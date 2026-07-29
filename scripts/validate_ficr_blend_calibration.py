"""
FICR 경계 후처리 보정 — 1차 검증(개념 증명).

배경: FICR 보상은 계단함수다(오차율 <= 6% capacity -> 4원, <=8% -> 3원,
초과 -> 0원, src/metrics.py의 error_rate = |forecast-actual|/capacity).
지금까지의 접근(잔차보정 stage-2, FICR-shaped objective, quantile
regression)은 전부 "모델이 뭘 예측할지"를 바꾸는 방식이었다. 이 스크립트는
질적으로 다른 방식을 시험한다 — 모델 재학습 없이, 이미 나온 예측값을
물리 파워커브 추정치(src/manufacturer_power_curve.py)와 블렌딩해서
FICR을 직접 높일 수 있는지 본다.

가설: 물리 파워커브는 개별 모델보다 평균적으로 덜 정확하지만(29번 섹션에서
이미 확인) 분산이 작고 극단적인 outlier 오차를 잘 안 낸다 — 모델 예측과
소량만 블렌딩(final = (1-α)*model_pred + α*physics_est)하면 모델의 큰
outlier 오차 일부를 깎아 FICR 경계(6%/8%) 안쪽으로 끌어들일 수 있을지도
모른다. NMAE은 소폭 나빠질 수 있지만 FICR이 그보다 더 크게 오르면 total
score(0.5*(1-NMAE)+0.5*FICR)에 순이득.

이번엔 순수 개념 증명 — 각 그룹의 "배포된 stage-1 objective"(group1=quantile
alpha=0.60, group2=FICR objective ficr_weight=0.008, group3=L2)만 재현하고
(잔차보정 stage-2는 아직 얹지 않음), 그 예측을 물리 추정치와 블렌딩하는
α를 LOYO 폴드에서 그리드서치해 total score를 직접 최대화하는 α가 있는지,
있다면 얼마나 이득인지 확인한다.

실행: python3 scripts/validate_ficr_blend_calibration.py <group_id>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import lightgbm as lgb

from src.data_cleaning import remove_curtailment
from src.features import build_baseline_features, get_feature_cols
from src.ficr_objective import make_ficr_objective
from src.manufacturer_power_curve import estimate_group_power_kwh
from src.metrics import CAPACITY_KWH, validate_single_group
from src.power_curve import apply_power_curve_models, fit_power_curve_models
from src.preprocess import build_group_dataset
from src.validation import time_based_split_leave_year_out
from validate_loyo import RECIPE_CHOICE, VALID_YEARS, build_physics_features, load_params
from validate_loyo_candidates import get_baseline_stats

STAGE1_STYLE = {1: "quantile", 2: "ficr", 3: "l2"}
QUANTILE_ALPHA = {1: 0.60}
FICR_WEIGHT = {2: 0.008}
ALPHA_GRID = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]


def get_stage1_params(group_id: int, recipe: str):
    params = dict(load_params(group_id, recipe))
    style = STAGE1_STYLE[group_id]
    if style == "quantile":
        params["objective"] = "quantile"
        params["alpha"] = QUANTILE_ALPHA[group_id]
    elif style == "ficr":
        capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]
        params["objective"] = make_ficr_objective(capacity, ficr_weight=FICR_WEIGHT[group_id], lambda_l2=1.0, T=0.01)
    return params


def run_fold(group_id: int, df, holdout_year: int) -> dict:
    recipe = RECIPE_CHOICE[group_id]
    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]

    train_raw, holdout_raw = time_based_split_leave_year_out(
        df, holdout_year=holdout_year, valid_years=VALID_YEARS[group_id]
    )
    cleaned_train, n_removed = remove_curtailment(train_raw, capacity=capacity)

    if recipe == "physics":
        train_feat = build_physics_features(cleaned_train)
        holdout_feat = build_physics_features(holdout_raw)
    else:
        train_feat = build_baseline_features(cleaned_train)
        holdout_feat = build_baseline_features(holdout_raw)
        curve_models = fit_power_curve_models(train_feat, capacity=capacity)
        train_feat = apply_power_curve_models(train_feat, curve_models)
        holdout_feat = apply_power_curve_models(holdout_feat, curve_models)

    feature_cols = get_feature_cols(train_feat)
    params = get_stage1_params(group_id, recipe)

    bagging_freq = 1 if params.get("bagging_fraction", 1.0) < 1.0 else 0
    model = lgb.LGBMRegressor(**params, random_state=42, bagging_freq=bagging_freq, verbosity=-1)
    model.fit(train_feat[feature_cols], train_feat["y"])
    model_pred = model.predict(holdout_feat[feature_cols]).clip(min=0, max=capacity)

    physics_est = np.clip(
        estimate_group_power_kwh(
            holdout_feat["ldaps_hub_speed"], holdout_feat["ldaps_air_density"], group_id, capacity
        ),
        0, capacity,
    )

    y_holdout = holdout_feat["y"].to_numpy()
    return {"holdout_year": holdout_year, "y_holdout": y_holdout, "model_pred": model_pred, "physics_est": np.asarray(physics_est), "capacity": capacity}


def score_of(y, pred, group_id):
    r = validate_single_group(y, pred, group_id=group_id)
    return 0.5 * r["one_minus_nmae"] + 0.5 * r["ficr"], r["nmae"], r["ficr"]


def main(group_id: int):
    baseline_stats = get_baseline_stats()
    l2_base_mean = baseline_stats[group_id][0]
    print(f"=== group{group_id} L2 baseline mean={l2_base_mean:.4f} (stage-1 style={STAGE1_STYLE[group_id]}) ===")

    df = build_group_dataset(group_id, split="train").dropna(subset=["y"]).reset_index(drop=True)
    fold_data = [run_fold(group_id, df, year) for year in VALID_YEARS[group_id]]

    for r in fold_data:
        s0, n0, f0 = score_of(r["y_holdout"], r["model_pred"], group_id)
        print(f"  holdout={r['holdout_year']}: stage-1 단독(α=0) score={s0:.4f} NMAE={n0:.4f} FICR={f0:.4f}")

    print(f"\n  α(물리 블렌드 비중) 그리드서치 — total score 직접 최대화:")
    best_alpha, best_mean = 0.0, -1.0
    for alpha in ALPHA_GRID:
        scores, nmaes, ficrs = [], [], []
        for r in fold_data:
            blended = np.clip((1 - alpha) * r["model_pred"] + alpha * r["physics_est"], 0, r["capacity"])
            s, n, f = score_of(r["y_holdout"], blended, group_id)
            scores.append(s); nmaes.append(n); ficrs.append(f)
        mean_s, mean_n, mean_f = np.mean(scores), np.mean(nmaes), np.mean(ficrs)
        marker = ""
        if mean_s > best_mean:
            best_mean, best_alpha = mean_s, alpha
            marker = "  <- best"
        print(f"    alpha={alpha:.2f}: mean_score={mean_s:.4f}(delta_vs_stage1_alone={mean_s-scores_at_zero(fold_data, group_id):+.4f}) "
              f"NMAE={mean_n:.4f} FICR={mean_f:.4f}{marker}")

    print(f"\n  최적 alpha={best_alpha}, score={best_mean:.4f} "
          f"(stage-1 단독 대비 delta={best_mean-scores_at_zero(fold_data, group_id):+.4f})")


def scores_at_zero(fold_data, group_id):
    scores = [score_of(r["y_holdout"], r["model_pred"], group_id)[0] for r in fold_data]
    return float(np.mean(scores))


if __name__ == "__main__":
    main(int(sys.argv[1]))
