"""
33번 섹션 후속 — FICR 경계 후처리 물리 블렌딩을 "stage-1 단독"이 아니라
**배포된 stage-1+잔차보정(stage-2) 최종 예측**에 적용해서 재검증한다.

33번 섹션은 stage-2를 얹기 전 stage-1만으로 블렌딩을 시험했는데, 세 그룹
모두 방향은 양수였지만 delta가 각 그룹의 fold-std보다 작아 "노이즈 수준"
판정을 받았다(보류). 잔차보정 stage-2가 이미 고출력 구간 편향을 상당히
잡아준 뒤에는 남은 오차의 성격이 달라져 블렌딩 효과가 다르게(더 크게 or
더 작게) 나타날 수 있다는 가설을 이번에 검증한다.

그룹별 배포 설정 그대로 재현:
  group1: quantile objective(alpha=0.60) stage-1 + 잔차보정
    (threshold/stage2_params: group1_residual_stage_quantile_best_config.json)
  group2: FICR-shaped objective(ficr_weight=0.008) stage-1 + 잔차보정
    (threshold/stage2_params: group2_residual_stage_ficr_best_config.json)
  group3: 배포본은 잔차보정 없음(L2 단독) — 33번 섹션에서 이미 검증 완료라
    이 스크립트에서는 생략.

실행: python3 scripts/validate_ficr_blend_full_pipeline.py <group_id>
"""
import json
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
from validate_loyo import OUT_DIR, RECIPE_CHOICE, VALID_YEARS, build_physics_features, load_params
from validate_loyo_candidates import get_baseline_stats

STAGE1_STYLE = {1: "quantile", 2: "ficr"}
QUANTILE_ALPHA = {1: 0.60}
FICR_WEIGHT = {2: 0.008}
CONFIG_SUFFIX = {1: "quantile", 2: "ficr"}
ALPHA_GRID = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]

MIN_REGIME_SAMPLES = 50
OOF_SPLITS = 3


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


def _fit_lgbm(X, y, params, seed=42):
    bagging_freq = 1 if params.get("bagging_fraction", 1.0) < 1.0 else 0
    model = lgb.LGBMRegressor(**params, random_state=seed, bagging_freq=bagging_freq, verbosity=-1)
    model.fit(X, y)
    return model


def get_oof_stage1_preds(train_feat, feature_cols, params, n_splits=OOF_SPLITS):
    n = len(train_feat)
    idx = np.arange(n)
    blocks = np.array_split(idx, n_splits)
    oof = np.zeros(n)
    for i in range(n_splits):
        test_idx = blocks[i]
        train_idx = np.concatenate([blocks[j] for j in range(n_splits) if j != i])
        model = _fit_lgbm(train_feat.iloc[train_idx][feature_cols], train_feat.iloc[train_idx]["y"], params)
        oof[test_idx] = model.predict(train_feat.iloc[test_idx][feature_cols]).clip(min=0)
    return oof


def run_fold(group_id: int, df, holdout_year: int, threshold: float, stage2_params: dict) -> dict:
    recipe = RECIPE_CHOICE[group_id]
    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]

    train_raw, holdout_raw = time_based_split_leave_year_out(
        df, holdout_year=holdout_year, valid_years=VALID_YEARS[group_id]
    )
    cleaned_train, _ = remove_curtailment(train_raw, capacity=capacity)

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
    stage1_params = get_stage1_params(group_id, recipe)

    stage1_model = _fit_lgbm(train_feat[feature_cols], train_feat["y"], stage1_params)
    stage1_holdout_pred = stage1_model.predict(holdout_feat[feature_cols]).clip(min=0, max=capacity)
    oof_pred = get_oof_stage1_preds(train_feat, feature_cols, stage1_params)

    y_train = train_feat["y"].to_numpy()
    train_frac = y_train / capacity
    regime_mask_train = train_frac >= threshold
    final_pred = stage1_holdout_pred.copy()
    if regime_mask_train.sum() >= MIN_REGIME_SAMPLES:
        residual_train = y_train - oof_pred
        X_stage2 = train_feat.loc[regime_mask_train, feature_cols].copy()
        X_stage2["stage1_pred"] = oof_pred[regime_mask_train]
        y_stage2 = residual_train[regime_mask_train]
        stage2_model = _fit_lgbm(X_stage2, y_stage2, stage2_params)

        holdout_frac = stage1_holdout_pred / capacity
        regime_mask_holdout = holdout_frac >= threshold
        if regime_mask_holdout.sum() > 0:
            X_holdout_stage2 = holdout_feat.loc[regime_mask_holdout, feature_cols].copy()
            X_holdout_stage2["stage1_pred"] = stage1_holdout_pred[regime_mask_holdout]
            correction = stage2_model.predict(X_holdout_stage2)
            final_pred[regime_mask_holdout] = stage1_holdout_pred[regime_mask_holdout] + correction
    final_pred = np.clip(final_pred, 0, capacity)

    physics_est = np.clip(
        estimate_group_power_kwh(
            holdout_feat["ldaps_hub_speed"], holdout_feat["ldaps_air_density"], group_id, capacity
        ),
        0, capacity,
    )

    return {
        "holdout_year": holdout_year, "y_holdout": holdout_feat["y"].to_numpy(),
        "final_pred": final_pred, "physics_est": np.asarray(physics_est), "capacity": capacity,
    }


def score_of(y, pred, group_id):
    r = validate_single_group(y, pred, group_id=group_id)
    return 0.5 * r["one_minus_nmae"] + 0.5 * r["ficr"], r["nmae"], r["ficr"]


def main(group_id: int):
    suffix = CONFIG_SUFFIX[group_id]
    config_path = OUT_DIR / f"group{group_id}_residual_stage_{suffix}_best_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    threshold, stage2_params = config["threshold"], config["stage2_params"]
    print(f"=== group{group_id} 배포 설정: stage-1={STAGE1_STYLE[group_id]}, "
          f"잔차보정 threshold={threshold} ===")

    baseline_stats = get_baseline_stats()
    l2_base_mean = baseline_stats[group_id][0]

    df = build_group_dataset(group_id, split="train").dropna(subset=["y"]).reset_index(drop=True)
    fold_data = [run_fold(group_id, df, year, threshold, stage2_params) for year in VALID_YEARS[group_id]]

    for r in fold_data:
        s0, n0, f0 = score_of(r["y_holdout"], r["final_pred"], group_id)
        print(f"  holdout={r['holdout_year']}: 최종(stage1+잔차보정) score={s0:.4f} NMAE={n0:.4f} FICR={f0:.4f}")

    base_scores = [score_of(r["y_holdout"], r["final_pred"], group_id)[0] for r in fold_data]
    base_mean = float(np.mean(base_scores))
    print(f"\n  잔차보정 최종 예측 단독(α=0): mean={base_mean:.4f} (delta_vs_L2_baseline={base_mean-l2_base_mean:+.4f})")

    print(f"\n  α(물리 블렌드 비중) 그리드서치:")
    best_alpha, best_mean = 0.0, base_mean
    for alpha in ALPHA_GRID:
        scores, nmaes, ficrs = [], [], []
        for r in fold_data:
            blended = np.clip((1 - alpha) * r["final_pred"] + alpha * r["physics_est"], 0, r["capacity"])
            s, n, f = score_of(r["y_holdout"], blended, group_id)
            scores.append(s); nmaes.append(n); ficrs.append(f)
        mean_s, mean_n, mean_f = float(np.mean(scores)), float(np.mean(nmaes)), float(np.mean(ficrs))
        marker = ""
        if mean_s > best_mean:
            best_mean, best_alpha = mean_s, alpha
            marker = "  <- best"
        print(f"    alpha={alpha:.2f}: mean_score={mean_s:.4f}(delta_vs_final={mean_s-base_mean:+.4f}) "
              f"NMAE={mean_n:.4f} FICR={mean_f:.4f}{marker}")

    print(f"\n  최적 alpha={best_alpha}, score={best_mean:.4f} "
          f"(최종예측 단독 대비 delta={best_mean-base_mean:+.4f}, "
          f"L2 baseline 대비 delta={best_mean-l2_base_mean:+.4f})")


if __name__ == "__main__":
    main(int(sys.argv[1]))
