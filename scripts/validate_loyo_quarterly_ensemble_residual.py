"""
42번 섹션 후속 — group1 stage-1 objective 앙상블(L2+quantile+FICR)이
stage-1 단독으로는 quantile 단독보다 짝지은 std가 2.6배 작아졌다(더
안정적)는 걸 확인했다. group2 사례(FICR 단독은 노이즈 수준이었다가
잔차보정을 얹은 뒤 1.81배까지 올라감, 19~20/40번 섹션)와 같은 패턴이
이 앙상블에도 적용되는지, **앙상블 stage-1 + 잔차보정(stage-2)**을
분기 LOQO(12-fold, 진짜 재학습)로 검증한다.

앙상블 가중치는 42번 섹션에서 delta/std 비율이 가장 좋았던 "균등
(1/3씩)"을 우선 사용(WEIGHTS 변경으로 다른 조합도 재사용 가능).
잔차보정 threshold/stage2_params는 기존 group1 quantile 배포본의
튜닝값(group1_residual_stage_quantile_best_config.json, threshold=0.8)을
그대로 재사용 — 이번 1차 검증에서는 재탐색하지 않고 "같은 잔차보정
레시피 위에 stage-1만 바꾸면 어떻게 되는가"를 먼저 본다.

OOF 앙상블 예측 = 3개 objective 각각 3-fold OOF 예측을 얻은 뒤 같은
가중치로 평균 — 잔차보정 학습에 쓰는 residual target(y - oof_pred)이
실제 holdout에서 쓰는 앙상블 예측과 같은 방식으로 만들어지도록 보장.

비용 경고: fold당 3(최종 fit) + 9(3-objective x 3-fold OOF) + 1(stage2)
= 13회 LightGBM 학습 — 단일-objective 잔차보정 fold(~13초, 5회 학습)의
약 2.5배(~30~40초)라 45초 제약상 fold당 1회씩만 호출 가능.

실행: python3 scripts/validate_loyo_quarterly_ensemble_residual.py [YYYY-Q ...]
결과: experiments/baseline_lgbm/loyo_quarterly_ensemble_residual_results.json
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import lightgbm as lgb

from src.data_cleaning import remove_curtailment
from src.features import (
    add_default_wind_features,
    add_lag_rolling_features,
    add_physics_features,
    add_time_features,
    get_feature_cols,
)
from src.ficr_objective import make_ficr_objective
from src.metrics import CAPACITY_KWH, validate_single_group
from src.preprocess import build_group_dataset
from src.validation import time_based_split_leave_quarter_out

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "baseline_lgbm"
RESULTS_PATH = OUT_DIR / "loyo_quarterly_ensemble_residual_results.json"

GROUP_ID = 1
RECIPE = "physics"
PARAMS_SOURCE = "yearly"
VALID_YEARS = [2022, 2023, 2024]
QUANTILE_ALPHA = 0.60
FICR_WEIGHT, LAMBDA_L2, T = 0.003, 1.0, 0.01
WEIGHTS = (1 / 3, 1 / 3, 1 / 3)  # (L2, quantile, FICR) -- 균등, 42번 섹션 최고 delta/std
OOF_SPLITS = 3
MIN_REGIME_SAMPLES = 30


def all_quarters():
    return [(y, q) for y in VALID_YEARS for q in [1, 2, 3, 4]]


def build_physics_features(df):
    df = add_default_wind_features(df)
    df = add_physics_features(df)
    df = add_time_features(df)
    speed_cols = [c for c in df.columns if c.endswith("_speed")]
    df = add_lag_rolling_features(df, cols=speed_cols, lags=[1, 2, 3], windows=[3, 6, 24])
    return df


def load_base_params():
    path = OUT_DIR / f"group{GROUP_ID}_{PARAMS_SOURCE}_best_params_{RECIPE}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_residual_config():
    path = OUT_DIR / f"group{GROUP_ID}_residual_stage_quantile_best_config.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fit_lgbm(X, y, params, seed=42):
    bagging_freq = 1 if params.get("bagging_fraction", 1.0) < 1.0 else 0
    model = lgb.LGBMRegressor(**params, random_state=seed, bagging_freq=bagging_freq, verbosity=-1)
    model.fit(X, y)
    return model


def stage1_param_variants(base_params, capacity):
    l2_params = dict(base_params)
    q_params = dict(base_params)
    q_params["objective"] = "quantile"
    q_params["alpha"] = QUANTILE_ALPHA
    f_params = dict(base_params)
    f_params["objective"] = make_ficr_objective(capacity, ficr_weight=FICR_WEIGHT, lambda_l2=LAMBDA_L2, T=T)
    return [l2_params, q_params, f_params]


def get_oof_ensemble_preds(train_feat, feature_cols, param_list, weights, n_splits=OOF_SPLITS):
    n = len(train_feat)
    idx = np.arange(n)
    blocks = np.array_split(idx, n_splits)
    oof_per_objective = [np.zeros(n) for _ in param_list]
    for i in range(n_splits):
        test_idx = blocks[i]
        train_idx = np.concatenate([blocks[j] for j in range(n_splits) if j != i])
        for oi, params in enumerate(param_list):
            model = _fit_lgbm(train_feat.iloc[train_idx][feature_cols], train_feat.iloc[train_idx]["y"], params)
            oof_per_objective[oi][test_idx] = model.predict(train_feat.iloc[test_idx][feature_cols]).clip(min=0)
    ensemble_oof = sum(w * o for w, o in zip(weights, oof_per_objective))
    return ensemble_oof


def run_fold(df, holdout_quarter: tuple[int, int], threshold: float, stage2_params: dict) -> dict:
    capacity = CAPACITY_KWH[f"kpx_group_{GROUP_ID}"]
    train_raw, holdout_raw = time_based_split_leave_quarter_out(
        df, holdout_quarter=holdout_quarter, valid_quarters=all_quarters()
    )
    cleaned_train, n_removed = remove_curtailment(train_raw, capacity=capacity)
    train_feat = build_physics_features(cleaned_train)
    holdout_feat = build_physics_features(holdout_raw)
    feature_cols = get_feature_cols(train_feat)
    base_params = load_base_params()
    param_list = stage1_param_variants(base_params, capacity)

    # 최종 stage-1 모델 3개 (holdout 예측용)
    holdout_preds = []
    for params in param_list:
        model = _fit_lgbm(train_feat[feature_cols], train_feat["y"], params)
        holdout_preds.append(model.predict(holdout_feat[feature_cols]).clip(min=0, max=capacity))
    ensemble_holdout_pred = sum(w * p for w, p in zip(WEIGHTS, holdout_preds))

    # OOF 앙상블 예측 (잔차보정 학습용)
    ensemble_oof_pred = get_oof_ensemble_preds(train_feat, feature_cols, param_list, WEIGHTS)

    y_train = train_feat["y"].to_numpy()
    train_frac = y_train / capacity
    regime_mask_train = train_frac >= threshold
    final_pred = ensemble_holdout_pred.copy()
    n_regime_train = int(regime_mask_train.sum())
    if n_regime_train >= MIN_REGIME_SAMPLES:
        residual_train = y_train - ensemble_oof_pred
        X_stage2 = train_feat.loc[regime_mask_train, feature_cols].copy()
        X_stage2["stage1_pred"] = ensemble_oof_pred[regime_mask_train]
        y_stage2 = residual_train[regime_mask_train]
        stage2_model = _fit_lgbm(X_stage2, y_stage2, stage2_params)

        holdout_frac = ensemble_holdout_pred / capacity
        regime_mask_holdout = holdout_frac >= threshold
        if regime_mask_holdout.sum() > 0:
            X_holdout_stage2 = holdout_feat.loc[regime_mask_holdout, feature_cols].copy()
            X_holdout_stage2["stage1_pred"] = ensemble_holdout_pred[regime_mask_holdout]
            correction = stage2_model.predict(X_holdout_stage2)
            final_pred[regime_mask_holdout] = ensemble_holdout_pred[regime_mask_holdout] + correction
    final_pred = np.clip(final_pred, 0, capacity)

    y_holdout = holdout_feat["y"].to_numpy()

    def score_of(pred):
        r = validate_single_group(y_holdout, pred, group_id=GROUP_ID)
        return 0.5 * r["one_minus_nmae"] + 0.5 * r["ficr"]

    return {
        "group_id": GROUP_ID,
        "holdout_year": holdout_quarter[0],
        "holdout_quarter": holdout_quarter[1],
        "score_ensemble_stage1_only": score_of(ensemble_holdout_pred),
        "score_ensemble_plus_residual": score_of(final_pred),
        "n_regime_train": n_regime_train,
        "n_train": len(train_feat),
        "n_holdout": len(holdout_feat),
        "n_curtailment_removed": n_removed,
    }


def parse_quarter_tokens(tokens):
    return [(int(t.split("-")[0]), int(t.split("-")[1])) for t in tokens]


def main():
    tokens = sys.argv[1:]
    quarters = parse_quarter_tokens(tokens) if tokens else all_quarters()
    config = load_residual_config()
    threshold, stage2_params = config["threshold"], config["stage2_params"]

    all_results = []
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    done = {(r["holdout_year"], r["holdout_quarter"]) for r in all_results}
    todo = [q for q in quarters if q not in done]

    if not todo:
        print("요청된 분기 전부 이미 처리됨 (스킵)")
    else:
        df = build_group_dataset(GROUP_ID, split="train").dropna(subset=["y"]).reset_index(drop=True)
        print(f"=== group{GROUP_ID} 앙상블(균등)+잔차보정(threshold={threshold}) 분기 LOQO ({len(todo)}개) ===")
        for hq in todo:
            t0 = time.time()
            r = run_fold(df, hq, threshold, stage2_params)
            all_results.append(r)
            print(f"  holdout={hq[0]}-Q{hq[1]}: stage1만={r['score_ensemble_stage1_only']:.4f} "
                  f"+잔차보정={r['score_ensemble_plus_residual']:.4f} [{time.time()-t0:.1f}s]")
            with open(RESULTS_PATH, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"저장: {RESULTS_PATH}")

    print("\n=== 누적 요약 ===")
    for key in ["score_ensemble_stage1_only", "score_ensemble_plus_residual"]:
        vals = np.array([r[key] for r in all_results])
        std = vals.std(ddof=1) if len(vals) > 1 else 0.0
        print(f"  {key}: n={len(vals)}/12 mean={vals.mean():.4f} std={std:.4f}")


if __name__ == "__main__":
    main()
