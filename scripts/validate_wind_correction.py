"""
예보 풍속(허브높이 외삽값) 사후 보정이 holdout 성능을 개선하는지 검증.

기존 calibration(모델 출력 보정, 기각됨)과 다른 점: 여기서는 train_df
전체(80%, 다양한 계절 포함)로 "예보풍속 -> 실측풍속" 보정기를 학습한다 —
좁은 시간대 slice가 아니므로 계절적 과적합 위험이 훨씬 낮을 것으로 기대.

A/B 비교:
  - baseline: 보정 없이 build_baseline_features 그대로
  - corrected: train_raw로 학습한 wind_correctors를 적용한 build_baseline_features

둘 다 같은 outer holdout(시간순 20%)에서 평가한다.

실행: (레포 루트에서) python3 scripts/validate_wind_correction.py [group_id ...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb

from src.features import build_baseline_features, get_feature_cols
from src.metrics import CAPACITY_KWH, analyze_error_bands, validate_single_group
from src.power_curve import apply_power_curve_models, fit_power_curve_models
from src.preprocess import build_group_dataset
from src.validation import time_based_split
from src.wind_bias_correction import fit_wind_bias_correctors_from_raw

ROOT = Path(__file__).resolve().parents[1]
PARAM_DIR = ROOT / "experiments" / "baseline_lgbm"

HOLDOUT_RATIO = 0.2
DEFAULT_LGBM_PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=31)


def load_tuned_params(group_id: int) -> dict:
    params = dict(DEFAULT_LGBM_PARAMS)
    best_path = PARAM_DIR / f"group{group_id}_best_params.json"
    if best_path.exists():
        with open(best_path, "r", encoding="utf-8") as f:
            params.update(json.load(f))
    return params


def fit_and_eval(train_df, holdout_df, feature_cols, params, group_id, capacity):
    bagging_freq = 1 if params.get("bagging_fraction", 1.0) < 1.0 else 0
    model = lgb.LGBMRegressor(**params, random_state=42, bagging_freq=bagging_freq, verbosity=-1)
    model.fit(train_df[feature_cols], train_df["y"])
    pred = model.predict(holdout_df[feature_cols]).clip(min=0)
    result = validate_single_group(holdout_df["y"].to_numpy(), pred, group_id=group_id)
    score = 0.5 * result["one_minus_nmae"] + 0.5 * result["ficr"]
    bands = analyze_error_bands(holdout_df["y"].to_numpy(), pred, capacity)
    return score, result, bands


def run_group(group_id: int) -> dict:
    raw_df = build_group_dataset(group_id, split="train")
    raw_df = raw_df.dropna(subset=["y"]).reset_index(drop=True)
    train_raw, holdout_raw = time_based_split(raw_df, holdout_ratio=HOLDOUT_RATIO)

    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]
    params = load_tuned_params(group_id)

    # --- baseline (보정 없음) ---
    train_base = build_baseline_features(train_raw)
    holdout_base = build_baseline_features(holdout_raw)
    curve_models_base = fit_power_curve_models(train_base, capacity=capacity)
    train_base = apply_power_curve_models(train_base, curve_models_base)
    holdout_base = apply_power_curve_models(holdout_base, curve_models_base)
    feature_cols_base = get_feature_cols(train_base)
    base_score, base_result, base_bands = fit_and_eval(train_base, holdout_base, feature_cols_base, params, group_id, capacity)

    # --- corrected (예보풍속 보정, train_raw 전체로 fit) ---
    correctors = fit_wind_bias_correctors_from_raw(train_raw)
    train_corr = build_baseline_features(train_raw, wind_correctors=correctors)
    holdout_corr = build_baseline_features(holdout_raw, wind_correctors=correctors)
    curve_models_corr = fit_power_curve_models(train_corr, capacity=capacity)
    train_corr = apply_power_curve_models(train_corr, curve_models_corr)
    holdout_corr = apply_power_curve_models(holdout_corr, curve_models_corr)
    feature_cols_corr = get_feature_cols(train_corr)
    corr_score, corr_result, corr_bands = fit_and_eval(train_corr, holdout_corr, feature_cols_corr, params, group_id, capacity)

    return {
        "group_id": group_id,
        "base_score": base_score,
        "corr_score": corr_score,
        "base_result": base_result,
        "corr_result": corr_result,
        "base_bands": base_bands,
        "corr_bands": corr_bands,
    }


def main():
    groups = [int(a) for a in sys.argv[1:]] or [1, 2, 3]
    for gid in groups:
        r = run_group(gid)
        print(f"\n=== group{gid} ===")
        print(f"  baseline:  score={r['base_score']:.4f}  nmae={r['base_result']['nmae']:.4f}  ficr={r['base_result']['ficr']:.4f}")
        print(f"  corrected: score={r['corr_score']:.4f}  nmae={r['corr_result']['nmae']:.4f}  ficr={r['corr_result']['ficr']:.4f}")
        print(f"  delta score: {r['corr_score'] - r['base_score']:+.4f}")

        bb, cb = r["base_bands"]["overall"], r["corr_bands"]["overall"]
        print(f"  [전체] baseline >8%: {bb['pct_over8']:.3f} | corrected >8%: {cb['pct_over8']:.3f}")

        for band in ["70-80%", "80-90%", "90-100%"]:
            bband = r["base_bands"]["by_capacity_band"].loc[band]
            cband = r["corr_bands"]["by_capacity_band"].loc[band]
            print(
                f"  [{band}, n={int(bband['n'])}] baseline mean_err={bband['mean_error_rate']:.4f} "
                f"bias={bband['mean_bias']:+.4f} pct_over8={bband['pct_over8']:.3f}  ->  "
                f"corrected mean_err={cband['mean_error_rate']:.4f} bias={cband['mean_bias']:+.4f} "
                f"pct_over8={cband['pct_over8']:.3f}"
            )


if __name__ == "__main__":
    main()
