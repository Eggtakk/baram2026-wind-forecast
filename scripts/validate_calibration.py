"""
정격출력 구간 under-prediction 사후 보정(bias calibration) 검증 스크립트.

구조 (누수 방지를 위해 2단 split):
  전체 train
    -> outer: train_df(80%) / holdout_df(20%, 최종 검증용 = 리더보드 대리)
      -> inner: fit_df(80%) / calib_df(20%, 보정 학습 전용)

1. fit_df로 모델 학습 (파워커브도 fit_df에서만 fit)
2. calib_df에서 예측 -> 실제값 쌍으로 등단조회귀 보정기(calibrator) 학습
3. holdout_df에서: 보정 전(raw) vs 보정 후(calibrated) 점수/구간별 오차 비교

실행: (레포 루트에서) python3 scripts/validate_calibration.py [group_id ...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb

from src.calibration import apply_bias_calibrator, fit_bias_calibrator
from src.features import build_baseline_features, get_feature_cols
from src.metrics import CAPACITY_KWH, analyze_error_bands, validate_single_group
from src.power_curve import apply_power_curve_models, fit_power_curve_models
from src.preprocess import build_group_dataset
from src.validation import time_based_split

ROOT = Path(__file__).resolve().parents[1]
PARAM_DIR = ROOT / "experiments" / "baseline_lgbm"

OUTER_HOLDOUT_RATIO = 0.2
INNER_CALIB_RATIO = 0.2

DEFAULT_LGBM_PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=31)


def load_tuned_params(group_id: int) -> dict:
    params = dict(DEFAULT_LGBM_PARAMS)
    best_path = PARAM_DIR / f"group{group_id}_best_params.json"
    if best_path.exists():
        with open(best_path, "r", encoding="utf-8") as f:
            params.update(json.load(f))
    return params


def run_group(group_id: int) -> dict:
    df = build_group_dataset(group_id, split="train")
    df = build_baseline_features(df)
    df = df.dropna(subset=["y"]).reset_index(drop=True)

    train_df, holdout_df = time_based_split(df, holdout_ratio=OUTER_HOLDOUT_RATIO)
    fit_df, calib_df = time_based_split(train_df, holdout_ratio=INNER_CALIB_RATIO)

    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]

    curve_models = fit_power_curve_models(fit_df, capacity=capacity)
    fit_df = apply_power_curve_models(fit_df, curve_models)
    calib_df = apply_power_curve_models(calib_df, curve_models)
    holdout_df = apply_power_curve_models(holdout_df, curve_models)

    feature_cols = get_feature_cols(fit_df)
    params = load_tuned_params(group_id)
    bagging_freq = 1 if params.get("bagging_fraction", 1.0) < 1.0 else 0

    model = lgb.LGBMRegressor(**params, random_state=42, bagging_freq=bagging_freq, verbosity=-1)
    model.fit(fit_df[feature_cols], fit_df["y"])

    calib_pred = model.predict(calib_df[feature_cols]).clip(min=0)
    calibrator = fit_bias_calibrator(calib_pred, calib_df["y"].to_numpy(), capacity=capacity)

    raw_pred = model.predict(holdout_df[feature_cols]).clip(min=0)
    cal_pred = apply_bias_calibrator(raw_pred, calibrator)

    raw_result = validate_single_group(holdout_df["y"].to_numpy(), raw_pred, group_id=group_id)
    cal_result = validate_single_group(holdout_df["y"].to_numpy(), cal_pred, group_id=group_id)
    raw_score = 0.5 * raw_result["one_minus_nmae"] + 0.5 * raw_result["ficr"]
    cal_score = 0.5 * cal_result["one_minus_nmae"] + 0.5 * cal_result["ficr"]

    raw_bands = analyze_error_bands(holdout_df["y"].to_numpy(), raw_pred, capacity)
    cal_bands = analyze_error_bands(holdout_df["y"].to_numpy(), cal_pred, capacity)

    return {
        "group_id": group_id,
        "raw_score": raw_score,
        "cal_score": cal_score,
        "raw_result": raw_result,
        "cal_result": cal_result,
        "raw_bands": raw_bands,
        "cal_bands": cal_bands,
        "n_fit": len(fit_df),
        "n_calib": len(calib_df),
        "n_holdout": len(holdout_df),
    }


def main():
    groups = [int(a) for a in sys.argv[1:]] or [1, 2, 3]
    for gid in groups:
        r = run_group(gid)
        print(f"\n=== group{gid} (fit={r['n_fit']}, calib={r['n_calib']}, holdout={r['n_holdout']}) ===")
        print(f"  raw:        score={r['raw_score']:.4f}  nmae={r['raw_result']['nmae']:.4f}  ficr={r['raw_result']['ficr']:.4f}")
        print(f"  calibrated: score={r['cal_score']:.4f}  nmae={r['cal_result']['nmae']:.4f}  ficr={r['cal_result']['ficr']:.4f}")
        print(f"  delta score: {r['cal_score'] - r['raw_score']:+.4f}")

        rb, cb = r["raw_bands"]["overall"], r["cal_bands"]["overall"]
        print(f"  [전체] raw >8%: {rb['pct_over8']:.3f} | calibrated >8%: {cb['pct_over8']:.3f}")

        for band in ["70-80%", "80-90%", "90-100%"]:
            rband = r["raw_bands"]["by_capacity_band"].loc[band]
            cband = r["cal_bands"]["by_capacity_band"].loc[band]
            print(
                f"  [{band}, n={int(rband['n'])}] raw mean_err={rband['mean_error_rate']:.4f} "
                f"bias={rband['mean_bias']:+.4f} pct_over8={rband['pct_over8']:.3f}  ->  "
                f"calibrated mean_err={cband['mean_error_rate']:.4f} bias={cband['mean_bias']:+.4f} "
                f"pct_over8={cband['pct_over8']:.3f}"
            )


if __name__ == "__main__":
    main()
