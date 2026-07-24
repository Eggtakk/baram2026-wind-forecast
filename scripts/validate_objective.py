"""
LightGBM objective(손실함수) 비교: 기본값(L2, 'regression') vs 'mae'(L1) vs 'huber'.

대회 점수(NMAE, FICR)는 둘 다 "절대오차/임계값" 기반이라 L2(제곱오차)보다
L1(MAE) 계열이 목적함수 자체로 더 잘 맞을 가능성이 있다. 지금까지의
하이퍼파라미터 탐색은 objective를 한 번도 바꿔본 적이 없었다(항상 기본 L2).

실행: (레포 루트에서) python3 scripts/validate_objective.py [group_id ...]
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

ROOT = Path(__file__).resolve().parents[1]
PARAM_DIR = ROOT / "experiments" / "baseline_lgbm"
HOLDOUT_RATIO = 0.2
DEFAULT_LGBM_PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=31)

OBJECTIVES = ["regression", "mae", "huber"]


def load_tuned_params(group_id: int) -> dict:
    params = dict(DEFAULT_LGBM_PARAMS)
    best_path = PARAM_DIR / f"group{group_id}_best_params.json"
    if best_path.exists():
        with open(best_path, "r", encoding="utf-8") as f:
            params.update(json.load(f))
    return params


def run_group(group_id: int):
    df = build_group_dataset(group_id, split="train")
    df = build_baseline_features(df)
    df = df.dropna(subset=["y"]).reset_index(drop=True)
    train_df, holdout_df = time_based_split(df, holdout_ratio=HOLDOUT_RATIO)

    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]
    curve_models = fit_power_curve_models(train_df, capacity=capacity)
    train_df = apply_power_curve_models(train_df, curve_models)
    holdout_df = apply_power_curve_models(holdout_df, curve_models)
    feature_cols = get_feature_cols(train_df)

    base_params = load_tuned_params(group_id)
    bagging_freq = 1 if base_params.get("bagging_fraction", 1.0) < 1.0 else 0

    print(f"\n=== group{group_id} ===")
    for obj in OBJECTIVES:
        model = lgb.LGBMRegressor(**base_params, objective=obj, random_state=42, bagging_freq=bagging_freq, verbosity=-1)
        model.fit(train_df[feature_cols], train_df["y"])
        pred = model.predict(holdout_df[feature_cols]).clip(min=0)
        result = validate_single_group(holdout_df["y"].to_numpy(), pred, group_id=group_id)
        score = 0.5 * result["one_minus_nmae"] + 0.5 * result["ficr"]
        bands = analyze_error_bands(holdout_df["y"].to_numpy(), pred, capacity)
        b90 = bands["by_capacity_band"].loc["90-100%"]
        print(
            f"  objective={obj:12s} score={score:.4f} nmae={result['nmae']:.4f} ficr={result['ficr']:.4f} "
            f"| 90-100% mean_err={b90['mean_error_rate']:.4f} pct_over8={b90['pct_over8']:.3f}"
        )


def main():
    groups = [int(a) for a in sys.argv[1:]] or [1, 2, 3]
    for gid in groups:
        run_group(gid)


if __name__ == "__main__":
    main()
