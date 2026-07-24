"""
FICR 가중치 특성을 반영한 sample_weight 재학습 검증.

대회 산식을 다시 보면: FICR = sum(a_i * unit_price_i) / sum(a_i * 4) — 즉
FICR은 "실제 발전량(a)"으로 가중평균된 지표다. 반면 NMAE는 자격 시간대
전체를 단순 평균한다. 이 말은, 고출력 시간대(90-100% capacity 등)의 오차가
NMAE보다 FICR에 훨씬 크게 반영된다는 뜻이다 — 우리가 계속 씨름해온 "정격출력
구간 오차"가 실제로 리더보드 점수(특히 FICR)에 불균형하게 큰 악영향을 준다는
것을 산식 자체가 뒷받침한다.

그런데 지금까지 모델은 학습 시 모든 시간대를 동일 가중치로 취급했다(기본
L2 loss, sample_weight 없음). FICR가 고출력 시간대에 더 민감하다면, 학습
단계에서부터 고출력 시간대에 더 큰 가중치를 줘서 그쪽 오차를 우선적으로
줄이도록 유도하면 전체 대회 점수가 개선될 수 있다 (NMAE는 다소 희생될 수
있지만, 리더보드 격차 분석상 FICR 쪽 개선 여지가 훨씬 크다).

실행: (레포 루트에서) python3 scripts/validate_sample_weight.py [group_id ...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import numpy as np

from src.features import build_baseline_features, get_feature_cols
from src.metrics import CAPACITY_KWH, analyze_error_bands, validate_single_group
from src.power_curve import apply_power_curve_models, fit_power_curve_models
from src.preprocess import build_group_dataset
from src.validation import time_based_split

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


WEIGHT_SCHEMES = {
    "none": lambda y, cap: None,
    "linear_y": lambda y, cap: (y / cap).clip(lower=0.05),
    "sqrt_y": lambda y, cap: np.sqrt((y / cap).clip(lower=0.05)),
}


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
    for name, weight_fn in WEIGHT_SCHEMES.items():
        w = weight_fn(train_df["y"], capacity)
        model = lgb.LGBMRegressor(**base_params, random_state=42, bagging_freq=bagging_freq, verbosity=-1)
        model.fit(train_df[feature_cols], train_df["y"], sample_weight=w)
        pred = model.predict(holdout_df[feature_cols]).clip(min=0)
        result = validate_single_group(holdout_df["y"].to_numpy(), pred, group_id=group_id)
        score = 0.5 * result["one_minus_nmae"] + 0.5 * result["ficr"]
        bands = analyze_error_bands(holdout_df["y"].to_numpy(), pred, capacity)
        b90 = bands["by_capacity_band"].loc["90-100%"]
        print(
            f"  weight={name:10s} score={score:.4f} nmae={result['nmae']:.4f} ficr={result['ficr']:.4f} "
            f"| 90-100% mean_err={b90['mean_error_rate']:.4f} pct_over8={b90['pct_over8']:.3f}"
        )


def main():
    groups = [int(a) for a in sys.argv[1:]] or [1, 2, 3]
    for gid in groups:
        run_group(gid)


if __name__ == "__main__":
    main()
