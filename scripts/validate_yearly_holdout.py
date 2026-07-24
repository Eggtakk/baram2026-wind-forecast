"""
연도 단위 holdout 재검증 (기존 holdout 결함 대응).

문제: 기존 time_based_split(holdout_ratio=0.2)는 시간순 뒤쪽 20%를 그대로
자르는데, train 기간이 2023-01-01~2025-01-01(2년)이라 뒤쪽 20%는 8~12월
(약 5개월)에만 몰려 있고 1~7월이 전혀 없다. 반면 실제 test 기간은 2025년
"전체"(1~12월) — holdout이 계절적으로 test와 전혀 다른 분포라서, holdout
점수가 실제 리더보드 점수를 신뢰성 있게 예측하지 못했다 (오늘 saturation
+재튜닝 버전이 holdout에서는 더 높았지만 실제 제출에서는 더 낮게 나온 것이
그 증거).

수정: train=2023년 전체(1년), holdout=2024년 전체(1년)로 나눠 test(2025년
전체)와 계절 구성이 같은 "연 단위" 검증을 한다. 이걸로 어떤 feature/튜닝이
실제로 일반화되는지 다시 확인한다.

실행: (레포 루트에서) python3 scripts/validate_yearly_holdout.py [group_id ...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb

from src.features import build_baseline_features, get_feature_cols
from src.metrics import CAPACITY_KWH, validate_single_group
from src.power_curve import apply_power_curve_models, fit_power_curve_models
from src.preprocess import build_group_dataset
from src.validation import time_based_split_by_date

ROOT = Path(__file__).resolve().parents[1]
PARAM_DIR = ROOT / "experiments" / "baseline_lgbm"
SPLIT_DATE = "2024-01-01"

DEFAULT_LGBM_PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=31)


def load_tuned_params(group_id: int) -> dict:
    params = dict(DEFAULT_LGBM_PARAMS)
    best_path = PARAM_DIR / f"group{group_id}_best_params.json"
    if best_path.exists():
        with open(best_path, "r", encoding="utf-8") as f:
            params.update(json.load(f))
    return params


def run_variant(train_df, holdout_df, feature_cols, params, group_id, capacity, label):
    bagging_freq = 1 if params.get("bagging_fraction", 1.0) < 1.0 else 0
    model = lgb.LGBMRegressor(**{k: v for k, v in params.items() if k != "bagging_freq"}, random_state=42, bagging_freq=bagging_freq, verbosity=-1)
    model.fit(train_df[feature_cols], train_df["y"])
    pred = model.predict(holdout_df[feature_cols]).clip(min=0)
    result = validate_single_group(holdout_df["y"].to_numpy(), pred, group_id=group_id)
    score = 0.5 * result["one_minus_nmae"] + 0.5 * result["ficr"]
    print(f"  [{label}] score={score:.4f} nmae={result['nmae']:.4f} ficr={result['ficr']:.4f} n_holdout={len(holdout_df)}")
    return score


def run_group(group_id: int):
    df = build_group_dataset(group_id, split="train")
    df = df.dropna(subset=["y"]).reset_index(drop=True)
    train_raw, holdout_raw = time_based_split_by_date(df, split_date=SPLIT_DATE)
    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]

    print(f"\n=== group{group_id} (train={train_raw['forecast_kst_dtm'].min().date()}~{train_raw['forecast_kst_dtm'].max().date()}, "
          f"holdout={holdout_raw['forecast_kst_dtm'].min().date()}~{holdout_raw['forecast_kst_dtm'].max().date()}) ===")

    default_params = dict(DEFAULT_LGBM_PARAMS)
    tuned_params = load_tuned_params(group_id)

    # variant A: 물리 feature만(saturation 이전), 기본 파라미터 -- 원래 첫 제출과 가장 가까운 버전
    from src.features import add_default_wind_features, add_physics_features, add_time_features, add_lag_rolling_features

    def build_v1_features(d):
        d = add_default_wind_features(d)
        d = add_physics_features(d)
        d = add_time_features(d)
        speed_cols = [c for c in d.columns if c.endswith("_speed")]
        d = add_lag_rolling_features(d, cols=speed_cols, lags=[1, 2, 3], windows=[3, 6, 24])
        return d

    train_v1 = build_v1_features(train_raw)
    holdout_v1 = build_v1_features(holdout_raw)
    feat_v1 = get_feature_cols(train_v1)
    run_variant(train_v1, holdout_v1, feat_v1, default_params, group_id, capacity, "v1: physics-only + default params")
    run_variant(train_v1, holdout_v1, feat_v1, tuned_params, group_id, capacity, "v1b: physics-only + tuned params")

    # variant B: saturation + power curve + tuned params -- 오늘 제출한 버전
    train_v2 = build_baseline_features(train_raw)
    holdout_v2 = build_baseline_features(holdout_raw)
    curve_models = fit_power_curve_models(train_v2, capacity=capacity)
    train_v2 = apply_power_curve_models(train_v2, curve_models)
    holdout_v2 = apply_power_curve_models(holdout_v2, curve_models)
    feat_v2 = get_feature_cols(train_v2)
    run_variant(train_v2, holdout_v2, feat_v2, tuned_params, group_id, capacity, "v2: saturation+powercurve + tuned params")


def main():
    groups = [int(a) for a in sys.argv[1:]] or [1, 2, 3]
    for gid in groups:
        run_group(gid)


if __name__ == "__main__":
    main()
