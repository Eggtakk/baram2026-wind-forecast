"""
커틀먼트/고장 의심 구간을 train에서만 제거하고 재학습했을 때 연 단위 holdout
점수가 개선되는지 검증.

diagnose_curtailment.py에서 확인된 것: 실측 풍속>=8m/s인데 발전량이 경험적
파워커브보다 30%p 이상 낮은 시간대가 그룹당 1.0~1.4% 존재하고, 겨울철
(12~2월)에 집중되며 최대 83시간까지 이어지는 연속 구간도 있음 — 측정
노이즈가 아니라 착빙/커틀먼트/고장 등 실제 운영 이벤트로 추정됨.

주의(누수 방지): holdout(2024년 전체)은 절대 건드리지 않는다 — 실제
test도 이런 이벤트를 포함할 수 있으므로, holdout 성능은 "정리된 데이터로
학습했을 때 정상 구간을 더 잘 맞추는지"를 보여줘야지, 이벤트 구간을
숨겨서 점수를 인위적으로 올리면 안 된다. 오직 train_raw(2024년 이전)에서만
의심 구간을 제거한다.

실행: (레포 루트에서) python3 scripts/validate_curtailment_removal.py [group_id ...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

from src.features import (
    add_default_wind_features,
    add_lag_rolling_features,
    add_physics_features,
    add_time_features,
    build_baseline_features,
    get_feature_cols,
)
from src.metrics import CAPACITY_KWH, validate_single_group
from src.power_curve import apply_power_curve_models, fit_power_curve_models
from src.preprocess import build_group_dataset
from src.validation import time_based_split_by_date

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "baseline_lgbm"
SPLIT_DATE = "2024-01-01"

RESIDUAL_THRESHOLD_RATIO = 0.30
HIGH_WIND_THRESHOLD = 8.0

DEFAULT_LGBM_PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=31)


# 지금까지 확인된, 커틀먼트 정리 전 기준 그룹별/레시피별 진짜 최고 파라미터 출처
# (group1/2는 그리드 탐색이, group3/physics는 optuna가, group3/full은 그리드가 근소 우세)
BEST_SOURCE = {
    (1, "full"): "yearly", (1, "physics"): "yearly",
    (2, "full"): "yearly", (2, "physics"): "yearly",
    (3, "full"): "yearly", (3, "physics"): "optuna",
}


def load_best_params(group_id: int, recipe: str, source: str | None = None) -> dict:
    source = source or BEST_SOURCE.get((group_id, recipe), "yearly")
    path = OUT_DIR / f"group{group_id}_{source}_best_params_{recipe}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return dict(DEFAULT_LGBM_PARAMS)


def flag_curtailment(train_raw, capacity):
    """train_raw(SCADA 포함)에서 의심 구간의 boolean mask를 반환."""
    valid = train_raw.dropna(subset=["scada_mean_ws"])
    curve = IsotonicRegression(y_min=0, y_max=capacity, increasing=True, out_of_bounds="clip")
    curve.fit(valid["scada_mean_ws"], valid["y"])

    est = curve.predict(train_raw["scada_mean_ws"].fillna(0))
    residual_ratio = (train_raw["y"] - est) / capacity
    mask = (residual_ratio < -RESIDUAL_THRESHOLD_RATIO) & (train_raw["scada_mean_ws"] >= HIGH_WIND_THRESHOLD)
    return mask.fillna(False)


def build_physics_features(df):
    df = add_default_wind_features(df)
    df = add_physics_features(df)
    df = add_time_features(df)
    speed_cols = [c for c in df.columns if c.endswith("_speed")]
    df = add_lag_rolling_features(df, cols=speed_cols, lags=[1, 2, 3], windows=[3, 6, 24])
    return df


def run_group(group_id: int, recipe: str = "full"):
    df = build_group_dataset(group_id, split="train", include_scada=True)
    df = df.dropna(subset=["y"]).reset_index(drop=True)
    train_raw, holdout_raw = time_based_split_by_date(df, split_date=SPLIT_DATE)
    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]
    params = load_best_params(group_id, recipe)

    def build_and_score(train_part, label):
        if recipe == "physics":
            train_df = build_physics_features(train_part)
            holdout_df = build_physics_features(holdout_raw)
        else:
            train_df = build_baseline_features(train_part)
            holdout_df = build_baseline_features(holdout_raw)
            curve_models = fit_power_curve_models(train_df, capacity=capacity)
            train_df = apply_power_curve_models(train_df, curve_models)
            holdout_df = apply_power_curve_models(holdout_df, curve_models)
        feature_cols = get_feature_cols(train_df)

        bagging_freq = 1 if params.get("bagging_fraction", 1.0) < 1.0 else 0
        model = lgb.LGBMRegressor(**params, random_state=42, bagging_freq=bagging_freq, verbosity=-1)
        model.fit(train_df[feature_cols], train_df["y"])
        pred = model.predict(holdout_df[feature_cols]).clip(min=0)
        r = validate_single_group(holdout_df["y"].to_numpy(), pred, group_id=group_id)
        score = 0.5 * r["one_minus_nmae"] + 0.5 * r["ficr"]
        print(f"  [{label}] n_train={len(train_part)} score={score:.4f} nmae={r['nmae']:.4f} ficr={r['ficr']:.4f}")
        return score

    print(f"\n=== group{group_id} (recipe={recipe}) ===")
    base_score = build_and_score(train_raw, "원본(정리 전)")

    mask = flag_curtailment(train_raw, capacity)
    n_flagged = int(mask.sum())
    print(f"  의심 구간 {n_flagged}행 ({n_flagged/len(train_raw)*100:.2f}%) 제거")
    cleaned_train = train_raw[~mask].reset_index(drop=True)
    clean_score = build_and_score(cleaned_train, "커틀먼트 제거 후")

    print(f"  => delta: {clean_score - base_score:+.4f}")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("groups", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--recipe", choices=["physics", "full"], default="full")
    args = ap.parse_args()
    for gid in args.groups:
        run_group(gid, recipe=args.recipe)


if __name__ == "__main__":
    main()
