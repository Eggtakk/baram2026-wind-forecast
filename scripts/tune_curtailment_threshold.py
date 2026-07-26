"""
커틀먼트 탐지 임계값(RESIDUAL_THRESHOLD_RATIO, HIGH_WIND_THRESHOLD) 자체를
연 단위 holdout 기준으로 그리드 탐색.

이전 실험(validate_curtailment_removal.py)에서는 임계값을 첫 추정치
(0.30, 8.0m/s)로 고정한 채 "제거하면 도움되는지"만 검증했음. 이 스크립트는
임계값 자체를 바꿔가며 어떤 조합이 연 단위 holdout 점수를 가장 높이는지
탐색한다.

주의(누수 방지): isotonic 파워커브는 train_raw(2024년 이전)로만 fit하고,
플래그도 train_raw에만 적용한다. holdout(2024년 전체)은 절대 건드리지 않는다.

실행: python3 scripts/tune_curtailment_threshold.py [group_id ...] [--recipe physics|full]
출력: experiments/baseline_lgbm/group{n}_curtailment_threshold_best.json,
      experiments/baseline_lgbm/curtailment_threshold_log.csv
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import pandas as pd
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
OUT_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DATE = "2024-01-01"

RATIO_GRID = [0.20, 0.25, 0.30, 0.35, 0.40]
WIND_GRID = [6.0, 7.0, 8.0, 9.0]

RECIPE_CHOICE = {1: "physics", 2: "full", 3: "full"}
PARAMS_SOURCE = {1: "yearly", 2: "yearly", 3: "optuna"}


def load_params(group_id: int, recipe: str) -> dict:
    source = PARAMS_SOURCE[group_id]
    path = OUT_DIR / f"group{group_id}_{source}_best_params_{recipe}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_default_wind_features(df)
    df = add_physics_features(df)
    df = add_time_features(df)
    speed_cols = [c for c in df.columns if c.endswith("_speed")]
    df = add_lag_rolling_features(df, cols=speed_cols, lags=[1, 2, 3], windows=[3, 6, 24])
    return df


def flag_curtailment(train_raw: pd.DataFrame, capacity: float, ratio: float, wind_thresh: float) -> pd.Series:
    valid = train_raw.dropna(subset=["scada_mean_ws"])
    curve = IsotonicRegression(y_min=0, y_max=capacity, increasing=True, out_of_bounds="clip")
    curve.fit(valid["scada_mean_ws"], valid["y"])
    est = curve.predict(train_raw["scada_mean_ws"].fillna(0))
    residual_ratio = (train_raw["y"] - est) / capacity
    mask = (residual_ratio < -ratio) & (train_raw["scada_mean_ws"] >= wind_thresh)
    return mask.fillna(False)


def run_group(group_id: int, recipe: str | None = None, ratio_grid=None, wind_grid=None):
    recipe = recipe or RECIPE_CHOICE[group_id]
    ratio_grid = ratio_grid or RATIO_GRID
    wind_grid = wind_grid or WIND_GRID
    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]
    params = load_params(group_id, recipe)

    df = build_group_dataset(group_id, split="train", include_scada=True)
    df = df.dropna(subset=["y"]).reset_index(drop=True)
    train_raw, holdout_raw = time_based_split_by_date(df, split_date=SPLIT_DATE)

    holdout_physics = build_physics_features(holdout_raw) if recipe == "physics" else None

    results = []
    print(f"\n=== group{group_id} (recipe={recipe}) threshold grid ===")
    for ratio in ratio_grid:
        for wind_thresh in wind_grid:
            mask = flag_curtailment(train_raw, capacity, ratio, wind_thresh)
            n_flagged = int(mask.sum())
            cleaned = train_raw[~mask].reset_index(drop=True)

            if recipe == "physics":
                train_df = build_physics_features(cleaned)
                hdf = holdout_physics
            else:
                train_df = build_baseline_features(cleaned)
                hdf = build_baseline_features(holdout_raw)
                curve_models = fit_power_curve_models(train_df, capacity=capacity)
                train_df = apply_power_curve_models(train_df, curve_models)
                hdf = apply_power_curve_models(hdf, curve_models)

            feature_cols = get_feature_cols(train_df)
            bagging_freq = 1 if params.get("bagging_fraction", 1.0) < 1.0 else 0
            model = lgb.LGBMRegressor(**params, random_state=42, bagging_freq=bagging_freq, verbosity=-1)
            model.fit(train_df[feature_cols], train_df["y"])
            pred = model.predict(hdf[feature_cols]).clip(min=0)
            r = validate_single_group(hdf["y"].to_numpy(), pred, group_id=group_id)
            score = 0.5 * r["one_minus_nmae"] + 0.5 * r["ficr"]

            results.append(
                {
                    "group_id": group_id, "recipe": recipe, "ratio": ratio, "wind_thresh": wind_thresh,
                    "n_flagged": n_flagged, "score": score, "nmae": r["nmae"], "ficr": r["ficr"],
                }
            )
            print(f"  ratio={ratio:.2f} wind={wind_thresh:.1f} n_flagged={n_flagged:>4d} score={score:.4f} nmae={r['nmae']:.4f} ficr={r['ficr']:.4f}")

    best = max(results, key=lambda r: r["score"])
    print(f"[group{group_id}/{recipe}] BEST ratio={best['ratio']} wind={best['wind_thresh']} score={best['score']:.4f}")

    out_path = OUT_DIR / f"group{group_id}_curtailment_threshold_best.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)

    log_path = OUT_DIR / "curtailment_threshold_log.csv"
    log_df = pd.DataFrame(results)
    if log_path.exists():
        log_df = pd.concat([pd.read_csv(log_path), log_df], ignore_index=True)
    log_df.to_csv(log_path, index=False)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--recipe", choices=["physics", "full"], default=None)
    ap.add_argument("--ratios", type=float, nargs="*", default=None)
    ap.add_argument("--winds", type=float, nargs="*", default=None)
    args = ap.parse_args()
    for gid in args.groups:
        run_group(gid, recipe=args.recipe, ratio_grid=args.ratios, wind_grid=args.winds)


if __name__ == "__main__":
    main()
