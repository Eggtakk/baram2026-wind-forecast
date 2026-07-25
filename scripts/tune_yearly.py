"""
연 단위 holdout(train=이전 연도 전체, holdout=마지막 연도 전체) 기준 하이퍼파라미터
랜덤서치. 기존 tune_baseline.py는 뒤쪽 20% row-cut(=계절 편중)으로 튜닝해서
실제 test(연중 전체)로 일반화가 안 됐던 것으로 확인됨
(experiments/baseline_lgbm/rated_output_investigation.md 참고). 이 스크립트는
그 결함을 고친 버전.

두 feature 레시피를 각각 지원한다:
  --recipe physics  : add_default_wind_features + add_physics_features + time + lag/rolling
                       (saturation/power-curve 이전, 원래 첫 제출과 가까운 구성)
  --recipe full      : build_baseline_features(saturation 포함) + power curve feature

실행: (레포 루트에서) python3 scripts/tune_yearly.py --group 3 --recipe physics
출력: experiments/baseline_lgbm/group{n}_yearly_best_params_{recipe}.json, yearly_tuning_log.csv
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import pandas as pd

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
SEED = 42

PARAM_GRID = {
    "n_estimators": [200, 400, 600],
    "learning_rate": [0.03, 0.05, 0.08],
    "num_leaves": [15, 31, 63],
    "min_child_samples": [10, 30],
    "feature_fraction": [0.8, 1.0],
    "bagging_fraction": [0.8, 1.0],
}


def build_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_default_wind_features(df)
    df = add_physics_features(df)
    df = add_time_features(df)
    speed_cols = [c for c in df.columns if c.endswith("_speed")]
    df = add_lag_rolling_features(df, cols=speed_cols, lags=[1, 2, 3], windows=[3, 6, 24])
    return df


def sample_params(rng: random.Random) -> dict:
    return {k: rng.choice(v) for k, v in PARAM_GRID.items()}


def tune_group(group_id: int, recipe: str, n_trials: int, rng: random.Random):
    df = build_group_dataset(group_id, split="train")
    df = df.dropna(subset=["y"]).reset_index(drop=True)
    train_raw, holdout_raw = time_based_split_by_date(df, split_date=SPLIT_DATE)

    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]

    if recipe == "physics":
        train_df = build_physics_features(train_raw)
        holdout_df = build_physics_features(holdout_raw)
    elif recipe == "full":
        train_df = build_baseline_features(train_raw)
        holdout_df = build_baseline_features(holdout_raw)
        curve_models = fit_power_curve_models(train_df, capacity=capacity)
        train_df = apply_power_curve_models(train_df, curve_models)
        holdout_df = apply_power_curve_models(holdout_df, curve_models)
    else:
        raise ValueError(recipe)

    feature_cols = get_feature_cols(train_df)

    trials = []
    seen = set()
    while len(trials) < n_trials:
        params = sample_params(rng)
        key = tuple(sorted(params.items()))
        if key in seen:
            continue
        seen.add(key)

        bagging_freq = 1 if params["bagging_fraction"] < 1.0 else 0
        model = lgb.LGBMRegressor(**params, random_state=SEED, verbosity=-1, bagging_freq=bagging_freq)
        model.fit(train_df[feature_cols], train_df["y"])
        pred = model.predict(holdout_df[feature_cols]).clip(min=0)
        r = validate_single_group(holdout_df["y"].to_numpy(), pred, group_id=group_id)
        score = 0.5 * r["one_minus_nmae"] + 0.5 * r["ficr"]

        trial = {"group_id": group_id, "recipe": recipe, "score": score, "nmae": r["nmae"], "ficr": r["ficr"], **params}
        trials.append(trial)
        print(f"  [group{group_id}/{recipe}] trial {len(trials)}/{n_trials}: score={score:.4f} nmae={r['nmae']:.4f} ficr={r['ficr']:.4f}")

    best = max(trials, key=lambda t: t["score"])
    return best, trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=int, required=True)
    ap.add_argument("--recipe", choices=["physics", "full"], required=True)
    ap.add_argument("--trials", type=int, default=6)
    args = ap.parse_args()

    rng = random.Random(SEED)
    best, trials = tune_group(args.group, args.recipe, args.trials, rng)

    best_params = {k: best[k] for k in PARAM_GRID}
    out_path = OUT_DIR / f"group{args.group}_yearly_best_params_{args.recipe}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    print(f"\n[group{args.group}/{args.recipe}] BEST score={best['score']:.4f} nmae={best['nmae']:.4f} ficr={best['ficr']:.4f}")
    print(f"params={best_params}")
    print(f"saved: {out_path}")

    log_path = OUT_DIR / "yearly_tuning_log.csv"
    log_df = pd.DataFrame(trials)
    if log_path.exists():
        log_df = pd.concat([pd.read_csv(log_path), log_df], ignore_index=True)
    log_df.to_csv(log_path, index=False)


if __name__ == "__main__":
    main()
