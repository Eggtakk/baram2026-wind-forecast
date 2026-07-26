"""
커틀먼트 임계값 튜닝(scripts/tune_curtailment_threshold.py)에서 찾은 그룹별
최적 임계값으로 train에서 의심 구간을 제거한 뒤, 그 정리된 데이터 기준으로
LightGBM 하이퍼파라미터를 처음부터 다시 그리드서치한다.

기존 tune_yearly.py의 best_params(json)는 전부 "정리 전" 데이터로 찾은 것을
재사용해왔음 — 커틀먼트를 제거하면 학습 데이터의 분포/난이도가 바뀌므로
파라미터도 다시 찾아야 한다는 가설을 검증.

임계값 소스: experiments/baseline_lgbm/group{n}_curtailment_threshold_best.json
(없으면 src.data_cleaning의 기본값 RESIDUAL_THRESHOLD_RATIO=0.30, HIGH_WIND_THRESHOLD=8.0 사용)

실행: python3 scripts/tune_yearly_clean.py --group 3 --recipe full [--trials 20]
출력: experiments/baseline_lgbm/group{n}_yearly_clean_best_params_{recipe}.json,
      experiments/baseline_lgbm/yearly_clean_tuning_log.csv
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.data_cleaning import HIGH_WIND_THRESHOLD, RESIDUAL_THRESHOLD_RATIO
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


def load_threshold(group_id: int) -> tuple[float, float]:
    path = OUT_DIR / f"group{group_id}_curtailment_threshold_best.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            best = json.load(f)
        return best["ratio"], best["wind_thresh"]
    return RESIDUAL_THRESHOLD_RATIO, HIGH_WIND_THRESHOLD


def flag_curtailment(train_raw: pd.DataFrame, capacity: float, ratio: float, wind_thresh: float) -> pd.Series:
    valid = train_raw.dropna(subset=["scada_mean_ws"])
    curve = IsotonicRegression(y_min=0, y_max=capacity, increasing=True, out_of_bounds="clip")
    curve.fit(valid["scada_mean_ws"], valid["y"])
    est = curve.predict(train_raw["scada_mean_ws"].fillna(0))
    residual_ratio = (train_raw["y"] - est) / capacity
    mask = (residual_ratio < -ratio) & (train_raw["scada_mean_ws"] >= wind_thresh)
    return mask.fillna(False)


def build_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_default_wind_features(df)
    df = add_physics_features(df)
    df = add_time_features(df)
    speed_cols = [c for c in df.columns if c.endswith("_speed")]
    df = add_lag_rolling_features(df, cols=speed_cols, lags=[1, 2, 3], windows=[3, 6, 24])
    return df


def sample_params(rng: random.Random) -> dict:
    return {k: rng.choice(v) for k, v in PARAM_GRID.items()}


def tune_group(group_id: int, recipe: str, n_trials: int, rng: random.Random, seen: set, trials_acc: list):
    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]
    ratio, wind_thresh = load_threshold(group_id)

    df = build_group_dataset(group_id, split="train", include_scada=True)
    df = df.dropna(subset=["y"]).reset_index(drop=True)
    train_raw, holdout_raw = time_based_split_by_date(df, split_date=SPLIT_DATE)

    mask = flag_curtailment(train_raw, capacity, ratio, wind_thresh)
    n_flagged = int(mask.sum())
    cleaned_train = train_raw[~mask].reset_index(drop=True)
    print(f"[group{group_id}/{recipe}] threshold ratio={ratio} wind={wind_thresh} -> {n_flagged}행 제거 ({n_flagged/len(train_raw)*100:.2f}%)")

    if recipe == "physics":
        train_df = build_physics_features(cleaned_train)
        holdout_df = build_physics_features(holdout_raw)
    elif recipe == "full":
        train_df = build_baseline_features(cleaned_train)
        holdout_df = build_baseline_features(holdout_raw)
        curve_models = fit_power_curve_models(train_df, capacity=capacity)
        train_df = apply_power_curve_models(train_df, curve_models)
        holdout_df = apply_power_curve_models(holdout_df, curve_models)
    else:
        raise ValueError(recipe)

    feature_cols = get_feature_cols(train_df)

    while len(trials_acc) < n_trials:
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
        trials_acc.append(trial)
        print(f"  [group{group_id}/{recipe}] trial {len(trials_acc)}/{n_trials}: score={score:.4f} nmae={r['nmae']:.4f} ficr={r['ficr']:.4f}")

    best = max(trials_acc, key=lambda t: t["score"])
    return best, trials_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=int, required=True)
    ap.add_argument("--recipe", choices=["physics", "full"], required=True)
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--resume", action="store_true", help="기존 로그에서 이어서 누적 (같은 group/recipe)")
    args = ap.parse_args()

    log_path = OUT_DIR / "yearly_clean_tuning_log.csv"
    seen = set()
    trials_acc = []
    if args.resume and log_path.exists():
        prev = pd.read_csv(log_path)
        prev = prev[(prev["group_id"] == args.group) & (prev["recipe"] == args.recipe)]
        for _, row in prev.iterrows():
            params = {k: row[k] for k in PARAM_GRID}
            trials_acc.append({**row.to_dict()})
            seen.add(tuple(sorted(params.items())))
        print(f"[resume] {len(trials_acc)}개 기존 trial 로드")

    rng = random.Random(SEED + len(trials_acc))
    best, trials_acc = tune_group(args.group, args.recipe, args.trials, rng, seen, trials_acc)

    best_params = {k: best[k] for k in PARAM_GRID}
    out_path = OUT_DIR / f"group{args.group}_yearly_clean_best_params_{args.recipe}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    print(f"\n[group{args.group}/{args.recipe}] BEST score={best['score']:.4f} nmae={best['nmae']:.4f} ficr={best['ficr']:.4f}")
    print(f"params={best_params}")
    print(f"saved: {out_path}")

    log_df = pd.DataFrame(trials_acc)
    if log_path.exists():
        prev_all = pd.read_csv(log_path)
        prev_other = prev_all[~((prev_all["group_id"] == args.group) & (prev_all["recipe"] == args.recipe))]
        log_df = pd.concat([prev_other, log_df], ignore_index=True)
    log_df.to_csv(log_path, index=False)


if __name__ == "__main__":
    main()
