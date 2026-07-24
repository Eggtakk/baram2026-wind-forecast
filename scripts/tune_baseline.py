"""
LightGBM 하이퍼파라미터 랜덤서치 (그룹별, 시간순 holdout 기준).

RMSE가 아니라 대회 공식 산식(local_group_score = 0.5*(1-NMAE)+0.5*FICR)을
직접 기준으로 최적 파라미터를 고른다 — 이 대회에서 실제로 중요한 건 오차의
크기(RMSE)가 아니라 오차율 구간(6%/8%)과 설비용량 대비 오차이기 때문.

실행: (레포 루트에서) python3 scripts/tune_baseline.py
출력: experiments/baseline_lgbm/group{n}_best_params.json, tuning_log.csv
"""
import itertools
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import pandas as pd

from src.features import build_baseline_features, get_feature_cols
from src.metrics import CAPACITY_KWH, validate_single_group
from src.power_curve import apply_power_curve_models, fit_power_curve_models
from src.preprocess import build_group_dataset
from src.validation import time_based_split

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "baseline_lgbm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HOLDOUT_RATIO = 0.2
N_TRIALS = 10
SEED = 42

PARAM_GRID = {
    "n_estimators": [200, 400, 600],
    "learning_rate": [0.03, 0.05, 0.08],
    "num_leaves": [15, 31, 63],
    "min_child_samples": [10, 30],
    "feature_fraction": [0.8, 1.0],
    "bagging_fraction": [0.8, 1.0],
}


def sample_params(rng: random.Random) -> dict:
    return {k: rng.choice(v) for k, v in PARAM_GRID.items()}


def tune_group(group_id: int, rng: random.Random) -> tuple[dict, list[dict]]:
    df = build_group_dataset(group_id, split="train")
    df = build_baseline_features(df)
    df = df.dropna(subset=["y"]).reset_index(drop=True)

    train_df, holdout_df = time_based_split(df, holdout_ratio=HOLDOUT_RATIO)

    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]
    curve_models = fit_power_curve_models(train_df, capacity=capacity)
    train_df = apply_power_curve_models(train_df, curve_models)
    holdout_df = apply_power_curve_models(holdout_df, curve_models)

    feature_cols = get_feature_cols(train_df)

    trials = []
    seen = set()
    while len(trials) < N_TRIALS:
        params = sample_params(rng)
        key = tuple(sorted(params.items()))
        if key in seen:
            continue
        seen.add(key)

        model = lgb.LGBMRegressor(
            **params, random_state=SEED, verbosity=-1, bagging_freq=1 if params["bagging_fraction"] < 1.0 else 0
        )
        model.fit(train_df[feature_cols], train_df["y"])
        pred = model.predict(holdout_df[feature_cols]).clip(min=0)
        r = validate_single_group(holdout_df["y"].to_numpy(), pred, group_id=group_id)
        score = 0.5 * r["one_minus_nmae"] + 0.5 * r["ficr"]

        trial = {"group_id": group_id, "score": score, "nmae": r["nmae"], "ficr": r["ficr"], **params}
        trials.append(trial)
        print(f"  [group{group_id}] trial {len(trials)}/{N_TRIALS}: score={score:.4f} nmae={r['nmae']:.4f} ficr={r['ficr']:.4f} params={params}")

    best = max(trials, key=lambda t: t["score"])
    return best, trials


def main():
    groups = [int(a) for a in sys.argv[1:]] or [1, 2, 3]
    rng = random.Random(SEED)
    all_trials = []
    for gid in groups:
        print(f"\n=== group{gid} tuning ({N_TRIALS} trials) ===")
        best, trials = tune_group(gid, rng)
        all_trials.extend(trials)

        best_params = {k: best[k] for k in PARAM_GRID}
        with open(OUT_DIR / f"group{gid}_best_params.json", "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2)
        print(f"[group{gid}] BEST score={best['score']:.4f} params={best_params}")

    log_path = OUT_DIR / "tuning_log.csv"
    log_df = pd.DataFrame(all_trials)
    if log_path.exists():
        log_df = pd.concat([pd.read_csv(log_path), log_df], ignore_index=True)
    log_df.to_csv(log_path, index=False)
    print(f"\n전체 시도 로그 저장: {log_path}")


if __name__ == "__main__":
    main()
