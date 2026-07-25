"""
Optuna(TPE, 베이지안 탐색) 기반 하이퍼파라미터 탐색 — 연 단위 holdout 기준.

scripts/tune_yearly.py(고정 그리드 + 랜덤서치)의 후속. 이산 그리드 대신
연속 구간에서 탐색하고, 이전 시도 결과를 반영해 다음 시도를 더 똑똑하게
고른다(TPE sampler). SQLite에 study를 저장하므로(load_if_exists=True)
샌드박스 45초 제약 때문에 여러 번에 나눠 호출해도 시도가 계속 누적된다 —
매번 처음부터 다시 세는 랜덤서치와 달리 이어서 탐색 가능.

실행: (레포 루트에서) python3 scripts/tune_optuna.py --group 3 --recipe physics --trials 15
  (같은 명령을 여러 번 실행하면 시도가 계속 누적됨)
출력: experiments/baseline_lgbm/optuna_studies.db (SQLite),
      experiments/baseline_lgbm/group{n}_optuna_best_params_{recipe}.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler

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

# 주의: sqlite DB는 마운트된 레포 폴더(네트워크/FUSE 파일시스템)에 두면 파일 락
# 문제로 "disk I/O error"가 난다. 로컬(비마운트) 경로에 두고, 최종 best
# params만 JSON으로 레포에 저장한다.
import tempfile

LOCAL_DB_DIR = Path(tempfile.gettempdir()) / "baram_optuna"
LOCAL_DB_DIR.mkdir(parents=True, exist_ok=True)
STORAGE = f"sqlite:///{LOCAL_DB_DIR / 'optuna_studies.db'}"
SPLIT_DATE = "2024-01-01"
SEED = 42


def build_physics_features(df):
    df = add_default_wind_features(df)
    df = add_physics_features(df)
    df = add_time_features(df)
    speed_cols = [c for c in df.columns if c.endswith("_speed")]
    df = add_lag_rolling_features(df, cols=speed_cols, lags=[1, 2, 3], windows=[3, 6, 24])
    return df


def load_data(group_id: int, recipe: str):
    df = build_group_dataset(group_id, split="train")
    df = df.dropna(subset=["y"]).reset_index(drop=True)
    train_raw, holdout_raw = time_based_split_by_date(df, split_date=SPLIT_DATE)
    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]

    if recipe == "physics":
        train_df = build_physics_features(train_raw)
        holdout_df = build_physics_features(holdout_raw)
    else:
        train_df = build_baseline_features(train_raw)
        holdout_df = build_baseline_features(holdout_raw)
        curve_models = fit_power_curve_models(train_df, capacity=capacity)
        train_df = apply_power_curve_models(train_df, curve_models)
        holdout_df = apply_power_curve_models(holdout_df, curve_models)

    feature_cols = get_feature_cols(train_df)
    return train_df, holdout_df, feature_cols


def make_objective(group_id: int, train_df, holdout_df, feature_cols):
    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            num_leaves=trial.suggest_int("num_leaves", 7, 127),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
        bagging_freq = 1 if params["bagging_fraction"] < 1.0 else 0
        model = lgb.LGBMRegressor(**params, random_state=SEED, verbosity=-1, bagging_freq=bagging_freq)
        model.fit(train_df[feature_cols], train_df["y"])
        pred = model.predict(holdout_df[feature_cols]).clip(min=0)
        r = validate_single_group(holdout_df["y"].to_numpy(), pred, group_id=group_id)
        score = 0.5 * r["one_minus_nmae"] + 0.5 * r["ficr"]
        trial.set_user_attr("nmae", r["nmae"])
        trial.set_user_attr("ficr", r["ficr"])
        return score

    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=int, required=True)
    ap.add_argument("--recipe", choices=["physics", "full"], required=True)
    ap.add_argument("--trials", type=int, default=8)
    args = ap.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train_df, holdout_df, feature_cols = load_data(args.group, args.recipe)

    study_name = f"group{args.group}_{args.recipe}"
    study = optuna.create_study(
        study_name=study_name,
        storage=STORAGE,
        direction="maximize",
        sampler=TPESampler(seed=SEED),
        load_if_exists=True,
    )
    objective = make_objective(args.group, train_df, holdout_df, feature_cols)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    best = study.best_trial
    print(
        f"[{study_name}] 누적 시도 수: {len(study.trials)}  BEST score={best.value:.4f} "
        f"nmae={best.user_attrs['nmae']:.4f} ficr={best.user_attrs['ficr']:.4f}"
    )
    print(f"params={best.params}")

    out_path = OUT_DIR / f"group{args.group}_optuna_best_params_{args.recipe}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best.params, f, indent=2)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
