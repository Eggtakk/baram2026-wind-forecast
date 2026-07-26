"""
XGBoost / CatBoost를 LightGBM과 동일한 수준으로 제대로 튜닝하기 위한
Optuna(TPE) 기반 하이퍼파라미터 탐색 — 연 단위 holdout 기준.

배경: scripts/validate_model_family.py에서 XGBoost/CatBoost를 "기본형"
파라미터로만 학습해 블렌딩했다가 실제 리더보드에서 역효과가 났음
(0.61034 -> 0.60894). LightGBM은 그리드+Optuna로 충분히 탐색된 반면
XGBoost/CatBoost는 전혀 튜닝되지 않아 단일 연도 holdout에 과적합됐던
것으로 추정됨. 이 스크립트는 그 격차를 없애기 위해 XGBoost/CatBoost도
동일하게 Optuna로 탐색한다.

scripts/train_final.py와 동일하게 커틀먼트 제거(기본 임계값 0.30/8.0)를
train에만 적용한 뒤, 그룹별 확정 레시피(RECIPE_CHOICE)로 학습한다.

SQLite에 study를 저장(load_if_exists=True)하므로 여러 번 나눠 호출해도
시도가 누적된다(45초 샌드박스 제약 대응). DB는 마운트 폴더가 아닌
로컬 임시 경로에 둔다(파일 락 이슈 회피).

실행: python3 scripts/tune_family_optuna.py --group 3 --model xgboost --trials 8
      python3 scripts/tune_family_optuna.py --group 3 --model catboost --trials 8
출력: experiments/baseline_lgbm/group{n}_{model}_optuna_best_params.json
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import catboost as cb
import optuna
import xgboost as xgb
from optuna.samplers import TPESampler

from src.data_cleaning import remove_curtailment
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

LOCAL_DB_DIR = Path(tempfile.gettempdir()) / "baram_optuna"
LOCAL_DB_DIR.mkdir(parents=True, exist_ok=True)
STORAGE = f"sqlite:///{LOCAL_DB_DIR / 'family_optuna_studies.db'}"
SPLIT_DATE = "2024-01-01"
SEED = 42

RECIPE_CHOICE = {1: "physics", 2: "full", 3: "full"}


def build_physics_features(df):
    df = add_default_wind_features(df)
    df = add_physics_features(df)
    df = add_time_features(df)
    speed_cols = [c for c in df.columns if c.endswith("_speed")]
    df = add_lag_rolling_features(df, cols=speed_cols, lags=[1, 2, 3], windows=[3, 6, 24])
    return df


def load_data(group_id: int):
    recipe = RECIPE_CHOICE[group_id]
    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]

    df = build_group_dataset(group_id, split="train")
    df = df.dropna(subset=["y"]).reset_index(drop=True)
    train_raw, holdout_raw = time_based_split_by_date(df, split_date=SPLIT_DATE)
    train_raw, n_removed = remove_curtailment(train_raw, capacity=capacity)

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
    return train_df, holdout_df, feature_cols, recipe


def make_objective_xgb(group_id, train_df, holdout_df, feature_cols):
    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
        model = xgb.XGBRegressor(**params, random_state=SEED, tree_method="hist", verbosity=0)
        model.fit(train_df[feature_cols], train_df["y"])
        pred = model.predict(holdout_df[feature_cols]).clip(min=0)
        r = validate_single_group(holdout_df["y"].to_numpy(), pred, group_id=group_id)
        score = 0.5 * r["one_minus_nmae"] + 0.5 * r["ficr"]
        trial.set_user_attr("nmae", r["nmae"])
        trial.set_user_attr("ficr", r["ficr"])
        return score

    return objective


def make_objective_catboost(group_id, train_df, holdout_df, feature_cols):
    def objective(trial: optuna.Trial) -> float:
        params = dict(
            iterations=trial.suggest_int("iterations", 100, 800, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            depth=trial.suggest_int("depth", 3, 10),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 1, 50),
        )
        model = cb.CatBoostRegressor(
            **params, bootstrap_type="Bernoulli", random_seed=SEED, verbose=False
        )
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
    ap.add_argument("--model", choices=["xgboost", "catboost"], required=True)
    ap.add_argument("--trials", type=int, default=8)
    args = ap.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train_df, holdout_df, feature_cols, recipe = load_data(args.group)

    study_name = f"group{args.group}_{args.model}"
    study = optuna.create_study(
        study_name=study_name,
        storage=STORAGE,
        direction="maximize",
        sampler=TPESampler(seed=SEED),
        load_if_exists=True,
    )

    if args.model == "xgboost":
        objective = make_objective_xgb(args.group, train_df, holdout_df, feature_cols)
    else:
        objective = make_objective_catboost(args.group, train_df, holdout_df, feature_cols)

    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    best = study.best_trial
    print(
        f"[{study_name}] recipe={recipe} 누적 시도 수: {len(study.trials)}  BEST score={best.value:.4f} "
        f"nmae={best.user_attrs['nmae']:.4f} ficr={best.user_attrs['ficr']:.4f}"
    )
    print(f"params={best.params}")

    out_path = OUT_DIR / f"group{args.group}_{args.model}_optuna_best_params.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best.params, f, indent=2)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
