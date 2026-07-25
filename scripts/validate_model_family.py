"""
LightGBM vs XGBoost vs CatBoost 모델 계열 비교 (연 단위 holdout, 커틀먼트 제거 반영).

scripts/train_final.py와 동일한 그룹별 레시피/커틀먼트 제거를 적용하고,
같은 feature로 세 boosting 라이브러리를 학습해 holdout 성능과 블렌딩 효과를
비교한다. 구조가 다른 모델은 physics/full 레시피 블렌딩(+0.0002, 거의
무의미)보다 더 큰 다양성을 만들어낼 가능성이 있다.

실행: (레포 루트에서) python3 scripts/validate_model_family.py [group_id ...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import catboost as cb
import lightgbm as lgb
import numpy as np
import xgboost as xgb

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
SPLIT_DATE = "2024-01-01"

RECIPE_CHOICE = {1: "physics", 2: "full", 3: "full"}
PARAMS_SOURCE = {1: "yearly", 2: "yearly", 3: "optuna"}


def build_physics_features(df):
    df = add_default_wind_features(df)
    df = add_physics_features(df)
    df = add_time_features(df)
    speed_cols = [c for c in df.columns if c.endswith("_speed")]
    df = add_lag_rolling_features(df, cols=speed_cols, lags=[1, 2, 3], windows=[3, 6, 24])
    return df


def load_lgbm_params(group_id: int, recipe: str) -> dict:
    source = PARAMS_SOURCE[group_id]
    path = OUT_DIR / f"group{group_id}_{source}_best_params_{recipe}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_group(group_id: int):
    recipe = RECIPE_CHOICE[group_id]
    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]

    df = build_group_dataset(group_id, split="train")
    df = df.dropna(subset=["y"]).reset_index(drop=True)
    train_raw, holdout_raw = time_based_split_by_date(df, split_date=SPLIT_DATE)
    train_raw, n_removed = remove_curtailment(train_raw, capacity=capacity)
    print(f"\n=== group{group_id} (recipe={recipe}, 커틀먼트 {n_removed}행 제거) ===")

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
    X_train, y_train = train_df[feature_cols], train_df["y"]
    X_holdout, y_holdout = holdout_df[feature_cols], holdout_df["y"].to_numpy()

    def score_of(pred):
        r = validate_single_group(y_holdout, pred, group_id=group_id)
        return 0.5 * r["one_minus_nmae"] + 0.5 * r["ficr"], r

    preds = {}

    # LightGBM (기존 최적 파라미터)
    lgb_params = load_lgbm_params(group_id, recipe)
    bagging_freq = 1 if lgb_params.get("bagging_fraction", 1.0) < 1.0 else 0
    m_lgb = lgb.LGBMRegressor(**lgb_params, random_state=42, bagging_freq=bagging_freq, verbosity=-1)
    m_lgb.fit(X_train, y_train)
    preds["lightgbm"] = m_lgb.predict(X_holdout).clip(min=0)

    # XGBoost (기본형 파라미터, 비슷한 복잡도)
    m_xgb = xgb.XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42,
        tree_method="hist", verbosity=0,
    )
    m_xgb.fit(X_train, y_train)
    preds["xgboost"] = m_xgb.predict(X_holdout).clip(min=0)

    # CatBoost (기본형 파라미터)
    m_cb = cb.CatBoostRegressor(
        iterations=400, learning_rate=0.05, depth=6, subsample=0.8,
        bootstrap_type="Bernoulli", reg_lambda=1.0, random_seed=42, verbose=False,
    )
    m_cb.fit(X_train, y_train)
    preds["catboost"] = m_cb.predict(X_holdout).clip(min=0)

    for name, pred in preds.items():
        score, r = score_of(pred)
        print(f"  {name:10s} score={score:.4f} nmae={r['nmae']:.4f} ficr={r['ficr']:.4f}")

    # 3-모델 균등 블렌드 + 최고 2개 블렌드
    names = list(preds.keys())
    scores = {n: score_of(preds[n])[0] for n in names}
    best2 = sorted(names, key=lambda n: -scores[n])[:2]

    blend_all = np.mean([preds[n] for n in names], axis=0)
    s_all, _ = score_of(blend_all)
    print(f"  blend(전체3개 균등) score={s_all:.4f}")

    blend2 = np.mean([preds[n] for n in best2], axis=0)
    s2, _ = score_of(blend2)
    print(f"  blend(상위2개 {best2}) score={s2:.4f}")

    best_single = max(scores.values())
    print(f"  => 단일 최고 {best_single:.4f} 대비 블렌드 최고 {max(s_all, s2):.4f} ({max(s_all, s2)-best_single:+.4f})")


def main():
    groups = [int(a) for a in sys.argv[1:]] or [1, 2, 3]
    for gid in groups:
        run_group(gid)


if __name__ == "__main__":
    main()
