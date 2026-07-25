"""
scripts/train_final.py + 모델 계열 블렌딩(scripts/validate_model_family.py 검증 결과)을
합친 최종 프로덕션 학습 스크립트.

그룹별 최종 구성:
  group1: LightGBM 단일 (physics, 커틀먼트 제거) -- 블렌딩이 오히려 소폭 악화되어 단일 유지
  group2: XGBoost 50% + LightGBM 50% (full, 커틀먼트 제거) -- 0.6508 -> 0.6547
  group3: LightGBM 50% + CatBoost 50% (full, 커틀먼트 제거) -- 0.5790 -> 0.5801

실행: (레포 루트에서) python3 scripts/train_final2.py
출력: experiments/baseline_lgbm/group{n}_final2_{model}.pkl,
      experiments/baseline_lgbm/group{n}_final2_meta.json,
      (full 레시피 그룹만) group{n}_final2_power_curve.pkl
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import catboost as cb
import joblib
import lightgbm as lgb
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
from src.metrics import CAPACITY_KWH
from src.power_curve import apply_power_curve_models, fit_power_curve_models, save_power_curve_models
from src.preprocess import build_group_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "baseline_lgbm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RECIPE_CHOICE = {1: "physics", 2: "full", 3: "full"}
PARAMS_SOURCE = {1: "yearly", 2: "yearly", 3: "optuna"}
BLEND_CONFIG = {
    1: [("lightgbm", 1.0)],
    2: [("xgboost", 0.5), ("lightgbm", 0.5)],
    3: [("lightgbm", 0.5), ("catboost", 0.5)],
}

XGB_PARAMS = dict(
    n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, tree_method="hist",
)
CB_PARAMS = dict(
    iterations=400, learning_rate=0.05, depth=6, subsample=0.8,
    bootstrap_type="Bernoulli", reg_lambda=1.0, verbose=False,
)


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


def fit_model(name: str, group_id: int, recipe: str, X, y):
    if name == "lightgbm":
        params = load_lgbm_params(group_id, recipe)
        bagging_freq = 1 if params.get("bagging_fraction", 1.0) < 1.0 else 0
        model = lgb.LGBMRegressor(**params, random_state=42, bagging_freq=bagging_freq, verbosity=-1)
    elif name == "xgboost":
        model = xgb.XGBRegressor(**XGB_PARAMS, random_state=42, verbosity=0)
    elif name == "catboost":
        model = cb.CatBoostRegressor(**CB_PARAMS, random_seed=42)
    else:
        raise ValueError(name)
    model.fit(X, y)
    return model


def train_group(group_id: int) -> dict:
    recipe = RECIPE_CHOICE[group_id]
    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]

    df = build_group_dataset(group_id, split="train")
    df = df.dropna(subset=["y"]).reset_index(drop=True)
    cleaned_df, n_removed = remove_curtailment(df, capacity=capacity)
    print(f"[group{group_id}] 커틀먼트 의심 {n_removed}행 제거 ({n_removed/len(df)*100:.2f}%)")

    if recipe == "physics":
        feat_df = build_physics_features(cleaned_df)
    else:
        feat_df = build_baseline_features(cleaned_df)
        curve_models = fit_power_curve_models(feat_df, capacity=capacity)
        feat_df = apply_power_curve_models(feat_df, curve_models)
        save_power_curve_models(curve_models, OUT_DIR / f"group{group_id}_final2_power_curve.pkl")

    feature_cols = get_feature_cols(feat_df)
    X, y = feat_df[feature_cols], feat_df["y"]

    blend = BLEND_CONFIG[group_id]
    for name, weight in blend:
        model = fit_model(name, group_id, recipe, X, y)
        joblib.dump(model, OUT_DIR / f"group{group_id}_final2_{name}.pkl")
        print(f"  [group{group_id}] {name}(weight={weight}) trained on {len(feat_df)} rows")

    meta = {
        "group_id": group_id,
        "recipe": recipe,
        "blend": blend,
        "feature_cols": feature_cols,
        "n_train_rows": len(feat_df),
        "n_curtailment_removed": n_removed,
    }
    with open(OUT_DIR / f"group{group_id}_final2_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def main():
    for gid in [1, 2, 3]:
        train_group(gid)
    print(f"\nAll final2 models saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
