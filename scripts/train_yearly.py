"""
연 단위 holdout으로 재검증한 하이퍼파라미터/레시피로 최종 프로덕션 모델을 학습.

scripts/tune_yearly.py 결과(experiments/baseline_lgbm/group{n}_yearly_best_params_{recipe}.json)
기준으로 그룹별로 더 점수가 높았던 레시피를 선택:
  group1: full (saturation+power curve)  yearly holdout 0.6102 > physics 0.6091
  group2: full (saturation+power curve)  yearly holdout 0.6493 > physics 0.6467
  group3: physics-only                    yearly holdout 0.5621 > full 0.5611

최종 모델은 전체 train 데이터(2024년 포함, test 직전까지 전부)로 학습한다 —
튜닝 단계에서는 2024년을 holdout으로 뗐지만, 프로덕션 모델은 가진 데이터를
전부 써야 하므로 다시 합쳐서 학습.

실행: (레포 루트에서) python3 scripts/train_yearly.py
출력: experiments/baseline_lgbm/group{n}_yearly_model.pkl, group{n}_yearly_meta.json,
      (recipe=full인 그룹만) group{n}_yearly_power_curve.pkl
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import lightgbm as lgb

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

RECIPE_CHOICE = {1: "full", 2: "full", 3: "physics"}


def build_physics_features(df):
    df = add_default_wind_features(df)
    df = add_physics_features(df)
    df = add_time_features(df)
    speed_cols = [c for c in df.columns if c.endswith("_speed")]
    df = add_lag_rolling_features(df, cols=speed_cols, lags=[1, 2, 3], windows=[3, 6, 24])
    return df


def load_yearly_params(group_id: int, recipe: str) -> dict:
    path = OUT_DIR / f"group{group_id}_yearly_best_params_{recipe}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def train_group(group_id: int) -> dict:
    recipe = RECIPE_CHOICE[group_id]
    df = build_group_dataset(group_id, split="train")
    df = df.dropna(subset=["y"]).reset_index(drop=True)

    if recipe == "physics":
        df = build_physics_features(df)
    else:
        df = build_baseline_features(df)
        capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]
        curve_models = fit_power_curve_models(df, capacity=capacity)
        df = apply_power_curve_models(df, curve_models)
        save_power_curve_models(curve_models, OUT_DIR / f"group{group_id}_yearly_power_curve.pkl")

    feature_cols = get_feature_cols(df)
    params = load_yearly_params(group_id, recipe)
    bagging_freq = 1 if params.get("bagging_fraction", 1.0) < 1.0 else 0

    model = lgb.LGBMRegressor(**params, random_state=42, bagging_freq=bagging_freq, verbosity=-1)
    model.fit(df[feature_cols], df["y"])

    model_path = OUT_DIR / f"group{group_id}_yearly_model.pkl"
    joblib.dump(model, model_path)

    meta = {
        "group_id": group_id,
        "recipe": recipe,
        "feature_cols": feature_cols,
        "params": params,
        "n_train_rows": len(df),
        "train_range": [str(df["forecast_kst_dtm"].min()), str(df["forecast_kst_dtm"].max())],
    }
    with open(OUT_DIR / f"group{group_id}_yearly_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[group{group_id}] recipe={recipe} trained on {len(df)} rows, {len(feature_cols)} features -> {model_path}")
    return meta


def main():
    for gid in [1, 2, 3]:
        train_group(gid)
    print(f"\nAll yearly models saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
