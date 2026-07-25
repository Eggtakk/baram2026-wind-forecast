"""
오늘 하루 검증한 모든 개선사항을 종합한 최종 프로덕션 학습 스크립트.

그룹별 최종 선택(연 단위 holdout + 커틀먼트 제거 기준, 상세 근거는
experiments/baseline_lgbm/rated_output_investigation.md):
  group1: physics-only 레시피, 그리드 탐색 파라미터, 커틀먼트 제거    (0.6091 -> 0.6113)
  group2: full(saturation+파워커브) 레시피, 그리드 탐색 파라미터, 커틀먼트 제거 (0.6493 -> 0.6508)
  group3: full(saturation+파워커브) 레시피, Optuna 탐색 파라미터, 커틀먼트 제거 (0.5607 -> 0.5790)

커틀먼트 제거(src.data_cleaning)는 SCADA 실측 풍속 기준 경험적 파워커브
대비 발전량이 크게 낮은 시간대(착빙/커틀먼트/고장 의심)를 학습 데이터에서만
제거한다 — holdout/test는 건드리지 않는다.

실행: (레포 루트에서) python3 scripts/train_final.py
출력: experiments/baseline_lgbm/group{n}_final_model.pkl, group{n}_final_meta.json,
      (full 레시피 그룹만) group{n}_final_power_curve.pkl
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import lightgbm as lgb

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


def build_physics_features(df):
    df = add_default_wind_features(df)
    df = add_physics_features(df)
    df = add_time_features(df)
    speed_cols = [c for c in df.columns if c.endswith("_speed")]
    df = add_lag_rolling_features(df, cols=speed_cols, lags=[1, 2, 3], windows=[3, 6, 24])
    return df


def load_params(group_id: int, recipe: str) -> dict:
    source = PARAMS_SOURCE[group_id]
    path = OUT_DIR / f"group{group_id}_{source}_best_params_{recipe}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def train_group(group_id: int) -> dict:
    recipe = RECIPE_CHOICE[group_id]
    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]

    df = build_group_dataset(group_id, split="train")  # include_scada=True by default
    df = df.dropna(subset=["y"]).reset_index(drop=True)

    cleaned_df, n_removed = remove_curtailment(df, capacity=capacity)
    print(f"[group{group_id}] 커틀먼트 의심 {n_removed}행 제거 ({n_removed/len(df)*100:.2f}%)")

    if recipe == "physics":
        feat_df = build_physics_features(cleaned_df)
    else:
        feat_df = build_baseline_features(cleaned_df)
        curve_models = fit_power_curve_models(feat_df, capacity=capacity)
        feat_df = apply_power_curve_models(feat_df, curve_models)
        save_power_curve_models(curve_models, OUT_DIR / f"group{group_id}_final_power_curve.pkl")

    feature_cols = get_feature_cols(feat_df)
    params = load_params(group_id, recipe)
    bagging_freq = 1 if params.get("bagging_fraction", 1.0) < 1.0 else 0

    model = lgb.LGBMRegressor(**params, random_state=42, bagging_freq=bagging_freq, verbosity=-1)
    model.fit(feat_df[feature_cols], feat_df["y"])

    model_path = OUT_DIR / f"group{group_id}_final_model.pkl"
    joblib.dump(model, model_path)

    meta = {
        "group_id": group_id,
        "recipe": recipe,
        "params_source": PARAMS_SOURCE[group_id],
        "feature_cols": feature_cols,
        "params": params,
        "n_train_rows": len(feat_df),
        "n_curtailment_removed": n_removed,
    }
    with open(OUT_DIR / f"group{group_id}_final_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[group{group_id}] recipe={recipe} trained on {len(feat_df)} rows, {len(feature_cols)} features -> {model_path}")
    return meta


def main():
    for gid in [1, 2, 3]:
        train_group(gid)
    print(f"\nAll final models saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
