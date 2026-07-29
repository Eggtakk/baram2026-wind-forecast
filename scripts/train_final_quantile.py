"""
30번 섹션 — LightGBM 네이티브 quantile objective(alpha>0.5)로 group1/group2의
stage-1을 재학습한다. `scripts/train_final_ficr.py`(FICR-shaped objective
버전)와 병렬 구조 — 프로덕션 레시피/하이퍼파라미터는 그대로 두고 objective만
`{"objective": "quantile", "alpha": alpha}`로 교체.

alpha는 LOYO 그리드서치(scripts/validate_quantile_objective.py, 30번 섹션)
결과 중 "델타는 크되 fold-std가 과도하게 커지기 전"인 보수적인 값을 선택:
group1=0.60(delta+0.0100), group2=0.65(delta+0.0232, fold-std가 baseline보다도
낮음).

group3는 이번에도 제외(FICR objective 때와 동일한 이유로 별도 검증 필요 —
아직 미착수).

실행: (레포 루트에서) python3 scripts/train_final_quantile.py
출력: experiments/baseline_lgbm/group{n}_final_quantile_model.pkl,
      group{n}_final_quantile_meta.json,
      (group2, full 레시피) group{n}_final_quantile_power_curve.pkl
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

RECIPE_CHOICE = {1: "physics", 2: "full", 3: "full"}
PARAMS_SOURCE = {1: "yearly", 2: "yearly", 3: "optuna"}
QUANTILE_ALPHA = {1: 0.60, 2: 0.65, 3: 0.65}


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
    alpha = QUANTILE_ALPHA[group_id]

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
        save_power_curve_models(curve_models, OUT_DIR / f"group{group_id}_final_quantile_power_curve.pkl")

    feature_cols = get_feature_cols(feat_df)

    base_params = load_params(group_id, recipe)
    params = dict(base_params)
    params["objective"] = "quantile"
    params["alpha"] = alpha

    bagging_freq = 1 if params.get("bagging_fraction", 1.0) < 1.0 else 0
    model = lgb.LGBMRegressor(**params, random_state=42, bagging_freq=bagging_freq, verbosity=-1)
    model.fit(feat_df[feature_cols], feat_df["y"])

    model_path = OUT_DIR / f"group{group_id}_final_quantile_model.pkl"
    joblib.dump(model, model_path)

    meta = {
        "group_id": group_id,
        "recipe": recipe,
        "model_type": "lightgbm",
        "params_source": PARAMS_SOURCE[group_id],
        "feature_cols": feature_cols,
        "params": base_params,
        "objective_type": "quantile",
        "quantile_alpha": alpha,
        "n_train_rows": len(feat_df),
        "n_curtailment_removed": n_removed,
    }
    with open(OUT_DIR / f"group{group_id}_final_quantile_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[group{group_id}] recipe={recipe} quantile_alpha={alpha} "
          f"trained on {len(feat_df)} rows, {len(feature_cols)} features -> {model_path}")
    return meta


def main():
    gids = [int(a) for a in sys.argv[1:]] or [1, 2]
    for gid in gids:
        train_group(gid)
    print(f"\nQuantile objective final models saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
