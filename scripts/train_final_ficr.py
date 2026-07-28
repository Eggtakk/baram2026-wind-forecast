"""
FICR-shaped custom objective(25번 섹션, src/ficr_objective.py)로 group1/group2의
stage-1 프로덕션 모델을 재학습한다. group3는 LOYO에서 개선이 뚜렷하지 않아
(25번 섹션, 버그 수정 후에도 노이즈 수준/평평) 제외 — 기존 group3_final_model.pkl
그대로 유지.

`scripts/train_final.py`(기존 프로덕션 L2 학습)와 완전히 동일한 데이터/레시피/
하이퍼파라미터를 쓰되 objective만 교체한다. 기존 `group{gid}_final_model.pkl`은
잔차보정(22·23번 섹션)이 이미 이 위에서 학습돼 실 배포 중이므로 덮어쓰지 않고
`group{gid}_final_ficr_model.pkl`(및 meta/power_curve)로 별도 저장한다 — 롤백
가능하게.

ficr_weight는 25번 섹션 LOYO 그리드서치 최적점: group1=0.003, group2=0.008.

실행: (레포 루트에서) python3 scripts/train_final_ficr.py
출력: experiments/baseline_lgbm/group{n}_final_ficr_model.pkl,
      group{n}_final_ficr_meta.json,
      (group2, full 레시피) group{n}_final_ficr_power_curve.pkl
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
from src.ficr_objective import make_ficr_objective
from src.metrics import CAPACITY_KWH
from src.power_curve import apply_power_curve_models, fit_power_curve_models, save_power_curve_models
from src.preprocess import build_group_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "baseline_lgbm"

RECIPE_CHOICE = {1: "physics", 2: "full"}
PARAMS_SOURCE = {1: "yearly", 2: "yearly"}
FICR_WEIGHT = {1: 0.003, 2: 0.008}
LAMBDA_L2 = 1.0
T = 0.01


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
    ficr_weight = FICR_WEIGHT[group_id]

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
        save_power_curve_models(curve_models, OUT_DIR / f"group{group_id}_final_ficr_power_curve.pkl")

    feature_cols = get_feature_cols(feat_df)

    base_params = load_params(group_id, recipe)
    params = dict(base_params)
    params["objective"] = make_ficr_objective(capacity, ficr_weight=ficr_weight, lambda_l2=LAMBDA_L2, T=T)

    bagging_freq = 1 if params.get("bagging_fraction", 1.0) < 1.0 else 0
    model = lgb.LGBMRegressor(**params, random_state=42, bagging_freq=bagging_freq, verbosity=-1)
    model.fit(feat_df[feature_cols], feat_df["y"])

    model_path = OUT_DIR / f"group{group_id}_final_ficr_model.pkl"
    joblib.dump(model, model_path)

    # meta에는 objective 콜러블을 그대로 못 담으니(JSON 직렬화 불가) ficr_weight
    # 등 재구성에 필요한 값만 저장하고, params는 objective를 뺀 원본 dict를 저장.
    meta = {
        "group_id": group_id,
        "recipe": recipe,
        "model_type": "lightgbm",
        "params_source": PARAMS_SOURCE[group_id],
        "feature_cols": feature_cols,
        "params": base_params,
        "objective_type": "ficr_shaped",
        "ficr_weight": ficr_weight,
        "lambda_l2": LAMBDA_L2,
        "T": T,
        "n_train_rows": len(feat_df),
        "n_curtailment_removed": n_removed,
    }
    with open(OUT_DIR / f"group{group_id}_final_ficr_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[group{group_id}] recipe={recipe} ficr_weight={ficr_weight} "
          f"trained on {len(feat_df)} rows, {len(feature_cols)} features -> {model_path}")
    return meta


def main():
    for gid in [1, 2]:
        train_group(gid)
    print(f"\nFICR objective final models saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
