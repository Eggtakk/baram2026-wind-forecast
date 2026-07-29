"""
34번 섹션 — FICR-shaped custom objective의 T/lambda_l2를 재튜닝한 값
(T=0.025, lambda_l2=0.5, ficr_weight=0.008 그대로)으로 group2의 stage-1
프로덕션 모델을 재학습한다. group1은 quantile objective로 이미 교체됐고
group3는 대상 아님 — group2만 해당.

`scripts/train_final_ficr.py`(T=0.01/lambda_l2=1.0, 기존 배포본)와 완전히
동일한 구조이되 T/lambda_l2만 다르다. 기존 `group2_final_ficr_*`는
덮어쓰지 않고 `group2_final_ficr_v2_*`로 별도 저장(롤백 가능).

실행: (레포 루트에서) python3 scripts/train_final_ficr_v2.py
출력: experiments/baseline_lgbm/group2_final_ficr_v2_model.pkl,
      group2_final_ficr_v2_meta.json, group2_final_ficr_v2_power_curve.pkl
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import lightgbm as lgb

from src.data_cleaning import remove_curtailment
from src.features import build_baseline_features, get_feature_cols
from src.ficr_objective import make_ficr_objective
from src.metrics import CAPACITY_KWH
from src.power_curve import apply_power_curve_models, fit_power_curve_models, save_power_curve_models
from src.preprocess import build_group_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "baseline_lgbm"

GROUP_ID = 2
RECIPE = "full"
PARAMS_SOURCE = "yearly"
FICR_WEIGHT = 0.008
LAMBDA_L2 = 0.5
T = 0.025


def load_params() -> dict:
    path = OUT_DIR / f"group{GROUP_ID}_{PARAMS_SOURCE}_best_params_{RECIPE}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    capacity = CAPACITY_KWH[f"kpx_group_{GROUP_ID}"]

    df = build_group_dataset(GROUP_ID, split="train").dropna(subset=["y"]).reset_index(drop=True)
    cleaned_df, n_removed = remove_curtailment(df, capacity=capacity)
    print(f"[group{GROUP_ID}] 커틀먼트 의심 {n_removed}행 제거 ({n_removed/len(df)*100:.2f}%)")

    feat_df = build_baseline_features(cleaned_df)
    curve_models = fit_power_curve_models(feat_df, capacity=capacity)
    feat_df = apply_power_curve_models(feat_df, curve_models)
    save_power_curve_models(curve_models, OUT_DIR / f"group{GROUP_ID}_final_ficr_v2_power_curve.pkl")

    feature_cols = get_feature_cols(feat_df)

    base_params = load_params()
    params = dict(base_params)
    params["objective"] = make_ficr_objective(capacity, ficr_weight=FICR_WEIGHT, lambda_l2=LAMBDA_L2, T=T)

    bagging_freq = 1 if params.get("bagging_fraction", 1.0) < 1.0 else 0
    model = lgb.LGBMRegressor(**params, random_state=42, bagging_freq=bagging_freq, verbosity=-1)
    model.fit(feat_df[feature_cols], feat_df["y"])

    model_path = OUT_DIR / f"group{GROUP_ID}_final_ficr_v2_model.pkl"
    joblib.dump(model, model_path)

    meta = {
        "group_id": GROUP_ID,
        "recipe": RECIPE,
        "model_type": "lightgbm",
        "params_source": PARAMS_SOURCE,
        "feature_cols": feature_cols,
        "params": base_params,
        "objective_type": "ficr_shaped",
        "ficr_weight": FICR_WEIGHT,
        "lambda_l2": LAMBDA_L2,
        "T": T,
        "n_train_rows": len(feat_df),
        "n_curtailment_removed": n_removed,
    }
    with open(OUT_DIR / f"group{GROUP_ID}_final_ficr_v2_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[group{GROUP_ID}] recipe={RECIPE} ficr_weight={FICR_WEIGHT} T={T} lambda_l2={LAMBDA_L2} "
          f"trained on {len(feat_df)} rows, {len(feature_cols)} features -> {model_path}")


if __name__ == "__main__":
    main()
