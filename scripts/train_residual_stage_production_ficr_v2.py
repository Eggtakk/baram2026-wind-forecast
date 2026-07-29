"""
group2 전용 — T/lambda_l2 재튜닝된 FICR objective stage-1
(`train_final_ficr_v2.py`, group2_final_ficr_v2_model.pkl) 위에 정격출력
2단계 잔차보정을 새로 학습한다. threshold/stage2_params는
`group2_residual_stage_ficr_v2_best_config.json`(34번 섹션 Optuna
재탐색 결과, threshold=0.65)을 사용 — 기존 threshold=0.60과 다름.

실행: (레포 루트에서, scripts/train_final_ficr_v2.py가 먼저 실행되어 있어야 함)
  python3 scripts/train_residual_stage_production_ficr_v2.py
출력: experiments/baseline_lgbm/group2_residual_stage_ficr_v2_model.pkl,
      experiments/baseline_lgbm/group2_residual_stage_ficr_v2_meta.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import lightgbm as lgb
import numpy as np

from src.data_cleaning import remove_curtailment
from src.features import build_baseline_features
from src.ficr_objective import make_ficr_objective
from src.metrics import CAPACITY_KWH
from src.power_curve import apply_power_curve_models, load_power_curve_models
from src.preprocess import build_group_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "baseline_lgbm"

GROUP_ID = 2
OOF_SPLITS = 3
MIN_REGIME_SAMPLES = 50


def _fit_lgbm(X, y, params, seed=42):
    bagging_freq = 1 if params.get("bagging_fraction", 1.0) < 1.0 else 0
    model = lgb.LGBMRegressor(**params, random_state=seed, bagging_freq=bagging_freq, verbosity=-1)
    model.fit(X, y)
    return model


def get_oof_stage1_preds(train_feat, feature_cols, params, n_splits=OOF_SPLITS):
    n = len(train_feat)
    idx = np.arange(n)
    blocks = np.array_split(idx, n_splits)
    oof = np.zeros(n)
    for i in range(n_splits):
        test_idx = blocks[i]
        train_idx = np.concatenate([blocks[j] for j in range(n_splits) if j != i])
        model = _fit_lgbm(train_feat.iloc[train_idx][feature_cols], train_feat.iloc[train_idx]["y"], params)
        oof[test_idx] = model.predict(train_feat.iloc[test_idx][feature_cols]).clip(min=0)
    return oof


def main():
    meta_path = OUT_DIR / f"group{GROUP_ID}_final_ficr_v2_meta.json"
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} 가 없습니다. 먼저 scripts/train_final_ficr_v2.py를 실행하세요.")
    with open(meta_path, "r", encoding="utf-8") as f:
        stage1_meta = json.load(f)
    recipe = stage1_meta["recipe"]
    feature_cols = stage1_meta["feature_cols"]
    capacity = CAPACITY_KWH[f"kpx_group_{GROUP_ID}"]

    stage1_params = dict(stage1_meta["params"])
    stage1_params["objective"] = make_ficr_objective(
        capacity, ficr_weight=stage1_meta["ficr_weight"], lambda_l2=stage1_meta["lambda_l2"], T=stage1_meta["T"],
    )

    config_path = OUT_DIR / f"group{GROUP_ID}_residual_stage_ficr_v2_best_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    threshold = config["threshold"]
    stage2_params = config["stage2_params"]
    print(f"[group{GROUP_ID}] recipe={recipe} ficr_weight={stage1_meta['ficr_weight']} "
          f"T={stage1_meta['T']} lambda_l2={stage1_meta['lambda_l2']} "
          f"threshold={threshold} stage2_params={stage2_params}")

    df = build_group_dataset(GROUP_ID, split="train").dropna(subset=["y"]).reset_index(drop=True)
    cleaned_df, n_removed = remove_curtailment(df, capacity=capacity)

    train_feat = build_baseline_features(cleaned_df)
    curve_path = OUT_DIR / f"group{GROUP_ID}_final_ficr_v2_power_curve.pkl"
    curve_models = load_power_curve_models(curve_path)
    train_feat = apply_power_curve_models(train_feat, curve_models)

    missing = [c for c in feature_cols if c not in train_feat.columns]
    if missing:
        raise ValueError(f"train_feat에 없는 stage-1 feature: {missing}")

    print(f"train_feat: {len(train_feat)}행, {len(feature_cols)} features (커틀먼트 제거 {n_removed}행)")

    oof_pred = get_oof_stage1_preds(train_feat, feature_cols, stage1_params)

    y_train = train_feat["y"].to_numpy()
    train_frac = y_train / capacity
    regime_mask = train_frac >= threshold
    n_regime = int(regime_mask.sum())
    print(f"regime(threshold={threshold}) 학습 행 수: {n_regime} / {len(train_feat)} "
          f"({n_regime/len(train_feat)*100:.1f}%)")
    if n_regime < MIN_REGIME_SAMPLES:
        raise SystemExit(f"regime 샘플이 너무 적음({n_regime} < {MIN_REGIME_SAMPLES})")

    residual = y_train - oof_pred
    X_stage2 = train_feat.loc[regime_mask, feature_cols].copy()
    X_stage2["stage1_pred"] = oof_pred[regime_mask]
    y_stage2 = residual[regime_mask]

    stage2_model = _fit_lgbm(X_stage2, y_stage2, stage2_params)

    model_path = OUT_DIR / f"group{GROUP_ID}_residual_stage_ficr_v2_model.pkl"
    joblib.dump(stage2_model, model_path)

    meta = {
        "group_id": GROUP_ID,
        "recipe": recipe,
        "threshold": threshold,
        "stage2_params": stage2_params,
        "stage1_feature_cols": feature_cols,
        "stage2_feature_cols": feature_cols + ["stage1_pred"],
        "n_regime_train": n_regime,
        "n_train_rows": len(train_feat),
        "oof_splits": OOF_SPLITS,
        "stage1_objective_type": "ficr_shaped_v2",
    }
    meta_out_path = OUT_DIR / f"group{GROUP_ID}_residual_stage_ficr_v2_meta.json"
    with open(meta_out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료: {model_path}")
    print(f"저장 완료: {meta_out_path}")


if __name__ == "__main__":
    main()
