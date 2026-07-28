"""
group1 또는 group2 전용 — 정격출력 구간 2단계 잔차(residual) 모델을 실제
test 추론용으로 학습(프로덕션 반영). scripts/train_group2_residual_stage.py를
group1까지 다루도록 일반화한 버전(group1은 physics 레시피라 파워커브가 없음).

배경: rated_output_investigation.md 19~22번 섹션. group2는 이미 이 방식으로
실제 제출해 0.61034->0.61446으로 개선 확인(22번 섹션). group1도 같은
패턴(OOF 3-fold, threshold=0.80, stage-2 Optuna 튜닝)으로 LOYO delta
+0.0073을 얻어(21번 섹션) 이번엔 group1도 실제 제출로 검증한다.

실행: (레포 루트에서, scripts/train_final.py가 이미 실행되어 있어야 함)
  python3 scripts/train_residual_stage_production.py <group_id>
출력: experiments/baseline_lgbm/group{gid}_residual_stage_model.pkl,
      experiments/baseline_lgbm/group{gid}_residual_stage_meta.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import lightgbm as lgb
import numpy as np

from src.data_cleaning import remove_curtailment
from src.features import (
    add_default_wind_features,
    add_lag_rolling_features,
    add_physics_features,
    add_time_features,
    build_baseline_features,
)
from src.metrics import CAPACITY_KWH
from src.power_curve import apply_power_curve_models, load_power_curve_models
from src.preprocess import build_group_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "baseline_lgbm"

OOF_SPLITS = 3
MIN_REGIME_SAMPLES = 50


def build_physics_features(df):
    df = add_default_wind_features(df)
    df = add_physics_features(df)
    df = add_time_features(df)
    speed_cols = [c for c in df.columns if c.endswith("_speed")]
    df = add_lag_rolling_features(df, cols=speed_cols, lags=[1, 2, 3], windows=[3, 6, 24])
    return df


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


def main(group_id: int):
    meta_path = OUT_DIR / f"group{group_id}_final_meta.json"
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} 가 없습니다. 먼저 scripts/train_final.py를 실행하세요.")
    with open(meta_path, "r", encoding="utf-8") as f:
        stage1_meta = json.load(f)
    assert stage1_meta["model_type"] == "lightgbm"
    recipe = stage1_meta["recipe"]
    stage1_params = stage1_meta["params"]
    feature_cols = stage1_meta["feature_cols"]

    config_path = OUT_DIR / f"group{group_id}_residual_stage_best_config_oof3.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    threshold = config["threshold"]
    stage2_params = config["stage2_params"]
    print(f"[group{group_id}] recipe={recipe} threshold={threshold} stage2_params={stage2_params}")

    # --- stage-1과 완전히 동일한 train feature 재구성 ---
    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]
    df = build_group_dataset(group_id, split="train").dropna(subset=["y"]).reset_index(drop=True)
    cleaned_df, n_removed = remove_curtailment(df, capacity=capacity)

    if recipe == "physics":
        train_feat = build_physics_features(cleaned_df)
    else:
        train_feat = build_baseline_features(cleaned_df)
        curve_path = OUT_DIR / f"group{group_id}_final_power_curve.pkl"
        curve_models = load_power_curve_models(curve_path)
        train_feat = apply_power_curve_models(train_feat, curve_models)

    missing = [c for c in feature_cols if c not in train_feat.columns]
    if missing:
        raise ValueError(f"train_feat에 없는 stage-1 feature: {missing}")

    print(f"train_feat: {len(train_feat)}행, {len(feature_cols)} features "
          f"(커틀먼트 제거 {n_removed}행)")

    # --- 3-fold OOF stage-1 예측 (잔차 추정용) ---
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

    model_path = OUT_DIR / f"group{group_id}_residual_stage_model.pkl"
    joblib.dump(stage2_model, model_path)

    meta = {
        "group_id": group_id,
        "recipe": recipe,
        "threshold": threshold,
        "stage2_params": stage2_params,
        "stage1_feature_cols": feature_cols,
        "stage2_feature_cols": feature_cols + ["stage1_pred"],
        "n_regime_train": n_regime,
        "n_train_rows": len(train_feat),
        "oof_splits": OOF_SPLITS,
    }
    meta_out_path = OUT_DIR / f"group{group_id}_residual_stage_meta.json"
    with open(meta_out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료: {model_path}")
    print(f"저장 완료: {meta_out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("사용법: python3 scripts/train_residual_stage_production.py <group_id>")
    main(int(sys.argv[1]))
