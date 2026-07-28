"""
group1/group2 전용 — FICR objective로 재학습한 stage-1(`train_final_ficr.py`,
group{gid}_final_ficr_model.pkl) 위에 정격출력 구간 2단계 잔차보정을 다시
학습한다. `scripts/train_residual_stage_production.py`와 동일한 구조이되:
  - stage-1을 `_final_meta.json`이 아니라 `_final_ficr_meta.json`에서 읽고,
  - OOF stage-1 예측을 만들 때도 동일한 FICR objective로 fit한다(잔차 타깃이
    "실제로 배포될 stage-1"과 같은 손실함수로 만들어져야 일관됨),
  - threshold/stage2_params는 재탐색하지 않고 기존
    `group{gid}_residual_stage_best_config_oof3.json`을 그대로 재사용한다
    (25번 섹션 tune_residual_stage_ficr.py에서 LOYO로 이미 "여전히
    도움된다"를 확인).

실행: (레포 루트에서, scripts/train_final_ficr.py가 이미 실행되어 있어야 함)
  python3 scripts/train_residual_stage_production_ficr.py <group_id>
출력: experiments/baseline_lgbm/group{gid}_residual_stage_ficr_model.pkl,
      experiments/baseline_lgbm/group{gid}_residual_stage_ficr_meta.json
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
from src.ficr_objective import make_ficr_objective
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
    meta_path = OUT_DIR / f"group{group_id}_final_ficr_meta.json"
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} 가 없습니다. 먼저 scripts/train_final_ficr.py를 실행하세요.")
    with open(meta_path, "r", encoding="utf-8") as f:
        stage1_meta = json.load(f)
    assert stage1_meta["model_type"] == "lightgbm"
    recipe = stage1_meta["recipe"]
    feature_cols = stage1_meta["feature_cols"]
    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]

    stage1_params = dict(stage1_meta["params"])
    stage1_params["objective"] = make_ficr_objective(
        capacity,
        ficr_weight=stage1_meta["ficr_weight"],
        lambda_l2=stage1_meta["lambda_l2"],
        T=stage1_meta["T"],
    )

    # FICR objective 전용으로 재탐색된 설정이 있으면 그걸 우선 사용(27번 섹션,
    # 예: group2는 stage-1이 FICR objective로 바뀌면서 threshold 최적점도
    # 0.65->0.60으로 이동). 없으면 기존 L2 stage-1 기준 oof3 설정을 재사용.
    ficr_config_path = OUT_DIR / f"group{group_id}_residual_stage_ficr_best_config.json"
    config_path = ficr_config_path if ficr_config_path.exists() else OUT_DIR / f"group{group_id}_residual_stage_best_config_oof3.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    print(f"[group{group_id}] 잔차보정 설정 소스: {config_path.name}")
    threshold = config["threshold"]
    stage2_params = config["stage2_params"]
    print(f"[group{group_id}] recipe={recipe} ficr_weight={stage1_meta['ficr_weight']} "
          f"threshold={threshold} stage2_params={stage2_params}")

    # --- stage-1과 완전히 동일한 train feature 재구성 ---
    df = build_group_dataset(group_id, split="train").dropna(subset=["y"]).reset_index(drop=True)
    cleaned_df, n_removed = remove_curtailment(df, capacity=capacity)

    if recipe == "physics":
        train_feat = build_physics_features(cleaned_df)
    else:
        train_feat = build_baseline_features(cleaned_df)
        curve_path = OUT_DIR / f"group{group_id}_final_ficr_power_curve.pkl"
        curve_models = load_power_curve_models(curve_path)
        train_feat = apply_power_curve_models(train_feat, curve_models)

    missing = [c for c in feature_cols if c not in train_feat.columns]
    if missing:
        raise ValueError(f"train_feat에 없는 stage-1 feature: {missing}")

    print(f"train_feat: {len(train_feat)}행, {len(feature_cols)} features "
          f"(커틀먼트 제거 {n_removed}행)")

    # --- 3-fold OOF stage-1 예측 (FICR objective로, 잔차 추정용) ---
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

    model_path = OUT_DIR / f"group{group_id}_residual_stage_ficr_model.pkl"
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
        "stage1_objective_type": "ficr_shaped",
    }
    meta_out_path = OUT_DIR / f"group{group_id}_residual_stage_ficr_meta.json"
    with open(meta_out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료: {model_path}")
    print(f"저장 완료: {meta_out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("사용법: python3 scripts/train_residual_stage_production_ficr.py <group_id>")
    main(int(sys.argv[1]))
