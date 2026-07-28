"""
group2 전용 — 정격출력 구간 2단계 잔차(residual) 모델을 실제 test 추론용으로
학습(프로덕션 반영).

배경: rated_output_investigation.md 19~21번 섹션. LOYO 검증 결과 group2가
이 세션에서 유일하게 재현된(2-fold/3-fold OOF 양쪽 모두, 3개 폴드 전부
양수, candidate std가 baseline std보다 항상 작음) 신뢰할 만한 개선이었다
(threshold=0.65, delta+0.0218 — baseline std 0.0243의 90%까지 근접,
그러나 엄격 기준은 아직 미충족). group1/3은 여전히 노이즈 수준이거나
과최적화 의심(group3)이라 이번엔 group2만 프로덕션에 반영해 실제
리더보드로 검증한다.

방법(LOYO 실험과 동일한 파이프라인, 다만 outer holdout이 실제 2025 test라는
점만 다름):
  1. stage-1: 이미 학습된 프로덕션 group2 모델(scripts/train_final.py 결과물,
     group2_final_model.pkl — full 레시피 LightGBM, yearly 파라미터)을 그대로
     재사용한다(재학습 없음 — 현재 제출본 0.61034와 완전히 동일한 stage-1).
  2. stage-1 학습에 쓰인 것과 동일한 cleaned/feature 데이터(전체 train)에서
     3-fold 시간순 out-of-fold(OOF) 예측을 만든다(각 블록을 나머지 2/3로
     학습한 모델로 예측 — 잔차 추정용, group2_final_model.pkl 자체와는 다른
     보조 모델 3개를 이 목적에만 씀).
  3. 실제 y/capacity >= 0.65(그리드서치로 찾은 최적 threshold, 21번 섹션)인
     행만 골라 residual = y - oof_pred를 타깃으로, stage-2 LightGBM을
     `experiments/baseline_lgbm/group2_residual_stage_best_config_oof3.json`
     의 튜닝된 파라미터로 학습(입력 feature = stage-1과 동일 + stage-1 OOF
     예측값).

실행: (레포 루트에서, scripts/train_final.py가 이미 실행되어
group2_final_model.pkl/group2_final_power_curve.pkl이 있어야 함)
  python3 scripts/train_group2_residual_stage.py
출력: experiments/baseline_lgbm/group2_residual_stage_model.pkl,
      experiments/baseline_lgbm/group2_residual_stage_meta.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import lightgbm as lgb
import numpy as np

from src.data_cleaning import remove_curtailment
from src.features import build_baseline_features, get_feature_cols
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
    meta_path = OUT_DIR / f"group{GROUP_ID}_final_meta.json"
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} 가 없습니다. 먼저 scripts/train_final.py를 실행하세요.")
    with open(meta_path, "r", encoding="utf-8") as f:
        stage1_meta = json.load(f)
    assert stage1_meta["recipe"] == "full" and stage1_meta["model_type"] == "lightgbm"
    stage1_params = stage1_meta["params"]
    feature_cols = stage1_meta["feature_cols"]

    config_path = OUT_DIR / "group2_residual_stage_best_config_oof3.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    threshold = config["threshold"]
    stage2_params = config["stage2_params"]
    print(f"threshold={threshold}, stage2_params={stage2_params}")

    # --- stage-1과 완전히 동일한 train feature 재구성 ---
    capacity = CAPACITY_KWH[f"kpx_group_{GROUP_ID}"]
    df = build_group_dataset(GROUP_ID, split="train").dropna(subset=["y"]).reset_index(drop=True)
    cleaned_df, n_removed = remove_curtailment(df, capacity=capacity)
    train_feat = build_baseline_features(cleaned_df)
    curve_path = OUT_DIR / f"group{GROUP_ID}_final_power_curve.pkl"
    curve_models = load_power_curve_models(curve_path)
    train_feat = apply_power_curve_models(train_feat, curve_models)

    missing = [c for c in feature_cols if c not in train_feat.columns]
    if missing:
        raise ValueError(f"train_feat에 없는 stage-1 feature: {missing}")

    print(f"train_feat: {len(train_feat)}행, {len(feature_cols)} features "
          f"(커틀먼트 제거 {n_removed}행)")

    # --- 3-fold OOF stage-1 예측 (잔차 추정용, group2_final_model.pkl과는 별개) ---
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

    model_path = OUT_DIR / f"group{GROUP_ID}_residual_stage_model.pkl"
    joblib.dump(stage2_model, model_path)

    meta = {
        "group_id": GROUP_ID,
        "threshold": threshold,
        "stage2_params": stage2_params,
        "stage1_feature_cols": feature_cols,
        "stage2_feature_cols": feature_cols + ["stage1_pred"],
        "n_regime_train": n_regime,
        "n_train_rows": len(train_feat),
        "oof_splits": OOF_SPLITS,
    }
    meta_out_path = OUT_DIR / f"group{GROUP_ID}_residual_stage_meta.json"
    with open(meta_out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료: {model_path}")
    print(f"저장 완료: {meta_out_path}")


if __name__ == "__main__":
    main()
