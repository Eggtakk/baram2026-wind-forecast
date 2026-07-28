"""
group1+group2에 정격출력 2단계 잔차 보정을 적용한 제출 파일 생성
(group3는 기존 프로덕션 그대로). scripts/inference_group2_residual.py를
group1까지 다루도록 일반화.

배경: group2 단독 적용 실제 제출 결과 0.61034->0.61446 확인(22번 섹션).
group1도 같은 방식(threshold=0.80, OOF 3-fold, stage-2 Optuna)으로 LOYO
delta +0.0073을 얻어 이번엔 group1까지 함께 적용해 실제 제출로 검증한다.

실행: (레포 루트에서, train_final.py + train_residual_stage_production.py
1/2가 이미 실행되어 있어야 함)
  python3 scripts/inference_residual_combined.py
출력: submissions/lgbm_group12_residual_submission.csv
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from src.data_loader import load_sample_submission
from src.metrics import CAPACITY_KWH
from src.features import (
    add_default_wind_features,
    add_lag_rolling_features,
    add_physics_features,
    add_time_features,
    build_baseline_features,
)
from src.power_curve import apply_power_curve_models, load_power_curve_models
from src.preprocess import build_group_dataset

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "experiments" / "baseline_lgbm"
SUBMISSION_DIR = ROOT / "submissions"
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = SUBMISSION_DIR / "lgbm_group12_residual_submission.csv"

RESIDUAL_GROUPS = {1, 2}  # group3는 기존 프로덕션 그대로


def build_physics_features(df):
    df = add_default_wind_features(df)
    df = add_physics_features(df)
    df = add_time_features(df)
    speed_cols = [c for c in df.columns if c.endswith("_speed")]
    df = add_lag_rolling_features(df, cols=speed_cols, lags=[1, 2, 3], windows=[3, 6, 24])
    return df


def predict_group_baseline(group_id: int) -> pd.DataFrame:
    meta_path = MODEL_DIR / f"group{group_id}_final_meta.json"
    model_path = MODEL_DIR / f"group{group_id}_final_model.pkl"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    model = joblib.load(model_path)
    recipe = meta["recipe"]

    df = build_group_dataset(group_id, split="test")
    if recipe == "physics":
        df = build_physics_features(df)
    else:
        df = build_baseline_features(df)
        curve_path = MODEL_DIR / f"group{group_id}_final_power_curve.pkl"
        if curve_path.exists():
            curve_models = load_power_curve_models(curve_path)
            df = apply_power_curve_models(df, curve_models)

    feature_cols = meta["feature_cols"]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"group{group_id}: test 데이터에 없는 학습 feature: {missing}")

    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]
    pred = model.predict(df[feature_cols]).clip(min=0, max=capacity)
    return pd.DataFrame({"forecast_kst_dtm": df["forecast_kst_dtm"], f"kpx_group_{group_id}": pred})


def predict_group_with_residual(group_id: int) -> pd.DataFrame:
    meta_path = MODEL_DIR / f"group{group_id}_final_meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        stage1_meta = json.load(f)
    stage1_model = joblib.load(MODEL_DIR / f"group{group_id}_final_model.pkl")
    feature_cols = stage1_meta["feature_cols"]
    recipe = stage1_meta["recipe"]

    residual_meta_path = MODEL_DIR / f"group{group_id}_residual_stage_meta.json"
    with open(residual_meta_path, "r", encoding="utf-8") as f:
        residual_meta = json.load(f)
    stage2_model = joblib.load(MODEL_DIR / f"group{group_id}_residual_stage_model.pkl")
    threshold = residual_meta["threshold"]

    df = build_group_dataset(group_id, split="test")
    if recipe == "physics":
        df = build_physics_features(df)
    else:
        df = build_baseline_features(df)
        curve_models = load_power_curve_models(MODEL_DIR / f"group{group_id}_final_power_curve.pkl")
        df = apply_power_curve_models(df, curve_models)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"group{group_id}: test 데이터에 없는 stage-1 feature: {missing}")

    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]
    stage1_pred = stage1_model.predict(df[feature_cols]).clip(min=0, max=capacity)

    regime_mask = (stage1_pred / capacity) >= threshold
    n_regime = int(regime_mask.sum())
    print(f"[group{group_id}] stage-1 예측 기준 regime(threshold={threshold}) 행 수: "
          f"{n_regime} / {len(df)} ({n_regime/len(df)*100:.1f}%)")

    final_pred = stage1_pred.copy()
    if n_regime > 0:
        X_stage2 = df.loc[regime_mask, feature_cols].copy()
        X_stage2["stage1_pred"] = stage1_pred[regime_mask]
        correction = stage2_model.predict(X_stage2)
        final_pred[regime_mask] = stage1_pred[regime_mask] + correction
        print(f"[group{group_id}] 보정 전/후 평균 발전량: "
              f"{stage1_pred.mean():.1f} -> {final_pred.mean():.1f} kWh "
              f"(regime 구간 평균 보정폭: {correction.mean():.1f} kWh)")
    final_pred = np.clip(final_pred, 0, capacity)

    return pd.DataFrame({"forecast_kst_dtm": df["forecast_kst_dtm"], f"kpx_group_{group_id}": final_pred})


def main():
    submission = load_sample_submission()[["forecast_id", "forecast_kst_dtm"]]

    for gid in [1, 2, 3]:
        pred_df = predict_group_with_residual(gid) if gid in RESIDUAL_GROUPS else predict_group_baseline(gid)
        before = len(submission)
        submission = submission.merge(pred_df, on="forecast_kst_dtm", how="left")
        assert len(submission) == before, f"group{gid} merge 후 행 수가 바뀜"
        n_missing = submission[f"kpx_group_{gid}"].isna().sum()
        if n_missing:
            print(f"[경고] group{gid}: {n_missing}개 시각에 예측값 없음 (0으로 채움)")
            submission[f"kpx_group_{gid}"] = submission[f"kpx_group_{gid}"].fillna(0)
        print(f"[group{gid}] 예측 완료: min={pred_df.iloc[:, 1].min():.1f}, max={pred_df.iloc[:, 1].max():.1f}")

    submission = submission[["forecast_id", "forecast_kst_dtm", "kpx_group_1", "kpx_group_2", "kpx_group_3"]]
    submission.to_csv(OUT_PATH, index=False)
    print(f"\n저장 완료: {OUT_PATH} (shape={submission.shape})")

    # --- 포맷 검증 ---
    sample = load_sample_submission()
    assert list(submission["forecast_id"]) == list(sample["forecast_id"]), "forecast_id 순서/값 불일치"
    assert list(submission["forecast_kst_dtm"]) == list(sample["forecast_kst_dtm"]), "forecast_kst_dtm 순서/값 불일치"
    assert submission.shape[0] == sample.shape[0], f"행 수 불일치: {submission.shape[0]} != {sample.shape[0]}"
    for gid in [1, 2, 3]:
        col = f"kpx_group_{gid}"
        assert submission[col].isna().sum() == 0, f"{col}에 NaN 존재"
        assert (submission[col] < 0).sum() == 0, f"{col}에 음수 존재"
        capacity = CAPACITY_KWH[col]
        assert (submission[col] > capacity + 1e-6).sum() == 0, f"{col}에 capacity 초과값 존재"
    print(f"포맷 검증 통과: {submission.shape[0]}행, NaN 0, 음수 0, capacity 초과 0")

    # 직전 제출본(group2만 보정) 대비 비교
    prev_path = SUBMISSION_DIR / "lgbm_group2_residual_submission.csv"
    if prev_path.exists():
        prev = pd.read_csv(prev_path, parse_dates=["forecast_kst_dtm"])
        for gid in [1, 2, 3]:
            col = f"kpx_group_{gid}"
            diff = (submission[col] - prev[col]).abs()
            print(f"  {col}: 직전 제출 대비 평균|delta|={diff.mean():.2f} kWh, "
                  f"delta!=0인 행 수={int((diff > 1e-6).sum())}")


if __name__ == "__main__":
    main()
