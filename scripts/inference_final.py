"""
scripts/train_final.py로 학습한 최종 모델(커틀먼트 제거 + 그룹별 최적 레시피/
파라미터)로 test 기간을 예측해 제출 파일을 만든다.

실행: (레포 루트에서, scripts/train_final.py를 먼저 실행한 뒤) python3 scripts/inference_final.py
출력: submissions/lgbm_final_submission.csv
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
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
OUT_PATH = SUBMISSION_DIR / "lgbm_final_submission.csv"


def build_physics_features(df):
    df = add_default_wind_features(df)
    df = add_physics_features(df)
    df = add_time_features(df)
    speed_cols = [c for c in df.columns if c.endswith("_speed")]
    df = add_lag_rolling_features(df, cols=speed_cols, lags=[1, 2, 3], windows=[3, 6, 24])
    return df


def predict_group(group_id: int) -> pd.DataFrame:
    meta_path = MODEL_DIR / f"group{group_id}_final_meta.json"
    model_path = MODEL_DIR / f"group{group_id}_final_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} 가 없습니다. 먼저 scripts/train_final.py를 실행하세요.")
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

    feature_cols = model.feature_name_
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"group{group_id}: test 데이터에 없는 학습 feature: {missing}")

    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]
    pred = model.predict(df[feature_cols]).clip(min=0, max=capacity)
    return pd.DataFrame({"forecast_kst_dtm": df["forecast_kst_dtm"], f"kpx_group_{group_id}": pred})


def main():
    submission = load_sample_submission()[["forecast_id", "forecast_kst_dtm"]]

    for gid in [1, 2, 3]:
        pred_df = predict_group(gid)
        before = len(submission)
        submission = submission.merge(pred_df, on="forecast_kst_dtm", how="left")
        assert len(submission) == before, f"group{gid} merge 후 행 수가 바뀜"
        n_missing = submission[f"kpx_group_{gid}"].isna().sum()
        if n_missing:
            print(f"[경고] group{gid}: {n_missing}개 시각에 예측값 없음 (0으로 채움)")
            submission[f"kpx_group_{gid}"] = submission[f"kpx_group_{gid}"].fillna(0)
        print(f"[group{gid}] 예측 완료: min={pred_df.iloc[:,1].min():.1f}, max={pred_df.iloc[:,1].max():.1f}")

    submission = submission[["forecast_id", "forecast_kst_dtm", "kpx_group_1", "kpx_group_2", "kpx_group_3"]]
    submission.to_csv(OUT_PATH, index=False)
    print(f"\n저장 완료: {OUT_PATH} (shape={submission.shape})")


if __name__ == "__main__":
    main()
