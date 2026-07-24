"""
베이스라인 추론 스크립트. scripts/train.py로 저장한 모델을 불러와 test 기간을
예측하고, sample_submission.csv 포맷에 맞춘 제출 파일을 만든다.

주의: 이 스크립트는 test 기간의 LDAPS/GFS 예보 데이터만 사용한다. SCADA나
라벨은 test 기간에 존재하지 않으므로 feature로 쓰지 않는다
(src/data_loader.py, src/features.NON_FEATURE_COLS 참고).

실행: (레포 루트에서, scripts/train.py를 먼저 실행한 뒤) python3 scripts/inference.py
출력: submissions/baseline_lgbm_submission.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from src.data_loader import load_sample_submission
from src.features import build_baseline_features
from src.preprocess import build_group_dataset

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "experiments" / "baseline_lgbm"
SUBMISSION_DIR = ROOT / "submissions"
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = SUBMISSION_DIR / "baseline_lgbm_submission.csv"


def predict_group(group_id: int) -> pd.DataFrame:
    model_path = MODEL_DIR / f"group{group_id}_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} 가 없습니다. 먼저 scripts/train.py를 실행하세요.")
    model = joblib.load(model_path)

    df = build_group_dataset(group_id, split="test")
    df = build_baseline_features(df)

    feature_cols = model.feature_name_
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"group{group_id}: test 데이터에 없는 학습 feature: {missing}")

    pred = model.predict(df[feature_cols]).clip(min=0)
    return pd.DataFrame({"forecast_kst_dtm": df["forecast_kst_dtm"], f"kpx_group_{group_id}": pred})


def main():
    submission = load_sample_submission()[["forecast_id", "forecast_kst_dtm"]]

    for gid in [1, 2, 3]:
        pred_df = predict_group(gid)
        before = len(submission)
        submission = submission.merge(pred_df, on="forecast_kst_dtm", how="left")
        assert len(submission) == before, f"group{gid} merge 후 행 수가 바뀜 (중복 시각 의심)"
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
