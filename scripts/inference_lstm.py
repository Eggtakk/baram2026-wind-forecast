"""
LSTM 추론 스크립트 (torch 필요). scripts/train_lstm.py로 저장한 체크포인트를
불러와 test 기간을 예측하고 sample_submission 포맷 제출 파일을 만든다.

실행: (레포 루트에서, scripts/train_lstm.py를 먼저 실행한 뒤)
    python3 scripts/inference_lstm.py
출력: submissions/lstm_submission.csv
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

try:
    import torch
except ImportError as e:
    raise SystemExit("torch가 설치되어 있지 않습니다. `pip install torch` 후 다시 시도하세요.") from e

from src.data_loader import load_sample_submission
from src.lstm_model import WindLSTM
from src.sequence_data import build_test_sequences

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "experiments" / "lstm"
SUBMISSION_DIR = ROOT / "submissions"
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = SUBMISSION_DIR / "lstm_submission.csv"


def predict_group(group_id: int) -> pd.DataFrame:
    meta_path = MODEL_DIR / f"group{group_id}_lstm_meta.json"
    model_path = MODEL_DIR / f"group{group_id}_lstm.pt"
    scaler_path = MODEL_DIR / f"group{group_id}_scaler.npz"
    for p in (meta_path, model_path, scaler_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} 가 없습니다. 먼저 scripts/train_lstm.py --group {group_id} 를 실행하세요.")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    scaler = np.load(scaler_path)
    mean, std = scaler["mean"], scaler["std"]

    data = build_test_sequences(group_id, seq_len=meta["seq_len"])
    if data["feature_cols"] != meta["feature_cols"]:
        raise ValueError(f"group{group_id}: 학습 때와 feature 목록이 다릅니다. 코드가 바뀌었다면 재학습이 필요합니다.")

    X = (data["X"] - mean) / std

    model = WindLSTM(n_features=X.shape[-1], hidden_size=meta["hidden_size"], dropout=meta.get("dropout", 0.2))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        pred_norm = model(torch.from_numpy(X.astype(np.float32))).numpy()
    pred = np.clip(pred_norm * meta["capacity_kwh"], a_min=0, a_max=None)  # 학습 때와 동일하게 kWh로 복원

    return pd.DataFrame({"forecast_kst_dtm": data["timestamps"], f"kpx_group_{group_id}": pred})


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
