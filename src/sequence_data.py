"""
LSTM 입력용 시퀀스(sliding window) 데이터 생성.

LightGBM(src/preprocess.py, src/features.py)과 동일한 feature 파이프라인을
그대로 재사용해서 "같은 feature, 다른 모델 구조(tree vs recurrent)"로 공정하게
비교할 수 있게 만들었다. torch에 의존하지 않는 순수 pandas/numpy 코드라서
(torch가 설치되지 않은 환경에서도) 독립적으로 테스트할 수 있다.

핵심 이슈: test 기간(ldaps_test/gfs_test)은 2025-01-01 01:00부터 시작하는데,
seq_len=24 윈도우를 쓰려면 2025-01-01 01:00 예측에 2024-12-31의 데이터가
필요하다. test 파일에는 그 이전 시각이 없으므로, train 구간의 마지막
LOOKBACK_BUFFER 시간을 test 앞에 이어붙인 뒤 lag/rolling feature를 다시
계산하고(경계에서 값이 끊기지 않도록), 그 다음 test 구간에 해당하는 행만
예측 대상으로 잘라낸다.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import CONFIG_DIR, DATA_DIR
from src.features import build_sequence_features, get_feature_cols
from src.preprocess import build_group_dataset

LOOKBACK_BUFFER = 60  # train 꼬리에서 test 앞에 붙일 시간 수 (seq_len + rolling window 여유분)


def build_train_sequences(
    group_id: int,
    seq_len: int = 24,
    config_dir: Path = CONFIG_DIR,
    data_dir: Path = DATA_DIR,
) -> dict:
    """학습 구간 전체에 대해 (N, seq_len, F) 시퀀스와 (N,) 타깃을 만든다.

    y가 결측인 시각은 타깃으로 쓸 수 없으므로 제외한다 (window 내 과거 시점에
    y가 없어도 feature만 있으면 되므로 입력으로는 문제 없음 — y 컬럼은애초에
    feature에 포함되지 않는다).
    """
    df = build_group_dataset(group_id, split="train", config_dir=config_dir, data_dir=data_dir)
    df = build_sequence_features(df)
    df = df.sort_values("forecast_kst_dtm").reset_index(drop=True)

    feature_cols = get_feature_cols(df)
    feat = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["y"].to_numpy(dtype=np.float32)
    timestamps = df["forecast_kst_dtm"].to_numpy()

    valid_target_idx = np.where(~np.isnan(y))[0]
    valid_target_idx = valid_target_idx[valid_target_idx >= seq_len - 1]

    X, Y, T = [], [], []
    for i in valid_target_idx:
        window = feat[i - seq_len + 1 : i + 1]
        if np.isnan(window).any():
            continue
        X.append(window)
        Y.append(y[i])
        T.append(timestamps[i])

    return {
        "X": np.stack(X).astype(np.float32),
        "y": np.array(Y, dtype=np.float32),
        "timestamps": np.array(T),
        "feature_cols": feature_cols,
    }


def build_test_sequences(
    group_id: int,
    seq_len: int = 24,
    config_dir: Path = CONFIG_DIR,
    data_dir: Path = DATA_DIR,
) -> dict:
    """평가(test) 기간 전체(8,760시간)에 대한 (N, seq_len, F) 시퀀스를 만든다.

    test 파일에 없는 앞부분 이력은 train 구간 꼬리에서 가져와 이어붙인 뒤
    feature를 다시 계산하고, test 기간에 해당하는 행만 잘라 window를 만든다.
    """
    train_raw = build_group_dataset(group_id, split="train", config_dir=config_dir, data_dir=data_dir)
    test_raw = build_group_dataset(group_id, split="test", config_dir=config_dir, data_dir=data_dir)

    weather_cols = [c for c in train_raw.columns if c not in ("y",) and not c.startswith("scada_")]
    train_tail = train_raw[weather_cols].sort_values("forecast_kst_dtm").tail(LOOKBACK_BUFFER)
    test_full = test_raw[weather_cols].sort_values("forecast_kst_dtm")

    combined = pd.concat([train_tail, test_full], ignore_index=True).sort_values("forecast_kst_dtm").reset_index(drop=True)
    combined = build_sequence_features(combined)

    feature_cols = get_feature_cols(combined.assign(y=np.nan))  # y 없어도 동일 feature 목록 생성
    feat = combined[feature_cols].to_numpy(dtype=np.float32)
    timestamps = combined["forecast_kst_dtm"].to_numpy()

    test_start = test_full["forecast_kst_dtm"].min()
    target_positions = np.where(timestamps >= np.datetime64(test_start))[0]
    target_positions = target_positions[target_positions >= seq_len - 1]

    X, T = [], []
    for i in target_positions:
        window = feat[i - seq_len + 1 : i + 1]
        X.append(window)
        T.append(timestamps[i])

    X = np.stack(X).astype(np.float32)
    T = np.array(T)

    n_missing = len(test_full) - len(T)
    if n_missing != 0:
        raise RuntimeError(
            f"group{group_id}: test 시퀀스 {len(T)}개가 test 행 수 {len(test_full)}개와 다릅니다 "
            f"(LOOKBACK_BUFFER를 seq_len보다 크게 늘려야 할 수 있음)."
        )

    return {"X": X, "timestamps": T, "feature_cols": feature_cols}
