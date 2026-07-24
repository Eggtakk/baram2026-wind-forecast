"""
시계열 검증(holdout) 유틸리티.

풍력 발전량은 계절성이 강해서(experiments/group3_eda/summary.md 참고) 랜덤
split을 쓰면 미래 정보가 과거 예측에 새어 들어가 validation score가
실제보다 낙관적으로 나온다. 항상 시간순으로 정렬한 뒤 뒤쪽 구간을
holdout으로 떼어낸다.
"""
import numpy as np
import pandas as pd


def time_based_split(
    df: pd.DataFrame,
    time_col: str = "forecast_kst_dtm",
    holdout_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """시간순 정렬 후 뒤쪽 holdout_ratio 비율을 holdout으로 분리.

    df는 이미 라벨(y)이 결측이 아닌 행만 담고 있다고 가정한다
    (build_group_dataset()의 결과에서 dropna(subset=['y'])한 뒤 호출).
    """
    df = df.sort_values(time_col).reset_index(drop=True)
    cutoff = int(len(df) * (1 - holdout_ratio))
    train = df.iloc[:cutoff].copy()
    holdout = df.iloc[cutoff:].copy()
    return train, holdout


def array_time_split(
    *arrays: np.ndarray,
    holdout_ratio: float = 0.2,
) -> tuple:
    """time_based_split의 numpy 배열 버전 (LSTM 시퀀스 X/y/timestamps 등에 사용).

    배열들은 이미 시간순 정렬되어 있다고 가정한다 (src/sequence_data.py의
    build_train_sequences가 정렬된 상태로 반환함). 여러 배열을 한 번에 같은
    지점에서 잘라준다: array_time_split(X, y, ts) -> (X_tr,y_tr,ts_tr), (X_ho,y_ho,ts_ho)
    """
    n = len(arrays[0])
    cutoff = int(n * (1 - holdout_ratio))
    train_part = tuple(a[:cutoff] for a in arrays)
    holdout_part = tuple(a[cutoff:] for a in arrays)
    return train_part, holdout_part


def time_based_split_by_date(
    df: pd.DataFrame,
    split_date: str,
    time_col: str = "forecast_kst_dtm",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """특정 날짜를 기준으로 train/holdout을 나눈다 (예: 마지막 1년을 통째로 holdout).

    split_date 이전 = train, split_date 이후(포함) = holdout.
    """
    df = df.sort_values(time_col).reset_index(drop=True)
    cutoff = pd.Timestamp(split_date)
    train = df[df[time_col] < cutoff].copy()
    holdout = df[df[time_col] >= cutoff].copy()
    return train, holdout
