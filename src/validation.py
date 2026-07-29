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


def _year_bounds(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year + 1}-01-01")


def time_based_split_leave_year_out(
    df: pd.DataFrame,
    holdout_year: int,
    valid_years: list[int],
    time_col: str = "forecast_kst_dtm",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-year-out 분할: holdout_year 1년을 통째로 holdout으로 떼어내고,
    valid_years 중 나머지 연도를 전부 합쳐 train으로 쓴다.

    단일 연도 holdout(time_based_split_by_date)은 어느 한 해의 특이 이벤트
    (예: 2024년에 유독 많았던 착빙/커틀먼트)에 결과가 좌우될 위험이 있다
    (experiments/baseline_lgbm/rated_output_investigation.md 참고 — holdout에서
    좋아 보인 개선이 실제 제출에서 3번 연속 뒤집힌 사례들). valid_years에 있는
    연도 수만큼 폴드를 돌려 평균/표준편차를 같이 보면, 특정 연도 하나에 대한
    과적합인지 실제로 일반화되는 개선인지 더 신뢰성 있게 판단할 수 있다.

    각 연도 구간은 [1/1 00:00, 다음해 1/1 00:00) 로 자른다. valid_years에
    없는 연도의 데이터(예: 라벨 경계에 걸친 자투리 행)는 train/holdout
    어디에도 포함되지 않고 자동으로 제외된다.
    """
    df = df.sort_values(time_col).reset_index(drop=True)

    ho_lo, ho_hi = _year_bounds(holdout_year)
    holdout = df[(df[time_col] >= ho_lo) & (df[time_col] < ho_hi)].copy()

    train_mask = pd.Series(False, index=df.index)
    for year in valid_years:
        if year == holdout_year:
            continue
        lo, hi = _year_bounds(year)
        train_mask |= (df[time_col] >= lo) & (df[time_col] < hi)
    train = df[train_mask].copy()

    return train, holdout


def _quarter_bounds(year: int, quarter: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_month = 3 * (quarter - 1) + 1
    start = pd.Timestamp(year=year, month=start_month, day=1)
    if quarter == 4:
        end = pd.Timestamp(year=year + 1, month=1, day=1)
    else:
        end = pd.Timestamp(year=year, month=start_month + 3, day=1)
    return start, end


def time_based_split_leave_quarter_out(
    df: pd.DataFrame,
    holdout_quarter: tuple[int, int],
    valid_quarters: list[tuple[int, int]],
    time_col: str = "forecast_kst_dtm",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-quarter-out 분할: (year, quarter) 튜플로 지정한 분기 하나를
    통째로 holdout으로 떼어내고, valid_quarters 중 나머지 분기를 (연도 경계를
    걸쳐) 전부 합쳐 train으로 쓴다.

    leave-one-year-out(연도 전체를 통째로 떼어냄, 3(2)-fold뿐)보다 fold 수를
    최대 12(group3는 8)까지 늘려 "진짜 재학습된" 독립적인 검증 지점을 더
    많이 얻기 위한 것 — 단, 각 fold의 train이 "이전 연도 전체"가 아니라
    "다른 연도들 + 같은 연도의 다른 분기들"이 섞인 형태가 되어, 실제
    프로덕션(이전 연도 전체로 학습해 다음 연도를 예측)과는 학습 데이터
    구성 방식이 달라진다는 trade-off가 있다(experiments/baseline_lgbm/
    rated_output_investigation.md 38번 섹션 참고).

    holdout_quarter/valid_quarters의 각 원소는 (year, quarter) — quarter는
    1~4(1=1~3월, 2=4~6월, 3=7~9월, 4=10~12월).
    """
    df = df.sort_values(time_col).reset_index(drop=True)

    ho_lo, ho_hi = _quarter_bounds(*holdout_quarter)
    holdout = df[(df[time_col] >= ho_lo) & (df[time_col] < ho_hi)].copy()

    train_mask = pd.Series(False, index=df.index)
    for q in valid_quarters:
        if q == holdout_quarter:
            continue
        lo, hi = _quarter_bounds(*q)
        train_mask |= (df[time_col] >= lo) & (df[time_col] < hi)
    train = df[train_mask].copy()

    return train, holdout
