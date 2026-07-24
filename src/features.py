"""
파생 feature 생성 함수 모음. src.preprocess.build_group_dataset()의 출력에 적용한다.
"""
import numpy as np
import pandas as pd

# (u_col, v_col, 출력 접두어) — build_group_dataset의 컬럼명 규칙(ldaps_/gfs_ 접두어)에 맞춤.
# 존재하는 쌍만 자동으로 적용되므로, config/그룹에 따라 컬럼이 없어도 에러 없이 스킵된다.
WIND_UV_PAIRS = [
    ("ldaps_heightAboveGround_10_10u", "ldaps_heightAboveGround_10_10v", "ldaps_ws10"),
    ("gfs_heightAboveGround_10_10u", "gfs_heightAboveGround_10_10v", "gfs_ws10"),
    ("gfs_heightAboveGround_80_u", "gfs_heightAboveGround_80_v", "gfs_ws80"),
    ("gfs_heightAboveGround_100_100u", "gfs_heightAboveGround_100_100v", "gfs_ws100"),
    ("gfs_planetaryBoundaryLayer_0_u", "gfs_planetaryBoundaryLayer_0_v", "gfs_ws_pbl"),
]


def add_wind_speed_direction(df: pd.DataFrame, u_col: str, v_col: str, prefix: str) -> pd.DataFrame:
    """u/v 성분으로부터 풍속(ws)과 기상학적 풍향(wd, 바람이 불어오는 방향, 0=N/360)을 계산."""
    df = df.copy()
    df[f"{prefix}_speed"] = np.sqrt(df[u_col] ** 2 + df[v_col] ** 2)
    df[f"{prefix}_dir"] = (np.rad2deg(np.arctan2(-df[u_col], -df[v_col])) + 360) % 360
    return df


def add_default_wind_features(df: pd.DataFrame) -> pd.DataFrame:
    """WIND_UV_PAIRS 중 df에 실제로 존재하는 컬럼쌍에 대해서만 풍속/풍향을 추가."""
    for u_col, v_col, prefix in WIND_UV_PAIRS:
        if u_col in df.columns and v_col in df.columns:
            df = add_wind_speed_direction(df, u_col, v_col, prefix)
    return df


def add_time_features(df: pd.DataFrame, time_col: str = "forecast_kst_dtm") -> pd.DataFrame:
    """월/시간의 계절성·일중 패턴을 반영하기 위한 캘린더 + 주기(sin/cos) feature."""
    df = df.copy()
    t = df[time_col]
    df["month"] = t.dt.month
    df["hour"] = t.dt.hour
    df["dayofweek"] = t.dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    return df


def add_lag_rolling_features(
    df: pd.DataFrame,
    cols: list[str],
    lags: list[int] = (1, 2, 3, 6),
    windows: list[int] = (3, 6, 24),
    time_col: str = "forecast_kst_dtm",
) -> pd.DataFrame:
    """지정한 컬럼들에 대해 lag / rolling mean·std feature를 추가.

    df는 반드시 시간순 정렬 + 결측 없는 연속 시간축(1h 간격)이어야 lag가 의미를
    가진다. 필요하면 호출 전 `df.set_index(time_col).asfreq('h')`로 gap을
    메운 뒤 다시 reset_index() 하고 넘길 것.

    주의: LDAPS/GFS 기반 feature에만 사용할 것. SCADA/라벨 기반 lag는 그
    시점 실측이 필요해 test 추론 시 사용할 수 없다(과거 라벨 lag 제외).
    """
    df = df.sort_values(time_col).copy()
    for col in cols:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
        for w in windows:
            df[f"{col}_roll{w}_mean"] = df[col].shift(1).rolling(w, min_periods=1).mean()
            df[f"{col}_roll{w}_std"] = df[col].shift(1).rolling(w, min_periods=1).std()
    return df
