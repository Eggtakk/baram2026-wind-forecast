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


HUB_HEIGHT_M = 117.0  # info.xlsx 기준 3개 그룹 터빈 모두 동일 (Hub Height(m)=117)
GAS_CONSTANT_DRY_AIR = 287.05  # J / (kg*K)
DEFAULT_SHEAR_EXPONENT = 1 / 7  # 관측 높이가 하나뿐일 때 쓰는 표준 근사 지수(오픈 터레인 관례값)


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """물리적으로 동기 부여된 파생 feature.

    발전량은 대략 P ∝ rho * v_hub^3 (rho=공기밀도, v_hub=허브높이 풍속)을
    따르므로, 관측 높이의 풍속을 허브높이(117m)로 외삽하고 공기밀도를 곁들인
    "이론적 파워 프록시"를 넣어준다. 트리 모델은 비선형 관계를 스스로 학습할
    수 있지만, 관측치가 부족한 구간(고풍속 등)에서는 물리 식이 외삽에 도움될
    수 있어 시도해본다.
    """
    df = df.copy()

    # 1) 허브높이 외삽 풍속
    #    GFS는 10m/100m 두 높이가 있어 전단지수(shear exponent)를 직접 추정 가능.
    if {"gfs_ws10_speed", "gfs_ws100_speed"}.issubset(df.columns):
        v10 = df["gfs_ws10_speed"].clip(lower=0.1)
        v100 = df["gfs_ws100_speed"].clip(lower=0.1)
        alpha = np.log(v100 / v10) / np.log(100 / 10)
        df["gfs_shear_exponent"] = alpha
        df["gfs_hub_speed"] = v100 * (HUB_HEIGHT_M / 100) ** alpha

    #    LDAPS는 10m만 있으므로 표준 전단지수로 단순 외삽.
    if "ldaps_ws10_speed" in df.columns:
        df["ldaps_hub_speed"] = df["ldaps_ws10_speed"] * (HUB_HEIGHT_M / 10) ** DEFAULT_SHEAR_EXPONENT

    # 2) 공기밀도 (이상기체 근사: rho = P / (R*T))
    if {"ldaps_surface_0_sp", "ldaps_heightAboveGround_2_t"}.issubset(df.columns):
        df["ldaps_air_density"] = df["ldaps_surface_0_sp"] / (
            GAS_CONSTANT_DRY_AIR * df["ldaps_heightAboveGround_2_t"]
        )

    # 3) 파워 프록시: rho * v_hub^3 (스케일은 임의, 모델이 학습으로 흡수)
    if "ldaps_hub_speed" in df.columns and "ldaps_air_density" in df.columns:
        df["ldaps_power_proxy"] = df["ldaps_air_density"] * df["ldaps_hub_speed"] ** 3
    for col in ["ldaps_ws10_speed", "gfs_ws100_speed", "gfs_hub_speed"]:
        if col in df.columns:
            df[f"{col}_cubed"] = df[col] ** 3

    # 4) LDAPS 50m 성분 변동폭 -> 돌풍/난류 프록시
    ldaps_50m_cols = {
        "u_max": "ldaps_heightAboveGround_50_50MUmax",
        "u_min": "ldaps_heightAboveGround_50_50MUmin",
        "v_max": "ldaps_heightAboveGround_50_50MVmax",
        "v_min": "ldaps_heightAboveGround_50_50MVmin",
    }
    if set(ldaps_50m_cols.values()).issubset(df.columns):
        du = df[ldaps_50m_cols["u_max"]] - df[ldaps_50m_cols["u_min"]]
        dv = df[ldaps_50m_cols["v_max"]] - df[ldaps_50m_cols["v_min"]]
        df["ldaps_gust_proxy_50m"] = np.sqrt(du**2 + dv**2)

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


NON_FEATURE_COLS = {
    "forecast_kst_dtm",
    "ldaps_data_available_kst_dtm",
    "y",
    "group_id",
}


def build_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """train.py / inference.py / validate_baseline.py가 공유하는 기본 feature 레시피.

    train과 inference가 서로 다른 feature 로직을 쓰면 학습-추론 불일치(스큐)가
    생기므로, 반드시 이 함수 하나만 양쪽에서 호출한다.
    """
    df = add_default_wind_features(df)
    df = add_physics_features(df)
    df = add_time_features(df)
    speed_cols = [c for c in df.columns if c.endswith("_speed")]
    df = add_lag_rolling_features(df, cols=speed_cols, lags=[1, 2, 3], windows=[3, 6, 24])
    return df


def build_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """LSTM(src/sequence_data.py)이 쓰는 feature 레시피. build_baseline_features와
    거의 같지만 lag/rolling 파생 feature는 뺀다.

    LSTM은 과거 seq_len시간을 그대로 입력으로 받으므로, LightGBM처럼 매
    시점마다 "과거 24시간 평균/표준편차"를 미리 손으로 계산해 넣어주는 게
    오히려 같은 정보를 창(window) 안에 여러 번 중복해서 넣는 꼴이 된다.
    실제로 154개 feature(=lag/rolling 포함)로 학습했더니 3~8 epoch 만에
    holdout score가 정점을 찍고 그 뒤로는 과적합으로 계속 나빠지는 문제가
    있었다 — feature 수를 줄여(~90개) 과적합 압력을 낮추기 위한 변형이다.
    """
    df = add_default_wind_features(df)
    df = add_physics_features(df)
    df = add_time_features(df)
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """build_baseline_features() 결과에서 모델 입력으로 쓸 컬럼만 골라낸다.

    scada_* 컬럼은 test 기간에 존재하지 않으므로(data_loader.py 참고) 항상 제외한다.
    """
    scada_cols = [c for c in df.columns if c.startswith("scada_")]
    return [c for c in df.columns if c not in NON_FEATURE_COLS and c not in scada_cols]


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
