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


def add_forecast_disagreement_features(df: pd.DataFrame) -> pd.DataFrame:
    """LDAPS와 GFS, 두 예보 소스 간 불일치(disagreement)를 feature로 추가.

    (실험용 — 아직 프로덕션 레시피(build_physics_features/build_baseline_features)에는
    포함되지 않음. scripts/validate_loyo_candidates.py에서 LOYO로 검증 중.
    검증 통과 시 build_physics_features/build_baseline_features에 편입할 것.)

    근거: `scripts/analyze_forecast_accuracy.py` 결과, 정격출력 구간으로 갈수록
    예보풍속의 MAE/분산이 뚜렷하게 커진다(예: group1 gfs_hub_speed MAE가
    0-10% 출력구간 1.8m/s -> 90-100% 구간 6.8~7.4m/s로 증가,
    rated_output_investigation.md 4번 섹션 참고). "지금 이 시각 예보가
    얼마나 불확실한가"를 모델이 직접 참고할 수 있는 신호를 주면, 불확실성이
    큰 시간대를 다르게 다루는 법을 학습할 여지가 생긴다는 가설.

    이전에 시도했던 "예보풍속 자체를 실측 기준으로 보정"(5번 섹션, 기각)과는
    질적으로 다르다 — 트리 기반 모델은 개별 feature의 단조(monotonic)
    변환에는 사실상 불변이라 그 보정은 추가 정보가 거의 없었지만, 두 예보
    소스의 차이(disagreement)는 원본 feature들의 비단조 조합(interaction)이라
    트리가 기존 feature만으로 스스로 만들어낼 수 없는 새로운 정보다.
    """
    df = df.copy()
    if {"ldaps_hub_speed", "gfs_hub_speed"}.issubset(df.columns):
        df["hub_speed_disagreement"] = (df["ldaps_hub_speed"] - df["gfs_hub_speed"]).abs()
    if {"ldaps_ws10_speed", "gfs_ws10_speed"}.issubset(df.columns):
        df["ws10_speed_disagreement"] = (df["ldaps_ws10_speed"] - df["gfs_ws10_speed"]).abs()
    if {"ldaps_ws10_dir", "gfs_ws10_dir"}.issubset(df.columns):
        # 풍향은 원형(circular) 변수라 단순 차가 아니라 0~180도 범위로 감아준다.
        diff = (df["ldaps_ws10_dir"] - df["gfs_ws10_dir"]).abs() % 360
        df["ws10_dir_disagreement"] = np.minimum(diff, 360 - diff)
    return df


def add_manufacturer_power_curve_feature(df: pd.DataFrame, group_id: int) -> pd.DataFrame:
    """터빈 제작사 공식 파워커브(외부 공개 데이터) 기반 발전량 추정 feature.

    (실험용 — 아직 프로덕션 레시피에는 편입되지 않음. src/manufacturer_power_curve.py
    상단 docstring에 출처/재현성 설명, docs/external_data_manufacturer_power_curve.md
    참고. scripts/validate_loyo_candidates.py "manufacturer_curve" 후보로 LOYO 검증 중.)

    기존 `src/power_curve.py`의 isotonic 커브는 이 대회의 train 데이터(풍속-발전량)
    자체에서 통계적으로 학습한 것이라, train에 없는 극단적 고풍속 구간에서는
    외삽(extrapolation)에 의존한다. 이 feature는 반대로 터빈 제작사가 공개한
    실제 설계 파워커브를 그대로 조회하는 것이라, train 데이터의 분포와 무관하게
    물리적으로 타당한 값을 준다 — 특히 정격출력(rated) 부근 오차가 가장 컸던
    문제(rated_output_investigation.md 1~4번 섹션)에 직접 도움이 될 수 있다는
    가설.
    """
    df = df.copy()
    if not {"ldaps_hub_speed", "ldaps_air_density"}.issubset(df.columns):
        return df
    from src.manufacturer_power_curve import TURBINE_BY_GROUP, estimate_power_kw

    turbine = TURBINE_BY_GROUP[group_id]
    df["manufacturer_curve_est"] = estimate_power_kw(
        df["ldaps_hub_speed"], df["ldaps_air_density"], turbine
    )
    return df


def add_wind_direction_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """`*_dir`(0~360도, 원형 변수) 컬럼들에 sin/cos 인코딩을 추가.

    (실험용 — 아직 프로덕션 레시피에는 편입되지 않음. add_forecast_disagreement_features와
    같은 방식으로 scripts/validate_loyo_candidates.py에서 LOYO로 검증 중.)

    현재 add_default_wind_features()가 만드는 `*_dir` 컬럼들은 0~360도의
    raw 각도 그대로다. 트리 모델은 이 값 자체가 아니라 분할 임계값을
    학습하므로 359도와 1도가 물리적으로는 거의 같은 방향이라는 걸 표현하려면
    "> 350 또는 < 10" 같은 분할을 여러 겹 겹쳐야 한다 — sin/cos로 인코딩하면
    이 wrap-around를 좌표 하나로 바로 표현할 수 있다. 산악 지형 풍력단지라
    지형 채널링/웨이크 효과로 풍향 자체가 의미 있는 신호일 가능성이 있는데
    (group3 EDA의 야간 활강풍 패턴 참고), 지금은 raw 각도로만 들어가 있어
    이 정보를 모델이 비효율적으로만 활용하고 있을 수 있다.
    """
    df = df.copy()
    dir_cols = [c for c in df.columns if c.endswith("_dir")]
    for col in dir_cols:
        rad = np.deg2rad(df[col])
        df[f"{col}_sin"] = np.sin(rad)
        df[f"{col}_cos"] = np.cos(rad)
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


SATURATION_WIND_COLS = ["ldaps_ws10_speed", "ldaps_hub_speed", "gfs_hub_speed", "gfs_ws100_speed"]
SATURATION_CLIP_SPEEDS = [6, 7, 8, 9, 10, 12]  # 실측 기준 90~100% 출력대 풍속이 대략 7~9 m/s(10m 기준)였음


def add_saturation_features(df: pd.DataFrame) -> pd.DataFrame:
    """정격출력(파워커브 포화) 구간을 모델이 더 잘 잡도록 돕는 결정론적 feature.

    발전량은 rated wind speed 이상에서는 풍속이 더 올라가도 출력이 더 안 올라가고
    평평해지는데(포화), 원시 풍속이나 v^3 feature만으로는 트리 모델이 이 "꺾이는
    지점"을 스스로 여러 번 분할해서 근사해야 한다. min(v, 임계값) 형태의 클리핑
    feature를 미리 여러 임계값으로 만들어주면 그 분할을 대신 해주는 효과가 있다.
    실측 확인 결과(그룹1: 90~100% 출력 구간 ldaps_ws10_speed 중앙값 ≈ 6.8m/s,
    그룹3 ≈ 9.0m/s) 근처 값들로 후보를 잡았다.
    """
    df = df.copy()
    for col in SATURATION_WIND_COLS:
        if col not in df.columns:
            continue
        for clip in SATURATION_CLIP_SPEEDS:
            df[f"{col}_clip{clip}"] = df[col].clip(upper=clip)
        # 완만한 시그모이드 포화 지표 (중심 8m/s 근처 — 위 실측값들의 중간)
        df[f"{col}_saturation"] = 1 / (1 + np.exp(-(df[col] - 8)))
        # 임계값을 넘어선 "초과분"만 따로 (포화 이후 완만한 증가/정체를 분리해서 표현)
        df[f"{col}_excess8"] = (df[col] - 8).clip(lower=0)
    return df


def build_baseline_features(df: pd.DataFrame, wind_correctors: dict | None = None) -> pd.DataFrame:
    """train.py / inference.py / validate_baseline.py가 공유하는 기본 feature 레시피.

    train과 inference가 서로 다른 feature 로직을 쓰면 학습-추론 불일치(스큐)가
    생기므로, 반드시 이 함수 하나만 양쪽에서 호출한다.

    wind_correctors: src.wind_bias_correction.fit_wind_bias_correctors_from_raw()로
        학습한 예보풍속 보정기(train 전용으로 fit). 주어지면 물리 feature 계산
        직후, 포화(saturation) feature 계산 전에 허브높이 예보풍속을 보정한다.
    """
    df = add_default_wind_features(df)
    df = add_physics_features(df)
    if wind_correctors:
        from src.wind_bias_correction import apply_wind_bias_correction

        df = apply_wind_bias_correction(df, wind_correctors)
    df = add_saturation_features(df)
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
