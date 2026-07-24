"""
그룹(1/2/3) 공용 전처리 파이프라인.

핵심 함수는 `build_group_dataset(group_id, split)` 하나로, config만 바뀌면
세 그룹 모두 같은 코드로 동작하도록 만들었다.

시간 정렬 규칙
--------------
- LDAPS/GFS: `forecast_kst_dtm`이 곧 예측 대상 시각이며 label(kst_dtm)과
  동일한 시간 기준이다. EDA 결과 lag=0에서 예보-발전량 상관이 가장 높게
  나와, 별도의 시간 이동(shift) 없이 그대로 merge key로 사용한다
  (experiments/group3_eda/summary.md 참고).
- SCADA: 10분 단위 실측이며, `train_labels.csv`의 `kst_dtm`은 "집계 구간의
  종료 시각"이다. 즉 kst_dtm=T 라벨은 (T-1h, T] 구간의 발전량이므로,
  SCADA도 동일한 컨벤션(label='right', closed='right')으로 시간당
  리샘플링해야 라벨과 정확히 맞물린다. 10분 power 값(kW10m, 10분 평균
  출력)의 시간당 합산 에너지(kWh)는 산술적으로 "그 시간에 속한 6개
  10분값의 평균"과 같다 (각 10분 구간 에너지 = power_kW / 6).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import (
    CONFIG_DIR,
    DATA_DIR,
    load_group_config,
    load_labels,
    load_scada,
    load_weather,
)


def aggregate_weather_grids(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """선정된 grid들을 forecast_kst_dtm 기준으로 평균내어 1행/시각으로 축소한다.

    반환 컬럼은 원본 기상 변수명 앞에 f"{source}_" 접두어를 붙여 LDAPS/GFS를
    합칠 때 컬럼명이 충돌하지 않게 한다. (예: ldaps_heightAboveGround_10_10u)
    """
    meta_cols = {"forecast_kst_dtm", "data_available_kst_dtm", "grid_id", "latitude", "longitude"}
    value_cols = [c for c in df.columns if c not in meta_cols]

    agg = df.groupby("forecast_kst_dtm")[value_cols].mean()
    if "data_available_kst_dtm" in df.columns:
        avail = df.groupby("forecast_kst_dtm")["data_available_kst_dtm"].first()
        agg["data_available_kst_dtm"] = avail
    agg = agg.reset_index()

    rename = {c: f"{source}_{c}" for c in value_cols}
    return agg.rename(columns=rename)


def aggregate_scada_hourly(df: pd.DataFrame, turbines: list[str]) -> pd.DataFrame:
    """10분 SCADA를 label과 동일한 컨벤션(구간 종료 시각)으로 시간당 집계.

    반환: kst_dtm, scada_total_power_kwh(터빈 합, kWh 근사), scada_mean_ws, scada_mean_wd
    """
    df = df.set_index("kst_dtm").sort_index()
    power_cols = [c for c in df.columns if c.endswith("_power_kw10m")]
    ws_cols = [c for c in df.columns if c.endswith("_ws")]
    wd_cols = [c for c in df.columns if c.endswith("_wd")]

    hourly = df[power_cols + ws_cols].resample("1h", label="right", closed="right").mean()

    out = pd.DataFrame(index=hourly.index)
    # 10분 평균출력(kW)의 시간 평균 = 시간당 에너지(kWh)와 산술적으로 동일
    out["scada_total_power_kwh"] = hourly[power_cols].sum(axis=1)
    out["scada_mean_ws"] = hourly[ws_cols].mean(axis=1)

    if wd_cols:
        # 풍향은 원형(circular) 변수이므로 단순 평균 대신 벡터 평균 사용
        rad = np.deg2rad(df[wd_cols])
        sin_mean = np.sin(rad).resample("1h", label="right", closed="right").mean().mean(axis=1)
        cos_mean = np.cos(rad).resample("1h", label="right", closed="right").mean().mean(axis=1)
        out["scada_mean_wd"] = (np.rad2deg(np.arctan2(sin_mean, cos_mean)) + 360) % 360

    return out.reset_index().rename(columns={"kst_dtm": "forecast_kst_dtm"})


def build_group_dataset(
    group_id: int,
    split: str = "train",
    include_scada: bool | None = None,
    config_dir: Path = CONFIG_DIR,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """그룹(1/2/3)의 병합된 시간당 데이터셋을 만든다.

    split="train": ldaps/gfs + (옵션) scada + 라벨(y) 을 모두 merge.
    split="test" : ldaps/gfs만 사용 (라벨/scada는 test 기간에 존재하지 않음).
    include_scada: None이면 split=="train"일 때만 자동으로 포함.
        추론용 feature로는 사용하지 말 것 (data_loader.load_scada 참고).
    """
    if include_scada is None:
        include_scada = split == "train"
    if split == "test" and include_scada:
        raise ValueError("SCADA is not available for the test split.")

    config = load_group_config(group_id, config_dir=config_dir)

    ldaps = load_weather("ldaps", split=split, grids=config["ldaps_grids"], data_dir=data_dir)
    gfs = load_weather("gfs", split=split, grids=config["gfs_grids"], data_dir=data_dir)

    ldaps_agg = aggregate_weather_grids(ldaps, source="ldaps")
    gfs_agg = aggregate_weather_grids(gfs, source="gfs")

    # data_available_kst_dtm은 둘 다 있으니 ldaps 기준 하나만 남긴다
    gfs_agg = gfs_agg.drop(columns=["gfs_data_available_kst_dtm"], errors="ignore")

    data = ldaps_agg.merge(gfs_agg, on="forecast_kst_dtm", how="outer")

    if split == "train":
        labels = load_labels(data_dir=data_dir)
        y = labels[["kst_dtm", config["label_column"]]].rename(
            columns={"kst_dtm": "forecast_kst_dtm", config["label_column"]: "y"}
        )
        data = data.merge(y, on="forecast_kst_dtm", how="left")

        if include_scada:
            scada_raw = load_scada(config, split="train", data_dir=data_dir)
            scada_hourly = aggregate_scada_hourly(scada_raw, config["turbines"])
            data = data.merge(scada_hourly, on="forecast_kst_dtm", how="left")

    data["group_id"] = group_id
    return data.sort_values("forecast_kst_dtm").reset_index(drop=True)
