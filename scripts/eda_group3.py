"""
Group 3 (KPX group 3 / UNISON 터빈) 초기 EDA 스크립트.

담당: 탁예린 (그룹3)

수행 내용
1. info.xlsx에서 UNISON 터빈(그룹3) 좌표를 읽어 centroid 계산 후,
   LDAPS(16격자) / GFS(9격자) 중 가장 가까운 격자를 선정한다.
2. train_labels.csv의 kpx_group_3 컬럼에 대한 결측/이상치/분포를 확인한다.
3. 월별/시간대별 발전량 패턴(계절성, 일중 패턴)을 확인한다.
4. 선정된 격자의 풍속 예보값과 실제 발전량 간 동시상관 및 lag(-6~+6h)
   교차상관, lead-time(예보 시계)별 상관계수를 확인한다.

결과물: experiments/group3_eda/ 아래 PNG, summary.md
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "open"
OUT = ROOT / "experiments" / "group3_eda"
OUT.mkdir(parents=True, exist_ok=True)

CAPACITY_KWH = 21_000  # kpx_group_3 설비용량 21.0MW * 1h


def dms_to_dd(s: str) -> float:
    m = re.match(r"(\d+)\D+(\d+)\D+([\d.]+)\"?([NSEW])", s.strip())
    d, mi, se, hemi = m.groups()
    val = float(d) + float(mi) / 60 + float(se) / 3600
    return -val if hemi in ("S", "W") else val


def find_group3_centroid() -> tuple[float, float]:
    df = pd.ExcelFile(DATA / "info.xlsx").parse("info", header=3)
    df.columns = [str(c).strip() for c in df.columns]
    unison = df[df["제작사"] == "UNISON"].copy()
    coords = unison["좌표(Google)"].astype(str)
    lats, lons = [], []
    for c in coords:
        lat_s, lon_s = c.split()
        lats.append(dms_to_dd(lat_s))
        lons.append(dms_to_dd(lon_s))
    return float(np.mean(lats)), float(np.mean(lons))


def nearest_grids(csv_path: Path, target_lat: float, target_lon: float, n: int) -> list[int]:
    df = pd.read_csv(csv_path, usecols=["grid_id", "latitude", "longitude"], nrows=200_000)
    g = df.drop_duplicates("grid_id")[["grid_id", "latitude", "longitude"]]
    g["dist"] = np.sqrt((g["latitude"] - target_lat) ** 2 + (g["longitude"] - target_lon) ** 2)
    return g.sort_values("dist")["grid_id"].head(n).astype(int).tolist()


def load_labels() -> pd.DataFrame:
    labels = pd.read_csv(DATA / "train" / "train_labels.csv", parse_dates=["kst_dtm"])
    return labels[["kst_dtm", "kpx_group_3"]].rename(columns={"kst_dtm": "forecast_kst_dtm"})


def analyze_label_distribution(g3: pd.DataFrame) -> dict:
    valid = g3.dropna(subset=["kpx_group_3"])
    full_range = pd.date_range(valid["forecast_kst_dtm"].min(), valid["forecast_kst_dtm"].max(), freq="h")
    gaps = full_range.difference(valid["forecast_kst_dtm"])
    stats = {
        "total_rows": len(g3),
        "missing_rows": int(g3["kpx_group_3"].isna().sum()),
        "valid_range": (valid["forecast_kst_dtm"].min(), valid["forecast_kst_dtm"].max()),
        "valid_rows": len(valid),
        "internal_gaps": len(gaps),
        "negative_count": int((valid["kpx_group_3"] < 0).sum()),
        "over_capacity_count": int((valid["kpx_group_3"] > CAPACITY_KWH).sum()),
        "zero_count": int((valid["kpx_group_3"] == 0).sum()),
        "describe": valid["kpx_group_3"].describe(),
    }
    return stats


def plot_seasonality(g3: pd.DataFrame):
    valid = g3.dropna(subset=["kpx_group_3"]).copy()
    valid["month"] = valid["forecast_kst_dtm"].dt.month
    valid["hour"] = valid["forecast_kst_dtm"].dt.hour

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    valid.boxplot(column="kpx_group_3", by="month", ax=axes[0, 0])
    axes[0, 0].set_title("Monthly distribution")
    axes[0, 0].set_xlabel("month")
    axes[0, 0].set_ylabel("kWh")

    valid.boxplot(column="kpx_group_3", by="hour", ax=axes[0, 1])
    axes[0, 1].set_title("Hourly distribution")
    axes[0, 1].set_xlabel("hour")
    axes[0, 1].set_ylabel("kWh")

    axes[1, 0].hist(valid["kpx_group_3"], bins=60)
    axes[1, 0].set_title("Histogram")
    axes[1, 0].set_xlabel("kWh")

    valid.set_index("forecast_kst_dtm")["kpx_group_3"].plot(ax=axes[1, 1], linewidth=0.3)
    axes[1, 1].set_title("Time series (valid range)")

    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(OUT / "label_distribution_seasonality.png", dpi=110)
    plt.close(fig)

    return (
        valid.groupby("month")["kpx_group_3"].mean() / CAPACITY_KWH * 100,
        valid.groupby("hour")["kpx_group_3"].mean(),
    )


def build_weather_features(ldaps_grids: list[int], gfs_grids: list[int]) -> pd.DataFrame:
    cols_l = [
        "forecast_kst_dtm", "data_available_kst_dtm", "grid_id",
        "heightAboveGround_10_10u", "heightAboveGround_10_10v",
    ]
    ldaps = pd.read_csv(
        DATA / "train" / "ldaps_train.csv", usecols=cols_l,
        parse_dates=["forecast_kst_dtm", "data_available_kst_dtm"],
    )
    ldaps = ldaps[ldaps["grid_id"].isin(ldaps_grids)]
    ldaps["ws10"] = np.sqrt(ldaps["heightAboveGround_10_10u"] ** 2 + ldaps["heightAboveGround_10_10v"] ** 2)
    ldaps_agg = (
        ldaps.groupby("forecast_kst_dtm")
        .agg(ws10_ldaps=("ws10", "mean"), data_available_kst_dtm=("data_available_kst_dtm", "first"))
        .reset_index()
    )

    cols_g = [
        "forecast_kst_dtm", "grid_id",
        "heightAboveGround_10_10u", "heightAboveGround_10_10v",
        "heightAboveGround_100_100u", "heightAboveGround_100_100v",
    ]
    gfs = pd.read_csv(DATA / "train" / "gfs_train.csv", usecols=cols_g, parse_dates=["forecast_kst_dtm"])
    gfs = gfs[gfs["grid_id"].isin(gfs_grids)]
    gfs["ws10"] = np.sqrt(gfs["heightAboveGround_10_10u"] ** 2 + gfs["heightAboveGround_10_10v"] ** 2)
    gfs["ws100"] = np.sqrt(gfs["heightAboveGround_100_100u"] ** 2 + gfs["heightAboveGround_100_100v"] ** 2)
    gfs_agg = gfs.groupby("forecast_kst_dtm").agg(ws10_gfs=("ws10", "mean"), ws100_gfs=("ws100", "mean")).reset_index()

    return ldaps_agg.merge(gfs_agg, on="forecast_kst_dtm", how="outer")


def analyze_lag_correlation(g3: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    merged = g3.merge(weather, on="forecast_kst_dtm", how="left").dropna(subset=["kpx_group_3"])
    merged = merged.set_index("forecast_kst_dtm").sort_index().asfreq("h")

    lags = range(-6, 7)
    rows = []
    for lag in lags:
        shifted = merged[["ws10_ldaps", "ws10_gfs", "ws100_gfs"]].shift(lag)
        rows.append({
            "lag": lag,
            "corr_ldaps_ws10": merged["kpx_group_3"].corr(shifted["ws10_ldaps"]),
            "corr_gfs_ws10": merged["kpx_group_3"].corr(shifted["ws10_gfs"]),
            "corr_gfs_ws100": merged["kpx_group_3"].corr(shifted["ws100_gfs"]),
        })
    res = pd.DataFrame(rows)

    plt.figure(figsize=(8, 5))
    plt.plot(res["lag"], res["corr_ldaps_ws10"], marker="o", label="LDAPS ws10")
    plt.plot(res["lag"], res["corr_gfs_ws10"], marker="o", label="GFS ws10")
    plt.plot(res["lag"], res["corr_gfs_ws100"], marker="o", label="GFS ws100")
    plt.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("lag (h)")
    plt.ylabel("correlation with kpx_group_3")
    plt.title("Cross-correlation: forecast wind speed vs actual power (Group 3)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "lag_crosscorrelation.png", dpi=110)
    plt.close()

    return res


def main():
    lat, lon = find_group3_centroid()
    ldaps_grids = nearest_grids(DATA / "train" / "ldaps_train.csv", lat, lon, n=4)
    gfs_grids = nearest_grids(DATA / "train" / "gfs_train.csv", lat, lon, n=1)
    print(f"Group3 centroid: ({lat:.4f}, {lon:.4f})")
    print(f"Selected LDAPS grids: {ldaps_grids}")
    print(f"Selected GFS grids: {gfs_grids}")

    g3 = load_labels()
    stats = analyze_label_distribution(g3)
    print("\n--- label distribution ---")
    for k, v in stats.items():
        print(k, ":", v)

    monthly_cf, hourly_mean = plot_seasonality(g3)
    print("\n--- monthly capacity factor (%) ---")
    print(monthly_cf.round(1))

    weather = build_weather_features(ldaps_grids, gfs_grids)
    lag_res = analyze_lag_correlation(g3, weather)
    print("\n--- lag correlation ---")
    print(lag_res.to_string(index=False))

    lag_res.to_csv(OUT / "lag_correlation.csv", index=False)
    print(f"\nSaved plots and results to {OUT}")


if __name__ == "__main__":
    main()
