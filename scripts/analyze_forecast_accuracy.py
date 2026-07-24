"""
LDAPS/GFS 예보 풍속 자체의 정확도를, 특히 고풍속(정격출력 근처) 구간에서
검증하는 스크립트.

SCADA의 scada_mean_ws(나셀 풍속계 실측, 허브 높이 근처)를 "정답"으로 삼아,
예보 풍속(원시 10m/100m 및 허브높이 외삽값)과 비교한다. 목표: 정격출력
구간(90-100% capacity)에서 관측된 발전량 과소예측(under-prediction) bias가
"모델이 못 배운 것"이 아니라 "예보 풍속 입력 자체가 그 구간에서 부정확한
것"인지 확인.

이 스크립트는 모델을 학습하지 않는다 — 순수 입력 데이터 품질 분석.

실행: (레포 루트에서) python3 scripts/analyze_forecast_accuracy.py [group_id ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.features import add_default_wind_features, add_physics_features
from src.metrics import CAPACITY_KWH
from src.preprocess import build_group_dataset

FORECAST_COLS = ["ldaps_ws10_speed", "ldaps_hub_speed", "gfs_ws10_speed", "gfs_ws100_speed", "gfs_hub_speed"]

WS_BINS = [0, 2, 4, 6, 8, 10, 12, 14, 100]
WS_LABELS = ["0-2", "2-4", "4-6", "6-8", "8-10", "10-12", "12-14", "14+"]

CAP_EDGES = np.arange(0.0, 1.01, 0.1)
CAP_LABELS = [f"{int(CAP_EDGES[i]*100)}-{int(CAP_EDGES[i+1]*100)}%" for i in range(len(CAP_EDGES) - 1)]


def load_group(group_id: int) -> tuple[pd.DataFrame, float]:
    df = build_group_dataset(group_id, split="train", include_scada=True)
    df = add_default_wind_features(df)
    df = add_physics_features(df)
    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]
    df = df.dropna(subset=["y", "scada_mean_ws"]).reset_index(drop=True)
    return df, capacity


def by_wind_speed_bin(df: pd.DataFrame, col: str) -> pd.DataFrame:
    actual = df["scada_mean_ws"]
    forecast = df[col]
    bias = forecast - actual
    abs_err = bias.abs()
    band = pd.cut(actual, bins=WS_BINS, labels=WS_LABELS, right=False)
    out = pd.DataFrame({"band": band, "bias": bias, "abs_err": abs_err})
    return out.groupby("band", observed=True).agg(n=("bias", "size"), mean_bias=("bias", "mean"), mae=("abs_err", "mean"))


def by_capacity_bin(df: pd.DataFrame, col: str, capacity: float) -> pd.DataFrame:
    actual_ws = df["scada_mean_ws"]
    forecast_ws = df[col]
    ws_bias = forecast_ws - actual_ws
    ws_abs_err = ws_bias.abs()

    cap_ratio = df["y"] / capacity
    band = pd.cut(cap_ratio, bins=CAP_EDGES, labels=CAP_LABELS, right=True, include_lowest=True)

    out = pd.DataFrame({"band": band, "ws_bias": ws_bias, "ws_abs_err": ws_abs_err})
    return out.groupby("band", observed=True).agg(
        n=("ws_bias", "size"), mean_ws_bias=("ws_bias", "mean"), ws_mae=("ws_abs_err", "mean")
    )


def main():
    groups = [int(a) for a in sys.argv[1:]] or [1, 2, 3]
    for gid in groups:
        df, capacity = load_group(gid)
        print(f"\n{'='*70}\ngroup{gid} (capacity={capacity} kWh, n={len(df)})\n{'='*70}")

        for col in FORECAST_COLS:
            if col not in df.columns:
                continue
            overall_bias = (df[col] - df["scada_mean_ws"]).mean()
            overall_mae = (df[col] - df["scada_mean_ws"]).abs().mean()
            corr = df[col].corr(df["scada_mean_ws"])
            print(f"\n--- {col} vs scada_mean_ws --- overall bias={overall_bias:+.3f} MAE={overall_mae:.3f} corr={corr:.3f}")
            print("  [실측 풍속 구간별]")
            print(by_wind_speed_bin(df, col).to_string())

        # 정격출력(y/capacity) 구간별로, 대표 forecast(허브높이 외삽)의 풍속오차 확인
        rep_col = "ldaps_hub_speed" if "ldaps_hub_speed" in df.columns else FORECAST_COLS[0]
        print(f"\n  [정격출력(y/capacity) 구간별 {rep_col} 풍속오차]")
        print(by_capacity_bin(df, rep_col, capacity).to_string())

        if "gfs_hub_speed" in df.columns:
            print(f"\n  [정격출력(y/capacity) 구간별 gfs_hub_speed 풍속오차]")
            print(by_capacity_bin(df, "gfs_hub_speed", capacity).to_string())


if __name__ == "__main__":
    main()
