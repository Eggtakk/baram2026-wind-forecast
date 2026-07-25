"""
SCADA 커틀먼트/고장 의심 구간 진단.

실측 풍속(scada_mean_ws, 나셀 풍속계)을 기준으로 "이 풍속이면 보통 이 정도
발전한다"는 경험적 파워커브(isotonic, y ~ scada_mean_ws)를 만들고, 실제
발전량(y)이 그 커브보다 크게 낮은 시간대를 찾는다. 예보가 아니라 "실측"
풍속 기준이므로, 여기서 나오는 잔차는 예보 오차가 아니라 순수하게
"바람은 있었는데 발전이 안 된" 운영상 이유(커틀먼트, 정지, 고장 등)를
가리킬 가능성이 높다.

실행: (레포 루트에서) python3 scripts/diagnose_curtailment.py [group_id ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.metrics import CAPACITY_KWH
from src.preprocess import build_group_dataset

# 실측 풍속 대비 잔차가 "설비용량 대비 이 비율"보다 더 낮으면 의심 구간으로 표시.
RESIDUAL_THRESHOLD_RATIO = 0.30
# 이 실측 풍속(m/s) 이상인데 발전량이 낮으면 특히 의심스러움 (전형적 rated 근방).
HIGH_WIND_THRESHOLD = 8.0


def run_group(group_id: int):
    df = build_group_dataset(group_id, split="train", include_scada=True)
    df = df.dropna(subset=["y", "scada_mean_ws"]).reset_index(drop=True)
    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]

    curve = IsotonicRegression(y_min=0, y_max=capacity, increasing=True, out_of_bounds="clip")
    curve.fit(df["scada_mean_ws"], df["y"])
    df["curve_est"] = curve.predict(df["scada_mean_ws"])
    df["residual"] = df["y"] - df["curve_est"]
    df["residual_ratio"] = df["residual"] / capacity

    suspect = df[(df["residual_ratio"] < -RESIDUAL_THRESHOLD_RATIO) & (df["scada_mean_ws"] >= HIGH_WIND_THRESHOLD)]
    print(f"\n=== group{group_id} (capacity={capacity}, n_total={len(df)}) ===")
    print(f"의심 구간(실측풍속>={HIGH_WIND_THRESHOLD}m/s, 커브 대비 {RESIDUAL_THRESHOLD_RATIO*100:.0f}%p 이상 저발전): "
          f"{len(suspect)}행 ({len(suspect)/len(df)*100:.2f}%)")

    if len(suspect) == 0:
        return

    print(f"  실측풍속 평균: {suspect['scada_mean_ws'].mean():.2f} m/s, "
          f"y 평균: {suspect['y'].mean():.0f} kWh (커브 예측 평균: {suspect['curve_est'].mean():.0f} kWh)")
    print(f"  y=0인 행: {(suspect['y'] < capacity*0.01).sum()}행 ({(suspect['y'] < capacity*0.01).sum()/len(suspect)*100:.1f}%)")

    # 월별/연도별 분포 -- 특정 기간에 몰려있는지(정비/커틀먼트 정책 등) 확인
    suspect_by_month = suspect.groupby(suspect["forecast_kst_dtm"].dt.to_period("M")).size()
    print("  월별 분포(상위 10개):")
    print(suspect_by_month.sort_values(ascending=False).head(10).to_string())

    # 연속 구간(몇 시간 이상 이어지는지) 확인 -- 길게 이어지면 정비/고장, 흩어져 있으면 노이즈
    all_idx = df.index.to_numpy()
    suspect_idx = set(suspect.index)
    streaks = []
    cur_streak = 0
    for i in all_idx:
        if i in suspect_idx:
            cur_streak += 1
        else:
            if cur_streak > 0:
                streaks.append(cur_streak)
            cur_streak = 0
    if cur_streak > 0:
        streaks.append(cur_streak)
    if streaks:
        streaks = np.array(streaks)
        print(f"  연속 구간(streak) 개수: {len(streaks)}, 최대 길이: {streaks.max()}시간, "
              f"6시간 이상 streak 개수: {(streaks >= 6).sum()}")


def main():
    groups = [int(a) for a in sys.argv[1:]] or [1, 2, 3]
    for gid in groups:
        run_group(gid)


if __name__ == "__main__":
    main()
