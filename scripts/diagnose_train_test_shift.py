"""
35번 섹션 후속 — 실 제출에서 반복적으로 관측된 "LOYO 개선이 부호까지
뒤집히는" 현상의 원인을 검증한다: 2025년 test 기간의 예보 입력 분포가
2022~2024년 train 라벨 연도들과 실제로 다른지 직접 비교.

비교 대상: ldaps_hub_speed(허브고도 환산 LDAPS 풍속), ldaps_air_density
(지상기압에서 유도한 공기밀도, "기압" 프록시), 월별(계절) 구성 비율.

train은 연도별로 쪼개서(2022/2023/2024) 따로 보고, test(2025, 라벨 없음)와
비교한다 — group별로 실행(공통 LDAPS/GFS 격자 배정이 group마다 다름).

실행: python3 scripts/diagnose_train_test_shift.py <group_id>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats

from src.features import add_default_wind_features, add_physics_features
from src.preprocess import build_group_dataset


def load_with_physics(group_id: int, split: str) -> pd.DataFrame:
    df = build_group_dataset(group_id, split=split)
    df = add_default_wind_features(df)
    df = add_physics_features(df)
    df["year"] = df["forecast_kst_dtm"].dt.year
    df["month"] = df["forecast_kst_dtm"].dt.month
    return df


def summarize(df: pd.DataFrame, col: str, label: str):
    s = df[col].dropna()
    print(f"    {label}: n={len(s)} mean={s.mean():.3f} std={s.std():.3f} "
          f"p10={s.quantile(0.1):.3f} p50={s.quantile(0.5):.3f} p90={s.quantile(0.9):.3f}")


def main(group_id: int):
    train = load_with_physics(group_id, "train")
    test = load_with_physics(group_id, "test")

    print(f"=== group{group_id} ===")
    print(f"train 연도: {sorted(train['year'].unique())}, test 연도: {sorted(test['year'].unique())}")

    for col in ["ldaps_hub_speed", "ldaps_air_density"]:
        print(f"\n--- {col} ---")
        for y in sorted(train["year"].unique()):
            summarize(train[train["year"] == y], col, f"train {y}")
        summarize(test, col, "test(2025)")

        # KS test: 각 train 연도 vs test
        for y in sorted(train["year"].unique()):
            a = train.loc[train["year"] == y, col].dropna()
            b = test[col].dropna()
            ks_stat, p = stats.ks_2samp(a, b)
            flag = " <- 분포 유의하게 다름(p<0.01)" if p < 0.01 else ""
            print(f"    KS(train{y} vs test): stat={ks_stat:.4f} p={p:.4g}{flag}")

    print(f"\n--- 월별 구성 비율 ---")
    for y in sorted(train["year"].unique()):
        counts = train[train["year"] == y]["month"].value_counts(normalize=True).sort_index()
        print(f"  train {y}: " + ", ".join(f"{m}월={v*100:.1f}%" for m, v in counts.items()))
    counts = test["month"].value_counts(normalize=True).sort_index()
    print(f"  test(2025): " + ", ".join(f"{m}월={v*100:.1f}%" for m, v in counts.items()))

    # 고풍속(고출력) 구간 비율 비교 - 이 세션 내내 문제였던 구간
    print(f"\n--- ldaps_hub_speed >= 12 m/s (고풍속) 비율 ---")
    for y in sorted(train["year"].unique()):
        frac = (train.loc[train["year"] == y, "ldaps_hub_speed"] >= 12).mean()
        print(f"  train {y}: {frac*100:.2f}%")
    frac = (test["ldaps_hub_speed"] >= 12).mean()
    print(f"  test(2025): {frac*100:.2f}%")


if __name__ == "__main__":
    main(int(sys.argv[1]))
