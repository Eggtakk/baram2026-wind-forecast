"""
그룹1/2/3 각각에 대해 preprocess -> features -> 시간순 holdout -> LightGBM
베이스라인 -> 공식 산식(1-NMAE, FICR, Score) 검증까지 한 번에 실행하는 스크립트.

실행: (레포 루트에서) python3 scripts/validate_baseline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import pandas as pd

from src.features import build_baseline_features, get_feature_cols
from src.metrics import validate_single_group
from src.preprocess import build_group_dataset
from src.validation import time_based_split

HOLDOUT_RATIO = 0.2


def run_group(group_id: int) -> dict:
    df = build_group_dataset(group_id, split="train")
    df = build_baseline_features(df)
    df = df.dropna(subset=["y"]).reset_index(drop=True)

    train_df, holdout_df = time_based_split(df, holdout_ratio=HOLDOUT_RATIO)

    feature_cols = get_feature_cols(df)

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1,
    )
    model.fit(train_df[feature_cols], train_df["y"])

    pred = model.predict(holdout_df[feature_cols]).clip(min=0)
    result = validate_single_group(holdout_df["y"].to_numpy(), pred, group_id=group_id)
    result.update(
        {
            "train_range": (train_df["forecast_kst_dtm"].min(), train_df["forecast_kst_dtm"].max()),
            "holdout_range": (holdout_df["forecast_kst_dtm"].min(), holdout_df["forecast_kst_dtm"].max()),
            "n_train": len(train_df),
            "n_features": len(feature_cols),
        }
    )
    return result


def main():
    rows = []
    for gid in [1, 2, 3]:
        r = run_group(gid)
        rows.append(r)
        print(
            f"[group{gid}] train={r['train_range'][0].date()}~{r['train_range'][1].date()} "
            f"({r['n_train']}행) | holdout={r['holdout_range'][0].date()}~{r['holdout_range'][1].date()} "
            f"({r['n_total']}행, 평가대상 {r['n_eval']}행)"
        )
        print(
            f"          NMAE={r['nmae']:.4f}  1-NMAE={r['one_minus_nmae']:.4f}  FICR={r['ficr']:.4f}  "
            f"local_group_score={0.5*r['one_minus_nmae']+0.5*r['ficr']:.4f}"
        )

    print("\n=== 요약 ===")
    summary = pd.DataFrame(rows)[["group_id", "nmae", "one_minus_nmae", "ficr", "n_eval", "n_total"]]
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
