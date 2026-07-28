"""
현재(26번 섹션) 배포된 FICR objective stage-1 + 잔차보정 결합 모델의 holdout
예측을 실제로 만들어(tune_residual_stage_ficr.py의 캐시 재사용) 오차율 구간별
분포(analyze_error_bands)를 본다 — "아직 어디서 8% 벽을 못 넘고 있는지"를
찾아 다음 FICR 개선 후보의 방향을 잡기 위함.

실행: python3 scripts/diagnose_ficr_error_bands.py <group_id>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src.metrics import CAPACITY_KWH, analyze_error_bands
from tune_residual_stage import _fit_lgbm, MIN_REGIME_SAMPLES
from tune_residual_stage_ficr import load_cache, BEST_CONFIG_PATH_TMPL
import json


def final_pred_for_fold(fold, threshold, stage2_params):
    capacity = fold["capacity"]
    y_train, oof_pred = fold["y_train"], fold["oof_pred"]
    train_frac = y_train / capacity
    regime_mask_train = train_frac >= threshold
    stage1_holdout_pred = fold["stage1_holdout_pred"]
    if regime_mask_train.sum() < MIN_REGIME_SAMPLES:
        return stage1_holdout_pred
    residual_train = y_train - oof_pred
    X_stage2 = fold["X_train"].loc[regime_mask_train].copy()
    X_stage2["stage1_pred"] = oof_pred[regime_mask_train]
    y_stage2 = residual_train[regime_mask_train]
    stage2_model = _fit_lgbm(X_stage2, y_stage2, stage2_params)

    holdout_frac = stage1_holdout_pred / capacity
    regime_mask_holdout = holdout_frac >= threshold
    final_pred = stage1_holdout_pred.copy()
    if regime_mask_holdout.sum() > 0:
        X_holdout_stage2 = fold["X_holdout"].loc[regime_mask_holdout].copy()
        X_holdout_stage2["stage1_pred"] = stage1_holdout_pred[regime_mask_holdout]
        correction = stage2_model.predict(X_holdout_stage2)
        final_pred[regime_mask_holdout] = stage1_holdout_pred[regime_mask_holdout] + correction
    return np.clip(final_pred, 0, capacity)


def main(group_id: int):
    config_path = Path(BEST_CONFIG_PATH_TMPL.format(gid=group_id))
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    threshold, stage2_params = config["threshold"], config["stage2_params"]

    cache = load_cache(group_id)
    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]

    all_actual, all_forecast = [], []
    for year, fold in cache["folds"].items():
        final_pred = final_pred_for_fold(fold, threshold, stage2_params)
        all_actual.append(fold["y_holdout"])
        all_forecast.append(final_pred)
        band = analyze_error_bands(fold["y_holdout"], final_pred, capacity)
        print(f"\n=== group{group_id} holdout={year} ===")
        print(f"  overall: le6%={band['overall']['pct_le6']*100:.1f}% "
              f"6-8%={band['overall']['pct_6to8']*100:.1f}% "
              f"over8%={band['overall']['pct_over8']*100:.1f}% "
              f"mean_err={band['overall']['mean_error_rate']*100:.2f}%")

    actual = np.concatenate(all_actual)
    forecast = np.concatenate(all_forecast)
    band = analyze_error_bands(actual, forecast, capacity)
    print(f"\n=== group{group_id} 전체 폴드 합산 ===")
    print(f"  overall: le6%={band['overall']['pct_le6']*100:.1f}% "
          f"6-8%={band['overall']['pct_6to8']*100:.1f}% "
          f"over8%={band['overall']['pct_over8']*100:.1f}% "
          f"mean_err={band['overall']['mean_error_rate']*100:.2f}%")
    print("\n  capacity_ratio(실제발전량/설비용량) 구간별:")
    print(band["by_capacity_band"].to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]))
