"""
27번 섹션 진단(고출력 구간 과소예측이 threshold를 낮추는 것만으로는 안 잡힘)의
후속 — 정격출력 잔차보정 stage-2를 두 가지 방향으로 확장해 LOYO 검증한다.

1) 2-tier regime: 지금은 threshold 이상 전체를 하나의 stage-2 모델이 담당하는데,
   [t_low, t_mid) / [t_mid, 1.0] 두 구간으로 나눠 각각 별도 stage-2 모델을
   학습 — 60~80%/80~100% 등 구간별로 잔차의 성질이 다를 수 있다는 가설
   (group1: 70~80% bias -0.076, 80~90% -0.128, 90~100% -0.141로 구간마다
   편향 크기가 다름 — 하나의 모델로는 이 이질성을 다 못 담을 수 있음).

2) 고출력 특화 feature: src/features.py에 이미 구현/LOYO 테스트까지 됐지만
   전역 채택은 보류됐던 add_forecast_disagreement_features(예보 소스 간
   불일치)와 add_wind_direction_cyclical_features(풍향 sin/cos)를 stage-1
   전체가 아니라 stage-2(고출력 구간만)에만 국한해서 추가 — 12/13번 섹션에서
   이 feature들이 "약하지만 방향이 일관된" 양성 신호를 보였던 게 정확히
   고출력 구간에서 예보 불확실성이 커지기 때문이라는 가설(4번 섹션 근거)과
   맞다면, 전역이 아니라 고출력 구간에 국한했을 때 신호 대 잡음비가 더
   좋아질 수 있다.

stage-1(FICR objective, 26번 섹션)과 OOF는 이미 tune_residual_stage_ficr.py의
캐시(/tmp/residual_stage_ficr_cache_group{1,2}.pkl)에 있고, 이 두 확장 모두
stage-2에만 영향을 주므로 캐시를 그대로 재사용한다(재계산 불필요) — 단,
disagreement/cyclical feature는 캐시된 X_train/X_holdout에 이미 있는
ldaps_hub_speed/gfs_hub_speed/ldaps_ws10_speed/gfs_ws10_speed/*_dir 컬럼만
사용해 파생하므로 별도 재계산 없이 그 자리에서 추가 가능
(LDAPS 격자 spread는 원시 grid 데이터가 캐시에 없어 이번엔 제외 — 필요하면
fold 재생성 필요, 후속 과제로 남김).

실행: python3 scripts/tune_residual_stage_v2.py <group_id>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import numpy as np
import pandas as pd

from src.features import add_forecast_disagreement_features, add_wind_direction_cyclical_features
from src.metrics import validate_single_group
from tune_residual_stage import _fit_lgbm, MIN_REGIME_SAMPLES
from tune_residual_stage_ficr import load_cache, BEST_CONFIG_PATH_TMPL


def augment_extra_features(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """캐시된 X(이미 stage-1 feature_cols로 구성됨)에 예보 불일치 + 풍향
    sin/cos feature를 추가. 반환: (augmented X, 새로 추가된 컬럼명 리스트)."""
    before = set(X.columns)
    X = add_forecast_disagreement_features(X)
    X = add_wind_direction_cyclical_features(X)
    new_cols = [c for c in X.columns if c not in before]
    return X, new_cols


def eval_fold_v2(fold, thresholds, stage2_params, use_extra: bool = False):
    """thresholds: float(단일 threshold, 기존과 동일) 또는 (t_low, t_mid) 튜플(2-tier).
    stage2_params: dict(단일) 또는 [dict, dict](2-tier, 티어별 다른 파라미터 허용)."""
    capacity = fold["capacity"]
    y_train, oof_pred = fold["y_train"], fold["oof_pred"]
    X_train, X_holdout = fold["X_train"], fold["X_holdout"]
    stage1_holdout_pred = fold["stage1_holdout_pred"]

    if use_extra:
        X_train, extra_cols = augment_extra_features(X_train)
        X_holdout, _ = augment_extra_features(X_holdout)
    else:
        extra_cols = []

    train_frac = y_train / capacity
    holdout_frac = stage1_holdout_pred / capacity

    two_tier = isinstance(thresholds, (tuple, list))
    if not two_tier:
        tiers = [(thresholds, 1.01)]
        params_list = [stage2_params]
    else:
        t_low, t_mid = thresholds
        tiers = [(t_low, t_mid), (t_mid, 1.01)]
        params_list = stage2_params if isinstance(stage2_params, list) else [stage2_params, stage2_params]

    final_pred = stage1_holdout_pred.copy()
    n_regimes = []
    for (lo, hi), params in zip(tiers, params_list):
        regime_mask_train = (train_frac >= lo) & (train_frac < hi)
        n_regime = int(regime_mask_train.sum())
        n_regimes.append(n_regime)
        if n_regime < MIN_REGIME_SAMPLES:
            continue
        residual_train = y_train - oof_pred
        cols = fold["feature_cols"] + extra_cols
        X_stage2 = X_train.loc[regime_mask_train, cols].copy()
        X_stage2["stage1_pred"] = oof_pred[regime_mask_train]
        y_stage2 = residual_train[regime_mask_train]
        stage2_model = _fit_lgbm(X_stage2, y_stage2, params)

        regime_mask_holdout = (holdout_frac >= lo) & (holdout_frac < hi)
        if regime_mask_holdout.sum() > 0:
            X_holdout_stage2 = X_holdout.loc[regime_mask_holdout, cols].copy()
            X_holdout_stage2["stage1_pred"] = stage1_holdout_pred[regime_mask_holdout]
            correction = stage2_model.predict(X_holdout_stage2)
            final_pred[regime_mask_holdout] = stage1_holdout_pred[regime_mask_holdout] + correction

    final_pred = np.clip(final_pred, 0, capacity)
    result = validate_single_group(fold["y_holdout"], final_pred, group_id=fold.get("group_id", 0))
    score = 0.5 * result["one_minus_nmae"] + 0.5 * result["ficr"]
    return score, n_regimes, result["nmae"], result["ficr"]


def run_grid(group_id: int, configs: list[dict]):
    """configs: [{"label":..., "thresholds":..., "stage2_params":..., "use_extra":bool}, ...]"""
    cache = load_cache(group_id)
    folds = cache["folds"]
    ficr_only = np.array([f["baseline_score"] for f in folds.values()])
    print(f"group{group_id} FICR stage-1 단독 baseline mean={ficr_only.mean():.4f}")

    for cfg in configs:
        scores, nmaes, ficrs = [], [], []
        for year, fold in folds.items():
            fw = dict(fold, group_id=group_id)
            s, n_regimes, nmae, ficr = eval_fold_v2(
                fw, cfg["thresholds"], cfg["stage2_params"], use_extra=cfg.get("use_extra", False)
            )
            scores.append(s); nmaes.append(nmae); ficrs.append(ficr)
        scores = np.array(scores)
        print(f"  [{cfg['label']}] mean={scores.mean():.4f} delta_vs_ficr_stage1={scores.mean()-ficr_only.mean():+.4f} "
              f"NMAE={np.mean(nmaes):.4f} FICR={np.mean(ficrs):.4f} fold_scores={scores.round(4).tolist()}")


def load_current_best_config(group_id: int) -> dict:
    """27번 섹션에서 FICR objective 전용으로 재탐색된 설정(있으면)을 우선
    사용 — 현재 실제 배포된 threshold와 반드시 일치시키기 위함."""
    from pathlib import Path as _Path
    ficr_path = _Path(f"experiments/baseline_lgbm/group{group_id}_residual_stage_ficr_best_config.json")
    path = ficr_path if ficr_path.exists() else _Path(BEST_CONFIG_PATH_TMPL.format(gid=group_id))
    print(f"[group{group_id}] baseline config 소스: {path.name}")
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    gid = int(sys.argv[1])
    base_config = load_current_best_config(gid)
    base_thr = base_config["threshold"]
    base_params = base_config["stage2_params"]

    configs = [
        {"label": "baseline(단일 tier, 기존 feature)", "thresholds": base_thr, "stage2_params": base_params},
        {"label": "baseline + extra feature", "thresholds": base_thr, "stage2_params": base_params, "use_extra": True},
    ]
    run_grid(gid, configs)
