"""
3그룹 pooled 학습 후보 — 그룹별 개별 모델 대신 하나의 LightGBM 모델을
group1/2/3 데이터를 다 합쳐서 학습하고, group_id를 categorical feature로
줘서 그룹별 차이는 모델이 스스로 분리하게 한다.

배경: experiments/baseline_lgbm/rated_output_investigation.md. 지금까지
시도한 feature 튜닝/모델 계열 교체/커틀먼트 임계값 재탐색은 전부 LOYO
노이즈 밴드(group1 std=0.0148, group2 std=0.0243, group3 std=0.0098)를
넘지 못했다(11/13번 섹션). group3는 세 그룹 중 라벨이 가장 적고(2023~2024
2개년, ~14000~17500행 vs group1/2의 ~26000행) 점수도 가장 낮다(baseline
0.5720) — "같은 데이터를 더 잘 짜는" 방향은 수확체감에 도달한 것으로
보이니, group1/2의 더 많은 데이터로 group3를 보강할 수 있는지 확인하는
더 근본적인 방향(pooled 학습)을 시도한다.

설계:
- 타깃 정규화: 그룹별 설비용량이 다르므로(21,600 / 21,600 / 21,000 kWh)
  raw kWh 대신 y_frac = y / capacity_kwh(대략 0~1의 설비이용률)를 예측하게
  해서 세 그룹이 같은 스케일을 공유하게 한다. 예측 후 그룹별 capacity를
  다시 곱해 kWh로 되돌린 뒤 공식 지표(validate_single_group)로 평가.
- group_id를 categorical feature로 추가해 그룹별 고유 패턴(터빈/지형 차이)은
  모델이 분리해서 학습할 여지를 준다.
- 레시피는 physics-only(build_physics_features)로 세 그룹 전부 통일한다
  (단순화 — group2/3의 프로덕션 레시피는 saturation+파워커브가 포함된 "full"
  이라 group2/3 baseline과는 완전히 공정한 비교가 아님, 이 스크립트 하단
  docstring/로그에 명시. group1 baseline은 원래도 physics-only라 공정한
  비교). saturation/파워커브를 pooled에 넣으려면 그룹별로 파워커브를 따로
  fit해야 해서 복잡도가 커지므로, 이번엔 "pooling 자체가 도움이 되는가"만
  먼저 확인하고, 도움이 되면 다음 단계로 full 레시피 pooled 버전을 시도한다.
- 하이퍼파라미터는 튜닝 없이 pooled 데이터 크기(그룹당 데이터의 약 3배)에
  맞춰 적당히 키운 값을 씀(n_estimators=600, num_leaves=63) — 미세튜닝
  전이라는 한계가 있음, 결과가 유의미하면 그때 Optuna 등으로 재탐색.

폴드는 scripts/validate_loyo.py와 동일한 leave-one-year-out(2022/2023/2024)
을 쓰되, 이번엔 그룹별로 따로 도는 게 아니라 **한 번의 학습으로 세 그룹을
동시에** 처리한다. group3는 2022년 라벨이 없으므로 holdout_year=2022일 때는
자동으로 평가 대상에서 빠진다(학습에는 계속 아무 영향 없음 — 애초에
2022년 데이터가 없을 뿐).

실행: python3 scripts/validate_loyo_pooled.py [holdout_year ...]
  (인자 없으면 2022 2023 2024 전부)
결과: experiments/baseline_lgbm/loyo_pooled_results.json
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data_cleaning import remove_curtailment
from src.features import (
    add_default_wind_features,
    add_lag_rolling_features,
    add_physics_features,
    add_time_features,
    build_baseline_features,
)
from src.metrics import CAPACITY_KWH, validate_single_group
from src.power_curve import apply_power_curve_models, fit_power_curve_models
from src.preprocess import build_group_dataset
from src.validation import time_based_split_leave_year_out

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "baseline_lgbm"
BASELINE_PATH = OUT_DIR / "loyo_validation_results.json"
RESULTS_PATH = OUT_DIR / "loyo_pooled_results.json"

VALID_YEARS = {1: [2022, 2023, 2024], 2: [2022, 2023, 2024], 3: [2023, 2024]}
ALL_HOLDOUT_YEARS = [2022, 2023, 2024]
GROUP_IDS = [1, 2, 3]

# 튜닝 없이 pooled 데이터 크기(그룹당의 약 3배)에 맞춰 적당히 키운 기본값.
POOLED_PARAMS = dict(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=30)

NON_FEATURE_COLS_POOLED = {"forecast_kst_dtm", "ldaps_data_available_kst_dtm", "y", "y_frac"}


def build_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """scripts/validate_loyo.py와 동일한 physics-only 레시피 (세 그룹 전부 이걸로 통일)."""
    df = add_default_wind_features(df)
    df = add_physics_features(df)
    df = add_time_features(df)
    speed_cols = [c for c in df.columns if c.endswith("_speed")]
    df = add_lag_rolling_features(df, cols=speed_cols, lags=[1, 2, 3], windows=[3, 6, 24])
    return df


def get_pooled_feature_cols(df: pd.DataFrame) -> list[str]:
    """get_feature_cols()와 달리 group_id를 feature로 포함시킨다(pooled 모델의 핵심)."""
    scada_cols = [c for c in df.columns if c.startswith("scada_")]
    return [c for c in df.columns if c not in NON_FEATURE_COLS_POOLED and c not in scada_cols]


def _build_group_features(cleaned_train, holdout_raw, capacity: float, recipe: str):
    """레시피별 feature 생성 (+ full 레시피는 그룹별 파워커브를 train에만 fit해서 적용).

    physics: scripts/validate_loyo.py와 동일한 physics-only.
    full: build_baseline_features(물리+saturation) + 그룹별 isotonic 파워커브
        (src/power_curve.py — train에서만 fit, holdout엔 적용만, 프로덕션과
        동일한 누수 방지 원칙).
    """
    has_holdout = len(holdout_raw) > 0
    if recipe == "physics":
        train_feat = build_physics_features(cleaned_train)
        holdout_feat = build_physics_features(holdout_raw) if has_holdout else holdout_raw
    elif recipe == "full":
        train_feat = build_baseline_features(cleaned_train)
        holdout_feat = build_baseline_features(holdout_raw) if has_holdout else holdout_raw
        curve_models = fit_power_curve_models(train_feat, capacity=capacity)
        train_feat = apply_power_curve_models(train_feat, curve_models)
        if has_holdout:
            holdout_feat = apply_power_curve_models(holdout_feat, curve_models)
    else:
        raise ValueError(f"unknown recipe: {recipe}")
    return train_feat, holdout_feat


def run_pooled_fold(holdout_year: int, recipe: str = "physics", params: dict | None = None) -> list[dict]:
    """pooled(3그룹 동시) 학습 1개 폴드.

    recipe: "physics"(기본, 세 그룹 동일 단순 레시피) 또는 "full"(saturation+
        그룹별 파워커브 — 프로덕션 group2/3와 공정 비교하려면 이걸 써야 함).
    params: LightGBM 하이퍼파라미터 override(없으면 POOLED_PARAMS 기본값 —
        튜닝 없는 값). scripts/tune_pooled_optuna.py에서 Optuna로 찾은 값을
        넘겨 재사용.
    """
    t0 = time.time()
    params = params or POOLED_PARAMS
    train_frames = []
    holdout_by_group = {}  # group_id -> (holdout_feat, capacity)

    for gid in GROUP_IDS:
        capacity = CAPACITY_KWH[f"kpx_group_{gid}"]
        df = build_group_dataset(gid, split="train").dropna(subset=["y"]).reset_index(drop=True)
        train_raw, holdout_raw = time_based_split_leave_year_out(
            df, holdout_year=holdout_year, valid_years=VALID_YEARS[gid]
        )
        cleaned_train, n_removed = remove_curtailment(train_raw, capacity=capacity)

        train_feat, holdout_feat = _build_group_features(cleaned_train, holdout_raw, capacity, recipe)
        train_feat["group_id"] = gid
        train_feat["y_frac"] = train_feat["y"] / capacity
        train_frames.append(train_feat)

        if len(holdout_raw) > 0:
            holdout_feat["group_id"] = gid
            holdout_by_group[gid] = (holdout_feat, capacity, n_removed)

    pooled_train = pd.concat(train_frames, ignore_index=True)
    pooled_train["group_id"] = pd.Categorical(pooled_train["group_id"], categories=GROUP_IDS)

    feature_cols = get_pooled_feature_cols(pooled_train)
    model = lgb.LGBMRegressor(**params, random_state=42, verbosity=-1)
    model.fit(
        pooled_train[feature_cols],
        pooled_train["y_frac"],
        categorical_feature=["group_id"],
    )

    fold_results = []
    for gid, (holdout_feat, capacity, n_removed) in holdout_by_group.items():
        holdout_feat = holdout_feat.copy()
        holdout_feat["group_id"] = pd.Categorical(holdout_feat["group_id"], categories=GROUP_IDS)
        pred_frac = model.predict(holdout_feat[feature_cols]).clip(min=0)
        pred_y = pred_frac * capacity

        result = validate_single_group(holdout_feat["y"].to_numpy(), pred_y, group_id=gid)
        score = 0.5 * result["one_minus_nmae"] + 0.5 * result["ficr"]
        fold_results.append({
            "group_id": gid,
            "holdout_year": holdout_year,
            "model": f"pooled_lightgbm_{recipe}",
            "score": score,
            "nmae": result["nmae"],
            "one_minus_nmae": result["one_minus_nmae"],
            "ficr": result["ficr"],
            "n_train_pooled": len(pooled_train),
            "n_holdout": len(holdout_feat),
            "n_eval": result["n_eval"],
            "n_curtailment_removed": n_removed,
        })
        print(f"  group{gid} holdout={holdout_year}: score={score:.4f} "
              f"(1-NMAE={result['one_minus_nmae']:.4f}, FICR={result['ficr']:.4f}) "
              f"n_holdout={len(holdout_feat)}")

    print(f"  [pooled recipe={recipe} n_train={len(pooled_train)}, {len(feature_cols)} features, "
          f"holdout_year={holdout_year}] {time.time()-t0:.1f}s")
    return fold_results


def load_baseline_stats():
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    stats = {}
    for gid in GROUP_IDS:
        scores = np.array([r["score"] for r in baseline if r["group_id"] == gid])
        if len(scores) == 0:
            continue
        std = float(scores.std(ddof=1)) if len(scores) > 1 else 0.0
        stats[gid] = (float(scores.mean()), std, len(scores))
    return stats


def main():
    years = [int(a) for a in sys.argv[1:]] or ALL_HOLDOUT_YEARS

    existing = []
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        existing = [r for r in existing if r["holdout_year"] not in years]

    all_results = existing
    for year in years:
        print(f"\n=== pooled fold holdout_year={year} ===")
        all_results.extend(run_pooled_fold(year))

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장(누적): {RESULTS_PATH}")

    baseline_stats = load_baseline_stats()
    print("\n=== pooled vs baseline(그룹별 개별 모델) ===")
    for gid in GROUP_IDS:
        scores = np.array([r["score"] for r in all_results if r["group_id"] == gid])
        if len(scores) == 0 or gid not in baseline_stats:
            continue
        pooled_mean = float(scores.mean())
        pooled_std = float(scores.std(ddof=1)) if len(scores) > 1 else 0.0
        base_mean, base_std, n_folds = baseline_stats[gid]
        delta = pooled_mean - base_mean
        verdict = "신뢰 가능(노이즈 초과)" if abs(delta) > base_std else "노이즈 수준(불확실)"
        print(f"  group{gid}: baseline={base_mean:.4f}(std={base_std:.4f}, n={n_folds}) -> "
              f"pooled={pooled_mean:.4f}(std={pooled_std:.4f}, n={len(scores)})  "
              f"delta={delta:+.4f}  [{verdict}]")


if __name__ == "__main__":
    main()
