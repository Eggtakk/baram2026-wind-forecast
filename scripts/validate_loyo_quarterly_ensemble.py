"""
41번 섹션 후속 — "기각된 후보 재검토"가 아니라 **처음부터 분기 LOQO로
검증하는 새 아이디어**: group1 stage-1을 하나의 objective가 아니라
L2 + quantile(alpha=0.60) + FICR-shaped(ficr_weight=0.003, lambda_l2=1.0,
T=0.01, group1의 과거 FICR 배포 파라미터 재사용) **세 objective의 평균
앙상블**로 바꿔본다.

동기: 40번 섹션에서 지금 배포된 group1(quantile 단독+잔차보정)이 분기
LOQO 기준으로 delta/std=0.42배에 그쳐(4/12 분기에서는 오히려 baseline
보다 나쁨) 신뢰도가 약하다는 게 드러났다. 반면 group2(FICR 단독+잔차
보정)는 1.81배로 훨씬 안정적이었다. 과거 세션에서 시도한 앙상블(2번
섹션: 같은 L2 objective+다른 seed만 다름 → 완전 상관이라 무의미,
9번 섹션: 모델 계열 XGBoost/CatBoost 블렌딩 → 실 제출 실패)과 달리,
이번엔 **같은 LightGBM·같은 feature, objective 함수만 다른 3개 모델**을
섞는다 — objective가 다르면 예측이 서로 다른 방식으로 틀리기 쉬워서
(quantile은 상방 편향, FICR은 6%/8% 경계 근처를 정교하게 맞춤, L2는
평균제곱오차 최소화) 앙상블 다양성 효과를 기대할 수 있다는 가설.

이번엔 stage-1만 앙상블하고(잔차보정 결합은 이 결과가 유망할 때 다음
단계로), 분기 LOQO(12-fold, 진짜 재학습)로 baseline(L2 단독, 39번
섹션 결과 재사용)과 짝지어(paired) 비교한다.

실행: python3 scripts/validate_loyo_quarterly_ensemble.py [YYYY-Q ...]
(그룹은 group1 고정 — 이 아이디어의 동기가 group1 전용이므로)
결과: experiments/baseline_lgbm/loyo_quarterly_ensemble_results.json
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import lightgbm as lgb

from src.data_cleaning import remove_curtailment
from src.features import (
    add_default_wind_features,
    add_lag_rolling_features,
    add_physics_features,
    add_time_features,
    get_feature_cols,
)
from src.ficr_objective import make_ficr_objective
from src.metrics import CAPACITY_KWH, validate_single_group
from src.preprocess import build_group_dataset
from src.validation import time_based_split_leave_quarter_out

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "baseline_lgbm"
RESULTS_PATH = OUT_DIR / "loyo_quarterly_ensemble_results.json"

GROUP_ID = 1
RECIPE = "physics"
PARAMS_SOURCE = "yearly"
VALID_YEARS = [2022, 2023, 2024]
QUANTILE_ALPHA = 0.60
FICR_WEIGHT, LAMBDA_L2, T = 0.003, 1.0, 0.01
# 앙상블 가중치 후보: 균등, quantile 편중, FICR 편중, L2 배제(quantile+FICR만)
WEIGHT_GRID = {
    "equal": (1 / 3, 1 / 3, 1 / 3),
    "quantile_heavy": (0.2, 0.6, 0.2),
    "ficr_heavy": (0.2, 0.2, 0.6),
    "no_l2": (0.0, 0.5, 0.5),
}


def all_quarters():
    return [(y, q) for y in VALID_YEARS for q in [1, 2, 3, 4]]


def build_physics_features(df):
    df = add_default_wind_features(df)
    df = add_physics_features(df)
    df = add_time_features(df)
    speed_cols = [c for c in df.columns if c.endswith("_speed")]
    df = add_lag_rolling_features(df, cols=speed_cols, lags=[1, 2, 3], windows=[3, 6, 24])
    return df


def load_base_params():
    path = OUT_DIR / f"group{GROUP_ID}_{PARAMS_SOURCE}_best_params_{RECIPE}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fit_lgbm(X, y, params, seed=42):
    bagging_freq = 1 if params.get("bagging_fraction", 1.0) < 1.0 else 0
    model = lgb.LGBMRegressor(**params, random_state=seed, bagging_freq=bagging_freq, verbosity=-1)
    model.fit(X, y)
    return model


def run_fold(df, holdout_quarter: tuple[int, int]) -> dict:
    capacity = CAPACITY_KWH[f"kpx_group_{GROUP_ID}"]
    train_raw, holdout_raw = time_based_split_leave_quarter_out(
        df, holdout_quarter=holdout_quarter, valid_quarters=all_quarters()
    )
    cleaned_train, n_removed = remove_curtailment(train_raw, capacity=capacity)
    train_feat = build_physics_features(cleaned_train)
    holdout_feat = build_physics_features(holdout_raw)
    feature_cols = get_feature_cols(train_feat)
    base_params = load_base_params()

    # L2 (기본 objective)
    l2_model = _fit_lgbm(train_feat[feature_cols], train_feat["y"], base_params)
    pred_l2 = l2_model.predict(holdout_feat[feature_cols]).clip(min=0, max=capacity)

    # quantile
    q_params = dict(base_params)
    q_params["objective"] = "quantile"
    q_params["alpha"] = QUANTILE_ALPHA
    q_model = _fit_lgbm(train_feat[feature_cols], train_feat["y"], q_params)
    pred_q = q_model.predict(holdout_feat[feature_cols]).clip(min=0, max=capacity)

    # FICR-shaped
    f_params = dict(base_params)
    f_params["objective"] = make_ficr_objective(capacity, ficr_weight=FICR_WEIGHT, lambda_l2=LAMBDA_L2, T=T)
    f_model = _fit_lgbm(train_feat[feature_cols], train_feat["y"], f_params)
    pred_f = f_model.predict(holdout_feat[feature_cols]).clip(min=0, max=capacity)

    y_holdout = holdout_feat["y"].to_numpy()

    def score_of(pred):
        r = validate_single_group(y_holdout, pred, group_id=GROUP_ID)
        return 0.5 * r["one_minus_nmae"] + 0.5 * r["ficr"]

    result = {
        "group_id": GROUP_ID,
        "holdout_year": holdout_quarter[0],
        "holdout_quarter": holdout_quarter[1],
        "n_train": len(train_feat),
        "n_holdout": len(holdout_feat),
        "n_curtailment_removed": n_removed,
        "score_l2": score_of(pred_l2),
        "score_quantile": score_of(pred_q),
        "score_ficr": score_of(pred_f),
    }
    for name, (wl2, wq, wf) in WEIGHT_GRID.items():
        blended = np.clip(wl2 * pred_l2 + wq * pred_q + wf * pred_f, 0, capacity)
        result[f"score_ens_{name}"] = score_of(blended)
    return result


def parse_quarter_tokens(tokens: list[str]) -> list[tuple[int, int]]:
    return [(int(t.split("-")[0]), int(t.split("-")[1])) for t in tokens]


def main():
    tokens = sys.argv[1:]
    quarters = parse_quarter_tokens(tokens) if tokens else all_quarters()

    all_results = []
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            all_results = json.load(f)

    done = {(r["holdout_year"], r["holdout_quarter"]) for r in all_results}
    todo = [q for q in quarters if q not in done]

    if not todo:
        print("요청된 분기 전부 이미 처리됨 (스킵)")
    else:
        df = build_group_dataset(GROUP_ID, split="train").dropna(subset=["y"]).reset_index(drop=True)
        print(f"=== group{GROUP_ID} stage-1 objective 앙상블 분기 LOQO ({len(todo)}개 분기) ===")
        for hq in todo:
            t0 = time.time()
            r = run_fold(df, hq)
            all_results.append(r)
            print(f"  holdout={hq[0]}-Q{hq[1]}: L2={r['score_l2']:.4f} quantile={r['score_quantile']:.4f} "
                  f"FICR={r['score_ficr']:.4f} | equal={r['score_ens_equal']:.4f} "
                  f"q_heavy={r['score_ens_quantile_heavy']:.4f} f_heavy={r['score_ens_ficr_heavy']:.4f} "
                  f"no_l2={r['score_ens_no_l2']:.4f} [{time.time()-t0:.1f}s]")
            with open(RESULTS_PATH, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"저장: {RESULTS_PATH}")

    print("\n=== 누적 요약 ===")
    n = len(all_results)
    for key in ["score_l2", "score_quantile", "score_ficr", "score_ens_equal",
                "score_ens_quantile_heavy", "score_ens_ficr_heavy", "score_ens_no_l2"]:
        vals = np.array([r[key] for r in all_results])
        std = vals.std(ddof=1) if len(vals) > 1 else 0.0
        print(f"  {key}: n={len(vals)}/12 mean={vals.mean():.4f} std={std:.4f}")


if __name__ == "__main__":
    main()
