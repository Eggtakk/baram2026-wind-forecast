"""
38번 섹션 후속 — 진짜 독립적으로 재학습된 fold 수를 늘리기 위한
leave-one-quarter-out(LOQO) 검증. 38번의 월별-블록 확장은 "같은 3(2)개
모델"을 평가만 잘게 쪼갠 것이라 새로운 일반화 증거가 아니었다는 한계가
있었다 — 이번엔 실제로 분기(quarter) 단위로 holdout을 바꿔가며 매번
새로 학습한다. group1/2는 2022~2024 3년 x 4분기 = 12-fold, group3는
2023~2024 2년 x 4분기 = 8-fold.

주의(trade-off, 문서화됨): 각 fold의 train은 "이전 연도 전체"가 아니라
"holdout 분기를 제외한 나머지 모든 분기"(여러 연도가 섞인 형태)라
실제 프로덕션 학습 방식과 다르다. lag/rolling feature는 분기 경계에서
비연속 구간이 이어붙는 지점(seam)이 생겨 그 부분만 약간 부정확할 수
있음(연 단위 LOYO도 연도 경계에서 동일한 근사를 이미 쓰고 있음).

45초 bash 제약으로 한 번에 전체 12(8)-fold를 다 못 돌릴 수 있어, 그룹별로
처리할 (year, quarter) 목록을 CLI 인자로 받아 여러 번 나눠 호출 가능:
  python3 scripts/validate_loyo_quarterly.py 1 2022-1 2022-2 2022-3 2022-4
  python3 scripts/validate_loyo_quarterly.py 1 2023-1 2023-2 2023-3 2023-4
  python3 scripts/validate_loyo_quarterly.py 1 2024-1 2024-2 2024-3 2024-4
인자 없이 그룹 번호만 주면 해당 그룹의 전체 분기를 한 번에 시도한다
(시간 안에 끝나면 전체, 타임아웃되면 일부만 저장되고 나머지는 재실행 시
이어서 채워짐 — 이미 처리된 (group,quarter)는 건너뜀).
결과: experiments/baseline_lgbm/loyo_quarterly_results.json
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
    build_baseline_features,
    get_feature_cols,
)
from src.metrics import CAPACITY_KWH, validate_single_group
from src.power_curve import apply_power_curve_models, fit_power_curve_models
from src.preprocess import build_group_dataset
from src.validation import time_based_split_leave_quarter_out

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "baseline_lgbm"
RESULTS_PATH = OUT_DIR / "loyo_quarterly_results.json"

RECIPE_CHOICE = {1: "physics", 2: "full", 3: "full"}
PARAMS_SOURCE = {1: "yearly", 2: "yearly", 3: "optuna"}
VALID_YEARS = {1: [2022, 2023, 2024], 2: [2022, 2023, 2024], 3: [2023, 2024]}


def all_quarters(group_id: int) -> list[tuple[int, int]]:
    return [(y, q) for y in VALID_YEARS[group_id] for q in [1, 2, 3, 4]]


def build_physics_features(df):
    df = add_default_wind_features(df)
    df = add_physics_features(df)
    df = add_time_features(df)
    speed_cols = [c for c in df.columns if c.endswith("_speed")]
    df = add_lag_rolling_features(df, cols=speed_cols, lags=[1, 2, 3], windows=[3, 6, 24])
    return df


def load_params(group_id: int, recipe: str) -> dict:
    source = PARAMS_SOURCE[group_id]
    path = OUT_DIR / f"group{group_id}_{source}_best_params_{recipe}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_fold(group_id: int, df, holdout_quarter: tuple[int, int], seed: int = 42) -> dict:
    recipe = RECIPE_CHOICE[group_id]
    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]

    train_raw, holdout_raw = time_based_split_leave_quarter_out(
        df, holdout_quarter=holdout_quarter, valid_quarters=all_quarters(group_id)
    )
    cleaned_train, n_removed = remove_curtailment(train_raw, capacity=capacity)

    if recipe == "physics":
        train_feat = build_physics_features(cleaned_train)
        holdout_feat = build_physics_features(holdout_raw)
    else:
        train_feat = build_baseline_features(cleaned_train)
        holdout_feat = build_baseline_features(holdout_raw)
        curve_models = fit_power_curve_models(train_feat, capacity=capacity)
        train_feat = apply_power_curve_models(train_feat, curve_models)
        holdout_feat = apply_power_curve_models(holdout_feat, curve_models)

    feature_cols = get_feature_cols(train_feat)
    params = load_params(group_id, recipe)
    bagging_freq = 1 if params.get("bagging_fraction", 1.0) < 1.0 else 0
    model = lgb.LGBMRegressor(**params, random_state=seed, bagging_freq=bagging_freq, verbosity=-1)
    model.fit(train_feat[feature_cols], train_feat["y"])

    pred = model.predict(holdout_feat[feature_cols]).clip(min=0)
    result = validate_single_group(holdout_feat["y"].to_numpy(), pred, group_id=group_id)
    score = 0.5 * result["one_minus_nmae"] + 0.5 * result["ficr"]

    return {
        "group_id": group_id,
        "holdout_year": holdout_quarter[0],
        "holdout_quarter": holdout_quarter[1],
        "recipe": recipe,
        "score": score,
        "nmae": result["nmae"],
        "one_minus_nmae": result["one_minus_nmae"],
        "ficr": result["ficr"],
        "n_train": len(train_feat),
        "n_holdout": len(holdout_feat),
        "n_eval": result["n_eval"],
        "n_curtailment_removed": n_removed,
    }


def parse_quarter_tokens(tokens: list[str]) -> list[tuple[int, int]]:
    out = []
    for t in tokens:
        y, q = t.split("-")
        out.append((int(y), int(q)))
    return out


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit("사용법: validate_loyo_quarterly.py <group_id> [YYYY-Q ...]")
    group_id = int(args[0])
    quarter_tokens = args[1:]
    quarters = parse_quarter_tokens(quarter_tokens) if quarter_tokens else all_quarters(group_id)

    all_results = []
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            all_results = json.load(f)

    done = {(r["group_id"], r["holdout_year"], r["holdout_quarter"]) for r in all_results}
    todo = [q for q in quarters if (group_id, q[0], q[1]) not in done]

    if not todo:
        print(f"group{group_id}: 요청된 분기 전부 이미 처리됨 (스킵)")
    else:
        df = build_group_dataset(group_id, split="train").dropna(subset=["y"]).reset_index(drop=True)
        print(f"=== group{group_id} leave-one-quarter-out ({len(todo)}개 분기 처리) ===")
        for hq in todo:
            t0 = time.time()
            r = run_fold(group_id, df, hq)
            all_results.append(r)
            print(f"  holdout={hq[0]}-Q{hq[1]}: score={r['score']:.4f} "
                  f"(1-NMAE={r['one_minus_nmae']:.4f}, FICR={r['ficr']:.4f}) "
                  f"n_train={r['n_train']} n_holdout={r['n_holdout']} [{time.time()-t0:.1f}s]")

        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"저장: {RESULTS_PATH}")

    # 현재까지 누적된 결과로 그룹별 요약(전체 분기가 다 모이지 않았어도 진행 상황 출력)
    print("\n=== 누적 요약 ===")
    for gid in sorted({r["group_id"] for r in all_results}):
        scores = np.array([r["score"] for r in all_results if r["group_id"] == gid])
        n = len(scores)
        expected = len(all_quarters(gid))
        std = scores.std(ddof=1) if n > 1 else 0.0
        sem = std / np.sqrt(n) if n > 0 else float("nan")
        print(f"  group{gid}: n={n}/{expected} mean={scores.mean():.4f} std={std:.4f} SEM={sem:.4f}")


if __name__ == "__main__":
    main()
