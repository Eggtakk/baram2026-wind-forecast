"""
30번 섹션 후속 재검증 — group3 quantile+잔차보정이 LOYO(2-fold)를 통과하고도
실 제출에서 크게 뒤집힌 사례(31번 섹션) 이후, "LOYO fold 수가 적으면 특정
연도 하나에 좌우된 델타를 과신할 위험이 있다"는 교훈을 group1에도 적용한다.

group1은 이미 quantile alpha=0.60(보수적 선택)으로 배포됐지만, 30번 섹션
자체에서 "alpha를 올릴수록 fold-std가 같이 커지는 패턴(0.014->0.030)이라
alpha=0.65의 최고 delta가 2024년 폴드 하나에 과도하게 좌우된 결과일 위험"을
이미 경고해뒀었다 — 즉 alpha=0.60 선택 자체도 "그나마 안전해 보이는 값을
사람이 골랐다"는 수준이지, 통계적으로 재검증된 적은 없다.

이 스크립트는 (연도 fold를 늘릴 수는 없으니 — group1은 2022/2023/2024
3개 라벨 연도가 전부) 대신 "같은 연도 fold를 여러 random seed로 반복
학습"해서 폴드별 점수의 seed-variance와 fold-variance를 분리해서 본다.
seed-variance가 크면 "이 delta는 모델 학습 노이즈일 뿐"이라는 뜻이고,
fold-variance(연도별 평균 점수의 차이)가 크면 "이 delta는 특정 연도에
좌우된다"는 뜻 — 둘 다 작아야 진짜 신뢰할 수 있는 alpha다.

45초 bash 타임아웃 제약 때문에 한 번 호출에 alpha 1개 x seed 2~3개 정도만
처리 가능 — 그래서 결과를 (year, seed) 단위 raw score로 저장해두고, 같은
alpha를 다른 seed 목록으로 여러 번 호출해도 기존 raw score에 새 seed
결과를 병합(overwrite 아님)한 뒤 매번 전체 통계를 재계산한다.

실행 (레포 루트에서, 여러 번 나눠 호출 가능):
  python3 scripts/validate_quantile_seed_repeat.py <group_id> --alphas=a1,a2 --seeds=s1,s2
기본값: alphas=0.55,0.58,0.60,0.62,0.65  seeds=42,7,123,2024,99
결과 누적: experiments/baseline_lgbm/loyo_quantile_seed_repeat_results.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from validate_loyo import RECIPE_CHOICE, VALID_YEARS, load_params, run_group
from validate_loyo_candidates import get_baseline_stats

RESULTS_PATH = Path(__file__).resolve().parents[1] / "experiments" / "baseline_lgbm" / "loyo_quantile_seed_repeat_results.json"

DEFAULT_ALPHAS = [0.55, 0.58, 0.60, 0.62, 0.65]
DEFAULT_SEEDS = [42, 7, 123, 2024, 99]


def run_alpha_seed(group_id: int, alpha: float, seed: int):
    recipe = RECIPE_CHOICE[group_id]
    params = dict(load_params(group_id, recipe))
    params["objective"] = "quantile"
    params["alpha"] = alpha
    return run_group(group_id, model_params=params, seed=seed)


def summarize(group_id: int, alpha: float, raw: dict, base_mean: float, base_std: float):
    """raw: {year_str: {seed_str: score}} -> 통계 dict."""
    all_scores = []
    year_means = []
    seed_stds = []
    per_year_str = {}
    for year, seed_scores in raw.items():
        scores = list(seed_scores.values())
        all_scores.extend(scores)
        year_means.append(float(np.mean(scores)))
        if len(scores) > 1:
            seed_stds.append(float(np.std(scores, ddof=1)))
        per_year_str[year] = float(np.mean(scores))

    all_scores = np.array(all_scores)
    year_means_arr = np.array(year_means)
    overall_mean = float(all_scores.mean())
    overall_std = float(all_scores.std(ddof=1)) if len(all_scores) > 1 else 0.0
    fold_std = float(year_means_arr.std(ddof=1)) if len(year_means_arr) > 1 else 0.0
    seed_std = float(np.mean(seed_stds)) if seed_stds else 0.0
    delta = overall_mean - base_mean
    verdict = "신뢰 가능(노이즈 초과)" if abs(delta) > base_std else "노이즈 수준"
    return {
        "overall_mean": overall_mean, "delta": delta, "overall_std": overall_std,
        "fold_std": fold_std, "seed_std": seed_std, "verdict": verdict,
        "n_samples": int(len(all_scores)), "year_means": per_year_str,
        "n_seeds_per_year": {y: len(s) for y, s in raw.items()},
    }


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    group_id = int(args[0])
    alphas = DEFAULT_ALPHAS
    seeds = DEFAULT_SEEDS
    for a in args[1:]:
        if a.startswith("--alphas="):
            alphas = [float(x) for x in a.split("=", 1)[1].split(",")]
        elif a.startswith("--seeds="):
            seeds = [int(x) for x in a.split("=", 1)[1].split(",")]

    existing = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing.setdefault(str(group_id), {})

    baseline_stats = get_baseline_stats()
    base_mean, base_std, _ = baseline_stats[group_id]
    n_years = len(VALID_YEARS[group_id])
    print(f"\n=== group{group_id} L2 baseline mean={base_mean:.4f} std={base_std:.4f} "
          f"(연도 fold={n_years}, 이번 호출 seed={seeds}) ===")

    for alpha in alphas:
        alpha_key = str(alpha)
        entry = existing[str(group_id)].setdefault(alpha_key, {"raw": {}})
        raw = entry.setdefault("raw", {})

        new_seeds = [s for s in seeds if str(s) not in {
            s2 for year_scores in raw.values() for s2 in year_scores.keys()
        }] if raw else seeds
        # 이미 모든 연도에 대해 이 seed가 있으면 스킵(중복 학습 방지). 연도별로 다를 수 있으니
        # 안전하게 "어느 연도든 하나라도 이 seed가 없으면" 다시 돌린다.
        seeds_to_run = []
        for s in seeds:
            has_all_years = all(str(s) in raw.get(str(y), {}) for y in VALID_YEARS[group_id])
            if not has_all_years:
                seeds_to_run.append(s)

        for seed in seeds_to_run:
            fold_results = run_alpha_seed(group_id, alpha, seed)
            for r in fold_results:
                year_key = str(r["holdout_year"])
                raw.setdefault(year_key, {})[str(seed)] = r["score"]

        stats = summarize(group_id, alpha, raw, base_mean, base_std)
        entry.update(stats)
        entry["raw"] = raw

        print(f"\n  alpha={alpha}: overall_mean={stats['overall_mean']:.4f}(delta={stats['delta']:+.4f}, "
              f"n={stats['n_samples']}) overall_std={stats['overall_std']:.4f} "
              f"fold_std(연도간)={stats['fold_std']:.4f} seed_std(같은연도내)={stats['seed_std']:.4f} "
              f"[{stats['verdict']}]")
        print(f"    연도별 평균(누적 seed 수={stats['n_seeds_per_year']}): " +
              ", ".join(f"{y}={m:.4f}" for y, m in stats["year_means"].items()))

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장(누적): {RESULTS_PATH}")

    print(f"\n=== group{group_id} alpha별 요약 (seed_std가 fold_std보다 훨씬 작아야 '진짜' 신호) ===")
    for a, r in sorted(existing[str(group_id)].items(), key=lambda kv: float(kv[0])):
        if "delta" not in r:
            continue
        print(f"  alpha={a}: delta={r['delta']:+.4f} fold_std={r['fold_std']:.4f} "
              f"seed_std={r['seed_std']:.4f} n={r['n_samples']} [{r['verdict']}]")


if __name__ == "__main__":
    main()
