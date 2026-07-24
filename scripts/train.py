"""
베이스라인 학습 스크립트 (LightGBM, 그룹1/2/3 공통 파이프라인).

대회 산출물 제출 규정상 학습 코드와 추론 코드는 분리해야 하므로, 이 파일은
학습만 담당한다. 추론은 scripts/inference.py 참고.

각 그룹의 전체 학습 데이터(y가 있는 전체 기간)로 최종 모델을 학습해 저장한다.
holdout으로 성능을 먼저 보고 싶으면 scripts/validate_baseline.py를 먼저 실행할 것.

실행: (레포 루트에서) python3 scripts/train.py
출력: experiments/baseline_lgbm/group{n}_model.pkl, feature_cols.json, train_meta.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import lightgbm as lgb

from src.features import build_baseline_features, get_feature_cols
from src.preprocess import build_group_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "baseline_lgbm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_LGBM_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbosity=-1,
)


def load_params(group_id: int) -> dict:
    """scripts/tune_baseline.py가 만든 group{n}_best_params.json이 있으면
    그 값으로 기본 파라미터를 덮어쓴다 (없으면 DEFAULT_LGBM_PARAMS 그대로)."""
    params = dict(DEFAULT_LGBM_PARAMS)
    best_path = OUT_DIR / f"group{group_id}_best_params.json"
    if best_path.exists():
        with open(best_path, "r", encoding="utf-8") as f:
            tuned = json.load(f)
        params.update(tuned)
        if tuned.get("bagging_fraction", 1.0) < 1.0:
            params["bagging_freq"] = 1
        print(f"[group{group_id}] {best_path.name} 적용: {tuned}")
    return params


def train_group(group_id: int) -> dict:
    df = build_group_dataset(group_id, split="train")
    df = build_baseline_features(df)
    df = df.dropna(subset=["y"]).reset_index(drop=True)

    feature_cols = get_feature_cols(df)
    params = load_params(group_id)

    model = lgb.LGBMRegressor(**params)
    model.fit(df[feature_cols], df["y"])

    model_path = OUT_DIR / f"group{group_id}_model.pkl"
    joblib.dump(model, model_path)

    meta = {
        "group_id": group_id,
        "feature_cols": feature_cols,
        "n_train_rows": len(df),
        "train_range": [str(df["forecast_kst_dtm"].min()), str(df["forecast_kst_dtm"].max())],
        "params": params,
    }
    with open(OUT_DIR / f"group{group_id}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[group{group_id}] trained on {len(df)} rows, {len(feature_cols)} features -> {model_path}")
    return meta


def main():
    for gid in [1, 2, 3]:
        train_group(gid)
    print(f"\nAll models saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
