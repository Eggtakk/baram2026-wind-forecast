"""
원본 데이터 로딩 함수 모음.

주의: `scada_*_train.csv` 와 `train_labels.csv`는 **학습 기간에만** 제공되며,
평가(test) 기간에는 존재하지 않는다 (data_description.md 5절, 12절 참고).
따라서 SCADA 실측치를 모델의 입력 feature로 그대로 사용하면 추론 시점에
값이 없어 파이프라인이 깨진다. SCADA는 (a) 라벨/파이프라인 검증(sanity check),
(b) 학습 데이터 증강을 위한 보조 타깃 정도로만 사용하고, 실제 추론에 쓰는
feature는 반드시 `ldaps_*`/`gfs_*` (train/test 모두 존재)에서 만들어야 한다.
"""
from pathlib import Path

import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "open"
CONFIG_DIR = ROOT_DIR / "configs"


def load_group_config(group_id: int, config_dir: Path = CONFIG_DIR) -> dict:
    """configs/group{group_id}.yaml 로드. (scripts/generate_group_configs.py 로 생성됨)"""
    path = Path(config_dir) / f"group{group_id}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_labels(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """train_labels.csv 전체 로드 (kst_dtm, kpx_group_1/2/3). 학습 기간만 존재."""
    path = Path(data_dir) / "train" / "train_labels.csv"
    return pd.read_csv(path, parse_dates=["kst_dtm"])


def load_weather(
    source: str,
    split: str = "train",
    grids: list[int] | None = None,
    usecols: list[str] | None = None,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """LDAPS 또는 GFS 기상예보 데이터 로드.

    Parameters
    ----------
    source: "ldaps" | "gfs"
    split: "train" | "test"
    grids: 지정하면 해당 grid_id만 필터링 (그룹 config의 ldaps_grids/gfs_grids 사용 권장)
    usecols: 특정 컬럼만 읽고 싶을 때 (메모리 절약, 파일이 크므로 필요한 컬럼만 읽는 걸 권장)
    """
    assert source in ("ldaps", "gfs")
    assert split in ("train", "test")
    path = Path(data_dir) / split / f"{source}_{split}.csv"
    parse_dates = ["forecast_kst_dtm", "data_available_kst_dtm"]
    if usecols is not None:
        parse_dates = [c for c in parse_dates if c in usecols]
    df = pd.read_csv(path, usecols=usecols, parse_dates=parse_dates)
    if grids is not None:
        df = df[df["grid_id"].isin(grids)].reset_index(drop=True)
    return df


def load_scada(config: dict, split: str = "train", data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """그룹 config에 지정된 제조사의 SCADA 데이터를 로드하고, 해당 그룹 터빈 컬럼만 남긴다.

    학습 기간에만 존재 (split='train'만 지원). 10분 단위 데이터.
    """
    if split != "train":
        raise ValueError("SCADA data is only available for the train split.")
    path = Path(data_dir) / "train" / config["scada_file"]
    df = pd.read_csv(path, parse_dates=["kst_dtm"])

    keep_cols = ["kst_dtm"]
    for t in config["turbines"]:
        keep_cols += [c for c in df.columns if c.startswith(f"{t}_")]
    return df[keep_cols]


def load_sample_submission(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = Path(data_dir) / "sample_submission.csv"
    return pd.read_csv(path, parse_dates=["forecast_kst_dtm"])
