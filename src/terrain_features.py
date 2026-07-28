"""
지형(DEM) 기반 방향별 풍상측(upwind) 고도차 feature (외부 공개 데이터, 실험용).

배경: 이 대회 터빈은 전부 태백 고지대 능선(태백가덕산/태백원동) 위에
설치돼 있다(`data/open/info.xlsx` 확인, group1/2=VESTAS V126, group3=UNISON
U136, 전부 hub height 117m). LDAPS/GFS는 격자 평균 지형을 반영한 예보라
실제 능선 지형의 국지 효과(산등성이를 넘으며 가속되는 지형풍, 계곡에서
불어오를 때의 활강풍 등 — group3 EDA에서 이미 확인된 야간 활강풍 패턴
참고)를 완전히 반영하지 못할 수 있다. "지금 부는 바람이 어느 방향에서
오는가"에 따라 그 방향의 실제 지형(계곡에서 올라오는지, 능선을 타고 오는지)
이 다르므로, 풍향별로 "풍상측 지형이 터빈 위치보다 얼마나 낮은가/높은가"를
feature로 주면 트리 모델이 방향별 지형 효과를 학습할 여지가 생긴다는 가설.

이전에 시도했던 정적(static) 외부 데이터(16번 섹션, 터빈 제작사 파워커브)와
질적으로 다르다 — 파워커브 feature는 그룹 내에서 사실상 상수(풍속의 단조
변환)라 트리가 새로 학습할 게 거의 없었지만, 이 feature는 매 예보 시각의
풍향(시간에 따라 계속 바뀜)에 연동되므로 그룹 내에서도 실제로 값이
변한다 — 즉 원본 feature들의 비단조 interaction과 유사한 성격.

데이터 출처: SRTM 30m DEM (NASA/USGS, 2000년 수집, 2015년 전세계
30m 해상도로 공개 — 이 대회 학습 기간(2022~)보다 훨씬 이전), opentopodata.org
공개 API(`srtm30m` 데이터셋)로 조회. 재현성 문서:
`docs/external_data_terrain_dem.md`.

방법: `data/open/info.xlsx`에서 그룹별 터빈 좌표를 파싱해 중심점(centroid)을
구하고, 중심점에서 8방위(45도 간격) x 2거리(1000m, 2500m) 지점의 고도를
조회한다. 각 방위에 대해 "중심점 고도 - 그 방위 프로필 평균 고도"를
`upwind_drop`으로 정의(+ = 그 방향은 터빈보다 낮은 지형/계곡, - = 그
방향은 터빈보다 높은 지형/능선). 결과 lookup table은
`experiments/baseline_lgbm/terrain_upwind_lookup.json`에 저장.

(실험용 — 아직 프로덕션 레시피에는 편입되지 않음. scripts/validate_loyo_candidates.py
"terrain_exposure" 후보로 LOYO 검증 중.)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOOKUP_PATH = ROOT / "experiments" / "baseline_lgbm" / "terrain_upwind_lookup.json"

_DIR_BINS = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])


def _load_lookup() -> dict:
    with open(LOOKUP_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _nearest_dir_bin(deg: np.ndarray) -> np.ndarray:
    """0~360도 각도를 가장 가까운 8방위(0/45/.../315) bin으로 반올림(circular)."""
    idx = np.round((deg % 360) / 45.0).astype(int) % 8
    return _DIR_BINS[idx]


def add_terrain_upwind_feature(df: pd.DataFrame, group_id: int, dir_col: str = "ldaps_ws10_dir") -> pd.DataFrame:
    """`dir_col`(풍향, 0~360도)을 8방위로 bin해 그룹별 지형 lookup에서
    `terrain_upwind_drop`을 조회해 추가한다. `dir_col`이 없으면 원본을 그대로 반환.
    """
    df = df.copy()
    if dir_col not in df.columns:
        return df

    lookup = _load_lookup()
    dir_table = lookup[str(group_id)]["upwind_drop_by_dir"]
    # JSON 키가 "0.0", "45.0" 형태의 문자열이므로 float 매핑 딕셔너리로 변환.
    value_by_bin = {float(k): v for k, v in dir_table.items()}

    binned = _nearest_dir_bin(df[dir_col].to_numpy())
    df["terrain_upwind_drop"] = np.array([value_by_bin[b] for b in binned], dtype=float)
    return df
