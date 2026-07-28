# 외부 공개 데이터 — SRTM DEM 방향별 풍상측(upwind) 지형고도차

대회 규칙(대회 개요 페이지 "대회규칙 및 유의사항" 3~4절) 상 외부 데이터는
아래 조건을 만족하면 사용 가능하다:

- 예측기준시점 이전에 생성·공개·확정되어 실제로 활용 가능했던 정보일 것
- 출처, 수집 방법, 수집 시점, 사용 변수, 라이선스, 전처리 코드를 포함해
  재현 가능하게 문서화할 것 (2차 평가 대상자로 지정될 경우 제출 필요)

이 문서는 `src/terrain_features.py`가 사용하는 SRTM DEM(수치표고모델)
기반 방향별 지형고도차 feature의 출처와 재현 방법을 기록한다.

## 왜 이 데이터가 규칙을 만족하는가

SRTM(Shuttle Radar Topography Mission)은 2000년 2월 NASA 우주왕복선
임무로 지구 표면을 스캔해 만든 DEM으로, 2015년경부터 전세계 30m 해상도
버전이 공개됐다. 지형 고도 자체는 이 대회의 학습 기간(2022년~) 및
평가 기간(2025년) 어느 시점보다도 훨씬 이전에 이미 확정·공개된 정적
데이터이며, 지형은 시간에 따라 변하지 않으므로 leakage 위험이 없다.

## 터빈 좌표 확인 근거

`data/open/info.xlsx`(대회 공식 제공 파일, `info` 시트)의 "좌표(Google)"
컬럼(DMS 형식)을 파싱해 그룹별 터빈 중심점(centroid)을 계산:

| KPX 그룹 | 터빈 수 | 중심점 좌표 | 중심점 고도(SRTM) |
|---|---|---|---|
| 1 (태백가덕산 1~6호기, VESTAS V126) | 6 | 37.287127°N, 128.952021°E | 1062 m |
| 2 (태백가덕산 7~12호기, VESTAS V126) | 6 | 37.282255°N, 128.965148°E | 1076 m |
| 3 (태백원동 UNISON U136) | 5 | 37.275199°N, 128.971444°E | 965 m |

## 데이터 출처 및 수집 방법

- API: https://api.opentopodata.org/v1/srtm30m (opentopodata.org, SRTM
  30m 데이터셋을 제공하는 공개 REST API)
- 조회일: 2026-07-28
- 수집 방법: 각 그룹 중심점에서 8방위(0/45/.../315도, 진북 기준) x
  2거리(1000m, 2500m) = 16개 지점 + 중심점 1개(총 17개/그룹)의 좌표를
  계산(구면 좌표 근사, `R=6371km`)해 API로 고도 조회. 원시 조회 결과는
  `experiments/baseline_lgbm/terrain_upwind_lookup.json`에 사람이 읽을 수
  있는 형태로 저장.
- 라이선스: SRTM 데이터는 NASA/USGS의 공개 데이터셋(미국 정부 저작물,
  공공 도메인에 준함). opentopodata.org는 이 공개 데이터를 무료 API로
  제공하는 오픈소스 프로젝트(요청 빈도 제한 있음, 상업적 재배포가 아닌
  분석 목적의 조회는 이용약관상 허용).

## 방법론 — 방향별 upwind_drop 정의

각 그룹의 터빈은 능선(산등성이) 위에 위치한다(그룹3 EDA에서 이미 확인된
야간 활강풍 패턴과 일치). 예보 풍향에 따라 "지금 부는 바람이 어느 지형을
타고 올라오는가"가 달라지므로, 8방위 각각에 대해:

```
upwind_drop(방향) = 중심점 고도 - 그 방향 프로필(1000m, 2500m 지점) 평균 고도
```

으로 정의한다. 양수(+)면 그 방향은 터빈보다 낮은 지형(계곡 방향 — 바람이
그 방향에서 불어올 때 능선을 타고 오르며 가속되는 지형풍 가능성), 음수(-)면
그 방향은 터빈보다 높은 지형(능선 반대편 — 지형에 막혀 약해질 가능성)을
뜻한다. 실제 조회 결과(`terrain_upwind_lookup.json`)를 보면 대부분의
방향에서 양수(+50~+165m)가 나와, 세 그룹 모두 주변보다 확실히 높은
능선 위에 있다는 게 정량적으로 확인된다.

## 재현 방법

```python
from src.terrain_features import add_terrain_upwind_feature
# df: forecast_kst_dtm, ldaps_ws10_dir 등을 포함한 그룹별 데이터프레임
df = add_terrain_upwind_feature(df, group_id=1, dir_col="ldaps_ws10_dir")
# -> df["terrain_upwind_drop"] 컬럼 추가 (ldaps_ws10_dir을 8방위로 반올림해
#    experiments/baseline_lgbm/terrain_upwind_lookup.json에서 조회)
```

좌표 파싱 → centroid 계산 → 프로필 포인트 생성 → API 조회 → lookup table
저장까지 전 과정이 결정적(deterministic)이라 동일한 입력(info.xlsx,
SRTM 30m 데이터셋)에서 항상 동일한 결과가 재현된다.

## 현재 상태 (2026-07-28)

`scripts/validate_loyo_candidates.py`의 "terrain_exposure" 후보로 LOYO
검증했으나, 세 그룹 모두 baseline std 대비 유의미한 delta를 만들지 못해
**프로덕션 레시피에는 아직 편입하지 않았다**(상세:
`experiments/baseline_lgbm/rated_output_investigation.md` 18번 섹션).
