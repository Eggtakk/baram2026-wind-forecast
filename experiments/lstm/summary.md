# LSTM 실험 요약 및 결론

담당: 탁예린 · 관련 코드: `src/sequence_data.py`, `src/lstm_model.py`, `scripts/train_lstm.py`, `scripts/inference_lstm.py`

## 결론: LightGBM(`experiments/baseline_lgbm/`)을 주력 모델로 확정

LSTM은 group1/2에서 LightGBM에 근접했지만(차이 0.5~1%p) 앞서지 못했고,
group3에서는 오히려 더 낮게 나와 3그룹 전체로는 LightGBM이 우세하다고 판단.
**대회 제출은 `submissions/baseline_lgbm_submission.csv` 기준으로 진행.**

## holdout local_group_score 비교 (뒤 20% 시간순 holdout, 동일 조건)

| group | LightGBM (튜닝+물리feature) | LSTM 1차 (154 feature, lag/rolling 포함) | LSTM 2차 (91 feature, dropout+early stop) |
|---|---|---|---|
| 1 | **0.6205** | 0.5968 | 0.6165 |
| 2 | **0.6410** | 0.6264 | 0.6350 |
| 3 | **0.6073** | 0.6111 | 0.5949 |

## 시도 및 디버깅 히스토리

1. **1차 시도** (lr=1e-3, y 정규화 없음, 154 feature): 20 epoch에도 holdout NMAE 35~44%대에 머묾.
   원인 — 발전량(0~21,000kWh) 그대로 MSE를 학습해 loss 스케일이 너무 커서 사실상 못 움직임.
   → y를 설비용량으로 나눠 0~1로 정규화해서 해결.
2. **2차 시도** (lr=3e-4, patience=10, 154 feature): 3~8 epoch 만에 holdout score가 정점을 찍고
   그 뒤로 계속 나빠짐(train_loss는 계속 감소) — 학습률이 아니라 **과적합** 문제로 확인.
   → lag/rolling feature 제거(154→91, `build_sequence_features`), LSTM 출력에 dropout 추가,
   Adam weight_decay 추가, early stopping 추가.
3. **3차 시도** (dropout+weight_decay+91 feature): group1/2는 개선(LightGBM에 근접),
   **group3는 오히려 하락** — group3는 학습 샘플이 더 적어서(14,030행 vs 20,940행) lag/rolling
   정보가 과적합보다 유용한 신호였던 것으로 추정. 그룹마다 최적 feature 구성이 다를 수 있음을 시사.

세 그룹 모두 여전히 4~8 epoch 근처에서 최고점을 찍고 이후 정체/하락하는 패턴이 반복됨 —
현재 구조(1-layer LSTM, 같은 시각까지의 24시간 창)로는 LightGBM 대비 소폭 열세인 지점이
사실상의 성능 한계에 가까워 보임. 추가로 시도해볼 수 있는 방향(현재는 보류):

- 2-layer LSTM + 진짜 recurrent dropout, attention, 또는 seq_len 확장(48h)
- group3만 154 feature로 되돌려서 재시도
- LightGBM + LSTM 앙상블(가중평균)

## 재현 방법

```bash
python3 scripts/train_lstm.py --epochs 100   # experiments/lstm/에 체크포인트 저장
python3 scripts/inference_lstm.py            # submissions/lstm_submission.csv 생성 (참고용, 미제출)
```
