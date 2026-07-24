"""
LSTM 회귀 모델 정의 (torch 필요).

과거 seq_len시간의 feature 시퀀스를 받아 현재 시각의 발전량(kWh)을 예측하는
many-to-one 구조. src/sequence_data.py가 만든 (N, seq_len, F) 텐서를 그대로
입력으로 받는다.
"""
try:
    import torch.nn as nn
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "torch가 설치되어 있지 않습니다. 로컬 환경에서 `pip install torch` 를 실행한 뒤 사용하세요."
    ) from e


class WindLSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # nn.LSTM 내부 dropout은 num_layers>1일 때만 레이어 사이에 적용됨
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # 1-layer LSTM에서는 위 dropout이 무효화되므로, LSTM 출력과 head 사이에
        # 별도로 dropout을 둔다 (첫 시도에서 3~8 epoch 만에 과적합되는 문제가
        # 있어 추가했다 — src/features.build_sequence_features의 lag/rolling
        # 제거와 함께 과적합 억제용).
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]  # 마지막 타임스텝의 hidden state만 사용
        last_step = self.dropout(last_step)
        return self.head(last_step).squeeze(-1)
