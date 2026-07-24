"""
LSTM 학습 스크립트 (torch 필요, 로컬 환경에서 실행할 것).

이 저장소를 만든 샌드박스 환경은 네트워크 제약으로 torch(CPU wheel
~400MB+)를 설치할 수 없어서, 이 스크립트는 로컬에서 직접 실행/디버깅해야
한다. 코드 자체는 src/sequence_data.py(윈도우 생성, torch 없이 검증 완료)와
src/metrics.py(공식 산식, 단위테스트 완료)를 그대로 재사용하므로 로직상
위험은 최소화했지만, 실제 학습 루프(torch 텐서 흐름)는 로컬 첫 실행 시
확인이 필요하다.

설치 (레포 루트, conda 환경에서):
    pip install torch
    # 또는: conda install pytorch cpuonly -c pytorch

실행 예:
    python3 scripts/train_lstm.py --group 3 --epochs 60
    python3 scripts/train_lstm.py --epochs 60          # 그룹 1,2,3 순서대로 전부

디버깅 히스토리 (같은 문제를 또 만나지 않도록 기록):
1차: y를 정규화하지 않고 MSE를 그대로 학습 -> 20 epoch에도 holdout NMAE가
    35~44%대. 원인은 loss 스케일이 너무 커서 lr(1e-3)로 사실상 못 움직임.
    -> y를 설비용량으로 나눠 0~1로 정규화해서 해결.
2차: lr을 3e-4로 낮추고 scheduler patience를 10으로 늘렸는데도 3~8 epoch
    만에 holdout score가 정점을 찍고 그 뒤로 계속 나빠짐(train_loss는 계속
    감소) -> 이건 학습률 문제가 아니라 **과적합**이었다. 154개 feature
    (lag/rolling 포함) 대비 학습 샘플이 14,030~20,940개뿐이라 LSTM이 너무
    빨리 train set을 외워버림.
    -> (a) src/features.build_sequence_features로 lag/rolling 제거(154->91
       feature — 시퀀스 자체가 이미 시간축 정보를 담고 있어 중복),
       (b) LSTM 출력에 dropout 추가, (c) Adam에 weight_decay 추가,
       (d) 학습 루프에 early stopping 추가(개선 없으면 --early-stop-patience
       epoch 후 조기 종료)로 대응.

이 구조에서도 여전히 몇 epoch 만에 정체된다면 --dropout을 더 올리거나
(0.2->0.4) --hidden-size를 낮춰볼 것.

출력: experiments/lstm/group{n}_lstm.pt (state_dict, holdout 최고점 기준 저장),
      group{n}_scaler.npz (feature 표준화 평균/표준편차),
      group{n}_lstm_meta.json (feature 목록, 최고 성능 기록)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as e:
    raise SystemExit(
        "torch가 설치되어 있지 않습니다.\n"
        "  pip install torch\n"
        "실행 후 다시 시도하세요. (자세한 배경은 이 파일 상단 주석 참고)"
    ) from e

from src.lstm_model import WindLSTM
from src.metrics import CAPACITY_KWH, validate_single_group
from src.sequence_data import build_train_sequences
from src.validation import array_time_split

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "lstm"

SEQ_LEN = 24
HOLDOUT_RATIO = 0.2

# group1/2는 학습 데이터가 더 많아(20,940행 vs group3 14,030행) 더 큰 모델을
# 감당할 여유가 있어 hidden_size를 키운다. --hidden-size를 명시하면 이 값을
# 무시하고 모든 그룹에 그 값을 그대로 쓴다.
DEFAULT_HIDDEN_SIZE = {1: 96, 2: 96, 3: 64}


def train_group(
    group_id: int,
    epochs: int = 20,
    batch_size: int = 256,
    hidden_size: int = 64,
    lr: float = 1e-3,
    dropout: float = 0.2,
    weight_decay: float = 1e-5,
    early_stop_patience: int = 15,
    seq_len: int = SEQ_LEN,
    device: str = "cpu",
) -> float:
    print(f"\n=== group{group_id}: 시퀀스 생성 (seq_len={seq_len}) ===")
    data = build_train_sequences(group_id, seq_len=seq_len)
    X, y = data["X"], data["y"]

    (Xtr, ytr), (Xho, yho) = array_time_split(X, y, holdout_ratio=HOLDOUT_RATIO)
    print(f"train={len(Xtr)}  holdout={len(Xho)}  n_features={X.shape[-1]}")

    # feature 표준화 (train 통계만 사용, test/holdout에 그대로 적용 — 누수 방지)
    mean = Xtr.mean(axis=(0, 1), keepdims=True)
    std = Xtr.std(axis=(0, 1), keepdims=True) + 1e-6
    Xtr_n = (Xtr - mean) / std
    Xho_n = (Xho - mean) / std

    # 타깃(y)도 스케일링 필수: kWh 단위 그대로(0~21000) MSE를 학습하면 loss
    # 스케일이 너무 커서 Adam 기본 lr(1e-3)로는 수렴이 극도로 느리다
    # (실측: 20 epoch에도 holdout NMAE가 35%대에서 못 벗어남).
    # 설비용량으로 나눠 대략 0~1 범위로 맞춘다.
    capacity = CAPACITY_KWH[f"kpx_group_{group_id}"]
    ytr_n = ytr / capacity
    print(f"y를 설비용량({capacity}kWh)으로 나눠 0~1 범위로 정규화해서 학습")

    dev = torch.device(device)
    model = WindLSTM(n_features=X.shape[-1], hidden_size=hidden_size, dropout=dropout).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # patience를 넉넉하게 줘서 학습률이 너무 빨리 깎여 초반 수렴점에 갇히는 걸 방지
    # (1차 시도에서 lr=1e-3 + patience=5로 3~5 epoch 만에 정체되는 문제가 있었음).
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr_n), torch.from_numpy(ytr_n)),
        batch_size=batch_size,
        shuffle=True,
    )
    Xho_t = torch.from_numpy(Xho_n).to(dev)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_score = -1e9
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        train_loss_norm = total_loss / len(Xtr)  # 정규화된(0~1) 스케일의 MSE

        model.eval()
        with torch.no_grad():
            pred_ho_norm = model(Xho_t).cpu().numpy()
        pred_ho = np.clip(pred_ho_norm * capacity, a_min=0, a_max=None)  # kWh 스케일로 복원

        r = validate_single_group(yho, pred_ho, group_id=group_id)
        score = 0.5 * r["one_minus_nmae"] + 0.5 * r["ficr"]
        scheduler.step(score)
        cur_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[group{group_id}] epoch {epoch:3d}/{epochs}  train_loss(norm)={train_loss_norm:.5f}  "
            f"holdout_nmae={r['nmae']:.4f}  holdout_ficr={r['ficr']:.4f}  score={score:.4f}  lr={cur_lr:.2e}"
        )

        if score > best_score:
            best_score = score
            epochs_no_improve = 0
            torch.save(model.state_dict(), OUT_DIR / f"group{group_id}_lstm.pt")
            np.savez(OUT_DIR / f"group{group_id}_scaler.npz", mean=mean, std=std)
            meta = {
                "group_id": group_id,
                "seq_len": seq_len,
                "hidden_size": hidden_size,
                "dropout": dropout,
                "capacity_kwh": capacity,
                "feature_cols": data["feature_cols"],
                "best_epoch": epoch,
                "best_score": best_score,
                "best_nmae": r["nmae"],
                "best_ficr": r["ficr"],
            }
            with open(OUT_DIR / f"group{group_id}_lstm_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(
                    f"[group{group_id}] {early_stop_patience} epoch 연속 개선 없음 -> "
                    f"epoch {epoch}에서 조기 종료"
                )
                break

    print(f"[group{group_id}] 최종 최고 holdout score={best_score:.4f} -> {OUT_DIR}")
    return best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=int, default=None, help="1/2/3 중 하나. 생략하면 세 그룹 모두 순서대로 학습")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="생략하면 그룹별 기본값 사용 (group1/2=96, group3=64). 명시하면 모든 그룹에 그 값 적용.",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="초기 학습률")
    parser.add_argument("--dropout", type=float, default=0.2, help="과적합 억제용 dropout (2차 디버깅에서 추가)")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Adam L2 정규화")
    parser.add_argument(
        "--early-stop-patience", type=int, default=15, help="이 epoch 동안 holdout score 개선이 없으면 조기 종료"
    )
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--device", type=str, default="cpu", help="'cpu' 또는 'cuda' (GPU 있을 때)")
    args = parser.parse_args()

    groups = [args.group] if args.group else [1, 2, 3]
    results = {}
    for gid in groups:
        hidden_size = args.hidden_size if args.hidden_size is not None else DEFAULT_HIDDEN_SIZE[gid]
        results[gid] = train_group(
            gid,
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_size=hidden_size,
            lr=args.lr,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            early_stop_patience=args.early_stop_patience,
            seq_len=args.seq_len,
            device=args.device,
        )

    print("\n=== 요약 (holdout local_group_score) ===")
    for gid, score in results.items():
        print(f"group{gid}: {score:.4f}")


if __name__ == "__main__":
    main()
