import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from src.ml.dino.config import DinoEnvConfig
from src.ml.dino.action_spec import DinoAction
from src.ml.model.dino_sl_model import DinoSLModel


MODEL_PATH = "dino_sl_cnn.pt"
SEED = 42


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_class_weights(y, num_classes):
    # 데이터에 없는 클래스(count=0)는 weight=0 으로 두어 학습에서 무시한다.
    counts = torch.bincount(y, minlength=num_classes).float()

    w = torch.zeros(num_classes)
    nonzero = counts > 0

    w[nonzero] = counts.sum() / counts[nonzero]      # 역비율 가중치
    w = w / (w[nonzero].mean() + 1e-9)               # 있는 클래스만 평균 1로 정규화

    print("class counts:", counts.tolist())
    print("class weights:", w.tolist())

    return w


def evaluate(model, loader, device):
    # 검증 정확도 계산
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / total if total else 0.0


def load_checkpoint(model, path, device):
    # 신규 포맷(전체 모델) 우선, 구버전(backbone-only) 은 backbone 에만 로드해 호환
    state = torch.load(path, map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        print("full-model load 실패 -> 구버전 backbone-only 체크포인트로 간주하고 backbone 에 로드")
        model.backbone.load_state_dict(state)


def main():

    seed_everything(SEED)

    env = DinoEnvConfig()

    # 데이터셋 로드
    data = np.load("dino_sl_dataset.npz")

    X = data["obs"]      # (N,H,W,1)
    y = data["label"]

    assert X.shape[1] == env.obs_size and X.shape[2] == env.obs_size, \
        f"dataset obs_size={X.shape[1:3]} != env.obs_size={env.obs_size}"
    assert X.shape[3] == env.n_stack, \
        f"dataset n_stack={X.shape[3]} != env.n_stack={env.n_stack} (재녹화 필요)"

    X = torch.from_numpy(X).permute(0, 3, 1, 2).float() / 255.0
    y = torch.from_numpy(y).long()

    print("dataset size:", len(y))
    print("unique labels:", torch.unique(y).tolist())

    # 학습/검증 분할
    n = len(y)
    idx = torch.randperm(n)

    split = int(n * 0.9)
    tr, va = idx[:split], idx[split:]

    assert len(tr) > 0 and len(va) > 0, \
        f"데이터가 너무 적어 분할 불가 (n={n}). 더 길게 녹화하세요."

    # 데이터가 이미 메모리에 올라와 있으므로 worker 프로세스는 오히려 오버헤드 -> num_workers=0
    train_loader = DataLoader(
        TensorDataset(X[tr], y[tr]),
        batch_size=128,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        TensorDataset(X[va], y[va]),
        batch_size=256,
        num_workers=0,
        pin_memory=True,
    )

    # 디바이스
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = DinoSLModel(
        n_actions=len(DinoAction), obs_size=env.obs_size, in_channels=env.n_stack
    ).to(device)

    # Fine-tuning: 기존 체크포인트가 있으면 이어서 학습
    best_acc = 0.0
    if os.path.exists(MODEL_PATH):
        print("기존 모델 로드 (fine-tuning)")
        load_checkpoint(model, MODEL_PATH, device)
        # 이미 학습된 모델을 더 나쁜 가중치로 덮어쓰지 않도록 현재 정확도로 best_acc 를 시드
        best_acc = evaluate(model, val_loader, device)
        print(f"loaded model val acc: {best_acc:.4f}")
    else:
        print("처음부터 학습")

    # 손실: 학습 분할의 라벨만으로 클래스 가중치 계산 (검증 분포 누수 방지)
    weights = build_class_weights(y[tr], len(DinoAction)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # 옵티마이저
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    # LR 자동 감소 (성능 안정화)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    # AMP (GPU 가속)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # 학습 루프
    patience = 6
    patience_counter = 0
    MAX_EPOCH = 50

    for epoch in range(1, MAX_EPOCH + 1):

        model.train()
        loss_sum = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # gradient explosion 방지

            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item()

        val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:02d} | "
            f"loss {loss_sum/len(train_loader):.4f} | "
            f"val acc {val_acc:.4f}"
        )

        scheduler.step(val_acc)

        # 더 좋은 모델만 저장 (전체 모델 state dict; sl/play.py 가 그대로 로드)
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print("BEST MODEL SAVED")
        else:
            patience_counter += 1
            print(f"no improvement ({patience_counter}/{patience})")

        # Early stopping
        if patience_counter >= patience:
            print("Early stopping")
            break

    print("Training finished. Best val acc:", best_acc)


if __name__ == "__main__":
    main()
