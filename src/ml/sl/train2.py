import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from ml.env.config import DinoEnvConfig
from src.ml.env.action_spec import DinoAction
from src.ml.sl.model_dino import DinoCNN


MODEL_PATH = "dino_sl_cnn.pt"
SEED = 42


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_class_weights(y, num_classes):
    counts = torch.bincount(y, minlength=num_classes).float()

    w = torch.zeros(num_classes)
    nonzero = counts > 0

    w[nonzero] = counts.sum() / counts[nonzero]
    w = w / (w[nonzero].mean() + 1e-9)

    print("class counts:", counts.tolist())
    print("class weights:", w.tolist())

    return w


def main():

    seed_everything(SEED)

    env = DinoEnvConfig()

    # =========================
    # Load dataset
    # =========================
    data = np.load("dino_sl_dataset.npz")

    X = data["obs"]      # (N,H,W,1)
    y = data["label"]

    assert X.shape[1] == env.obs_size and X.shape[2] == env.obs_size

    X = torch.from_numpy(X).permute(0, 3, 1, 2).float() / 255.0
    y = torch.from_numpy(y).long()

    print("dataset size:", len(y))
    print("unique labels:", torch.unique(y).tolist())

    # =========================
    # Split
    # =========================
    n = len(y)
    idx = torch.randperm(n)

    split = int(n * 0.9)
    tr, va = idx[:split], idx[split:]

    train_loader = DataLoader(
        TensorDataset(X[tr], y[tr]),
        batch_size=128,        # ⭐ 가능하면 크게
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        TensorDataset(X[va], y[va]),
        batch_size=256,
        num_workers=2,
        pin_memory=True
    )

    # =========================
    # Device
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = DinoCNN(n_actions=len(DinoAction)).to(device)

    # ⭐ Fine-tuning
    if os.path.exists(MODEL_PATH):
        print("🔥 Loading existing model (fine-tuning)")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("🔥 Training from scratch")

    # =========================
    # Loss
    # =========================
    weights = build_class_weights(y, len(DinoAction)).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)

    # =========================
    # Optimizer (modern default)
    # =========================
    optimizer = optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4
    )

    # ⭐ LR 자동 감소 (성능 안정화 핵심)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    # ⭐ AMP (GPU 가속)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # =========================
    # Training loop
    # =========================
    best_acc = 0.0
    patience = 6
    patience_counter = 0

    MAX_EPOCH = 50

    for epoch in range(1, MAX_EPOCH + 1):

        # -------- TRAIN --------
        model.train()
        loss_sum = 0.0

        for xb, yb in train_loader:

            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()

            # ⭐ gradient explosion 방지
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item()

        # -------- VALIDATION --------
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for xb, yb in val_loader:

                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                pred = model(xb).argmax(dim=1)

                correct += (pred == yb).sum().item()
                total += yb.numel()

        val_acc = correct / total

        print(
            f"Epoch {epoch:02d} | "
            f"loss {loss_sum/len(train_loader):.4f} | "
            f"val acc {val_acc:.4f}"
        )

        scheduler.step(val_acc)

        # ⭐ Best model 저장
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0

            torch.save(model.state_dict(), MODEL_PATH)
            print("✅ BEST MODEL SAVED")

        else:
            patience_counter += 1
            print(f"no improvement ({patience_counter}/{patience})")

        # ⭐ Early stopping
        if patience_counter >= patience:
            print("🛑 Early stopping triggered")
            break

    print("🔥 Training finished")
    print("Best val acc:", best_acc)


if __name__ == "__main__":
    main()
