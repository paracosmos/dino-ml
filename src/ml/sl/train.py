import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from src.ml.dino.config import DinoEnvConfig
from src.ml.dino.action_spec import DinoAction
from src.ml.model.dino_sl_model import DinoSLModel  

MODEL_PATH = "dino_sl_cnn.pt"

def main():
    env = DinoEnvConfig()

    data = np.load("dino_sl_dataset.npz")
    X = data["obs"]     # (N,obs,obs,1) uint8
    y = data["label"]   # (N,)

    # 안전 체크: env.obs_size와 데이터 크기 일치
    assert X.shape[1] == env.obs_size and X.shape[2] == env.obs_size, \
        f"dataset obs_size={X.shape[1:3]} != env.obs_size={env.obs_size}"

    # torch: (N,1,H,W) float
    X = torch.from_numpy(X).permute(0, 3, 1, 2).float() / 255.0
    y = torch.from_numpy(y).long()

    # ✅ 진단: 라벨이 실제로 어떤 값들로 구성됐는지 확인 (DUCK=2가 아예 없을 수 있음)
    # 예) [0,1] 이면 DUCK(2) 샘플이 0개 → 기존 weight 계산이 shape=[2]가 되어 에러 발생
    print("unique labels:", torch.unique(y).tolist())

    n = len(y)
    idx = torch.randperm(n)
    split = int(n * 0.9)
    tr, va = idx[:split], idx[split:]

    train_loader = DataLoader(TensorDataset(X[tr], y[tr]), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X[va], y[va]), batch_size=256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DinoSLModel(n_actions=len(DinoAction)).to(device)

     # ⭐ Fine-tuning
    if os.path.exists(MODEL_PATH):
        print("🔥 Loading existing model (fine-tuning)")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("⭐ Training from scratch")


    # 데이터에 없는 클래스(count=0)는 weight를 0으로 둬서(=학습에서 무시) 안정적으로 학습 진행
    num_classes = len(DinoAction)
    counts = torch.bincount(y, minlength=num_classes).float()  # ✅ minlength로 길이 보장

    w = torch.zeros(num_classes, dtype=torch.float32)
    nonzero = counts > 0
    w[nonzero] = counts.sum() / counts[nonzero]               # 역비율 가중치
    w = w / (w[nonzero].mean() + 1e-9)                        # 있는 클래스만 평균 1로 정규화

    print("class counts:", counts.tolist(), "weights:", w.tolist())

    criterion = nn.CrossEntropyLoss(weight=w.to(device))

    opt = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 11):
        model.train()
        loss_sum = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.numel()

        print(f"Epoch {epoch} | loss {loss_sum/len(train_loader):.4f} | val acc {correct/total:.3f}")

    torch.save(model.backbone.state_dict(), "dino_sl_cnn.pt")
    print("Saved model: dino_sl_cnn.pt")

if __name__ == "__main__":
    main()
