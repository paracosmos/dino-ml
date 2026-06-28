import numpy as np

from src.ml.dino.action_spec import DinoAction


def class_distribution(labels, n_classes: int) -> dict:
    """라벨 배열의 클래스별 개수와 비율 (M5 데이터 품질 점검)."""
    labels = np.asarray(labels).astype(np.int64)
    counts = np.bincount(labels, minlength=n_classes)[:n_classes]
    total = int(counts.sum())
    return {
        "counts": counts.tolist(),
        "total": total,
        "ratios": [(int(c) / total if total else 0.0) for c in counts],
    }


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    """행=실제, 열=예측 인 혼동 행렬 (n_classes x n_classes) (M5 평가)."""
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def main():
    # 녹화된 데이터셋의 클래스 분포 리포트
    data = np.load("dino_sl_dataset.npz")
    dist = class_distribution(data["label"], len(DinoAction))
    names = [a.name for a in DinoAction]
    print("dataset size:", dist["total"])
    for name, c, r in zip(names, dist["counts"], dist["ratios"]):
        print(f"  {name:5s} {c:7d}  ({r:.1%})")


if __name__ == "__main__":
    main()
