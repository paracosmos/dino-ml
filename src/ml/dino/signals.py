import numpy as np

from src.ml.dino.config import DinoEnvConfig


def detect_dead(gray: np.ndarray, cfg: DinoEnvConfig) -> bool:
    """game-over 화면은 밝으므로 중앙 crop 평균 밝기로 죽음을 추정한다 (M4).

    순수 함수로 분리해 단위테스트가 가능하도록 했다. 임계값은 cfg.dead_threshold.
    """
    h, w = gray.shape[:2]
    roi = gray[int(h * 0.2):int(h * 0.45), int(w * 0.35):int(w * 0.65)]
    if roi.size == 0:
        return False
    return float(roi.mean()) > cfg.dead_threshold


def score_crop(gray: np.ndarray) -> np.ndarray:
    """점수가 표시되는 우상단 영역 crop (M4 점수 OCR 의 입력 후보).

    실제 OCR 연동은 후속 작업으로 남기고, 여기서는 crop 만 제공한다.
    """
    h, w = gray.shape[:2]
    return gray[0:int(h * 0.4), int(w * 0.7):w]
