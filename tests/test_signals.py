import numpy as np

from src.ml.dino.config import DinoEnvConfig
from src.ml.dino.signals import detect_dead, score_crop


def test_detect_dead_true_on_bright():
    cfg = DinoEnvConfig()
    assert detect_dead(np.full((100, 100), 255, dtype=np.uint8), cfg) is True


def test_detect_dead_false_on_dark():
    cfg = DinoEnvConfig()
    assert detect_dead(np.zeros((100, 100), dtype=np.uint8), cfg) is False


def test_detect_dead_empty_roi_is_false():
    cfg = DinoEnvConfig()
    assert detect_dead(np.zeros((1, 1), dtype=np.uint8), cfg) is False


def test_score_crop_is_top_right():
    crop = score_crop(np.zeros((100, 200), dtype=np.uint8))
    assert crop.shape == (40, 60)
