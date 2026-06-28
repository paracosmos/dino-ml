import numpy as np

from src.ml.dino.config import DinoEnvConfig
from src.ml.dino.preprocess import preprocess_obs, preprocess_obs_float


def _dummy_bgr(cfg, value=0):
    return np.full((cfg.roi_height, cfg.roi_width, 3), value, dtype=np.uint8)


def test_preprocess_obs_shape_and_dtype():
    cfg = DinoEnvConfig()
    obs = preprocess_obs(_dummy_bgr(cfg), cfg)
    assert obs.shape == (cfg.obs_size, cfg.obs_size, 1)
    assert obs.dtype == np.uint8


def test_preprocess_obs_float_shape_and_range():
    cfg = DinoEnvConfig()
    obs = preprocess_obs_float(_dummy_bgr(cfg, value=255), cfg)
    assert obs.shape == (1, 1, cfg.obs_size, cfg.obs_size)
    assert obs.dtype == np.float32
    assert obs.min() >= 0.0 and obs.max() <= 1.0
