import cv2
import numpy as np
from src.ml.dino.config import DinoEnvConfig

def preprocess_obs(bgr: np.ndarray, cfg: DinoEnvConfig) -> np.ndarray:
    """BGR -> (obs_size, obs_size, 1) uint8"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (cfg.obs_size, cfg.obs_size), interpolation=cv2.INTER_AREA)
    return resized[:, :, None].astype(np.uint8)

def preprocess_obs_float(bgr: np.ndarray, cfg: DinoEnvConfig) -> np.ndarray:
    """BGR -> (1,1,obs_size,obs_size) float32 in [0,1] for inference"""
    obs_u8 = preprocess_obs(bgr, cfg)               # (H,W,1) uint8
    obs = obs_u8.astype(np.float32) / 255.0
    # (1,1,H,W)
    return obs.transpose(2, 0, 1)[None, ...]
