import numpy as np
import torch
from gymnasium import spaces

from src.ml.dino.config import DinoEnvConfig
from src.ml.dino.action_spec import DinoAction
from src.ml.model.cnn_backbone import DinoCNNBackbone
from src.ml.model.dino_sl_model import DinoSLModel
from src.ml.rl.policy import DinoFeatureExtractor


def test_backbone_features_dim_is_positive_int():
    bb = DinoCNNBackbone()
    assert isinstance(bb.features_dim, int)
    assert bb.features_dim > 0


def test_backbone_forward_shape():
    bb = DinoCNNBackbone()
    out = bb(torch.zeros(2, 1, 84, 84))
    assert out.shape == (2, bb.features_dim)


def test_backbone_respects_obs_size():
    cfg = DinoEnvConfig()
    bb = DinoCNNBackbone(obs_size=cfg.obs_size)
    out = bb(torch.zeros(1, 1, cfg.obs_size, cfg.obs_size))
    assert out.shape == (1, bb.features_dim)


def test_sl_model_outputs_one_logit_per_action():
    n = len(DinoAction)
    model = DinoSLModel(n_actions=n)
    out = model(torch.zeros(4, 1, 84, 84))
    assert out.shape == (4, n)


def test_sl_model_threads_obs_size():
    # obs_size 가 백본까지 실제로 전달되어 다른 입력 크기에서도 동작해야 한다.
    n = len(DinoAction)
    model = DinoSLModel(n_actions=n, obs_size=64)
    out = model(torch.zeros(2, 1, 64, 64))
    assert out.shape == (2, n)


def test_rl_feature_extractor_threads_obs_size():
    obs_space = spaces.Box(low=0, high=255, shape=(64, 64, 1), dtype=np.uint8)
    fe = DinoFeatureExtractor(obs_space, features_dim=128)
    out = fe(torch.zeros(2, 1, 64, 64))
    assert out.shape == (2, 128)


def test_backbone_multichannel_forward():
    bb = DinoCNNBackbone(obs_size=84, in_channels=4)
    out = bb(torch.zeros(2, 4, 84, 84))
    assert out.shape == (2, bb.features_dim)


def test_sl_model_multichannel_forward():
    n = len(DinoAction)
    model = DinoSLModel(n_actions=n, obs_size=84, in_channels=4)
    out = model(torch.zeros(2, 4, 84, 84))
    assert out.shape == (2, n)


def test_rl_extractor_handles_both_channel_layouts():
    # channels-last (H, W, C): env 가 정의한 원래 관측 공간
    last = spaces.Box(low=0, high=255, shape=(84, 84, 4), dtype=np.uint8)
    assert DinoFeatureExtractor(last, 256)(torch.zeros(2, 4, 84, 84)).shape == (2, 256)
    # channels-first (C, H, W): SB3 가 이미지 관측을 transpose 한 형태
    first = spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
    assert DinoFeatureExtractor(first, 256)(torch.zeros(2, 4, 84, 84)).shape == (2, 256)


def test_rl_feature_extractor_forward_shape():
    obs_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
    fe = DinoFeatureExtractor(obs_space, features_dim=256)
    out = fe(torch.zeros(3, 1, 84, 84))
    assert out.shape == (3, 256)
