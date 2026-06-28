import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

from src.ml.model.cnn_backbone import DinoCNNBackbone

class DinoFeatureExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        # 관측 공간에서 obs_size / 채널 수를 추출해 백본에 전달.
        # SB3 는 이미지 관측을 channels-first 로 transpose 하므로 두 레이아웃
        # ((H, W, C) / (C, H, W)) 모두를 견고하게 처리한다.
        # 프레임은 정사각(H == W)이고 채널 수(n_stack)는 obs_size 와 다르므로
        # 두 spatial 차원이 같은 쪽을 H/W 로 판별할 수 있다.
        shape = observation_space.shape
        if shape[0] == shape[1]:                  # (H, W, C) channels-last
            obs_size, in_channels = shape[0], shape[2]
        else:                                     # (C, H, W) channels-first
            in_channels, obs_size = shape[0], shape[1]

        self.cnn = DinoCNNBackbone(obs_size, in_channels)

        self.linear = nn.Sequential(
            nn.Linear(self.cnn.features_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        x = self.cnn(obs)
        return self.linear(x)
