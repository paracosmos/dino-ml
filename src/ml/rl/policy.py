import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

from src.ml.model.cnn_backbone import DinoCNNBackbone

class DinoFeatureExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        self.cnn = DinoCNNBackbone()

        self.linear = nn.Sequential(
            nn.Linear(self.cnn._features_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        x = self.cnn(obs)
        return self.linear(x)
