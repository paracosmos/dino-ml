import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

from src.ml.model.cnn_backbone import DinoCNNBackbone


class DinoRLFeatureExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: spaces.Box, features_dim=256):
        super().__init__(observation_space, features_dim)

        self.backbone = DinoCNNBackbone(features_dim)

    def forward(self, observations: th.Tensor):
        return self.backbone(observations)
