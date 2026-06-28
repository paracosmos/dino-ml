import torch
import torch.nn as nn


class DinoCNNBackbone(nn.Module):

    def __init__(self, obs_size: int = 84):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        # obs_size must match DinoEnvConfig.obs_size used to build observations.
        with torch.no_grad():
            sample = torch.zeros(1, 1, obs_size, obs_size)
            self.features_dim = self.cnn(sample).shape[1]

    def forward(self, x):
        return self.cnn(x)
