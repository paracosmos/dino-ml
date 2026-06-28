import torch
import torch.nn as nn


class DinoCNNBackbone(nn.Module):

    def __init__(self, obs_size: int = 84, in_channels: int = 1):
        super().__init__()

        self.in_channels = in_channels

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        # obs_size / in_channels 는 DinoEnvConfig.obs_size / n_stack 과 일치해야 한다.
        with torch.no_grad():
            sample = torch.zeros(1, in_channels, obs_size, obs_size)
            self.features_dim = self.cnn(sample).shape[1]

    def forward(self, x):
        return self.cnn(x)
