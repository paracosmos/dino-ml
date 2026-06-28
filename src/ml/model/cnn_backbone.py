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

        # obs_size 는 관측을 만들 때 쓰는 DinoEnvConfig.obs_size 와 일치해야 한다.
        with torch.no_grad():
            sample = torch.zeros(1, 1, obs_size, obs_size)
            self.features_dim = self.cnn(sample).shape[1]

    def forward(self, x):
        return self.cnn(x)
