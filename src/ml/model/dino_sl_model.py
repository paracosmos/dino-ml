import torch.nn as nn

from src.ml.model.cnn_backbone import DinoCNNBackbone

class DinoSLModel(nn.Module):

    def __init__(self, n_actions: int, obs_size: int = 84):
        super().__init__()

        self.backbone = DinoCNNBackbone(obs_size)

        self.head = nn.Sequential(
            nn.Linear(self.backbone.features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)

