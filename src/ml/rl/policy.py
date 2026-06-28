import os

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.ml.model.cnn_backbone import DinoCNNBackbone


def load_backbone_weights(backbone: DinoCNNBackbone, sl_ckpt_path) -> bool:
    """SL 체크포인트의 backbone 가중치를 RL 백본에 로드한다 (warm start). 성공 여부를 반환.

    - 전체 DinoSLModel 체크포인트(`backbone.*` / `head.*`)와 구버전 backbone-only
      체크포인트(`cnn.*`)를 모두 처리한다.
    - 형상이 맞는 키만 부분 로드(strict=False)하고, 하나도 못 맞추면 False.
    - 로그를 찍지 않는다(조용히 동작). 결과 로깅은 호출 측 책임.

    주의: PPO(ortho_init=True)는 policy build 시 feature extractor 를 직교 초기화로
    재설정하므로, 이 함수는 반드시 PPO 생성 이후에 호출해야 한다. 그 전에 로드하면
    build 과정에서 덮어써져 warm start 가 무효가 된다.
    """
    if not sl_ckpt_path or not os.path.exists(sl_ckpt_path):
        return False
    try:
        state = torch.load(sl_ckpt_path, map_location="cpu")
    except Exception:
        return False

    # 전체 모델이면 backbone.* 만 추출, 구버전(backbone-only)이면 head.* 제외 후 그대로.
    backbone_state = {
        k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")
    }
    if not backbone_state:
        backbone_state = {k: v for k, v in state.items() if not k.startswith("head.")}

    # 형상이 일치하는 키만 선별해 부분 전이를 허용한다.
    own = backbone.state_dict()
    matched = {
        k: v for k, v in backbone_state.items()
        if k in own and own[k].shape == v.shape
    }
    if not matched:
        return False

    backbone.load_state_dict(matched, strict=False)
    return True


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
