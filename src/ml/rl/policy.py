import os

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.ml.model.cnn_backbone import DinoCNNBackbone


def load_backbone_weights(backbone: DinoCNNBackbone, sl_ckpt_path: str) -> bool:
    """SL 체크포인트(dino_sl_cnn.pt)의 backbone 가중치를 RL 백본에 로드한다 (warm start).

    SL 은 전체 DinoSLModel(state dict key 가 `backbone.*` / `head.*`)을 저장하므로
    `backbone.` 접두어를 떼어 백본 전용 state dict 로 만든 뒤 로드한다.
    실패(파일 없음/형상 불일치) 시 조용히 건너뛰고 False 를 반환한다.
    """
    if not sl_ckpt_path or not os.path.exists(sl_ckpt_path):
        return False
    try:
        state = torch.load(sl_ckpt_path, map_location="cpu")
        prefix = "backbone."
        backbone_state = {
            k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)
        }
        if not backbone_state:
            print(f"[warmstart] {sl_ckpt_path} 에 backbone.* 키가 없어 건너뜀")
            return False
        backbone.load_state_dict(backbone_state)
        print(f"[warmstart] SL backbone 가중치 로드 완료: {sl_ckpt_path}")
        return True
    except Exception as e:
        print(f"[warmstart] 로드 실패 ({e}); 무작위 초기화로 진행")
        return False


class DinoFeatureExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space, features_dim=256, pretrained_backbone=None):
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

        # SL 로 사전학습한 표현을 재사용 (plan M4: SL -> RL 전이학습)
        if pretrained_backbone:
            load_backbone_weights(self.cnn, pretrained_backbone)

        self.linear = nn.Sequential(
            nn.Linear(self.cnn.features_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        x = self.cnn(obs)
        return self.linear(x)
