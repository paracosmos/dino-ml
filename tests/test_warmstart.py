import numpy as np
import torch
from gymnasium import spaces

from src.ml.dino.action_spec import DinoAction
from src.ml.model.dino_sl_model import DinoSLModel
from src.ml.rl.policy import DinoFeatureExtractor, load_backbone_weights
from src.ml.model.cnn_backbone import DinoCNNBackbone


def _sl_checkpoint(tmp_path, n_stack=4):
    sl = DinoSLModel(n_actions=len(DinoAction), obs_size=84, in_channels=n_stack)
    path = tmp_path / "dino_sl_cnn.pt"
    torch.save(sl.state_dict(), path)
    return sl, str(path)


def test_warm_start_copies_sl_backbone_into_extractor(tmp_path):
    n_stack = 4
    sl, ckpt = _sl_checkpoint(tmp_path, n_stack)

    obs_space = spaces.Box(low=0, high=255, shape=(84, 84, n_stack), dtype=np.uint8)
    fe = DinoFeatureExtractor(obs_space, features_dim=256, pretrained_backbone=ckpt)

    # RL 백본 가중치가 SL 체크포인트의 backbone 과 동일해야 한다 (전이 성공)
    for k, v in sl.backbone.state_dict().items():
        assert torch.equal(fe.cnn.state_dict()[k], v)


def test_warm_start_missing_path_is_ignored():
    obs_space = spaces.Box(low=0, high=255, shape=(84, 84, 4), dtype=np.uint8)
    # 존재하지 않는 경로여도 크래시 없이 무작위 초기화로 생성되어야 한다
    fe = DinoFeatureExtractor(obs_space, 256, pretrained_backbone="/no/such/file.pt")
    assert fe.cnn.features_dim > 0


def test_load_backbone_weights_returns_false_without_file():
    bb = DinoCNNBackbone(obs_size=84, in_channels=4)
    assert load_backbone_weights(bb, "/no/such/file.pt") is False
    assert load_backbone_weights(bb, None) is False
