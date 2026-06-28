import torch

from src.ml.dino.action_spec import DinoAction
from src.ml.model.dino_sl_model import DinoSLModel
from src.ml.model.cnn_backbone import DinoCNNBackbone
from src.ml.rl.policy import load_backbone_weights


def _save(obj, tmp_path, name="ckpt.pt"):
    p = tmp_path / name
    torch.save(obj, p)
    return str(p)


def test_warm_start_loads_full_model_checkpoint(tmp_path):
    # SL 이 저장하는 전체 DinoSLModel(state dict: backbone.* / head.*) 체크포인트
    sl = DinoSLModel(n_actions=len(DinoAction), obs_size=84, in_channels=4)
    ckpt = _save(sl.state_dict(), tmp_path)

    bb = DinoCNNBackbone(obs_size=84, in_channels=4)
    assert load_backbone_weights(bb, ckpt) is True
    for k, v in sl.backbone.state_dict().items():
        assert torch.equal(bb.state_dict()[k], v)


def test_warm_start_loads_legacy_backbone_only_checkpoint(tmp_path):
    # 구버전: backbone(cnn.*) state dict 만 저장된 체크포인트도 받아들여야 한다
    src = DinoCNNBackbone(obs_size=84, in_channels=4)
    ckpt = _save(src.state_dict(), tmp_path, "legacy.pt")

    dst = DinoCNNBackbone(obs_size=84, in_channels=4)
    assert load_backbone_weights(dst, ckpt) is True
    for k, v in src.state_dict().items():
        assert torch.equal(dst.state_dict()[k], v)


def test_warm_start_partial_transfer_on_channel_mismatch(tmp_path):
    # in_channels 가 다르면 첫 conv 만 형상 불일치 -> 나머지는 전이, 첫 conv 는 건너뜀
    src = DinoCNNBackbone(obs_size=84, in_channels=1)
    ckpt = _save(src.state_dict(), tmp_path, "mismatch.pt")

    dst = DinoCNNBackbone(obs_size=84, in_channels=4)
    before_first = dst.state_dict()["cnn.0.weight"].clone()
    assert load_backbone_weights(dst, ckpt) is True
    # 첫 conv: 형상 달라 그대로(랜덤 초기화 유지)
    assert torch.equal(dst.state_dict()["cnn.0.weight"], before_first)
    # 둘째 conv: 형상 같아 전이됨
    assert torch.equal(dst.state_dict()["cnn.2.weight"], src.state_dict()["cnn.2.weight"])


def test_warm_start_head_only_checkpoint_returns_false(tmp_path):
    ckpt = _save({"head.0.weight": torch.zeros(4, 4)}, tmp_path, "headonly.pt")
    bb = DinoCNNBackbone(obs_size=84, in_channels=4)
    assert load_backbone_weights(bb, ckpt) is False


def test_warm_start_missing_path_returns_false():
    bb = DinoCNNBackbone(obs_size=84, in_channels=4)
    assert load_backbone_weights(bb, "/no/such/file.pt") is False
    assert load_backbone_weights(bb, None) is False
