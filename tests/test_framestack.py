import numpy as np

from src.ml.dino.framestack import FrameStacker


def test_reset_fills_all_channels():
    fs = FrameStacker(4)
    out = fs.reset(np.full((84, 84, 1), 7, dtype=np.uint8))
    assert out.shape == (84, 84, 4)
    assert out.dtype == np.uint8
    assert (out == 7).all()


def test_append_auto_inits_when_empty():
    fs = FrameStacker(3)
    out = fs.append(np.zeros((10, 10), dtype=np.uint8))
    assert out.shape == (10, 10, 3)


def test_append_shifts_oldest_to_newest():
    fs = FrameStacker(3)
    fs.reset(np.full((2, 2, 1), 1, dtype=np.uint8))
    fs.append(np.full((2, 2, 1), 2, dtype=np.uint8))
    out = fs.append(np.full((2, 2, 1), 3, dtype=np.uint8))
    # 채널 순서: 가장 오래된 프레임(1) -> 최신 프레임(3)
    assert [int(out[0, 0, c]) for c in range(3)] == [1, 2, 3]
