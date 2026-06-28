from collections import deque

import numpy as np


class FrameStacker:
    """최근 n장의 단일 채널 프레임을 (H, W, n) 관측으로 누적한다.

    단일 프레임은 장애물의 속도·접근을 담지 못해 state 가 비-Markov 가 된다.
    연속 프레임을 채널로 쌓아 움직임 정보를 모델에 제공하기 위한 헬퍼.
    """

    def __init__(self, n_stack: int):
        assert n_stack >= 1, "n_stack must be >= 1"
        self.n_stack = n_stack
        self._frames = deque(maxlen=n_stack)

    @staticmethod
    def _to_hw(frame: np.ndarray) -> np.ndarray:
        # (H, W, 1) 또는 (H, W) 를 (H, W) 로 정규화
        if frame.ndim == 3:
            frame = frame[..., 0]
        return frame

    def reset(self, frame: np.ndarray) -> np.ndarray:
        """첫 프레임으로 스택을 가득 채우고 (H, W, n) 반환 (에피소드 시작용)."""
        f = self._to_hw(frame)
        self._frames.clear()
        for _ in range(self.n_stack):
            self._frames.append(f)
        return self._stacked()

    def append(self, frame: np.ndarray) -> np.ndarray:
        """프레임 1장을 넣고 (H, W, n) 반환. 비어 있으면 첫 프레임으로 자동 채운다."""
        f = self._to_hw(frame)
        if not self._frames:
            for _ in range(self.n_stack):
                self._frames.append(f)
        else:
            self._frames.append(f)
        return self._stacked()

    def _stacked(self) -> np.ndarray:
        # 가장 오래된 프레임이 channel 0, 가장 최근이 channel n-1
        return np.stack(self._frames, axis=-1).astype(np.uint8)
