import threading


class LatestFrameBuffer:
    """생산자(캡처 스레드)가 최신 프레임을 쓰고 소비자(추론)가 읽는 thread-safe 홀더 (M9).

    추론이 캡처보다 느려도 항상 '가장 최근' 프레임만 보게 하여 실시간성을 유지한다.
    seq 는 프레임이 갱신될 때마다 증가하므로 소비자가 새 프레임 여부를 알 수 있다.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0

    def put(self, frame):
        with self._lock:
            self._frame = frame
            self._seq += 1

    def get(self):
        """(frame, seq) 를 반환. 아직 프레임이 없으면 (None, 0)."""
        with self._lock:
            return self._frame, self._seq
