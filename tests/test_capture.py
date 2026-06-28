import time

from src.ml.util.capture import CaptureThread
from src.ml.util.framebuffer import LatestFrameBuffer


def test_capture_once_writes_latest_to_buffer():
    buf = LatestFrameBuffer()
    seq = iter(range(100))
    ct = CaptureThread(lambda: next(seq), buf)
    ct._capture_once()
    ct._capture_once()
    frame, s = buf.get()
    assert frame == 1 and s == 2


def test_capture_thread_runs_then_stops():
    buf = LatestFrameBuffer()
    ct = CaptureThread(lambda: 7, buf, poll_sleep=0.001)
    ct.start()
    for _ in range(2000):                 # 최신 프레임이 채워질 때까지 대기 (timeout)
        if buf.get()[1] > 0:
            break
        time.sleep(0.001)
    ct.stop()
    ct.join(timeout=1.0)
    assert ct.is_alive() is False
    assert buf.get()[0] == 7
