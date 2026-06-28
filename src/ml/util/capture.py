import threading
import time


class CaptureThread(threading.Thread):
    """grab_fn() 으로 프레임을 받아 LatestFrameBuffer 에 계속 써 넣는 생산자 스레드 (M9).

    소비자(추론)는 buffer.get() 으로 항상 '가장 최근' 프레임만 읽으므로, 추론이
    캡처보다 느려도 오래된 프레임에 막히지 않고 실시간성을 유지한다.
    """

    def __init__(self, grab_fn, buffer, poll_sleep: float = 0.0):
        super().__init__(daemon=True)
        self._grab = grab_fn
        self._buffer = buffer
        self._poll_sleep = poll_sleep
        # 주의: 속성명을 _stop 으로 두면 threading.Thread 내부 메서드를 가려
        # join() 이 깨진다. _stop_event 로 분리한다.
        self._stop_event = threading.Event()

    def _capture_once(self):
        self._buffer.put(self._grab())

    def run(self):
        while not self._stop_event.is_set():
            self._capture_once()
            if self._poll_sleep:
                time.sleep(self._poll_sleep)

    def stop(self):
        self._stop_event.set()
