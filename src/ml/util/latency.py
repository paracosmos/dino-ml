class LatencyTracker:
    """프레임 처리 지연(초) 표본을 모아 통계를 낸다 (M3 실시간 제어 계측).

    실화면 게임은 캡처-추론-입력이 fps 예산(예: 30fps -> 33ms) 안에 끝나야
    의미가 있으므로, 루프 지연 분포를 측정해 병목을 드러내기 위한 헬퍼.
    """

    def __init__(self):
        self._samples = []

    def add(self, dt: float):
        self._samples.append(float(dt))

    def summary(self) -> dict:
        s = self._samples
        if not s:
            return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
        return {
            "count": len(s),
            "mean": sum(s) / len(s),
            "min": min(s),
            "max": max(s),
        }

    def fps_estimate(self) -> float:
        mean = self.summary()["mean"]
        return (1.0 / mean) if mean > 0 else 0.0
