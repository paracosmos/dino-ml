def _stats(xs) -> dict:
    xs = list(xs)
    if not xs:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    return {"mean": mean, "std": var ** 0.5, "min": min(xs), "max": max(xs)}


def summarize_episodes(returns, lengths) -> dict:
    """에피소드 리턴/길이 집계 (M8 평가 하니스의 표준 메트릭).

    실화면이라 학습과 동시 평가가 불가하므로, 별도 평가 세션에서 모은
    에피소드 리턴/생존 길이를 요약하는 데 쓴다.
    """
    return {
        "episodes": len(list(returns)),
        "return": _stats(returns),
        "length": _stats(lengths),
    }
