def roi_from_window(left, top, width, height, inset: int = 0) -> dict:
    """윈도우 bbox 로부터 mss 캡처용 ROI dict 를 만든다 (M2).

    inset 만큼 안쪽으로 좁혀 창 테두리를 제외할 수 있다.
    """
    return {
        "left": int(left + inset),
        "top": int(top + inset),
        "width": int(max(0, width - 2 * inset)),
        "height": int(max(0, height - 2 * inset)),
    }


def resolve_roi(cfg, title_substr: str = "chrome") -> dict:
    """제목에 title_substr 가 포함된 창을 찾아 ROI 를 자동 산출한다 (M2).

    하드코딩된 절대좌표 의존을 줄이기 위한 자동화. 창 탐지에 실패하면
    cfg.roi(하드코딩 값)로 안전하게 폴백한다.

    pygetwindow 는 디스플레이가 필요하므로 함수 내부에서 임포트한다(headless 안전).
    """
    try:
        import pygetwindow as gw

        wins = [
            w for w in gw.getAllWindows()
            if title_substr.lower() in (getattr(w, "title", "") or "").lower()
        ]
        if wins:
            w = wins[0]
            return roi_from_window(w.left, w.top, w.width, w.height)
    except Exception:
        pass
    return cfg.roi
