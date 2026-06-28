import numpy as np


def binarize(gray, thresh: int = 128) -> np.ndarray:
    """점수 글자는 배경보다 어두우므로 (gray < thresh) 를 전경 마스크로 본다 (M4 OCR)."""
    return np.asarray(gray) < thresh


def segment_digits(mask: np.ndarray):
    """전경이 있는 연속 열 구간을 좌 -> 우 순서로 [(c0, c1), ...] 로 반환한다."""
    cols = mask.any(axis=0)
    spans = []
    start = None
    for i, on in enumerate(cols):
        if on and start is None:
            start = i
        elif not on and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(cols)))
    return spans


def _resize_to(cell: np.ndarray, shape) -> np.ndarray:
    """최근접 보간으로 cell 을 shape(H, W) 로 리사이즈 (numpy 만 사용)."""
    h, w = shape
    ch, cw = cell.shape
    if ch == 0 or cw == 0:
        return np.zeros(shape, dtype=np.float32)
    ys = (np.arange(h) * ch) // h
    xs = (np.arange(w) * cw) // w
    return cell.astype(np.float32)[ys][:, xs]


def match_digit(cell: np.ndarray, templates: dict):
    """cell 을 각 숫자 템플릿과 비교(L1)해 가장 가까운 숫자를 반환한다."""
    best_dist, best_digit = None, None
    for digit, tmpl in templates.items():
        resized = _resize_to(cell, tmpl.shape)
        dist = float(np.abs(resized - tmpl.astype(np.float32)).sum())
        if best_dist is None or dist < best_dist:
            best_dist, best_digit = dist, digit
    return best_digit


def read_score(gray, templates: dict, thresh: int = 128):
    """점수 strip(gray)에서 숫자를 읽어 정수로 반환. 템플릿/전경이 없으면 None.

    templates: {int_digit: 2D array}. 실제 템플릿은 게임 화면에서 숫자 스프라이트를
    한 번 캡처해 만들어야 한다(폰트 의존). 여기서는 매칭 알고리즘만 제공한다.
    """
    if not templates:
        return None
    mask = binarize(gray, thresh)
    spans = segment_digits(mask)
    if not spans:
        return None
    digits = [str(match_digit(mask[:, c0:c1], templates)) for c0, c1 in spans]
    try:
        return int("".join(digits))
    except ValueError:
        return None
