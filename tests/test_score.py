import numpy as np

from src.ml.dino.score import binarize, segment_digits, match_digit, read_score


# 서로 다른, 모든 열에 전경이 있는 2x2 숫자 템플릿 (합성)
T2 = np.array([[1, 1], [1, 0]], dtype=bool)
T5 = np.array([[1, 1], [0, 1]], dtype=bool)
TEMPLATES = {2: T2, 5: T5}


def _strip(*masks):
    # 숫자 사이에 빈 열(gap)을 끼워 strip mask 를 만든 뒤, 전경=0(어두움) gray 로 변환
    gap = np.zeros((masks[0].shape[0], 1), dtype=bool)
    pieces = []
    for i, m in enumerate(masks):
        if i:
            pieces.append(gap)
        pieces.append(m)
    mask = np.concatenate(pieces, axis=1)
    return np.where(mask, 0, 255).astype(np.uint8)


def test_binarize_marks_dark_as_foreground():
    gray = np.array([[0, 255], [10, 200]], dtype=np.uint8)
    assert binarize(gray, 128).tolist() == [[True, False], [True, False]]


def test_segment_digits_splits_on_gaps():
    gray = _strip(T2, T5)
    spans = segment_digits(binarize(gray))
    assert spans == [(0, 2), (3, 5)]


def test_match_digit_picks_closest_template():
    assert match_digit(T2, TEMPLATES) == 2
    assert match_digit(T5, TEMPLATES) == 5


def test_read_score_reads_multi_digit_number():
    assert read_score(_strip(T2, T5), TEMPLATES) == 25
    assert read_score(_strip(T5, T2, T2), TEMPLATES) == 522


def test_read_score_without_templates_is_none():
    assert read_score(_strip(T2), {}) is None
