from src.ml.dino.roi import roi_from_window


def test_roi_from_window_basic():
    assert roi_from_window(10, 20, 300, 200) == {
        "left": 10, "top": 20, "width": 300, "height": 200,
    }


def test_roi_from_window_inset_shrinks_both_sides():
    assert roi_from_window(0, 0, 100, 80, inset=5) == {
        "left": 5, "top": 5, "width": 90, "height": 70,
    }


def test_roi_from_window_inset_clamps_to_zero():
    roi = roi_from_window(0, 0, 4, 4, inset=10)
    assert roi["width"] == 0 and roi["height"] == 0
