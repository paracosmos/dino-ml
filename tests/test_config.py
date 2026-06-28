from src.ml.dino.config import DinoEnvConfig


def test_roi_keys_match_fields():
    cfg = DinoEnvConfig()
    roi = cfg.roi
    assert set(roi) == {"left", "top", "width", "height"}
    assert roi["left"] == cfg.roi_left
    assert roi["width"] == cfg.roi_width
    assert roi["height"] == cfg.roi_height


def test_focus_point_inside_roi():
    cfg = DinoEnvConfig()
    fx, fy = cfg.focus_point
    assert cfg.roi_left <= fx <= cfg.roi_left + cfg.roi_width
    assert cfg.roi_top <= fy <= cfg.roi_top + cfg.roi_height
