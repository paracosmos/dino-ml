from src.ml.util.latency import LatencyTracker
from src.ml.util.metrics import summarize_episodes
from src.ml.util.experiment import save_run_config, load_run_config
from src.ml.util.framebuffer import LatestFrameBuffer


def test_latency_summary_and_fps():
    t = LatencyTracker()
    for dt in (0.01, 0.02, 0.03):
        t.add(dt)
    s = t.summary()
    assert s["count"] == 3
    assert abs(s["mean"] - 0.02) < 1e-9
    assert s["min"] == 0.01 and s["max"] == 0.03
    assert abs(t.fps_estimate() - 50.0) < 1e-9


def test_latency_empty():
    t = LatencyTracker()
    assert t.summary()["count"] == 0
    assert t.fps_estimate() == 0.0


def test_summarize_episodes():
    s = summarize_episodes([10.0, 20.0, 30.0], [5, 7, 9])
    assert s["episodes"] == 3
    assert abs(s["return"]["mean"] - 20.0) < 1e-9
    assert s["length"]["min"] == 5 and s["length"]["max"] == 9


def test_summarize_episodes_empty():
    s = summarize_episodes([], [])
    assert s["episodes"] == 0
    assert s["return"]["mean"] == 0.0


def test_experiment_config_round_trip(tmp_path):
    cfg = {"lr": 0.001, "n_stack": 4, "name": "run-a"}
    path = save_run_config(cfg, str(tmp_path / "sub" / "run.json"))
    assert load_run_config(path) == cfg


def test_framebuffer_get_before_put():
    fb = LatestFrameBuffer()
    assert fb.get() == (None, 0)


def test_framebuffer_keeps_latest_and_counts():
    fb = LatestFrameBuffer()
    fb.put("a")
    fb.put("b")
    assert fb.get() == ("b", 2)
