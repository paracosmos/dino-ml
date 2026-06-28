from src.ml.dino.config import DinoEnvConfig
from src.ml.dino.reward import compute_reward


def test_reward_alive_and_death():
    cfg = DinoEnvConfig()
    assert compute_reward(False, cfg) == cfg.reward_alive
    assert compute_reward(True, cfg) == cfg.reward_death


def test_reward_respects_config_override():
    cfg = DinoEnvConfig(reward_alive=2.0, reward_death=-50.0)
    assert compute_reward(False, cfg) == 2.0
    assert compute_reward(True, cfg) == -50.0
