from src.ml.dino.config import DinoEnvConfig


def compute_reward(dead: bool, cfg: DinoEnvConfig) -> float:
    """프레임당 보상: 생존이면 reward_alive, 죽으면 reward_death (M7 reward shaping).

    보상 스케일을 config 로 빼서 실험(생존/죽음 균형 튜닝)을 쉽게 한다.
    """
    return cfg.reward_death if dead else cfg.reward_alive
