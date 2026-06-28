from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from src.ml.dino.config import DinoEnvConfig
from src.ml.dino.dino_env import DinoEnv
from src.ml.rl.policy import DinoFeatureExtractor, load_backbone_weights
from src.ml.sl.train import MODEL_PATH as SL_BACKBONE_PATH


def make_env():
    def _init():
        return Monitor(DinoEnv(DinoEnvConfig()))
    return _init


def warm_start(model) -> bool:
    # DQN 은 q_net / q_net_target 두 네트워크가 각자 feature extractor 를 가진다.
    # build(직교 초기화) 이후 q_net 의 backbone 을 SL 가중치로 채우고,
    # target net 은 q_net 전체를 복사해 동기화한다(정상적인 DQN 초기 상태 유지).
    q_net = getattr(model.policy, "q_net", None)
    fe = getattr(q_net, "features_extractor", None) if q_net is not None else None
    if fe is None or not hasattr(fe, "cnn"):
        return False

    if not load_backbone_weights(fe.cnn, SL_BACKBONE_PATH):
        return False

    target = getattr(model.policy, "q_net_target", None)
    if target is not None:
        target.load_state_dict(q_net.state_dict())
    return True


def main():
    # 단일 물리 화면 -> env 1개. off-policy(DQN)는 replay 로 경험을 재사용해
    # on-policy(PPO)보다 표본효율이 좋아 실화면 학습에 유리하다 (plan M7).
    env = DummyVecEnv([make_env()])

    policy_kwargs = dict(
        features_extractor_class=DinoFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[256, 256],
    )

    model = DQN(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=64,
        gamma=0.995,
        train_freq=4,
        target_update_interval=2_000,
        exploration_fraction=0.2,
        exploration_final_eps=0.02,
        learning_rate=1e-4,
        device="cuda",
        tensorboard_log="./dqn_dino_tensorboard/",
    )

    if warm_start(model):
        print(f"[warmstart] SL backbone 로드 성공: {SL_BACKBONE_PATH}")
    else:
        print("[warmstart] SL backbone 없음/비호환 -> 무작위 초기화로 학습")

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="./checkpoints_dqn/",
        name_prefix="dqn_dino",
    )

    model.learn(
        total_timesteps=1_000_000,
        callback=[checkpoint_callback],
        progress_bar=True,
    )

    model.save("dqn_dino_final")
    env.close()


if __name__ == "__main__":
    main()
