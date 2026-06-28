import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from src.ml.dino.config import DinoEnvConfig
from src.ml.dino.dino_env import DinoEnv
from src.ml.rl.policy import DinoFeatureExtractor


# SL 로 사전학습한 backbone 체크포인트. 존재하면 RL 백본을 warm start 한다. (plan M4)
SL_BACKBONE_PATH = "dino_sl_cnn.pt"


############################################
# env factory
############################################
def make_env():
    def _init():
        env = DinoEnv(DinoEnvConfig())
        env = Monitor(env)   # reward / episode 길이 자동 기록
        return env
    return _init


############################################
# main
############################################
def main():

    # 단일 물리 화면/키보드를 캡처·제어하므로 env 는 1개만 사용한다.
    # (병렬 env 는 같은 화면을 동시에 조작해 서로 간섭하므로 불가)
    # 같은 이유로 별도 eval_env 도 두지 않는다 — 평가 에피소드가 학습 중인
    # 게임 화면을 도중에 리셋/조작해 학습 신호를 망가뜨리기 때문이다.
    env = DummyVecEnv([make_env()])

    # SL backbone 이 있으면 warm start, 없으면 무작위 초기화로 학습
    pretrained = SL_BACKBONE_PATH if os.path.exists(SL_BACKBONE_PATH) else None
    if pretrained:
        print(f"[warmstart] SL backbone 으로 초기화: {pretrained}")
    else:
        print("[warmstart] SL backbone 없음 -> 무작위 초기화로 학습")

    policy_kwargs = dict(
        features_extractor_class=DinoFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=256,
            pretrained_backbone=pretrained,
        ),

        # CNN 이후 MLP
        net_arch=dict(
            pi=[256, 256],
            vf=[256, 256],
        ),
    )

    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,

        ############################
        # Dino에 매우 좋은 세팅
        ############################
        n_steps=4096,        # 중요 (2048보다 안정적)
        batch_size=256,
        learning_rate=2.5e-4,

        gamma=0.995,        # 오래 살아남는 정책 학습
        gae_lambda=0.98,

        clip_range=0.15,    # PPO 안정화
        ent_coef=0.01,      # 탐험 증가

        device="cuda",
        tensorboard_log="./ppo_dino_tensorboard/",
    )

    ############################################
    # callbacks
    ############################################

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="./checkpoints/",
        name_prefix="ppo_dino",
    )

    ############################################
    # train
    ############################################

    model.learn(
        total_timesteps=1_000_000,
        callback=[checkpoint_callback],
        progress_bar=True,
    )

    model.save("ppo_dino_final")

    env.close()


if __name__ == "__main__":
    main()
