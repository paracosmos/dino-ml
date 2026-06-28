from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from src.ml.dino.config import DinoEnvConfig
from src.ml.dino.dino_env import DinoEnv
from src.ml.rl.policy import DinoFeatureExtractor, load_backbone_weights
from src.ml.sl.train import MODEL_PATH as SL_BACKBONE_PATH   # SL 체크포인트 경로 단일 출처


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

    policy_kwargs = dict(
        features_extractor_class=DinoFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),

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
    # warm start (plan M4): SL backbone -> RL backbone
    # PPO build(ortho_init) 이후에 로드해야 직교 초기화에 덮어써지지 않는다.
    ############################################
    if load_backbone_weights(model.policy.features_extractor.cnn, SL_BACKBONE_PATH):
        print(f"[warmstart] SL backbone 로드 성공: {SL_BACKBONE_PATH}")
    else:
        print("[warmstart] SL backbone 없음/비호환 -> 무작위 초기화로 학습")

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
