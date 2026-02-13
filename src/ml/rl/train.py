from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.ml.dino.config import DinoEnvConfig
from src.ml.dino.dino_env import DinoEnv
from src.ml.rl.policy import DinoFeatureExtractor


############################################
# env factory (멀티 프로세스용)
############################################
def make_env(rank: int):
    def _init():
        env = DinoEnv(DinoEnvConfig())
        env = Monitor(env)   # ⭐ reward / episode 길이 자동 기록
        return env
    return _init


############################################
# main
############################################
def main():

    # ⭐⭐⭐ 매우 중요 — RL 속도 3~6배 상승
    N_ENVS = 4

    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])

    ############################################
    # 평가용 env (best 모델 저장용)
    ############################################
    eval_env = SubprocVecEnv([make_env(999)])

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
        # 🔥 Dino에 매우 좋은 세팅
        ############################
        n_steps=4096,        # ← 중요 (2048보다 안정적)
        batch_size=256,
        learning_rate=2.5e-4,

        gamma=0.995,        # 오래 살아남는 정책 학습
        gae_lambda=0.98,

        clip_range=0.15,    # PPO 안정화
        ent_coef=0.01,      # 탐험 증가 ⭐⭐⭐

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

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        eval_freq=25_000,
        deterministic=True,
        render=False,
    )

    ############################################
    # train
    ############################################

    model.learn(
        total_timesteps=1_000_000,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    model.save("ppo_dino_final")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
