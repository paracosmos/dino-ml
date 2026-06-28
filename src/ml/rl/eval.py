import argparse

from stable_baselines3 import PPO

from src.ml.dino.dino_env import DinoEnv
from src.ml.util.metrics import summarize_episodes


def evaluate(model_path: str = "ppo_dino_final", n_episodes: int = 10) -> dict:
    """학습된 정책을 실게임에서 N 에피소드 돌려 리턴/생존길이를 집계한다 (M8).

    실화면이라 학습 루프와 동시에 평가할 수 없으므로, 학습이 끝난 뒤 또는
    별도 세션에서 호출하는 평가 전용 하니스.
    """
    env = DinoEnv()
    model = PPO.load(model_path)

    returns, lengths = [], []
    try:
        for _ in range(n_episodes):
            obs, _ = env.reset()
            ep_ret, ep_len = 0.0, 0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_ret += reward
                ep_len += 1
                if terminated or truncated:
                    break
            returns.append(ep_ret)
            lengths.append(ep_len)
    finally:
        env.close()

    return summarize_episodes(returns, lengths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ppo_dino_final")
    ap.add_argument("--episodes", type=int, default=10)
    args = ap.parse_args()
    print(evaluate(args.model, args.episodes))


if __name__ == "__main__":
    main()
