import cv2
import time
from stable_baselines3 import PPO

from src.ml.dino.dino_env import DinoEnv

def main():
    env = DinoEnv()
    model = PPO.load("ppo_dino")

    obs, _ = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        frame = obs[:, :, 0]
        frame = cv2.resize(frame, None, fx=5, fy=5)

        cv2.imshow("RL", frame)

        if terminated:
            obs, _ = env.reset()
            time.sleep(0.2)

        if cv2.waitKey(1) == ord("q"):
            break
