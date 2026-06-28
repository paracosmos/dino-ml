import time
import cv2
import numpy as np
import pyautogui
import gymnasium as gym
from gymnasium import spaces
from mss import mss

from src.ml.dino.config import DinoEnvConfig
from src.ml.dino.preprocess import preprocess_obs
from src.ml.dino.framestack import FrameStacker
from src.ml.dino.signals import detect_dead
from src.ml.dino.reward import compute_reward


class DinoEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, config: DinoEnvConfig | None = None):
        super().__init__()

        self.cfg = config or DinoEnvConfig()

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.cfg.obs_size, self.cfg.obs_size, self.cfg.n_stack),
            dtype=np.uint8,
        )

        self.stacker = FrameStacker(self.cfg.n_stack)

        self.sct = mss()
        self.monitor = self.cfg.roi

        pyautogui.FAILSAFE = False
        self.steps_alive = 0

    # ---------- capture ----------
    def _grab_bgr(self):
        img = np.array(self.sct.grab(self.monitor))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def _act(self, action):

        if action == 1:
            pyautogui.press(self.cfg.key_jump)

        elif action == 2:
            pyautogui.press(self.cfg.key_duck)

    # ---------- gym ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps_alive = 0

        # focus
        pyautogui.click(*self.cfg.focus_point)
        time.sleep(0.1)

        pyautogui.press(self.cfg.key_jump)
        time.sleep(self.cfg.reset_sleep)

        bgr = self._grab_bgr()
        frame = preprocess_obs(bgr, self.cfg)
        obs = self.stacker.reset(frame)        # 첫 프레임으로 스택을 채움

        return obs, {}

    def step(self, action):

        # action repeat: 한 번 입력하고 frame_skip 프레임 동안 관측을 누적한다 (M3).
        self._act(action)

        total_reward = 0.0
        dead = False
        obs = None

        for _ in range(max(1, self.cfg.frame_skip)):
            time.sleep(self.cfg.step_sleep)

            bgr = self._grab_bgr()
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            obs = self.stacker.append(preprocess_obs(bgr, self.cfg))
            dead = detect_dead(gray, self.cfg)
            total_reward += compute_reward(dead, self.cfg)

            if dead:
                break

            self.steps_alive += 1

        return obs, total_reward, dead, False, {"steps": self.steps_alive}

    def close(self):
        self.sct.close()
        super().close()
