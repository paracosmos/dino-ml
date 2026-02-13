import time
import cv2
import numpy as np
import pyautogui
import gymnasium as gym
from gymnasium import spaces
from mss import mss

from src.ml.dino.config import DinoEnvConfig
from src.ml.dino.preprocess import preprocess_obs


class DinoEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, config: DinoEnvConfig | None = None):
        super().__init__()

        self.cfg = config or DinoEnvConfig()

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.cfg.obs_size, self.cfg.obs_size, 1),
            dtype=np.uint8,
        )

        self.sct = mss()
        self.monitor = self.cfg.roi

        pyautogui.FAILSAFE = False
        self.steps_alive = 0

    # ---------- capture ----------
    def _grab_bgr(self):
        img = np.array(self.sct.grab(self.monitor))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def _is_dead(self, gray):
        h, w = gray.shape
        roi = gray[int(h*0.2):int(h*0.45), int(w*0.35):int(w*0.65)]
        return roi.mean() > 140

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
        obs = preprocess_obs(bgr, self.cfg)

        return obs, {}

    def step(self, action):

        self._act(action)
        time.sleep(self.cfg.step_sleep)

        bgr = self._grab_bgr()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        obs = preprocess_obs(bgr, self.cfg)

        dead = self._is_dead(gray)

        reward = -100.0 if dead else 1.0

        if not dead:
            self.steps_alive += 1

        return obs, reward, dead, False, {"steps": self.steps_alive}

    def close(self):
        self.sct.close()
        super().close()
