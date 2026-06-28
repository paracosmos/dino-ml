import os
import time

import numpy as np
from mss import mss
import pyautogui
import torch

from src.ml.dino.config import DinoEnvConfig
from src.ml.dino.preprocess import preprocess_obs, stack_to_float
from src.ml.dino.framestack import FrameStacker
from src.ml.dino.action_spec import DinoAction
from src.ml.model.dino_sl_model import DinoSLModel
from src.ml.util.framebuffer import LatestFrameBuffer
from src.ml.util.capture import CaptureThread
from src.ml.util.latency import LatencyTracker


# 비동기 추론 데모 (plan M9): 캡처는 별도 스레드, 추론은 항상 최신 프레임만 소비.
# 컨트롤은 간결화를 위해 JUMP 임계치 + 쿨다운만 둔다(전체 히스테리시스는 sl/play.py 참고).
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0

JUMP_TH = 0.6
JUMP_COOLDOWN_FRAMES = 12
MODEL_PATH = "dino_sl_cnn.pt"


def main():
    env = DinoEnvConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] model not found: {MODEL_PATH}")
        return

    model = DinoSLModel(
        n_actions=len(DinoAction), obs_size=env.obs_size, in_channels=env.n_stack
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    sct = mss()
    roi = env.roi

    def grab():
        return np.array(sct.grab(roi))[:, :, :3]

    buffer = LatestFrameBuffer()
    capture = CaptureThread(grab, buffer)
    capture.start()

    stacker = FrameStacker(env.n_stack)
    latency = LatencyTracker()
    jump_cd = 0
    last_seq = -1

    print("Async player start in 3 sec. Focus Chrome Dino window!")
    time.sleep(3)

    try:
        while True:
            t0 = time.time()

            frame, seq = buffer.get()
            if frame is None or seq == last_seq:
                continue                      # 새 프레임이 없으면 대기
            last_seq = seq

            stacked = stacker.append(preprocess_obs(frame, env))
            x = torch.from_numpy(stack_to_float(stacked)).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
            p_jump = float(probs[int(DinoAction.JUMP)])

            if jump_cd > 0:
                jump_cd -= 1
            if jump_cd == 0 and p_jump >= JUMP_TH:
                pyautogui.press(env.key_jump)
                jump_cd = JUMP_COOLDOWN_FRAMES

            latency.add(time.time() - t0)
    except KeyboardInterrupt:
        print("\nstop. latency:", latency.summary(), "fps_est:", round(latency.fps_estimate(), 1))
    finally:
        capture.stop()
        capture.join(timeout=1.0)


if __name__ == "__main__":
    main()
