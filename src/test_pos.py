import time
import cv2
import numpy as np
from mss import mss

from src.ml.dino.config import DinoEnvConfig
from src.ml.dino.preprocess import preprocess_obs

def main():
    env = DinoEnvConfig()
    sct = mss()

    print("3초 후 ROI를 1장 캡처합니다. Chrome Dino 창을 화면에 띄워두세요.")
    time.sleep(3)

    # ✅ raw capture (BGR)
    raw = np.array(sct.grab(env.roi))[:, :, :3]  # BGRA -> BGR
    cv2.imshow("RAW CAPTURE (BGR)", raw)
    cv2.imwrite("debug_raw.png", raw)
    print("debug_raw.png saved")

    # ✅ optional: preprocessed obs (84x84 grayscale)
    obs = preprocess_obs(raw, env)          # (84,84,1) uint8
    obs2d = obs[:, :, 0]                    # (84,84)
    cv2.imshow("OBS (84x84)", obs2d)
    cv2.imwrite("debug_obs.png", obs2d)
    print("debug_obs.png saved")

    print("Press any key on the image window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
