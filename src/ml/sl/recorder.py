import time
import numpy as np
from mss import mss
from pynput import keyboard
from dataclasses import dataclass
from collections import Counter, deque

from src.ml.dino.config import DinoEnvConfig
from src.ml.dino.preprocess import preprocess_obs
from src.ml.dino.action_spec import DinoAction


@dataclass
class RecorderCfg:
    noop_keep_prob: float = 0.1
    out_path: str = "dino_sl_dataset.npz"
    label_shift_frames: int = 3
    log_every_sec: float = 5.0

class KeyState:
    def __init__(self):
        self.space = False
        self.down = False

    def on_press(self, key):
        if key == keyboard.Key.space:
            self.space = True
        elif key == keyboard.Key.down:
            self.down = True

    def on_release(self, key):
        if key == keyboard.Key.space:
            self.space = False
        elif key == keyboard.Key.down:
            self.down = False

    def label(self) -> int:
        if self.space:
            return int(DinoAction.JUMP)
        if self.down:
            return int(DinoAction.DUCK)
        return int(DinoAction.NOOP)

def main():
    env = DinoEnvConfig()
    cfg = RecorderCfg()
    ks = KeyState()

    listener = keyboard.Listener(on_press=ks.on_press, on_release=ks.on_release)
    listener.start()

    sct = mss()
    obs_list, label_list = [], []
    dt = 1.0 / env.fps

    obs_buf = deque(maxlen=cfg.label_shift_frames + 1)
    label_buf = deque(maxlen=cfg.label_shift_frames + 1)
    last_log_t = time.time()

    print("Recording... (Ctrl+C to stop)")
    try:
        while True:
            t0 = time.time()

            frame = np.array(sct.grab(env.roi))[:, :, :3]
            obs = preprocess_obs(frame, env)
            y = ks.label()

            obs_buf.append(obs)
            label_buf.append(y)

            if len(obs_buf) == obs_buf.maxlen:
                shifted_obs = obs_buf[0]
                shifted_label = label_buf[-1]

                if shifted_label != DinoAction.NOOP or (np.random.rand() < cfg.noop_keep_prob):
                    obs_list.append(shifted_obs)
                    label_list.append(shifted_label)

            now = time.time()
            if now - last_log_t >= cfg.log_every_sec and len(label_list) > 0:
                cnt = Counter(label_list)
                total = len(label_list)
                noop = cnt.get(int(DinoAction.NOOP), 0)
                jump = cnt.get(int(DinoAction.JUMP), 0)
                duck = cnt.get(int(DinoAction.DUCK), 0)

                print(
                    f"[STAT] total={total} | "
                    f"NOOP={noop}({noop/total:.1%}) "
                    f"JUMP={jump}({jump/total:.1%}) "
                    f"DUCK={duck}({duck/total:.1%}) | "
                    f"noop_keep_prob={cfg.noop_keep_prob} "
                    f"label_shift={cfg.label_shift_frames}f"
                )
                last_log_t = now

            time.sleep(max(0.0, dt - (time.time() - t0)))

    except KeyboardInterrupt:
        print("\nStop recording.")
    finally:
        listener.stop()

    # ✅ finally 밖에서 처리 (return/저장 분기)
    if len(obs_list) == 0:
        print("No samples collected. (ROI/Focus/Key capture 확인)")
        return

    X = np.stack(obs_list, axis=0)
    y = np.array(label_list, dtype=np.int64)
    np.savez_compressed(cfg.out_path, obs=X, label=y)

    print(f"Saved: {cfg.out_path}")
    print(f"Samples: {len(y)}  (NOOP:{(y==0).sum()}, JUMP:{(y==1).sum()}, DUCK:{(y==2).sum()})")
    print(f"X shape: {X.shape}")

if __name__ == "__main__":
    main()
