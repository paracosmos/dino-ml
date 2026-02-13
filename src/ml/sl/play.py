import time
import numpy as np
from mss import mss
import pyautogui
import torch
import os

from src.ml.dino.config import DinoEnvConfig
from src.ml.dino.preprocess import preprocess_obs_float
from src.ml.dino.action_spec import DinoAction
from src.ml.model.dino_sl_model import DinoSLModel


# =========================
# Safety / UX settings
# =========================
# pyautogui는 마우스를 화면 모서리로 옮기면 FailSafe로 예외가 나며 종료될 수 있음.
# 원치 않으면 False로. (디버깅 중에는 True 유지 추천)
pyautogui.FAILSAFE = True

# 키 입력 사이에 작은 딜레이(너무 빠르면 일부 환경에서 키가 씹히는 경우 방지)
pyautogui.PAUSE = 0.0


def tap_space():
    """짧게 Space 탭. (점프)"""
    pyautogui.keyDown("space")
    pyautogui.keyUp("space")


def duck_down(on: bool):
    """Down 키 누르기/떼기 (숙이기)"""
    if on:
        pyautogui.keyDown("down")
    else:
        pyautogui.keyUp("down")


def main():
    env = DinoEnvConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================
    # Model load
    # =========================
    model_path = "dino_sl_cnn.pt"
    if not os.path.exists(model_path):
        # 파일이 없으면 흔히 경로 문제임. (예: models/ 폴더로 옮겨둔 경우)
        print(f"[ERROR] model not found: {model_path}")
        print(" - dino_sl_cnn.pt 위치를 확인하거나 model_path를 수정하세요.")
        return

    model = DinoSLModel(n_actions=len(DinoAction)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # =========================
    # Capture loop timing
    # =========================
    sct = mss()
    dt = 1.0 / env.fps

    # =========================
    # Control state
    # =========================
    jump_cd = 0          # jump cooldown frames
    ducking = False      # 현재 down키 누르고 있는 상태

    # =========================
    # Threshold / stabilization
    # =========================
    # (1) JUMP 임계치: 점프가 너무 안 나가면 0.70 -> 0.60으로 낮춰보는 게 정석
    JUMP_TH = 0.70

    # (2) DUCK 히스테리시스: 누를 때/뗄 때 임계치를 다르게 해서 떨림 방지
    DUCK_ON_TH = 0.60
    DUCK_OFF_TH = 0.45

    # (3) 점프 후 쿨다운: 점프 스팸 방지. (스팸이면 12 -> 16)
    JUMP_COOLDOWN_FRAMES = 12

    # =========================
    # Debug logging
    # =========================
    # 확률/액션을 출력해서 "왜 안 뛰지?"를 바로 확인 가능
    DEBUG = True
    LOG_EVERY_SEC = 1.0
    last_log_t = time.time()

    print("Player start in 3 sec. Focus Chrome Dino window!")
    print(" - Chrome Dino 창을 클릭해서 포커스를 주고, 게임이 시작 상태인지 확인하세요.")
    print(" - 종료는 Ctrl+C (또는 pyautogui FailSafe: 마우스를 좌상단 모서리로 이동)")
    time.sleep(3)

    # 선택: 게임 시작을 확실히 하고 싶으면 아래 한 줄을 주석 해제
    # tap_space()

    try:
        while True:
            t0 = time.time()

            # =========================
            # Capture + preprocess
            # =========================
            frame = np.array(sct.grab(env.roi))[:, :, :3]
            x = preprocess_obs_float(frame, env)        # (1,1,H,W) float32
            xb = torch.from_numpy(x).to(device)

            # =========================
            # Inference
            # =========================
            with torch.no_grad():
                probs = torch.softmax(model(xb), dim=1)[0].cpu().numpy()

            # 주의: 모델이 3클래스로 학습되었다는 가정 (NOOP, JUMP, DUCK 순)
            p_noop, p_jump, p_duck = probs

            # =========================
            # Decide action (threshold + cooldown)
            # =========================
            action = DinoAction.NOOP

            # 점프 쿨다운 감소
            if jump_cd > 0:
                jump_cd -= 1

            # 점프 우선 로직
            if jump_cd == 0 and p_jump >= JUMP_TH:
                action = DinoAction.JUMP
            else:
                # DUCK은 히스테리시스로 상태 기반 제어가 더 안정적
                # 여기서 action을 DUCK으로 직접 두기보다는 ducking 상태를 바꿔서 처리
                pass

            # =========================
            # Execute action (stabilized)
            # =========================
            # (A) JUMP 실행
            if action == DinoAction.JUMP:
                if ducking:
                    duck_down(False)
                    ducking = False

                tap_space()
                jump_cd = JUMP_COOLDOWN_FRAMES

            # (B) DUCK 상태 업데이트 (히스테리시스)
            # - 점프가 아닌 경우에만 duck 상태를 업데이트하는 게 일반적으로 안정적
            if action != DinoAction.JUMP:
                if ducking:
                    # 떼는 기준(더 낮게)
                    if p_duck < DUCK_OFF_TH:
                        duck_down(False)
                        ducking = False
                else:
                    # 누르는 기준(더 높게)
                    if p_duck >= DUCK_ON_TH:
                        duck_down(True)
                        ducking = True

            # =========================
            # Debug log
            # =========================
            now = time.time()
            if DEBUG and (now - last_log_t) >= LOG_EVERY_SEC:
                print(
                    f"[PROB] noop={p_noop:.2f} jump={p_jump:.2f} duck={p_duck:.2f} | "
                    f"ducking={ducking} jump_cd={jump_cd}"
                )
                last_log_t = now

            # =========================
            # FPS control
            # =========================
            time.sleep(max(0.0, dt - (time.time() - t0)))

    except KeyboardInterrupt:
        print("\nStop player.")
    finally:
        # 종료 시 down키가 눌린 채로 남아있지 않도록 반드시 해제
        if ducking:
            duck_down(False)


if __name__ == "__main__":
    main()
