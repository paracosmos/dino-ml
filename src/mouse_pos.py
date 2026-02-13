import pyautogui
import time

print("마우스 좌표 출력 시작 (Ctrl+C 로 종료)")
time.sleep(1)

while True:
    x, y = pyautogui.position()
    print(f"x={x}, y={y}")
    time.sleep(1)

# x=3840, y=547  좌측상단
# x=3840, y=2145 좌측하단

# x=6399, y=547  우측상단
# x=6399, y=2146 우측하단


# x=3930, y=871  좌측상단
# x=3930, y=1020 좌측하단

# x=4482, y=871  우측상단
# x=4482, y=1020 우측하단


# capture_width = 4482 - 3930 = 552
# capture_height = 1020 - 871 = 149