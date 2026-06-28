from dataclasses import dataclass

@dataclass
class DinoEnvConfig:
    # -------- observation --------
    obs_size: int = 84
    n_stack: int = 4        # 연속 프레임 수 (속도/움직임 정보를 채널로 제공)
    fps: int = 30

    # -------- screen ROI --------
    roi_left: int = 3930
    roi_top: int = 871
    roi_width: int = 552
    roi_height: int = 149

    # -------- control --------
    key_jump: str = "space"
    key_duck: str = "down"
    frame_skip: int = 1        # 한 action 을 몇 프레임 유지할지 (M3, action repeat)

    # -------- timing --------
    step_sleep: float = 0.02
    reset_sleep: float = 0.4

    # -------- reward shaping (M7) --------
    reward_alive: float = 1.0
    reward_death: float = -100.0

    # -------- death detection (M4) --------
    dead_threshold: float = 140.0   # 중앙 crop 평균 밝기 임계 (이상이면 game-over)

    @property
    def roi(self):
        return {
            "left": self.roi_left,
            "top": self.roi_top,
            "width": self.roi_width,
            "height": self.roi_height,
        }

    @property
    def focus_point(self):
        """게임 클릭 위치"""
        return (self.roi_left + 20, self.roi_top + 20)
