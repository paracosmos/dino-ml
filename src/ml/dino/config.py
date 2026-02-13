from dataclasses import dataclass

@dataclass
class DinoEnvConfig:
    # -------- observation --------
    obs_size: int = 84
    fps: int = 30

    # -------- screen ROI --------
    roi_left: int = 3930
    roi_top: int = 871
    roi_width: int = 552
    roi_height: int = 149

    # -------- control --------
    key_jump: str = "space"
    key_duck: str = "down"

    # -------- timing --------
    step_sleep: float = 0.02
    reset_sleep: float = 0.4

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
