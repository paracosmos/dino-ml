import json
import os


def save_run_config(config: dict, path: str) -> str:
    """실행 설정을 json 으로 저장한다 (M8 재현성).

    결과와 함께 어떤 하이퍼파라미터/ROI 로 돌렸는지를 남겨 재현 가능하게 한다.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2, sort_keys=True)
    return path


def load_run_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
