"""알고리즘별 파라미터 명세. 파라미터 화면은 이 목록으로 입력 폼을 만든다."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ParamSpec:
    key: str
    label: str
    default: int | str
    choices: tuple[str, ...] | None = None
    lo: int = 0
    hi: int = 0
    step: int = 1


PARAMS: dict[str, list[ParamSpec]] = {
    "PatchCore": [
        ParamSpec("width", "입력 폭 (px)", 640, lo=64, hi=2048, step=32),
        ParamSpec("coreset", "메모리 크기 (패치 수)", 20000, lo=1000, hi=100000, step=1000),
        ParamSpec("erode", "마스크 침식 (px)", 5, lo=0, hi=50),
    ],
    "YOLO": [
        ParamSpec("model", "모델", "yolov8n.pt", choices=("yolov8n.pt", "yolov8s.pt")),
        ParamSpec("epochs", "학습 epoch", 30, lo=1, hi=500),
        ParamSpec("k", "교차검증 겹 수", 3, lo=2, hi=10),
        ParamSpec("imgsz", "입력 크기 (px)", 640, lo=320, hi=1280, step=32),
    ],
}


def defaults(name: str) -> dict:
    return {s.key: s.default for s in PARAMS[name]}
