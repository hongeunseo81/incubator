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
    help: str = ""


PARAMS: dict[str, list[ParamSpec]] = {
    "PatchCore": [
        ParamSpec("width", "입력 폭 (px)", 640, lo=64, hi=2048, step=32,
                  help="사진을 얼마나 크게 놓고 볼지. 작은 이물을 놓치면 ↑ (느려짐)"),
        ParamSpec("coreset", "메모리 크기 (패치 수)", 20000, lo=1000, hi=100000, step=1000,
                  help="기억해 두는 정상 조각 수. 정상인데 NG(오검출)가 많으면 ↑ (느려짐)"),
        ParamSpec("erode", "마스크 침식 (px)", 5, lo=0, hi=50,
                  help="검사 영역 가장자리를 깎는 폭. 판 테두리에서 오검출이 나면 ↑, 가장자리 이물을 놓치면 ↓"),
    ],
    "YOLO": [
        ParamSpec("model", "모델", "yolov8n.pt", choices=("yolov8n.pt", "yolov8s.pt"),
                  help="n: 작고 빠름 · s: 크고 정확함"),
        ParamSpec("epochs", "학습 epoch", 30, lo=1, hi=500,
                  help="몇 번 반복해서 학습할지. 놓침이 많으면 ↑ (오래 걸림)"),
        ParamSpec("k", "교차검증 겹 수", 3, lo=2, hi=10,
                  help="간섭 사진을 몇 묶음으로 나눠 번갈아 학습·채점할지. ↑ 하면 학습 횟수도 늘어남"),
        ParamSpec("imgsz", "입력 크기 (px)", 640, lo=320, hi=1280, step=32,
                  help="사진을 얼마나 크게 놓고 볼지. 작은 이물을 놓치면 ↑ (느려짐)"),
    ],
}


def defaults(name: str) -> dict:
    return {s.key: s.default for s in PARAMS[name]}
