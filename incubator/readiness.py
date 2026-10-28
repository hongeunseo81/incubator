"""실행 전 준비 상태 검사와 실행 시점 스냅샷."""
import copy
from dataclasses import dataclass

from incubator.dataset import Dataset, Label
from incubator.split import Split, SplitError, split_normals

YOLO_MIN_BOXED = 2


def readiness(ds: Dataset | None, algo: str) -> list[str]:
    """비어 있으면 실행 준비 완료. 아니면 사용자에게 보여줄 문제 목록."""
    if ds is None:
        return ["데이터셋을 고르세요"]
    problems = []
    try:
        split_normals(ds)
    except SplitError as e:
        problems.append(str(e))
    interf = ds.usable(Label.INTERFERENCE)
    if not interf:
        problems.append("간섭 이미지가 없습니다")
    if algo == "YOLO":
        boxed = sum(bool(i.boxes) for i in interf)
        if boxed < YOLO_MIN_BOXED:
            problems.append(f"YOLO는 박스 있는 간섭이 {YOLO_MIN_BOXED}장 이상 필요합니다 (현재 {boxed}장)")
    return problems


@dataclass
class Snapshot:
    ds: Dataset
    split: Split


def snapshot(ds: Dataset) -> Snapshot:
    """실행 중 라벨을 고쳐도 결과가 섞이지 않도록 데이터셋을 깊은 복사해 고정한다."""
    frozen = copy.deepcopy(ds)
    return Snapshot(frozen, split_normals(frozen))
