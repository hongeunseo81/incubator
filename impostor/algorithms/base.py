"""알고리즘 공통 형식."""
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from impostor.dataset import Dataset, Item, Label
from impostor.split import Split

Progress = Callable[[str, float], None]   # (단계 문구, 0~1)
ShouldStop = Callable[[], bool]


class Cancelled(Exception):
    pass


@dataclass
class AlgoResult:
    name: str
    scores: dict[str, float] = field(default_factory=dict)             # 경로 → 점수
    maps: dict[str, np.ndarray] = field(default_factory=dict)          # 경로 → float16 이상 지도
    pred_boxes: dict[str, list[tuple[int, int, int, int, float]]] = field(default_factory=dict)
    peaks: dict[str, tuple[int, int] | None] = field(default_factory=dict)
    shifts: dict[str, tuple[float, float, float]] = field(default_factory=dict)   # 경로 → (dx, dy, 신뢰도)
    sec_per_image: float = 0.0
    skipped: str | None = None                                          # 실행하지 않은 사유


class Algorithm(Protocol):
    name: str

    def run(self, ds: Dataset, split: Split, progress: Progress, should_stop: ShouldStop) -> AlgoResult: ...


def eval_items(ds: Dataset, split: Split) -> tuple[list[Item], list[Item], list[Item]]:
    """(간섭, 임계값용 정상, 평가 정상)"""
    return ds.usable(Label.INTERFERENCE), split.thresh, split.test


def check_stop(should_stop: ShouldStop) -> None:
    if should_stop():
        raise Cancelled()
