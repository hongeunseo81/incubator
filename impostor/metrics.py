"""평가 지표. 판정: 점수 >= 임계값 → NG(간섭)."""
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from sklearn.metrics import roc_auc_score

Point = tuple[int, int]  # x, y


@dataclass
class Metrics:
    missed: int
    n_interference: int
    false_pos: int
    n_normal_test: int
    auc: float | None
    hit: int
    n_hit_eval: int
    threshold: float


def auto_threshold(thresh_scores: Iterable[float]) -> float:
    """임계값용 정상 점수 최댓값 바로 위 — 그 세트에서 오검출 0건."""
    m = max(thresh_scores)
    return m + max(1e-6, 1e-3 * abs(m))


def location_hit(peak: Point | None, boxes: Sequence) -> bool:
    if peak is None:
        return False
    x, y = peak
    return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in boxes)


def evaluate(interf: Mapping[str, float], normal_test: Mapping[str, float], threshold: float,
             peaks: Mapping[str, Point | None], gt_boxes: Mapping[str, Sequence]) -> Metrics:
    missed = sum(s < threshold for s in interf.values())
    false_pos = sum(s >= threshold for s in normal_test.values())

    auc = None
    if interf and normal_test:
        y = [0] * len(normal_test) + [1] * len(interf)
        auc = float(roc_auc_score(y, list(normal_test.values()) + list(interf.values())))

    boxed = [p for p in interf if gt_boxes.get(p)]
    hit = sum(location_hit(peaks.get(p), gt_boxes[p]) for p in boxed)
    return Metrics(missed, len(interf), false_pos, len(normal_test), auc, hit, len(boxed), threshold)
