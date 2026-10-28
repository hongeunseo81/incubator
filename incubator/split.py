"""정상 이미지를 학습 / 임계값용 / 평가로 그룹 단위 분할."""
import random
from dataclasses import dataclass

from incubator.dataset import Dataset, Item, Label, Pin


class SplitError(ValueError):
    pass


@dataclass
class Split:
    train: list[Item]
    thresh: list[Item]
    test: list[Item]


def split_normals(ds: Dataset, seed: int = 0, ratios=(0.6, 0.2, 0.2)) -> Split:
    by_group: dict[str, list[Item]] = {}
    for it in ds.usable(Label.NORMAL):
        by_group.setdefault(it.group, []).append(it)

    n = len(by_group)
    fixed_train, fixed_test, free = [], [], []
    for g, its in by_group.items():
        pins = {i.pin for i in its}
        if Pin.EVAL in pins:
            fixed_test.append(g)
        elif Pin.TRAIN in pins:
            fixed_train.append(g)
        else:
            free.append(g)

    free.sort()
    random.Random(seed).shuffle(free)
    n_thresh = max(1, round(n * ratios[1]))
    n_test = max(0, max(1, round(n * ratios[2])) - len(fixed_test))
    thresh_g = free[:n_thresh]
    test_g = fixed_test + free[n_thresh:n_thresh + n_test]
    train_g = fixed_train + free[n_thresh + n_test:]

    if not (train_g and thresh_g and test_g):
        raise SplitError(f"정상 그룹이 부족합니다: 최소 3개 필요 (현재 {n}개)")

    pick = lambda gs: [i for g in sorted(gs) for i in by_group[g]]
    return Split(pick(train_g), pick(thresh_g), pick(test_g))
