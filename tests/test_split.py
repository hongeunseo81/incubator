import pytest

from incubator.dataset import Dataset, Item, Label, Pin
from incubator.split import SplitError, split_normals


def make_ds(tmp_path, n_groups, per_group=2):
    items = []
    for g in range(n_groups):
        for k in range(per_group):
            p = tmp_path / f"g{g}_{k}.png"
            p.write_bytes(b"x")
            items.append(Item(str(p), f"g{g}", Label.NORMAL))
    return Dataset("t", "m.png", items)


def groups(items):
    return {i.group for i in items}


def test_groups_never_straddle(tmp_path):
    s = split_normals(make_ds(tmp_path, 10))
    sets = [groups(s.train), groups(s.thresh), groups(s.test)]
    assert sum(len(x) for x in sets) == 10
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])


def test_ratios_10_groups(tmp_path):
    s = split_normals(make_ds(tmp_path, 10))
    assert (len(groups(s.train)), len(groups(s.thresh)), len(groups(s.test))) == (6, 2, 2)


def test_pins(tmp_path):
    ds = make_ds(tmp_path, 10)
    ds.items[0].pin = Pin.TRAIN      # g0
    ds.items[3].pin = Pin.EVAL       # g1
    s = split_normals(ds)
    assert "g0" in groups(s.train) and "g1" in groups(s.test)
    assert len(groups(s.test)) == 2


def test_deterministic(tmp_path):
    ds = make_ds(tmp_path, 10)
    assert split_normals(ds) == split_normals(ds)


def test_unlabeled_and_interference_ignored(tmp_path):
    ds = make_ds(tmp_path, 10)
    ds.items[0].label = Label.UNLABELED
    ds.items[1].label = Label.INTERFERENCE
    s = split_normals(ds)
    used = s.train + s.thresh + s.test
    assert ds.items[0] not in used and ds.items[1] not in used


def test_too_few_groups_raises(tmp_path):
    with pytest.raises(SplitError, match="최소 3개"):
        split_normals(make_ds(tmp_path, 2))
