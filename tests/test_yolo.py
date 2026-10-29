import glob
import os
import tempfile

import pytest

from incubator.algorithms import REGISTRY
from incubator.algorithms.base import eval_items
from incubator.algorithms.yolo import FAST, TMP_PREFIX, Yolo
from incubator.split import split_normals
from tests.conftest import make_synthetic


def noop(*_):
    pass


def never():
    return False


def test_registered():
    assert REGISTRY["YOLO"] is Yolo


def test_yolo_skips_with_reason(tmp_path):
    ds = make_synthetic(tmp_path, n_boxed=1, n_boxless=2)
    r = Yolo().run(ds, split_normals(ds), noop, never)
    assert r.skipped and "2장 미만" in r.skipped and r.scores == {}


def tmp_dirs():
    return set(glob.glob(os.path.join(tempfile.gettempdir(), TMP_PREFIX + "*")))


@pytest.mark.slow
def test_yolo_scores_every_eval_item_and_cleans_up(tmp_path):
    ds = make_synthetic(tmp_path, n_boxed=4, n_boxless=2)
    split = split_normals(ds)
    interf, thresh, test = eval_items(ds, split)
    before = tmp_dirs()
    r = Yolo(dict(FAST, epochs=3)).run(ds, split, noop, never)
    assert set(r.scores) == {i.path for i in interf + thresh + test}
    assert all(0.0 <= s <= 1.0 for s in r.scores.values())
    assert r.skipped is None and r.sec_per_image > 0
    assert tmp_dirs() == before
    r.discard()


def test_yolo_rejects_size_mismatch(tmp_path):
    import numpy as np
    from pathlib import Path
    from tests.conftest import write_png
    ds = make_synthetic(tmp_path, n_boxed=2)
    boxed = [i for i in ds.items if i.boxes]
    write_png(Path(boxed[0].path), np.zeros((100, 100, 3), np.uint8))
    with pytest.raises(ValueError, match="마스크와 크기가 다른"):
        Yolo().run(ds, split_normals(ds), noop, never)


@pytest.mark.slow
def test_yolo_records_shifts_when_master_available(tmp_path):
    from tests.conftest import make_shifted
    ds, truth = make_shifted(tmp_path, n_normal=10, n_interf=3)
    split = split_normals(ds)
    r = Yolo(dict(FAST, epochs=2)).run(ds, split, noop, never)
    assert set(r.shifts) == set(r.scores)
    p = next(iter(r.shifts))
    assert abs(r.shifts[p][0] - truth[p][0]) <= 1.5


@pytest.mark.slow
def test_yolo_keeps_fold_weights(tmp_path):
    ds = make_synthetic(tmp_path, n_boxed=4)
    r = Yolo(dict(FAST, epochs=2, k=2)).run(ds, split_normals(ds), noop, never)
    assert len(r.model.weights) == 2 and all(os.path.isfile(w) for w in r.model.weights)
    assert r.model.imgsz == FAST["imgsz"]
    r.discard()
    assert not os.path.exists(r.model.root)


def test_sweep_stale_temp_removes_only_old_folders(tmp_path):
    import time
    from incubator.algorithms.yolo import MODEL_PREFIX, sweep_stale_temp
    old = tmp_path / (MODEL_PREFIX + "old")
    new = tmp_path / (TMP_PREFIX + "new")
    other = tmp_path / "keep_me"
    for d in (old, new, other):
        d.mkdir()
    past = time.time() - 2 * 86400
    os.utime(old, (past, past))
    os.utime(other, (past, past))
    sweep_stale_temp(str(tmp_path))
    assert not old.exists() and new.exists() and other.exists()
