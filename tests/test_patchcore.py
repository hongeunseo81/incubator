import pytest

from incubator.algorithms.base import Cancelled, eval_items
from incubator.algorithms.patchcore import PatchCore
from incubator.metrics import location_hit
from incubator.split import split_normals
from tests.conftest import make_synthetic

pytestmark = pytest.mark.slow


def noop(*_):
    pass


def test_patchcore_flags_synthetic_blob(tmp_path):
    ds = make_synthetic(tmp_path)
    split = split_normals(ds)
    interf, thresh, test = eval_items(ds, split)
    r = PatchCore().run(ds, split, noop, lambda: False)

    assert set(r.scores) == {i.path for i in interf + thresh + test}
    normal_max = max(r.scores[i.path] for i in thresh + test)
    assert all(r.scores[i.path] > normal_max for i in interf)
    assert all(location_hit(r.peaks[i.path], i.boxes) for i in interf)
    assert r.maps[interf[0].path].shape == (224, 224)
    assert r.sec_per_image > 0 and r.skipped is None


def test_patchcore_cancel(tmp_path):
    ds = make_synthetic(tmp_path)
    with pytest.raises(Cancelled):
        PatchCore().run(ds, split_normals(ds), noop, lambda: True)


def test_patchcore_aligns_mask_per_image(tmp_path):
    from tests.conftest import make_shifted
    ds, truth = make_shifted(tmp_path)
    split = split_normals(ds)
    interf, thresh, test = eval_items(ds, split)
    r = PatchCore().run(ds, split, noop, lambda: False)
    assert len(r.shifts) == len(interf + thresh + test)
    for p, (dx, dy) in truth.items():
        if p in r.shifts:
            sx, sy, _ = r.shifts[p]
            assert abs(sx - dx) <= 1.5 and abs(sy - dy) <= 1.5
    normal_max = max(r.scores[i.path] for i in thresh + test)
    assert all(r.scores[i.path] > normal_max for i in interf)
    assert all(location_hit(r.peaks[i.path], i.boxes) for i in interf)


def test_patchcore_keeps_model(tmp_path):
    ds = make_synthetic(tmp_path)
    p = PatchCore()
    r = p.run(ds, split_normals(ds), noop, lambda: False)
    m = r.model
    assert m.bank.shape[1] == 1536 and len(m.bank) <= p.params["coreset"]
    assert m.input_hw[1] == p.params["width"] and m.input_hw[0] % 8 == 0
    assert m.score_mask.shape == (224, 224) and m.score_mask.dtype == bool
