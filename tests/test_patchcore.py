import pytest

from impostor.algorithms.base import Cancelled, eval_items
from impostor.algorithms.patchcore import PatchCore
from impostor.metrics import location_hit
from impostor.split import split_normals
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
