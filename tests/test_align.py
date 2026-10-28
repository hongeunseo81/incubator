import numpy as np
import pytest

from impostor.align import Aligner, reference_for, shift_box, shift_mask
from tests.conftest import make_mask, write_png


def noise(h=240, w=320, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (h // 8, w // 8), np.uint8)
    import cv2
    img = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def moved(img, dx, dy):
    return np.roll(np.roll(img, dy, axis=0), dx, axis=1)


def test_measures_known_shift():
    ref = noise()
    mask = make_mask(320, 240) > 0
    a = Aligner(ref, mask)
    dx, dy, resp = a.shift(moved(ref, 12, -7))
    assert abs(dx - 12) <= 1 and abs(dy + 7) <= 1 and resp > 0.3


def test_shift_mask_moves_region():
    m = np.zeros((20, 20), bool)
    m[5:10, 5:10] = True
    s = shift_mask(m, 3, -2)
    ys, xs = np.where(s)
    assert (ys.min(), xs.min(), ys.max(), xs.max()) == (3, 8, 7, 12)


def test_shift_box_keeps_size_and_stays_inside():
    assert shift_box((10, 50, 20, 80), 5, -3, (100, 100)) == (7, 47, 25, 85)
    assert shift_box((10, 50, 20, 80), 50, -30, (100, 100)) == (0, 40, 40, 100)


def test_reference_for_finds_master_texture(tmp_path):
    img = np.zeros((4, 4, 3), np.uint8)
    mask = write_png(tmp_path / "Master" / "JG" / "fastenLHCam_02_IMG_Kungfu.png", img)
    tex = write_png(tmp_path / "Master" / "JG" / "fastenLHCam_02_IMG_Texture_8Bit.png", img)
    assert reference_for(mask) == tex
    assert reference_for(str(tmp_path / "other_mask.png")) is None


def test_for_dataset_none_without_reference(tmp_path):
    from tests.conftest import make_synthetic
    assert Aligner.for_dataset(make_synthetic(tmp_path)) is None


def test_for_dataset_respects_off_switch(tmp_path):
    from tests.conftest import make_shifted
    ds, _ = make_shifted(tmp_path)
    assert Aligner.for_dataset(ds) is not None
    ds.align = False
    assert Aligner.for_dataset(ds) is None
