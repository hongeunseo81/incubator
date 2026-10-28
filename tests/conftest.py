import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def write_png(path: Path, img: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    buf.tofile(str(path))
    return str(path)


def make_mask(w: int, h: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    m[h // 4: h * 3 // 4, w // 4: w * 3 // 4] = 255
    return m


def make_synthetic(tmp_path: Path, n_normal=15, n_boxed=2, n_boxless=0, size=224):
    """격자 무늬+잡음 정상, 마스크 안에 흰 사각형(40px)이 있는 간섭. Dataset 반환."""
    from impostor.dataset import Dataset, Item, Label

    rng = np.random.default_rng(0)
    base = np.full((size, size), 110, np.uint8)
    base[::16, :] = 60
    base[:, ::16] = 60

    def noisy():
        g = np.clip(base + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    items = []
    for k in range(n_normal):
        items.append(Item(write_png(tmp_path / f"n{k:02d}" / "img.png", noisy()), f"n{k:02d}", Label.NORMAL))
    for k in range(n_boxed + n_boxless):
        img = noisy()
        x0 = size // 4 + 10 + (k % 3) * 20
        y0 = size // 4 + 10 + (k // 3) * 20
        img[y0:y0 + 40, x0:x0 + 40] = 250
        it = Item(write_png(tmp_path / f"i{k:02d}" / "img.png", img), f"i{k:02d}", Label.INTERFERENCE)
        if k < n_boxed:
            it.add_box((x0, y0, x0 + 40, y0 + 40))
        items.append(it)
    mask = write_png(tmp_path / "mask.png", make_mask(size, size))
    return Dataset("synthetic", mask, items)


@dataclass
class TmpImages:
    root: str
    mask: str
    any_image: str


@pytest.fixture
def tmp_images(tmp_path) -> TmpImages:
    """한글·연속 공백 경로에 회차 폴더 2개, 64x48 이미지."""
    root = tmp_path / "간섭 폴더"
    rng = np.random.default_rng(0)
    first = None
    for case in ("1_7C  0001", "1_7C  0002"):
        img = rng.integers(0, 255, (48, 64, 3), np.uint8)
        p = write_png(root / case / "cam_03_IMG_Texture_8Bit.png", img)
        write_png(root / case / "cam_01_IMG_Texture_8Bit.png", img)
        first = first or p
    mask = write_png(tmp_path / "마스크" / "mask.png", make_mask(64, 48))
    return TmpImages(str(root), mask, first)
