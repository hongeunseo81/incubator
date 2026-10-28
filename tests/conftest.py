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
