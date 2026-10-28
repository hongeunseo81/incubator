"""이미지별 위치 보정. 부품이 놓인 위치가 마스터와 수십 px씩 달라서(FEM LH 중앙값 14px, 최대 38px)
마스터 마스크를 그대로 쓰면 경계가 판 바깥에 걸친다. 마스터 이미지와의 평행이동을 재서 마스크를 옮겨 쓴다.
"""
from pathlib import Path

import cv2
import numpy as np

from incubator.dataset import Dataset
from incubator.imageio import imread, load_mask

PAD = 40               # 위치 측정에 쓰는 영역 = 마스크 bbox + PAD
LOW_CONFIDENCE = 0.05  # phaseCorrelate 응답이 이보다 낮으면 측정을 믿기 어렵다 (이물이 크게 가린 경우 등)

Shift = tuple[float, float, float]   # dx, dy, 신뢰도(응답)


def reference_for(mask_path: str) -> str | None:
    """마스터 마스크(…_IMG_Kungfu.png) 옆의 같은 카메라 마스터 이미지(…_IMG_Texture_8Bit.png)."""
    p = Path(mask_path)
    if not p.name.endswith("_IMG_Kungfu.png"):
        return None
    tex = p.with_name(p.name.replace("_IMG_Kungfu.png", "_IMG_Texture_8Bit.png"))
    return str(tex) if tex.is_file() else None


def shift_mask(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    out = cv2.warpAffine(mask.astype(np.uint8), m, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST)
    return out > 0


def shift_box(box, dx: float, dy: float, shape) -> tuple[int, int, int, int]:
    """(y0, y1, x0, x1) 을 옮기되 크기는 유지하고 이미지 안으로 밀어 넣는다."""
    y0, y1, x0, x1 = box
    h, w = y1 - y0, x1 - x0
    ny0 = min(max(int(round(y0 + dy)), 0), shape[0] - h)
    nx0 = min(max(int(round(x0 + dx)), 0), shape[1] - w)
    return ny0, ny0 + h, nx0, nx0 + w


class Aligner:
    def __init__(self, ref_bgr: np.ndarray, mask: np.ndarray):
        ys, xs = np.where(mask)
        self.region = (max(ys.min() - PAD, 0), min(ys.max() + PAD, mask.shape[0]),
                       max(xs.min() - PAD, 0), min(xs.max() + PAD, mask.shape[1]))
        self.shape = mask.shape
        self.ref = self._gray(ref_bgr)
        self.win = cv2.createHanningWindow((self.ref.shape[1], self.ref.shape[0]), cv2.CV_32F)

    @classmethod
    def for_dataset(cls, ds: Dataset) -> "Aligner | None":
        ref = reference_for(ds.mask_path) if ds.align else None
        if ref is None:
            return None
        try:
            mask = load_mask(ds.mask_path)
            img = imread(ref)
        except OSError:
            return None
        if img.shape[:2] != mask.shape or not mask.any():
            return None
        return cls(img, mask)

    def _gray(self, bgr: np.ndarray) -> np.ndarray:
        y0, y1, x0, x1 = self.region
        return cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)

    def shift(self, bgr: np.ndarray) -> Shift:
        """이 이미지의 내용이 마스터에서 (dx, dy) 만큼 옮겨져 있다."""
        if bgr.shape[:2] != self.shape:
            return 0.0, 0.0, 0.0
        (dx, dy), resp = cv2.phaseCorrelate(self.ref, self._gray(bgr), self.win)
        return float(dx), float(dy), float(resp)


def describe(s: Shift | None) -> str:
    """화면 표시용: '위치 보정 dx +12 · dy -5 (신뢰도 0.31)'"""
    if s is None:
        return "위치 보정 없음 (마스터 이미지를 찾지 못함)"
    dx, dy, resp = s
    text = f"위치 보정 dx {dx:+.0f} · dy {dy:+.0f} (신뢰도 {resp:.2f})"
    if resp < LOW_CONFIDENCE:
        text += " — 측정 불확실: 이물이 크게 가렸을 수 있음"
    return text
