"""이미지 읽기. 경로에 한글·연속 공백이 흔해서 cv2.imread 대신 이 모듈만 쓴다."""
import cv2
import numpy as np


class ImageReadError(OSError):
    def __init__(self, path: str):
        super().__init__(f"이미지를 읽을 수 없습니다: {path}")
        self.path = path


def imread(path: str) -> np.ndarray:
    """BGR uint8 이미지."""
    try:
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    except OSError as e:
        raise ImageReadError(path) from e
    if img is None:
        raise ImageReadError(path)
    return img


def imsize(path: str) -> tuple[int, int]:
    """(h, w)"""
    return imread(path).shape[:2]


def load_mask(path: str) -> np.ndarray:
    """흰색(>0) = 검사 영역인 bool 마스크."""
    m = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise ImageReadError(path)
    return m > 0
