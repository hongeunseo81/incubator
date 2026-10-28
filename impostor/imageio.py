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


PNG_SIGNATURE = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])   # PNG 파일 시그니처


def imsize(path: str) -> tuple[int, int]:
    """(h, w). PNG는 헤더(IHDR)만 읽는다 — 수백 장 크기 검사를 전체 디코딩 없이."""
    try:
        with open(path, "rb") as f:
            head = f.read(24)
    except OSError as e:
        raise ImageReadError(path) from e
    if head[:8] == PNG_SIGNATURE and head[12:16] == b"IHDR":
        return int.from_bytes(head[20:24], "big"), int.from_bytes(head[16:20], "big")
    return imread(path).shape[:2]


def load_mask(path: str) -> np.ndarray:
    """흰색(>0) = 검사 영역인 bool 마스크."""
    m = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise ImageReadError(path)
    return m > 0
