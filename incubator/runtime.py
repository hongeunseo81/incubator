"""내보낸 묶음만으로 채점한다. torch·ultralytics 없이 onnxruntime + OpenCV 만 쓴다.

현장 코드(C#)가 따라야 할 기준 구현이기도 하다 — manifest.json 의 값과 여기 순서대로 하면 툴과 같은 점수가 나온다.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from incubator.align import shift_box, shift_mask
from incubator.imageio import imread

FORMAT = "incubator-export"
VERSION = 1
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
BLUR_SIGMA = 4
PAD_VALUE = 114


def letterbox(img: np.ndarray, s: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """비율 유지 리사이즈 + 가운데 회색 패딩 (Ultralytics LetterBox 와 같은 반올림). → (1x3xSxS RGB/255, 배율, (좌, 상) 패딩)"""
    h, w = img.shape[:2]
    r = min(s / h, s / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (s - nw) / 2, (s - nh) / 2
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255
    return x, r, (left, top)


def unletterbox(xyxy: np.ndarray, scale: float, pad: tuple[int, int]) -> np.ndarray:
    out = xyxy.astype(np.float32).copy()
    out[:, [0, 2]] -= pad[0]
    out[:, [1, 3]] -= pad[1]
    return out / scale


def phase_shift(master_gray: np.ndarray, img: np.ndarray, region) -> tuple[float, float, float]:
    """입력 내용이 마스터에서 (dx, dy) 만큼 옮겨져 있다. → (dx, dy, 응답)"""
    y0, y1, x0, x1 = region
    ref = master_gray[y0:y1, x0:x1].astype(np.float32)
    cur = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    win = cv2.createHanningWindow((ref.shape[1], ref.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(ref, cur, win)
    return float(dx), float(dy), float(resp)


def patchcore_input(img: np.ndarray, crop, input_hw) -> np.ndarray:
    y0, y1, x0, x1 = crop
    rs = cv2.resize(img[y0:y1, x0:x1], (input_hw[1], input_hw[0]), interpolation=cv2.INTER_AREA)
    x = (cv2.cvtColor(rs, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - MEAN) / STD
    return x.transpose(2, 0, 1)[None].astype(np.float32)


def patchcore_score(dist: np.ndarray, crop, score_mask: np.ndarray, shape):
    """패치 거리 지도 → (점수, peak (x, y), 원본 크기 이상 지도)"""
    y0, y1, x0, x1 = crop
    up = cv2.resize(dist.astype(np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    full = np.zeros(shape[:2], np.float32)
    full[y0:y1, x0:x1] = cv2.GaussianBlur(up, (0, 0), BLUR_SIGMA)
    masked = np.where(score_mask, full, -np.inf)
    y, x = np.unravel_index(int(np.argmax(masked)), masked.shape)
    return float(masked[y, x]), (int(x), int(y)), full


def yolo_score(dets: np.ndarray, mask: np.ndarray):
    """원본 좌표 검출 [N, 6] (x1, y1, x2, y2, conf, cls) → (마스크와 겹치는 박스 최고 conf, peak, 박스들)"""
    boxes = []
    for x0, y0, x1, y1, c, _ in dets:
        if c <= 0:
            continue
        xa, ya, xb, yb = int(x0), int(y0), int(np.ceil(x1)), int(np.ceil(y1))
        if mask[max(ya, 0):yb, max(xa, 0):xb].any():
            boxes.append((xa, ya, xb, yb, float(c)))
    boxes.sort(key=lambda b: -b[4])
    if not boxes:
        return 0.0, None, []
    b = boxes[0]
    return b[4], ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2), boxes


def read_mask(path: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE) > 0


@dataclass
class Scored:
    score: float
    peak: tuple[int, int] | None
    boxes: list
    shift: tuple[float, float, float] | None


class Bundle:
    def __init__(self, root: Path, manifest: dict, sessions: list, masks: dict, master_gray: np.ndarray | None):
        self.root, self.manifest, self.sessions = root, manifest, sessions
        self.masks, self.master_gray = masks, master_gray

    @property
    def threshold(self) -> float:
        return self.manifest["threshold"]

    @classmethod
    def load(cls, path, providers=("CPUExecutionProvider",)) -> "Bundle":
        import onnxruntime as ort

        root = Path(path)
        m = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if m.get("format") != FORMAT or m.get("version") != VERSION:
            raise ValueError(f"Incubator 내보내기 묶음이 아닙니다: {root}")
        files = m["models"] if m["algorithm"] == "YOLO" else [m["model"]]
        sessions = [ort.InferenceSession(str(root / f), providers=list(providers)) for f in files]
        masks = {k: read_mask(root / m[k]) for k in ("mask", "score_mask") if k in m}
        master = None
        if m["align"]:
            master = cv2.cvtColor(imread(str(root / m["align"]["master"])), cv2.COLOR_BGR2GRAY)
        return cls(root, m, sessions, masks, master)

    def score(self, img: np.ndarray) -> Scored:
        m = self.manifest
        if list(img.shape[:2]) != m["image_size"]:
            raise ValueError(f"이미지 크기 {img.shape[:2]} 가 모델 {m['image_size']} 와 다릅니다")
        shift = None
        dx = dy = 0.0
        if self.master_gray is not None:
            shift = phase_shift(self.master_gray, img, m["align"]["region"])
            dx, dy = shift[0], shift[1]
        if m["algorithm"] == "PatchCore":
            crop = shift_box(m["crop"], dx, dy, img.shape)
            x = patchcore_input(img, crop, m["input_hw"])
            dist = self.sessions[0].run(None, {"image": x})[0]
            score, peak, _ = patchcore_score(dist, crop, shift_mask(self.masks["score_mask"], dx, dy), img.shape)
            return Scored(score, peak, [], shift)
        mask = shift_mask(self.masks["mask"], dx, dy)
        x, scale, pad = letterbox(img, m["imgsz"])
        best = (0.0, None, [])
        for sess in self.sessions:
            out = sess.run(None, {"images": x})[0][0]
            dets = out.copy()
            dets[:, :4] = unletterbox(out[:, :4], scale, pad)
            r = yolo_score(dets, mask)
            if r[0] > best[0]:
                best = r
        return Scored(best[0], best[1], best[2], shift)
