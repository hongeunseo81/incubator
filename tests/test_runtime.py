import cv2
import numpy as np
import pytest

from incubator.runtime import letterbox, patchcore_score, phase_shift, unletterbox, yolo_score


def test_letterbox_geometry():
    img = np.zeros((100, 200, 3), np.uint8)
    x, scale, pad = letterbox(img, 64)
    assert x.shape == (1, 3, 64, 64) and x.dtype == np.float32
    assert scale == pytest.approx(0.32) and pad == (0, 16)
    assert x[0, :, 0, 0] == pytest.approx([114 / 255] * 3)      # 위쪽 패딩
    assert x[0, :, 32, 32] == pytest.approx([0, 0, 0])           # 이미지 영역


def test_letterbox_converts_bgr_to_rgb():
    img = np.zeros((64, 64, 3), np.uint8)
    img[..., 0] = 255                                             # 파랑
    x, _, _ = letterbox(img, 64)
    assert x[0, 2, 10, 10] == pytest.approx(1.0) and x[0, 0, 10, 10] == 0


def test_unletterbox_inverse():
    boxes = np.array([[10.0, 20.0, 50.0, 40.0]])
    in_lb = boxes * 0.32 + np.array([0, 16, 0, 16])
    assert unletterbox(in_lb, 0.32, (0, 16)) == pytest.approx(boxes)


def test_patchcore_score_peak_in_mask():
    dist = np.zeros((4, 4), np.float32)
    dist[0, 0] = 100                                              # 마스크 밖
    dist[2, 2] = 10                                               # 마스크 안
    mask = np.zeros((200, 200), bool)
    mask[100:200, 100:200] = True
    score, (x, y), full = patchcore_score(dist, (40, 200, 40, 200), mask, (200, 200))
    assert full.shape == (200, 200) and full[:40].max() == 0
    assert mask[y, x] and score == pytest.approx(full[mask].max())
    assert score < full.max()                                     # 마스크 밖 큰 값은 무시


def test_yolo_score_ignores_boxes_outside_mask():
    mask = np.zeros((100, 100), bool)
    mask[50:, 50:] = True
    dets = np.array([[0, 0, 20, 20, 0.9, 0], [60, 60, 80, 80, 0.4, 0], [0, 0, 0, 0, 0, 0]], np.float32)
    score, peak, boxes = yolo_score(dets, mask)
    assert score == pytest.approx(0.4) and peak == (70, 70) and len(boxes) == 1
    assert yolo_score(dets[:1], mask)[0] == 0.0


def test_phase_shift_recovers_translation():
    rng = np.random.default_rng(0)
    tex = cv2.GaussianBlur(rng.uniform(0, 255, (200, 200)).astype(np.float32), (0, 0), 3)
    master = cv2.cvtColor(np.clip(tex, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    moved = cv2.warpAffine(master, np.float32([[1, 0, 7], [0, 1, -4]]), (200, 200))
    dx, dy, resp = phase_shift(cv2.cvtColor(master, cv2.COLOR_BGR2GRAY), moved, (40, 160, 40, 160))
    assert dx == pytest.approx(7, abs=0.5) and dy == pytest.approx(-4, abs=0.5) and resp > 0.05


def test_runtime_does_not_need_torch():
    import subprocess
    import sys
    code = "import sys, incubator.runtime; print('torch' in sys.modules or 'ultralytics' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"
