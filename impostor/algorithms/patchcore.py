"""PatchCore 이상탐지. C:\\DATA\\EV\\FEM\\clip\\anomaly\\code\\patchcore.py 를 이식.

정상 영상만으로 패치 특징 메모리를 만들고, 각 패치의 최근접 거리로 이상 지도를 만든다.
점수 = 침식한 마스크 안 이상 지도 최댓값.
"""
import random
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from impostor.algorithms.params import defaults
from impostor.align import Aligner, shift_box, shift_mask
from impostor.algorithms.base import AlgoResult, Progress, ShouldStop, check_stop, eval_items
from impostor.dataset import Dataset
from impostor.imageio import imread, load_mask
from impostor.split import Split

BANK_PER_IMAGE = 2000       # 영상당 랜덤 추출 패치 수 (coreset 전)
BOX_PAD = 20
BLUR_SIGMA = 4
SEED = 0

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Extractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        net = torchvision.models.wide_resnet50_2(weights="IMAGENET1K_V1")
        self.stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.l2, self.l3 = net.layer2, net.layer3

    @torch.no_grad()
    def forward(self, x):
        f2 = self.l2(self.stem(x))
        f3 = self.l3(f2)
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        f3 = F.interpolate(F.avg_pool2d(f3, 3, 1, 1), size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], 1)  # B x 1536 x h x w (stride 8)


def mask_geometry(mask: np.ndarray, erode: int):
    ys, xs = np.where(mask)
    box = (max(ys.min() - BOX_PAD, 0), min(ys.max() + BOX_PAD, mask.shape[0]),
           max(xs.min() - BOX_PAD, 0), min(xs.max() + BOX_PAD, mask.shape[1]))
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((2 * erode + 1,) * 2, np.uint8)) > 0
    return box, eroded


class PatchCore:
    name = "PatchCore"

    def __init__(self, params: dict | None = None):
        # width: 마스크 bbox 를 이 폭으로 리사이즈 / coreset: 최종 메모리 크기 / erode: 마스크 경계 오정렬 완화(px)
        self.params = defaults(self.name) | (params or {})
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"

    def _prep(self, img, box):
        y0, y1, x0, x1 = box
        crop = img[y0:y1, x0:x1]
        h = max(8, int(round(crop.shape[0] * self.params["width"] / crop.shape[1] / 8)) * 8)
        rs = cv2.resize(crop, (self.params["width"], h), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(cv2.cvtColor(rs, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0) / 255
        return ((t - MEAN) / STD).to(self.dev)

    def _coreset(self, feats, n):
        """greedy k-center 로 n 개 대표 패치 선택 (128차원 랜덤 투영에서 거리 계산)."""
        if len(feats) <= n:
            return feats
        proj = feats @ torch.randn(feats.shape[1], 128, device=self.dev) / 128 ** 0.5
        sel = [random.randrange(len(feats))]
        mind = torch.cdist(proj, proj[sel[-1]:sel[-1] + 1]).squeeze(1)
        for _ in range(n - 1):
            i = int(torch.argmax(mind))
            sel.append(i)
            mind = torch.minimum(mind, torch.cdist(proj, proj[i:i + 1]).squeeze(1))
        return feats[sel]

    def _score_map(self, model, bank, x):
        f = model(x)[0]
        c, h, w = f.shape
        q = f.permute(1, 2, 0).reshape(-1, c)
        d = torch.cat([torch.cdist(q[i:i + 4096], bank).min(1).values for i in range(0, len(q), 4096)])
        return d.reshape(h, w).cpu().numpy()

    @staticmethod
    def _to_full(amap, box, shape):
        y0, y1, x0, x1 = box
        up = cv2.resize(amap, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        up = cv2.GaussianBlur(up, (0, 0), BLUR_SIGMA)
        full = np.zeros(shape[:2], np.float32)
        full[y0:y1, x0:x1] = up
        return full

    def _read(self, path, mask):
        img = imread(path)
        if img.shape[:2] != mask.shape:
            raise ValueError(f"마스크와 크기가 다른 이미지입니다: {path}")
        return img

    def run(self, ds: Dataset, split: Split, progress: Progress, should_stop: ShouldStop) -> AlgoResult:
        random.seed(SEED)
        torch.manual_seed(SEED)
        check_stop(should_stop)
        mask = load_mask(ds.mask_path)
        box, eroded = mask_geometry(mask, self.params["erode"])
        aligner = Aligner.for_dataset(ds)       # 마스터 이미지가 있으면 이미지마다 위치를 맞춘다

        def placed(img):
            """이 이미지에 맞게 옮긴 (특징 추출 영역, 점수 마스크, 측정값)"""
            if aligner is None:
                return box, eroded, None
            s = aligner.shift(img)
            return shift_box(box, s[0], s[1], img.shape), shift_mask(eroded, s[0], s[1]), s
        progress("PatchCore 모델 준비", 0.0)
        model = Extractor().to(self.dev).eval()

        feats = []
        for k, it in enumerate(split.train):
            check_stop(should_stop)
            progress("PatchCore 정상 특징 수집", 0.5 * k / len(split.train))
            img = self._read(it.path, mask)
            f = model(self._prep(img, placed(img)[0]))[0]
            f = f.permute(1, 2, 0).reshape(-1, f.shape[0])
            feats.append(f[torch.randperm(len(f), device=self.dev)[:BANK_PER_IMAGE]])
        progress("PatchCore 메모리 축약", 0.5)
        bank = self._coreset(torch.cat(feats), self.params["coreset"])
        del feats

        interf, thresh, test = eval_items(ds, split)
        targets = interf + thresh + test
        res = AlgoResult(self.name)
        t0 = time.perf_counter()
        for k, it in enumerate(targets):
            check_stop(should_stop)
            progress("PatchCore 점수 계산", 0.5 + 0.5 * k / len(targets))
            img = self._read(it.path, mask)
            b, m, s = placed(img)
            full = self._to_full(self._score_map(model, bank, self._prep(img, b)), b, img.shape)
            masked = np.where(m, full, -np.inf)
            if s is not None:
                res.shifts[it.path] = s
            y, x = np.unravel_index(int(np.argmax(masked)), masked.shape)
            res.scores[it.path] = float(masked[y, x])
            res.peaks[it.path] = (int(x), int(y))
            res.maps[it.path] = full.astype(np.float16)
        res.sec_per_image = (time.perf_counter() - t0) / max(1, len(targets))
        progress("PatchCore 완료", 1.0)
        return res
