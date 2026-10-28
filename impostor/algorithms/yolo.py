"""YOLO 검출. 간섭 사례가 적어 박스 있는 간섭을 k-겹으로 나눠 교차검증한다.

이미지 점수 = 마스크와 겹치는 예측 박스 중 최고 신뢰도 (없으면 0).
- 박스 있는 간섭: 자기를 학습하지 않은 겹 모델의 점수
- 박스 없는 간섭: k개 모델 점수 평균 (모두 처음 보는 이미지)
- 정상: k개 모델 점수 최댓값 (정상 NG를 과소평가하지 않도록)
"""
import random
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from impostor.algorithms.params import defaults
from impostor.align import Aligner, shift_mask
from impostor.algorithms.base import AlgoResult, Cancelled, Progress, ShouldStop, check_stop, eval_items
from impostor.dataset import DATASET_DIR, Dataset, Item
from impostor.imageio import imread, imsize, load_mask
from impostor.split import Split

FAST = defaults("YOLO") | {"batch": 8}   # 빠른 모드 — 화면에 없는 batch 만 여기서 정한다
TMP_PREFIX = "impostor_yolo_"
WEIGHTS_DIR = DATASET_DIR.parent / "weights"
SEED = 0


def _write_image(img: np.ndarray, path: Path) -> None:
    ok, buf = cv2.imencode(".png", img)
    buf.tofile(str(path))


def _label_lines(item: Item, w: int, h: int) -> str:
    lines = []
    for x0, y0, x1, y1 in item.boxes:
        lines.append(f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
    return "\n".join(lines)


class Yolo:
    name = "YOLO"

    def __init__(self, params: dict | None = None):
        self.params = FAST | (params or {})
        self.device = 0 if torch.cuda.is_available() else "cpu"

    def _write_trainset(self, root: Path, items: list[Item]) -> Path:
        (root / "images" / "train").mkdir(parents=True)
        (root / "labels" / "train").mkdir(parents=True)
        for k, it in enumerate(items):
            img = imread(it.path)
            _write_image(img, root / "images" / "train" / f"{k}.png")
            (root / "labels" / "train" / f"{k}.txt").write_text(_label_lines(it, img.shape[1], img.shape[0]))
        yaml = root / "data.yaml"
        yaml.write_text(f"path: {root.as_posix()}\ntrain: images/train\nval: images/train\nnames:\n  0: foreign\n",
                        encoding="utf-8")
        return yaml

    def _train(self, yaml: Path, root: Path, should_stop: ShouldStop):
        from ultralytics import YOLO

        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        model = YOLO(str(WEIGHTS_DIR / self.params["model"]))

        def stop_cb(trainer):
            if should_stop():
                raise Cancelled()

        model.add_callback("on_train_epoch_end", stop_cb)
        model.train(data=str(yaml), epochs=self.params["epochs"], imgsz=self.params["imgsz"],
                    batch=self.params["batch"], device=self.device, project=str(root / "runs"), name="f",
                    exist_ok=True, val=False, plots=False, workers=0, verbose=False, seed=SEED)
        return model

    def _predict(self, model, img: np.ndarray, mask: np.ndarray):
        r = model.predict(img, conf=0.001, imgsz=self.params["imgsz"], device=self.device, verbose=False)[0]
        boxes = []
        for (x0, y0, x1, y1), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            xa, ya, xb, yb = int(x0), int(y0), int(np.ceil(x1)), int(np.ceil(y1))
            if mask[max(ya, 0):yb, max(xa, 0):xb].any():
                boxes.append((xa, ya, xb, yb, float(c)))
        boxes.sort(key=lambda b: -b[4])
        score = boxes[0][4] if boxes else 0.0
        peak = ((boxes[0][0] + boxes[0][2]) // 2, (boxes[0][1] + boxes[0][3]) // 2) if boxes else None
        return score, peak, boxes

    def run(self, ds: Dataset, split: Split, progress: Progress, should_stop: ShouldStop) -> AlgoResult:
        interf, thresh, test = eval_items(ds, split)
        boxed = [i for i in interf if i.boxes]
        boxless = [i for i in interf if not i.boxes]
        if len(boxed) < 2:
            return AlgoResult(self.name, skipped="박스 있는 간섭이 2장 미만이라 YOLO를 실행하지 않았습니다")

        mask = load_mask(ds.mask_path)
        for it in interf + thresh + test + split.train:   # 학습 전에 확인해야 빨리 실패한다
            if imsize(it.path) != mask.shape:
                raise ValueError(f"마스크와 크기가 다른 이미지입니다: {it.path}")
        aligner = Aligner.for_dataset(ds)       # 마스터 이미지가 있으면 이미지마다 마스크 위치를 맞춘다
        shifts: dict[str, tuple] = {}

        def mask_for(it, img):
            if aligner is None:
                return mask
            if it.path not in shifts:
                shifts[it.path] = aligner.shift(img)
            dx, dy, _ = shifts[it.path]
            return shift_mask(mask, dx, dy)

        k = min(self.params["k"], len(boxed))
        order = sorted(boxed, key=lambda i: i.path)
        random.Random(SEED).shuffle(order)
        folds = [order[f::k] for f in range(k)]

        # 경로 → 겹별 (점수, peak, 박스)
        per: dict[str, list[tuple[float, tuple | None, list]]] = {}
        n_pred, t_pred = 0, 0.0
        for f, fold in enumerate(folds):
            check_stop(should_stop)
            progress(f"YOLO {f + 1}/{k}겹 학습", f / k)
            root = Path(tempfile.mkdtemp(prefix=TMP_PREFIX))
            try:
                train = [i for i in boxed if i not in fold] + split.train
                model = self._train(self._write_trainset(root, train), root, should_stop)
                progress(f"YOLO {f + 1}/{k}겹 점수 계산", (f + 0.9) / k)
                for it in fold + boxless + thresh + test:
                    check_stop(should_stop)
                    img = imread(it.path)
                    t0 = time.perf_counter()
                    per.setdefault(it.path, []).append(self._predict(model, img, mask_for(it, img)))
                    t_pred += time.perf_counter() - t0
                    n_pred += 1
            finally:
                shutil.rmtree(root, ignore_errors=True)

        res = AlgoResult(self.name, shifts=shifts)
        for it in interf + thresh + test:
            preds = per[it.path]
            best = max(preds, key=lambda p: p[0])
            if it in boxed:
                score = preds[0][0]
            elif it in boxless:
                score = float(np.mean([p[0] for p in preds]))
            else:
                score = best[0]
            res.scores[it.path] = float(score)
            res.peaks[it.path] = best[1]
            res.pred_boxes[it.path] = best[2]
        res.sec_per_image = t_pred / max(1, n_pred)
        progress("YOLO 완료", 1.0)
        return res
