"""검증한 모델을 현장용 묶음(ONNX + manifest + 마스크·마스터 + report)으로 내보낸다.

묶음 형식은 docs/superpowers/specs/2026-10-29-model-export-design.md, 채점 순서는 incubator/runtime.py 참고.
"""
import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from incubator.algorithms.base import AlgoResult, Progress, eval_items
from incubator.align import Aligner, reference_for
from incubator.imageio import imread, load_mask
from incubator.metrics import Metrics
from incubator.readiness import Snapshot
from incubator.runtime import BLUR_SIGMA, FORMAT, MEAN, PAD_VALUE, STD, STRIDE, VERSION, Bundle, rect_shape

YOLO_CONF = 0.001     # 툴 채점과 같은 최소 신뢰도
OPSET = 17


@dataclass
class Verification:
    n: int
    mismatched: int      # 판정(점수 ≥ 임계값)이 달라진 장수
    max_diff: float
    passed: bool


def _write_png(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise OSError(f"PNG 저장 실패: {path}")
    buf.tofile(str(path))


def _bundle_dir(out_root: Path, ds_name: str, algo: str) -> Path:
    name = re.sub(r'[\\/:*?"<>|]', "_", f"{ds_name}_{algo}_{datetime.now():%Y%m%d-%H%M}")
    out = Path(out_root) / name
    k = 2
    while out.exists():
        out = Path(out_root) / f"{name}_{k}"
        k += 1
    out.mkdir(parents=True)
    return out


def _export_patchcore(model, out: Path) -> dict:
    import torch

    from incubator.algorithms.patchcore import PatchCoreGraph

    graph = PatchCoreGraph(torch.from_numpy(model.bank)).eval()
    h, w = model.input_hw
    torch.onnx.export(graph, torch.zeros(1, 3, h, w), str(out / "model.onnx"), opset_version=OPSET,
                      input_names=["image"], output_names=["distance"])
    _write_png(out / "score_mask.png", model.score_mask.astype(np.uint8) * 255)
    return {"model": "model.onnx", "score_mask": "score_mask.png", "crop": [int(v) for v in model.crop],
            "input_hw": list(model.input_hw), "input": "image", "output": "distance", "stride": 8,
            "mean": MEAN.tolist(), "std": STD.tolist(), "blur_sigma": BLUR_SIGMA}


def _export_yolo(model, out: Path, image_hw, progress: Progress) -> dict:
    """.pt 예측은 rect 레터박스를 쓴다 — ONNX 도 같은 고정 입력 크기로 내보내야 점수가 같다."""
    from ultralytics import YOLO

    input_hw = rect_shape(image_hw, model.imgsz)
    names = []
    for k, w in enumerate(model.weights):
        progress(f"YOLO {k + 1}/{len(model.weights)}겹 ONNX 변환", 0.1 + 0.4 * k / len(model.weights))
        onnx = YOLO(w).export(format="onnx", imgsz=list(input_hw), nms=True, conf=YOLO_CONF, verbose=False)
        names.append(f"fold{k + 1}.onnx")
        shutil.move(onnx, out / names[-1])
    return {"models": names, "imgsz": model.imgsz, "letterbox": "rect", "stride": STRIDE, "input_hw": list(input_hw),
            "input": "images", "output": "output0",
            "conf": YOLO_CONF, "pad_value": PAD_VALUE, "combine": "max"}


def _versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for pkg in ("torch", "torchvision", "ultralytics", "onnx", "onnxruntime", "opencv-python"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return out


def export_bundle(res: AlgoResult, snap: Snapshot, threshold: float, metrics: Metrics, out_root,
                  progress: Progress, params: dict | None = None) -> Path:
    if res.model is None:
        raise ValueError("내보낼 모델이 없습니다. 먼저 학습하세요.")
    ds = snap.ds
    progress("묶음 준비", 0.0)
    out = _bundle_dir(Path(out_root), ds.name, res.name)
    try:
        mask = load_mask(ds.mask_path)
        _write_png(out / "mask.png", mask.astype(np.uint8) * 255)
        align = None
        aligner = Aligner.for_dataset(ds)
        if aligner is not None:
            _write_png(out / "master.png", imread(reference_for(ds.mask_path)))
            align = {"master": "master.png", "region": [int(v) for v in aligner.region], "method": "phaseCorrelate"}
        manifest = {"format": FORMAT, "version": VERSION, "algorithm": res.name, "threshold": float(threshold),
                    "image_size": [int(v) for v in mask.shape], "align": align, "mask": "mask.png"}
        progress(f"{res.name} ONNX 변환", 0.1)
        if res.name == "PatchCore":
            manifest |= _export_patchcore(res.model, out)
        else:
            manifest |= _export_yolo(res.model, out, mask.shape, progress)
        (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        interf, thresh, test = eval_items(ds, snap.split)
        report = {"dataset": ds.name, "versions": _versions(), "algorithm": res.name, "created": datetime.now().isoformat(timespec="seconds"),
                  "params": params if params is not None else getattr(res.model, "params", {}),
                  "threshold": float(threshold), "metrics": asdict(metrics),
                  "counts": {"train": len(snap.split.train), "thresh": len(thresh), "test": len(test),
                             "interference": len(interf)},
                  "verification": None}
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)      # 반쯤 쓴 묶음을 남기지 않는다
        raise
    progress("내보내기 완료", 0.5)
    return out


def verify(bundle_dir, res: AlgoResult, snap: Snapshot, progress: Progress) -> Verification:
    """묶음만으로 다시 채점해 화면 점수와 비교한다. YOLO 간섭은 자기 겹 모델로만 채점했으므로 정상만 비교한다."""
    bundle = Bundle.load(bundle_dir)
    thr = bundle.threshold
    interf, thresh, test = eval_items(snap.ds, snap.split)
    items = (interf if res.name == "PatchCore" else []) + thresh + test
    mismatched, max_diff = 0, 0.0
    for k, it in enumerate(items):
        progress("내보낸 모델로 다시 채점", 0.5 + 0.5 * k / max(1, len(items)))
        got, want = bundle.score(imread(it.path)).score, res.scores[it.path]
        max_diff = max(max_diff, abs(got - want))
        mismatched += (got >= thr) != (want >= thr)
    v = Verification(len(items), mismatched, max_diff, mismatched == 0 and max_diff <= max(1e-3, 0.01 * abs(thr)))
    path = Path(bundle_dir) / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["verification"] = asdict(v)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    progress("검증 완료", 1.0)
    return v
