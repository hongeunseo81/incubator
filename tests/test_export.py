import json

import pytest

from incubator.algorithms.base import eval_items
from incubator.algorithms.patchcore import PatchCore
from incubator.algorithms.yolo import FAST, Yolo
from incubator.export import export_bundle, verify
from incubator.metrics import auto_threshold, evaluate
from incubator.readiness import snapshot
from tests.conftest import make_shifted, make_synthetic

pytestmark = pytest.mark.slow


def noop(*_):
    pass


def run_and_export(ds, algo, out_root):
    snap = snapshot(ds)
    res = algo.run(snap.ds, snap.split, noop, lambda: False)
    interf, thresh, test = eval_items(snap.ds, snap.split)
    thr = auto_threshold(res.scores[i.path] for i in thresh)
    m = evaluate({i.path: res.scores[i.path] for i in interf}, {i.path: res.scores[i.path] for i in test},
                 thr, res.peaks, {i.path: i.boxes for i in interf})
    out = export_bundle(res, snap, thr, m, out_root, noop)
    return snap, res, thr, out


def test_export_patchcore_roundtrip(tmp_path):
    ds, _ = make_shifted(tmp_path)
    snap, res, thr, out = run_and_export(ds, PatchCore(), tmp_path / "내보내기")
    assert out.parent == tmp_path / "내보내기" and out.name.startswith("shifted_PatchCore_")
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert man["format"] == "incubator-export" and man["version"] == 1 and man["algorithm"] == "PatchCore"
    assert man["threshold"] == thr and man["image_size"] == [224, 224]
    assert man["align"]["master"] == "master.png" and len(man["align"]["region"]) == 4
    assert man["input_hw"] == list(res.model.input_hw) and man["crop"] == list(res.model.crop)
    for f in ("model.onnx", "mask.png", "score_mask.png", "master.png", "report.json"):
        assert (out / f).is_file()

    v = verify(out, res, snap, noop)
    interf, thresh, test = eval_items(snap.ds, snap.split)
    assert v.n == len(interf + thresh + test) and v.mismatched == 0 and v.passed
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["verification"]["passed"] and report["metrics"]["n_interference"] == len(interf)


def test_export_without_align(tmp_path):
    ds, _ = make_shifted(tmp_path)
    ds.align = False
    snap, res, _, out = run_and_export(ds, PatchCore(), tmp_path / "out")
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert man["align"] is None and not (out / "master.png").exists()
    assert verify(out, res, snap, noop).passed


def test_export_yolo_roundtrip(tmp_path):
    ds = make_synthetic(tmp_path, n_boxed=4)
    snap, res, _, out = run_and_export(ds, Yolo(dict(FAST, epochs=3, k=2)), tmp_path / "out")
    try:
        man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert man["algorithm"] == "YOLO" and man["models"] == ["fold1.onnx", "fold2.onnx"]
        assert all((out / f).is_file() for f in man["models"])
        v = verify(out, res, snap, noop)
        assert v.n == len(snap.split.thresh + snap.split.test) and v.passed
    finally:
        res.discard()
