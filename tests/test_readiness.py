from incubator.dataset import Label
from incubator.readiness import readiness, snapshot
from tests.conftest import make_synthetic


def test_ready_patchcore(tmp_path):
    ds = make_synthetic(tmp_path, n_normal=10, n_boxed=0, n_boxless=2)
    assert readiness(ds, "PatchCore") == []


def test_yolo_needs_boxes(tmp_path):
    ds = make_synthetic(tmp_path, n_normal=10, n_boxed=1, n_boxless=2)
    problems = readiness(ds, "YOLO")
    assert len(problems) == 1 and "2장 이상" in problems[0] and "현재 1장" in problems[0]


def test_no_dataset():
    assert readiness(None, "PatchCore") == ["데이터셋을 고르세요"]


def test_split_error_reported(tmp_path):
    ds = make_synthetic(tmp_path, n_normal=2)
    assert any("최소 3개" in p for p in readiness(ds, "PatchCore"))


def test_no_interference(tmp_path):
    ds = make_synthetic(tmp_path, n_normal=10, n_boxed=0)
    assert "간섭 이미지가 없습니다" in readiness(ds, "PatchCore")


def test_snapshot_isolated_from_edits(tmp_path):
    ds = make_synthetic(tmp_path, n_normal=10, n_boxed=2)
    s = snapshot(ds)
    ds.items[0].label = Label.UNLABELED
    ds.items[-1].boxes.clear()
    assert s.ds.items[0].label is Label.NORMAL and s.ds.items[-1].boxes
    assert len(s.split.train) + len(s.split.thresh) + len(s.split.test) == 10
