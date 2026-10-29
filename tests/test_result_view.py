import time

import pytest

from incubator.algorithms.base import AlgoResult, eval_items
from incubator.dataset import Label
from incubator.ui import result_view as rv
from tests.conftest import make_synthetic


class FakeAlgo:
    """간섭 [3.0, 1.0], 임계값용 정상 2.0, 평가 정상 0.5"""
    name = "Fake"
    calls = 0
    delay = 0.0

    def __init__(self, params=None):
        self.params = params

    def run(self, ds, split, progress, should_stop):
        type(self).calls += 1
        time.sleep(self.delay)
        interf, thresh, test = eval_items(ds, split)
        r = AlgoResult(self.name)
        for k, it in enumerate(interf):
            r.scores[it.path] = 3.0 if k == 0 else 1.0
            r.peaks[it.path] = None
        for it in thresh:
            r.scores[it.path] = 2.0
        for it in test:
            r.scores[it.path] = 0.5
        r.sec_per_image = 0.08
        return r


class SlowFake(FakeAlgo):
    delay = 0.5


class SkipAlgo(FakeAlgo):
    def run(self, ds, split, progress, should_stop):
        return AlgoResult("Fake", skipped="박스 있는 간섭이 2장 미만")


class FailAlgo(FakeAlgo):
    def run(self, ds, split, progress, should_stop):
        raise ValueError("마스크와 크기가 다른 이미지입니다: x.png")


@pytest.fixture
def make_view(qtbot, monkeypatch, tmp_path):
    def _make(cls):
        FakeAlgo.calls = 0
        monkeypatch.setattr(rv, "REGISTRY", {"Fake": cls})
        v = rv.ResultView()
        qtbot.addWidget(v)
        ds = make_synthetic(tmp_path, n_normal=10, n_boxed=2)
        v.set_job(ds, "Fake", {})
        return v, ds
    return _make


def run_and_wait(qtbot, v):
    v.run_btn.click()
    qtbot.waitUntil(lambda: not v.running, timeout=10000)


def cell(v, col):
    return v.table.item(0, rv.COLUMNS.index(col)).text()


def table_text(v):
    return " ".join(v.table.item(0, c).text() for c in range(v.table.columnCount()) if v.table.item(0, c))


def test_row_after_run(qtbot, make_view):
    v, _ = make_view(FakeAlgo)
    run_and_wait(qtbot, v)
    assert (cell(v, "놓침"), cell(v, "오검출"), cell(v, "처리 시간")) == ("1/2", "0/2", "80 ms")


def test_slider_and_auto(qtbot, make_view):
    v, _ = make_view(FakeAlgo)
    run_and_wait(qtbot, v)
    v.slider.slider.setValue(0)
    assert (cell(v, "놓침"), cell(v, "오검출")) == ("0/2", "2/2")
    v.slider.auto_btn.click()
    assert cell(v, "놓침") == "1/2"
    assert FakeAlgo.calls == 1


def test_failure_reason_shown(qtbot, make_view):
    v, _ = make_view(FailAlgo)
    run_and_wait(qtbot, v)
    assert "실행 실패" in table_text(v) and "x.png" in table_text(v)


def test_skipped_reason_shown(qtbot, make_view):
    v, _ = make_view(SkipAlgo)
    run_and_wait(qtbot, v)
    assert "2장 미만" in table_text(v)


def test_images_sorted_by_score(qtbot, make_view):
    v, _ = make_view(FakeAlgo)
    run_and_wait(qtbot, v)
    scores = [float(v.images.item(r, 2).text()) for r in range(v.images.rowCount())]
    assert scores == sorted(scores, reverse=True)


def test_result_uses_snapshot(qtbot, make_view):
    v, ds = make_view(FakeAlgo)
    run_and_wait(qtbot, v)
    for it in ds.items:
        it.label = Label.UNLABELED
    v.images.selectRow(0)
    assert cell(v, "놓침") == "1/2" and v.images.rowCount() == 4
    assert set(v.views) == {"원본", "Fake"}


def test_running_changed_signal(qtbot, make_view):
    v, _ = make_view(SlowFake)
    seen = []
    v.running_changed.connect(seen.append)
    run_and_wait(qtbot, v)
    assert seen == [True, False]


def test_clear(qtbot, make_view):
    v, _ = make_view(FakeAlgo)
    run_and_wait(qtbot, v)
    v.clear()
    assert v.table.rowCount() == 0 and v.images.rowCount() == 0 and not v.slider.isEnabled()


def test_run_split_error_shown(qtbot, monkeypatch, tmp_path):
    monkeypatch.setattr(rv, "REGISTRY", {"Fake": FakeAlgo})
    v = rv.ResultView()
    qtbot.addWidget(v)
    v.set_job(make_synthetic(tmp_path, n_normal=1, n_boxed=2), "Fake", {})
    v.run_btn.click()
    assert not v.running and "최소 3개" in table_text(v)


def test_no_algorithm_column(qtbot, make_view):
    v, _ = make_view(FakeAlgo)
    run_and_wait(qtbot, v)
    headers = [v.table.horizontalHeaderItem(c).text() for c in range(v.table.columnCount())]
    assert "알고리즘" not in headers and headers[0] == "놓침"


def test_result_outline_uses_recorded_shift(qtbot, monkeypatch, tmp_path):
    from tests.conftest import make_shifted

    class ShiftFake(FakeAlgo):
        def run(self, ds, split, progress, should_stop):
            r = FakeAlgo.run(self, ds, split, progress, should_stop)
            r.shifts = {p: (10.0, -5.0, 0.5) for p in r.scores}
            return r

    monkeypatch.setattr(rv, "REGISTRY", {"Fake": ShiftFake})
    v = rv.ResultView()
    qtbot.addWidget(v)
    ds, _ = make_shifted(tmp_path)
    v.set_job(ds, "Fake", {})
    run_and_wait(qtbot, v)
    v.images.selectRow(0)
    rect = v.views["원본"].mask_rect()
    assert abs(rect.left() - (224 // 4 + 10)) <= 2 and abs(rect.top() - (224 // 4 - 5)) <= 2
    assert "dx +10" in v.shift_label.text()


def test_target_banner_and_header(qtbot, make_view):
    v, ds = make_view(FakeAlgo)
    run_and_wait(qtbot, v)
    assert "SCAN COMPLETE" in v.header.text() and "1 TARGETS" in v.header.text()   # 간섭 3.0 하나만 임계값 이상
    v.images.selectRow(0)                       # 점수 최고 = 간섭 3.0
    assert "TARGET LOCKED" in v.banner.text()
    v.images.selectRow(v.images.rowCount() - 1)  # 최저 = 정상 0.5
    assert "CLEAR" in v.banner.text()


def test_no_target_on_clear(qtbot, make_view):
    v, ds = make_view(FakeAlgo)
    run_and_wait(qtbot, v)
    v.res.peaks = {p: (10, 10) for p in v.res.scores}      # 모든 이미지에 이상 지점이 있다고 치고
    v.images.selectRow(0)                                   # 점수 최고 = NG
    assert v.views["Fake"].target_items
    v.images.selectRow(v.images.rowCount() - 1)             # 점수 최저 = CLEAR
    assert all(not c.target_items for c in v.views.values())


def test_result_sections_and_big_slider(qtbot, make_view):
    v, ds = make_view(FakeAlgo)
    run_and_wait(qtbot, v)
    assert list(v.sections) == ["실행", "임계값", "이미지"]
    assert v.sections["임계값"].isAncestorOf(v.slider)
    from PySide6.QtWidgets import QLabel
    texts = " ".join(l.text() for l in v.sections["임계값"].findChildren(QLabel))
    assert "더 많이 잡음" not in texts and "오검출 줄임" not in texts and "이상이면" not in texts
    assert v.slider.slider.minimumHeight() >= 28


def test_slider_always_visible_and_locked_until_run(qtbot, make_view):
    v, _ = make_view(FakeAlgo)
    assert v.slider is not None and not v.slider.isEnabled()
    assert v.slider.value_edit.text() == "—"
    run_and_wait(qtbot, v)
    assert v.slider.isEnabled()
    v.clear()
    assert v.slider is not None and not v.slider.isEnabled()


def test_run_button_prominent(qtbot, make_view):
    v, _ = make_view(FakeAlgo)
    assert "실행" in v.run_btn.text() and v.run_btn.minimumHeight() >= 40 and v.run_btn.minimumWidth() >= 140


class FakeModel:
    discarded = 0

    def discard(self):
        type(self).discarded += 1


class ModelAlgo(FakeAlgo):
    def run(self, ds, split, progress, should_stop):
        r = super().run(ds, split, progress, should_stop)
        r.model = FakeModel()
        return r


def test_export_button_disabled_without_model(qtbot, make_view):
    v, _ = make_view(FakeAlgo)
    assert not v.export_btn.isEnabled()
    run_and_wait(qtbot, v)
    assert not v.export_btn.isEnabled()


def test_export_button_disabled_when_skipped(qtbot, make_view):
    v, _ = make_view(SkipAlgo)
    run_and_wait(qtbot, v)
    assert not v.export_btn.isEnabled()


def test_export_button_enabled_after_result_with_model(qtbot, make_view):
    v, _ = make_view(ModelAlgo)
    run_and_wait(qtbot, v)
    assert v.export_btn.isEnabled()


def test_clear_discards_model(qtbot, make_view):
    v, _ = make_view(ModelAlgo)
    run_and_wait(qtbot, v)
    FakeModel.discarded = 0
    v.clear()
    assert FakeModel.discarded == 1 and not v.export_btn.isEnabled()


def test_export_reports_path_and_verification(qtbot, make_view, monkeypatch, tmp_path):
    from incubator.export import Verification
    from incubator.ui import worker
    calls = {}

    def fake_export(res, snap, threshold, metrics, out_root, progress, params=None):
        calls["threshold"], calls["root"] = threshold, out_root
        return tmp_path / "번들"

    monkeypatch.setattr(worker, "export_bundle", fake_export)
    monkeypatch.setattr(worker, "verify", lambda *a: Verification(4, 0, 1e-5, True))
    v, _ = make_view(ModelAlgo)
    run_and_wait(qtbot, v)
    monkeypatch.setattr(v, "pick_dir", lambda: str(tmp_path))
    v.export_btn.click()
    qtbot.waitUntil(lambda: v.exporter is None, timeout=10000)
    assert calls == {"threshold": v.threshold, "root": str(tmp_path)}
    assert "번들" in v.stage.text() and "일치 4/4" in v.stage.text()
    assert v.export_btn.isEnabled() and v.run_btn.isEnabled()


def test_export_failure_is_shown(qtbot, make_view, monkeypatch, tmp_path):
    from incubator.ui import worker

    def boom(*a, **k):
        raise OSError("디스크 가득")

    monkeypatch.setattr(worker, "export_bundle", boom)
    v, _ = make_view(ModelAlgo)
    run_and_wait(qtbot, v)
    monkeypatch.setattr(v, "pick_dir", lambda: str(tmp_path))
    v.export_btn.click()
    qtbot.waitUntil(lambda: v.exporter is None, timeout=10000)
    assert "디스크 가득" in v.stage.text()


def test_export_cancelled_when_no_dir(qtbot, make_view, monkeypatch):
    v, _ = make_view(ModelAlgo)
    run_and_wait(qtbot, v)
    monkeypatch.setattr(v, "pick_dir", lambda: "")
    v.export_btn.click()
    assert v.exporter is None and v.export_btn.isEnabled()


def test_exporting_counts_as_running(qtbot, make_view, monkeypatch, tmp_path):
    import threading
    from incubator.export import Verification
    from incubator.ui import worker
    gate = threading.Event()

    def slow_export(*a, **k):
        gate.wait(5)
        return tmp_path

    monkeypatch.setattr(worker, "export_bundle", slow_export)
    monkeypatch.setattr(worker, "verify", lambda *a: Verification(1, 0, 0.0, True))
    v, _ = make_view(ModelAlgo)
    run_and_wait(qtbot, v)
    monkeypatch.setattr(v, "pick_dir", lambda: str(tmp_path))
    v.export_btn.click()
    assert v.running and not v.run_btn.isEnabled()
    v.clear()                                   # 내보내는 중엔 결과를 지우지 않는다
    assert v.res is not None
    gate.set()
    qtbot.waitUntil(lambda: not v.running, timeout=10000)


def test_failed_verification_removes_bundle(qtbot, make_view, monkeypatch, tmp_path):
    from incubator.ui import worker
    bundle = tmp_path / "번들"
    bundle.mkdir()

    def bad_verify(*a):
        raise RuntimeError("ORT 오류")

    monkeypatch.setattr(worker, "export_bundle", lambda *a, **k: bundle)
    monkeypatch.setattr(worker, "verify", bad_verify)
    v, _ = make_view(ModelAlgo)
    run_and_wait(qtbot, v)
    monkeypatch.setattr(v, "pick_dir", lambda: str(tmp_path))
    v.export_btn.click()
    qtbot.waitUntil(lambda: v.exporter is None, timeout=10000)
    assert not bundle.exists() and "ORT 오류" in v.stage.text() and "지웠" in v.stage.text()


def test_patchcore_hides_location_hit_and_boxes(qtbot, monkeypatch, tmp_path):
    monkeypatch.setattr(rv, "REGISTRY", {"PatchCore": FakeAlgo, "YOLO": FakeAlgo})
    v = rv.ResultView()
    qtbot.addWidget(v)
    ds = make_synthetic(tmp_path, n_normal=10, n_boxed=2)
    hit = rv.COLUMNS.index("위치 적중")
    v.set_job(ds, "PatchCore", {})
    run_and_wait(qtbot, v)
    assert v.table.isColumnHidden(hit)
    row = next(r for r, (it, _) in enumerate(v.rows) if it.boxes)
    v.images.selectRow(row)
    assert all(c.box_count() == 0 for c in v.views.values())
    v.set_job(ds, "YOLO", {})
    assert not v.table.isColumnHidden(hit)


def test_threshold_typed_directly(qtbot, make_view):
    v, _ = make_view(FakeAlgo)
    run_and_wait(qtbot, v)
    v.slider.value_edit.setText("0.4")              # 평가 정상 0.5 → 전부 오검출, 범위 밖 값도 그대로
    v.slider.value_edit.editingFinished.emit()
    assert v.threshold == pytest.approx(0.4) and cell(v, "오검출") == "2/2"
    v.slider.value_edit.setText("abc")              # 숫자가 아니면 이전 값으로 되돌린다
    v.slider.value_edit.editingFinished.emit()
    assert v.threshold == pytest.approx(0.4) and v.slider.value_edit.text() == "0.4"


class DoneAlgo(ModelAlgo):
    def run(self, ds, split, progress, should_stop):
        progress("Fake 완료", 1.0)
        return super().run(ds, split, progress, should_stop)


def test_finished_run_clears_stage_text(qtbot, make_view):
    v, _ = make_view(DoneAlgo)
    run_and_wait(qtbot, v)
    assert v.stage.text() == "" and v.export_btn.isEnabled()
