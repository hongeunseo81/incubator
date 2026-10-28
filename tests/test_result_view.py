import time

import pytest

from impostor.algorithms.base import AlgoResult, eval_items
from impostor.dataset import Label
from impostor.ui import result_view as rv
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
    assert list(v.sections) == ["실행", "판정 기준", "이미지"]
    assert v.sections["판정 기준"].isAncestorOf(v.slider)
    assert v.slider.slider.minimumHeight() >= 28


def test_slider_always_visible_and_locked_until_run(qtbot, make_view):
    v, _ = make_view(FakeAlgo)
    assert v.slider is not None and not v.slider.isEnabled()
    assert "실행하면" in v.slider.value_label.text()
    run_and_wait(qtbot, v)
    assert v.slider.isEnabled()
    v.clear()
    assert v.slider is not None and not v.slider.isEnabled()


def test_run_button_prominent(qtbot, make_view):
    v, _ = make_view(FakeAlgo)
    assert "실행" in v.run_btn.text() and v.run_btn.minimumHeight() >= 40 and v.run_btn.minimumWidth() >= 140
