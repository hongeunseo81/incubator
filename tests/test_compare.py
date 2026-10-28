import time

import pytest
from PySide6.QtWidgets import QMessageBox

from impostor.algorithms.base import AlgoResult, eval_items
from impostor.ui import compare_tab as ct
from tests.conftest import make_synthetic


class FakeAlgo:
    """간섭 [3.0, 1.0], 임계값용 정상 2.0, 평가 정상 0.5"""
    name = "Fake"
    calls = 0
    delay = 0.0

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


class SkipAlgo:
    name = "Skip"

    def run(self, ds, split, progress, should_stop):
        return AlgoResult(self.name, skipped="박스 있는 간섭이 2장 미만")


@pytest.fixture
def make_tab(qtbot, monkeypatch, tmp_path):
    def _make(registry, **kw):
        FakeAlgo.calls = 0
        monkeypatch.setattr(ct, "REGISTRY", registry)
        tab = ct.CompareTab()
        qtbot.addWidget(tab)
        tab.set_dataset(make_synthetic(tmp_path, **({"n_normal": 10, "n_boxed": 2} | kw)))
        return tab
    return _make


def run_and_wait(qtbot, tab):
    tab.run_btn.click()
    qtbot.waitUntil(lambda: tab.run_btn.isEnabled(), timeout=10000)


def cell(tab, row, col_name):
    return tab.table.item(row, ct.COLUMNS.index(col_name)).text()


def test_scoreboard_row_after_run(qtbot, make_tab):
    tab = make_tab({"Fake": FakeAlgo})
    run_and_wait(qtbot, tab)
    assert cell(tab, 0, "알고리즘") == "Fake"
    assert cell(tab, 0, "놓침") == "1/2"
    assert cell(tab, 0, "오검출") == "0/2"
    assert cell(tab, 0, "처리 시간") == "80 ms"


def test_slider_recomputes_without_rerun(qtbot, make_tab):
    tab = make_tab({"Fake": FakeAlgo})
    run_and_wait(qtbot, tab)
    tab.sliders["Fake"].slider.setValue(0)
    assert cell(tab, 0, "놓침") == "0/2"
    assert cell(tab, 0, "오검출") == "2/2"
    tab.sliders["Fake"].auto_btn.click()
    assert cell(tab, 0, "놓침") == "1/2"
    assert FakeAlgo.calls == 1


def test_run_button_disabled_while_running(qtbot, make_tab):
    tab = make_tab({"Fake": SlowFake})
    tab.run_btn.click()
    assert not tab.run_btn.isEnabled()
    qtbot.waitUntil(lambda: tab.run_btn.isEnabled(), timeout=10000)


def test_split_error_shown(qtbot, make_tab, monkeypatch):
    seen = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: seen.append(a[2]))
    tab = make_tab({"Fake": FakeAlgo}, n_normal=2)
    tab.run_btn.click()
    assert seen and "최소 3개" in seen[0]
    assert tab.run_btn.isEnabled()


def test_skipped_reason_shown(qtbot, make_tab):
    tab = make_tab({"Skip": SkipAlgo})
    run_and_wait(qtbot, tab)
    assert "2장 미만" in cell(tab, 0, "놓침")


def test_selecting_image_shows_views(qtbot, make_tab):
    tab = make_tab({"Fake": FakeAlgo})
    run_and_wait(qtbot, tab)
    tab.images.selectRow(0)
    assert set(tab.views) == {"원본", "Fake"}


def test_label_change_clears_stale_results(qtbot, make_tab):
    tab = make_tab({"Fake": FakeAlgo})
    run_and_wait(qtbot, tab)
    tab.labels_changed()
    assert tab.table.rowCount() == 0 and tab.images.rowCount() == 0


def test_main_window_hosts_compare_tab(qtbot, monkeypatch):
    from impostor.ui.app import MainWindow
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Discard)
    w = MainWindow()
    qtbot.addWidget(w)
    assert isinstance(w.compare_tab, ct.CompareTab)
