import pytest
from PySide6.QtWidgets import QMessageBox

from impostor.dataset import Label
from impostor.ui import result_view as rv
from impostor.ui.wizard import Wizard
from tests.conftest import make_synthetic
from tests.test_result_view import FakeAlgo, SlowFake


@pytest.fixture
def wizard(qtbot):
    w = Wizard()
    qtbot.addWidget(w)
    return w


def choose(w, algo):
    w.algo.buttons[algo].click()


def test_next_blocked_until_algo_selected(wizard):
    assert wizard.step == 0 and not wizard.next_btn.isEnabled()
    choose(wizard, "PatchCore")
    assert wizard.next_btn.isEnabled()


def test_next_blocked_without_boxes(wizard, tmp_path):
    choose(wizard, "YOLO")
    wizard.next_btn.click()
    wizard.set_dataset(make_synthetic(tmp_path, n_normal=10, n_boxed=0, n_boxless=2))
    assert wizard.step == 1 and not wizard.next_btn.isEnabled()
    assert "2장 이상" in wizard.data.status.text()


def test_next_blocked_without_dataset(wizard):
    choose(wizard, "PatchCore")
    wizard.next_btn.click()
    assert not wizard.next_btn.isEnabled() and "데이터셋을 고르세요" in wizard.data.status.text()


def test_patchcore_flow_reaches_result(wizard, tmp_path):
    choose(wizard, "PatchCore")
    wizard.next_btn.click()
    wizard.set_dataset(make_synthetic(tmp_path, n_normal=10, n_boxed=2))
    wizard.next_btn.click()
    wizard.next_btn.click()
    assert wizard.step == 3 and wizard.result.algo == "PatchCore"
    assert wizard.result.params == {"width": 640, "coreset": 20000, "erode": 5}
    assert wizard.result.table.rowCount() == 0
    assert "PatchCore" in wizard.step_labels[0].text()


def test_param_form_follows_algo(wizard, tmp_path):
    wizard.set_dataset(make_synthetic(tmp_path, n_normal=10, n_boxed=2))
    choose(wizard, "YOLO")
    wizard.go(2)
    assert "epochs" in wizard.params.widgets
    wizard.go(0)
    choose(wizard, "PatchCore")
    wizard.go(2)
    assert "coreset" in wizard.params.widgets and "epochs" not in wizard.params.widgets


def test_param_change_reaches_result(wizard, tmp_path):
    choose(wizard, "PatchCore")
    wizard.next_btn.click()
    wizard.set_dataset(make_synthetic(tmp_path, n_normal=10, n_boxed=2))
    wizard.next_btn.click()
    wizard.params.widgets["coreset"].setValue(5000)
    wizard.next_btn.click()
    assert wizard.result.params["coreset"] == 5000
    wizard.params.reset_btn.click()
    assert wizard.params.values()["coreset"] == 20000


def to_result(qtbot, wizard, tmp_path, monkeypatch, cls):
    monkeypatch.setattr(rv, "REGISTRY", {"PatchCore": cls})
    FakeAlgo.calls = 0
    choose(wizard, "PatchCore")
    wizard.next_btn.click()
    ds = make_synthetic(tmp_path, n_normal=10, n_boxed=2)
    wizard.set_dataset(ds)
    wizard.next_btn.click()
    wizard.next_btn.click()
    return ds


def test_navigation_locked_while_running(qtbot, wizard, tmp_path, monkeypatch):
    to_result(qtbot, wizard, tmp_path, monkeypatch, SlowFake)
    wizard.result.run_btn.click()
    assert not wizard.prev_btn.isEnabled()
    qtbot.waitUntil(lambda: not wizard.result.running, timeout=10000)
    assert wizard.prev_btn.isEnabled()


def test_results_cleared_after_edit(qtbot, wizard, tmp_path, monkeypatch):
    to_result(qtbot, wizard, tmp_path, monkeypatch, FakeAlgo)
    wizard.result.run_btn.click()
    qtbot.waitUntil(lambda: not wizard.result.running, timeout=10000)
    assert wizard.result.table.rowCount() == 1
    wizard.prev_btn.click()
    wizard.prev_btn.click()
    wizard.data.label_tab.set_label(Label.UNLABELED)
    wizard.go(3)
    assert wizard.result.table.rowCount() == 0


def test_results_kept_when_just_browsing_back(qtbot, wizard, tmp_path, monkeypatch):
    to_result(qtbot, wizard, tmp_path, monkeypatch, FakeAlgo)
    wizard.result.run_btn.click()
    qtbot.waitUntil(lambda: not wizard.result.running, timeout=10000)
    wizard.prev_btn.click()
    wizard.next_btn.click()
    assert wizard.result.table.rowCount() == 1


def test_main_window_uses_wizard(qtbot, monkeypatch):
    from impostor.ui.app import MainWindow
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Discard)
    w = MainWindow()
    qtbot.addWidget(w)
    assert isinstance(w.wizard, Wizard) and w.label_tab is w.wizard.data.label_tab


def test_go_cannot_skip_ungated_steps(wizard):
    wizard.go(3)
    assert wizard.step == 0
    choose(wizard, "YOLO")
    wizard.go(3)
    assert wizard.step == 1


def test_opening_unready_dataset_at_result_goes_back(wizard, tmp_path):
    choose(wizard, "PatchCore")
    wizard.set_dataset(make_synthetic(tmp_path / "a", n_normal=10, n_boxed=2))
    wizard.go(3)
    assert wizard.step == 3
    wizard.set_dataset(make_synthetic(tmp_path / "b", n_normal=1, n_boxed=2))
    assert wizard.step == 1 and "최소 3개" in wizard.data.status.text()


def test_opening_ready_dataset_at_result_rebinds(wizard, tmp_path):
    choose(wizard, "PatchCore")
    wizard.set_dataset(make_synthetic(tmp_path / "a", n_normal=10, n_boxed=2))
    wizard.go(3)
    b = make_synthetic(tmp_path / "b", n_normal=10, n_boxed=2)
    wizard.set_dataset(b)
    assert wizard.result.ds is b


def test_size_mismatch_shown_in_status(wizard, tmp_path):
    import numpy as np
    from pathlib import Path
    from tests.conftest import write_png
    ds = make_synthetic(tmp_path, n_normal=10, n_boxed=2)
    write_png(Path(ds.items[0].path), np.zeros((50, 50, 3), np.uint8))
    choose(wizard, "PatchCore")
    wizard.set_dataset(ds)
    assert "크기가 다른 이미지 1개" in wizard.data.status.text()


def test_size_mismatch_cleared_by_unlabeling(wizard, tmp_path):
    import numpy as np
    from pathlib import Path
    from tests.conftest import write_png
    ds = make_synthetic(tmp_path, n_normal=10, n_boxed=2)
    write_png(Path(ds.items[0].path), np.zeros((50, 50, 3), np.uint8))
    choose(wizard, "PatchCore")
    wizard.set_dataset(ds)
    wizard.data.label_tab.set_label(Label.UNLABELED)     # 첫 항목 = 크기 다른 이미지
    assert "크기가 다른" not in wizard.data.status.text()
