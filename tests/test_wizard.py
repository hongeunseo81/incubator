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


def tab(w, i):
    w.tabs[i].click()


def test_no_prev_next_buttons(wizard):
    assert not hasattr(wizard, "next_btn") and not hasattr(wizard, "prev_btn")


def test_tabs_locked_until_algo_selected(wizard):
    assert wizard.step == 0 and not wizard.tabs[1].isEnabled() and not wizard.tabs[2].isEnabled()
    choose(wizard, "PatchCore")
    assert wizard.step == 1                       # 카드를 누르면 바로 데이터 준비로
    assert wizard.tabs[0].isEnabled() and wizard.tabs[1].isEnabled() and not wizard.tabs[2].isEnabled()


def test_algo_cards_are_large(wizard):
    card = wizard.algo.buttons["PatchCore"]
    assert card.minimumWidth() >= 360 and card.minimumHeight() >= 240


def test_result_tab_blocked_without_boxes(wizard, tmp_path):
    choose(wizard, "YOLO")
    wizard.set_dataset(make_synthetic(tmp_path, n_normal=10, n_boxed=0, n_boxless=2))
    tab(wizard, 2)
    assert wizard.step == 1 and not wizard.tabs[2].isEnabled()
    assert "2장 이상" in wizard.data.status.text() and "2장 이상" in wizard.reason.text()


def test_result_tab_blocked_without_dataset(wizard):
    choose(wizard, "PatchCore")
    assert not wizard.tabs[2].isEnabled() and "데이터셋을 고르세요" in wizard.data.status.text()


def test_patchcore_flow_reaches_result(wizard, tmp_path):
    choose(wizard, "PatchCore")
    wizard.set_dataset(make_synthetic(tmp_path, n_normal=10, n_boxed=2))
    tab(wizard, 2)
    assert wizard.step == 2 and wizard.result.algo == "PatchCore"
    assert len(wizard.tabs) == 3
    assert wizard.result.params == {"width": 640, "coreset": 20000, "erode": 5}
    assert wizard.result.table.rowCount() == 0
    assert "PatchCore" in wizard.tabs[0].text()


def test_param_form_follows_algo(wizard, tmp_path):
    wizard.set_dataset(make_synthetic(tmp_path, n_normal=10, n_boxed=2))
    choose(wizard, "YOLO")
    wizard.go(2)
    assert "epochs" in wizard.params.widgets and wizard.params.isVisible() == wizard.result.isVisible()
    wizard.go(0)
    choose(wizard, "PatchCore")
    wizard.go(2)
    assert "coreset" in wizard.params.widgets and "epochs" not in wizard.params.widgets


def test_param_change_reaches_result(wizard, tmp_path):
    choose(wizard, "PatchCore")
    wizard.set_dataset(make_synthetic(tmp_path, n_normal=10, n_boxed=2))
    tab(wizard, 2)
    wizard.params.widgets["coreset"].setValue(5000)
    assert wizard.result.params["coreset"] == 5000
    wizard.params.reset_btn.click()
    assert wizard.params.values()["coreset"] == 20000


def to_result(qtbot, wizard, tmp_path, monkeypatch, cls):
    monkeypatch.setattr(rv, "REGISTRY", {"PatchCore": cls})
    FakeAlgo.calls = 0
    choose(wizard, "PatchCore")
    ds = make_synthetic(tmp_path, n_normal=10, n_boxed=2)
    wizard.set_dataset(ds)
    tab(wizard, 2)
    return ds


def test_navigation_locked_while_running(qtbot, wizard, tmp_path, monkeypatch):
    to_result(qtbot, wizard, tmp_path, monkeypatch, SlowFake)
    wizard.result.run_btn.click()
    assert not wizard.tabs[0].isEnabled() and not wizard.tabs[1].isEnabled() and not wizard.params.isEnabled()
    qtbot.waitUntil(lambda: not wizard.result.running, timeout=10000)
    assert wizard.tabs[1].isEnabled() and wizard.params.isEnabled()


def test_results_cleared_after_edit(qtbot, wizard, tmp_path, monkeypatch):
    to_result(qtbot, wizard, tmp_path, monkeypatch, FakeAlgo)
    wizard.result.run_btn.click()
    qtbot.waitUntil(lambda: not wizard.result.running, timeout=10000)
    assert wizard.result.table.rowCount() == 1
    tab(wizard, 1)
    wizard.data.grid.view.setCurrentRow(0)
    wizard.data.grid.label_current(Label.UNLABELED)
    tab(wizard, 2)
    assert wizard.result.table.rowCount() == 0


def test_results_kept_when_just_browsing_back(qtbot, wizard, tmp_path, monkeypatch):
    to_result(qtbot, wizard, tmp_path, monkeypatch, FakeAlgo)
    wizard.result.run_btn.click()
    qtbot.waitUntil(lambda: not wizard.result.running, timeout=10000)
    tab(wizard, 1)
    tab(wizard, 2)
    assert wizard.result.table.rowCount() == 1


def test_main_window_uses_wizard(qtbot, monkeypatch):
    from impostor.ui.app import MainWindow
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Discard)
    w = MainWindow()
    qtbot.addWidget(w)
    assert isinstance(w.wizard, Wizard)


def test_go_cannot_skip_ungated_steps(wizard):
    wizard.go(2)
    assert wizard.step == 0
    choose(wizard, "YOLO")
    wizard.go(2)
    assert wizard.step == 1


def test_opening_unready_dataset_at_result_goes_back(wizard, tmp_path):
    choose(wizard, "PatchCore")
    wizard.set_dataset(make_synthetic(tmp_path / "a", n_normal=10, n_boxed=2))
    wizard.go(2)
    assert wizard.step == 2
    wizard.set_dataset(make_synthetic(tmp_path / "b", n_normal=1, n_boxed=2))
    assert wizard.step == 1 and "최소 3개" in wizard.data.status.text()


def test_opening_ready_dataset_at_result_rebinds(wizard, tmp_path):
    choose(wizard, "PatchCore")
    wizard.set_dataset(make_synthetic(tmp_path / "a", n_normal=10, n_boxed=2))
    wizard.go(2)
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
    wizard.data.grid.view.setCurrentRow(0)
    wizard.data.grid.label_current(Label.UNLABELED)     # 첫 항목 = 크기 다른 이미지
    assert "크기가 다른" not in wizard.data.status.text()


def test_reason_shown_next_to_tabs(wizard):
    choose(wizard, "PatchCore")
    assert "데이터셋을 고르세요" in wizard.reason.text()


def test_entering_data_step_opens_latest_saved_dataset(qtbot, tmp_path, monkeypatch):
    import os
    import time
    from impostor import dataset as dataset_mod
    from impostor.dataset import save_dataset
    monkeypatch.setattr(dataset_mod, "DATASET_DIR", tmp_path / "saved")
    old = make_synthetic(tmp_path / "a", n_normal=10, n_boxed=2)
    old.name = "old"
    new = make_synthetic(tmp_path / "b", n_normal=10, n_boxed=2)
    new.name = "new"
    save_dataset(old, tmp_path / "saved" / "old.json")
    save_dataset(new, tmp_path / "saved" / "new.json")
    os.utime(tmp_path / "saved" / "old.json", (time.time() - 100, time.time() - 100))
    w = Wizard()
    qtbot.addWidget(w)
    opened = []
    w.data.open_requested.connect(opened.append)
    choose(w, "PatchCore")
    assert opened == [str(tmp_path / "saved" / "new.json")]


def test_param_change_clears_result(qtbot, wizard, tmp_path, monkeypatch):
    to_result(qtbot, wizard, tmp_path, monkeypatch, FakeAlgo)
    wizard.result.run_btn.click()
    qtbot.waitUntil(lambda: not wizard.result.running, timeout=10000)
    wizard.params.widgets["coreset"].setValue(5000)
    assert wizard.result.table.rowCount() == 0


def test_param_help_shown(wizard, tmp_path):
    choose(wizard, "PatchCore")
    wizard.set_dataset(make_synthetic(tmp_path, n_normal=10, n_boxed=2))
    wizard.go(2)
    assert "오검출" in wizard.params.help_text("coreset")
