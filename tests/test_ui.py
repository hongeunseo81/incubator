import os

import pytest
from PySide6.QtCore import Qt

from impostor.dataset import Label, Pin, scan_dataset
from impostor.ui.label_tab import LabelTab

PATTERN = "*/cam_03_IMG_Texture_8Bit.png"


@pytest.fixture
def small_ds(tmp_images):
    return scan_dataset("t", tmp_images.root, PATTERN, tmp_images.mask)


@pytest.fixture
def label_tab(qtbot, small_ds):
    tab = LabelTab()
    qtbot.addWidget(tab)
    tab.set_dataset(small_ds)
    return tab


def test_label_tab_shortcut_sets_normal(qtbot, label_tab, small_ds):
    qtbot.keyClick(label_tab.list, Qt.Key_N)
    assert small_ds.items[0].label is Label.NORMAL


def test_shortcut_moves_to_next(qtbot, label_tab, small_ds):
    qtbot.keyClick(label_tab.list, Qt.Key_Right)
    qtbot.keyClick(label_tab.list, Qt.Key_U)
    qtbot.keyClick(label_tab.list, Qt.Key_I)
    assert small_ds.items[1].label is Label.INTERFERENCE


def test_box_added_marks_interference(label_tab, small_ds):
    label_tab.canvas.box_added.emit((1, 1, 20, 20))
    assert small_ds.items[0].label is Label.INTERFERENCE
    assert small_ds.items[0].boxes == [(1, 1, 20, 20)]


def test_box_deleted(label_tab, small_ds):
    label_tab.canvas.box_added.emit((1, 1, 20, 20))
    label_tab.canvas.box_deleted.emit(0)
    assert small_ds.items[0].boxes == []


def test_pin_buttons(label_tab, small_ds):
    label_tab.pin_buttons[Pin.EVAL].click()
    assert small_ds.items[0].pin is Pin.EVAL


def test_changed_signal(qtbot, label_tab):
    with qtbot.waitSignal(label_tab.changed, timeout=1000):
        qtbot.keyClick(label_tab.list, Qt.Key_N)


def test_missing_file_does_not_crash(qtbot, small_ds):
    os.remove(small_ds.items[0].path)
    tab = LabelTab()
    qtbot.addWidget(tab)
    tab.set_dataset(small_ds)
    assert "파일 없음" in tab.list.item(0).text()


# ---- Task 7: 메인 창 ----
from PySide6.QtWidgets import QMessageBox

from impostor.dataset import save_dataset
from impostor.ui.app import MainWindow
from impostor.ui.dataset_dialogs import ImportDialog, default_label_for


@pytest.fixture
def main_window(qtbot, monkeypatch):
    # 정리 단계에서 창을 닫을 때 "저장할까요?" 모달이 테스트를 멈추지 않도록
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Discard)
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def test_main_window_opens(main_window):
    assert main_window.windowTitle().startswith("Impostor")


def test_open_dataset_warns_missing(main_window, small_ds, tmp_path, monkeypatch):
    p = tmp_path / "t.json"
    save_dataset(small_ds, p)
    os.remove(small_ds.items[0].path)
    seen = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: seen.append(a[2]))
    main_window.open_dataset(str(p))
    assert main_window.dataset.name == "t"
    assert seen and "찾을 수 없습니다" in seen[0]


def test_dirty_marker(qtbot, main_window, small_ds):
    main_window.set_dataset(small_ds)
    assert "*" not in main_window.windowTitle()
    qtbot.keyClick(main_window.label_tab.list, Qt.Key_N)
    assert "*" in main_window.windowTitle()


def test_save_clears_dirty(qtbot, main_window, small_ds, tmp_path, monkeypatch):
    monkeypatch.setattr("impostor.ui.app.DATASET_DIR", tmp_path)
    main_window.set_dataset(small_ds)
    qtbot.keyClick(main_window.label_tab.list, Qt.Key_N)
    main_window.save()
    assert (tmp_path / "t.json").exists() and "*" not in main_window.windowTitle()


def test_import_default_labels():
    assert default_label_for("normal") is Label.NORMAL
    assert default_label_for("inside") is Label.INTERFERENCE
    assert default_label_for("edge") is Label.INTERFERENCE
    assert default_label_for("minor") is Label.UNLABELED


def test_import_dialog_builds_dataset(qtbot, tmp_path):
    import numpy as np
    from tests.conftest import write_png
    img = np.zeros((48, 64, 3), np.uint8)
    write_png(tmp_path / "data" / "normal" / "a__LH.png", img)
    write_png(tmp_path / "data" / "inside" / "b__LH.png", img)
    d = ImportDialog()
    qtbot.addWidget(d)
    d.name.setText("fem")
    d.set_root(str(tmp_path / "data"))
    d.name_glob.setText("*__LH.png")
    d.mask.setText(str(tmp_path / "m.png"))
    ds = d.build()
    assert ds.name == "fem" and sorted(i.label for i in ds.items) == sorted([Label.NORMAL, Label.INTERFERENCE])


def test_scan_dialog_lists_kinds(qtbot, tmp_images):
    from impostor.ui.dataset_dialogs import ScanDialog
    d = ScanDialog()
    qtbot.addWidget(d)
    d.set_root(tmp_images.root)
    kinds = [d.kind.itemData(i) for i in range(d.kind.count())]
    assert kinds == ["cam_01_IMG_Texture_8Bit.png", "cam_03_IMG_Texture_8Bit.png"]
    d.kind.setCurrentIndex(1)
    d.name.setText("t")
    d.mask.setText(tmp_images.mask)
    assert len(d.build().items) == 2


def test_data_step_grid_double_click_switches_to_single(qtbot, small_ds):
    from impostor.ui.wizard import DataStep
    s = DataStep()
    qtbot.addWidget(s)
    s.set_dataset(small_ds)
    s.grid.open_item.emit(small_ds.items[1])
    assert s.views.currentWidget() is s.label_tab and s.label_tab.current() is small_ds.items[1]


def test_scan_dialog_suggests_master_mask(qtbot, tmp_path):
    import numpy as np
    from tests.conftest import write_png
    from impostor.ui.dataset_dialogs import ScanDialog
    img = np.zeros((48, 64, 3), np.uint8)
    write_png(tmp_path / "Source" / "car1" / "fastenLHCam_02_IMG_Texture_8Bit.png", img)
    m = write_png(tmp_path / "Master" / "JG" / "fastenLHCam_02_IMG_Kungfu.png", img)
    write_png(tmp_path / "Master" / "JK" / "fastenLHCam_02_IMG_Kungfu.png", img)
    d = ScanDialog()
    qtbot.addWidget(d)
    d.set_root(str(tmp_path))
    assert d.mask.text() == m
