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
