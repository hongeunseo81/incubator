import pytest

from impostor.dataset import Label, scan_dataset
from impostor.ui.thumb_grid import ThumbGrid

PATTERN = "*/cam_03_IMG_Texture_8Bit.png"


@pytest.fixture
def grid(qtbot, tmp_images):
    g = ThumbGrid()
    qtbot.addWidget(g)
    ds = scan_dataset("t", tmp_images.root, PATTERN, tmp_images.mask)
    g.set_dataset(ds)
    return g, ds


def select(g, rows):
    g.view.clearSelection()
    for r in rows:
        g.view.item(r).setSelected(True)


def test_lists_folder_names_without_decoding(grid):
    g, ds = grid
    assert g.view.count() == len(ds.items)
    assert "1_7C  0001" in g.view.item(0).text()
    assert g.view.item(0).icon().isNull()           # 목록에는 그림을 만들지 않는다


def test_preview_shows_current_only(qtbot, grid):
    g, ds = grid
    g.view.setCurrentRow(1)
    assert g.preview_path == ds.items[1].path and g.preview.has_image()


def test_preview_wheel_zoom_kept_between_images(qtbot, grid):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QApplication
    g, ds = grid
    g.view.setCurrentRow(0)
    before = g.preview.transform().m11()
    ev = QWheelEvent(QPointF(10, 10), QPointF(10, 10), QPoint(0, 0), QPoint(0, 120), Qt.NoButton,
                     Qt.NoModifier, Qt.NoScrollPhase, False)
    QApplication.sendEvent(g.preview.viewport(), ev)
    zoomed = g.preview.transform().m11()
    assert zoomed > before
    g.view.setCurrentRow(1)
    assert g.preview.transform().m11() == zoomed


def test_bulk_label_selection(qtbot, grid):
    g, ds = grid
    select(g, [0, 1])
    with qtbot.waitSignal(g.changed, timeout=1000):
        g.buttons[Label.INTERFERENCE].click()
    assert all(i.label is Label.INTERFERENCE for i in ds.items)
    assert "간섭" in g.view.item(0).text()


def test_rest_to_normal(grid):
    g, ds = grid
    select(g, [0])
    g.buttons[Label.INTERFERENCE].click()
    g.rest_normal_btn.click()
    assert [i.label for i in ds.items] == [Label.INTERFERENCE, Label.NORMAL]


def test_double_click_opens_item(qtbot, grid):
    g, ds = grid
    with qtbot.waitSignal(g.open_item, timeout=1000) as sig:
        g.view.itemDoubleClicked.emit(g.view.item(1))
    assert sig.args == [ds.items[1]]


def test_filter(grid):
    g, ds = grid
    select(g, [0])
    g.buttons[Label.NORMAL].click()
    g.filter.setCurrentText("미분류")
    assert g.view.count() == 1


def test_shortcuts_apply_to_selection(qtbot, grid):
    from PySide6.QtCore import Qt
    g, ds = grid
    select(g, [0, 1])
    qtbot.keyClick(g.view, Qt.Key_I)
    assert all(i.label is Label.INTERFERENCE for i in ds.items)
    select(g, [1])
    qtbot.keyClick(g.view, Qt.Key_N)
    assert ds.items[1].label is Label.NORMAL
    qtbot.keyClick(g.view, Qt.Key_U)
    assert ds.items[1].label is Label.UNLABELED
    assert "(I)" in g.buttons[Label.INTERFERENCE].text()


def test_preview_outline_follows_alignment(qtbot, tmp_path):
    from tests.conftest import make_shifted
    ds, truth = make_shifted(tmp_path)
    g = ThumbGrid()
    qtbot.addWidget(g)
    g.set_dataset(ds)
    g.view.setCurrentRow(0)
    dx, dy = truth[ds.items[0].path]
    rect = g.preview.mask_rect()
    assert abs(rect.left() - (224 // 4 + dx)) <= 2 and abs(rect.top() - (224 // 4 + dy)) <= 2
    assert "위치 보정" in g.shift_label.text()


def test_list_sets_usage_for_selection(qtbot, grid):
    from impostor.dataset import Pin
    g, ds = grid
    select(g, [0, 1])
    g.buttons[Label.NORMAL].click()
    select(g, [1])
    g.pin_buttons[Pin.TRAIN].click()
    assert ds.items[1].pin is Pin.TRAIN and ds.items[0].pin is Pin.NONE
    assert "학습" in g.view.item(1).text()
    g.pin_buttons[Pin.NONE].click()
    assert ds.items[1].pin is Pin.NONE and "학습" not in g.view.item(1).text()


def test_list_usage_hidden_for_interference(qtbot, grid):
    g, ds = grid
    select(g, [0])
    g.buttons[Label.INTERFERENCE].click()
    g.view.setCurrentRow(0)
    assert g.pin_box.isHidden()
    g.buttons[Label.NORMAL].click()
    assert not g.pin_box.isHidden()
