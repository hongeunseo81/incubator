import pytest

from incubator.dataset import Label, scan_dataset
from incubator.ui.thumb_grid import ThumbGrid

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
    from incubator.dataset import Pin
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


def test_list_usage_shows_auto_by_default_and_reflects_choice(qtbot, grid):
    from incubator.dataset import Pin
    g, ds = grid
    select(g, [0])
    g.buttons[Label.NORMAL].click()
    g.view.setCurrentRow(0)
    assert g.pin_buttons[Pin.NONE].isChecked() and not g.pin_buttons[Pin.TRAIN].isChecked()
    g.pin_buttons[Pin.TRAIN].click()
    assert g.pin_buttons[Pin.TRAIN].isChecked() and not g.pin_buttons[Pin.NONE].isChecked()


# ---- 목록에서 바로 박스 그리기 (간섭 사진만) ----
def test_preview_editable_only_for_interference(qtbot, grid):
    g, ds = grid
    g.view.setCurrentRow(0)
    assert not g.preview.editable                       # 미분류
    g.label_current(Label.INTERFERENCE)
    assert g.preview.editable
    g.label_current(Label.NORMAL)
    assert not g.preview.editable


def test_draw_and_delete_box_in_list(qtbot, grid):
    g, ds = grid
    g.view.setCurrentRow(1)
    g.label_current(Label.INTERFERENCE)
    with qtbot.waitSignal(g.changed, timeout=1000):
        g.preview.box_added.emit((1, 1, 20, 20))
    assert ds.items[1].boxes == [(1, 1, 20, 20)] and "박스1" in g.view.item(1).text()
    g.preview.box_deleted.emit(0)
    assert ds.items[1].boxes == [] and "박스" not in g.view.item(1).text()


def test_preview_size_stable_when_switching(qtbot, grid):
    g, ds = grid
    g.resize(1200, 700)
    g.show()
    qtbot.waitExposed(g)
    g.view.setCurrentRow(0)
    before = g.preview.size()
    g.shift_label.setText("x" * 500)
    g.view.setCurrentRow(1)
    qtbot.wait(10)
    assert g.preview.size() == before


def test_missing_file_listed(qtbot, small_ds):
    import os
    os.remove(small_ds.items[0].path)
    g = ThumbGrid()
    qtbot.addWidget(g)
    g.set_dataset(small_ds)
    g.view.setCurrentRow(0)
    assert "파일 없음" in g.view.item(0).text()


def test_boxes_hidden_when_disabled(qtbot, grid):
    g, ds = grid
    g.view.setCurrentRow(1)
    g.label_current(Label.INTERFERENCE)
    g.preview.box_added.emit((1, 1, 20, 20))
    assert "박스1" in g.view.item(1).text()
    g.set_boxes_enabled(False)                          # PatchCore
    assert "박스" not in g.view.item(1).text()
    assert not g.preview.editable and g.preview.box_count() == 0
    assert ds.items[1].boxes == [(1, 1, 20, 20)]         # 데이터는 그대로
    g.set_boxes_enabled(True)                           # YOLO
    assert "박스1" in g.view.item(1).text() and g.preview.editable and g.preview.box_count() == 1
