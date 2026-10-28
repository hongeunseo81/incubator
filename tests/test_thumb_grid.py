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
    qtbot.waitUntil(lambda: g.loaded == len(ds.items), timeout=5000)
    return g, ds


def select(g, rows):
    g.view.clearSelection()
    for r in rows:
        g.view.item(r).setSelected(True)


def test_shows_every_image(grid):
    g, ds = grid
    assert g.view.count() == len(ds.items) and not g.view.item(0).icon().isNull()


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
