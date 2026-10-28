from PySide6.QtWidgets import QApplication

from impostor.ui import theme
from impostor.ui.canvas import ImageCanvas


def test_apply_sets_dark_stylesheet(qapp):
    theme.apply(qapp)
    assert theme.BG in qapp.styleSheet() and theme.ACCENT in qapp.styleSheet()


def test_target_reticle_locked_and_clear(qtbot):
    import numpy as np
    c = ImageCanvas()
    qtbot.addWidget(c)
    c.set_image(np.zeros((100, 120, 3), np.uint8))
    c.set_target((60, 50), locked=True)
    assert c.target_items and c.target_color().name() == theme.ALERT.lower()
    c.set_target((60, 50), locked=False)
    assert c.target_color().name() == theme.ACCENT.lower()
    c.set_target(None, locked=False)
    assert not c.target_items
