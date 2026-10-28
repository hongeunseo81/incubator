from impostor.ui.boxes import normalize_box


def test_normalize_reverse_drag():
    assert normalize_box(50, 40, 10, 5, 100, 100) == (10, 5, 50, 40)


def test_normalize_clamps():
    assert normalize_box(-5, -5, 200, 200, 100, 80) == (0, 0, 100, 80)


def test_normalize_too_small():
    assert normalize_box(10, 10, 11, 40, 100, 100) is None


def test_normalize_rounds_floats():
    assert normalize_box(10.4, 10.6, 30.5, 40.2, 100, 100) == (10, 11, 30, 40)
