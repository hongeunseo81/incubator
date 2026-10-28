from incubator.metrics import auto_threshold, evaluate, location_hit


def test_auto_threshold_just_above_max():
    t = auto_threshold([1.0, 2.5, 2.0])
    assert 2.5 < t < 2.51


def test_auto_threshold_zero_scores():
    assert auto_threshold([0.0, 0.0]) > 0.0


def test_auto_threshold_gives_zero_fp_on_its_own_set():
    s = {f"n{i}": v for i, v in enumerate([0.3, 1.7, 0.9, 1.7])}
    m = evaluate({}, s, auto_threshold(s.values()), {}, {})
    assert m.false_pos == 0


def test_counts():
    m = evaluate({"a": 3.0, "b": 1.0}, {"c": 0.5, "d": 2.6}, 2.0, {}, {})
    assert (m.missed, m.n_interference, m.false_pos, m.n_normal_test) == (1, 2, 1, 2)


def test_equal_score_is_ng():
    m = evaluate({"a": 2.0}, {"c": 2.0}, 2.0, {}, {})
    assert m.missed == 0 and m.false_pos == 1


def test_auc():
    m = evaluate({"a": 3.0, "b": 1.0}, {"c": 0.5, "d": 2.6}, 2.0, {}, {})
    assert m.auc == 0.75


def test_auc_none_when_one_side_empty():
    assert evaluate({}, {"c": 0.5}, 1.0, {}, {}).auc is None
    assert evaluate({"a": 0.5}, {}, 1.0, {}, {}).auc is None


def test_location_hit_inside_and_outside():
    boxes = [(10, 10, 20, 20), (50, 50, 60, 60)]
    assert location_hit((20, 15), boxes)
    assert location_hit((55, 55), boxes)
    assert not location_hit((30, 30), boxes)
    assert not location_hit(None, boxes)


def test_location_hit_excludes_boxless():
    m = evaluate({"a": 3.0, "b": 3.0, "c": 3.0}, {}, 1.0,
                 {"a": (15, 15), "b": (0, 0), "c": (5, 5)},
                 {"a": [(10, 10, 20, 20)], "b": [(10, 10, 20, 20)], "c": []})
    assert (m.hit, m.n_hit_eval) == (1, 2)
