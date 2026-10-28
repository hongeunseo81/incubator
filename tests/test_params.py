from incubator.algorithms import PARAMS, defaults
from incubator.algorithms.patchcore import PatchCore
from incubator.algorithms.yolo import Yolo


def test_defaults_match_v1():
    assert defaults("PatchCore") == {"width": 640, "coreset": 20000, "erode": 5}
    assert defaults("YOLO") == {"model": "yolov8n.pt", "epochs": 30, "k": 3, "imgsz": 640}


def test_param_specs_have_ranges():
    for specs in PARAMS.values():
        for s in specs:
            if s.choices:
                assert s.default in s.choices
            else:
                assert s.lo < s.hi and s.lo <= s.default <= s.hi
    by_key = {s.key: s for s in PARAMS["YOLO"]}
    assert by_key["epochs"].lo >= 1 and by_key["k"].lo >= 2


def test_algorithms_accept_params():
    assert PatchCore({"width": 320, "coreset": 1000, "erode": 2}).params["width"] == 320
    assert PatchCore().params == defaults("PatchCore")
    y = Yolo({"epochs": 5})
    assert y.params["epochs"] == 5 and y.params["batch"] == 8 and y.params["k"] == 3
