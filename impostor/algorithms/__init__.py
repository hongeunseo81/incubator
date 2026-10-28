from impostor.algorithms.params import PARAMS, ParamSpec, defaults
from impostor.algorithms.patchcore import PatchCore
from impostor.algorithms.yolo import Yolo

REGISTRY = {"PatchCore": PatchCore, "YOLO": Yolo}
