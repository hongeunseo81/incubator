from incubator.algorithms.params import PARAMS, ParamSpec, defaults
from incubator.algorithms.patchcore import PatchCore
from incubator.algorithms.yolo import Yolo

REGISTRY = {"PatchCore": PatchCore, "YOLO": Yolo}
