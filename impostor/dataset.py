"""데이터셋 = 원본 이미지 경로 + 라벨 + 박스. 이미지는 복사하지 않는다."""
import fnmatch
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from impostor.imageio import imsize, load_mask

DATASET_DIR = Path.home() / ".impostor" / "datasets"

Box = tuple[int, int, int, int]  # x0, y0, x1, y1 (원본 픽셀)


class Label(str, Enum):
    NORMAL = "normal"
    INTERFERENCE = "interference"
    UNLABELED = "unlabeled"


class Pin(str, Enum):
    NONE = "none"
    TRAIN = "train"
    EVAL = "eval"


@dataclass
class Item:
    path: str
    group: str
    label: Label = Label.UNLABELED
    boxes: list[Box] = field(default_factory=list)
    pin: Pin = Pin.NONE

    def add_box(self, box: Box) -> None:
        self.boxes.append(tuple(box))
        self.label = Label.INTERFERENCE


@dataclass
class Dataset:
    name: str
    mask_path: str
    items: list[Item] = field(default_factory=list)
    align: bool = True        # 이미지별 위치 보정 (마스터 이미지가 있을 때만 동작)

    def exists(self, item: Item) -> bool:
        return os.path.isfile(item.path)

    def missing(self) -> list[Item]:
        return [i for i in self.items if not self.exists(i)]

    def usable(self, label: Label) -> list[Item]:
        return [i for i in self.items if i.label is label and self.exists(i)]

    def unlabeled_to_normal(self) -> int:
        """표시 안 한 이미지를 전부 정상으로. 바꾼 개수."""
        n = 0
        for i in self.items:
            if i.label is Label.UNLABELED:
                i.label = Label.NORMAL
                n += 1
        return n

    def check_sizes(self) -> list[str]:
        """마스크와 크기가 다른(또는 읽을 수 없는) 이미지 경로."""
        shape = load_mask(self.mask_path).shape
        bad = []
        for i in self.items:
            if not self.exists(i):
                continue
            try:
                if imsize(i.path) != shape:
                    bad.append(i.path)
            except OSError:
                bad.append(i.path)
        return bad


def _match(name: str, name_glob: str) -> bool:
    return any(fnmatch.fnmatch(name, p.strip()) for p in name_glob.split(";") if p.strip())


TEXTURE_GLOB = "*Texture_8Bit.png"


def _excluded(path: Path, root: Path, exclude) -> bool:
    """루트 아래 경로의 폴더 이름에 exclude 단어가 들어 있으면 제외 (대소문자 무시). 예: Master, 3dvs_Master"""
    parts = [s.lower() for s in path.relative_to(root).parts[:-1]]
    return any(w.lower() in s for w in exclude if w for s in parts)


def find_image_kinds(root: str, exclude=()) -> list[tuple[str, int]]:
    """루트 아래(깊이 무관) Texture 이미지 파일 이름 종류와 개수. 예: fastenLHCam_02_IMG_Texture_8Bit.png"""
    counts: dict[str, int] = {}
    for p in Path(root).rglob(TEXTURE_GLOB):
        if p.is_file() and not _excluded(p, Path(root), exclude):
            counts[p.name] = counts.get(p.name, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))   # 많은 것(검사 카메라)부터


def find_mask_for(root: str, kind: str) -> str | None:
    """카메라 이미지 이름(…_IMG_Texture_8Bit.png)에 맞는 마스크(…_IMG_Kungfu.png)를 루트 아래에서 찾는다.
    master 가 들어간 폴더 중 가장 얕은 것을 우선한다 (회차 폴더마다 있는 마스크보다 마스터 것)."""
    if not kind.endswith("_IMG_Texture_8Bit.png"):
        return None
    name = kind.replace("_IMG_Texture_8Bit.png", "_IMG_Kungfu.png")
    found = sorted(Path(root).rglob(name),
                   key=lambda p: ("master" not in str(p).lower(), len(p.parts), str(p)))
    return str(found[0]) if found else None


def scan_dataset(name: str, root: str, pattern: str, mask_path: str, exclude=()) -> Dataset:
    paths = sorted(Path(root).glob(pattern))
    items = [Item(str(p), p.parent.name) for p in paths if p.is_file() and not _excluded(p, Path(root), exclude)]
    return Dataset(name, mask_path, items)


def import_folders(name: str, root: str, mapping: dict[str, Label], name_glob: str,
                   mask_path: str) -> Dataset:
    items = []
    for sub in sorted(Path(root).iterdir()):
        if not sub.is_dir() or sub.name not in mapping:
            continue
        for p in sorted(sub.iterdir()):
            if p.is_file() and _match(p.name, name_glob):
                items.append(Item(str(p), p.stem, mapping[sub.name]))
    return Dataset(name, mask_path, items)


def save_dataset(ds: Dataset, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": 1,
        "name": ds.name,
        "mask_path": ds.mask_path,
        "align": ds.align,
        "items": [{"path": i.path, "group": i.group, "label": i.label.value,
                   "boxes": [list(b) for b in i.boxes], "pin": i.pin.value} for i in ds.items],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def load_dataset(path) -> Dataset:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = [Item(d["path"], d["group"], Label(d["label"]), [tuple(b) for b in d["boxes"]],
                  Pin(d["pin"])) for d in data["items"]]
    return Dataset(data["name"], data["mask_path"], items, data.get("align", True))
