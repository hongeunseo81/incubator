"""썸네일 격자: 여러 장을 한눈에 보고 골라서 한 번에 정상/간섭/미분류로 지정한다."""
import cv2
import numpy as np
from PySide6.QtCore import QSize, QTimer, Signal
from PySide6.QtGui import QBrush, QIcon
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QHBoxLayout, QLabel, QListView, QListWidget,
                               QListWidgetItem, QPushButton, QVBoxLayout, QWidget)

from impostor.dataset import Dataset, Item, Label
from impostor.imageio import imread, load_mask
from impostor.ui.canvas import to_pixmap
from impostor.ui.label_tab import LABEL_COLOR, LABEL_TEXT

THUMB_W = 220
BORDER = 6
BATCH = 8          # 한 번에 만드는 썸네일 수 — 나머지는 이어서 만들어 화면이 멈추지 않게
FILTERS = ["전체", "정상", "간섭", "미분류"]


def mask_crop_box(mask: np.ndarray | None, pad: int = 20):
    if mask is None or not mask.any():
        return None
    ys, xs = np.where(mask)
    return (max(ys.min() - pad, 0), min(ys.max() + pad, mask.shape[0]),
            max(xs.min() - pad, 0), min(xs.max() + pad, mask.shape[1]))


class ThumbGrid(QWidget):
    changed = Signal()
    open_item = Signal(object)   # Item — 더블클릭하면 한 장씩 보기로

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ds: Dataset | None = None
        self.crop = None
        self.mask_shape = None     # 크롭은 마스크와 크기가 같은 이미지에만 적용
        self.thumbs: dict[str, np.ndarray] = {}   # 경로 → 테두리 없는 썸네일(BGR)
        self.loaded = 0
        self.shown: list[Item] = []
        self._queue: list[Item] = []
        self._timer = QTimer(self)
        self._timer.setInterval(0)
        self._timer.timeout.connect(self._load_some)

        self.filter = QComboBox()
        self.filter.addItems(FILTERS)
        self.filter.currentIndexChanged.connect(self._refill)
        self.counts = QLabel()
        self.buttons: dict[Label, QPushButton] = {}
        top = QHBoxLayout()
        top.addWidget(QLabel("보기"))
        top.addWidget(self.filter)
        top.addWidget(self.counts)
        top.addStretch()
        top.addWidget(QLabel("선택한 이미지를"))
        for lab in (Label.NORMAL, Label.INTERFERENCE, Label.UNLABELED):
            b = QPushButton(LABEL_TEXT[lab])
            b.clicked.connect(lambda _=False, l=lab: self.apply(l))
            top.addWidget(b)
            self.buttons[lab] = b
        self.rest_normal_btn = QPushButton("미분류 전부 → 정상")
        self.rest_normal_btn.setToolTip("간섭만 골라 표시한 뒤 누르면 나머지가 모두 정상이 됩니다")
        self.rest_normal_btn.clicked.connect(self.rest_to_normal)
        top.addWidget(self.rest_normal_btn)

        self.view = QListWidget()
        self.view.setViewMode(QListView.IconMode)
        self.view.setResizeMode(QListView.Adjust)
        self.view.setMovement(QListView.Static)
        self.view.setIconSize(QSize(THUMB_W, THUMB_W))
        self.view.setSpacing(6)
        self.view.setUniformItemSizes(True)
        self.view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.view.itemDoubleClicked.connect(lambda w: self.open_item.emit(self.shown[self.view.row(w)]))
        hint = QLabel("드래그·Ctrl/Shift 클릭으로 여러 장 선택 · 더블클릭하면 크게 보고 박스 그리기")
        hint.setStyleSheet("color: gray")

        v = QVBoxLayout(self)
        v.addLayout(top)
        v.addWidget(self.view, 1)
        v.addWidget(hint)

    # ---- 데이터 ----
    def set_dataset(self, ds: Dataset) -> None:
        self.ds = ds
        try:
            mask = load_mask(ds.mask_path)
            self.crop, self.mask_shape = mask_crop_box(mask), mask.shape
        except OSError:
            self.crop, self.mask_shape = None, None
        self.thumbs = {}
        self.loaded = 0
        self._queue = list(ds.items)
        self._refill()
        self._timer.start()

    def refresh(self) -> None:
        """다른 화면에서 라벨이 바뀌었을 때 표시만 다시."""
        self._refill()

    def _make_thumb(self, it: Item) -> np.ndarray:
        if not self.ds.exists(it):
            img = np.full((THUMB_W * 3 // 4, THUMB_W, 3), 60, np.uint8)
            cv2.putText(img, "missing", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            return img
        try:
            img = imread(it.path)
        except OSError:
            return np.full((THUMB_W * 3 // 4, THUMB_W, 3), 60, np.uint8)
        if self.crop is not None and img.shape[:2] == self.mask_shape:
            y0, y1, x0, x1 = self.crop
            img = img[y0:y1, x0:x1]
        h = max(1, round(img.shape[0] * THUMB_W / img.shape[1]))
        return cv2.resize(img, (THUMB_W, h), interpolation=cv2.INTER_AREA)

    def _load_some(self) -> None:
        for _ in range(BATCH):
            if not self._queue:
                self._timer.stop()
                return
            it = self._queue.pop(0)
            self.thumbs[it.path] = self._make_thumb(it)
            self.loaded += 1
            if it in self.shown:
                self._decorate(self.shown.index(it))

    # ---- 표시 ----
    def _passes(self, it: Item) -> bool:
        f = self.filter.currentText()
        return f == "전체" or LABEL_TEXT[it.label] == f

    def _refill(self) -> None:
        if self.ds is None:
            return
        self.view.clear()
        self.shown = [i for i in self.ds.items if self._passes(i)]
        for _ in self.shown:
            self.view.addItem(QListWidgetItem())
        for r in range(len(self.shown)):
            self._decorate(r)
        self._update_counts()

    def _decorate(self, row: int) -> None:
        it = self.shown[row]
        w = self.view.item(row)
        w.setText(f"[{LABEL_TEXT[it.label]}] {it.group}")
        w.setForeground(QBrush(LABEL_COLOR[it.label]))
        thumb = self.thumbs.get(it.path)
        if thumb is not None:
            c = LABEL_COLOR[it.label]
            framed = cv2.copyMakeBorder(thumb, BORDER, BORDER, BORDER, BORDER, cv2.BORDER_CONSTANT,
                                        value=(c.blue(), c.green(), c.red()))
            w.setIcon(QIcon(to_pixmap(framed)))

    def _update_counts(self) -> None:
        c = {lab: sum(i.label is lab for i in self.ds.items) for lab in Label}
        self.counts.setText(f"정상 {c[Label.NORMAL]} · 간섭 {c[Label.INTERFERENCE]} · 미분류 {c[Label.UNLABELED]}")

    # ---- 동작 ----
    def apply(self, label: Label) -> None:
        rows = sorted({self.view.row(w) for w in self.view.selectedItems()})
        if not rows:
            return
        for r in rows:
            self.shown[r].label = label
        if self.filter.currentText() == "전체":
            for r in rows:
                self._decorate(r)
            self._update_counts()
        else:
            self._refill()
        self.changed.emit()

    def rest_to_normal(self) -> None:
        if self.ds is not None and self.ds.unlabeled_to_normal():
            self._refill()
            self.changed.emit()

