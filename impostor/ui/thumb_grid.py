"""목록 보기: 폴더 이름 목록에서 여러 장을 골라 한 번에 정상/간섭/미분류로 지정한다.

그림은 미리 만들지 않는다(수백 장 디코딩이 화면을 멈추게 함). 지금 고른 한 장만 오른쪽에 미리보기.
"""
import cv2
import numpy as np
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QBrush
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                               QPushButton, QSizePolicy, QSplitter, QVBoxLayout, QWidget)

from impostor.dataset import Dataset, Item, Label
from impostor.imageio import imread, load_mask
from impostor.ui.canvas import to_pixmap
from impostor.ui.label_tab import KEY_LABEL, LABEL_COLOR, LABEL_TEXT

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
        self.shown: list[Item] = []
        self.preview_path: str | None = None

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
        top.addWidget(QLabel("선택한 항목을"))
        for lab, key in ((Label.NORMAL, "N"), (Label.INTERFERENCE, "I"), (Label.UNLABELED, "U")):
            b = QPushButton(f"{LABEL_TEXT[lab]} ({key})")
            b.clicked.connect(lambda _=False, l=lab: self.apply(l))
            top.addWidget(b)
            self.buttons[lab] = b
        self.rest_normal_btn = QPushButton("미분류 전부 → 정상")
        self.rest_normal_btn.setToolTip("간섭만 골라 표시한 뒤 누르면 나머지가 모두 정상이 됩니다")
        self.rest_normal_btn.clicked.connect(self.rest_to_normal)
        top.addWidget(self.rest_normal_btn)

        self.view = QListWidget()
        self.view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.view.currentRowChanged.connect(self._preview)
        self.view.itemDoubleClicked.connect(lambda w: self.open_item.emit(self.shown[self.view.row(w)]))
        self.view.installEventFilter(self)
        self.preview = QLabel("항목을 클릭하면 여기에 보입니다")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumWidth(400)
        self.preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.preview.setStyleSheet("background: #282828; color: #aaa")
        sp = QSplitter()
        sp.addWidget(self.view)
        sp.addWidget(self.preview)
        sp.setSizes([420, 800])
        hint = QLabel("N 정상 · I 간섭 · U 미분류 (선택한 항목에 적용) · ↑/↓로 넘기며 보기 · 드래그·Ctrl/Shift 클릭으로 여러 개 선택 · 더블클릭하면 크게 보고 박스 그리기")
        hint.setStyleSheet("color: gray")

        v = QVBoxLayout(self)
        v.addLayout(top)
        v.addWidget(sp, 1)
        v.addWidget(hint)

    # ---- 데이터 ----
    def set_dataset(self, ds: Dataset) -> None:
        self.ds = ds
        try:
            mask = load_mask(ds.mask_path)
            self.crop, self.mask_shape = mask_crop_box(mask), mask.shape
        except OSError:
            self.crop, self.mask_shape = None, None
        self._refill()

    def refresh(self) -> None:
        """다른 화면에서 라벨이 바뀌었을 때 표시만 다시."""
        self._refill()

    # ---- 표시 ----
    def _passes(self, it: Item) -> bool:
        f = self.filter.currentText()
        return f == "전체" or LABEL_TEXT[it.label] == f

    def _refill(self) -> None:
        if self.ds is None:
            return
        self.view.blockSignals(True)
        self.view.clear()
        self.shown = [i for i in self.ds.items if self._passes(i)]
        for _ in self.shown:
            self.view.addItem(QListWidgetItem())
        for r in range(len(self.shown)):
            self._decorate(r)
        self.view.blockSignals(False)
        self._update_counts()

    def _decorate(self, row: int) -> None:
        it = self.shown[row]
        w = self.view.item(row)
        miss = "  [파일 없음]" if not self.ds.exists(it) else ""
        w.setText(f"[{LABEL_TEXT[it.label]}] {it.group}{miss}")
        w.setForeground(QBrush(LABEL_COLOR[it.label]))

    def _update_counts(self) -> None:
        c = {lab: sum(i.label is lab for i in self.ds.items) for lab in Label}
        self.counts.setText(f"정상 {c[Label.NORMAL]} · 간섭 {c[Label.INTERFERENCE]} · 미분류 {c[Label.UNLABELED]}")

    def _preview(self, row: int) -> None:
        if not 0 <= row < len(self.shown):
            return
        it = self.shown[row]
        self.preview_path = it.path
        try:
            img = imread(it.path)
        except OSError as e:
            self.preview.setText(str(e))
            return
        if self.crop is not None and img.shape[:2] == self.mask_shape:
            y0, y1, x0, x1 = self.crop
            img = img[y0:y1, x0:x1]
        w = max(200, self.preview.width())
        h = max(1, round(img.shape[0] * w / img.shape[1]))
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        self.preview.setPixmap(to_pixmap(img))

    def eventFilter(self, obj, e) -> bool:
        # 목록이 글자 키로 항목을 찾지 않도록 먼저 가로챈다
        if obj is self.view and e.type() == QEvent.KeyPress and e.key() in KEY_LABEL:
            self.apply(KEY_LABEL[e.key()])
            return True
        return super().eventFilter(obj, e)

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
