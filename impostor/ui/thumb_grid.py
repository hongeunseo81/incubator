"""목록 보기: 폴더 이름 목록에서 여러 장을 골라 한 번에 정상/간섭/미분류로 지정한다.

그림은 미리 만들지 않는다(수백 장 디코딩이 화면을 멈추게 함). 지금 고른 한 장만 오른쪽에 미리보기.
"""
import numpy as np
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QBrush
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                               QPushButton, QSizePolicy, QVBoxLayout, QWidget)

from impostor.align import Aligner, describe, shift_mask
from impostor.dataset import Dataset, Item, Label, Pin
from impostor.imageio import imread, load_mask
from impostor.ui import theme
from impostor.ui.canvas import ImageCanvas
from impostor.ui.labels import KEY_LABEL, LABEL_COLOR, LABEL_TEXT, LIST_W, PIN_HELP, PIN_TEXT

FILTERS = ["전체", "정상", "간섭", "미분류"]


def mask_crop_box(mask: np.ndarray | None, pad: int = 20):
    if mask is None or not mask.any():
        return None
    ys, xs = np.where(mask)
    return (max(ys.min() - pad, 0), min(ys.max() + pad, mask.shape[0]),
            max(xs.min() - pad, 0), min(xs.max() + pad, mask.shape[1]))


class ThumbGrid(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ds: Dataset | None = None
        self.mask = None
        self.aligner: Aligner | None = None
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
        # 용도: 정상 사진을 학습/시험 중 어디에 쓸지 (지금 보는 사진이 정상일 때만 보임)
        self.pin_box = QWidget()
        pb = QHBoxLayout(self.pin_box)
        pb.setContentsMargins(16, 0, 0, 0)
        usage = QLabel("용도")
        usage.setToolTip(PIN_HELP)
        pb.addWidget(usage)
        self.pin_buttons: dict[Pin, QPushButton] = {}
        for p in (Pin.NONE, Pin.TRAIN, Pin.EVAL):
            b = QPushButton(PIN_TEXT[p])
            b.setCheckable(True)                 # 지금 보는 사진의 용도가 눌린 모양 (기본 자동)
            b.setToolTip(PIN_HELP)
            b.clicked.connect(lambda _=False, pp=p: self.apply_pin(pp))
            pb.addWidget(b)
            self.pin_buttons[p] = b
        self.pin_box.hide()
        top.addWidget(self.pin_box)

        self.view = QListWidget()
        self.view.setFixedWidth(LIST_W)      # 목록 폭 고정 → 미리보기 크기가 바뀌지 않는다
        self.view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.view.currentRowChanged.connect(self._preview)
        self.view.installEventFilter(self)
        self.preview = ImageCanvas()     # 휠 확대·축소, 가운데 버튼 이동. 확대 상태는 다음 이미지에도 유지
        self.preview.box_added.connect(self._box_added)       # 간섭 사진이면 바로 박스 그리기
        self.preview.box_deleted.connect(self._box_deleted)
        self.preview.setMinimumWidth(400)
        self.preview.clear_all("항목을 클릭하면 여기에 보입니다")
        self.shift_label = QLabel()
        self.shift_label.setStyleSheet(f"color: {theme.MUTED}")
        self.shift_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)   # 글자 길이·종류가 레이아웃을 밀지 않게
        self.shift_label.setFixedHeight(self.shift_label.fontMetrics().height() + 6)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self.preview, 1)
        rv.addWidget(self.shift_label)
        sp = QWidget()
        sl = QHBoxLayout(sp)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(self.view)
        sl.addWidget(right, 1)
        hint = QLabel("N 정상 · I 간섭 · U 미분류 (선택한 항목에 적용) · ↑/↓로 넘기며 보기 · Ctrl/Shift 클릭으로 여러 개 선택 · 간섭 사진은 오른쪽에서 드래그로 박스 그리기, 박스 클릭 후 Delete로 삭제 · 휠 확대/축소 · 가운데 버튼 이동")
        hint.setStyleSheet(f"color: {theme.MUTED}")

        v = QVBoxLayout(self)
        v.addLayout(top)
        v.addWidget(sp, 1)
        v.addWidget(hint)

    # ---- 데이터 ----
    def set_dataset(self, ds: Dataset) -> None:
        self.ds = ds
        try:
            mask = load_mask(ds.mask_path)
            self.mask, self.crop, self.mask_shape = mask, mask_crop_box(mask), mask.shape
        except OSError:
            self.mask, self.crop, self.mask_shape = None, None, None
        self.aligner = Aligner.for_dataset(ds)
        self.preview.clear_all("항목을 클릭하면 여기에 보입니다")
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
        usage = f"·{PIN_TEXT[it.pin]}" if it.pin is not Pin.NONE else ""
        boxes = f"·박스{len(it.boxes)}" if it.boxes else ""
        w.setText(f"[{LABEL_TEXT[it.label]}{usage}{boxes}] {it.group}{miss}")
        w.setForeground(QBrush(LABEL_COLOR[it.label]))

    def _update_counts(self) -> None:
        c = {lab: sum(i.label is lab for i in self.ds.items) for lab in Label}
        self.counts.setText(f"정상 {c[Label.NORMAL]} · 간섭 {c[Label.INTERFERENCE]} · 미분류 {c[Label.UNLABELED]}")

    def _preview(self, row: int) -> None:
        if not 0 <= row < len(self.shown):
            return
        it = self.shown[row]
        self.preview_path = it.path
        self._show_usage(it)
        try:
            img = imread(it.path)
        except OSError as e:
            self.preview.clear_all(str(e))
            return
        first = not self.preview.has_image()
        self.preview.set_image(img)            # 크기가 같으면 확대·위치 유지
        same = img.shape[:2] == self.mask_shape
        s = self.aligner.shift(img) if (self.aligner is not None and same) else None
        mask = shift_mask(self.mask, s[0], s[1]) if s is not None else self.mask
        self.preview.set_mask_outline(mask if same else None)
        self.preview.set_boxes(it.boxes)
        self.shift_label.setText(describe(s))
        if first and self.crop is not None and same:
            y0, y1, x0, x1 = self.crop
            self.preview.fit_rect(x0, y0, x1, y1)   # 처음엔 판(검사 영역)에 맞춰

    def eventFilter(self, obj, e) -> bool:
        # 목록이 글자 키로 항목을 찾지 않도록 먼저 가로챈다
        if obj is self.view and e.type() == QEvent.KeyPress and e.key() in KEY_LABEL:
            self.apply(KEY_LABEL[e.key()])
            return True
        if obj is self.view and e.type() == QEvent.KeyPress and e.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            return self.preview.delete_selected()
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
            self._show_current_usage()
        else:
            self._refill()
        self.changed.emit()

    def apply_pin(self, pin: Pin) -> None:
        rows = sorted({self.view.row(w) for w in self.view.selectedItems()})
        if not rows:
            return
        for r in rows:
            self.shown[r].pin = pin
            self._decorate(r)
        self._show_current_usage()
        self.changed.emit()

    def _show_usage(self, it: Item) -> None:
        self.pin_box.setVisible(it.label is Label.NORMAL)
        self.preview.editable = it.label is Label.INTERFERENCE     # 간섭 사진만 박스를 그린다
        for p, b in self.pin_buttons.items():
            b.setChecked(p is it.pin)

    def _show_current_usage(self) -> None:
        cur = self.view.currentRow()
        if 0 <= cur < len(self.shown):
            self._show_usage(self.shown[cur])

    def current(self) -> Item | None:
        r = self.view.currentRow()
        return self.shown[r] if 0 <= r < len(self.shown) else None

    def label_current(self, label: Label) -> None:
        """지금 보고 있는 사진 하나의 라벨을 바꾼다."""
        r = self.view.currentRow()
        if not 0 <= r < len(self.shown):
            return
        self.view.clearSelection()
        self.view.item(r).setSelected(True)
        self.apply(label)

    def _box_added(self, box) -> None:
        it = self.current()
        if it is None or it.label is not Label.INTERFERENCE:
            return
        it.add_box(box)
        self._after_box_change(it)

    def _box_deleted(self, idx: int) -> None:
        it = self.current()
        if it is None or not 0 <= idx < len(it.boxes):
            return
        del it.boxes[idx]
        self._after_box_change(it)

    def _after_box_change(self, it: Item) -> None:
        self.preview.set_boxes(it.boxes)
        self._decorate(self.view.currentRow())
        self.changed.emit()

    def rest_to_normal(self) -> None:
        if self.ds is not None and self.ds.unlabeled_to_normal():
            self._refill()
            self.changed.emit()
