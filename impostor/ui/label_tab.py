"""라벨링 탭: 이미지 목록 · 캔버스(박스 그리기) · 라벨/고정 버튼."""
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QGroupBox, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QPushButton, QRadioButton, QSplitter, QVBoxLayout, QWidget)

from impostor.dataset import Dataset, Item, Label, Pin
from impostor.imageio import imread, load_mask
from impostor.ui.canvas import ImageCanvas

LABEL_TEXT = {Label.NORMAL: "정상", Label.INTERFERENCE: "간섭", Label.UNLABELED: "미분류"}
LABEL_COLOR = {Label.NORMAL: QColor(60, 160, 60), Label.INTERFERENCE: QColor(210, 60, 60),
               Label.UNLABELED: QColor(140, 140, 140)}
PIN_TEXT = {Pin.NONE: "고정 없음", Pin.TRAIN: "항상 학습", Pin.EVAL: "항상 평가"}
FILTERS = ["전체", "정상", "간섭", "미분류", "파일 없음"]
KEY_LABEL = {Qt.Key_N: Label.NORMAL, Qt.Key_I: Label.INTERFERENCE, Qt.Key_U: Label.UNLABELED}


class LabelTab(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ds: Dataset | None = None
        self.mask = None
        self.shown: list[Item] = []

        self.filter = QComboBox()
        self.filter.addItems(FILTERS)
        self.filter.currentIndexChanged.connect(self._refill)
        self.counts = QLabel()
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._show)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(self.filter)
        lv.addWidget(self.counts)
        lv.addWidget(self.list)

        self.canvas = ImageCanvas(editable=True)
        self.canvas.box_added.connect(self._box_added)
        self.canvas.box_deleted.connect(self._box_deleted)

        right = QWidget()
        rv = QVBoxLayout(right)
        lab_box = QGroupBox("라벨")
        lb = QVBoxLayout(lab_box)
        self.label_buttons = {}
        for lab, key in ((Label.NORMAL, "N"), (Label.INTERFERENCE, "I"), (Label.UNLABELED, "U")):
            b = QPushButton(f"{LABEL_TEXT[lab]} ({key})")
            b.clicked.connect(lambda _=False, l=lab: self.set_label(l))
            lb.addWidget(b)
            self.label_buttons[lab] = b
        # 학습/시험 직접 지정 — 헷갈려서 화면에서는 숨긴다 (저장된 지정값은 그대로 적용됨)
        self.pin_box = pin_box = QGroupBox("분할 고정")
        pin_box.hide()
        pb = QVBoxLayout(pin_box)
        self.pin_group = QButtonGroup(self)
        self.pin_buttons = {}
        for p in Pin:
            r = QRadioButton(PIN_TEXT[p])
            r.clicked.connect(lambda _=False, pp=p: self.set_pin(pp))
            self.pin_group.addButton(r)
            pb.addWidget(r)
            self.pin_buttons[p] = r
        self.info = QLabel()
        self.info.setWordWrap(True)
        help_ = QLabel("드래그: 박스 그리기 (그리면 간섭)\n박스 클릭 후 Delete: 삭제\n←/→: 이전/다음\n휠: 확대, 가운데 버튼: 이동")
        help_.setStyleSheet("color: gray")
        for w in (lab_box, pin_box, self.info, help_):
            rv.addWidget(w)
        rv.addStretch()

        sp = QSplitter()
        sp.addWidget(left)
        sp.addWidget(self.canvas)
        sp.addWidget(right)
        sp.setStretchFactor(1, 1)
        sp.setSizes([280, 900, 200])
        QHBoxLayout(self).addWidget(sp)

        for w in (self.list, self.canvas):
            w.installEventFilter(self)

    # ---- 데이터 ----
    def set_dataset(self, ds: Dataset) -> None:
        self.ds = ds
        try:
            self.mask = load_mask(ds.mask_path)
        except OSError:
            self.mask = None
        self._refill()

    def select_item(self, item: Item) -> None:
        if self.filter.currentText() != "전체":
            self.filter.setCurrentText("전체")
        else:
            self._refill()
        if item in self.shown:
            self.list.setCurrentRow(self.shown.index(item))

    def current(self) -> Item | None:
        r = self.list.currentRow()
        return self.shown[r] if 0 <= r < len(self.shown) else None

    def _passes(self, it: Item) -> bool:
        f = self.filter.currentText()
        if f == "전체":
            return True
        if f == "파일 없음":
            return not self.ds.exists(it)
        return LABEL_TEXT[it.label] == f

    def _item_text(self, it: Item) -> str:
        file = Path(it.path)
        name = file.name if it.group == file.stem else f"{it.group} / {file.name}"
        tag = LABEL_TEXT[it.label] + (f" ·박스{len(it.boxes)}" if it.boxes else "")
        miss = "  [파일 없음]" if not self.ds.exists(it) else ""
        return f"[{tag}] {name}{miss}"

    def _decorate(self, row: int) -> None:
        it = self.shown[row]
        w = self.list.item(row)
        w.setText(self._item_text(it))
        w.setForeground(QBrush(LABEL_COLOR[it.label] if self.ds.exists(it) else QColor(120, 120, 120)))

    def _update_counts(self) -> None:
        c = {lab: sum(i.label is lab for i in self.ds.items) for lab in Label}
        miss = len(self.ds.missing())
        self.counts.setText(f"정상 {c[Label.NORMAL]} · 간섭 {c[Label.INTERFERENCE]} · "
                            f"미분류 {c[Label.UNLABELED]}" + (f" · 파일 없음 {miss}" if miss else ""))

    def _refill(self) -> None:
        if self.ds is None:
            return
        self.list.blockSignals(True)
        self.list.clear()
        self.shown = [i for i in self.ds.items if self._passes(i)]
        for _ in self.shown:
            self.list.addItem(QListWidgetItem())
        for r in range(len(self.shown)):
            self._decorate(r)
        self.list.blockSignals(False)
        self._update_counts()
        if self.shown:
            self.list.setCurrentRow(0)
            self._show(0)

    def _show(self, row: int) -> None:
        it = self.current()
        if it is None:
            self.canvas.clear_all()
            return
        if not self.ds.exists(it):
            self.canvas.clear_all("파일 없음: " + it.path)
        else:
            try:
                self.canvas.set_image(imread(it.path))
                self.canvas.set_mask_outline(self.mask)
                self.canvas.set_boxes(it.boxes)
            except OSError as e:
                self.canvas.clear_all(str(e))
        self.pin_buttons[it.pin].setChecked(True)
        self.info.setText(it.path)

    def _touched(self) -> None:
        r = self.list.currentRow()
        if r >= 0:
            self._decorate(r)
        self._update_counts()
        self.changed.emit()

    # ---- 동작 ----
    def set_label(self, label: Label) -> None:
        it = self.current()
        if it is not None and it.label is not label:
            it.label = label
            self._touched()

    def set_pin(self, pin: Pin) -> None:
        it = self.current()
        if it is not None and it.pin is not pin:
            it.pin = pin
            self._touched()

    def _box_added(self, box) -> None:
        it = self.current()
        if it is None:
            return
        it.add_box(box)
        self.canvas.set_boxes(it.boxes)
        self._touched()

    def _box_deleted(self, idx: int) -> None:
        it = self.current()
        if it is None or not 0 <= idx < len(it.boxes):
            return
        del it.boxes[idx]
        self.canvas.set_boxes(it.boxes)
        self._touched()

    def step(self, d: int) -> None:
        r = self.list.currentRow() + d
        if 0 <= r < self.list.count():
            self.list.setCurrentRow(r)

    def eventFilter(self, obj: QObject, e: QEvent) -> bool:
        if e.type() == QEvent.KeyPress:
            k = e.key()
            if k in KEY_LABEL:
                self.set_label(KEY_LABEL[k])
                return True
            # ←/→ 는 어디서나 이동, ↑/↓ 는 목록이 스스로 처리
            if k == Qt.Key_Right or (k == Qt.Key_Down and obj is not self.list):
                self.step(1)
                return True
            if k == Qt.Key_Left or (k == Qt.Key_Up and obj is not self.list):
                self.step(-1)
                return True
            if k in (Qt.Key_Delete, Qt.Key_Backspace):
                return self.canvas.delete_selected()
        return super().eventFilter(obj, e)
