"""한 장씩 탭: 위 한 줄(보기·라벨·용도) · 왼쪽 목록 · 오른쪽 사진(박스 그리기). 목록 화면과 같은 배치."""
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

from impostor.dataset import Dataset, Item, Label, Pin
from impostor.imageio import imread, load_mask
from impostor.ui.canvas import ImageCanvas

LABEL_TEXT = {Label.NORMAL: "정상", Label.INTERFERENCE: "간섭", Label.UNLABELED: "미분류"}
LABEL_COLOR = {Label.NORMAL: QColor(60, 160, 60), Label.INTERFERENCE: QColor(210, 60, 60),
               Label.UNLABELED: QColor(140, 140, 140)}
PIN_TEXT = {Pin.NONE: "자동", Pin.TRAIN: "학습", Pin.EVAL: "시험"}
PIN_HELP = "정상 사진을 학습/시험 중 어디에 쓸지. 보통은 자동 (예: 손글씨 판은 학습, 헷갈리는 정상은 시험)"
FILTERS = ["전체", "정상", "간섭", "미분류", "파일 없음"]
LIST_W = 420          # 목록 폭 — 목록 화면과 한 장씩 화면이 같게
KEY_LABEL = {Qt.Key_N: Label.NORMAL, Qt.Key_I: Label.INTERFERENCE, Qt.Key_U: Label.UNLABELED}


class LabelTab(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ds: Dataset | None = None
        self.mask = None
        self.shown: list[Item] = []

        # 맨 위 한 줄: 목록 화면과 같은 배치 — 보기 · 개수 · 라벨 버튼 · 용도
        self.top_bar = QWidget()
        top = QHBoxLayout(self.top_bar)
        top.setContentsMargins(0, 0, 0, 0)
        self.filter = QComboBox()
        self.filter.addItems(FILTERS)
        self.filter.currentIndexChanged.connect(self._refill)
        self.counts = QLabel()
        top.addWidget(QLabel("보기"))
        top.addWidget(self.filter)
        top.addWidget(self.counts)
        top.addStretch()
        top.addWidget(QLabel("선택한 항목을"))
        self.label_buttons = {}
        for lab, key in ((Label.NORMAL, "N"), (Label.INTERFERENCE, "I"), (Label.UNLABELED, "U")):
            b = QPushButton(f"{LABEL_TEXT[lab]} ({key})", self.top_bar)
            b.clicked.connect(lambda _=False, l=lab: self.set_label(l))
            top.addWidget(b)
            self.label_buttons[lab] = b
        # 용도: 정상 사진을 학습/시험 중 어디에 쓸지 (정상일 때만 보임)
        self.pin_box = QWidget()
        pb = QHBoxLayout(self.pin_box)
        pb.setContentsMargins(16, 0, 0, 0)
        usage = QLabel("용도")
        usage.setToolTip(PIN_HELP)
        pb.addWidget(usage)
        self.pin_buttons = {}
        for p in (Pin.NONE, Pin.TRAIN, Pin.EVAL):
            b = QPushButton(PIN_TEXT[p])
            b.setCheckable(True)
            b.setToolTip(PIN_HELP)
            b.clicked.connect(lambda _=False, pp=p: self.set_pin(pp))
            pb.addWidget(b)
            self.pin_buttons[p] = b
        top.addWidget(self.pin_box)

        self.list = QListWidget()
        self.list.setFixedWidth(LIST_W)      # 목록 폭 고정 → 사진 영역 크기가 바뀌지 않는다
        self.list.currentRowChanged.connect(self._show)
        self.canvas = ImageCanvas(editable=True)
        self.canvas.box_added.connect(self._box_added)
        self.canvas.box_deleted.connect(self._box_deleted)
        middle = QHBoxLayout()
        middle.addWidget(self.list)
        middle.addWidget(self.canvas, 1)

        self.info = QLabel()
        self.info.setStyleSheet("color: gray")
        self.info.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)   # 긴 경로가 레이아웃을 밀지 않게
        hint = QLabel("드래그: 박스 그리기 (그리면 간섭) · 박스 클릭 후 Delete: 삭제 · ←/→: 이전/다음 · "
                      "휠: 확대/축소 · 가운데 버튼: 이동")
        hint.setStyleSheet("color: gray")

        v = QVBoxLayout(self)
        v.addWidget(self.top_bar)
        v.addLayout(middle, 1)
        v.addWidget(self.info)
        v.addWidget(hint)

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
        self._show_usage(it)
        self.info.setText(it.path)

    def _show_usage(self, it: Item) -> None:
        self.pin_box.setVisible(it.label is Label.NORMAL)
        for p, b in self.pin_buttons.items():
            b.setChecked(p is it.pin)

    def _touched(self) -> None:
        r = self.list.currentRow()
        if r >= 0:
            self._decorate(r)
            self._show_usage(self.shown[r])
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
