"""데이터셋 만들기 대화상자: 폴더 스캔 / 정리된 폴더 가져오기."""
from pathlib import Path

from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox, QPushButton, QTableWidget, QVBoxLayout, QWidget)

from impostor.dataset import Dataset, Label, import_folders, scan_dataset
from impostor.ui.label_tab import LABEL_TEXT

DEFAULT_PATTERN = "*/cam_03_IMG_Texture_8Bit.png"


def default_label_for(folder: str) -> Label:
    f = folder.lower()
    if f == "normal":
        return Label.NORMAL
    if f in ("inside", "edge"):
        return Label.INTERFERENCE
    return Label.UNLABELED


def _path_row(edit: QLineEdit, pick) -> QWidget:
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    b = QPushButton("찾아보기")
    b.clicked.connect(pick)
    h.addWidget(edit)
    h.addWidget(b)
    return w


class _Base(QDialog):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(640, 420)
        self.name = QLineEdit()
        self.root = QLineEdit()
        self.mask = QLineEdit()
        self.form = QFormLayout()
        self.form.addRow("이름", self.name)
        self.form.addRow("루트 폴더", _path_row(self.root, self._pick_root))
        self.result_info = QLabel()
        self.result_info.setWordWrap(True)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)
        self.dataset: Dataset | None = None

    def _layout(self, *extra: QWidget) -> None:
        self.form.addRow("마스크 파일", _path_row(self.mask, self._pick_mask))
        v = QVBoxLayout(self)
        v.addLayout(self.form)
        for w in extra:
            v.addWidget(w)
        v.addWidget(self.result_info)
        v.addWidget(self.buttons)

    def _pick_root(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "루트 폴더", self.root.text())
        if d:
            self.set_root(d)

    def set_root(self, d: str) -> None:
        self.root.setText(d)

    def _pick_mask(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "마스크 파일", self.root.text(), "이미지 (*.png *.bmp *.jpg)")
        if f:
            self.mask.setText(f)

    def build(self) -> Dataset:
        raise NotImplementedError

    def _accept(self) -> None:
        if not self.name.text().strip() or not self.root.text() or not self.mask.text():
            QMessageBox.warning(self, "입력 확인", "이름, 루트 폴더, 마스크 파일을 모두 입력하세요.")
            return
        ds = self.build()
        if not ds.items:
            QMessageBox.warning(self, "입력 확인", "조건에 맞는 이미지가 없습니다.")
            return
        try:
            bad = ds.check_sizes()
        except OSError as e:
            QMessageBox.warning(self, "마스크 오류", str(e))
            return
        if bad and QMessageBox.question(
                self, "크기 불일치", f"마스크와 크기가 다른 이미지 {len(bad)}개가 있습니다 (예: {bad[0]}).\n"
                                    "이 이미지들을 빼고 진행할까요?") != QMessageBox.Yes:
            return
        ds.items = [i for i in ds.items if i.path not in set(bad)]
        self.dataset = ds
        self.accept()


class ScanDialog(_Base):
    """원본 스캔 폴더를 패턴으로 훑는다. 그룹 = 부모 폴더(회차)."""

    def __init__(self, parent=None):
        super().__init__("새 데이터셋 (폴더 스캔)", parent)
        self.pattern = QLineEdit(DEFAULT_PATTERN)
        self.form.addRow("파일 패턴", self.pattern)
        preview = QPushButton("미리보기")
        preview.clicked.connect(self.preview)
        self._layout(preview)

    def build(self) -> Dataset:
        return scan_dataset(self.name.text().strip(), self.root.text(), self.pattern.text(), self.mask.text())

    def preview(self) -> None:
        ds = self.build()
        msg = f"찾은 이미지 {len(ds.items)}개"
        if ds.items and self.mask.text():
            try:
                bad = ds.check_sizes()
                if bad:
                    msg += f" · 마스크와 크기가 다른 이미지 {len(bad)}개"
            except OSError as e:
                msg += f" · {e}"
        self.result_info.setText(msg)


class ImportDialog(_Base):
    """normal/inside/edge 처럼 하위 폴더로 나눠 둔 데이터를 가져온다. 그룹 = 파일 이름."""

    def __init__(self, parent=None):
        super().__init__("폴더 가져오기", parent)
        self.name_glob = QLineEdit("*.png")
        self.name_glob.setToolTip("여러 패턴은 ; 로 구분 (예: *__LH.png;*retake*)")
        self.form.addRow("파일 이름 필터", self.name_glob)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["하위 폴더", "라벨"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self._layout(self.table)

    def set_root(self, d: str) -> None:
        super().set_root(d)
        subs = sorted(p.name for p in Path(d).iterdir() if p.is_dir()) if Path(d).is_dir() else []
        self.table.setRowCount(len(subs))
        for r, s in enumerate(subs):
            self.table.setCellWidget(r, 0, QLabel(s))
            c = QComboBox()
            for lab in Label:
                c.addItem(LABEL_TEXT[lab], lab)
            c.setCurrentIndex(list(Label).index(default_label_for(s)))
            self.table.setCellWidget(r, 1, c)

    def mapping(self) -> dict[str, Label]:
        m = {}
        for r in range(self.table.rowCount()):
            m[self.table.cellWidget(r, 0).text()] = self.table.cellWidget(r, 1).currentData()
        return m

    def build(self) -> Dataset:
        return import_folders(self.name.text().strip(), self.root.text(), self.mapping(),
                              self.name_glob.text(), self.mask.text())
