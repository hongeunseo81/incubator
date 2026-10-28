"""데이터셋 만들기 대화상자: 폴더 스캔 / 정리된 폴더 가져오기."""
from pathlib import Path

from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox, QPushButton, QTableWidget, QVBoxLayout, QWidget)

from impostor.dataset import Dataset, Label, find_image_kinds, find_mask_for, import_folders, scan_dataset
from impostor.ui.label_tab import LABEL_TEXT

DEFAULT_EXCLUDE = "master"


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
    """원본 스캔 폴더를 깊이와 상관없이 훑는다. 그룹 = 이미지가 들어 있는 회차 폴더."""

    def __init__(self, parent=None):
        super().__init__("새 데이터셋 (원본 폴더 스캔)", parent)
        self.exclude = QLineEdit(DEFAULT_EXCLUDE)
        self.exclude.setToolTip("폴더 이름에 이 단어가 들어 있으면 뺍니다 (쉼표로 여러 개, 예: master, calib)")
        self.exclude.editingFinished.connect(lambda: self.set_root(self.root.text()))
        self.kind = QComboBox()
        self.kind.setToolTip("회차 폴더마다 있는 카메라 이미지 중 어떤 것을 볼지")
        self.kind.currentIndexChanged.connect(lambda _: self._kind_changed())
        self.form.addRow("제외할 폴더", self.exclude)
        self.form.addRow("카메라 이미지", self.kind)
        self._layout()

    def excludes(self) -> list[str]:
        return [w.strip() for w in self.exclude.text().split(",") if w.strip()]

    def set_root(self, d: str) -> None:
        super().set_root(d)
        self.kind.blockSignals(True)
        self.kind.clear()
        for name, n in find_image_kinds(d, self.excludes()) if Path(d).is_dir() else []:
            self.kind.addItem(f"{name}  ({n}장)", name)
        self.kind.setCurrentIndex(0)      # 가장 많은 것 = 보통 검사 카메라
        self.kind.blockSignals(False)
        if not self.name.text() and Path(d).is_dir():
            self.name.setText(Path(d).name)
        self._kind_changed()

    def _kind_changed(self) -> None:
        # 같은 카메라의 마스터 마스크를 찾아 채운다 (직접 고른 값이 있어도 카메라를 바꾸면 다시 제안)
        kind = self.kind.currentData()
        m = find_mask_for(self.root.text(), kind) if kind else None
        if m:
            self.mask.setText(m)
        self.preview()

    def build(self) -> Dataset:
        kind = self.kind.currentData() or ""
        return scan_dataset(self.name.text().strip(), self.root.text(), f"**/{kind}", self.mask.text(),
                            self.excludes())

    def preview(self) -> None:
        if not self.kind.currentData():
            self.result_info.setText("루트 폴더에서 *Texture_8Bit.png 이미지를 찾지 못했습니다" if self.root.text() else "")
            return
        ds = self.build()
        groups = len({i.group for i in ds.items})
        self.result_info.setText(f"찾은 이미지 {len(ds.items)}장 (회차 폴더 {groups}개) — 처음엔 전부 미분류로 들어갑니다")


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
