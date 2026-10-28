"""메인 창: 라벨링 탭 + 비교 탭, 데이터셋 메뉴."""
import sys
from pathlib import Path

from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox, QTabWidget

from impostor.dataset import DATASET_DIR, Dataset, load_dataset, save_dataset
from impostor.ui.result_view import ResultView
from impostor.ui.dataset_dialogs import ImportDialog, ScanDialog
from impostor.ui.label_tab import LabelTab

APP_NAME = "Impostor"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.dataset: Dataset | None = None
        self.dirty = False
        self.resize(1500, 900)

        self.tabs = QTabWidget()
        self.label_tab = LabelTab()
        self.label_tab.changed.connect(lambda: self._set_dirty(True))
        self.compare_tab = ResultView()   # 임시 — Task 4 에서 마법사로 교체
        self.label_tab.changed.connect(self.compare_tab.clear)
        self.tabs.addTab(self.label_tab, "라벨링")
        self.tabs.addTab(self.compare_tab, "비교")
        self.setCentralWidget(self.tabs)

        menu = self.menuBar().addMenu("데이터셋")
        for text, slot, key in (("새로 만들기 (폴더 스캔)…", self.new_scan, None),
                                ("폴더 가져오기…", self.new_import, None),
                                ("열기…", self.open_dialog, QKeySequence.Open),
                                ("저장", self.save, QKeySequence.Save)):
            a = QAction(text, self)
            a.triggered.connect(slot)
            if key is not None:
                a.setShortcut(key)
            menu.addAction(a)
        self._update_title()

    # ---- 상태 ----
    def _update_title(self) -> None:
        name = f" — {self.dataset.name}" if self.dataset else ""
        self.setWindowTitle(f"{APP_NAME}{name}{' *' if self.dirty else ''}")

    def _set_dirty(self, v: bool) -> None:
        self.dirty = v
        self._update_title()

    def set_dataset(self, ds: Dataset, dirty: bool = False) -> None:
        self.dataset = ds
        self.label_tab.set_dataset(ds)
        self.compare_tab.clear()
        self._set_dirty(dirty)

    # ---- 메뉴 ----
    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        r = QMessageBox.question(self, APP_NAME, "저장하지 않은 변경이 있습니다. 저장할까요?",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Save:
            self.save()
            return True
        return r == QMessageBox.Discard

    def new_scan(self) -> None:
        self._new(ScanDialog(self))

    def new_import(self) -> None:
        self._new(ImportDialog(self))

    def _new(self, dlg) -> None:
        if self._confirm_discard() and dlg.exec() and dlg.dataset is not None:
            self.set_dataset(dlg.dataset, dirty=True)

    def open_dialog(self) -> None:
        if not self._confirm_discard():
            return
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        f, _ = QFileDialog.getOpenFileName(self, "데이터셋 열기", str(DATASET_DIR), "데이터셋 (*.json)")
        if f:
            self.open_dataset(f)

    def open_dataset(self, path: str) -> None:
        try:
            ds = load_dataset(path)
        except (OSError, ValueError, KeyError) as e:
            QMessageBox.warning(self, APP_NAME, f"데이터셋을 열 수 없습니다: {e}")
            return
        self.set_dataset(ds)
        miss = ds.missing()
        if miss:
            QMessageBox.warning(self, APP_NAME, f"{len(miss)}개 파일을 찾을 수 없습니다 — 실행에서 제외됩니다.\n"
                                                f"예: {miss[0].path}")

    def save(self) -> None:
        if self.dataset is None:
            return
        save_dataset(self.dataset, Path(DATASET_DIR) / f"{self.dataset.name}.json")
        self._set_dirty(False)

    def closeEvent(self, e) -> None:
        if not self._confirm_discard():
            e.ignore()
            return
        self.compare_tab.shutdown()
        super().closeEvent(e)


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
