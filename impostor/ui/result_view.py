"""④ 실행·결과: 알고리즘 1개, 실행 1회분. 결과는 실행 시점 스냅샷만 쓴다."""
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QGroupBox, QHBoxLayout, QLabel, QProgressBar, QPushButton, QSlider, QSplitter,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from impostor.algorithms import REGISTRY
from impostor.algorithms.base import AlgoResult, eval_items
from impostor.dataset import Dataset, Item, Label
from impostor.imageio import imread, load_mask
from impostor.metrics import auto_threshold, evaluate
from impostor.readiness import Snapshot, snapshot
from impostor.ui.canvas import ImageCanvas
from impostor.ui.label_tab import LABEL_TEXT
from impostor.ui.worker import RunWorker

COLUMNS = ["알고리즘", "놓침", "오검출", "AUC", "임계값", "위치 적중", "처리 시간"]
STEPS = 1000
MISS_BG = QColor(255, 200, 200)
FP_BG = QColor(255, 225, 170)


class ThresholdRow(QWidget):
    changed = Signal(float)

    def __init__(self, lo: float, hi: float, auto: float):
        super().__init__()
        self.lo, self.hi, self.auto, self.value = lo, hi, auto, auto
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, STEPS)
        self.value_label = QLabel()
        self.auto_btn = QPushButton("자동")
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        for w in (QLabel("임계값"), self.slider, self.value_label, self.auto_btn):
            h.addWidget(w)
        h.setStretch(1, 1)
        self.slider.valueChanged.connect(self._slid)
        self.auto_btn.clicked.connect(lambda: self.set_value(self.auto))
        self.set_value(auto)

    def _slid(self, pos: int) -> None:
        self.value = self.lo + (self.hi - self.lo) * pos / STEPS
        self._emit()

    def set_value(self, v: float) -> None:
        self.value = v
        span = self.hi - self.lo
        self.slider.blockSignals(True)
        self.slider.setValue(int(round((v - self.lo) / span * STEPS)) if span > 0 else 0)
        self.slider.blockSignals(False)
        self._emit()

    def _emit(self) -> None:
        self.value_label.setText(f"{self.value:.4g}")
        self.changed.emit(self.value)


class ResultView(QWidget):
    running_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ds: Dataset | None = None
        self.algo: str | None = None
        self.params: dict = {}
        self.snap: Snapshot | None = None
        self.res: AlgoResult | None = None
        self.threshold = 0.0
        self.rows: list[tuple[Item, Label]] = []
        self.slider: ThresholdRow | None = None
        self.views: dict[str, ImageCanvas] = {}
        self.worker: RunWorker | None = None
        self.mask = None

        self.title = QLabel()
        self.run_btn = QPushButton("실행")
        self.run_btn.clicked.connect(self.run)
        self.stop_btn = QPushButton("중단")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.stage = QLabel()
        top = QHBoxLayout()
        for w in (self.title, self.run_btn, self.stop_btn, self.bar, self.stage):
            top.addWidget(w)
        top.setStretch(3, 1)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setMaximumHeight(80)
        self.slider_box = QVBoxLayout()

        self.images = QTableWidget(0, 3)
        self.images.setHorizontalHeaderLabels(["이미지", "라벨", "점수"])
        self.images.setSelectionBehavior(QTableWidget.SelectRows)
        self.images.setSelectionMode(QTableWidget.SingleSelection)
        self.images.setEditTriggers(QTableWidget.NoEditTriggers)
        self.images.itemSelectionChanged.connect(self._show_selected)
        self.view_row = QHBoxLayout()
        views = QWidget()
        views.setLayout(self.view_row)
        bottom = QSplitter()
        bottom.addWidget(self.images)
        bottom.addWidget(views)
        bottom.setSizes([380, 1000])

        v = QVBoxLayout(self)
        v.addLayout(top)
        v.addWidget(self.table)
        v.addLayout(self.slider_box)
        v.addWidget(bottom, 1)

    # ---- 상태 ----
    @property
    def running(self) -> bool:
        return self.worker is not None

    def set_job(self, ds: Dataset, algo: str, params: dict) -> None:
        if ds is not self.ds or (algo, params) != (self.algo, self.params):
            self.clear()
        self.ds, self.algo, self.params = ds, algo, dict(params)
        self.title.setText(f"<b>{algo}</b> · {ds.name}")

    def clear(self) -> None:
        if self.running:
            return
        self.snap = self.res = None
        self.rows = []
        self.table.setRowCount(0)
        if self.slider is not None:
            self.slider.setParent(None)
            self.slider = None
        self.images.setRowCount(0)
        self._set_views([])
        self.stage.setText("")
        self.bar.setValue(0)

    # ---- 실행 ----
    def run(self) -> None:
        if self.ds is None or self.running:
            return
        self.clear()
        self.snap = snapshot(self.ds)
        try:
            self.mask = load_mask(self.snap.ds.mask_path)
        except OSError:
            self.mask = None
        self.worker = RunWorker(self.snap, self.algo, REGISTRY[self.algo], self.params, self)
        self.worker.progress.connect(self._progress)
        self.worker.done.connect(self._done)
        self.worker.finished.connect(self._finished)
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.worker.start()
        self.running_changed.emit(True)

    def stop(self) -> None:
        if self.worker is not None:
            self.worker.request_stop()
            self.stage.setText("중단 요청…")

    def shutdown(self) -> None:
        if self.worker is not None:
            self.worker.request_stop()
            self.worker.wait()

    def _progress(self, stage: str, frac: float) -> None:
        self.stage.setText(stage)
        self.bar.setValue(int(frac * 1000))

    def _finished(self) -> None:
        self.worker = None
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.bar.setValue(1000)
        self.running_changed.emit(False)

    def _done(self, res: AlgoResult) -> None:
        self.res = res
        self.table.setRowCount(1)
        self.table.setItem(0, 0, QTableWidgetItem(self.algo))
        if res.skipped:
            self.table.setItem(0, 1, QTableWidgetItem(res.skipped))
            self.table.setSpan(0, 1, 1, len(COLUMNS) - 1)
            self.stage.setText(res.skipped)
            return
        self.table.setSpan(0, 1, 1, 1)
        interf, thresh, test = eval_items(self.snap.ds, self.snap.split)
        values = list(res.scores.values())
        auto = auto_threshold(res.scores[i.path] for i in thresh)
        self.slider = ThresholdRow(min(values), max(max(values), auto), auto)
        self.slider.changed.connect(self._recompute)
        self.slider_box.addWidget(self.slider)
        rows = [(i, Label.INTERFERENCE) for i in interf] + [(i, Label.NORMAL) for i in test]
        self.rows = sorted(rows, key=lambda r: -res.scores[r[0].path])
        self._fill_images()
        self._set_views(["원본", self.algo])
        self._recompute(self.slider.value)

    # ---- 지표 ----
    def _recompute(self, threshold: float) -> None:
        self.threshold = threshold
        res = self.res
        interf = [i for i, lab in self.rows if lab is Label.INTERFERENCE]
        test = [i for i, lab in self.rows if lab is Label.NORMAL]
        m = evaluate({i.path: res.scores[i.path] for i in interf}, {i.path: res.scores[i.path] for i in test},
                     threshold, res.peaks, {i.path: i.boxes for i in interf})
        cells = [self.algo, f"{m.missed}/{m.n_interference}", f"{m.false_pos}/{m.n_normal_test}",
                 "-" if m.auc is None else f"{m.auc:.3f}", f"{m.threshold:.4g}",
                 f"{m.hit}/{m.n_hit_eval}" if m.n_hit_eval else "-", f"{res.sec_per_image * 1000:.0f} ms"]
        for c, text in enumerate(cells):
            self.table.setItem(0, c, QTableWidgetItem(text))
        self._color_images()

    # ---- 이미지 ----
    def _fill_images(self) -> None:
        self.images.blockSignals(True)
        self.images.setRowCount(len(self.rows))
        for r, (it, lab) in enumerate(self.rows):
            p = Path(it.path)
            name = p.name if it.group == p.stem else f"{it.group} / {p.name}"
            self.images.setItem(r, 0, QTableWidgetItem(name))
            self.images.setItem(r, 1, QTableWidgetItem(LABEL_TEXT[lab]))
            self.images.setItem(r, 2, QTableWidgetItem(f"{self.res.scores[it.path]:.4g}"))
        self.images.blockSignals(False)

    def _color_images(self) -> None:
        for r, (it, lab) in enumerate(self.rows):
            ng = self.res.scores[it.path] >= self.threshold
            bg = MISS_BG if (lab is Label.INTERFERENCE and not ng) else FP_BG if (lab is Label.NORMAL and ng) else None
            for c in range(3):
                self.images.item(r, c).setBackground(QBrush(bg) if bg else QBrush())

    def _set_views(self, titles: list[str]) -> None:
        while self.view_row.count():
            w = self.view_row.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
        self.views = {}
        for title in titles:
            box = QGroupBox(title)
            QVBoxLayout(box).addWidget(canvas := ImageCanvas())
            self.view_row.addWidget(box)
            self.views[title] = canvas

    def _show_selected(self) -> None:
        sel = self.images.selectionModel().selectedRows()
        if not sel or not self.views:
            return
        it, _ = self.rows[sel[0].row()]
        try:
            img = imread(it.path)
        except OSError as e:
            for c in self.views.values():
                c.clear_all(str(e))
            return
        vals = np.array(list(self.res.scores.values()))
        for title, canvas in self.views.items():
            canvas.set_image(img)
            canvas.set_mask_outline(self.mask)
            if title != "원본":
                if it.path in self.res.maps:
                    canvas.set_heatmap(self.res.maps[it.path], float(vals.min()), float(vals.max()))
                if it.path in self.res.pred_boxes:
                    canvas.set_pred_boxes(self.res.pred_boxes[it.path])
            canvas.set_boxes(it.boxes)
