"""비교 탭 (A안 스코어보드): 실행 · 성적표 · 임계값 슬라이더 · 이미지별 히트맵 나란히."""
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QCheckBox, QGroupBox, QHBoxLayout, QLabel, QMessageBox, QProgressBar,
                               QPushButton, QSlider, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from impostor.algorithms import REGISTRY
from impostor.algorithms.base import AlgoResult, eval_items
from impostor.dataset import Dataset, Label
from impostor.imageio import imread, load_mask
from impostor.metrics import Metrics, auto_threshold, evaluate
from impostor.split import Split, SplitError, split_normals
from impostor.ui.canvas import ImageCanvas
from impostor.ui.label_tab import LABEL_TEXT
from impostor.ui.worker import RunWorker

COLUMNS = ["알고리즘", "놓침", "오검출", "AUC", "임계값", "위치 적중", "처리 시간"]
STEPS = 1000
MISS_BG = QColor(255, 200, 200)
FP_BG = QColor(255, 225, 170)


class ThresholdRow(QWidget):
    changed = Signal(float)

    def __init__(self, name: str, lo: float, hi: float, auto: float):
        super().__init__()
        self.lo, self.hi, self.auto, self.value = lo, hi, auto, auto
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, STEPS)
        self.value_label = QLabel()
        self.auto_btn = QPushButton("자동")
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        name_label = QLabel(name)
        name_label.setMinimumWidth(90)
        for w in (name_label, self.slider, self.value_label, self.auto_btn):
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


class CompareTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ds: Dataset | None = None
        self.split: Split | None = None
        self.results: dict[str, AlgoResult] = {}
        self.metrics: dict[str, Metrics] = {}
        self.sliders: dict[str, ThresholdRow] = {}
        self.views: dict[str, ImageCanvas] = {}
        self.worker: RunWorker | None = None
        self.mask = None

        # 왼쪽 패널
        self.ds_label = QLabel("데이터셋 없음")
        self.ds_label.setWordWrap(True)
        self.counts = QLabel()
        self.counts.setWordWrap(True)
        algo_box = QGroupBox("알고리즘")
        av = QVBoxLayout(algo_box)
        self.algo_checks = {}
        for name in REGISTRY:
            c = QCheckBox(name)
            c.setChecked(True)
            av.addWidget(c)
            self.algo_checks[name] = c
        self.run_btn = QPushButton("실행")
        self.run_btn.clicked.connect(self.run)
        self.stop_btn = QPushButton("중단")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.stage = QLabel()
        self.stage.setWordWrap(True)
        left = QWidget()
        lv = QVBoxLayout(left)
        for w in (self.ds_label, self.counts, algo_box, self.run_btn, self.stop_btn, self.bar, self.stage):
            lv.addWidget(w)
        lv.addStretch()

        # 오른쪽: 성적표 + 슬라이더 + 이미지 목록 + 나란히 보기
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.slider_box = QVBoxLayout()
        sliders = QWidget()
        sliders.setLayout(self.slider_box)

        self.images = QTableWidget(0, 2)
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
        top = QWidget()
        tv = QVBoxLayout(top)
        tv.setContentsMargins(0, 0, 0, 0)
        tv.addWidget(self.table)
        tv.addWidget(sliders)
        right = QSplitter(Qt.Vertical)
        right.addWidget(top)
        right.addWidget(bottom)
        right.setSizes([260, 640])

        sp = QSplitter()
        sp.addWidget(left)
        sp.addWidget(right)
        sp.setSizes([220, 1300])
        QHBoxLayout(self).addWidget(sp)

    # ---- 데이터셋 ----
    def set_dataset(self, ds: Dataset) -> None:
        self.ds = ds
        self._clear_results()
        self.ds_label.setText(f"데이터셋: {ds.name}")
        try:
            self.mask = load_mask(ds.mask_path)
        except OSError:
            self.mask = None
        self._update_counts()

    def labels_changed(self) -> None:
        """라벨이 바뀌면 평가 대상이 달라지므로 이전 결과를 비운다 (실행 중이면 그대로 둔다)."""
        if self.worker is None:
            self._clear_results()
        self._update_counts()

    def _update_counts(self) -> None:
        if self.ds is None:
            return
        interf = self.ds.usable(Label.INTERFERENCE)
        boxed = sum(bool(i.boxes) for i in interf)
        text = f"간섭 {len(interf)} (박스 {boxed})"
        try:
            s = split_normals(self.ds)
            text += f"\n정상 학습 {len(s.train)} · 임계값 {len(s.thresh)} · 평가 {len(s.test)}"
        except SplitError as e:
            text += f"\n{e}"
        self.counts.setText(text)

    def _clear_results(self) -> None:
        self.results, self.metrics = {}, {}
        self.table.setRowCount(0)
        for row in self.sliders.values():
            row.setParent(None)
        self.sliders = {}
        self.images.setRowCount(0)
        self._set_views([])

    # ---- 실행 ----
    def run(self) -> None:
        if self.ds is None or self.worker is not None:
            return
        try:
            self.split = split_normals(self.ds)
        except SplitError as e:
            QMessageBox.warning(self, "실행할 수 없습니다", str(e))
            return
        algos = {n: REGISTRY[n] for n, c in self.algo_checks.items() if c.isChecked()}
        if not algos:
            return
        self._update_counts()
        self._clear_results()
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.bar.setValue(0)
        self.worker = RunWorker(self.ds, self.split, algos, self)
        self.worker.progress.connect(self._progress)
        self.worker.algo_done.connect(self._algo_done)
        self.worker.failed.connect(lambda msg: self.stage.setText(msg))
        self.worker.finished.connect(self._finished)
        self.worker.start()

    def stop(self) -> None:
        if self.worker is not None:
            self.worker.request_stop()
            self.stage.setText("중단 요청…")

    def shutdown(self) -> None:
        """창을 닫을 때: 작업 중단 요청 후 끝날 때까지 기다린다."""
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

    def _algo_done(self, res: AlgoResult) -> None:
        self.results[res.name] = res
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(res.name))
        if res.skipped:
            self.table.setItem(row, 1, QTableWidgetItem(res.skipped))
            self.table.setSpan(row, 1, 1, len(COLUMNS) - 1)
            return
        _, thresh, _ = eval_items(self.ds, self.split)
        values = list(res.scores.values())
        tr = ThresholdRow(res.name, min(values), max(values), auto_threshold(res.scores[i.path] for i in thresh))
        tr.changed.connect(lambda v, n=res.name: self._recompute(n, v))
        self.sliders[res.name] = tr
        self.slider_box.addWidget(tr)
        self._recompute(res.name, tr.value)
        self._rebuild_images()

    # ---- 지표 ----
    def _row_of(self, name: str) -> int:
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0).text() == name:
                return r
        return -1

    def _recompute(self, name: str, threshold: float) -> None:
        res = self.results[name]
        interf, _, test = eval_items(self.ds, self.split)
        m = evaluate({i.path: res.scores[i.path] for i in interf}, {i.path: res.scores[i.path] for i in test},
                     threshold, res.peaks, {i.path: i.boxes for i in interf})
        self.metrics[name] = m
        cells = [name, f"{m.missed}/{m.n_interference}", f"{m.false_pos}/{m.n_normal_test}",
                 "-" if m.auc is None else f"{m.auc:.3f}", f"{m.threshold:.4g}",
                 f"{m.hit}/{m.n_hit_eval}" if m.n_hit_eval else "-", f"{res.sec_per_image * 1000:.0f} ms"]
        r = self._row_of(name)
        for c, text in enumerate(cells):
            self.table.setItem(r, c, QTableWidgetItem(text))
        self._color_images()

    # ---- 이미지 목록 / 나란히 보기 ----
    def _scored(self) -> list[str]:
        return [n for n, r in self.results.items() if not r.skipped]

    def _eval_list(self):
        interf, _, test = eval_items(self.ds, self.split)
        return [(i, Label.INTERFERENCE) for i in interf] + [(i, Label.NORMAL) for i in test]

    def _rebuild_images(self) -> None:
        names = self._scored()
        rows = self._eval_list()
        self.images.blockSignals(True)
        self.images.setColumnCount(2 + len(names))
        self.images.setHorizontalHeaderLabels(["이미지", "라벨"] + names)
        self.images.setRowCount(len(rows))
        for r, (it, lab) in enumerate(rows):
            p = Path(it.path)
            name = p.name if it.group == p.stem else f"{it.group} / {p.name}"
            self.images.setItem(r, 0, QTableWidgetItem(name))
            self.images.setItem(r, 1, QTableWidgetItem(LABEL_TEXT[lab]))
            for c, n in enumerate(names):
                self.images.setItem(r, 2 + c, QTableWidgetItem(f"{self.results[n].scores[it.path]:.4g}"))
        self.images.blockSignals(False)
        self._color_images()
        self._set_views(names)

    def _color_images(self) -> None:
        names = self._scored()
        rows = self._eval_list() if self.split else []
        if self.images.rowCount() != len(rows):
            return
        for r, (it, lab) in enumerate(rows):
            for c, n in enumerate(names):
                cell = self.images.item(r, 2 + c)
                if cell is None or n not in self.metrics:
                    continue
                ng = self.results[n].scores[it.path] >= self.metrics[n].threshold
                bg = MISS_BG if (lab is Label.INTERFERENCE and not ng) else FP_BG if (lab is Label.NORMAL and ng) else None
                cell.setBackground(QBrush(bg) if bg else QBrush())

    def _set_views(self, names: list[str]) -> None:
        while self.view_row.count():
            w = self.view_row.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
        self.views = {}
        if not names:
            return
        for title in ["원본"] + names:
            box = QGroupBox(title)
            v = QVBoxLayout(box)
            canvas = ImageCanvas()
            v.addWidget(canvas)
            self.view_row.addWidget(box)
            self.views[title] = canvas

    def _show_selected(self) -> None:
        rows = self.images.selectionModel().selectedRows()
        if not rows or not self.views:
            return
        it, _ = self._eval_list()[rows[0].row()]
        try:
            img = imread(it.path)
        except OSError as e:
            for c in self.views.values():
                c.clear_all(str(e))
            return
        for title, canvas in self.views.items():
            canvas.set_image(img)
            canvas.set_mask_outline(self.mask)
            res = self.results.get(title)
            if res is not None:
                if it.path in res.maps:
                    vals = np.array(list(res.scores.values()))
                    canvas.set_heatmap(res.maps[it.path], float(vals.min()), float(vals.max()))
                if it.path in res.pred_boxes:
                    canvas.set_pred_boxes(res.pred_boxes[it.path])
            canvas.set_boxes(it.boxes)
