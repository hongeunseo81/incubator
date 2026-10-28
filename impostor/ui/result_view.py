"""④ 실행·결과: 알고리즘 1개, 실행 1회분. 결과는 실행 시점 스냅샷만 쓴다."""
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QGroupBox, QHBoxLayout, QLabel, QProgressBar, QPushButton, QSizePolicy, QSlider, QSplitter,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from impostor.algorithms import REGISTRY
from impostor.algorithms.base import AlgoResult, eval_items
from impostor.align import describe, shift_mask
from impostor.dataset import Dataset, Item, Label
from impostor.imageio import imread, load_mask
from impostor.metrics import auto_threshold, evaluate
from impostor.readiness import Snapshot, snapshot
from impostor.split import SplitError
from impostor.ui.canvas import ImageCanvas
from impostor.ui import theme
from impostor.ui.labels import LABEL_TEXT
from impostor.ui.sections import section
from impostor.ui.worker import RunWorker

COLUMNS = ["놓침", "오검출", "AUC", "임계값", "위치 적중", "처리 시간"]
STEPS = 1000
MISS_BG = QColor("#4a1c24")      # 놓침 — 어두운 빨강
FP_BG = QColor("#4a3618")        # 오검출 — 어두운 주황


class ThresholdRow(QWidget):
    changed = Signal(float)

    def __init__(self):
        super().__init__()
        self.lo, self.hi, self.auto, self.value = 0.0, 1.0, 0.0, 0.0
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, STEPS)
        self.slider.setMinimumHeight(32)
        self.slider.setStyleSheet(
            f"QSlider::groove:horizontal {{ height: 10px; border-radius: 5px; "
            f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {theme.ACCENT}, stop:1 {theme.ALERT}); }} "
            f"QSlider::handle:horizontal {{ background: white; border: 3px solid {theme.ALERT}; "
            f"width: 22px; margin: -9px 0; border-radius: 13px; }} "
            f"QSlider::groove:horizontal:disabled {{ background: {theme.GRID}; }} "      # 결과 전 잠김 → 회색
            f"QSlider::handle:horizontal:disabled {{ background: #2a3a4e; border-color: #3a4a5e; }}")
        self.value_label = QLabel()
        self.value_label.setMinimumWidth(90)
        self.value_label.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {theme.ALERT}; "
                                       f"font-family: {theme.MONO}")
        self.auto_btn = QPushButton("자동")
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        for w in (self.slider, self.value_label, self.auto_btn):
            h.addWidget(w)
        h.setStretch(0, 1)
        self.slider.valueChanged.connect(self._slid)
        self.auto_btn.clicked.connect(lambda: self.set_value(self.auto))
        self.reset()

    def reset(self) -> None:
        """결과가 없을 때: 잠그고 안내만 보인다."""
        self.setEnabled(False)
        self.slider.blockSignals(True)
        self.slider.setValue(STEPS // 2)
        self.slider.blockSignals(False)
        self.value_label.setText("—")
        self.value_label.setStyleSheet(f"font-size: 14px; color: {theme.MUTED}")

    def configure(self, lo: float, hi: float, auto: float) -> None:
        self.lo, self.hi, self.auto = lo, hi, auto
        self.value_label.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {theme.ALERT}; "
                                       f"font-family: {theme.MONO}")
        self.setEnabled(True)
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
        self.slider = ThresholdRow()           # 항상 보인다 — 결과 전엔 잠김
        self.slider.changed.connect(self._recompute)
        self.views: dict[str, ImageCanvas] = {}
        self.worker: RunWorker | None = None
        self.mask = None

        self.title = QLabel()
        self.header = QLabel()                 # ■ SCAN COMPLETE · 알고리즘 · 데이터셋        ⚠ N TARGETS
        self.header.setFont(theme.mono(11, bold=True))
        self.banner = QLabel()                 # TARGET LOCKED / CLEAR
        self.banner.setFont(theme.mono(12, bold=True))
        self.banner.setAlignment(Qt.AlignCenter)
        self.banner.setFixedHeight(28)
        self.run_btn = QPushButton("▶  실행")
        self.run_btn.setMinimumSize(160, 44)
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self.run_btn.setStyleSheet(
            f"QPushButton {{ background: {theme.ACCENT}; color: {theme.BG}; font-size: 18px; font-weight: bold; "
            f"border: 0; border-radius: 8px; }} QPushButton:hover {{ background: #6fe3d8; }} "
            f"QPushButton:disabled {{ background: #1d3a44; color: #3f6470; }}")
        self.run_btn.clicked.connect(self.run)
        self.stop_btn = QPushButton("■  중단")
        self.stop_btn.setMinimumSize(110, 44)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.stage = QLabel()
        top = QHBoxLayout()
        for w in (self.run_btn, self.stop_btn, self.bar, self.stage):
            top.addWidget(w)
        top.setStretch(2, 1)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setMaximumHeight(80)
        self.slider_box = QVBoxLayout()
        self.slider_box.addWidget(self.slider)

        self.images = QTableWidget(0, 3)
        self.images.setHorizontalHeaderLabels(["이미지", "라벨", "점수"])
        self.images.setSelectionBehavior(QTableWidget.SelectRows)
        self.images.setSelectionMode(QTableWidget.SingleSelection)
        self.images.setEditTriggers(QTableWidget.NoEditTriggers)
        self.images.itemSelectionChanged.connect(self._show_selected)
        self.view_row = QHBoxLayout()
        self.shift_label = QLabel()
        self.shift_label.setStyleSheet(f"color: {theme.MUTED}")
        self.shift_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)   # 글자가 사진 영역을 밀지 않게
        self.shift_label.setFixedHeight(self.shift_label.fontMetrics().height() + 6)
        views = QWidget()
        vv = QVBoxLayout(views)
        vv.setContentsMargins(0, 0, 0, 0)
        vv.addLayout(self.view_row, 1)
        vv.addWidget(self.banner)
        vv.addWidget(self.shift_label)
        bottom = QSplitter()
        bottom.addWidget(self.images)
        bottom.addWidget(views)
        bottom.setSizes([380, 1000])

        run_box, rv = section("■ 실행")
        rv.addLayout(top)
        rv.addWidget(self.header)
        rv.addWidget(self.table)
        thr_box, tv = section("■ 임계값")
        tv.addLayout(self.slider_box)
        img_box, iv = section("■ 이미지")
        iv.addWidget(bottom, 1)
        self.sections = {"실행": run_box, "임계값": thr_box, "이미지": img_box}
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(10)
        v.addWidget(run_box)
        v.addWidget(thr_box)
        v.addWidget(img_box, 1)

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
        self.slider.reset()
        self.images.setRowCount(0)
        self._set_views([])
        self.stage.setText("")
        self.bar.setValue(0)
        self.header.setText("")
        self._set_banner(None, None)

    # ---- 실행 ----
    def run(self) -> None:
        if self.ds is None or self.running:
            return
        self.clear()
        try:
            self.snap = snapshot(self.ds)
        except SplitError as e:      # 준비 검사를 우회한 경우에도 사유를 남긴다
            self._done(AlgoResult(self.algo, skipped=str(e)))
            return
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
        if res.skipped:
            self.table.setItem(0, 0, QTableWidgetItem(res.skipped))
            self.table.setSpan(0, 0, 1, len(COLUMNS))
            self.stage.setText(res.skipped)
            return
        interf, thresh, test = eval_items(self.snap.ds, self.snap.split)
        values = list(res.scores.values())
        auto = auto_threshold(res.scores[i.path] for i in thresh)
        self.slider.blockSignals(True)            # 목록·화면을 다 준비한 뒤 한 번만 다시 계산
        self.slider.configure(min(values), max(max(values), auto), auto)
        self.slider.blockSignals(False)
        rows = [(i, Label.INTERFERENCE) for i in interf] + [(i, Label.NORMAL) for i in test]
        self.rows = sorted(rows, key=lambda r: -res.scores[r[0].path])
        self._fill_images()
        self._set_views(["원본", self.algo])
        self._recompute(self.slider.value)

    # ---- 지표 ----
    def _recompute(self, threshold: float) -> None:
        if self.res is None or self.res.skipped:
            return
        self.threshold = threshold
        res = self.res
        interf = [i for i, lab in self.rows if lab is Label.INTERFERENCE]
        test = [i for i, lab in self.rows if lab is Label.NORMAL]
        m = evaluate({i.path: res.scores[i.path] for i in interf}, {i.path: res.scores[i.path] for i in test},
                     threshold, res.peaks, {i.path: i.boxes for i in interf})
        cells = [f"{m.missed}/{m.n_interference}", f"{m.false_pos}/{m.n_normal_test}",
                 "-" if m.auc is None else f"{m.auc:.3f}", f"{m.threshold:.4g}",
                 f"{m.hit}/{m.n_hit_eval}" if m.n_hit_eval else "-", f"{res.sec_per_image * 1000:.0f} ms"]
        for c, text in enumerate(cells):
            self.table.setItem(0, c, QTableWidgetItem(text))
        self._color_images()
        targets = sum(self.res.scores[i.path] >= threshold for i, _ in self.rows)
        self.header.setText(f"<span style='color:{theme.ACCENT}'>■ SCAN COMPLETE · {self.algo} · {self.snap.ds.name}</span>"
                            f"&nbsp;&nbsp;&nbsp;<span style='color:{theme.ALERT}'>⚠ {targets} TARGETS</span>")
        self._show_selected()

    def _set_banner(self, score: float | None, threshold: float | None) -> None:
        if score is None:
            self.banner.setText("")
            self.banner.setStyleSheet("background: transparent")
            return
        if score >= threshold:
            self.banner.setText(f"TARGET LOCKED — 점수 {score:.3g} ≥ {threshold:.3g}")
            self.banner.setStyleSheet(f"background: {theme.ALERT}; color: {theme.BG}; border-radius: 4px")
        else:
            self.banner.setText(f"CLEAR — 점수 {score:.3g} < {threshold:.3g}")
            self.banner.setStyleSheet(f"background: transparent; color: {theme.ACCENT}; "
                                      f"border: 1px solid {theme.ACCENT}; border-radius: 4px")

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
        score = self.res.scores[it.path]
        locked = score >= self.threshold
        self._set_banner(score, self.threshold)
        vals = np.array(list(self.res.scores.values()))
        s = self.res.shifts.get(it.path)
        mask = shift_mask(self.mask, s[0], s[1]) if (s is not None and self.mask is not None) else self.mask
        self.shift_label.setText(describe(s))
        for title, canvas in self.views.items():
            canvas.set_image(img)
            canvas.set_mask_outline(mask)
            if title != "원본":
                if it.path in self.res.maps:
                    canvas.set_heatmap(self.res.maps[it.path], float(vals.min()), float(vals.max()))
                if it.path in self.res.pred_boxes:
                    canvas.set_pred_boxes(self.res.pred_boxes[it.path])
            canvas.set_boxes(it.boxes)
            canvas.set_target(self.res.peaks.get(it.path) if locked else None, locked)   # 정상 판정엔 조준선 없음
