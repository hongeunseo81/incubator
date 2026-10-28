"""단계 마법사: ① 알고리즘 → ② 데이터 준비 → ③ 파라미터 → ④ 실행·결과."""
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QSpinBox, QStackedWidget, QVBoxLayout, QWidget)

from impostor import dataset as dataset_mod
from impostor.algorithms import PARAMS, REGISTRY
from impostor.dataset import Dataset, Label
from impostor.align import reference_for
from impostor.readiness import readiness
from impostor.ui import theme
from impostor.ui.result_view import ResultView
from impostor.ui.thumb_grid import ThumbGrid

STEP_TITLES = ["① 알고리즘", "② 데이터 준비", "③ 실행·결과"]
RUN = 2   # 마지막 단계: 왼쪽 파라미터 + 오른쪽 실행·결과
DESCRIPTIONS = {
    "PatchCore": "이상탐지 — 정상 이미지만으로 학습합니다. 처음 보는 이물도 잡습니다.\n필요: 정상/간섭 라벨 (박스는 선택, 그리면 위치 적중 계산)",
    "YOLO": "검출기 — 간섭 박스로 학습합니다. 이물 위치를 박스로 알려줍니다.\n필요: 정상/간섭 라벨 + 간섭 박스 2장 이상",
}
VALUE_W = 460   # 데이터셋 콤보와 마스터 파일 경로 칸의 폭 (버튼 위치를 맞춘다)
OK_STYLE = f"color: {theme.OK}"
BAD_STYLE = f"color: {theme.ALERT}"


def _row_title(text: str) -> QLabel:
    """데이터셋 줄과 마스터 파일 줄의 이름표 폭을 맞춰 값 칸이 세로로 줄 서게 한다."""
    lab = QLabel(text)
    lab.setFixedWidth(80)
    return lab


class AlgoStep(QWidget):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.selected: str | None = None
        self.buttons: dict[str, QPushButton] = {}
        v = QVBoxLayout(self)
        v.addWidget(QLabel("<h3>어떤 알고리즘을 시험할까요?</h3>"))
        row = QHBoxLayout()
        for name in REGISTRY:
            b = QPushButton(f"{name}\n\n{DESCRIPTIONS.get(name, '')}")
            b.setCheckable(True)
            b.setMinimumSize(320, 160)
            b.setStyleSheet("QPushButton { text-align: left; padding: 12px; } "
                            f"QPushButton:checked {{ border: 2px solid {theme.ACCENT}; color: {theme.ACCENT}; }}")
            b.clicked.connect(lambda _=False, n=name: self.select(n))
            row.addWidget(b)
            self.buttons[name] = b
        row.addStretch()
        v.addLayout(row)
        v.addStretch()

    def select(self, name: str) -> None:
        for n, b in self.buttons.items():
            b.setChecked(n == name)
        if name != self.selected:
            self.selected = name
            self.changed.emit()


class DataStep(QWidget):
    changed = Signal()
    open_requested = Signal(str)
    new_scan_requested = Signal()
    import_requested = Signal()
    save_requested = Signal()

    def __init__(self):
        super().__init__()
        self.ds: Dataset | None = None
        self.algo: str | None = None
        self.size_bad: set[str] = set()   # 데이터셋을 열 때 한 번만 확인 (이미지 전부 디코딩이라 비쌈)
        self.mask_error = ""
        self.saved = QComboBox()
        self.saved.setFixedWidth(VALUE_W)
        self.saved.activated.connect(self._picked)
        self.bar_buttons = []
        bar = QHBoxLayout()
        bar.addWidget(_row_title("데이터셋"))
        bar.addWidget(self.saved)
        for text, sig in (("원본 폴더 스캔", self.new_scan_requested), ("폴더 가져오기", self.import_requested),
                          ("저장", self.save_requested)):
            b = QPushButton(text)
            b.clicked.connect(sig.emit)
            bar.addWidget(b)
            self.bar_buttons.append(b)
        bar.addStretch()
        self.hint = QLabel()
        # 데이터셋 줄과 같은 모양: 이름표 · 값 · 버튼 · (체크) · 상태
        self.master_path = QLineEdit()
        self.master_path.setReadOnly(True)
        self.master_path.setFixedWidth(VALUE_W)
        self.master_btn = QPushButton("변경")
        self.master_btn.clicked.connect(self._pick_master)
        self.align_check = QCheckBox("위치 보정")
        self.align_check.setToolTip("이미지마다 마스터와의 위치 차이를 재서 마스크를 옮겨 씁니다")
        self.align_check.toggled.connect(self._align_toggled)
        self.master_label = QLabel()
        self.bar_buttons_extra = [self.master_btn, self.align_check]
        master_row = QHBoxLayout()
        master_row.addWidget(_row_title("마스터 파일"))
        master_row.addWidget(self.master_path)
        master_row.addWidget(self.master_btn)
        master_row.addWidget(self.align_check)
        master_row.addWidget(self.master_label)
        master_row.addStretch()
        self.grid = ThumbGrid()
        self.grid.changed.connect(self._edited)
        self.status = QLabel()
        self.status.setWordWrap(True)
        v = QVBoxLayout(self)
        v.addLayout(bar)
        v.addLayout(master_row)
        v.addWidget(self.hint)
        v.addWidget(self.grid, 1)
        v.addWidget(self.status)
        self.refresh_saved()
        self.update_status()

    def refresh_saved(self) -> None:
        self.saved.blockSignals(True)
        self.saved.clear()
        self.saved.addItem("— 저장된 데이터셋 —", None)
        d = Path(dataset_mod.DATASET_DIR)
        for p in sorted(d.glob("*.json")) if d.is_dir() else []:
            self.saved.addItem(p.stem, str(p))
            if self.ds is not None and p.stem == self.ds.name:
                self.saved.setCurrentIndex(self.saved.count() - 1)
        self.saved.blockSignals(False)

    def latest_saved(self) -> str | None:
        d = Path(dataset_mod.DATASET_DIR)
        files = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime) if d.is_dir() else []
        return str(files[-1]) if files else None

    def _picked(self, idx: int) -> None:
        path = self.saved.itemData(idx)
        if path:
            self.open_requested.emit(path)

    def set_dataset(self, ds: Dataset) -> None:
        self.ds = ds
        try:
            self.size_bad = set(ds.check_sizes())
        except OSError as e:
            self.size_bad, self.mask_error = set(), str(e)
        else:
            self.mask_error = ""
        self.grid.set_dataset(ds)
        self._update_master_label()
        self.refresh_saved()
        self.update_status()

    def _update_master_label(self) -> None:
        if self.ds is None:
            self.master_path.setText("")
            self.master_label.setText("")
            return
        ref = reference_for(self.ds.mask_path)
        self.master_path.setText(self.ds.mask_path)
        self.master_path.setToolTip(self.ds.mask_path)
        self.align_check.blockSignals(True)
        self.align_check.setEnabled(ref is not None)
        self.align_check.setChecked(ref is not None and self.ds.align)
        self.align_check.blockSignals(False)
        if ref is None:
            state = f"<span style='color:{theme.ALERT}'>위치 보정 안 됨 — 옆에 마스터 이미지가 없음</span>"
        elif self.ds.align:
            state = f"<span style='color:{theme.OK}'>위치 보정 켜짐</span>"
        else:
            state = f"<span style='color:{theme.MUTED}'>위치 보정 꺼짐</span>"
        self.master_label.setText(state)

    def _align_toggled(self, on: bool) -> None:
        if self.ds is None or self.ds.align == on:
            return
        self.ds.align = on
        self.grid.set_dataset(self.ds)     # 미리보기도 보정 여부를 따른다
        self._update_master_label()
        self.changed.emit()

    def _pick_master(self) -> None:
        if self.ds is None:
            return
        f, _ = QFileDialog.getOpenFileName(self, "마스터 파일 (검사 영역 마스크)", str(Path(self.ds.mask_path).parent),
                                           "마스크 (*_IMG_Kungfu.png);;이미지 (*.png *.bmp)")
        if f:
            self.set_mask_path(f)

    def set_mask_path(self, path: str) -> None:
        """기존 데이터셋의 마스터 파일을 바꾼다 (크기 검사·위치 보정을 다시 준비)."""
        self.ds.mask_path = path
        self.set_dataset(self.ds)
        self.changed.emit()

    def set_algo(self, name: str | None) -> None:
        self.algo = name
        self.hint.setText("YOLO: 간섭 사진을 고르고 오른쪽 사진 위에 드래그해서 이물에 박스를 그리세요 (2장 이상)."
                          if name == "YOLO" else
                          "PatchCore: 목록에서 정상/간섭만 나누면 됩니다 (정상으로 학습, 간섭으로 채점). 박스는 선택.")
        self.update_status()

    def problems(self) -> list[str]:
        if not self.algo:
            return ["알고리즘을 고르세요"]
        p = readiness(self.ds, self.algo)
        if self.ds is not None and self.mask_error:
            p.append(self.mask_error)
        bad = []
        if self.ds is not None:
            bad = [i for i in self.ds.items if i.path in self.size_bad and i.label is not Label.UNLABELED]
        if bad:
            p.append(f"마스크와 크기가 다른 이미지 {len(bad)}개 — 미분류로 바꾸세요")
        return p

    def update_status(self) -> None:
        p = self.problems()
        self.status.setText("준비 완료 — 다음으로 넘어가세요" if not p else "준비 필요: " + " · ".join(p))
        self.status.setStyleSheet(OK_STYLE if not p else BAD_STYLE)

    def _params_changed(self) -> None:
        # 실행 화면 안에서 바꾸므로 바로 결과 화면에 반영 (값이 달라지면 이전 결과는 지워진다)
        if self.step == RUN:
            self.result.set_job(self.ds, self.algo.selected, self.params.values())
        else:
            self.result.clear()

    def _running(self, running: bool) -> None:
        self.params.setEnabled(not running)
        self._update_nav()

    def _edited(self) -> None:
        self.update_status()
        self.changed.emit()


class ParamStep(QWidget):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.algo: str | None = None
        self.widgets: dict[str, QWidget] = {}
        self.helps: dict[str, QLabel] = {}
        self.setMaximumWidth(320)
        self.title = QLabel()
        self.form_host = QWidget()
        self.reset_btn = QPushButton("기본값으로")
        self.reset_btn.clicked.connect(self.reset)
        v = QVBoxLayout(self)
        v.addWidget(self.title)
        v.addWidget(self.form_host)
        v.addWidget(self.reset_btn, 0, Qt.AlignLeft)
        v.addStretch()

    def set_algo(self, name: str) -> None:
        if name == self.algo:
            return
        self.algo = name
        self.title.setText(f"<h3>{name} 파라미터</h3>")
        old = self.form_host.layout()
        if old is not None:
            QWidget().setLayout(old)    # 기존 폼 버리기
        form = QVBoxLayout(self.form_host)        # 세로로 쌓아야 줄바꿈된 설명이 겹치지 않는다
        form.setContentsMargins(0, 0, 0, 0)
        self.widgets, self.helps = {}, {}
        for s in PARAMS[name]:
            if s.choices:
                w = QComboBox()
                w.addItems(list(s.choices))
                w.currentIndexChanged.connect(lambda _: self.changed.emit())
            else:
                w = QSpinBox()
                w.setRange(s.lo, s.hi)
                w.setSingleStep(s.step)
                w.valueChanged.connect(lambda _: self.changed.emit())
            hint = QLabel(s.help)
            hint.setWordWrap(True)
            hint.setStyleSheet(f"color: {theme.MUTED}; font-size: 11px")
            form.addWidget(QLabel(f"<b>{s.label}</b>"))
            form.addWidget(w)
            form.addWidget(hint)
            form.addSpacing(10)
            self.widgets[s.key] = w
            self.helps[s.key] = hint
        self.reset()

    def reset(self) -> None:
        for s in PARAMS[self.algo]:
            w = self.widgets[s.key]
            if s.choices:
                w.setCurrentText(s.default)
            else:
                w.setValue(s.default)

    def help_text(self, key: str) -> str:
        return self.helps[key].text()

    def values(self) -> dict:
        return {k: (w.currentText() if isinstance(w, QComboBox) else w.value()) for k, w in self.widgets.items()}


class Wizard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.step = 0
        self.algo = AlgoStep()
        self.data = DataStep()
        self.params = ParamStep()
        self.result = ResultView()
        run_page = QWidget()
        rp = QHBoxLayout(run_page)
        rp.setContentsMargins(0, 0, 0, 0)
        rp.addWidget(self.params)
        rp.addWidget(self.result, 1)
        self.stack = QStackedWidget()
        for w in (self.algo, self.data, run_page):
            self.stack.addWidget(w)

        self.step_labels = [QLabel(t) for t in STEP_TITLES]
        bar = QHBoxLayout()
        for lab in self.step_labels:
            bar.addWidget(lab)
        bar.addStretch()
        self.prev_btn = QPushButton("◀ 이전")
        self.next_btn = QPushButton("다음 ▶")
        self.prev_btn.clicked.connect(lambda: self.go(self.step - 1))
        self.next_btn.clicked.connect(lambda: self.go(self.step + 1))
        self.reason = QLabel()          # "다음"이 막힌 이유를 버튼 옆에 항상 보여준다
        self.reason.setStyleSheet(BAD_STYLE)
        nav = QHBoxLayout()
        nav.addStretch()
        nav.addWidget(self.reason)
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)
        v = QVBoxLayout(self)
        v.addLayout(bar)
        v.addWidget(self.stack, 1)
        v.addLayout(nav)

        self.algo.changed.connect(self._algo_changed)
        self.data.changed.connect(self._edited)
        self.params.changed.connect(self._params_changed)
        self.result.running_changed.connect(self._running)
        self._update_nav()

    # ---- 상태 ----
    @property
    def ds(self) -> Dataset | None:
        return self.data.ds

    def set_dataset(self, ds: Dataset) -> None:
        self.data.set_dataset(ds)
        self.result.clear()
        if self.step == RUN:
            # ③·④에서 다른 데이터셋을 열면 준비 검사부터 다시: 안 되면 ②로, 되면 ④를 새 데이터셋에 묶는다
            self.go(self.step)
        self._update_nav()

    def _algo_changed(self) -> None:
        name = self.algo.selected
        self.data.set_algo(name)
        self.params.set_algo(name)
        self.result.clear()
        self._update_nav()

    def _params_changed(self) -> None:
        # 실행 화면 안에서 바꾸므로 바로 결과 화면에 반영 (값이 달라지면 이전 결과는 지워진다)
        if self.step == RUN:
            self.result.set_job(self.ds, self.algo.selected, self.params.values())
        else:
            self.result.clear()

    def _running(self, running: bool) -> None:
        self.params.setEnabled(not running)
        self._update_nav()

    def _edited(self) -> None:
        self.result.clear()
        self._update_nav()

    def can_leave(self, step: int) -> bool:
        if step == 0:
            return self.algo.selected is not None
        if step == 1:
            return not self.data.problems()
        return False

    # ---- 이동 ----
    def go(self, i: int) -> None:
        if self.result.running or not 0 <= i < len(STEP_TITLES):
            return
        for k in range(i):          # 앞 단계 준비가 안 됐으면 거기서 멈춘다
            if not self.can_leave(k):
                i = k
                break
        if i == RUN:
            self.result.set_job(self.ds, self.algo.selected, self.params.values())
        self.step = i
        self.stack.setCurrentIndex(i)
        self._update_nav()
        if i == 1 and self.ds is None:
            latest = self.data.latest_saved()
            if latest:
                self.data.open_requested.emit(latest)

    def _update_nav(self) -> None:
        running = self.result.running
        self.prev_btn.setEnabled(self.step > 0 and not running)
        ok = self.can_leave(self.step)
        self.next_btn.setEnabled(self.step < RUN and ok and not running)
        why = "" if ok or self.step == RUN else (
            "알고리즘을 고르세요" if self.step == 0 else " · ".join(self.data.problems()))
        self.next_btn.setToolTip(why)
        self.reason.setText(why)
        summaries = [self.algo.selected, self.ds.name if self.ds else None, None]
        for k, lab in enumerate(self.step_labels):
            text = STEP_TITLES[k]
            if k < self.step and summaries[k]:
                text = f"{STEP_TITLES[k]}: {summaries[k]} ✓"
            lab.setText(text)
            lab.setStyleSheet("padding: 4px 12px; border-radius: 10px; "
                              + (f"background: {theme.ACCENT}; color: {theme.BG}; font-weight: bold;" if k == self.step
                                 else f"background: {theme.PANEL}; color: {theme.MUTED}; border: 1px solid {theme.GRID};"))
