# Incubator v2 (단계 마법사) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** v1의 "라벨링 탭 + 비교 탭"을 "① 알고리즘 선택 → ② 데이터 준비 → ③ 파라미터 → ④ 실행·결과" 단계 마법사로 바꾸고, v1 최종 리뷰의 결함(실행 중 데이터 변경, 실패 사유 소실, YOLO 크기 검사)을 고친다.

**Architecture:** 코어에 알고리즘 파라미터 명세(`PARAMS`)·준비 상태 검사(`readiness`)·실행 스냅샷(`snapshot`)을 더한다. UI는 `Wizard`(QStackedWidget + 단계 표시줄 + 이전/다음)가 네 단계 위젯을 관리한다. ② 단계는 v1 `LabelTab`을 재사용하고, ④ 단계는 v1 `CompareTab`을 단일 알고리즘 `ResultView`로 줄인다.

**Tech Stack:** v1과 같음 (Python 3.11, PySide6, torch 2.5.1+cu121, ultralytics, pytest-qt)

**Spec:** `docs/superpowers/specs/2026-10-28-incubator-design.md` (9절 v2 개정이 우선)

## Global Constraints
- v1 Global Constraints 전부 유지 (`imageio.imread`만 사용, 한국어 UI, 원본 복사 금지, 60/20/20 그룹 분할, 점수 ≥ 임계값 → NG, 결과 비저장, 학습은 작업 스레드).
- 나란히 비교 없음: 결과 화면은 항상 알고리즘 1개, 실행 1회분.
- 실행 중에는 단계 이동·데이터셋 메뉴 불가.

## Review Focus
1. **실행 중 이전 단계로 가서 라벨 수정** — 막혀야 하고, 막히지 않더라도 결과는 실행 시점 사본 기준. → Task 2 `test_snapshot_isolated_from_edits`, Task 4 `test_navigation_locked_while_running`
2. **파라미터에 말이 안 되는 값** (epoch 0, 겹 수 1) — 입력 위젯 범위로 막는다. → Task 1 `test_param_specs_have_ranges`
3. **알고리즘 실패** (크기 다른 이미지 등) — 결과 화면에 사유가 남는다. → Task 3 `test_failure_reason_shown`
4. **YOLO 고른 뒤 박스 없이 다음** — 막히고 이유 표시. → Task 4 `test_next_blocked_without_boxes`
5. **결과 본 뒤 이전으로 가서 데이터 변경** — 결과가 지워진다. → Task 4 `test_results_cleared_after_edit`

---

### Task 1: 알고리즘 파라미터 명세 + YOLO 크기 검사

**Files:** Modify `incubator/algorithms/__init__.py`, `patchcore.py`, `yolo.py`; Create `tests/test_params.py`; Modify `tests/test_yolo.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) ParamSpec`: `key: str`, `label: str`, `default: int | str`, `choices: tuple[str, ...] | None = None`, `lo: int = 0`, `hi: int = 0`, `step: int = 1`
  - `PARAMS: dict[str, list[ParamSpec]]` (알고리즘 이름 → 명세), `defaults(name) -> dict`
    - PatchCore: `width` 입력 폭 640 [64, 2048] step 32 · `coreset` 메모리 크기 20000 [1000, 100000] step 1000 · `erode` 마스크 침식(px) 5 [0, 50]
    - YOLO: `model` 모델 choices ("yolov8n.pt", "yolov8s.pt") · `epochs` 30 [1, 500] · `k` 겹 수 3 [2, 10] · `imgsz` 입력 크기 640 [320, 1280] step 32
  - `PatchCore(params: dict | None = None)` — None이면 기본값. 모듈 상수 `WIDTH/CORESET/MASK_ERODE` 대신 `self.params` 사용.
  - `Yolo(params: dict | None = None)` — `FAST`와 합쳐 쓴다(`batch`는 FAST 값 고정).
  - Yolo: 마스크와 크기 다른 이미지 → `ValueError("마스크와 크기가 다른 이미지입니다: <경로>")`

- [ ] **Step 1: 실패하는 테스트**

```python
def test_defaults_match_v1(): defaults("PatchCore") == {"width":640,"coreset":20000,"erode":5}; defaults("YOLO") == {"model":"yolov8n.pt","epochs":30,"k":3,"imgsz":640}
def test_param_specs_have_ranges(): 모든 숫자형 spec 에 lo < hi 이고 lo <= default <= hi; epochs.lo >= 1; k.lo >= 2
def test_algorithms_accept_params(): PatchCore({"width":320,"coreset":1000,"erode":2}).params["width"] == 320; Yolo({"epochs":5}).params["epochs"] == 5 and Yolo().params["batch"] == 8
# test_yolo.py
def test_yolo_rejects_size_mismatch(tmp_path): 박스 간섭 2장 중 하나를 100x100 으로 덮어씀 → Yolo().run(...) 이 ValueError("마스크와 크기가 다른") (학습 전에 검사하므로 빠름)
```

- [ ] **Step 2: 실패 확인** — `pytest tests/test_params.py tests/test_yolo.py -q`
- [ ] **Step 3: 구현** — Yolo는 `run` 시작에서 평가·학습 대상 전부 크기 검사(학습 전에 실패해야 빠름).
- [ ] **Step 4: 통과 확인** — 같은 명령 + `pytest -m slow -q` (PatchCore·YOLO 기존 slow 테스트 유지)
- [ ] **Step 5: 커밋** — `feat(algo): 알고리즘 파라미터 명세·YOLO 크기 검사`

---

### Task 2: 준비 상태 검사 + 실행 스냅샷

**Files:** Create `incubator/readiness.py`, `tests/test_readiness.py`

**Interfaces:**
- Consumes: `Dataset`, `split_normals`, `SplitError`
- Produces:
  - `readiness(ds: Dataset | None, algo: str) -> list[str]` — 문제 문구 목록(비면 준비 완료): 데이터셋 없음 → `"데이터셋을 고르세요"`; `SplitError` → 그 메시지; 간섭 0 → `"간섭 이미지가 없습니다"`; `algo == "YOLO"`이고 박스 있는 간섭 < 2 → `"YOLO는 박스 있는 간섭이 2장 이상 필요합니다 (현재 N장)"`
  - `@dataclass Snapshot`: `ds: Dataset`, `split: Split` — `snapshot(ds) -> Snapshot` = `copy.deepcopy(ds)` 후 그 사본으로 `split_normals`

- [ ] **Step 1: 실패하는 테스트**

```python
def test_ready_patchcore(tmp_path): make_synthetic(n_normal=10, n_boxed=0, n_boxless=2) → readiness(ds,"PatchCore") == []
def test_yolo_needs_boxes(tmp_path): 같은 ds → readiness(ds,"YOLO") 에 "2장 이상" 포함
def test_no_dataset(): readiness(None,"PatchCore") == ["데이터셋을 고르세요"]
def test_split_error_reported(tmp_path): n_normal=2 → "최소 3개" 포함
def test_snapshot_isolated_from_edits(tmp_path):   # Review Focus 1
    s = snapshot(ds); ds.items[0].label = Label.UNLABELED; ds.items[-1].boxes.clear()
    assert s.ds.items[0].label is Label.NORMAL and s.ds.items[-1].boxes
```

- [ ] **Step 2: 실패 확인** — `pytest tests/test_readiness.py -q`
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인**
- [ ] **Step 5: 커밋** — `feat(core): 준비 상태 검사·실행 스냅샷`

---

### Task 3: 결과 화면 (단일 알고리즘 ResultView)

**Files:** Create `incubator/ui/result_view.py`, `tests/test_result_view.py`; Modify `incubator/ui/worker.py`; Delete `incubator/ui/compare_tab.py`, `tests/test_compare.py`

**Interfaces:**
- Consumes: `Snapshot`, `REGISTRY`, `AlgoResult`, `auto_threshold`, `evaluate`, `ImageCanvas`, `ThresholdRow`(compare_tab에서 옮김)
- Produces:
  - `RunWorker(snap: Snapshot, algo_cls, params: dict)` — 시그널 `progress(str, float)`, `done(AlgoResult)`; 예외·중단도 `done(AlgoResult(name, skipped="실행 실패: …" | "중단했습니다"))`로 보낸다. `request_stop()`.
  - `ResultView(QWidget)`: `set_job(ds: Dataset, algo: str, params: dict)`, `clear()`, `running: bool`(property), 시그널 `running_changed(bool)`, 위젯 `run_btn`, `stop_btn`, `table`(1행, 열 = v1 `COLUMNS`), `slider: ThresholdRow | None`, `images`(QTableWidget), `views: dict[str, ImageCanvas]` 키 `"원본"`과 알고리즘 이름
  - 실행: `snapshot(ds)` → 워커. 결과 평가·이미지 목록·캔버스는 **스냅샷만** 사용.
  - 슬라이더 범위 상한 = `max(max(scores), auto)`. 이미지 목록 = 간섭 + 평가 정상, **점수 내림차순**. 놓침 빨강, 오검출 주황.
  - `shutdown()` — 중단 요청 후 `wait()`.

- [ ] **Step 1: 실패하는 테스트** (v1 test_compare의 FakeAlgo·SlowFake·SkipAlgo 재사용, `REGISTRY` monkeypatch)

```python
def test_row_after_run(): 실행 → 놓침 "1/2", 오검출 "0/2", 처리 시간 "80 ms"
def test_slider_and_auto(): slider.setValue(0) → 놓침 "0/2"·오검출 "2/2"; auto_btn → "1/2"; FakeAlgo.calls == 1
def test_failure_reason_shown():  # Review Focus 3 — run 에서 ValueError("마스크와 크기가 다른 이미지입니다: x.png") 를 내는 알고리즘
    → table 에 "실행 실패" 와 "x.png" 포함
def test_skipped_reason_shown()
def test_images_sorted_by_score(): images 첫 행 점수 >= 둘째 행 점수
def test_result_uses_snapshot(): 실행 후 원본 ds.items 라벨 변경해도 table·images 그대로, 이미지 선택 시 예외 없음
def test_running_changed_signal(): 실행 시 True, 끝나면 False
```

- [ ] **Step 2: 실패 확인** — `pytest tests/test_result_view.py -q`
- [ ] **Step 3: 구현** — compare_tab의 `ThresholdRow`, 지표 계산, 이미지 목록·캔버스 코드를 단일 알고리즘으로 줄여 옮기고 compare_tab·test_compare 삭제.
- [ ] **Step 4: 통과 확인** — `pytest -q`
- [ ] **Step 5: 커밋** — `refactor(ui): 결과 화면을 단일 알고리즘 ResultView 로`

---

### Task 4: 단계 마법사 + 메인 창 결선

**Files:** Create `incubator/ui/wizard.py`, `tests/test_wizard.py`; Modify `incubator/ui/app.py`, `incubator/ui/label_tab.py`(데이터셋 바 추가 없음 — 마법사가 위에 둠), `tests/test_ui.py`(MainWindow 테스트를 마법사 기준으로)

**Interfaces:**
- Consumes: `PARAMS`, `defaults`, `readiness`, `LabelTab`, `ResultView`, `ScanDialog`, `ImportDialog`, `load_dataset`, `save_dataset`, `DATASET_DIR`
- Produces:
  - `AlgoStep(QWidget)`: 알고리즘 카드(QPushButton checkable, `REGISTRY` 순서, 설명 한 줄), `selected: str | None`, 시그널 `changed()`
  - `DataStep(QWidget)`: 위에 데이터셋 바(저장된 데이터셋 콤보 `DATASET_DIR/*.json` · `새로 만들기` · `폴더 가져오기` · `저장`), 아래 `LabelTab`, 맨 아래 준비 상태 문구(`readiness` 결과, 빨강/초록). `set_algo(name)`: YOLO면 "간섭 박스 그리기 필수" 안내 표시. 시그널 `changed()`.
  - `ParamStep(QWidget)`: `set_algo(name)` → `PARAMS[name]`로 폼 생성(숫자 = QSpinBox(lo, hi, step), choices = QComboBox), `values() -> dict`, `기본값으로` 버튼, 시그널 `changed()`.
  - `Wizard(QWidget)`: 단계 표시줄(4개 라벨, 현재 강조, 지난 단계에 선택 내용 요약 예 "① YOLO ✓"), `QStackedWidget`, `prev_btn`/`next_btn`, `step: int`, `go(i)`.
    - "다음" 활성 조건: 1단계 → 알고리즘 선택됨; 2단계 → `readiness(ds, algo) == []`; 3단계 → 항상. 막혔으면 버튼 툴팁·준비 상태 문구에 이유.
    - 4단계 진입 시 `ResultView.set_job(ds, algo, params)`.
    - `ResultView.running_changed(True)` 동안 `prev_btn`·단계 표시줄·데이터셋 바 비활성.
    - 2·3단계에서 무엇이든 바뀌면(`changed`) 또는 알고리즘을 바꾸면 `ResultView.clear()`.
  - `MainWindow`: 중앙 위젯 = `Wizard`. 메뉴 `데이터셋`은 유지하되 마법사 데이터셋 바와 같은 동작 호출, 실행 중 비활성. 제목·`*` 표시·닫을 때 저장 질문·`shutdown()` 유지. `label_tab` 속성은 `wizard.data.label_tab`로 노출(기존 테스트 호환).

- [ ] **Step 1: 실패하는 테스트**

```python
def test_next_blocked_until_algo_selected(qtbot)
def test_next_blocked_without_boxes(qtbot, ...):   # Review Focus 4 — YOLO 선택, 박스 0 → 2단계에서 next 비활성, 문구에 "2장 이상"
def test_patchcore_flow_reaches_result(qtbot, ...): PatchCore → 데이터셋 설정 → next 두 번 → step == 3, result.views 비어 있음(실행 전)
def test_param_form_follows_algo(qtbot): YOLO 선택 후 3단계 폼에 "epochs" 위젯, PatchCore 로 바꾸면 "coreset" 위젯
def test_navigation_locked_while_running(qtbot, slow_fake):   # Review Focus 1
    4단계 실행 클릭 → prev_btn 비활성 → 끝나면 활성
def test_results_cleared_after_edit(qtbot, fake):   # Review Focus 5
    실행 완료 → prev 로 2단계 → 라벨 변경 → 4단계 → result.table.rowCount() == 0
```

- [ ] **Step 2: 실패 확인** — `pytest tests/test_wizard.py -q`
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — `pytest -q`, `python -m incubator` 5초 실행 확인
- [ ] **Step 5: 커밋** — `feat(ui): 알고리즘 먼저 고르는 단계 마법사`

---

### Task 5: 마무리

**Files:** Modify `README.md`

- [ ] **Step 1:** `pytest -m "slow or not slow" -q` 전부 PASS
- [ ] **Step 2:** README의 "라벨링 탭 / 비교 탭" 절을 4단계 사용법으로 교체 (첫 검증 결과 절은 유지)
- [ ] **Step 3: 커밋** — `docs: README v2 사용법`
