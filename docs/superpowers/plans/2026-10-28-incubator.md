# Incubator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 이물 간섭 검출 방법(PatchCore 이상탐지, YOLO)을 같은 데이터에 돌려 놓침·오검출·위치 적중을 나란히 비교하고, 툴 안에서 정상/간섭 라벨과 박스를 다는 PySide6 데스크톱 툴을 만든다.

**Architecture:** 순수 Python 코어(`dataset`, `split`, `metrics`, `algorithms`)와 PySide6 UI(`ui`)를 분리한다. 코어는 Qt 없이 pytest로 검증하고, UI는 코어를 작업 스레드에서 호출한다. 결과는 메모리에만 두고, 데이터셋 정의·라벨만 JSON으로 저장한다.

**Tech Stack:** Python 3.11, PySide6, torch 2.5.1+cu121 / torchvision 0.20.1, ultralytics (YOLOv8), opencv-python, numpy, scikit-learn, pytest, pytest-qt

**Spec:** `docs/superpowers/specs/2026-10-28-incubator-design.md`

## Global Constraints

- Windows 11, RTX 2080 Ti. 가상환경은 레포 안 `.venv` (`py -3.11 -m venv .venv`). torch는 `pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121`.
- UI 문구는 한국어.
- 원본 이미지는 복사하지 않는다. 데이터셋 JSON에는 원본 경로·라벨·박스·고정만 저장한다. (YOLO 학습용 임시 폴더는 실행 중 캐시이며 실행 후 지운다.)
- 이미지 읽기는 반드시 `incubator.imageio.imread` 하나로 한다 — 경로에 한글·연속 공백이 흔하다(`C:\DATA\EV\SeatLongRail\Source\CT2\간섭_L\2_7C  001227_...`). `cv2.imread` 직접 호출 금지.
- 라벨 3종: 정상 / 간섭 / 미분류. 미분류는 학습·평가 모두 제외.
- 정상 분할: 학습 60% / 임계값용 20% / 평가 20%, 고정 시드 0, 그룹 단위.
- 판정: 점수 ≥ 임계값 → NG(간섭). 자동 임계값 = 임계값용 정상 점수 최댓값 바로 위.
- 실행 기록은 저장하지 않는다. 결과는 앱이 켜져 있는 동안 메모리에만.
- 학습·추론은 UI 스레드에서 돌리지 않는다.

## Review Focus

1. **원본 파일이 사라졌거나 옮겨진 데이터셋** (2026-10-27에 실제로 SeatLongRail 소스를 이동했다) — 열 때 죽지 않고, 없는 파일 수를 보여주고, 실행에서 빼야 한다. → Task 1 `test_missing_files_excluded`
2. **마스크와 크기가 다른 이미지** (카메라가 다른 파일이 패턴에 섞임) — 해당 이미지 이름을 담은 오류로 알려야 하고, 실행 전체가 이상한 값을 내면 안 된다. → Task 1 `test_size_mismatch_reported`
3. **정상이 너무 적어 3분할이 안 되는 데이터셋** — "정상 그룹이 최소 3개 필요" 같은 분명한 메시지. → Task 2 `test_too_few_groups_raises`
4. **박스 있는 간섭이 2장 미만일 때 YOLO 실행** — 크래시 대신 성적표에 사유 표시. → Task 5 `test_yolo_skips_with_reason`
5. **학습 중 창 조작/닫기** — 작업 스레드가 돌고 있을 때 실행 버튼 비활성, 창 닫으면 중단 요청. → Task 8 `test_run_button_disabled_while_running`

---

## File Structure

```
incubator/
  pyproject.toml
  README.md
  incubator/
    __init__.py
    __main__.py              # python -m incubator → ui.app.main()
    imageio.py               # imread (한글 경로), 마스크 로드
    dataset.py               # Label, Pin, Item, Dataset, scan_dataset, import_folders, load/save
    split.py                 # Split, split_normals
    metrics.py               # auto_threshold, Metrics, evaluate, location_hit
    algorithms/
      __init__.py            # REGISTRY = {"PatchCore": ..., "YOLO": ...}
      base.py                # AlgoResult, Progress, eval_items
      patchcore.py
      yolo.py
    ui/
      app.py                 # QApplication, MainWindow
      boxes.py               # normalize_box (순수 함수)
      canvas.py              # ImageCanvas (QGraphicsView: 이미지·마스크 윤곽·박스·히트맵)
      label_tab.py
      dataset_dialogs.py     # 새 데이터셋(스캔), 폴더 가져오기
      compare_tab.py         # A안 스코어보드
      worker.py              # RunWorker (QThread)
  tests/
    conftest.py              # 합성 이미지·마스크 fixture
    test_dataset.py  test_split.py  test_metrics.py
    test_patchcore.py  test_yolo.py      # @pytest.mark.slow
    test_boxes.py  test_ui.py            # pytest-qt, offscreen
```

---

### Task 1: 프로젝트 골격 + 데이터셋 모델·입출력

**Files:**
- Create: `pyproject.toml`, `incubator/__init__.py`, `incubator/imageio.py`, `incubator/dataset.py`, `tests/conftest.py`, `tests/test_dataset.py`

**Interfaces:**
- Produces:
  - `imageio.imread(path: str) -> np.ndarray` (BGR uint8, `cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)`; 실패 시 `ImageReadError(path)`)
  - `imageio.load_mask(path: str) -> np.ndarray` (bool HxW, 픽셀 > 0)
  - `class Label(str, Enum)`: `NORMAL="normal"`, `INTERFERENCE="interference"`, `UNLABELED="unlabeled"`
  - `class Pin(str, Enum)`: `NONE="none"`, `TRAIN="train"`, `EVAL="eval"`
  - `@dataclass Item`: `path: str`, `group: str`, `label: Label = UNLABELED`, `boxes: list[tuple[int,int,int,int]]` (x0,y0,x1,y1), `pin: Pin = NONE`; 메서드 `add_box(box)` → 박스 추가 + `label = INTERFERENCE`
  - `@dataclass Dataset`: `name: str`, `mask_path: str`, `items: list[Item]`; `exists(item) -> bool`, `missing() -> list[Item]`, `usable(label: Label) -> list[Item]` (파일 있는 것만), `check_sizes() -> list[str]` (마스크와 크기 다른 이미지 경로)
  - `scan_dataset(name, root, pattern, mask_path) -> Dataset` — `Path(root).glob(pattern)` 정렬, `group` = 파일의 부모 폴더 이름, 라벨 전부 미분류
  - `import_folders(name, root, mapping: dict[str, Label], name_glob: str, mask_path) -> Dataset` — `root/<하위폴더>/` 중 mapping에 있는 것만, 파일명이 `name_glob`(fnmatch, `;`로 여러 패턴 OR)과 맞는 것만, `group` = 파일 stem
  - `save_dataset(ds, path)`, `load_dataset(path) -> Dataset` — JSON `{"version":1,"name","mask_path","items":[{"path","group","label","boxes","pin"}]}`, UTF-8, `ensure_ascii=False`
  - `DATASET_DIR = Path.home() / ".incubator" / "datasets"`

- [ ] **Step 1: 골격** — `pyproject.toml`(이름 `incubator`, 의존성 PySide6, ultralytics, opencv-python, numpy, scikit-learn; dev: pytest, pytest-qt; `[tool.pytest.ini_options] markers = ["slow: GPU/모델 다운로드 필요"]`, 기본 `-m "not slow"`). `.venv` 생성, torch는 Global Constraints 명령으로 설치, `pip install -e .[dev]`.

- [ ] **Step 2: 실패하는 테스트 작성** — `tests/conftest.py`에 `tmp_images` fixture: `tmp_path / "간섭 폴더" / "1_7C  0001"` 처럼 한글·연속공백 경로에 64×48 PNG 몇 장과 같은 크기 마스크(가운데 사각형 흰색)를 만든다(`cv2.imencode` + `tofile`로 저장).

```python
def test_imread_korean_path(tmp_images): assert imread(tmp_images.any_image).shape == (48, 64, 3)
def test_scan_groups_by_parent(tmp_images):
    ds = scan_dataset("t", tmp_images.root, "*/cam_03_IMG_Texture_8Bit.png", tmp_images.mask)
    assert {i.group for i in ds.items} == {"1_7C  0001", "1_7C  0002"}
    assert all(i.label is Label.UNLABELED for i in ds.items)
def test_import_maps_subfolders_and_filters(tmp_images):  # normal/, inside/, minor/ ; mapping minor→UNLABELED, name_glob "*__LH.png;*retake*" (RH 파일 1장은 빠져야 함)
    ...; assert counts == {Label.NORMAL: 2, Label.INTERFERENCE: 1, Label.UNLABELED: 1}
def test_add_box_sets_interference(): it = Item("a.png", "g"); it.add_box((1,2,10,20)); assert it.label is Label.INTERFERENCE
def test_json_roundtrip(tmp_path, tmp_images): save→load 후 items·boxes·pin·한글 경로 동일
def test_missing_files_excluded(tmp_images):   # Review Focus 1
    ds = ...; os.remove(ds.items[0].path)
    assert ds.missing() == [ds.items[0]] and ds.items[0] not in ds.usable(ds.items[0].label)
def test_size_mismatch_reported(tmp_images):   # Review Focus 2 — 100×100 이미지 하나 섞음
    assert ds.check_sizes() == [odd_path]
```

- [ ] **Step 3: 실행해 실패 확인** — `.venv\Scripts\pytest tests/test_dataset.py -v` → ImportError/FAIL
- [ ] **Step 4: `imageio.py`, `dataset.py` 구현**
- [ ] **Step 5: 통과 확인** — 같은 명령 → 전부 PASS
- [ ] **Step 6: 커밋** — `git commit -m "feat(core): 데이터셋 모델·스캔·폴더 가져오기·JSON 저장"`

---

### Task 2: 정상 3분할

**Files:** Create `incubator/split.py`, `tests/test_split.py`

**Interfaces:**
- Consumes: `Dataset`, `Item`, `Label`, `Pin` (Task 1)
- Produces: `@dataclass Split`: `train: list[Item]`, `thresh: list[Item]`, `test: list[Item]`; `split_normals(ds: Dataset, seed: int = 0, ratios=(0.6, 0.2, 0.2)) -> Split`; `class SplitError(ValueError)`

규칙: `ds.usable(Label.NORMAL)`만 대상. 그룹 단위로 나눈다. 그룹 안에 `Pin.TRAIN` 항목이 있으면 그 그룹은 train, `Pin.EVAL`이 있으면 test (둘 다 있으면 EVAL 우선). 고정 안 된 그룹을 `random.Random(seed).shuffle` 후 `n_thresh = max(1, round(n*0.2))`, `n_test = max(1, round(n*0.2)) - (이미 test로 고정된 그룹 수)`(0 미만이면 0), 나머지 train. train·thresh·test 중 하나라도 비면 `SplitError("정상 그룹이 부족합니다: 최소 3개 필요 (현재 N개)")`.

- [ ] **Step 1: 실패하는 테스트**

```python
def test_groups_never_straddle(): 10그룹×2장 → 각 그룹의 항목이 정확히 한 세트에만
def test_ratios_10_groups(): 10그룹 → train 6 / thresh 2 / test 2 그룹
def test_pins(): Pin.TRAIN 항목의 그룹은 train, Pin.EVAL 항목의 그룹은 test
def test_deterministic(): 같은 seed 두 번 → 같은 결과
def test_unlabeled_and_interference_ignored(): 미분류·간섭은 어느 세트에도 없음
def test_too_few_groups_raises(): 2그룹 → SplitError, 메시지에 "최소 3개"   # Review Focus 3
```

- [ ] **Step 2: 실패 확인** — `pytest tests/test_split.py -v`
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인**
- [ ] **Step 5: 커밋** — `feat(core): 정상 이미지 그룹 단위 3분할`

---

### Task 3: 평가 지표

**Files:** Create `incubator/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `auto_threshold(thresh_scores: Sequence[float]) -> float` = `m + max(1e-6, 1e-3*abs(m))`, `m = max(thresh_scores)`
  - `@dataclass Metrics`: `missed: int`, `n_interference: int`, `false_pos: int`, `n_normal_test: int`, `auc: float | None`, `hit: int`, `n_hit_eval: int`, `threshold: float`
  - `evaluate(interf: dict[str, float], normal_test: dict[str, float], threshold: float, peaks: dict[str, tuple[int,int] | None], gt_boxes: dict[str, list[box]]) -> Metrics`
    - missed = 간섭 중 `score < threshold`; false_pos = 정상 평가 중 `score >= threshold`
    - auc = `sklearn.metrics.roc_auc_score` (정상 평가=0, 간섭=1), 한쪽이 비면 None
    - 위치 적중: 박스가 있는 간섭만 분모(`n_hit_eval`), peak가 박스 중 하나 안(경계 포함)이면 hit
  - `location_hit(peak, boxes) -> bool`

- [ ] **Step 1: 실패하는 테스트**

```python
def test_auto_threshold_just_above_max(): t = auto_threshold([1.0, 2.5, 2.0]); assert 2.5 < t < 2.51
def test_auto_threshold_gives_zero_fp_on_its_own_set(): 임의 점수 → 같은 점수에 evaluate 시 false_pos == 0
def test_counts(): interf {a:3,b:1}, normal {c:0.5,d:2.6}, th 2.0 → missed 1, false_pos 1
def test_equal_score_is_ng(): score == threshold → NG (missed 아님, 정상이면 false_pos)
def test_auc_none_when_one_side_empty()
def test_location_hit_excludes_boxless(): 박스 없는 간섭은 n_hit_eval에서 빠짐
def test_location_hit_inside_and_outside()
```

- [ ] **Step 2: 실패 확인** — `pytest tests/test_metrics.py -v`
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인**
- [ ] **Step 5: 커밋** — `feat(core): 자동 임계값·놓침·오검출·AUC·위치 적중`

---

### Task 4: 알고리즘 공통 + PatchCore

**Files:** Create `incubator/algorithms/__init__.py`, `base.py`, `patchcore.py`, `tests/test_patchcore.py`

**Interfaces:**
- Consumes: `Dataset`, `Split`, `imread`, `load_mask`
- Produces:
  - `Progress = Callable[[str, float], None]` (단계 문구, 0~1). 작업 중단은 `should_stop: Callable[[], bool]` 인자로 받고, True면 `Cancelled` 예외.
  - `@dataclass AlgoResult`: `name: str`, `scores: dict[str, float]` (경로→점수, 평가 대상 전부: 간섭 + thresh + test), `maps: dict[str, np.ndarray]` (경로→float16 HxW 이상 지도, PatchCore만), `pred_boxes: dict[str, list[tuple[int,int,int,int,float]]]` (YOLO만), `peaks: dict[str, tuple[int,int] | None]`, `sec_per_image: float`, `skipped: str | None` (실행 안 한 사유)
  - `eval_items(ds, split) -> tuple[list[Item], list[Item], list[Item]]` = (간섭 usable, split.thresh, split.test)
  - `class Algorithm(Protocol)`: `name: str`; `run(ds, split, progress, should_stop) -> AlgoResult`
    (스펙 5.4의 train/score 대신 run 하나로 둔다 — YOLO k-겹은 학습과 점수를 겹마다 번갈아 해야 해서 분리할 수 없다.)
  - `REGISTRY: dict[str, type[Algorithm]]` = `{"PatchCore": PatchCore, "YOLO": Yolo}` (YOLO는 Task 5에서 추가)

PatchCore는 `C:\DATA\EV\FEM\clip\anomaly\code\patchcore.py`의 `Extractor`, `prep`, `coreset`, `score_map`, `to_full`, `load_mask`를 옮긴다. 값은 그대로: `WIDTH=640`, `BANK_PER_IMAGE=2000`, `CORESET=20000`, `MASK_ERODE=5`, bbox pad 20, GaussianBlur sigma 4, wide_resnet50_2 IMAGENET1K_V1 layer2+3. 경로·`HARD_NORMAL`·LH/RH 분기는 버린다(Pin과 데이터셋 분리로 대체). 학습 = `split.train`, 점수 = 침식 마스크 안 최댓값, peak = 그 최댓값 위치 (x, y). 시드 0 고정. `sec_per_image` = 점수 계산 구간 시간 / 이미지 수.

- [ ] **Step 1: 실패하는 테스트** (`@pytest.mark.slow`, 백본 다운로드 필요)

```python
def test_patchcore_flags_synthetic_blob(synthetic_ds):
    # 224×224 회색+잡음 정상 15그룹, 간섭 2장(마스크 안 40×40 흰 사각형, 박스 라벨 포함)
    r = PatchCore().run(ds, split_normals(ds), lambda *_: None, lambda: False)
    normal_max = max(r.scores[i.path] for i in split.thresh + split.test)
    assert all(r.scores[i.path] > normal_max for i in interf)
    assert all(location_hit(r.peaks[i.path], i.boxes) for i in interf)
    assert r.maps[interf[0].path].shape == (224, 224)
def test_patchcore_cancel(): should_stop → True 이면 Cancelled
```

- [ ] **Step 2: 실패 확인** — `pytest -m slow tests/test_patchcore.py -v`
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인**
- [ ] **Step 5: 커밋** — `feat(algo): PatchCore 이식`

---

### Task 5: YOLO (빠른 모드, k-겹)

**Files:** Create `incubator/algorithms/yolo.py`, `tests/test_yolo.py`; Modify `incubator/algorithms/__init__.py` (REGISTRY에 추가)

**Interfaces:**
- Consumes: Task 4의 `AlgoResult`, `Algorithm`, `eval_items`
- Produces: `class Yolo` (`name="YOLO"`), `FAST = dict(model="yolov8n.pt", epochs=30, imgsz=640, batch=8, k=3)`, `PRECISE = dict(model="yolov8s.pt", epochs=100, imgsz=960, batch=4, k=5)` (UI 노출 없음), 생성자 `Yolo(params=FAST)`

동작:
- 박스 있는 간섭 < 2 → `AlgoResult(skipped="박스 있는 간섭이 2장 미만이라 YOLO를 실행하지 않았습니다")` 반환.
- k = min(params.k, 박스 있는 간섭 수). 박스 있는 간섭을 시드 0으로 섞어 k겹.
- 겹마다 `tempfile.mkdtemp()` 아래 ultralytics 형식(images/train, labels/train, data.yaml, 클래스 1개 `"이물"`)으로 학습 세트 작성: 다른 겹의 박스 있는 간섭 + `split.train` 정상(빈 라벨 파일). 검증 세트는 학습 세트 그대로(빠른 모드). 이미지는 `imread` 후 `cv2.imencode(...).tofile`로 임시 폴더에 쓴다. 겹 끝나면 폴더 삭제.
- 점수: 예측 박스 중 마스크와 한 픽셀이라도 겹치는 박스의 최고 conf, 없으면 0.0. peak = 그 박스 중심.
  - 해당 겹의 간섭 → 그 겹 모델 점수
  - 박스 없는 간섭 → k개 모델 점수 평균 (pred_boxes·peak는 최고 점수 모델 것)
  - 정상(thresh, test) → k개 모델 점수 최댓값
- 학습 `conf=0.001`로 예측(낮은 점수까지 받아야 임계값 슬라이더가 의미 있음).
- `progress("YOLO 2/3겹 학습", 0.4)`처럼 겹·단계 표시. `should_stop`은 epoch 콜백(`model.add_callback("on_train_epoch_end", ...)`)에서 확인.
- 첫 실행 때 ultralytics가 사전학습 가중치를 내려받는다(인터넷 필요) — README에 적는다.

- [ ] **Step 1: 실패하는 테스트**

```python
def test_yolo_skips_with_reason(one_box_ds):   # Review Focus 4, 빠름(학습 안 함)
    r = Yolo().run(ds, split, noop, never); assert r.skipped and "2장 미만" in r.skipped and r.scores == {}
@pytest.mark.slow
def test_yolo_scores_every_eval_item(synthetic_ds):  # Task 4 fixture 재사용, 간섭 6장(박스 4, 박스 없음 2)
    r = Yolo(dict(FAST, epochs=3)).run(...)
    assert set(r.scores) == {i.path for i in interf + split.thresh + split.test}
    assert all(0.0 <= s <= 1.0 for s in r.scores.values())
@pytest.mark.slow
def test_yolo_cleans_temp_dirs(): 실행 전후 임시 폴더 수 동일
```

- [ ] **Step 2: 실패 확인** — `pytest tests/test_yolo.py -v` (빠른 것), `pytest -m slow tests/test_yolo.py -v`
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인**
- [ ] **Step 5: 커밋** — `feat(algo): YOLO 빠른 모드 k-겹 교차검증`

---

### Task 6: 이미지 캔버스 + 라벨링 탭

**Files:** Create `incubator/ui/boxes.py`, `ui/canvas.py`, `ui/label_tab.py`, `tests/test_boxes.py`, `tests/test_ui.py`

**Interfaces:**
- Consumes: `Dataset`, `Item`, `Label`, `Pin`, `imread`, `load_mask`
- Produces:
  - `normalize_box(x0, y0, x1, y1, w, h, min_size=3) -> tuple[int,int,int,int] | None` — 드래그 방향 무관하게 정렬, 이미지 밖은 잘라냄, 한 변이 `min_size` 미만이면 None
  - `class ImageCanvas(QGraphicsView)`: `set_image(bgr: np.ndarray)`, `set_mask_outline(mask: np.ndarray)`, `set_boxes(boxes, color)`, `set_heatmap(map: np.ndarray | None, vmin, vmax)` (jet, 알파 0.45), `set_pred_boxes(boxes_with_conf)`, `editable: bool`; 시그널 `box_added(tuple)`, `box_deleted(int)`. 휠 확대, 가운데 버튼 드래그 이동.
  - `class LabelTab(QWidget)`: `set_dataset(ds: Dataset)`; 시그널 `changed()` (저장 필요 표시용)

라벨링 탭 구성: 왼쪽 목록(필터 콤보 전체/정상/간섭/미분류/없는 파일, 항목 옆에 라벨 색, 위에 라벨별 개수) · 가운데 `ImageCanvas`(editable, 마스크 윤곽 표시) · 오른쪽 버튼(정상 / 간섭 / 미분류, 고정: 없음 / 항상 학습 / 항상 평가). 단축키: `N` 정상, `I` 간섭, `U` 미분류, `←`/`→` 이전/다음, `Delete` 선택 박스 삭제. 박스를 그리면 `Item.add_box` → 간섭. 없는 파일은 회색 + "파일 없음" 표시.

- [ ] **Step 1: 실패하는 테스트**

```python
def test_normalize_reverse_drag(): normalize_box(50,40,10,5,100,100) == (10,5,50,40)
def test_normalize_clamps(): normalize_box(-5,-5,200,200,100,80) == (0,0,100,80)
def test_normalize_too_small(): normalize_box(10,10,11,40,100,100) is None
def test_label_tab_shortcut_sets_normal(qtbot, small_ds):   # QT_QPA_PLATFORM=offscreen
    tab = LabelTab(); tab.set_dataset(ds); qtbot.keyClick(tab, Qt.Key_N); assert ds.items[0].label is Label.NORMAL
def test_box_added_marks_interference(qtbot, small_ds): canvas.box_added.emit((1,1,20,20)) → items[0].label is INTERFERENCE
def test_missing_file_does_not_crash(qtbot, small_ds): 첫 파일 지우고 set_dataset → 예외 없음, 목록에 "파일 없음"
```

- [ ] **Step 2: 실패 확인** — `pytest tests/test_boxes.py tests/test_ui.py -v`
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인**
- [ ] **Step 5: 커밋** — `feat(ui): 이미지 캔버스·라벨링 탭`

---

### Task 7: 메인 창 + 데이터셋 대화상자

**Files:** Create `incubator/__main__.py`, `ui/app.py`, `ui/dataset_dialogs.py`; Modify `tests/test_ui.py`

**Interfaces:**
- Consumes: Task 1의 `scan_dataset`, `import_folders`, `save_dataset`, `load_dataset`, `DATASET_DIR`; Task 6의 `LabelTab`
- Produces: `MainWindow(QMainWindow)` — 탭 `라벨링`(LabelTab), `비교`(Task 8의 CompareTab 자리, 이 Task에선 빈 위젯); `ui.app.main()`; `MainWindow.dataset: Dataset | None`, 시그널 `dataset_changed(Dataset)`

메뉴 `데이터셋`: 새로 만들기(스캔) / 폴더 가져오기 / 열기 / 저장(Ctrl+S). 저장 위치 `DATASET_DIR/<name>.json`. 저장 안 한 변경이 있으면 창 제목에 `*`, 닫을 때 저장 여부 질문.
- 새로 만들기: 이름, 루트 폴더(찾아보기), 파일 패턴(기본 `*/cam_03_IMG_Texture_8Bit.png`), 마스크 파일 → "미리보기" 누르면 찾은 개수 + `check_sizes()` 결과 표시.
- 폴더 가져오기: 루트 폴더 → 하위 폴더 목록을 표로(폴더명 | 라벨 콤보, 기본: `normal`→정상, `inside`/`edge`→간섭, 그 외 미분류), 파일명 필터(기본 `*.png`), 마스크.
- 데이터셋을 열 때 `missing()`이 있으면 "N개 파일을 찾을 수 없습니다 — 실행에서 제외됩니다" 알림.

- [ ] **Step 1: 실패하는 테스트**

```python
def test_main_window_opens(qtbot): w = MainWindow(); qtbot.addWidget(w); assert w.windowTitle().startswith("Incubator")
def test_open_dataset_warns_missing(qtbot, small_ds_json, monkeypatch): QMessageBox 가로채기 → 문구에 "찾을 수 없습니다"
def test_dirty_marker(qtbot, ...): 라벨 바꾸면 제목에 "*"
```

- [ ] **Step 2: 실패 확인** — `pytest tests/test_ui.py -v`
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인**, 그리고 `python -m incubator`로 창이 뜨는지 직접 확인
- [ ] **Step 5: 커밋** — `feat(ui): 메인 창·데이터셋 만들기/가져오기/열기/저장`

---

### Task 8: 비교 탭 (A안 스코어보드)

**Files:** Create `ui/worker.py`, `ui/compare_tab.py`; Modify `ui/app.py`(비교 탭 연결), `tests/test_ui.py`

**Interfaces:**
- Consumes: `split_normals`, `SplitError`, `REGISTRY`, `AlgoResult`, `auto_threshold`, `evaluate`, `ImageCanvas`
- Produces: `RunWorker(QThread)` — 생성자 `(ds, algo_names: list[str])`, 시그널 `progress(str, float)`, `algo_done(AlgoResult)`, `failed(str)`, 메서드 `request_stop()`; `CompareTab(QWidget)` — `set_dataset(ds)`

화면:
- 왼쪽 패널: 데이터셋 이름, 실행 대상 수(간섭 N / 정상 학습·임계값·평가), 알고리즘 체크박스(REGISTRY 순서), `실행` 버튼, 진행 막대 + 단계 문구. 실행 중엔 `실행` 비활성, `중단` 활성. `SplitError`는 메시지 상자로.
- 오른쪽 위 성적표(QTableWidget): 알고리즘 | 놓침 | 오검출 | AUC | 임계값 | 위치 적중 | 처리 시간. 형식: `1/15`, `0/30`, `0.998`, `2.65`, `12/15`, `80 ms`. `skipped`면 그 줄에 사유만.
- 성적표 아래 알고리즘마다 슬라이더 한 줄(이름 | 슬라이더 1000단계, 범위 = 그 알고리즘 점수 최소~최대 | 값 | `자동` 버튼). 처음 값 = `auto_threshold(thresh 점수)`. 움직이면 그 줄 지표와 이미지 목록 판정만 다시 계산(재실행 없음).
- 아래 왼쪽 평가 이미지 목록: 간섭 + 정상 평가(임계값용 제외), 열 = 이름 | 라벨 | 알고리즘별 점수. 놓침은 빨강, 오검출은 주황 배경.
- 아래 오른쪽: 선택 이미지의 `원본` + 알고리즘마다 `ImageCanvas` 하나씩 나란히(PatchCore = 히트맵, YOLO = 예측 박스, 모두 라벨 박스는 초록 테두리로). 확대·이동 동기화는 하지 않는다.
- 창을 닫으면 `request_stop()` 후 `wait()`.

- [ ] **Step 1: 실패하는 테스트** (가짜 알고리즘을 REGISTRY에 monkeypatch — 즉시 고정 점수 반환)

```python
def test_scoreboard_row_after_run(qtbot, labeled_ds, fake_algo): 실행 → 완료 대기 → 성적표 "Fake" 줄 놓침 "1/2"
def test_slider_recomputes_without_rerun(qtbot, ...): 슬라이더 최소로 → 놓침 "0/2", fake_algo.calls == 1
def test_run_button_disabled_while_running(qtbot, slow_fake_algo):   # Review Focus 5
    클릭 직후 run_btn.isEnabled() is False; 완료 후 True
def test_split_error_shown(qtbot, tiny_ds): 정상 2그룹 → 메시지에 "최소 3개"
def test_skipped_reason_shown(qtbot, ...): skipped AlgoResult → 그 줄에 사유 문구
```

- [ ] **Step 2: 실패 확인** — `pytest tests/test_ui.py -v`
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인**
- [ ] **Step 5: 커밋** — `feat(ui): 비교 탭 스코어보드·임계값 슬라이더·히트맵 나란히 보기`

---

### Task 9: FEM 실데이터 검증 + README

**Files:** Create `README.md`

- [ ] **Step 1: 전체 테스트** — `pytest -v` 와 `pytest -m slow -v` 모두 PASS
- [ ] **Step 2: FEM 데이터셋 만들기** — 앱에서 폴더 가져오기: 루트 `C:\DATA\EV\FEM\clip\anomaly\data`, normal→정상, inside·edge→간섭, minor→미분류, 파일명 필터 `*__LH.png;*retake*` (retake 2장 `cabletest__retake_*__unknown.png`은 이름에 LH가 없지만 LH 화각). 가져온 뒤 retake_NG가 간섭, retake_OK가 정상인지 확인하고 아니면 직접 고친다. 마스크 `data\masks\LH_cam02_master_mask.png`. 간섭 15장 확인.
- [ ] **Step 3: 고정 재현** — 기존 HARD_NORMAL(`001064_0002_261019_090031`, `001093_0032_261019_083842`, `001236_0001_261019_161007`, `retake_OK`)은 항상 평가, `260611_094322`는 항상 학습.
- [ ] **Step 4: PatchCore 실행** — 기대: 놓침 1~2/15, 오검출 0~1, AUC ≥ 0.95 (기존 14/15, 0/30, AUC 0.998 수준). 크게 벗어나면 이식 차이를 조사한다.
- [ ] **Step 5: 간섭 15장에 박스를 그리고 YOLO 실행** — 빠른 모드가 2080 Ti에서 수 분 안에 끝나는지, 성적표·히트맵 줄이 채워지는지 확인. 결과 수치를 README "첫 검증" 절에 기록.
- [ ] **Step 6: README** — 설치(venv, torch cu121 명령, `pip install -e .[dev]`), 실행(`python -m incubator`), 데이터셋 만들기 두 방식, 첫 YOLO 실행 때 가중치 다운로드(인터넷 필요), 첫 검증 결과.
- [ ] **Step 7: 커밋** — `docs: README·FEM 첫 검증 결과`
