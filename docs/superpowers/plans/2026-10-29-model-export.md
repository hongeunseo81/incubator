# 모델 내보내기 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 실행 결과로 검증한 PatchCore/YOLO 모델을 현장용 ONNX 묶음으로 내보내고, 내보낸 묶음으로 다시 채점해 화면 점수와 일치하는지 확인한다.

**Architecture:** 알고리즘 실행 결과(`AlgoResult.model`)가 학습된 모델을 들고 있는다. `incubator/export.py`가 묶음(ONNX, manifest, 마스크, 마스터, report)을 쓰고, torch 없이 묶음만으로 채점하는 `incubator/runtime.py`로 검증한다. 결과 화면의 버튼이 별도 스레드에서 둘을 실행한다.

**Tech Stack:** Python 3.11, torch 2.5.1(cu121), torch.onnx(opset 17), ultralytics 8.4 export(nms=True), onnx, onnxruntime(CPU), onnxslim, OpenCV, PySide6

**Spec:** `docs/superpowers/specs/2026-10-29-model-export-design.md`

## Global Constraints

- 내보내는 모델은 실행에서 검증한 그 모델이다. 재학습하지 않는다.
- 판정: 점수 ≥ threshold → 간섭. threshold = 내보낼 때 슬라이더 값.
- 묶음 폴더명: `{데이터셋}_{알고리즘}_{YYYYMMDD-HHMM}`
- manifest `format` = `"incubator-export"`, `version` = 1
- YOLO 이미지 점수 = 겹 모델 점수의 최댓값, ONNX 출력 [1, 300, 6], conf ≥ 0.001, 레터박스 패딩 114
- 검증 통과: 판정이 전부 같고 |차이| ≤ max(1e-3, 0.01 × threshold)
- `runtime.py`는 torch·ultralytics를 import하지 않는다
- 커밋에 Co-Authored-By 등 Claude 서명을 넣지 않는다

## Review Focus

- 위치 보정이 꺼졌거나 마스터를 못 찾은 데이터셋에서 내보내기 → `align: null`, master.png 없음, 채점은 보정 없이
- 위치 보정 박스가 이미지 가장자리를 넘는 경우 → 박스 이동 규칙(크기 유지, 안으로 밀기)이 툴과 같아야 함 (Task 2 테스트)
- 실행 중이거나 중단·실패한 결과에서 버튼 → 눌리지 않아야 함 (Task 4 테스트)
- 새 실행, 데이터·파라미터 변경, 앱 종료 시 YOLO 임시 가중치 폴더가 남지 않아야 함 (Task 1, 4 테스트)
- 한글 경로 저장 → cv2.imwrite 대신 imencode+tofile (Task 3 테스트는 한글 폴더명 사용)

---

### Task 1: 실행 결과가 학습된 모델을 보관

**Files:**
- Modify: `incubator/algorithms/base.py`, `incubator/algorithms/patchcore.py`, `incubator/algorithms/yolo.py`
- Test: `tests/test_patchcore.py`, `tests/test_yolo.py` (slow)

**Interfaces:**
- Produces:
  - `AlgoResult.model: object | None = None`, `AlgoResult.discard() -> None` (model에 `discard`가 있으면 호출)
  - `patchcore.PatchCoreModel` dataclass: `bank: np.ndarray` (N×1536 float32), `crop: tuple[int,int,int,int]` (y0,y1,x0,x1), `input_hw: tuple[int,int]`, `score_mask: np.ndarray` (bool, 침식), `params: dict`
  - `patchcore.PatchCoreGraph(torch.nn.Module)`: `__init__(bank: torch.Tensor)`, `forward(x) -> [H/8, W/8]` 거리 (matmul 식, 스파이크로 cdist 대비 4e-6 확인)
  - `yolo.YoloModel` dataclass: `weights: list[str]`, `root: str`, `imgsz: int`, `discard()` → root 삭제
  - `yolo.MODEL_PREFIX = "incubator_model_"`

- [ ] **Step 1: 실패하는 테스트** — `test_patchcore_keeps_model`: 합성 데이터 실행 후 `res.model.bank.shape[1] == 1536`, `len(bank) <= coreset`, `res.model.input_hw[1] == params["width"]`. `test_yolo_keeps_fold_weights`: k=2 실행 후 `len(res.model.weights) == 2`, 파일 존재, `res.discard()` 후 `root` 없음.
- [ ] **Step 2:** `pytest -m slow tests/test_patchcore.py tests/test_yolo.py -k keeps` → FAIL (`model` 없음)
- [ ] **Step 3:** 구현. PatchCore는 뱅크를 `.cpu().numpy()`로 보관, `input_hw`는 `_prep`과 같은 식. YOLO는 겹별 `runs/f/weights/last.pt`를 `tempfile.mkdtemp(prefix=MODEL_PREFIX)` 아래 `fold{n}.pt`로 복사한 뒤 학습 폴더를 지운다. 중단·실패 시 이 폴더도 지운다.
- [ ] **Step 4:** 같은 명령 → PASS, 빠른 테스트 전체 PASS
- [ ] **Step 5:** 커밋 `feat: 실행 결과에 학습된 모델 보관`

### Task 2: 묶음만으로 채점하는 runtime

**Files:**
- Create: `incubator/runtime.py`
- Test: `tests/test_runtime.py`

**Interfaces:**
- Produces:
  - `letterbox(img, s) -> (np.ndarray[1,3,s,s] float32, scale: float, pad: tuple[float,float])`
  - `unletterbox(xyxy: np.ndarray, scale, pad) -> np.ndarray`
  - `phase_shift(master_gray, img, region) -> (dx, dy, resp)`
  - `patchcore_score(dist, crop, score_mask, shape) -> (score, peak, full_map)`
  - `yolo_score(dets: np.ndarray[N,6] 원본 좌표, mask) -> (score, peak, boxes)` — 박스 정수화는 yolo.py `_predict`와 같은 규칙
  - `Bundle.load(path) -> Bundle`, `Bundle.score(img) -> Scored(score, peak, boxes, shift)`, `Bundle.threshold`, `Bundle.manifest`
- Consumes: `align.shift_box`, `align.shift_mask` (torch 무관)

- [ ] **Step 1: 실패하는 테스트** — `test_letterbox_geometry` (200×100 BGR, s=64 → scale 0.32, pad (0, 16), 출력 shape (1,3,64,64), 패딩 값 114/255), `test_unletterbox_inverse`, `test_patchcore_score_peak_in_mask` (거리 지도 한 칸만 큰 값, 그 위치가 score_mask 밖이면 무시), `test_yolo_score_ignores_boxes_outside_mask`, `test_phase_shift_recovers_translation` (합성 격자 이미지를 (7, -4) 이동 → ±0.5)
- [ ] **Step 2:** `pytest tests/test_runtime.py` → FAIL (모듈 없음)
- [ ] **Step 3:** 구현. `Bundle.score`는 manifest의 align → (PatchCore) crop 이동·리사이즈·정규화 → ORT 실행 → 후처리, (YOLO) 모델별 실행 후 최댓값.
- [ ] **Step 4:** PASS + 빠른 테스트 전체
- [ ] **Step 5:** 커밋 `feat: 내보낸 묶음 채점 runtime`

### Task 3: 묶음 쓰기와 검증

**Files:**
- Create: `incubator/export.py`
- Modify: `pyproject.toml` (onnx, onnxruntime, onnxslim 추가)
- Test: `tests/test_export.py` (slow)

**Interfaces:**
- Consumes: Task 1 `PatchCoreModel`, `PatchCoreGraph`, `YoloModel`; Task 2 `Bundle`; `readiness.Snapshot`; `metrics.Metrics`
- Produces:
  - `export_bundle(res: AlgoResult, snap: Snapshot, threshold: float, metrics: Metrics, out_root: Path, progress) -> Path`
  - `Verification` dataclass: `n: int`, `mismatched: int`, `max_diff: float`, `passed: bool`
  - `verify(bundle_dir: Path, res: AlgoResult, snap: Snapshot, progress) -> Verification` — 비교 대상은 spec "검증" 절, 결과를 report.json `verification`에 기록

- [ ] **Step 1: 실패하는 테스트** — `test_export_patchcore_roundtrip`: 합성 데이터 PatchCore 실행 → `export_bundle(..., tmp_path / "내보내기")` → manifest 필드(format, version, algorithm, threshold, image_size, crop, input), `model.onnx`·`mask.png`·`score_mask.png`·`report.json` 존재, `verify(...).passed` 그리고 `n == 간섭+임계값용+시험용 장수`. `test_export_yolo_roundtrip`: k=2, epochs 3 → `fold1.onnx`, `fold2.onnx`, `verify().n == 정상 임계값용+시험용`, passed. `test_export_without_align`: `ds.align=False` → `manifest["align"] is None`, master.png 없음.
- [ ] **Step 2:** `pytest -m slow tests/test_export.py` → FAIL
- [ ] **Step 3:** 구현. PatchCore ONNX: CPU에서 `torch.onnx.export(PatchCoreGraph(bank), zeros(1,3,H,W), opset_version=17, input_names=["image"], output_names=["distance"])`. YOLO: `YOLO(w).export(format="onnx", imgsz=imgsz, nms=True, conf=0.001)` 결과를 `fold{n}.onnx`로 옮긴다. PNG는 imencode+tofile.
- [ ] **Step 4:** PASS + 빠른 테스트 전체
- [ ] **Step 5:** 커밋 `feat: 모델 묶음 내보내기와 ONNX 재채점 검증`

### Task 4: 결과 화면 버튼

**Files:**
- Modify: `incubator/ui/worker.py`, `incubator/ui/result_view.py`
- Test: `tests/test_result_view.py`

**Interfaces:**
- Consumes: Task 3 `export_bundle`, `verify`, `Verification`
- Produces: `ExportWorker(QThread)` — `progress(str, float)`, `done(object)` (성공 `(Path, Verification)`, 실패 `str`); `ResultView.export_btn`, `ResultView.pick_dir() -> str` (QFileDialog, 테스트에서 monkeypatch), `ResultView.export()`

- [ ] **Step 1: 실패하는 테스트** — `test_export_button_disabled_without_model` (초기·skipped 결과), `test_export_button_enabled_after_result_with_model` (가짜 res.model), `test_clear_discards_model` (가짜 model.discard 호출 확인), `test_export_reports_path_and_verification` (export_bundle·verify monkeypatch, pick_dir → tmp, 완료 후 stage에 경로와 "일치" 포함), `test_export_cancelled_when_no_dir` (pick_dir → "" 이면 아무 일 없음)
- [ ] **Step 2:** `pytest tests/test_result_view.py` → FAIL
- [ ] **Step 3:** 구현. 버튼 "⇩  모델 내보내기"를 중단 버튼 옆에 둔다. 내보내는 동안 실행·내보내기 버튼을 잠근다. 불일치는 WARN 색.
- [ ] **Step 4:** PASS + 빠른 테스트 전체
- [ ] **Step 5:** 커밋 `feat(ui): 결과 화면 모델 내보내기 버튼`

### Task 5: 문서와 실제 데이터 확인

- [ ] README에 "모델 내보내기" 절(묶음 구성, 현장 적용 순서, spec 링크) 추가
- [ ] FEM_P2_LH로 PatchCore 실행→내보내기 스크립트 확인: 검증 통과, 묶음 크기 기록
- [ ] 전체 테스트(`pytest -m "slow or not slow"`) PASS
- [ ] 커밋 `docs: 모델 내보내기 안내`
