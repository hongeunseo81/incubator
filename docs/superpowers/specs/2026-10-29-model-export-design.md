# 모델 내보내기 설계

## 의도

Incubator에서 실험하고 검증한 모델을 현장 앱(CleVisionSystems, C#)이 매번 학습하지 않고 그대로 판정에 쓸 수 있게 파일 묶음으로 내보낸다.
현장 PC에는 GPU가 있고, C# 쪽은 ONNX Runtime(GPU)으로 모델을 불러온다.

## 사용자가 정한 것

- ③ 실행·결과 화면에서 결과가 괜찮으면 **모델 내보내기**를 누른다. 지금 슬라이더 임계값, 마스크, 마스터가 모델과 함께 한 폴더에 저장되고, 그 폴더를 현장 PC에 복사하면 끝난다.
- PatchCore와 YOLO 둘 다 내보낼 수 있다.
- **방금 실행해서 검증한 모델을 그대로** 내보낸다. 전체 데이터로 다시 학습하지 않는다. 화면에서 본 성능과 임계값이 현장에서도 그대로 나와야 하기 때문이다.
- 내보낸 직후 ONNX로 다시 채점해서 화면 점수와 같은지 확인한다.
- CleVisionSystems 연동 코드는 이번 범위가 아니다(CVS 레포에서 별도 PR).

## 목표 / 비목표 / 반목표

- 목표: 묶음 하나만 있으면 현장 코드가 툴과 같은 점수, 같은 판정을 재현할 수 있다.
- 비목표: C# 구현, 재학습, 여러 모델 앙상블 선택 UI, 버전 관리.
- 반목표: 화면 점수와 현장 점수가 조용히 달라지는 것. 검증에 실패하면 반드시 화면에 드러나야 한다.

## 묶음 구성

저장 위치는 사용자가 고른 폴더 아래 `{데이터셋}_{알고리즘}_{YYYYMMDD-HHMM}/`에 만든다.

| 파일 | 내용 |
|---|---|
| `manifest.json` | 현장 코드가 따라야 할 전·후처리 값 전부. 아래 참고 |
| `model.onnx` (PatchCore) / `fold1.onnx … foldk.onnx` (YOLO) | 모델 |
| `mask.png` | 마스터 마스크 원본(0/255) |
| `score_mask.png` | PatchCore 전용. 침식한 점수 마스크(0/255) |
| `master.png` | 위치 보정 기준 마스터 이미지. 마스터를 못 찾았거나 위치 보정이 꺼져 있으면 없음 |
| `report.json` | 데이터셋 이름, 내보낸 시각, 파라미터, 지표(놓친 건수, 오검출, AUC, 위치 적중), 장수, 검증 결과 |

### manifest.json 공통

- `format`: `"incubator-export"`, `version`: 1
- `algorithm`: `"PatchCore"` | `"YOLO"`
- `threshold`: 내보낼 때 슬라이더 값. **점수 ≥ threshold면 간섭**
- `image_size`: [높이, 너비]. 다른 크기면 판정하지 않는다
- `align`: 위치 보정. 꺼져 있으면 `null`. 켜져 있으면 다음과 같다.
  - `master`: `"master.png"`, `region`: [y0, y1, x0, x1] (마스크 bbox + 40)
  - 방법: master.png(3채널 BGR)와 입력의 region을 각각 그레이(float32)로 바꾼다. region 크기의 Hanning 창(`createHanningWindow`)을 써서 `phaseCorrelate(master, 입력, 창)`을 돌린다(첫 인자가 master). 결과 (dx, dy)는 입력 내용이 마스터에서 옮겨진 양이다
  - 마스크 이동: `warpAffine` 평행이동 (dx, dy)(실수 그대로), 최근접 보간, 바깥은 0
  - 박스 이동: 크기는 유지하고 시작점에 round(dx), round(dy)를 더한 뒤 이미지 안으로 밀어 넣는다. round는 짝수 반올림(banker's)이며, C# `Math.Round` 기본값과 같다
- 판정은 `image_size`가 같은 이미지에만 한다. 위치 보정 응답이 0.05 미만이면 측정이 불확실하다(툴은 표시만 하고 판정은 그대로 한다)

### PatchCore

- `crop`: [y0, y1, x0, x1] 마스터 기준 특징 추출 영역 (마스크 bbox + 20). 위치 보정 시 박스 이동 규칙으로 옮긴다
- `input`: [1, 3, H, W] float32, BGR→RGB, /255, ImageNet mean/std 정규화, crop을 (W, H)로 `INTER_AREA` 리사이즈
- 출력 `distance`: [H/8, W/8] 패치별 최근접 거리
- 후처리: crop 크기로 bilinear 업샘플 → `GaussianBlur(sigma=4)` → 원본 크기 0 배경에 crop 위치로 붙임 → (옮긴) score_mask 안에서 최댓값 = 점수, 그 위치 = peak
- ONNX 안에는 wide_resnet50_2 특징 추출기와 메모리 뱅크(coreset), 거리 계산이 들어간다. 거리는 `sqrt(max(|q|²−2q·b+|b|², 0))`의 최솟값이다

### YOLO

- `models`: ["fold1.onnx", …]. 시험 때 정상 점수 규칙과 같게 **모델별 점수의 최댓값**이 이미지 점수다
- `input`: [1, 3, H, W] float32 = `input_hw`, `letterbox: "rect"`. 툴의 `.pt` 예측이 rect 레터박스를 쓰기 때문에 ONNX도 같은 고정 크기로 내보낸다(정사각 640×640으로 내보내면 실제 FEM 데이터에서 점수가 최대 0.019 달라졌다).
  - r = min(S/h, S/w) (S = imgsz), nw = round(w·r), nh = round(h·r), `INTER_LINEAR` 리사이즈
  - 패딩 크기: 너비 (S−nw) mod 32, 높이 (S−nh) mod 32. 좌·상 = round(d/2 − 0.1), 우·하 = round(d/2 + 0.1), 값 114
  - BGR→RGB, /255, NCHW
  - 예: 2064×1544 → 640×479로 리사이즈한 뒤 480×640으로 패딩
- 출력 `output0`: [1, 300, 6] = 레터박스 좌표 x1, y1, x2, y2, conf, cls (NMS 포함: conf ≥ 0.001, IoU 0.7, 최대 300). 빈 행은 conf 0
- 원본 좌표 되돌리기: x는 (x − 좌패딩) / (nw/w), y는 (y − 상패딩) / (nh/h). 가로·세로 배율을 따로 쓰고, 결과는 [0, w], [0, h]로 자른다
- 마스크 겹침: xa = int(x1), ya = int(y1), xb = ceil(x2), yb = ceil(y2). mask[max(ya,0):yb, max(xa,0):xb]에 참이 하나라도 있으면 겹친다
- 후처리: 겹치는 박스의 최고 conf가 모델 점수이고, 없으면 0이다. 이미지 점수는 모델 점수들의 최댓값이다. peak는 그 박스의 중심(정수 나눗셈)이다
- `report.json`의 간섭 지표는 자기 겹 모델 점수(박스 없는 간섭은 평균)로 계산한 값이다. 현장은 최댓값을 쓰므로 간섭 점수가 그보다 같거나 높게 나온다. 즉 report의 놓친 건수는 보수적인 값이다

## 툴 쪽 변경

- 실행 결과가 학습된 모델을 들고 있어야 한다(`AlgoResult.model`). PatchCore는 메모리 뱅크(CPU)와 crop 정보를, YOLO는 겹별 가중치 파일 경로를 든다. YOLO 가중치는 결과를 지울 때(새 실행, 데이터·파라미터 변경, 앱 종료) 함께 지운다.
- 결과 화면 실행 카드에 **모델 내보내기** 버튼을 둔다. 결과에 모델이 있고 실행 중이 아닐 때만 눌린다.
- 내보내기와 검증은 별도 스레드에서 돌고, 진행 문구는 실행 카드의 진행 표시를 같이 쓴다.
- 완료되면 폴더 경로와 검증 결과를 보여 준다. 검증 불일치는 경고 색으로 표시한다.

## 검증

- `incubator/runtime.py`는 torch 없이 묶음 파일만으로 manifest대로 채점한다(onnxruntime CPU + OpenCV). C# 구현의 기준 코드 역할도 한다.
- 비교 대상
  - PatchCore: 간섭과 임계값용·시험용 정상 전부
  - YOLO: 임계값용·시험용 정상(화면 점수가 모델별 최댓값인 이미지)만. 간섭은 자기 겹 모델로만 채점했기 때문에 비교하지 않는다
- 통과 조건: 모든 이미지의 판정(점수 ≥ 임계값)이 같고, 점수 차이가 max(1e-3, 임계값의 1%) 이하
- 결과(비교 장수, 판정 불일치 수, 최대 차이, 통과 여부)는 `report.json`과 화면에 남긴다
- 검증 자체가 오류로 끝나면 검증 안 된 묶음이 현장에 가지 않도록 폴더를 지운다
- `report.json`에 torch, ultralytics, onnx, onnxruntime 버전을 남긴다. ultralytics는 레터박스 방식이 버전마다 달라질 수 있어 8.4.191로 고정한다

## 구현자 판단으로 남긴 것

- YOLO 레터박스는 화면 채점(직사각 최소 패딩)과 달리 정사각 고정 입력이다. 차이는 검증으로 확인한다
- 버튼 위치·문구, 스레드 구조, 테스트 데이터 구성
