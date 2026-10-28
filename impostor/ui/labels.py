"""라벨·용도 표시용 공통 값 (목록 화면·결과 화면·대화상자가 같이 쓴다)."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

from impostor.dataset import Label, Pin
from impostor.ui import theme

LABEL_TEXT = {Label.NORMAL: "정상", Label.INTERFERENCE: "간섭", Label.UNLABELED: "미분류"}
LABEL_COLOR = {Label.NORMAL: QColor(theme.OK), Label.INTERFERENCE: QColor(theme.ALERT),
               Label.UNLABELED: QColor(theme.MUTED)}
PIN_TEXT = {Pin.NONE: "자동", Pin.TRAIN: "학습", Pin.EVAL: "시험"}
PIN_HELP = "정상 사진을 학습/시험 중 어디에 쓸지. 보통은 자동 (예: 손글씨 판은 학습, 헷갈리는 정상은 시험)"
KEY_LABEL = {Qt.Key_N: Label.NORMAL, Qt.Key_I: Label.INTERFERENCE, Qt.Key_U: Label.UNLABELED}
LIST_W = 420          # 목록 폭 고정 → 사진 영역 크기가 바뀌지 않는다
