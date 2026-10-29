"""감시실 레이더 테마 — 남색 바탕, 청록(시스템), 빨강(경고·잡았다)."""
from PySide6.QtGui import QFont

BG = "#060a12"         # 창 바탕 (구역 카드보다 확실히 어둡게)
PANEL = "#111c2e"      # 입력칸·목록 바탕, 단계 탭 띠
PANEL_2 = "#142238"    # 구역 카드 바탕 (바탕과 대비)
EDGE = "#24557a"       # 구역 카드 테두리
GRID = "#13304a"       # 사진 영역 격자·테두리
TEXT = "#cfe8ff"
MUTED = "#6b86a3"
ACCENT = "#4fd1c5"     # 청록 — 시스템·정상 판정(CLEAR)
ALERT = "#ff5c5c"      # 빨강 — NG·놓침·TARGET LOCKED
WARN = "#ffb347"       # 주황 — 오검출
OK = "#3ddc97"         # 초록 — 정상 라벨

MONO = "Consolas"

STYLESHEET = f"""
QWidget {{ background: {BG}; color: {TEXT}; font-size: 13px; }}
QLabel {{ background: transparent; }}
QLineEdit, QComboBox, QSpinBox, QListWidget, QTableWidget {{
    background: {PANEL}; border: 1px solid {GRID}; border-radius: 4px; padding: 3px;
    selection-background-color: #1d4f6e; selection-color: {TEXT};
}}
QHeaderView::section {{ background: {PANEL}; color: {MUTED}; border: 0; border-bottom: 1px solid {GRID}; padding: 4px; }}
QTableWidget {{ gridline-color: {GRID}; }}
QListWidget::item {{ padding: 3px 4px; border-left: 4px solid transparent; }}
QListWidget::item:selected, QTableWidget::item:selected {{
    background: #1f6f7a; color: #ffffff; border-left: 4px solid {ACCENT};
}}
QListWidget::item:hover {{ background: #15304a; }}
QPushButton {{
    background: #16253b; border: 1px solid {GRID}; border-radius: 4px; padding: 5px 12px; color: {TEXT};
}}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:checked {{ background: #12434a; border-color: {ACCENT}; color: {ACCENT}; font-weight: bold; }}
QPushButton:disabled {{ color: #44566c; border-color: #1a2638; }}
QProgressBar {{ background: {PANEL}; border: 1px solid {GRID}; border-radius: 4px; text-align: center; color: {TEXT}; }}
QProgressBar::chunk {{ background: {ACCENT}; }}
QSlider::groove:horizontal {{ height: 4px; background: {GRID}; }}
QSlider::handle:horizontal {{ background: {ALERT}; width: 12px; margin: -6px 0; border-radius: 6px; }}
QGroupBox {{ border: 1px solid {GRID}; border-radius: 4px; margin-top: 8px; color: {ACCENT}; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
QMenuBar, QMenu {{ background: {PANEL}; color: {TEXT}; }}
QMenu::item:selected, QMenuBar::item:selected {{ background: #1d4f6e; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {ACCENT}; border-radius: 3px; background: {PANEL}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; }}
QToolTip {{ background: {PANEL}; color: {TEXT}; border: 1px solid {ACCENT}; }}
QScrollBar:vertical {{ background: {PANEL}; width: 12px; margin: 0; border: 0; }}
QScrollBar:horizontal {{ background: {PANEL}; height: 12px; margin: 0; border: 0; }}
QScrollBar::handle {{ background: {EDGE}; border-radius: 4px; margin: 2px; }}
QScrollBar::handle:vertical {{ min-height: 30px; }}
QScrollBar::handle:horizontal {{ min-width: 30px; }}
QScrollBar::handle:hover, QScrollBar::handle:pressed {{ background: {ACCENT}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; border: 0; background: none; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
QAbstractScrollArea::corner {{ background: {PANEL}; border: 0; }}
"""


def mono(size: int = 12, bold: bool = False) -> QFont:
    f = QFont(MONO, size)
    f.setBold(bold)
    return f


def apply(app) -> None:
    app.setStyleSheet(STYLESHEET)
