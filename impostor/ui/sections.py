"""제목이 붙은 테두리 카드 — 화면을 구역으로 나눌 때 쓴다."""
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout

from impostor.ui import theme


def section(title: str) -> tuple[QFrame, QVBoxLayout]:
    """제목이 붙은 테두리 카드. (카드, 안쪽 레이아웃)"""
    frame = QFrame()
    frame.setObjectName("section")
    frame.setStyleSheet(f"QFrame#section {{ background: {theme.PANEL_2}; border: 1px solid {theme.EDGE}; "
                        f"border-top: 3px solid {theme.ACCENT}; border-radius: 8px; }}")
    v = QVBoxLayout(frame)
    v.setContentsMargins(14, 10, 14, 12)
    v.setSpacing(8)
    head = QLabel(title)
    head.setStyleSheet(f"color: {theme.ACCENT}; font-weight: bold; font-size: 15px; font-family: {theme.MONO}; "
                       f"background: transparent; border: 0;")
    v.addWidget(head)
    return frame, v
