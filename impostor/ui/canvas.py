"""이미지 + 마스크 윤곽 + 박스 + 히트맵을 보여주는 뷰. 휠 확대, 가운데 버튼 드래그 이동."""
import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (QGraphicsPathItem, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene,
                               QGraphicsSimpleTextItem, QGraphicsView)

from impostor.ui.boxes import normalize_box

GT_COLOR = QColor(0, 220, 0)
PRED_COLOR = QColor(255, 140, 0)
MASK_COLOR = QColor(0, 200, 255)
HEAT_ALPHA = 0.45


def to_pixmap(img: np.ndarray) -> QPixmap:
    """BGR 또는 BGRA uint8 → QPixmap"""
    if img.shape[2] == 4:
        rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        q = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QImage.Format_RGBA8888)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        q = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(q.copy())


class ImageCanvas(QGraphicsView):
    box_added = Signal(tuple)
    box_deleted = Signal(int)

    def __init__(self, editable: bool = False, parent=None):
        super().__init__(parent)
        self.editable = editable
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor(40, 40, 40)))
        self._img = self._heat = self._mask = self._msg = None
        self._gt: list[QGraphicsRectItem] = []
        self._pred: list = []
        self._drag_start: QPointF | None = None
        self._rubber: QGraphicsRectItem | None = None
        self._pan: QPointF | None = None
        self._size = (0, 0)

    # ---- 내용 ----
    def clear_all(self, message: str = "") -> None:
        self.scene().clear()
        self._img = self._heat = self._mask = None
        self._gt, self._pred = [], []
        self._size = (0, 0)
        if message:
            self._msg = self.scene().addSimpleText(message)
            self._msg.setBrush(QBrush(Qt.white))

    def set_image(self, bgr: np.ndarray) -> None:
        first = self._size != (bgr.shape[1], bgr.shape[0])
        self.clear_all()
        self._size = (bgr.shape[1], bgr.shape[0])
        self._img = self.scene().addPixmap(to_pixmap(bgr))
        self.scene().setSceneRect(QRectF(0, 0, *self._size))
        if first:
            self.fit()

    def has_image(self) -> bool:
        return self._img is not None

    def fit_rect(self, x0: int, y0: int, x1: int, y1: int) -> None:
        self.fitInView(QRectF(x0, y0, x1 - x0, y1 - y0), Qt.KeepAspectRatio)

    def fit(self) -> None:
        if self._size != (0, 0):
            self.fitInView(QRectF(0, 0, *self._size), Qt.KeepAspectRatio)

    def set_mask_outline(self, mask: np.ndarray | None) -> None:
        if self._mask is not None:
            self.scene().removeItem(self._mask)
            self._mask = None
        if mask is None or self._img is None:
            return
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        path = QPainterPath()
        for c in contours:
            pts = c[:, 0, :]
            path.moveTo(*pts[0])
            for x, y in pts[1:]:
                path.lineTo(x, y)
            path.closeSubpath()
        pen = QPen(MASK_COLOR, 0, Qt.DashLine)
        self._mask = self.scene().addPath(path, pen)
        self._mask.setZValue(2)

    def set_heatmap(self, amap: np.ndarray | None, vmin: float = 0.0, vmax: float = 1.0) -> None:
        if self._heat is not None:
            self.scene().removeItem(self._heat)
            self._heat = None
        if amap is None or self._img is None:
            return
        norm = np.clip((amap.astype(np.float32) - vmin) / max(vmax - vmin, 1e-9), 0, 1)
        color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        bgra = np.dstack([color, np.full(norm.shape, int(255 * HEAT_ALPHA), np.uint8)])
        self._heat = self.scene().addPixmap(to_pixmap(bgra))
        self._heat.setZValue(1)

    def set_boxes(self, boxes) -> None:
        for r in self._gt:
            self.scene().removeItem(r)
        self._gt = []
        for x0, y0, x1, y1 in boxes:
            r = self.scene().addRect(QRectF(x0, y0, x1 - x0, y1 - y0), QPen(GT_COLOR, 0))
            r.setZValue(3)
            r.setFlag(QGraphicsRectItem.ItemIsSelectable, self.editable)
            self._gt.append(r)

    def set_pred_boxes(self, boxes) -> None:
        for it in self._pred:
            self.scene().removeItem(it)
        self._pred = []
        for x0, y0, x1, y1, conf in boxes[:5]:
            r = self.scene().addRect(QRectF(x0, y0, x1 - x0, y1 - y0), QPen(PRED_COLOR, 2))
            t = self.scene().addSimpleText(f"{conf:.2f}")
            t.setBrush(QBrush(PRED_COLOR))
            t.setPos(x0, max(0, y0 - 16))
            for it in (r, t):
                it.setZValue(4)
                self._pred.append(it)

    # ---- 마우스 ----
    def wheelEvent(self, e):
        f = 1.25 if e.angleDelta().y() > 0 else 0.8
        self.scale(f, f)

    def mousePressEvent(self, e):
        if e.button() == Qt.MiddleButton:
            self._pan = e.position()
            return
        if self.editable and e.button() == Qt.LeftButton and self._img is not None:
            hit = self.itemAt(e.position().toPoint())
            if hit in self._gt:
                super().mousePressEvent(e)
                return
            self.scene().clearSelection()
            self._drag_start = self.mapToScene(e.position().toPoint())
            self._rubber = self.scene().addRect(QRectF(self._drag_start, self._drag_start), QPen(GT_COLOR, 0, Qt.DashLine))
            self._rubber.setZValue(5)
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._pan is not None:
            d = e.position() - self._pan
            self._pan = e.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            return
        if self._rubber is not None:
            self._rubber.setRect(QRectF(self._drag_start, self.mapToScene(e.position().toPoint())).normalized())
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MiddleButton:
            self._pan = None
            return
        if self._rubber is not None:
            end = self.mapToScene(e.position().toPoint())
            self.scene().removeItem(self._rubber)
            self._rubber = None
            box = normalize_box(self._drag_start.x(), self._drag_start.y(), end.x(), end.y(), *self._size)
            self._drag_start = None
            if box:
                self.box_added.emit(box)
            return
        super().mouseReleaseEvent(e)

    def keyPressEvent(self, e):
        if self.editable and e.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if self.delete_selected():
                return
        super().keyPressEvent(e)

    def delete_selected(self) -> bool:
        for idx, r in enumerate(self._gt):
            if r.isSelected():
                self.box_deleted.emit(idx)
                return True
        return False
