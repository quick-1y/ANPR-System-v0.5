#!/usr/bin/env python3
# /anpr/ui/main_window.py
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import psutil
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets
from zoneinfo import ZoneInfo

from anpr.config import Config
from anpr.infrastructure.settings_manager import DEFAULT_ROI_POINTS
from anpr.postprocessing.country_config import CountryConfigLoader
from anpr.workers.channel_worker import ChannelWorker
from anpr.infrastructure.logging_manager import get_logger
from anpr.infrastructure.storage import EventDatabase

logger = get_logger(__name__)


class PixmapPool:
    """Простой пул QPixmap для повторного использования буферов по размеру."""

    def __init__(self, max_per_size: int = 5) -> None:
        self._pool: Dict[Tuple[int, int], List[QtGui.QPixmap]] = {}
        self._max_per_size = max_per_size

    def acquire(self, size: QtCore.QSize) -> QtGui.QPixmap:
        key = (size.width(), size.height())
        pixmaps = self._pool.get(key)
        if pixmaps:
            pixmap = pixmaps.pop()
        else:
            pixmap = QtGui.QPixmap(size)
        if pixmap.size() != size:
            pixmap = QtGui.QPixmap(size)
        return pixmap

    def release(self, pixmap: QtGui.QPixmap) -> None:
        key = (pixmap.width(), pixmap.height())
        pixmaps = self._pool.setdefault(key, [])
        if len(pixmaps) < self._max_per_size:
            pixmaps.append(pixmap)


class ChannelView(QtWidgets.QWidget):
    """Отображает поток канала с подсказками и индикатором движения."""

    channelDropped = QtCore.pyqtSignal(int, int)
    channelActivated = QtCore.pyqtSignal(str)
    dragStarted = QtCore.pyqtSignal()
    dragFinished = QtCore.pyqtSignal()

    def __init__(self, name: str, pixmap_pool: Optional[PixmapPool]) -> None:
        super().__init__()
        self.name = name
        self._pixmap_pool = pixmap_pool
        self._current_pixmap: Optional[QtGui.QPixmap] = None
        self._channel_name: Optional[str] = None
        self._grid_position: int = -1
        self._drag_start_pos: Optional[QtCore.QPoint] = None
        self.setAcceptDrops(True)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.video_label = QtWidgets.QLabel("Нет сигнала")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setStyleSheet(
            "background-color: #000; color: #ccc; border: 1px solid #2e2e2e; padding: 4px;"
        )
        self.video_label.setMinimumSize(220, 170)
        self.video_label.setScaledContents(False)
        self.video_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.video_label)

        self.motion_indicator = QtWidgets.QLabel("Движение")
        self.motion_indicator.setParent(self.video_label)
        self.motion_indicator.setStyleSheet(
            "background-color: rgba(220, 53, 69, 0.85); color: white;"
            "padding: 3px 6px; border-radius: 6px; font-weight: bold;"
        )
        self.motion_indicator.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.motion_indicator.hide()

        self.last_plate = QtWidgets.QLabel("—")
        self.last_plate.setParent(self.video_label)
        self.last_plate.setStyleSheet(
            "background-color: rgba(0, 0, 0, 0.55); color: white;"
            "padding: 2px 6px; border-radius: 4px; font-weight: bold;"
        )
        self.last_plate.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.last_plate.hide()

        self.status_hint = QtWidgets.QLabel("")
        self.status_hint.setParent(self.video_label)
        self.status_hint.setStyleSheet(
            "background-color: rgba(0, 0, 0, 0.55); color: #ddd; padding: 2px 4px;"
        )
        self.status_hint.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.status_hint.hide()

    def set_channel_name(self, channel_name: Optional[str]) -> None:
        self._channel_name = channel_name

    def channel_name(self) -> Optional[str]:
        return self._channel_name

    def set_grid_position(self, position: int) -> None:
        self._grid_position = position

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        rect = self.video_label.contentsRect()
        margin = 8
        indicator_size = self.motion_indicator.sizeHint()
        self.motion_indicator.move(
            rect.right() - indicator_size.width() - margin, rect.top() + margin
        )
        self.last_plate.move(rect.left() + margin, rect.top() + margin)
        status_size = self.status_hint.sizeHint()
        self.status_hint.move(rect.left() + margin, rect.bottom() - status_size.height() - margin)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.LeftButton:
            self._drag_start_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if (
            self._channel_name
            and event.buttons() & QtCore.Qt.LeftButton
            and self._drag_start_pos
            and (event.pos() - self._drag_start_pos).manhattanLength()
            >= QtWidgets.QApplication.startDragDistance()
        ):
            self.dragStarted.emit()
            drag = QtGui.QDrag(self)
            mime_data = QtCore.QMimeData()
            mime_data.setText(self._channel_name)
            mime_data.setData("application/x-channel-index", str(self._grid_position).encode())
            drag.setMimeData(mime_data)
            drag.exec_(QtCore.Qt.MoveAction)
            self.dragFinished.emit()
            return
        super().mouseMoveEvent(event)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasFormat("application/x-channel-index"):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:  # noqa: N802
        if event.mimeData().hasFormat("application/x-channel-index"):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:  # noqa: N802
        if not event.mimeData().hasFormat("application/x-channel-index"):
            event.ignore()
            return
        try:
            source_index = int(bytes(event.mimeData().data("application/x-channel-index")).decode())
        except (ValueError, TypeError):
            event.ignore()
            return
        self.channelDropped.emit(source_index, self._grid_position)
        event.acceptProposedAction()

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.LeftButton and self._channel_name:
            self.channelActivated.emit(self._channel_name)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def set_pixmap(self, pixmap: QtGui.QPixmap) -> None:
        if self._pixmap_pool and self._current_pixmap is not None:
            self._pixmap_pool.release(self._current_pixmap)
        self._current_pixmap = pixmap
        self.video_label.setPixmap(pixmap)

    def set_motion_active(self, active: bool) -> None:
        self.motion_indicator.setVisible(active)

    def set_last_plate(self, plate: str) -> None:
        self.last_plate.setVisible(bool(plate))
        self.last_plate.setText(plate or "—")
        self.last_plate.adjustSize()

    def set_status(self, text: str) -> None:
        self.status_hint.setVisible(bool(text))
        self.status_hint.setText(text)
        if text:
            self.status_hint.adjustSize()


class ROIEditor(QtWidgets.QLabel):
    """Виджет предпросмотра канала с настраиваемой областью распознавания."""

    roi_changed = QtCore.pyqtSignal(dict)
    plate_size_selected = QtCore.pyqtSignal(str, int, int)

    def __init__(self) -> None:
        super().__init__("Нет кадра")
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(400, 260)
        self.setStyleSheet(
            "background-color: #111; color: #888; border: 1px solid #444; padding: 6px;"
        )
        self._pixmap: Optional[QtGui.QPixmap] = None
        self._roi_data: Dict[str, Any] = {"unit": "px", "points": []}
        self._points: List[QtCore.QPointF] = []
        self._drag_index: Optional[int] = None
        self._size_rects: Dict[str, Optional[QtCore.QRectF]] = {"min": None, "max": None}
        self._active_size_target: Optional[str] = None
        self._active_size_handle: Optional[str] = None
        self._active_size_origin: Optional[QtCore.QPointF] = None
        self._active_size_rect: Optional[QtCore.QRectF] = None

    def image_size(self) -> Optional[QtCore.QSize]:
        return self._pixmap.size() if self._pixmap else None

    def current_pixmap(self) -> Optional[QtGui.QPixmap]:
        return self._pixmap

    def _clamp_points(self) -> None:
        if not self._pixmap:
            return
        width = self._pixmap.width()
        height = self._pixmap.height()
        clamped: List[QtCore.QPointF] = []
        for p in self._points:
            clamped.append(
                QtCore.QPointF(
                    max(0.0, min(float(width - 1), p.x())),
                    max(0.0, min(float(height - 1), p.y())),
                )
            )
        self._points = clamped

    def _recalculate_points(self) -> None:
        roi = self._roi_data or {}
        unit = str(roi.get("unit", "px")).lower()
        raw_points = roi.get("points") or []
        if not raw_points:
            self._points = []
            return

        if self._pixmap is None:
            self._points = [
                QtCore.QPointF(float(p.get("x", 0)), float(p.get("y", 0)))
                for p in raw_points
                if isinstance(p, dict)
            ]
            return

        width = self._pixmap.width()
        height = self._pixmap.height()
        if unit == "percent":
            self._points = [
                QtCore.QPointF(width * float(p.get("x", 0)) / 100.0, height * float(p.get("y", 0)) / 100.0)
                for p in raw_points
                if isinstance(p, dict)
            ]
        else:
            self._points = [
                QtCore.QPointF(float(p.get("x", 0)), float(p.get("y", 0)))
                for p in raw_points
                if isinstance(p, dict)
            ]
        self._clamp_points()

    def _clamp_rect(self, rect: QtCore.QRectF) -> QtCore.QRectF:
        if self._pixmap is None:
            return QtCore.QRectF(rect)
        width = max(1.0, float(self._pixmap.width()))
        height = max(1.0, float(self._pixmap.height()))
        left = max(0.0, min(rect.left(), width))
        top = max(0.0, min(rect.top(), height))
        right = max(left + 1.0, min(rect.right(), width))
        bottom = max(top + 1.0, min(rect.bottom(), height))
        return QtCore.QRectF(QtCore.QPointF(left, top), QtCore.QPointF(right, bottom))

    def set_roi(self, roi: Dict[str, Any]) -> None:
        self._roi_data = roi or {"unit": "px", "points": []}
        self._recalculate_points()
        self.update()

    def set_plate_sizes(
        self,
        min_width: int,
        min_height: int,
        max_width: int,
        max_height: int,
    ) -> None:
        self._update_size_rect("min", float(min_width), float(min_height))
        self._update_size_rect("max", float(max_width), float(max_height))
        self.update()

    def setPixmap(self, pixmap: Optional[QtGui.QPixmap]) -> None:  # noqa: N802
        self._pixmap = pixmap
        if pixmap is None:
            super().setPixmap(QtGui.QPixmap())
            self.setText("Нет кадра")
            self._size_capture_target = None
            self._size_capture_start = None
            self._size_capture_end = None
            return
        self._recalculate_points()
        self._clamp_size_rects()
        scaled = self._scaled_pixmap(self.size())
        super().setPixmap(scaled)
        self.setText("")

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._pixmap:
            super().setPixmap(self._scaled_pixmap(event.size()))

    def _update_size_rect(self, target: str, width: float, height: float) -> None:
        if width <= 0 or height <= 0:
            self._size_rects[target] = None
            return

        if self._pixmap is None:
            left = 0.0
            top = 0.0
        else:
            existing = self._size_rects.get(target)
            anchor = existing.center() if existing else QtCore.QPointF(
                float(self._pixmap.width()) / 2.0,
                float(self._pixmap.height()) / 2.0,
            )
            left = anchor.x() - width / 2.0
            top = anchor.y() - height / 2.0
        rect = QtCore.QRectF(left, top, width, height)
        self._size_rects[target] = self._clamp_rect(rect)

    def _clamp_size_rects(self) -> None:
        for key, rect in self._size_rects.items():
            if rect is not None:
                if (
                    self._pixmap is not None
                    and rect.topLeft() == QtCore.QPointF(0.0, 0.0)
                    and rect.width() > 0
                    and rect.height() > 0
                ):
                    centered = QtCore.QPointF(
                        float(self._pixmap.width()) / 2.0,
                        float(self._pixmap.height()) / 2.0,
                    )
                    rect = QtCore.QRectF(
                        centered.x() - rect.width() / 2.0,
                        centered.y() - rect.height() / 2.0,
                        rect.width(),
                        rect.height(),
                    )
                self._size_rects[key] = self._clamp_rect(rect)

    def _scaled_pixmap(self, size: QtCore.QSize) -> QtGui.QPixmap:
        assert self._pixmap is not None
        return self._pixmap.scaled(
            size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )

    def _image_geometry(self) -> Optional[Tuple[QtCore.QPoint, QtCore.QSize]]:
        if self._pixmap is None:
            return None
        pixmap = self._scaled_pixmap(self.size())
        area = self.contentsRect()
        x = area.x() + (area.width() - pixmap.width()) // 2
        y = area.y() + (area.height() - pixmap.height()) // 2
        return QtCore.QPoint(x, y), pixmap.size()

    def _image_to_widget(self, point: QtCore.QPointF) -> Optional[QtCore.QPointF]:
        geom = self._image_geometry()
        if geom is None or self._pixmap is None:
            return None
        offset, scaled_size = geom
        scale_x = scaled_size.width() / max(1, self._pixmap.width())
        scale_y = scaled_size.height() / max(1, self._pixmap.height())
        return QtCore.QPointF(
            offset.x() + point.x() * scale_x,
            offset.y() + point.y() * scale_y,
        )

    def _widget_to_image(self, point: QtCore.QPoint) -> Optional[QtCore.QPointF]:
        geom = self._image_geometry()
        if geom is None or self._pixmap is None:
            return None
        offset, scaled_size = geom
        rect = QtCore.QRect(offset, scaled_size)
        if not rect.contains(point):
            return None
        scale_x = max(1, self._pixmap.width()) / max(1, scaled_size.width())
        scale_y = max(1, self._pixmap.height()) / max(1, scaled_size.height())
        return QtCore.QPointF(
            (point.x() - offset.x()) * scale_x,
            (point.y() - offset.y()) * scale_y,
        )

    def _widget_to_image_clamped(self, point: QtCore.QPoint) -> Optional[QtCore.QPointF]:
        img_pos = self._widget_to_image(point)
        if img_pos is not None:
            return img_pos
        geom = self._image_geometry()
        if geom is None or self._pixmap is None:
            return None
        offset, scaled_size = geom
        rect = QtCore.QRect(offset, scaled_size)
        clamped = QtCore.QPoint(
            max(rect.left(), min(point.x(), rect.right())),
            max(rect.top(), min(point.y(), rect.bottom())),
        )
        scale_x = max(1, self._pixmap.width()) / max(1, scaled_size.width())
        scale_y = max(1, self._pixmap.height()) / max(1, scaled_size.height())
        return QtCore.QPointF(
            (clamped.x() - offset.x()) * scale_x,
            (clamped.y() - offset.y()) * scale_y,
        )

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        super().paintEvent(event)
        geom = self._image_geometry()
        if geom is None:
            return
        _, scaled_size = geom
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # ROI отрисовываем поверх, чтобы рамка выбора не закрывала границы полигона

        polygon_points = self._points
        if not polygon_points and self._pixmap:
            polygon_points = [
                QtCore.QPointF(0, 0),
                QtCore.QPointF(self._pixmap.width() - 1, 0),
                QtCore.QPointF(self._pixmap.width() - 1, self._pixmap.height() - 1),
                QtCore.QPointF(0, self._pixmap.height() - 1),
            ]

        if not polygon_points:
            return

        widget_points = [self._image_to_widget(p) for p in polygon_points]
        widget_points = [p for p in widget_points if p is not None]
        if len(widget_points) < 3:
            return

        pen = QtGui.QPen(QtGui.QColor(0, 200, 0))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(QtGui.QColor(0, 200, 0, 40))
        painter.drawPolygon(QtGui.QPolygonF(widget_points))

        handle_brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        painter.setBrush(handle_brush)
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 1))
        for p in widget_points:
            painter.drawEllipse(QtCore.QPointF(p), 5, 5)

        for target, color in ("min", QtGui.QColor(34, 211, 238)), ("max", QtGui.QColor(249, 115, 22)):
            rect = self._size_rects.get(target)
            if rect is None:
                continue
            top_left = self._image_to_widget(rect.topLeft())
            bottom_right = self._image_to_widget(rect.bottomRight())
            if top_left is None or bottom_right is None:
                continue
            widget_rect = QtCore.QRectF(top_left, bottom_right).normalized()
            pen = QtGui.QPen(color)
            pen.setWidth(2)
            pen.setStyle(QtCore.Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QtGui.QColor(color.red(), color.green(), color.blue(), 40))
            painter.drawRect(widget_rect)

            painter.setPen(QtGui.QPen(color))
            painter.setBrush(QtGui.QBrush(color))
            for corner in (
                widget_rect.topLeft(),
                widget_rect.topRight(),
                widget_rect.bottomLeft(),
                widget_rect.bottomRight(),
            ):
                painter.drawEllipse(corner, 5, 5)

            label = "минимальный" if target == "min" else "максимальный"
            metrics = painter.fontMetrics()
            text_rect = metrics.boundingRect(label)
            label_rect = QtCore.QRectF(
                widget_rect.left(),
                max(0.0, widget_rect.top() - text_rect.height() - 4),
                text_rect.width(),
                text_rect.height(),
            )
            painter.setPen(QtGui.QPen(color))
            painter.drawText(label_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, label)

    def _emit_roi(self) -> None:
        roi = {
            "unit": "px",
            "points": [
                {"x": int(p.x()), "y": int(p.y())}
                for p in self._points
            ],
        }
        self.roi_changed.emit(roi)

    @staticmethod
    def _point_to_segment_distance(point: QtCore.QPointF, a: QtCore.QPointF, b: QtCore.QPointF) -> float:
        ab_x = b.x() - a.x()
        ab_y = b.y() - a.y()
        ab_len_sq = ab_x * ab_x + ab_y * ab_y
        if ab_len_sq == 0:
            return math.hypot(point.x() - a.x(), point.y() - a.y())
        ap_x = point.x() - a.x()
        ap_y = point.y() - a.y()
        t = max(0.0, min(1.0, (ap_x * ab_x + ap_y * ab_y) / ab_len_sq))
        proj_x = a.x() + t * ab_x
        proj_y = a.y() + t * ab_y
        return math.hypot(point.x() - proj_x, point.y() - proj_y)

    def _find_insertion_index(self, img_pos: QtCore.QPointF) -> int:
        if len(self._points) < 2:
            return len(self._points)

        best_index = len(self._points)
        best_distance = float("inf")
        for i, start in enumerate(self._points):
            end = self._points[(i + 1) % len(self._points)]
            dist = self._point_to_segment_distance(img_pos, start, end)
            if dist < best_distance:
                best_distance = dist
                best_index = i + 1
        return best_index

    def _size_handle_at(self, pos: QtCore.QPoint) -> Optional[Tuple[str, str]]:
        if self._pixmap is None:
            return None
        handle_radius = 10
        for target in ("min", "max"):
            rect = self._size_rects.get(target)
            if rect is None:
                continue
            top_left = self._image_to_widget(rect.topLeft())
            bottom_right = self._image_to_widget(rect.bottomRight())
            if top_left is None or bottom_right is None:
                continue
            widget_rect = QtCore.QRectF(top_left, bottom_right).normalized()
            corners = {
                "tl": widget_rect.topLeft(),
                "tr": widget_rect.topRight(),
                "bl": widget_rect.bottomLeft(),
                "br": widget_rect.bottomRight(),
            }
            for name, corner in corners.items():
                if (corner - QtCore.QPointF(pos)).manhattanLength() <= handle_radius:
                    return target, name
            if widget_rect.contains(pos):
                return target, "move"
        return None

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() != QtCore.Qt.LeftButton:
            return

        size_handle = self._size_handle_at(event.pos())
        if size_handle:
            self._active_size_target, self._active_size_handle = size_handle
            self._active_size_origin = self._widget_to_image_clamped(event.pos())
            rect = self._size_rects.get(self._active_size_target)
            self._active_size_rect = QtCore.QRectF(rect) if rect else None
            return

        img_pos = self._widget_to_image(event.pos())
        if img_pos is None:
            return

        if self._size_capture_target:
            self._size_capture_start = self._size_capture_end = img_pos
            self.update()
            return

        handle_radius = 8
        closest_idx = None
        closest_dist = handle_radius + 1
        for idx, point in enumerate(self._points):
            widget_point = self._image_to_widget(point)
            if widget_point is None:
                continue
            dist = (widget_point - QtCore.QPointF(event.pos())).manhattanLength()
            if dist < closest_dist:
                closest_idx = idx
                closest_dist = dist

        if closest_idx is not None:
            self._drag_index = closest_idx
        else:
            self._drag_index = None

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if self._active_size_target and self._active_size_rect:
            img_pos = self._widget_to_image_clamped(event.pos())
            if img_pos is None:
                return
            rect = QtCore.QRectF(
                self._size_rects.get(self._active_size_target) or self._active_size_rect
            )
            handle = self._active_size_handle
            if handle == "move" and self._active_size_origin is not None:
                delta = img_pos - self._active_size_origin
                rect.translate(delta)
                self._active_size_origin = img_pos
            elif handle:
                if "l" in handle:
                    rect.setLeft(img_pos.x())
                if "r" in handle:
                    rect.setRight(img_pos.x())
                if "t" in handle:
                    rect.setTop(img_pos.y())
                if "b" in handle:
                    rect.setBottom(img_pos.y())
            rect = rect.normalized()
            rect = self._clamp_rect(rect)
            self._size_rects[self._active_size_target] = rect
            self.plate_size_selected.emit(
                self._active_size_target, int(rect.width()), int(rect.height())
            )
            self.update()
            return

        if self._drag_index is None:
            return
        img_pos = self._widget_to_image(event.pos())
        if img_pos is None:
            return
        self._points[self._drag_index] = img_pos
        self._clamp_points()
        self._emit_roi()
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if self._active_size_target:
            self._active_size_target = None
            self._active_size_handle = None
            self._active_size_origin = None
            self._active_size_rect = None
            return

        self._drag_index = None

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() != QtCore.Qt.LeftButton:
            return

        img_pos = self._widget_to_image(event.pos())
        if img_pos is None:
            return

        handle_radius = 8
        closest_idx = None
        closest_dist = handle_radius + 1
        for idx, point in enumerate(self._points):
            widget_point = self._image_to_widget(point)
            if widget_point is None:
                continue
            dist = (widget_point - QtCore.QPointF(event.pos())).manhattanLength()
            if dist < closest_dist:
                closest_idx = idx
                closest_dist = dist

        if closest_idx is not None:
            self._points.pop(closest_idx)
        else:
            insert_at = self._find_insertion_index(img_pos)
            self._points.insert(insert_at, img_pos)

        self._drag_index = None
        self._clamp_points()
        self._emit_roi()
        self.update()

class PreviewLoader(QtCore.QThread):
    """Фоновая загрузка превью кадра канала, чтобы не блокировать UI."""

    frameReady = QtCore.pyqtSignal(QtGui.QPixmap)
    failed = QtCore.pyqtSignal()

    def __init__(self, source: str) -> None:
        super().__init__()
        self.source = source

    def run(self) -> None:  # noqa: N802
        capture: Optional[cv2.VideoCapture] = None
        try:
            capture = cv2.VideoCapture(
                int(self.source) if self.source.isnumeric() else self.source
            )
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, frame = capture.read()
        except Exception:
            ret, frame = False, None
        finally:
            if capture is not None:
                capture.release()

        if self.isInterruptionRequested():
            return
        if not ret or frame is None:
            self.failed.emit()
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb_frame.shape
        bytes_per_line = 3 * width
        q_image = QtGui.QImage(
            rgb_frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888
        ).copy()
        if self.isInterruptionRequested():
            return
        self.frameReady.emit(QtGui.QPixmap.fromImage(q_image))


class EventDetailView(QtWidgets.QWidget):
    """Отображение выбранного события: метаданные, кадр и область номера."""

    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)

        self.setStyleSheet(
            "QWidget { background-color: #16181d; }"
            "QGroupBox { background-color: #0f1115; color: #f6f7fb; border: 1px solid #20242c; border-radius: 12px; padding: 10px; margin-top: 8px; }"
            "QLabel { color: #e5e7eb; }"
        )

        self._frame_image: Optional[QtGui.QImage] = None
        self._plate_image: Optional[QtGui.QImage] = None

        self.frame_preview = self._build_preview("Кадр распознавания", min_height=320, keep_aspect=True)
        layout.addWidget(self.frame_preview, stretch=3)

        bottom_row = QtWidgets.QHBoxLayout()
        self.plate_preview = self._build_preview("Кадр номера", min_size=QtCore.QSize(200, 140), keep_aspect=True)
        bottom_row.addWidget(self.plate_preview, 1)

        meta_group = QtWidgets.QGroupBox("Данные распознавания")
        meta_group.setMinimumWidth(220)
        meta_layout = QtWidgets.QFormLayout(meta_group)
        self.time_label = QtWidgets.QLabel("—")
        self.channel_label = QtWidgets.QLabel("—")
        self.country_label = QtWidgets.QLabel("—")
        self.plate_label = QtWidgets.QLabel("—")
        self.conf_label = QtWidgets.QLabel("—")
        self.direction_label = QtWidgets.QLabel("—")
        meta_layout.addRow("Дата/Время:", self.time_label)
        meta_layout.addRow("Канал:", self.channel_label)
        meta_layout.addRow("Страна:", self.country_label)
        meta_layout.addRow("Гос. номер:", self.plate_label)
        meta_layout.addRow("Уверенность:", self.conf_label)
        meta_layout.addRow("Направление:", self.direction_label)
        bottom_row.addWidget(meta_group, 1)

        layout.addLayout(bottom_row, stretch=1)

    def _build_preview(
        self,
        title: str,
        min_height: int = 180,
        min_size: Optional[QtCore.QSize] = None,
        keep_aspect: bool = False,
    ) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox(title)
        wrapper = QtWidgets.QVBoxLayout(group)
        label = QtWidgets.QLabel("Нет изображения")
        label.setAlignment(QtCore.Qt.AlignCenter)
        if min_size:
            label.setMinimumSize(min_size)
        else:
            label.setMinimumHeight(min_height)
        label.setStyleSheet(
            "background-color: #0b0c10; color: #9ca3af; border: 1px solid #1f2937; border-radius: 10px;"
        )
        label.setScaledContents(not keep_aspect)
        label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        wrapper.addWidget(label)
        group.display_label = label  # type: ignore[attr-defined]
        return group

    def clear(self) -> None:
        self.time_label.setText("—")
        self.channel_label.setText("—")
        self.country_label.setText("—")
        self.plate_label.setText("—")
        self.conf_label.setText("—")
        self.direction_label.setText("—")
        self._frame_image = None
        self._plate_image = None
        for group in (self.frame_preview, self.plate_preview):
            group.display_label.setPixmap(QtGui.QPixmap())  # type: ignore[attr-defined]
            group.display_label.setText("Нет изображения")  # type: ignore[attr-defined]

    def set_event(
        self,
        event: Optional[Dict],
        frame_image: Optional[QtGui.QImage] = None,
        plate_image: Optional[QtGui.QImage] = None,
    ) -> None:
        if event is None:
            self.clear()
            return

        self.time_label.setText(event.get("timestamp", "—"))
        self.channel_label.setText(event.get("channel", "—"))
        self.country_label.setText(event.get("country") or "—")
        plate = event.get("plate") or "—"
        self.plate_label.setText(plate)
        conf = event.get("confidence")
        self.conf_label.setText(f"{float(conf):.2f}" if conf is not None else "—")
        self.direction_label.setText(event.get("direction", "—"))

        self._set_image(self.frame_preview, frame_image, keep_aspect=True)
        self._set_image(self.plate_preview, plate_image, keep_aspect=True)

    def _set_image(
        self,
        group: QtWidgets.QGroupBox,
        image: Optional[QtGui.QImage],
        keep_aspect: bool = False,
    ) -> None:
        if group is self.frame_preview:
            self._frame_image = image
        elif group is self.plate_preview:
            self._plate_image = image

        self._apply_image_to_label(group, image, keep_aspect)

    def _apply_image_to_label(
        self,
        group: QtWidgets.QGroupBox,
        image: Optional[QtGui.QImage],
        keep_aspect: bool = False,
    ) -> None:
        label: QtWidgets.QLabel = group.display_label  # type: ignore[attr-defined]
        if image is None:
            label.setPixmap(QtGui.QPixmap())
            label.setText("Нет изображения")
            return

        label.setText("")
        target_size = label.contentsRect().size()
        if target_size.width() == 0 or target_size.height() == 0:
            return
        pixmap = QtGui.QPixmap.fromImage(image)
        if keep_aspect:
            pixmap = pixmap.scaled(
                target_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
        else:
            pixmap = pixmap.scaled(target_size, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
        label.setPixmap(pixmap)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply_image_to_label(self.frame_preview, self._frame_image, keep_aspect=True)
        self._apply_image_to_label(self.plate_preview, self._plate_image, keep_aspect=True)


class MainWindow(QtWidgets.QMainWindow):
    """Главное окно приложения ANPR с вкладками наблюдения, поиска и настроек."""

    GRID_VARIANTS = ["1x1", "1x2", "2x2", "2x3", "3x3"]
    MAX_IMAGE_CACHE = 200
    MAX_IMAGE_CACHE_BYTES = 256 * 1024 * 1024  # 256 MB
    ACCENT_COLOR = "#22d3ee"
    SURFACE_COLOR = "#16181d"
    PANEL_COLOR = "#0f1115"
    GROUP_BOX_STYLE = (
        "QGroupBox { background-color: #16181d; color: #f6f7fb; border: 1px solid #20242c; border-radius: 12px; padding: 12px; margin-top: 10px; }"
        "QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; font-weight: 700; color: #e2e8f0; }"
        "QLabel { color: #cbd5e1; font-size: 13px; }"
        "QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QDateTimeEdit { background-color: #0b0c10; color: #f8fafc; border: 1px solid #1f2937; border-radius: 8px; padding: 8px; }"
        "QPushButton { background-color: #22d3ee; color: #0b0c10; border-radius: 8px; padding: 8px 14px; font-weight: 700; letter-spacing: 0.2px; }"
        "QPushButton:hover { background-color: #4ddcf3; }"
        "QCheckBox { color: #e5e7eb; font-size: 13px; }"
    )
    FORM_STYLE = (
        "QLabel { color: #cbd5e1; font-size: 13px; }"
        "QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QDateTimeEdit { background-color: #0b0c10; color: #f8fafc; border: 1px solid #1f2937; border-radius: 8px; padding: 8px; }"
        "QPushButton { background-color: #22d3ee; color: #0b0c10; border-radius: 8px; padding: 8px 14px; font-weight: 700; letter-spacing: 0.2px; }"
        "QPushButton:hover { background-color: #4ddcf3; }"
        "QCheckBox { color: #e5e7eb; font-size: 13px; }"
        f"QWidget {{ background-color: {SURFACE_COLOR}; }}"
    )
    FIELD_MAX_WIDTH = 520
    COMPACT_FIELD_WIDTH = 180

    PRIMARY_HOLLOW_BUTTON = (
        "QPushButton { background-color: transparent; color: #ffffff; border: 1px solid #ffffff; border-radius: 8px; padding: 8px 14px; font-weight: 700; letter-spacing: 0.2px; }"
        "QPushButton:hover { background-color: #22d3ee; color: #0b0c10; }"
        "QPushButton:pressed { background-color: #1fb6d5; color: #0b0c10; }"
    )
    #f?
    TABLE_STYLE = (
        "QHeaderView::section { background-color: #11131a; color: #e2e8f0; padding: 8px; font-weight: 700; border: none; }"
        "QTableWidget { background-color: #0b0c10; color: #e5e7eb; gridline-color: #1f2937; selection-background-color: #11131a; }"
        "QTableWidget::item { border-bottom: 1px solid #1f2937; padding: 6px; }"
        "QTableWidget::item:selected { background-color: rgba(34,211,238,0.18); color: #22d3ee; border: 1px solid #22d3ee; }"
        "QScrollBar:vertical { background: #0b0c10; width: 12px; margin: 0px; border: 1px solid #1f2937; border-radius: 6px; }"
        "QScrollBar::handle:vertical { background: #1f2937; min-height: 24px; border-radius: 6px; }"
        "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }"
        "QScrollBar:horizontal { background: #0b0c10; height: 12px; margin: 0px; border: 1px solid #1f2937; border-radius: 6px; }"
        "QScrollBar::handle:horizontal { background: #1f2937; min-width: 24px; border-radius: 6px; }"
        "QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }"
    )
    LIST_STYLE = (
        "QListWidget { background-color: #0b0c10; color: #cbd5e1; border: 1px solid #1f2937; border-radius: 10px; padding: 6px; }"
        "QListWidget::item:selected { background-color: rgba(34,211,238,0.16); color: #22d3ee; border-radius: 6px; }"
        "QListWidget::item { padding: 8px 10px; margin: 2px 0; }"
    )

    @staticmethod
    def _default_roi_region() -> Dict[str, Any]:
        return {"unit": "px", "points": [point.copy() for point in DEFAULT_ROI_POINTS]}

    def __init__(self, settings: Optional[Config] = None) -> None:
        super().__init__()
        self.setWindowTitle("ANPR Desktop")
        self.resize(1280, 800)
        screen = QtWidgets.QApplication.primaryScreen()
        self._top_left = (
            screen.availableGeometry().topLeft()
            if screen is not None
            else QtCore.QPoint(0, 0)
        )
        self.move(self._top_left)
        self.setWindowFlags(
            self.windowFlags()
            | QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowSystemMenuHint
            | QtCore.Qt.WindowMinMaxButtonsHint
        )
        self._window_drag_pos: Optional[QtCore.QPoint] = None

        self.settings = settings or Config()
        self.current_grid = self.settings.get_grid()
        if self.current_grid not in self.GRID_VARIANTS:
            self.current_grid = self.GRID_VARIANTS[0]
        self.db = EventDatabase(self.settings.get_db_path())

        self._pixmap_pool = PixmapPool()
        self.channel_workers: List[ChannelWorker] = []
        self.channel_labels: Dict[str, ChannelView] = {}
        self.focused_channel_name: Optional[str] = None
        self._previous_grid: Optional[str] = None
        self.event_images: "OrderedDict[int, Tuple[Optional[QtGui.QImage], Optional[QtGui.QImage]]]" = OrderedDict()
        self._image_cache_bytes = 0
        self.event_cache: Dict[int, Dict] = {}
        self.search_results: Dict[int, Dict] = {}
        self.flag_cache: Dict[str, Optional[QtGui.QIcon]] = {}
        self.flag_dir = Path(__file__).resolve().parents[2] / "images" / "flags"
        self.country_display_names = self._load_country_names()
        self._pending_channels: Optional[List[Dict[str, Any]]] = None
        self._channel_save_timer = QtCore.QTimer(self)
        self._channel_save_timer.setSingleShot(True)
        self._channel_save_timer.timeout.connect(self._flush_pending_channels)
        self._drag_counter = 0
        self._skip_frame_updates = False
        self._preview_worker: Optional[PreviewLoader] = None
        self._preview_request_id = 0

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setStyleSheet(
            "QTabBar { font-weight: 700; }"
            f"QTabBar::tab {{ background: #0b0c10; color: #9ca3af; padding: 10px 18px; border: 1px solid #0f1115; border-top-left-radius: 10px; border-top-right-radius: 10px; margin-right: 6px; }}"
            f"QTabBar::tab:selected {{ background: #16181d; color: {self.ACCENT_COLOR}; border: 1px solid #20242c; border-bottom: 2px solid {self.ACCENT_COLOR}; }}"
            "QTabWidget::pane { border: 1px solid #20242c; border-radius: 10px; background-color: #16181d; top: -1px; }"
        )
        self.observation_tab = self._build_observation_tab()
        self.search_tab = self._build_search_tab()
        self.settings_tab = self._build_settings_tab()

        self.tabs.addTab(self.observation_tab, "Наблюдение")
        self.tabs.addTab(self.search_tab, "Журнал")
        self.tabs.addTab(self.settings_tab, "Настройки")

        root = QtWidgets.QWidget()
        root_layout = QtWidgets.QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        header = self._build_main_header()
        root_layout.addWidget(header)
        root_layout.addWidget(self.tabs, 1)

        self.setCentralWidget(root)
        self.setStyleSheet(
            "QMainWindow { background-color: #0b0c10; }"
            "QStatusBar { background-color: #0b0c10; color: #e5e7eb; padding: 4px; border-top: 1px solid #1f2937; }"
            "QToolButton[windowControl='true'] { background-color: transparent; border: none; color: white; padding: 6px; }"
            "QToolButton[windowControl='true']:hover { background-color: rgba(255,255,255,0.08); border-radius: 6px; }"
        )
        self._build_status_bar()
        self._start_system_monitoring()
        self._refresh_events_table()
        self._start_channels()

    def _build_main_header(self) -> QtWidgets.QWidget:
        header = QtWidgets.QFrame()
        header.setFixedHeight(48)
        header.setStyleSheet(
            "QFrame { background-color: #0b0c10; border-bottom: 1px solid #20242c; }"
        )
        layout = QtWidgets.QHBoxLayout(header)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(8)

        title = QtWidgets.QLabel("ANPR Desktop")
        title.setStyleSheet("color: white; font-weight: 800;")
        layout.addWidget(title)
        layout.addStretch()

        minimize_btn = QtWidgets.QToolButton()
        minimize_btn.setProperty("windowControl", True)
        minimize_btn.setText("–")
        minimize_btn.setToolTip("Свернуть")
        minimize_btn.clicked.connect(self.showMinimized)
        layout.addWidget(minimize_btn)

        maximize_btn = QtWidgets.QToolButton()
        maximize_btn.setProperty("windowControl", True)
        maximize_btn.setText("⛶")
        maximize_btn.setToolTip("Развернуть/свернуть окно")

        def toggle_maximize() -> None:
            if self.isFullScreen():
                self.showNormal()
                self.move(self._top_left)
            else:
                self.showFullScreen()

        maximize_btn.clicked.connect(toggle_maximize)
        layout.addWidget(maximize_btn)

        close_btn = QtWidgets.QToolButton()
        close_btn.setProperty("windowControl", True)
        close_btn.setText("✕")
        close_btn.setToolTip("Закрыть")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        def start_move(event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
            if event.button() == QtCore.Qt.LeftButton and not self.isFullScreen():
                self._window_drag_pos = event.globalPos() - self.frameGeometry().topLeft()
                event.accept()

        def move_window(event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
            if (
                event.buttons() & QtCore.Qt.LeftButton
                and self._window_drag_pos is not None
                and not self.isFullScreen()
            ):
                self.move(event.globalPos() - self._window_drag_pos)
                event.accept()

        def end_move(event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
            self._window_drag_pos = None
            event.accept()

        def double_click(event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
            if event.button() == QtCore.Qt.LeftButton:
                toggle_maximize()
                event.accept()

        header.mousePressEvent = start_move  # type: ignore[assignment]
        header.mouseMoveEvent = move_window  # type: ignore[assignment]
        header.mouseReleaseEvent = end_move  # type: ignore[assignment]
        header.mouseDoubleClickEvent = double_click  # type: ignore[assignment]
        return header

    def _build_status_bar(self) -> None:
        status = self.statusBar()
        status.setStyleSheet(
            "background-color: #0b0c10; color: #e5e7eb; padding: 6px; border-top: 1px solid #1f2937;"
        )
        status.setSizeGripEnabled(False)
        self.cpu_label = QtWidgets.QLabel("CPU: —")
        self.ram_label = QtWidgets.QLabel("RAM: —")
        status.addPermanentWidget(self.cpu_label)
        status.addPermanentWidget(self.ram_label)

    def _start_system_monitoring(self) -> None:
        self.stats_timer = QtCore.QTimer(self)
        self.stats_timer.setInterval(1000)
        self.stats_timer.timeout.connect(self._update_system_stats)
        self.stats_timer.start()
        self._update_system_stats()

    def _update_system_stats(self) -> None:
        cpu_percent = psutil.cpu_percent(interval=None)
        ram_percent = psutil.virtual_memory().percent
        self.cpu_label.setText(f"CPU: {cpu_percent:.0f}%")
        self.ram_label.setText(f"RAM: {ram_percent:.0f}%")

    # ------------------ Наблюдение ------------------
    def _build_observation_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setSpacing(10)

        left_column = QtWidgets.QVBoxLayout()
        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(8)

        chooser = QtWidgets.QVBoxLayout()
        chooser.setSpacing(6)
        chooser.setContentsMargins(4, 4, 4, 4)
        chooser_label = QtWidgets.QLabel("Сетка")
        chooser_label.setStyleSheet("color: #e5e7eb; font-weight: 800;")
        chooser.addWidget(chooser_label)
        self.grid_combo = QtWidgets.QComboBox()
        self.grid_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.grid_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed
        )
        self.grid_combo.setStyleSheet(
            "QComboBox { background-color: #0b0c10; color: #e5e7eb; border: 1px solid #1f2937; border-radius: 10px; padding: 6px 10px; min-width: 0px; }"
            "QComboBox::drop-down { border: 0px; width: 28px; }"
            "QComboBox::down-arrow { image: url(:/qt-project.org/styles/commonstyle/images/arrowdown.png); width: 12px; height: 12px; margin-right: 6px; }"
            "QComboBox QAbstractItemView { background-color: #0b0c10; color: #e5e7eb; selection-background-color: rgba(34,211,238,0.14); border: 1px solid #1f2937; padding: 6px; }"
        )
        for variant in self.GRID_VARIANTS:
            self.grid_combo.addItem(variant.replace("x", "×"), variant)
        current_index = self.grid_combo.findData(self.current_grid)
        if current_index >= 0:
            self.grid_combo.setCurrentIndex(current_index)
        self.grid_combo.currentIndexChanged.connect(self._on_grid_combo_changed)
        chooser.addWidget(self.grid_combo)

        controls.addLayout(chooser)
        controls.addStretch()
        left_column.addLayout(controls)

        self.grid_widget = QtWidgets.QWidget()
        self.grid_layout = QtWidgets.QGridLayout(self.grid_widget)
        self.grid_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.grid_layout.setSpacing(6)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        left_column.addWidget(self.grid_widget, stretch=4)

        layout.addLayout(left_column, stretch=3)

        right_column = QtWidgets.QVBoxLayout()
        details_group = QtWidgets.QGroupBox("Информация о событии")
        details_group.setStyleSheet(self.GROUP_BOX_STYLE)
        details_layout = QtWidgets.QVBoxLayout(details_group)
        self.event_detail = EventDetailView()
        details_layout.addWidget(self.event_detail)
        right_column.addWidget(details_group, stretch=3)

        events_group = QtWidgets.QGroupBox("События")
        events_group.setStyleSheet(self.GROUP_BOX_STYLE)
        events_layout = QtWidgets.QVBoxLayout(events_group)
        self.events_table = QtWidgets.QTableWidget(0, 5)
        self.events_table.setHorizontalHeaderLabels(["Дата/Время", "Гос. номер", "Страна", "Канал", "Направление"])
        self.events_table.setStyleSheet(self.TABLE_STYLE)
        header = self.events_table.horizontalHeader()
        header.setMinimumSectionSize(90)
        header.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.events_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.events_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.events_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.events_table.verticalHeader().setVisible(False)
        self.events_table.itemSelectionChanged.connect(self._on_event_selected)
        events_layout.addWidget(self.events_table)
        right_column.addWidget(events_group, stretch=1)

        layout.addLayout(right_column, stretch=2)

        self._draw_grid()
        return widget

    @staticmethod
    def _polish_button(button: QtWidgets.QPushButton, min_width: int = 140) -> None:
        button.setStyleSheet(MainWindow.PRIMARY_HOLLOW_BUTTON)
        button.setMinimumWidth(min_width)
        button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

    @staticmethod
    def _prepare_optional_datetime(widget: QtWidgets.QDateTimeEdit) -> None:
        widget.setCalendarPopup(True)
        widget.setDisplayFormat("dd.MM.yyyy HH:mm:ss")
        min_dt = QtCore.QDateTime.fromSecsSinceEpoch(0)
        widget.setMinimumDateTime(min_dt)
        widget.setSpecialValueText("Не выбрано")
        widget.setDateTime(min_dt)

    def _apply_dark_calendar_style(self, widget: QtWidgets.QDateTimeEdit) -> None:
        widget.setStyleSheet(
            "QDateTimeEdit { background-color: #0b0c10; color: #f8fafc; border: 1px solid #1f2937; border-radius: 8px; padding: 6px; }"
            "QDateTimeEdit::drop-down { width: 24px; }"
        )
        calendar = widget.calendarWidget()
        calendar.setStyleSheet(
            "QCalendarWidget QWidget { background-color: #0b0c10; color: #f8fafc; }"
            "QCalendarWidget QToolButton { background-color: #1f2937; color: #f8fafc; border: none; padding: 6px; }"
            "QCalendarWidget QToolButton#qt_calendar_prevmonth, QCalendarWidget QToolButton#qt_calendar_nextmonth { width: 24px; }"
            "QCalendarWidget QSpinBox { background-color: #1f2937; color: #f8fafc; }"
            "QCalendarWidget QTableView { selection-background-color: rgba(34,211,238,0.18); selection-color: #22d3ee; alternate-background-color: #11131a; }"
        )

    @staticmethod
    def _offset_label(minutes: int) -> str:
        sign = "+" if minutes >= 0 else "-"
        total = abs(minutes)
        hours = total // 60
        mins = total % 60
        return f"UTC{sign}{hours:02d}:{mins:02d}"

    @staticmethod
    def _available_offset_labels() -> List[str]:
        now = QtCore.QDateTime.currentDateTime()
        offsets = {0}
        for tz_id in QtCore.QTimeZone.availableTimeZoneIds():
            try:
                offset = QtCore.QTimeZone(tz_id).offsetFromUtc(now) // 60
                offsets.add(offset)
            except Exception:
                continue
        return [MainWindow._offset_label(minutes) for minutes in sorted(offsets)]

    @staticmethod
    def _get_datetime_value(widget: QtWidgets.QDateTimeEdit) -> Optional[str]:
        if widget.dateTime() == widget.minimumDateTime():
            return None
        return widget.dateTime().toString(QtCore.Qt.ISODate)

    @staticmethod
    def _format_timestamp(value: str, target_zone: Optional[ZoneInfo], offset_minutes: int) -> str:
        if not value:
            return "—"
        cleaned = value.replace("Z", "+00:00") if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(cleaned)
        except ValueError:
            return value

        if parsed.tzinfo is None:
            assumed_zone = target_zone
            if assumed_zone is None:
                try:
                    assumed_zone = datetime.now().astimezone().tzinfo
                except Exception:
                    assumed_zone = None
            if assumed_zone:
                parsed = parsed.replace(tzinfo=assumed_zone)
            elif target_zone:
                parsed = parsed.replace(tzinfo=target_zone)

        if offset_minutes:
            parsed = parsed + timedelta(minutes=offset_minutes)

        if target_zone:
            try:
                parsed = parsed.astimezone(target_zone)
            except Exception:
                pass
        return parsed.strftime("%d.%m.%Y %H:%M:%S")

    @staticmethod
    def _parse_utc_offset_minutes(label: str) -> Optional[int]:
        if not label or not label.upper().startswith("UTC"):
            return None
        raw = label[3:].strip()
        if not raw:
            return 0
        try:
            sign = 1
            if raw.startswith("-"):
                sign = -1
                raw = raw[1:]
            elif raw.startswith("+"):
                raw = raw[1:]
            if not raw:
                return 0
            parts = raw.split(":")
            hours = int(parts[0]) if parts[0] else 0
            minutes = int(parts[1]) if len(parts) > 1 and parts[1] else 0
            return sign * (hours * 60 + minutes)
        except (ValueError, TypeError):
            return None

    def _get_target_zone(self) -> Optional[timezone]:
        tz_value = self.settings.get_timezone()
        offset_minutes = self._parse_utc_offset_minutes(tz_value)
        if offset_minutes is not None:
            try:
                return timezone(timedelta(minutes=offset_minutes))
            except Exception:
                pass
        try:
            return ZoneInfo(tz_value)
        except Exception:
            return None

    def _draw_grid(self) -> None:
        for i in reversed(range(self.grid_layout.count())):
            item = self.grid_layout.takeAt(i)
            widget = item.widget()
            if widget:
                widget.setParent(None)

        self.channel_labels.clear()
        self.grid_cells: Dict[int, ChannelView] = {}
        for col in range(3):
            self.grid_layout.setColumnStretch(col, 0)
        for row in range(3):
            self.grid_layout.setRowStretch(row, 0)
        channels = self.settings.get_channels()
        if self.current_grid == "1x1" and self.focused_channel_name:
            focused = [
                channel
                for channel in channels
                if channel.get("name") == self.focused_channel_name
            ]
            if focused:
                channels = focused
        rows, cols = map(int, self.current_grid.split("x"))
        for col in range(cols):
            self.grid_layout.setColumnStretch(col, 1)
        for row in range(rows):
            self.grid_layout.setRowStretch(row, 1)
        index = 0
        for row in range(rows):
            for col in range(cols):
                label = ChannelView(f"Канал {index+1}", self._pixmap_pool)
                label.set_grid_position(index)
                channel_name: Optional[str] = None
                if index < len(channels):
                    channel_name = channels[index].get("name", f"Канал {index+1}")
                    self.channel_labels[channel_name] = label
                label.set_channel_name(channel_name)
                label.channelDropped.connect(self._on_channel_dropped)
                label.channelActivated.connect(self._on_channel_activated)
                label.dragStarted.connect(self._on_drag_started)
                label.dragFinished.connect(self._on_drag_finished)
                self.grid_layout.addWidget(label, row, col)
                self.grid_cells[index] = label
                index += 1

        self.grid_layout.invalidate()
        self.grid_widget.updateGeometry()

    def _on_grid_combo_changed(self, index: int) -> None:
        variant = self.grid_combo.itemData(index)
        if variant:
            self._select_grid(str(variant))

    def _on_drag_started(self) -> None:
        self._drag_counter += 1
        if self._drag_counter == 1:
            self._skip_frame_updates = True
            self.grid_widget.setUpdatesEnabled(False)

    def _on_drag_finished(self) -> None:
        if self._drag_counter:
            self._drag_counter -= 1
        if self._drag_counter == 0:
            self._skip_frame_updates = False
            self.grid_widget.setUpdatesEnabled(True)

    def _on_channel_dropped(self, source_index: int, target_index: int) -> None:
        channels = self.settings.get_channels()
        if (
            source_index == target_index
            or source_index < 0
            or source_index >= len(channels)
        ):
            return

        if target_index >= len(channels):
            channel = channels.pop(source_index)
            channels.append(channel)
        else:
            channels[source_index], channels[target_index] = (
                channels[target_index],
                channels[source_index],
            )

        self._update_channels_list_names(channels)
        self._schedule_channels_save(channels)
        self.grid_widget.setUpdatesEnabled(False)
        try:
            if not self._swap_channel_views(source_index, target_index):
                self._draw_grid()
        finally:
            self.grid_widget.setUpdatesEnabled(True)

    def _swap_channel_views(self, source_index: int, target_index: int) -> bool:
        source_view = getattr(self, "grid_cells", {}).get(source_index)
        target_view = getattr(self, "grid_cells", {}).get(target_index)
        if not source_view or not target_view:
            return False

        source_name = source_view.channel_name()
        target_name = target_view.channel_name()
        source_view.set_channel_name(target_name)
        target_view.set_channel_name(source_name)

        self.channel_labels.clear()
        for view in self.grid_cells.values():
            name = view.channel_name()
            if name:
                self.channel_labels[name] = view

        for index, view in self.grid_cells.items():
            view.set_grid_position(index)
        return True

    def _update_channels_list_names(self, channels: List[Dict[str, Any]]) -> None:
        current_row = self.channels_list.currentRow()
        self.channels_list.blockSignals(True)
        self.channels_list.clear()
        for channel in channels:
            self.channels_list.addItem(channel.get("name", "Канал"))
        self.channels_list.blockSignals(False)
        if self.channels_list.count():
            target_row = min(current_row if current_row >= 0 else 0, self.channels_list.count() - 1)
            self.channels_list.setCurrentRow(target_row)

    def _schedule_channels_save(self, channels: List[Dict[str, Any]]) -> None:
        self._pending_channels = [dict(channel) for channel in channels]
        self._channel_save_timer.start(150)

    def _flush_pending_channels(self) -> None:
        if self._pending_channels is None:
            return
        self.settings.save_channels(self._pending_channels)
        self._pending_channels = None

    def _on_channel_activated(self, channel_name: str) -> None:
        if self.current_grid == "1x1" and self._previous_grid:
            previous = self._previous_grid
            self._previous_grid = None
            self.focused_channel_name = None
            self._select_grid(previous)
            return

        self.focused_channel_name = channel_name
        self._select_grid("1x1", focused=True)

    def _select_grid(self, grid: str, focused: bool = False) -> None:
        if grid != "1x1":
            self.focused_channel_name = None if not focused else self.focused_channel_name
            self._previous_grid = None
        elif not self._previous_grid and self.current_grid != "1x1":
            self._previous_grid = self.current_grid
        self.current_grid = grid
        if hasattr(self, "grid_combo"):
            combo_index = self.grid_combo.findData(grid)
            if combo_index >= 0 and combo_index != self.grid_combo.currentIndex():
                self.grid_combo.blockSignals(True)
                self.grid_combo.setCurrentIndex(combo_index)
                self.grid_combo.blockSignals(False)
        self.settings.save_grid(grid)
        self._draw_grid()

    def _start_channels(self) -> None:
        self._stop_workers()
        self.channel_workers = []
        reconnect_conf = self.settings.get_reconnect()
        plate_settings = self.settings.get_plate_settings()
        for channel_conf in self.settings.get_channels():
            source = str(channel_conf.get("source", "")).strip()
            channel_name = channel_conf.get("name", "Канал")
            if not source:
                label = self.channel_labels.get(channel_name)
                if label:
                    label.set_status("Нет источника")
                continue
            worker = ChannelWorker(
                channel_conf,
                self.settings.get_db_path(),
                self.settings.get_screenshot_dir(),
                reconnect_conf,
                plate_settings,
            )
            worker.frame_ready.connect(self._update_frame)
            worker.event_ready.connect(self._handle_event)
            worker.status_ready.connect(self._handle_status)
            self.channel_workers.append(worker)
            worker.start()

    def _stop_workers(self) -> None:
        for worker in self.channel_workers:
            worker.stop()
            worker.wait(1000)
        self.channel_workers = []

    def _update_frame(self, channel_name: str, image: QtGui.QImage) -> None:
        if self._skip_frame_updates:
            return
        label = self.channel_labels.get(channel_name)
        if not label:
            return
        target_size = label.video_label.contentsRect().size()
        if target_size.isEmpty():
            return

        scaled_image = image.scaled(
            target_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        label.set_pixmap(QtGui.QPixmap.fromImage(scaled_image))

    @staticmethod
    def _load_image_from_path(path: Optional[str]) -> Optional[QtGui.QImage]:
        if not path or not os.path.exists(path):
            return None
        image = cv2.imread(path)
        if image is None:
            return None
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb.shape
        bytes_per_line = 3 * width
        return QtGui.QImage(rgb.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888).copy()

    @staticmethod
    def _image_weight(images: Tuple[Optional[QtGui.QImage], Optional[QtGui.QImage]]) -> int:
        frame_image, plate_image = images
        frame_bytes = frame_image.byteCount() if frame_image else 0
        plate_bytes = plate_image.byteCount() if plate_image else 0
        return frame_bytes + plate_bytes

    def _discard_event_images(self, event_id: int) -> None:
        images = self.event_images.pop(event_id, None)
        if images:
            self._image_cache_bytes = max(0, self._image_cache_bytes - self._image_weight(images))

    def _store_event_images(
        self,
        event_id: int,
        images: Tuple[Optional[QtGui.QImage], Optional[QtGui.QImage]],
    ) -> None:
        existing = self.event_images.pop(event_id, None)
        if existing:
            self._image_cache_bytes = max(0, self._image_cache_bytes - self._image_weight(existing))

        frame_image, plate_image = images
        if frame_image is None and plate_image is None:
            return

        self.event_images[event_id] = (frame_image, plate_image)
        self.event_images.move_to_end(event_id)
        self._image_cache_bytes += self._image_weight((frame_image, plate_image))
        self._prune_image_cache()

    def _handle_event(self, event: Dict) -> None:
        event_id = int(event.get("id", 0))
        frame_image = event.get("frame_image")
        plate_image = event.get("plate_image")
        if event_id:
            self._store_event_images(event_id, (frame_image, plate_image))
            self.event_cache[event_id] = event
        channel_label = self.channel_labels.get(event.get("channel", ""))
        if channel_label:
            channel_label.set_last_plate(event.get("plate", ""))
        if event_id:
            self._insert_event_row(event, position=0)
            self._trim_events_table()
        else:
            self._refresh_events_table()
        self._show_event_details(event_id)

    def _cleanup_event_images(self, valid_ids: set[int]) -> None:
        for stale_id in list(self.event_images.keys()):
            if stale_id not in valid_ids:
                self._discard_event_images(stale_id)
        self._prune_image_cache()

    def _get_flag_icon(self, country: Optional[str]) -> Optional[QtGui.QIcon]:
        if not country:
            return None
        code = str(country).lower()
        if code in self.flag_cache:
            return self.flag_cache[code]
        flag_path = self.flag_dir / f"{code}.png"
        if flag_path.exists():
            icon = QtGui.QIcon(str(flag_path))
            self.flag_cache[code] = icon
            return icon
        self.flag_cache[code] = None
        return None

    def _get_country_name(self, country: Optional[str]) -> str:
        if not country:
            return "—"
        code = str(country).upper()
        return self.country_display_names.get(code, code)

    @staticmethod
    def _format_direction(direction: Optional[str]) -> str:
        if not direction:
            return "—"
        normalized = str(direction).upper()
        mapping = {"APPROACHING": "К камере", "RECEDING": "От камеры"}
        return mapping.get(normalized, "—")

    def _prune_image_cache(self) -> None:
        """Ограничивает размер кеша изображений, удаляя самые старые записи."""

        valid_ids = set(self.event_cache.keys())
        for event_id in list(self.event_images.keys()):
            if event_id not in valid_ids:
                self._discard_event_images(event_id)

        while self.event_images and (
            len(self.event_images) > self.MAX_IMAGE_CACHE
            or self._image_cache_bytes > self.MAX_IMAGE_CACHE_BYTES
        ):
            stale_id, images = self.event_images.popitem(last=False)
            self._image_cache_bytes = max(0, self._image_cache_bytes - self._image_weight(images))

    def _insert_event_row(self, event: Dict, position: Optional[int] = None) -> None:
        row_index = position if position is not None else self.events_table.rowCount()
        self.events_table.insertRow(row_index)

        target_zone = self._get_target_zone()
        offset_minutes = self.settings.get_time_offset_minutes()
        timestamp = self._format_timestamp(event.get("timestamp", ""), target_zone, offset_minutes)
        plate = event.get("plate", "—")
        channel = event.get("channel", "—")
        event_id = int(event.get("id") or 0)
        country_code = (event.get("country") or "").upper()
        direction = self._format_direction(event.get("direction"))

        id_item = QtWidgets.QTableWidgetItem(timestamp)
        id_item.setData(QtCore.Qt.UserRole, event_id)
        self.events_table.setItem(row_index, 0, id_item)
        self.events_table.setItem(row_index, 1, QtWidgets.QTableWidgetItem(plate))
        country_item = QtWidgets.QTableWidgetItem("")
        country_item.setData(QtCore.Qt.UserRole, country_code)
        country_item.setData(QtCore.Qt.TextAlignmentRole, QtCore.Qt.AlignCenter)
        country_icon = self._get_flag_icon(event.get("country"))
        if country_icon:
            country_item.setData(QtCore.Qt.DecorationRole, country_icon)
        country_name = self._get_country_name(country_code)
        if country_name != "—":
            country_item.setToolTip(country_name)
        self.events_table.setItem(row_index, 2, country_item)
        self.events_table.setItem(row_index, 3, QtWidgets.QTableWidgetItem(channel))
        self.events_table.setItem(row_index, 4, QtWidgets.QTableWidgetItem(direction))

    def _trim_events_table(self, max_rows: int = 200) -> None:
        while self.events_table.rowCount() > max_rows:
            last_row = self.events_table.rowCount() - 1
            item = self.events_table.item(last_row, 0)
            event_id = int(item.data(QtCore.Qt.UserRole) or 0) if item else 0
            self.events_table.removeRow(last_row)
            if event_id and event_id in self.event_cache:
                self.event_cache.pop(event_id, None)
            if event_id and event_id in self.event_images:
                self._discard_event_images(event_id)

    def _handle_status(self, channel: str, status: str) -> None:
        label = self.channel_labels.get(channel)
        if label:
            normalized = status.lower()
            if "движ" in normalized or "motion" in normalized:
                label.set_status("")
            else:
                label.set_status(status)
            label.set_motion_active("обнаружено" in normalized)

    def _on_event_selected(self) -> None:
        row = self.events_table.currentRow()
        if row < 0:
            return
        event_id_item = self.events_table.item(row, 0)
        if event_id_item is None:
            return
        event_id = int(event_id_item.data(QtCore.Qt.UserRole) or 0)
        self._show_event_details(event_id)

    def _show_event_details(self, event_id: int) -> None:
        event = self.event_cache.get(event_id)
        images = self.event_images.get(event_id, (None, None))
        frame_image, plate_image = images
        if event:
            if frame_image is None and event.get("frame_path"):
                frame_image = self._load_image_from_path(event.get("frame_path"))
            if plate_image is None and event.get("plate_path"):
                plate_image = self._load_image_from_path(event.get("plate_path"))
            self._store_event_images(event_id, (frame_image, plate_image))
        display_event = dict(event) if event else None
        if display_event:
            target_zone = self._get_target_zone()
            offset_minutes = self.settings.get_time_offset_minutes()
            display_event["timestamp"] = self._format_timestamp(
                display_event.get("timestamp", ""), target_zone, offset_minutes
            )
            display_event["country"] = self._get_country_name(display_event.get("country"))
            display_event["direction"] = self._format_direction(display_event.get("direction"))
        self.event_detail.set_event(display_event, frame_image, plate_image)

    def _refresh_events_table(self, select_id: Optional[int] = None) -> None:
        rows = self.db.fetch_recent(limit=200)
        self.events_table.setRowCount(0)
        self.event_cache = {row["id"]: dict(row) for row in rows}
        valid_ids = set(self.event_cache.keys())

        for row_data in rows:
            self._insert_event_row(dict(row_data))

        self._cleanup_event_images(valid_ids)

        if select_id:
            for row in range(self.events_table.rowCount()):
                item = self.events_table.item(row, 0)
                if item and int(item.data(QtCore.Qt.UserRole) or 0) == select_id:
                    self.events_table.selectRow(row)
                    break

    # ------------------ Поиск ------------------
    def _build_search_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        widget.setStyleSheet(self.FORM_STYLE)

        filters_group = QtWidgets.QGroupBox("Фильтры поиска")
        filters_group.setStyleSheet(self.GROUP_BOX_STYLE)
        form = QtWidgets.QFormLayout(filters_group)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        metrics = self.fontMetrics()
        input_width = metrics.horizontalAdvance("00.00.0000 00:00:00") + 40

        self.search_plate = QtWidgets.QLineEdit()
        self.search_plate.setMinimumWidth(input_width)
        self.search_plate.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        self.search_from = QtWidgets.QDateTimeEdit()
        self._prepare_optional_datetime(self.search_from)
        self._apply_dark_calendar_style(self.search_from)
        self.search_from.setMinimumWidth(input_width)
        self.search_from.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed
        )
        self.search_to = QtWidgets.QDateTimeEdit()
        self._prepare_optional_datetime(self.search_to)
        self._apply_dark_calendar_style(self.search_to)
        self.search_to.setMinimumWidth(input_width)
        self.search_to.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed
        )

        self._reset_journal_range()

        form.addRow("Гос.номер:", self.search_plate)
        form.addRow("Дата с:", self.search_from)
        form.addRow("Дата по:", self.search_to)
        layout.addWidget(filters_group)

        button_row = QtWidgets.QHBoxLayout()
        button_row.setSpacing(10)
        search_btn = QtWidgets.QPushButton("Поиск")
        search_btn.clicked.connect(self._run_plate_search)
        search_btn.setMinimumWidth(150)
        button_row.addWidget(search_btn)

        reset_btn = QtWidgets.QPushButton("Сбросить фильтр")
        reset_btn.clicked.connect(self._reset_journal_filters)
        reset_btn.setMinimumWidth(180)
        button_row.addWidget(reset_btn)
        button_row.addStretch()
        layout.addLayout(button_row)

        self.search_table = QtWidgets.QTableWidget(0, 6)
        self.search_table.setHorizontalHeaderLabels(
            ["Дата/Время", "Канал", "Страна", "Гос. номер", "Уверенность", "Источник"]
        )
        header = self.search_table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        header.setStretchLastSection(True)
        header.setMinimumSectionSize(90)
        self.search_table.setColumnWidth(0, 220)
        self.search_table.setStyleSheet(self.TABLE_STYLE)
        self.search_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.search_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.search_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.search_table.verticalHeader().setVisible(False)
        self.search_table.itemActivated.connect(self._on_journal_event_activated)
        layout.addWidget(self.search_table)

        self._run_plate_search()

        return widget

    def _reset_journal_range(self) -> None:
        today = QtCore.QDate.currentDate()
        start_of_day = QtCore.QDateTime(today, QtCore.QTime(0, 0))
        end_of_day = QtCore.QDateTime(today, QtCore.QTime(23, 59))
        self.search_from.setDateTime(start_of_day)
        self.search_to.setDateTime(end_of_day)

    def _reset_journal_filters(self) -> None:
        self.search_plate.clear()
        self._reset_journal_range()
        self._run_plate_search()

    def _run_plate_search(self) -> None:
        start = self._get_datetime_value(self.search_from)
        end = self._get_datetime_value(self.search_to)
        plate_fragment = self.search_plate.text().strip()
        if plate_fragment:
            rows = self.db.search_by_plate(
                plate_fragment, start=start or None, end=end or None, limit=200
            )
        else:
            rows = self.db.fetch_filtered(start=start or None, end=end or None, limit=50)

        self._populate_journal_table(rows)

    def _populate_journal_table(self, rows: List[Dict]) -> None:
        self.search_table.setRowCount(0)
        self.search_results.clear()
        target_zone = self._get_target_zone()
        offset_minutes = self.settings.get_time_offset_minutes()

        for db_row in rows:
            row_data = dict(db_row)
            event_id = int(row_data["id"])
            self.search_results[event_id] = dict(row_data)
            row_index = self.search_table.rowCount()
            self.search_table.insertRow(row_index)
            formatted_time = self._format_timestamp(
                row_data["timestamp"], target_zone, offset_minutes
            )
            time_item = QtWidgets.QTableWidgetItem(formatted_time)
            time_item.setData(QtCore.Qt.UserRole, event_id)
            self.search_table.setItem(row_index, 0, time_item)
            self.search_table.setItem(row_index, 1, QtWidgets.QTableWidgetItem(row_data["channel"]))

            country_item = QtWidgets.QTableWidgetItem(row_data.get("country") or "")
            country_icon = self._get_flag_icon(row_data.get("country"))
            if country_icon:
                country_item.setIcon(country_icon)
            country_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.search_table.setItem(row_index, 2, country_item)

            self.search_table.setItem(row_index, 3, QtWidgets.QTableWidgetItem(row_data["plate"]))
            self.search_table.setItem(
                row_index, 4, QtWidgets.QTableWidgetItem(f"{row_data['confidence'] or 0:.2f}")
            )
            self.search_table.setItem(row_index, 5, QtWidgets.QTableWidgetItem(row_data["source"]))

    def _on_journal_event_activated(self, item: QtWidgets.QTableWidgetItem) -> None:
        row = item.row()
        if row < 0:
            return
        self._open_journal_event(row)

    def _open_journal_event(self, row: int) -> None:
        event_id_item = self.search_table.item(row, 0)
        if event_id_item is None:
            return
        event_id = int(event_id_item.data(QtCore.Qt.UserRole) or 0)
        event = self.search_results.get(event_id)
        if not event:
            return

        frame_image = self._load_image_from_path(event.get("frame_path"))
        plate_image = self._load_image_from_path(event.get("plate_path"))

        display_event = dict(event)
        target_zone = self._get_target_zone()
        offset_minutes = self.settings.get_time_offset_minutes()
        display_event["timestamp"] = self._format_timestamp(
            display_event.get("timestamp", ""), target_zone, offset_minutes
        )
        display_event["country"] = self._get_country_name(display_event.get("country"))

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Информация о событии")
        dialog.setMinimumSize(720, 640)
        dialog.setWindowFlags(
            QtCore.Qt.Dialog
            | QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowSystemMenuHint
            | QtCore.Qt.WindowMinMaxButtonsHint
        )
        dialog.setStyleSheet(
            f"QDialog {{ background-color: {self.SURFACE_COLOR}; border: 1px solid #20242c; border-radius: 12px; }}"
            f"QGroupBox {{ background-color: {self.PANEL_COLOR}; color: #f6f7fb; border: 1px solid #20242c; border-radius: 12px; padding: 10px; margin-top: 8px; }}"
            "QLabel { color: #e5e7eb; }"
            "QToolButton { background-color: transparent; border: none; color: white; padding: 6px; }"
            "QToolButton:hover { background-color: rgba(255,255,255,0.08); border-radius: 6px; }"
            f"QPushButton {{ background-color: {self.ACCENT_COLOR}; color: #0b0c10; border: none; border-radius: 8px; padding: 8px 16px; font-weight: 700; }}"
            f"QPushButton:hover {{ background-color: #4ddcf3; }}"
        )
        dialog_layout = QtWidgets.QVBoxLayout(dialog)
        dialog_layout.setContentsMargins(0, 0, 0, 12)

        header = QtWidgets.QFrame()
        header.setFixedHeight(44)
        header.setStyleSheet(
            f"QFrame {{ background-color: #0b0c10; border-bottom: 1px solid #20242c; border-top-left-radius: 12px; border-top-right-radius: 12px; }}"
        )
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(12, 6, 12, 6)
        header_layout.setSpacing(10)

        title = QtWidgets.QLabel("Информация о событии")
        title.setStyleSheet("color: white; font-weight: 800;")
        header_layout.addWidget(title)
        header_layout.addStretch()

        maximize_btn = QtWidgets.QToolButton()
        maximize_btn.setText("⛶")
        maximize_btn.setToolTip("Развернуть/свернуть окно")

        def toggle_maximize() -> None:
            if dialog.isMaximized():
                dialog.showNormal()
            else:
                dialog.showMaximized()

        maximize_btn.clicked.connect(toggle_maximize)
        header_layout.addWidget(maximize_btn)

        close_btn = QtWidgets.QToolButton()
        close_btn.setText("✕")
        close_btn.setToolTip("Закрыть")
        close_btn.clicked.connect(dialog.accept)
        header_layout.addWidget(close_btn)

        def start_move(event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
            if event.button() == QtCore.Qt.LeftButton:
                header._drag_pos = event.globalPos() - dialog.frameGeometry().topLeft()  # type: ignore[attr-defined]
                event.accept()

        def move_window(event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
            if event.buttons() & QtCore.Qt.LeftButton and hasattr(header, "_drag_pos") and not dialog.isMaximized():
                dialog.move(event.globalPos() - header._drag_pos)  # type: ignore[attr-defined]
                event.accept()

        header.mousePressEvent = start_move  # type: ignore[assignment]
        header.mouseMoveEvent = move_window  # type: ignore[assignment]

        dialog_layout.addWidget(header)

        details = EventDetailView()
        details.set_event(display_event, frame_image, plate_image)
        dialog_layout.addWidget(details)

        footer_row = QtWidgets.QHBoxLayout()
        footer_row.addStretch()
        bottom_close_btn = QtWidgets.QPushButton("Закрыть")
        bottom_close_btn.clicked.connect(dialog.accept)
        footer_row.addWidget(bottom_close_btn)
        dialog_layout.addLayout(footer_row)

        dialog.exec_()

    # ------------------ Настройки ------------------
    def _build_settings_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        content = QtWidgets.QFrame()
        content.setStyleSheet(
            f"QFrame {{ background-color: {self.SURFACE_COLOR}; border: none; border-radius: 14px; }}"
        )
        content_layout = QtWidgets.QHBoxLayout(content)
        content_layout.setContentsMargins(12, 12, 12, 12)
        content_layout.setSpacing(12)

        self.settings_nav = QtWidgets.QListWidget()
        self.settings_nav.setFixedWidth(220)
        self.settings_nav.setStyleSheet(self.LIST_STYLE)
        self.settings_nav.addItem("Общие")
        self.settings_nav.addItem("Каналы")
        content_layout.addWidget(self.settings_nav)

        self.settings_stack = QtWidgets.QStackedWidget()
        self.settings_stack.addWidget(self._build_general_settings_tab())
        self.settings_stack.addWidget(self._build_channel_settings_tab())
        content_layout.addWidget(self.settings_stack, 1)

        layout.addWidget(content, 1)

        self.settings_nav.currentRowChanged.connect(self.settings_stack.setCurrentIndex)
        self.settings_nav.setCurrentRow(0)
        return widget

    def _build_general_settings_tab(self) -> QtWidgets.QWidget:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setStyleSheet(
            f"QScrollArea {{ background: transparent; border: none; }}"
            f"QScrollArea > QWidget > QWidget {{ background-color: {self.SURFACE_COLOR}; }}"
        )

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        widget.setStyleSheet(self.FORM_STYLE)

        section_style = f"QFrame {{ background-color: {self.PANEL_COLOR}; border: none; border-radius: 12px; }}"

        def make_section(title: str) -> tuple[QtWidgets.QFrame, QtWidgets.QFormLayout]:
            frame = QtWidgets.QFrame()
            frame.setStyleSheet(section_style)
            frame_layout = QtWidgets.QVBoxLayout(frame)
            frame_layout.setContentsMargins(14, 12, 14, 12)
            frame_layout.setSpacing(10)

            header = QtWidgets.QLabel(title)
            header.setStyleSheet("font-size: 14px; font-weight: 800; color: #e5e7eb;")
            frame_layout.addWidget(header)

            form = QtWidgets.QFormLayout()
            form.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            form.setFormAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
            form.setHorizontalSpacing(14)
            form.setVerticalSpacing(10)
            form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
            frame_layout.addLayout(form)

            return frame, form

        reconnect_group, reconnect_form = make_section("Стабильность каналов")
        self.reconnect_on_loss_checkbox = QtWidgets.QCheckBox("Переподключение при потере сигнала")
        reconnect_form.addRow(self.reconnect_on_loss_checkbox)

        self.frame_timeout_input = QtWidgets.QSpinBox()
        self.frame_timeout_input.setMaximumWidth(140)
        self.frame_timeout_input.setRange(1, 300)
        self.frame_timeout_input.setSuffix(" с")
        self.frame_timeout_input.setToolTip("Сколько секунд ждать кадр перед попыткой переподключения")
        reconnect_form.addRow("Таймаут ожидания кадра:", self.frame_timeout_input)

        self.retry_interval_input = QtWidgets.QSpinBox()
        self.retry_interval_input.setMaximumWidth(140)
        self.retry_interval_input.setRange(1, 300)
        self.retry_interval_input.setSuffix(" с")
        self.retry_interval_input.setToolTip("Интервал между попытками переподключения при потере сигнала")
        reconnect_form.addRow("Интервал между попытками:", self.retry_interval_input)

        self.periodic_reconnect_checkbox = QtWidgets.QCheckBox("Переподключение по таймеру")
        reconnect_form.addRow(self.periodic_reconnect_checkbox)

        self.periodic_interval_input = QtWidgets.QSpinBox()
        self.periodic_interval_input.setMaximumWidth(140)
        self.periodic_interval_input.setRange(1, 1440)
        self.periodic_interval_input.setSuffix(" мин")
        self.periodic_interval_input.setToolTip("Плановое переподключение каждые N минут")
        reconnect_form.addRow("Интервал переподключения:", self.periodic_interval_input)

        storage_group, storage_form = make_section("Хранилище")
        storage_group.setMaximumWidth(self.FIELD_MAX_WIDTH + 220)
        storage_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldsStayAtSizeHint)

        db_row = QtWidgets.QHBoxLayout()
        db_row.setContentsMargins(0, 0, 0, 0)
        db_row.setSpacing(8)
        self.db_dir_input = QtWidgets.QLineEdit()
        self.db_dir_input.setMaximumWidth(self.FIELD_MAX_WIDTH)
        self.db_dir_input.setMinimumWidth(320)
        self.db_dir_input.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        browse_db_btn = QtWidgets.QPushButton("Выбрать...")
        self._polish_button(browse_db_btn, 130)
        browse_db_btn.clicked.connect(self._choose_db_dir)
        db_row.addWidget(self.db_dir_input)
        db_row.addWidget(browse_db_btn)
        db_container = QtWidgets.QWidget()
        db_container.setLayout(db_row)
        db_container.setMaximumWidth(self.FIELD_MAX_WIDTH + 60)
        db_container.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        storage_form.addRow("Папка БД:", db_container)

        screenshot_row = QtWidgets.QHBoxLayout()
        screenshot_row.setContentsMargins(0, 0, 0, 0)
        screenshot_row.setSpacing(8)
        self.screenshot_dir_input = QtWidgets.QLineEdit()
        self.screenshot_dir_input.setMaximumWidth(self.FIELD_MAX_WIDTH)
        self.screenshot_dir_input.setMinimumWidth(320)
        self.screenshot_dir_input.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        browse_screenshot_btn = QtWidgets.QPushButton("Выбрать...")
        self._polish_button(browse_screenshot_btn, 130)
        browse_screenshot_btn.clicked.connect(self._choose_screenshot_dir)
        screenshot_row.addWidget(self.screenshot_dir_input)
        screenshot_row.addWidget(browse_screenshot_btn)
        screenshot_container = QtWidgets.QWidget()
        screenshot_container.setLayout(screenshot_row)
        screenshot_container.setMaximumWidth(self.FIELD_MAX_WIDTH + 60)
        screenshot_container.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        storage_form.addRow("Папка для скриншотов:", screenshot_container)

        time_group, time_form = make_section("Дата и время")
        time_group.setMaximumWidth(self.FIELD_MAX_WIDTH + 220)

        self.timezone_combo = QtWidgets.QComboBox()
        self.timezone_combo.setEditable(False)
        self.timezone_combo.setMinimumWidth(self.COMPACT_FIELD_WIDTH + 80)
        self.timezone_combo.setStyleSheet(
            "QComboBox { background-color: #0b0c10; color: #f8fafc; border: 1px solid #1f2937; }"
            "QComboBox QAbstractItemView { background-color: #0b0c10; color: #f8fafc; selection-background-color: #1f2937; }"
            "QComboBox:on { padding-top: 3px; padding-left: 4px; }"
        )
        for label in self._available_offset_labels():
            self.timezone_combo.addItem(label, label)
        time_form.addRow("Часовой пояс:", self.timezone_combo)

        time_row = QtWidgets.QHBoxLayout()
        time_row.setSpacing(8)
        time_row.setContentsMargins(0, 0, 0, 0)
        self.time_correction_input = QtWidgets.QDateTimeEdit(QtCore.QDateTime.currentDateTime())
        self.time_correction_input.setDisplayFormat("dd.MM.yyyy HH:mm:ss")
        self.time_correction_input.setCalendarPopup(True)
        self.time_correction_input.setMaximumWidth(self.FIELD_MAX_WIDTH)
        self.time_correction_input.setStyleSheet(
            "QDateTimeEdit { background-color: #0b0c10; color: #f8fafc; border: 1px solid #1f2937; border-radius: 8px; padding: 8px; }"
            "QDateTimeEdit::drop-down { width: 26px; }")
        calendar = self.time_correction_input.calendarWidget()
        calendar.setStyleSheet(
            "QCalendarWidget QWidget { background-color: #0b0c10; color: #f8fafc; }"
            "QCalendarWidget QToolButton { background-color: #1f2937; color: #f8fafc; border: none; padding: 6px; }"
            "QCalendarWidget QToolButton#qt_calendar_prevmonth, QCalendarWidget QToolButton#qt_calendar_nextmonth { width: 24px; }"
            "QCalendarWidget QSpinBox { background-color: #1f2937; color: #f8fafc; }"
            "QCalendarWidget QTableView { selection-background-color: rgba(34,211,238,0.18); selection-color: #22d3ee; alternate-background-color: #11131a; }")
        sync_now_btn = QtWidgets.QPushButton("Текущее время")
        self._polish_button(sync_now_btn, 160)
        sync_now_btn.clicked.connect(self._sync_time_now)
        time_row.addWidget(self.time_correction_input)
        time_row.addWidget(sync_now_btn)
        time_row.addStretch()
        time_container = QtWidgets.QWidget()
        time_container.setLayout(time_row)
        time_form.addRow("Коррекция времени:", time_container)

        plate_group, plate_form = make_section("Валидация номеров")

        plate_dir_row = QtWidgets.QHBoxLayout()
        plate_dir_row.setContentsMargins(0, 0, 0, 0)
        plate_dir_row.setSpacing(8)
        self.country_config_dir_input = QtWidgets.QLineEdit()
        self.country_config_dir_input.setMaximumWidth(self.FIELD_MAX_WIDTH)
        self.country_config_dir_input.setMinimumWidth(320)
        self.country_config_dir_input.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        browse_country_btn = QtWidgets.QPushButton("Выбрать...")
        self._polish_button(browse_country_btn, 130)
        browse_country_btn.clicked.connect(self._choose_country_dir)
        self.country_config_dir_input.editingFinished.connect(self._reload_country_templates)
        plate_dir_row.addWidget(self.country_config_dir_input)
        plate_dir_row.addWidget(browse_country_btn)
        plate_dir_container = QtWidgets.QWidget()
        plate_dir_container.setLayout(plate_dir_row)
        plate_dir_container.setMaximumWidth(self.FIELD_MAX_WIDTH + 60)
        plate_dir_container.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        plate_form.addRow("Каталог шаблонов:", plate_dir_container)

        self.country_templates_list = QtWidgets.QListWidget()
        self.country_templates_list.setMaximumWidth(self.FIELD_MAX_WIDTH)
        self.country_templates_list.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        self.country_templates_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.country_templates_list.setStyleSheet(self.LIST_STYLE)
        plate_form.addRow("Активные страны:", self.country_templates_list)

        refresh_countries_btn = QtWidgets.QPushButton("Обновить список стран")
        self._polish_button(refresh_countries_btn, 180)
        refresh_countries_btn.clicked.connect(self._reload_country_templates)
        plate_form.addRow("", refresh_countries_btn)

        save_card = QtWidgets.QFrame()
        save_card.setStyleSheet(section_style)
        save_row = QtWidgets.QHBoxLayout(save_card)
        save_row.setContentsMargins(14, 12, 14, 12)
        save_row.setSpacing(10)
        save_general_btn = QtWidgets.QPushButton("Сохранить")
        self._polish_button(save_general_btn, 220)
        save_general_btn.clicked.connect(self._save_general_settings)
        save_row.addWidget(save_general_btn, 0)
        save_row.addStretch(1)

        layout.addWidget(reconnect_group)
        layout.addWidget(storage_group)
        layout.addWidget(time_group)
        layout.addWidget(plate_group)
        layout.addWidget(save_card)
        layout.addStretch()

        scroll.setWidget(widget)
        self._load_general_settings()
        return scroll

    def _build_channel_settings_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        widget.setStyleSheet(self.GROUP_BOX_STYLE)

        left_panel = QtWidgets.QVBoxLayout()
        left_panel.setSpacing(6)
        self.channels_list = QtWidgets.QListWidget()
        self.channels_list.setFixedWidth(180)
        self.channels_list.setStyleSheet(self.LIST_STYLE)
        self.channels_list.currentRowChanged.connect(self._load_channel_form)
        left_panel.addWidget(self.channels_list)

        list_buttons = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Добавить")
        self._polish_button(add_btn, 140)
        add_btn.clicked.connect(self._add_channel)
        remove_btn = QtWidgets.QPushButton("Удалить")
        self._polish_button(remove_btn, 140)
        remove_btn.clicked.connect(self._remove_channel)
        list_buttons.addWidget(add_btn)
        list_buttons.addWidget(remove_btn)
        left_panel.addLayout(list_buttons)
        layout.addLayout(left_panel)

        self.channel_details_container = QtWidgets.QWidget()
        details_layout = QtWidgets.QHBoxLayout(self.channel_details_container)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(12)

        center_panel = QtWidgets.QVBoxLayout()
        self.preview = ROIEditor()
        self.preview.roi_changed.connect(self._on_roi_drawn)
        self.preview.plate_size_selected.connect(self._on_plate_size_selected)
        center_panel.addWidget(self.preview)
        details_layout.addLayout(center_panel, 2)

        right_panel = QtWidgets.QVBoxLayout()
        right_panel.setSpacing(10)

        tab_styles = (
            "QTabBar { font-weight: 700; }"
            f"QTabBar::tab {{ background: #0b0c10; color: #9ca3af; padding: 8px 14px; border: 1px solid #0f1115; border-top-left-radius: 10px; border-top-right-radius: 10px; margin-right: 6px; }}"
            f"QTabBar::tab:selected {{ background: #16181d; color: {self.ACCENT_COLOR}; border: 1px solid #20242c; border-bottom: 2px solid {self.ACCENT_COLOR}; }}"
            "QTabWidget::pane { border: 1px solid #20242c; border-radius: 10px; background-color: #16181d; top: -1px; }"
            "QWidget { background-color: #16181d; color: #cbd5e1; }"
            "QLabel { color: #cbd5e1; font-size: 13px; }"
            "QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background-color: #0b0c10; color: #f8fafc; border: 1px solid #1f2937; border-radius: 8px; padding: 8px; }"
        )

        tabs = QtWidgets.QTabWidget()
        tabs.setStyleSheet(tab_styles)

        def make_form_tab() -> QtWidgets.QFormLayout:
            tab_widget = QtWidgets.QWidget()
            form = QtWidgets.QFormLayout(tab_widget)
            form.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            form.setFormAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
            form.setHorizontalSpacing(14)
            form.setVerticalSpacing(12)
            form.setContentsMargins(12, 12, 12, 12)
            tabs.addTab(tab_widget, "")
            return form

        channel_form = make_form_tab()
        tabs.setTabText(0, "Канал")
        self.channel_name_input = QtWidgets.QLineEdit()
        self.channel_name_input.setMaximumWidth(self.FIELD_MAX_WIDTH)
        self.channel_source_input = QtWidgets.QLineEdit()
        self.channel_source_input.setMaximumWidth(self.FIELD_MAX_WIDTH)
        channel_form.addRow("Название:", self.channel_name_input)
        channel_form.addRow("Источник/RTSP:", self.channel_source_input)

        recognition_form = make_form_tab()
        tabs.setTabText(1, "Распознавание")
        self.best_shots_input = QtWidgets.QSpinBox()
        self.best_shots_input.setRange(1, 50)
        self.best_shots_input.setMaximumWidth(self.COMPACT_FIELD_WIDTH)
        self.best_shots_input.setMinimumWidth(120)
        self.best_shots_input.setToolTip("Количество бстшотов, участвующих в консенсусе трека")
        recognition_form.addRow("Бестшоты на трек:", self.best_shots_input)

        self.cooldown_input = QtWidgets.QSpinBox()
        self.cooldown_input.setRange(0, 3600)
        self.cooldown_input.setMaximumWidth(self.COMPACT_FIELD_WIDTH)
        self.cooldown_input.setMinimumWidth(120)
        self.cooldown_input.setToolTip(
            "Интервал (в секундах), в течение которого не создается повторное событие для того же номера"
        )
        recognition_form.addRow("Пауза повтора (сек):", self.cooldown_input)

        self.min_conf_input = QtWidgets.QDoubleSpinBox()
        self.min_conf_input.setRange(0.0, 1.0)
        self.min_conf_input.setSingleStep(0.05)
        self.min_conf_input.setDecimals(2)
        self.min_conf_input.setMaximumWidth(self.COMPACT_FIELD_WIDTH)
        self.min_conf_input.setMinimumWidth(120)
        self.min_conf_input.setToolTip(
            "Минимальная уверенность OCR (0-1) для приема результата; ниже — помечается как нечитаемое"
        )
        recognition_form.addRow("Мин. уверенность OCR:", self.min_conf_input)

        debug_row = QtWidgets.QHBoxLayout()
        self.debug_detection_checkbox = QtWidgets.QCheckBox("Рамки детекции")
        self.debug_detection_checkbox.setToolTip(
            "Отображать рамки поиска номеров на кадре"
        )
        self.debug_ocr_checkbox = QtWidgets.QCheckBox("Символы OCR")
        self.debug_ocr_checkbox.setToolTip(
            "Выводить распознанные символы поверх рамок"
        )
        debug_row.addWidget(self.debug_detection_checkbox)
        debug_row.addWidget(self.debug_ocr_checkbox)
        debug_row.addStretch(1)
        recognition_form.addRow("Отладка:", debug_row)

        motion_form = make_form_tab()
        tabs.setTabText(2, "Детектор движения")
        self.detection_mode_input = QtWidgets.QComboBox()
        self.detection_mode_input.addItem("Постоянное", "continuous")
        self.detection_mode_input.addItem("Детектор движения", "motion")
        self.detection_mode_input.setMaximumWidth(self.FIELD_MAX_WIDTH)
        motion_form.addRow("Обнаружение ТС:", self.detection_mode_input)

        self.detector_stride_input = QtWidgets.QSpinBox()
        self.detector_stride_input.setRange(1, 12)
        self.detector_stride_input.setMaximumWidth(self.COMPACT_FIELD_WIDTH)
        self.detector_stride_input.setMinimumWidth(120)
        self.detector_stride_input.setToolTip(
            "Запускать YOLO на каждом N-м кадре в зоне распознавания, чтобы снизить нагрузку"
        )
        motion_form.addRow("Шаг инференса (кадр):", self.detector_stride_input)

        self.motion_threshold_input = QtWidgets.QDoubleSpinBox()
        self.motion_threshold_input.setRange(0.0, 1.0)
        self.motion_threshold_input.setDecimals(3)
        self.motion_threshold_input.setSingleStep(0.005)
        self.motion_threshold_input.setMaximumWidth(self.COMPACT_FIELD_WIDTH)
        self.motion_threshold_input.setMinimumWidth(120)
        self.motion_threshold_input.setToolTip("Порог чувствительности по площади изменения внутри ROI")
        motion_form.addRow("Порог движения:", self.motion_threshold_input)

        self.motion_stride_input = QtWidgets.QSpinBox()
        self.motion_stride_input.setRange(1, 30)
        self.motion_stride_input.setMaximumWidth(self.COMPACT_FIELD_WIDTH)
        self.motion_stride_input.setMinimumWidth(120)
        self.motion_stride_input.setToolTip("Обрабатывать каждый N-й кадр для поиска движения")
        motion_form.addRow("Частота анализа (кадр):", self.motion_stride_input)

        self.motion_activation_frames_input = QtWidgets.QSpinBox()
        self.motion_activation_frames_input.setRange(1, 60)
        self.motion_activation_frames_input.setMaximumWidth(self.COMPACT_FIELD_WIDTH)
        self.motion_activation_frames_input.setMinimumWidth(120)
        self.motion_activation_frames_input.setToolTip("Сколько кадров подряд должно быть движение, чтобы включить распознавание")
        motion_form.addRow("Мин. кадров с движением:", self.motion_activation_frames_input)

        self.motion_release_frames_input = QtWidgets.QSpinBox()
        self.motion_release_frames_input.setRange(1, 120)
        self.motion_release_frames_input.setMaximumWidth(self.COMPACT_FIELD_WIDTH)
        self.motion_release_frames_input.setMinimumWidth(120)
        self.motion_release_frames_input.setToolTip("Сколько кадров без движения нужно, чтобы остановить распознавание")
        motion_form.addRow("Мин. кадров без движения:", self.motion_release_frames_input)

        roi_form = make_form_tab()
        tabs.setTabText(3, "Зона распознавания")

        size_group = QtWidgets.QGroupBox("Фильтр по размеру рамки")
        size_layout = QtWidgets.QGridLayout()

        self.min_plate_width_input = QtWidgets.QSpinBox()
        self.min_plate_width_input.setRange(0, 5000)
        self.min_plate_width_input.setMaximumWidth(self.COMPACT_FIELD_WIDTH)
        self.min_plate_width_input.setToolTip("Минимальная ширина рамки, меньшие детекции будут отброшены")

        self.min_plate_width_input.valueChanged.connect(self._sync_plate_rects_from_inputs)

        self.min_plate_height_input = QtWidgets.QSpinBox()
        self.min_plate_height_input.setRange(0, 3000)
        self.min_plate_height_input.setMaximumWidth(self.COMPACT_FIELD_WIDTH)
        self.min_plate_height_input.setToolTip("Минимальная высота рамки, меньшие детекции будут отброшены")

        self.min_plate_height_input.valueChanged.connect(self._sync_plate_rects_from_inputs)

        self.max_plate_width_input = QtWidgets.QSpinBox()
        self.max_plate_width_input.setRange(0, 8000)
        self.max_plate_width_input.setMaximumWidth(self.COMPACT_FIELD_WIDTH)
        self.max_plate_width_input.setToolTip("Максимальная ширина рамки, более крупные детекции будут отброшены")

        self.max_plate_width_input.valueChanged.connect(self._sync_plate_rects_from_inputs)

        self.max_plate_height_input = QtWidgets.QSpinBox()
        self.max_plate_height_input.setRange(0, 4000)
        self.max_plate_height_input.setMaximumWidth(self.COMPACT_FIELD_WIDTH)
        self.max_plate_height_input.setToolTip("Максимальная высота рамки, более крупные детекции будут отброшены")

        self.max_plate_height_input.valueChanged.connect(self._sync_plate_rects_from_inputs)

        size_layout.addWidget(QtWidgets.QLabel("Мин. ширина (px):"), 0, 0)
        size_layout.addWidget(self.min_plate_width_input, 0, 1)
        size_layout.addWidget(QtWidgets.QLabel("Мин. высота (px):"), 0, 2)
        size_layout.addWidget(self.min_plate_height_input, 0, 3)
        size_layout.addWidget(QtWidgets.QLabel("Макс. ширина (px):"), 1, 0)
        size_layout.addWidget(self.max_plate_width_input, 1, 1)
        size_layout.addWidget(QtWidgets.QLabel("Макс. высота (px):"), 1, 2)
        size_layout.addWidget(self.max_plate_height_input, 1, 3)

        self.plate_size_hint = QtWidgets.QLabel(
            "Перетаскивайте прямоугольники мин/макс на превью слева, значения сохраняются автоматически"
        )
        self.plate_size_hint.setStyleSheet("color: #9ca3af; padding-top: 6px;")
        size_layout.addWidget(self.plate_size_hint, 2, 0, 1, 4)
        size_group.setLayout(size_layout)

        roi_form.addRow("", size_group)
        self.roi_points_table = QtWidgets.QTableWidget()
        self.roi_points_table.setColumnCount(2)
        self.roi_points_table.setHorizontalHeaderLabels(["X (px)", "Y (px)"])
        header = self.roi_points_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        header.setDefaultSectionSize(90)
        header.setMinimumSectionSize(80)
        header.setStyleSheet(
            "QHeaderView::section { background-color: #11131a; color: #cbd5e1; border: 1px solid #1f2937; }"
        )
        self.roi_points_table.verticalHeader().setVisible(False)
        self.roi_points_table.setEditTriggers(QtWidgets.QAbstractItemView.AllEditTriggers)
        self.roi_points_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.roi_points_table.setMaximumHeight(220)
        self.roi_points_table.itemChanged.connect(self._on_roi_table_changed)

        roi_buttons = QtWidgets.QHBoxLayout()
        add_point_btn = QtWidgets.QPushButton("Добавить точку")
        self._polish_button(add_point_btn, 150)
        add_point_btn.clicked.connect(self._add_roi_point)
        remove_point_btn = QtWidgets.QPushButton("Удалить точку")
        self._polish_button(remove_point_btn, 150)
        remove_point_btn.clicked.connect(self._remove_roi_point)
        clear_roi_btn = QtWidgets.QPushButton("Очистить зону")
        self._polish_button(clear_roi_btn, 150)
        clear_roi_btn.clicked.connect(self._clear_roi_points)
        roi_buttons.addWidget(add_point_btn)
        roi_buttons.addWidget(remove_point_btn)
        roi_buttons.addWidget(clear_roi_btn)

        roi_form.addRow("Точки ROI:", self.roi_points_table)
        roi_form.addRow("", roi_buttons)

        refresh_btn = QtWidgets.QPushButton("Обновить кадр")
        self._polish_button(refresh_btn, 160)
        refresh_btn.clicked.connect(self._refresh_preview_frame)
        roi_form.addRow("", refresh_btn)

        right_panel.addWidget(tabs)

        save_btn = QtWidgets.QPushButton("Сохранить канал")
        self._polish_button(save_btn, 200)
        save_btn.clicked.connect(self._save_channel)
        save_btn.setMaximumWidth(220)
        right_panel.addWidget(save_btn, alignment=QtCore.Qt.AlignLeft)
        right_panel.addStretch()

        details_layout.addLayout(right_panel, 2)

        layout.addWidget(self.channel_details_container, 1)

        self._load_general_settings()
        self._reload_channels_list()
        return widget

    def _reload_channels_list(self, target_index: Optional[int] = None) -> None:
        channels = self.settings.get_channels()
        current_row = self.channels_list.currentRow()
        self.channels_list.blockSignals(True)
        self.channels_list.clear()
        for channel in channels:
            self.channels_list.addItem(channel.get("name", "Канал"))
        self.channels_list.blockSignals(False)
        if self.channels_list.count():
            if target_index is None:
                target_index = current_row
            target_row = min(max(target_index if target_index is not None else 0, 0), self.channels_list.count() - 1)
            self.channels_list.setCurrentRow(target_row)
        else:
            self._set_channel_settings_visible(False)

    def _set_channel_settings_visible(self, visible: bool) -> None:
        self.channel_details_container.setVisible(visible)
        self.channel_details_container.setEnabled(visible)
        if not visible:
            self._clear_channel_form()

    def _clear_channel_form(self) -> None:
        self.channel_name_input.clear()
        self.channel_source_input.clear()
        self.best_shots_input.setValue(self.settings.get_best_shots())
        self.cooldown_input.setValue(self.settings.get_cooldown_seconds())
        self.min_conf_input.setValue(self.settings.get_min_confidence())
        self.detection_mode_input.setCurrentIndex(
            max(0, self.detection_mode_input.findData("motion"))
        )
        self.detector_stride_input.setValue(2)
        self.motion_threshold_input.setValue(0.01)
        self.motion_stride_input.setValue(1)
        self.motion_activation_frames_input.setValue(3)
        self.motion_release_frames_input.setValue(6)
        self.debug_detection_checkbox.setChecked(False)
        self.debug_ocr_checkbox.setChecked(False)
        size_defaults = self.settings.get_plate_size_defaults()
        min_size = size_defaults.get("min_plate_size", {})
        max_size = size_defaults.get("max_plate_size", {})
        self.min_plate_width_input.setValue(int(min_size.get("width", 0)))
        self.min_plate_height_input.setValue(int(min_size.get("height", 0)))
        self.max_plate_width_input.setValue(int(max_size.get("width", 0)))
        self.max_plate_height_input.setValue(int(max_size.get("height", 0)))
        self.plate_size_hint.setText(
            "Перетаскивайте прямоугольники мин/макс на превью слева, значения сохраняются автоматически"
        )
        self.preview.set_plate_sizes(
            self.min_plate_width_input.value(),
            self.min_plate_height_input.value(),
            self.max_plate_width_input.value(),
            self.max_plate_height_input.value(),
        )
        default_roi = self._default_roi_region()
        self.preview.setPixmap(None)
        self.preview.set_roi(default_roi)
        self._sync_roi_table(default_roi)

    def _load_general_settings(self) -> None:
        reconnect = self.settings.get_reconnect()
        signal_loss = reconnect.get("signal_loss", {})
        periodic = reconnect.get("periodic", {})
        self.db_dir_input.setText(self.settings.get_db_dir())
        self.screenshot_dir_input.setText(self.settings.get_screenshot_dir())
        plate_settings = self.settings.get_plate_settings()
        self.country_config_dir_input.setText(plate_settings.get("config_dir", "config/countries"))
        self._reload_country_templates(plate_settings.get("enabled_countries", []))

        self.reconnect_on_loss_checkbox.setChecked(bool(signal_loss.get("enabled", True)))
        self.frame_timeout_input.setValue(int(signal_loss.get("frame_timeout_seconds", 5)))
        self.retry_interval_input.setValue(int(signal_loss.get("retry_interval_seconds", 5)))

        self.periodic_reconnect_checkbox.setChecked(bool(periodic.get("enabled", False)))
        self.periodic_interval_input.setValue(int(periodic.get("interval_minutes", 60)))

        time_settings = self.settings.get_time_settings()
        tz_value = time_settings.get("timezone") or "UTC+00:00"
        offset_label = tz_value
        parsed_offset = self._parse_utc_offset_minutes(tz_value)
        if parsed_offset is None:
            try:
                qtz = QtCore.QTimeZone(tz_value.encode())
                parsed_offset = qtz.offsetFromUtc(QtCore.QDateTime.currentDateTime()) // 60
            except Exception:
                parsed_offset = 0
        if parsed_offset is not None:
            offset_label = self._offset_label(parsed_offset)
        index = self.timezone_combo.findData(offset_label)
        if index < 0:
            self.timezone_combo.addItem(offset_label, offset_label)
            index = self.timezone_combo.findData(offset_label)
        if index >= 0:
            self.timezone_combo.setCurrentIndex(index)
        offset_minutes = int(time_settings.get("offset_minutes", 0) or 0)
        adjusted_dt = QtCore.QDateTime.currentDateTime().addSecs(offset_minutes * 60)
        self.time_correction_input.setDateTime(adjusted_dt)

    def _sync_time_now(self) -> None:
        self.time_correction_input.setDateTime(QtCore.QDateTime.currentDateTime())

    def _choose_screenshot_dir(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Выбор папки для скриншотов")
        if directory:
            self.screenshot_dir_input.setText(directory)

    def _choose_db_dir(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Выбор папки базы данных")
        if directory:
            self.db_dir_input.setText(directory)

    def _choose_country_dir(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Выбор каталога шаблонов номеров")
        if directory:
            self.country_config_dir_input.setText(directory)
            self._reload_country_templates()

    def _load_country_names(self, config_dir: Optional[str] = None) -> Dict[str, str]:
        config_path = config_dir or self.settings.get_plate_settings().get("config_dir", "config/countries")
        loader = CountryConfigLoader(config_path)
        loader.ensure_dir()
        names: Dict[str, str] = {}
        for cfg in loader.available_configs():
            code = (cfg.get("code") or "").upper()
            name = cfg.get("name") or code
            if code:
                names[code] = name
        return names

    def _reload_country_templates(self, enabled: Optional[List[str]] = None) -> None:
        plate_settings = self.settings.get_plate_settings()
        config_dir = self.country_config_dir_input.text().strip() or plate_settings.get("config_dir", "config/countries")
        loader = CountryConfigLoader(config_dir)
        loader.ensure_dir()
        available = loader.available_configs()
        enabled_codes = set(enabled or plate_settings.get("enabled_countries", []))
        self.country_display_names = {cfg["code"].upper(): cfg.get("name") or cfg["code"] for cfg in available if cfg.get("code")}

        self.country_templates_list.clear()
        if not available:
            item = QtWidgets.QListWidgetItem("Конфигурации стран не найдены")
            item.setFlags(QtCore.Qt.NoItemFlags)
            self.country_templates_list.addItem(item)
            return

        for cfg in available:
            item = QtWidgets.QListWidgetItem(f"{cfg['code']} — {cfg['name']}")
            item.setData(QtCore.Qt.UserRole, cfg["code"])
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if cfg["code"] in enabled_codes else QtCore.Qt.Unchecked)
            self.country_templates_list.addItem(item)

    def _collect_enabled_countries(self) -> List[str]:
        codes: List[str] = []
        for idx in range(self.country_templates_list.count()):
            item = self.country_templates_list.item(idx)
            if item and item.flags() & QtCore.Qt.ItemIsUserCheckable and item.checkState() == QtCore.Qt.Checked:
                codes.append(str(item.data(QtCore.Qt.UserRole)))
        return codes

    def _save_general_settings(self) -> None:
        reconnect = {
            "signal_loss": {
                "enabled": self.reconnect_on_loss_checkbox.isChecked(),
                "frame_timeout_seconds": int(self.frame_timeout_input.value()),
                "retry_interval_seconds": int(self.retry_interval_input.value()),
            },
            "periodic": {
                "enabled": self.periodic_reconnect_checkbox.isChecked(),
                "interval_minutes": int(self.periodic_interval_input.value()),
            },
        }
        self.settings.save_reconnect(reconnect)
        db_dir = self.db_dir_input.text().strip() or "data/db"
        os.makedirs(db_dir, exist_ok=True)
        self.settings.save_db_dir(db_dir)
        screenshot_dir = self.screenshot_dir_input.text().strip() or "data/screenshots"
        self.settings.save_screenshot_dir(screenshot_dir)
        os.makedirs(screenshot_dir, exist_ok=True)
        tz_name = self.timezone_combo.currentData() or self.timezone_combo.currentText().strip() or "UTC+00:00"
        offset_minutes = int(
            QtCore.QDateTime.currentDateTime().secsTo(self.time_correction_input.dateTime()) / 60
        )
        self.settings.save_time_settings({"timezone": tz_name, "offset_minutes": offset_minutes})
        plate_settings = {
            "config_dir": self.country_config_dir_input.text().strip() or "config/countries",
            "enabled_countries": self._collect_enabled_countries(),
        }
        os.makedirs(plate_settings["config_dir"], exist_ok=True)
        self.settings.save_plate_settings(plate_settings)
        self.db = EventDatabase(self.settings.get_db_path())
        self._refresh_events_table()
        self._start_channels()

    def _load_channel_form(self, index: int) -> None:
        channels = self.settings.get_channels()
        has_channel = 0 <= index < len(channels)
        self._set_channel_settings_visible(has_channel)
        if has_channel:
            channel = channels[index]
            self.channel_name_input.setText(channel.get("name", ""))
            self.channel_source_input.setText(channel.get("source", ""))
            self.best_shots_input.setValue(int(channel.get("best_shots", self.settings.get_best_shots())))
            self.cooldown_input.setValue(int(channel.get("cooldown_seconds", self.settings.get_cooldown_seconds())))
            self.min_conf_input.setValue(float(channel.get("ocr_min_confidence", self.settings.get_min_confidence())))
            self.detection_mode_input.setCurrentIndex(
                max(0, self.detection_mode_input.findData(channel.get("detection_mode", "motion")))
            )
            self.detector_stride_input.setValue(int(channel.get("detector_frame_stride", 2)))
            self.motion_threshold_input.setValue(float(channel.get("motion_threshold", 0.01)))
            self.motion_stride_input.setValue(int(channel.get("motion_frame_stride", 1)))
            self.motion_activation_frames_input.setValue(int(channel.get("motion_activation_frames", 3)))
            self.motion_release_frames_input.setValue(int(channel.get("motion_release_frames", 6)))
            min_size = channel.get("min_plate_size", self.settings.get_plate_size_defaults().get("min_plate_size", {}))
            max_size = channel.get("max_plate_size", self.settings.get_plate_size_defaults().get("max_plate_size", {}))
            self.min_plate_width_input.setValue(int(min_size.get("width", 0)))
            self.min_plate_height_input.setValue(int(min_size.get("height", 0)))
            self.max_plate_width_input.setValue(int(max_size.get("width", 0)))
            self.max_plate_height_input.setValue(int(max_size.get("height", 0)))
            self.plate_size_hint.setText(
                f"Текущие рамки: мин {self.min_plate_width_input.value()}×{self.min_plate_height_input.value()} px, "
                f"макс {self.max_plate_width_input.value()}×{self.max_plate_height_input.value()} px"
            )
            self._sync_plate_rects_from_inputs()

            debug_conf = channel.get("debug", {})
            self.debug_detection_checkbox.setChecked(bool(debug_conf.get("show_detection_boxes", False)))
            self.debug_ocr_checkbox.setChecked(bool(debug_conf.get("show_ocr_text", False)))

            region = channel.get("region") or self._default_roi_region()
            if not region.get("points"):
                region = {"unit": region.get("unit", "px"), "points": self._default_roi_region()["points"]}
            self.preview.set_roi(region)
            self._sync_roi_table(region)
            self._refresh_preview_frame()

    def _add_channel(self) -> None:
        channels = self.settings.get_channels()
        new_id = max([c.get("id", 0) for c in channels] + [0]) + 1
        channels.append(
            {
                "id": new_id,
                "name": f"Канал {new_id}",
                "source": "",
                "best_shots": self.settings.get_best_shots(),
                "cooldown_seconds": self.settings.get_cooldown_seconds(),
                "ocr_min_confidence": self.settings.get_min_confidence(),
                "region": self._default_roi_region(),
                "detection_mode": "motion",
                "detector_frame_stride": 2,
                "motion_threshold": 0.01,
                "motion_frame_stride": 1,
                "motion_activation_frames": 3,
                "motion_release_frames": 6,
                "min_plate_size": self.settings.get_plate_size_defaults().get("min_plate_size"),
                "max_plate_size": self.settings.get_plate_size_defaults().get("max_plate_size"),
                "debug": {"show_detection_boxes": False, "show_ocr_text": False},
            }
        )
        self.settings.save_channels(channels)
        self._reload_channels_list(len(channels) - 1)
        self._draw_grid()
        self._start_channels()

    def _remove_channel(self) -> None:
        index = self.channels_list.currentRow()
        channels = self.settings.get_channels()
        if 0 <= index < len(channels):
            channels.pop(index)
            self.settings.save_channels(channels)
            self._reload_channels_list(index)
            self._draw_grid()
            self._start_channels()

    def _save_channel(self) -> None:
        index = self.channels_list.currentRow()
        channels = self.settings.get_channels()
        if 0 <= index < len(channels):
            channels[index]["name"] = self.channel_name_input.text()
            channels[index]["source"] = self.channel_source_input.text()
            channels[index]["best_shots"] = int(self.best_shots_input.value())
            channels[index]["cooldown_seconds"] = int(self.cooldown_input.value())
            channels[index]["ocr_min_confidence"] = float(self.min_conf_input.value())
            channels[index]["detection_mode"] = self.detection_mode_input.currentData()
            channels[index]["detector_frame_stride"] = int(self.detector_stride_input.value())
            channels[index]["motion_threshold"] = float(self.motion_threshold_input.value())
            channels[index]["motion_frame_stride"] = int(self.motion_stride_input.value())
            channels[index]["motion_activation_frames"] = int(self.motion_activation_frames_input.value())
            channels[index]["motion_release_frames"] = int(self.motion_release_frames_input.value())
            channels[index]["min_plate_size"] = {
                "width": int(self.min_plate_width_input.value()),
                "height": int(self.min_plate_height_input.value()),
            }
            channels[index]["max_plate_size"] = {
                "width": int(self.max_plate_width_input.value()),
                "height": int(self.max_plate_height_input.value()),
            }
            channels[index]["region"] = {"unit": "px", "points": self._collect_roi_points_from_table()}
            channels[index]["debug"] = {
                "show_detection_boxes": self.debug_detection_checkbox.isChecked(),
                "show_ocr_text": self.debug_ocr_checkbox.isChecked(),
            }
            self.settings.save_channels(channels)
            self._reload_channels_list(index)
            self._draw_grid()
            self._start_channels()

    def _sync_roi_table(self, roi: Dict[str, Any]) -> None:
        points = roi.get("points") or []
        self.roi_points_table.blockSignals(True)
        self.roi_points_table.setRowCount(len(points))
        for row, point in enumerate(points):
            x_item = QtWidgets.QTableWidgetItem(str(int(point.get("x", 0))))
            y_item = QtWidgets.QTableWidgetItem(str(int(point.get("y", 0))))
            self.roi_points_table.setItem(row, 0, x_item)
            self.roi_points_table.setItem(row, 1, y_item)
        self.roi_points_table.blockSignals(False)

    def _collect_roi_points_from_table(self) -> List[Dict[str, int]]:
        points: List[Dict[str, int]] = []
        for row in range(self.roi_points_table.rowCount()):
            x_item = self.roi_points_table.item(row, 0)
            y_item = self.roi_points_table.item(row, 1)
            try:
                x_val = int(x_item.text()) if x_item else 0
                y_val = int(y_item.text()) if y_item else 0
            except ValueError:
                continue
            points.append({"x": x_val, "y": y_val})
        return points

    def _on_roi_drawn(self, roi: Dict[str, Any]) -> None:
        self._sync_roi_table(roi)

    def _on_roi_table_changed(self) -> None:
        roi = {"unit": "px", "points": self._collect_roi_points_from_table()}
        self.preview.set_roi(roi)

    def _add_roi_point(self) -> None:
        img_size = self.preview.image_size()
        x = img_size.width() // 2 if img_size else 0
        y = img_size.height() // 2 if img_size else 0
        row = self.roi_points_table.rowCount()
        self.roi_points_table.blockSignals(True)
        self.roi_points_table.insertRow(row)
        self.roi_points_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(x)))
        self.roi_points_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(y)))
        self.roi_points_table.blockSignals(False)
        self._on_roi_table_changed()

    def _remove_roi_point(self) -> None:
        rows = sorted({index.row() for index in self.roi_points_table.selectedIndexes()}, reverse=True)
        if not rows and self.roi_points_table.rowCount():
            rows = [self.roi_points_table.rowCount() - 1]
        self.roi_points_table.blockSignals(True)
        for row in rows:
            self.roi_points_table.removeRow(row)
        self.roi_points_table.blockSignals(False)
        self._on_roi_table_changed()

    def _clear_roi_points(self) -> None:
        default_roi = self._default_roi_region()
        self._sync_roi_table(default_roi)
        self.preview.set_roi(default_roi)

    def _sync_plate_rects_from_inputs(self) -> None:
        self.preview.set_plate_sizes(
            self.min_plate_width_input.value(),
            self.min_plate_height_input.value(),
            self.max_plate_width_input.value(),
            self.max_plate_height_input.value(),
        )

    def _on_plate_size_selected(self, target: str, width: int, height: int) -> None:
        if target == "min":
            self.min_plate_width_input.setValue(width)
            self.min_plate_height_input.setValue(height)
            label = "мин"
        else:
            self.max_plate_width_input.setValue(width)
            self.max_plate_height_input.setValue(height)
            label = "макс"
        self.plate_size_hint.setText(
            f"Прямоугольник {label}: {width}×{height} px"
        )

    def _cancel_preview_worker(self) -> None:
        if self._preview_worker:
            self._preview_worker.requestInterruption()
            self._preview_worker.wait(1000)
            self._preview_worker.deleteLater()
            self._preview_worker = None

    def _refresh_preview_frame(self) -> None:
        index = self.channels_list.currentRow()
        channels = self.settings.get_channels()
        if not (0 <= index < len(channels)):
            return
        source = str(channels[index].get("source", ""))
        if not source:
            self.preview.setPixmap(None)
            return
        self._preview_request_id += 1
        request_id = self._preview_request_id
        self._cancel_preview_worker()
        self.preview.setPixmap(None)

        worker = PreviewLoader(source)
        self._preview_worker = worker

        def handle_frame(pixmap: QtGui.QPixmap, rid: int = request_id) -> None:
            if rid != self._preview_request_id:
                return
            self.preview.setPixmap(pixmap)
            self._preview_worker = None

        def handle_failure(rid: int = request_id) -> None:
            if rid != self._preview_request_id:
                return
            self.preview.setPixmap(None)
            self._preview_worker = None

        worker.frameReady.connect(handle_frame)
        worker.failed.connect(handle_failure)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    # ------------------ Жизненный цикл ------------------
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self._cancel_preview_worker()
        self._stop_workers()
        event.accept()
