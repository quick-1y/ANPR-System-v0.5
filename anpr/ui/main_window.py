#!/usr/bin/env python3
# /anpr/ui/main_window.py
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import psutil
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets
from zoneinfo import ZoneInfo

from anpr.postprocessing.country_config import CountryConfigLoader
from anpr.workers.channel_worker import ChannelWorker
from anpr.infrastructure.logging_manager import get_logger
from anpr.infrastructure.settings_manager import SettingsManager
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

    def __init__(self) -> None:
        super().__init__("Нет кадра")
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(400, 260)
        self.setStyleSheet(
            "background-color: #111; color: #888; border: 1px solid #444; padding: 6px;"
        )
        self._roi = {"x": 0, "y": 0, "width": 100, "height": 100}
        self._pixmap: Optional[QtGui.QPixmap] = None
        self._rubber_band = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)
        self._origin: Optional[QtCore.QPoint] = None

    def set_roi(self, roi: Dict[str, int]) -> None:
        self._roi = {
            "x": int(roi.get("x", 0)),
            "y": int(roi.get("y", 0)),
            "width": int(roi.get("width", 100)),
            "height": int(roi.get("height", 100)),
        }
        self._roi["width"] = min(self._roi["width"], max(1, 100 - self._roi["x"]))
        self._roi["height"] = min(self._roi["height"], max(1, 100 - self._roi["y"]))
        self.update()

    def setPixmap(self, pixmap: Optional[QtGui.QPixmap]) -> None:  # noqa: N802
        self._pixmap = pixmap
        if pixmap is None:
            super().setPixmap(QtGui.QPixmap())
            self.setText("Нет кадра")
            return
        scaled = self._scaled_pixmap(self.size())
        super().setPixmap(scaled)
        self.setText("")

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._pixmap:
            super().setPixmap(self._scaled_pixmap(event.size()))

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

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        super().paintEvent(event)
        geom = self._image_geometry()
        if geom is None:
            return
        offset, size = geom
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        roi_rect = QtCore.QRect(
            offset.x() + int(size.width() * self._roi["x"] / 100),
            offset.y() + int(size.height() * self._roi["y"] / 100),
            int(size.width() * self._roi["width"] / 100),
            int(size.height() * self._roi["height"] / 100),
        )
        pen = QtGui.QPen(QtGui.QColor(0, 200, 0))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(QtGui.QColor(0, 200, 0, 40))
        painter.drawRect(roi_rect)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        geom = self._image_geometry()
        if geom is None:
            return
        offset, size = geom
        area_rect = QtCore.QRect(offset, size)
        if not area_rect.contains(event.pos()):
            return
        self._origin = event.pos()
        self._rubber_band.setGeometry(QtCore.QRect(self._origin, QtCore.QSize()))
        self._rubber_band.show()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if self._origin is None:
            return
        rect = QtCore.QRect(self._origin, event.pos()).normalized()
        self._rubber_band.setGeometry(rect)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if self._origin is None:
            return
        geom = self._image_geometry()
        self._rubber_band.hide()
        if geom is None:
            self._origin = None
            return
        offset, size = geom
        selection = self._rubber_band.geometry().intersected(QtCore.QRect(offset, size))
        if selection.isValid() and selection.width() > 5 and selection.height() > 5:
            x_pct = max(0, min(100, int((selection.left() - offset.x()) * 100 / size.width())))
            y_pct = max(0, min(100, int((selection.top() - offset.y()) * 100 / size.height())))
            w_pct = max(1, min(100 - x_pct, int(selection.width() * 100 / size.width())))
            h_pct = max(1, min(100 - y_pct, int(selection.height() * 100 / size.height())))
            self._roi = {"x": x_pct, "y": y_pct, "width": w_pct, "height": h_pct}
            self.roi_changed.emit(self._roi)
        self._origin = None
        self.update()


class EventDetailView(QtWidgets.QWidget):
    """Отображение выбранного события: метаданные, кадр и область номера."""

    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)

        self.frame_preview = self._build_preview("Кадр распознавания", min_height=320, keep_aspect=True)
        layout.addWidget(self.frame_preview, stretch=3)

        bottom_row = QtWidgets.QHBoxLayout()
        self.plate_preview = self._build_preview("Кадр номера", min_size=QtCore.QSize(200, 140), keep_aspect=True)
        bottom_row.addWidget(self.plate_preview, 1)

        meta_group = QtWidgets.QGroupBox("Данные распознавания")
        meta_group.setStyleSheet(
            "QGroupBox { background-color: #000; color: white; border: 1px solid #2e2e2e; padding: 6px; }"
            "QLabel { color: white; }"
        )
        meta_group.setMinimumWidth(220)
        meta_layout = QtWidgets.QFormLayout(meta_group)
        self.time_label = QtWidgets.QLabel("—")
        self.channel_label = QtWidgets.QLabel("—")
        self.country_label = QtWidgets.QLabel("—")
        self.plate_label = QtWidgets.QLabel("—")
        self.conf_label = QtWidgets.QLabel("—")
        meta_layout.addRow("Дата/Время:", self.time_label)
        meta_layout.addRow("Канал:", self.channel_label)
        meta_layout.addRow("Страна:", self.country_label)
        meta_layout.addRow("Гос. номер:", self.plate_label)
        meta_layout.addRow("Уверенность:", self.conf_label)
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
        label.setScaledContents(False if keep_aspect else True)
        wrapper.addWidget(label)
        group.display_label = label  # type: ignore[attr-defined]
        return group

    def clear(self) -> None:
        self.time_label.setText("—")
        self.channel_label.setText("—")
        self.country_label.setText("—")
        self.plate_label.setText("—")
        self.conf_label.setText("—")
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

        self._set_image(self.frame_preview, frame_image, keep_aspect=True)
        self._set_image(self.plate_preview, plate_image, keep_aspect=True)

    def _set_image(
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
        pixmap = QtGui.QPixmap.fromImage(image)
        if keep_aspect:
            pixmap = pixmap.scaled(label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        label.setPixmap(pixmap)


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
    )
    LIST_STYLE = (
        "QListWidget { background-color: #0b0c10; color: #cbd5e1; border: 1px solid #1f2937; border-radius: 10px; padding: 6px; }"
        "QListWidget::item:selected { background-color: rgba(34,211,238,0.16); color: #22d3ee; border-radius: 6px; }"
        "QListWidget::item { padding: 8px 10px; margin: 2px 0; }"
    )

    def __init__(self, settings: Optional[SettingsManager] = None) -> None:
        super().__init__()
        self.setWindowTitle("ANPR Desktop")
        self.resize(1280, 800)

        self.settings = settings or SettingsManager()
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
        self.flag_cache: Dict[str, Optional[QtGui.QIcon]] = {}
        self.flag_dir = Path(__file__).resolve().parents[2] / "images" / "flags"
        self.country_display_names = self._load_country_names()
        self._pending_channels: Optional[List[Dict[str, Any]]] = None
        self._channel_save_timer = QtCore.QTimer(self)
        self._channel_save_timer.setSingleShot(True)
        self._channel_save_timer.timeout.connect(self._flush_pending_channels)
        self._drag_counter = 0
        self._skip_frame_updates = False

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
        self.tabs.addTab(self.search_tab, "Поиск")
        self.tabs.addTab(self.settings_tab, "Настройки")

        self.setCentralWidget(self.tabs)
        self.setStyleSheet(
            "QMainWindow { background-color: #0b0c10; }"
            "QStatusBar { background-color: #0b0c10; color: #e5e7eb; padding: 4px; border-top: 1px solid #1f2937; }"
        )
        self._build_status_bar()
        self._start_system_monitoring()
        self._refresh_events_table()
        self._start_channels()

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
        self.grid_combo.setStyleSheet(
            "QComboBox { background-color: #0b0c10; color: #e5e7eb; border: 1px solid #1f2937; border-radius: 10px; padding: 8px 12px; min-width: 140px; }"
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
        self.events_table = QtWidgets.QTableWidget(0, 4)
        self.events_table.setHorizontalHeaderLabels(["Дата/Время", "Гос. номер", "Страна", "Канал"])
        self.events_table.setStyleSheet(self.TABLE_STYLE)
        header = self.events_table.horizontalHeader()
        header.setMinimumSectionSize(70)
        header.setStretchLastSection(False)
        for index in range(4):
            header.setSectionResizeMode(index, QtWidgets.QHeaderView.Interactive)
        self.events_table.setColumnWidth(0, 220)
        self.events_table.setColumnWidth(1, 120)
        self.events_table.setColumnWidth(2, 90)
        self.events_table.setColumnWidth(3, 120)
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
        layout.setSpacing(12)

        widget.setStyleSheet(
            "QLabel { color: #f0f0f0; }"
            "QLineEdit, QDateTimeEdit { background-color: #111; color: #f0f0f0; border: 1px solid #333; padding: 4px; }"
            "QPushButton { background-color: #00ffff; color: #000; border-radius: 4px; padding: 6px 12px; font-weight: 600; }"
            "QPushButton:hover { background-color: #4dfefe; }"
        )

        filters_group = QtWidgets.QGroupBox("Фильтры поиска")
        filters_group.setStyleSheet(self.GROUP_BOX_STYLE)
        form = QtWidgets.QFormLayout(filters_group)
        self.search_plate = QtWidgets.QLineEdit()
        self.search_from = QtWidgets.QDateTimeEdit()
        self._prepare_optional_datetime(self.search_from)
        self.search_to = QtWidgets.QDateTimeEdit()
        self._prepare_optional_datetime(self.search_to)

        form.addRow("Номер:", self.search_plate)
        form.addRow("Дата с:", self.search_from)
        form.addRow("Дата по:", self.search_to)
        layout.addWidget(filters_group)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch()
        search_btn = QtWidgets.QPushButton("Искать")
        search_btn.clicked.connect(self._run_plate_search)
        button_row.addWidget(search_btn)
        layout.addLayout(button_row)

        self.search_table = QtWidgets.QTableWidget(0, 6)
        self.search_table.setHorizontalHeaderLabels(
            ["Дата/Время", "Канал", "Страна", "Номер", "Уверенность", "Источник"]
        )
        self.search_table.horizontalHeader().setStretchLastSection(True)
        self.search_table.setStyleSheet(self.TABLE_STYLE)
        self.search_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.search_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.search_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.search_table.verticalHeader().setVisible(False)
        layout.addWidget(self.search_table)

        return widget

    def _run_plate_search(self) -> None:
        start = self._get_datetime_value(self.search_from)
        end = self._get_datetime_value(self.search_to)
        plate_fragment = self.search_plate.text()
        rows = self.db.search_by_plate(plate_fragment, start=start or None, end=end or None)
        self.search_table.setRowCount(0)
        target_zone = self._get_target_zone()
        offset_minutes = self.settings.get_time_offset_minutes()
        for row_data in rows:
            row_index = self.search_table.rowCount()
            self.search_table.insertRow(row_index)
            formatted_time = self._format_timestamp(row_data["timestamp"], target_zone, offset_minutes)
            self.search_table.setItem(row_index, 0, QtWidgets.QTableWidgetItem(formatted_time))
            self.search_table.setItem(row_index, 1, QtWidgets.QTableWidgetItem(row_data["channel"]))
            country_item = QtWidgets.QTableWidgetItem(row_data["country"] or "")
            country_icon = self._get_flag_icon(row_data["country"])
            if country_icon:
                country_item.setIcon(country_icon)
            country_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.search_table.setItem(row_index, 2, country_item)
            self.search_table.setItem(row_index, 3, QtWidgets.QTableWidgetItem(row_data["plate"]))
            self.search_table.setItem(
                row_index, 4, QtWidgets.QTableWidgetItem(f"{row_data['confidence'] or 0:.2f}")
            )
            self.search_table.setItem(row_index, 5, QtWidgets.QTableWidgetItem(row_data["source"]))

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
        widget.setStyleSheet(
            "QLabel { color: #cbd5e1; font-size: 13px; }"
            "QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QDateTimeEdit { background-color: #0b0c10; color: #f8fafc; border: 1px solid #1f2937; border-radius: 8px; padding: 8px; }"
            "QPushButton { background-color: #22d3ee; color: #0b0c10; border-radius: 8px; padding: 8px 14px; font-weight: 700; letter-spacing: 0.2px; }"
            "QPushButton:hover { background-color: #4ddcf3; }"
            "QCheckBox { color: #e5e7eb; font-size: 13px; }"
            f"QWidget {{ background-color: {self.SURFACE_COLOR}; }}"
        )

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

        center_panel = QtWidgets.QVBoxLayout()
        self.preview = ROIEditor()
        self.preview.roi_changed.connect(self._on_roi_drawn)
        center_panel.addWidget(self.preview)
        layout.addLayout(center_panel, 2)

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
        roi_grid = QtWidgets.QGridLayout()
        roi_grid.setHorizontalSpacing(12)
        roi_grid.setVerticalSpacing(10)
        roi_grid.setContentsMargins(0, 0, 0, 0)
        self.roi_x_input = QtWidgets.QSpinBox()
        self.roi_x_input.setRange(0, 100)
        self.roi_x_input.setMaximumWidth(self.COMPACT_FIELD_WIDTH)
        self.roi_x_input.setMinimumWidth(120)
        self.roi_y_input = QtWidgets.QSpinBox()
        self.roi_y_input.setRange(0, 100)
        self.roi_y_input.setMaximumWidth(self.COMPACT_FIELD_WIDTH)
        self.roi_y_input.setMinimumWidth(120)
        self.roi_w_input = QtWidgets.QSpinBox()
        self.roi_w_input.setRange(1, 100)
        self.roi_w_input.setMaximumWidth(self.COMPACT_FIELD_WIDTH)
        self.roi_w_input.setMinimumWidth(120)
        self.roi_h_input = QtWidgets.QSpinBox()
        self.roi_h_input.setRange(1, 100)
        self.roi_h_input.setMaximumWidth(self.COMPACT_FIELD_WIDTH)
        self.roi_h_input.setMinimumWidth(120)

        for spin in (self.roi_x_input, self.roi_y_input, self.roi_w_input, self.roi_h_input):
            spin.valueChanged.connect(self._on_roi_inputs_changed)

        roi_grid.addWidget(QtWidgets.QLabel("X (%):"), 0, 0)
        roi_grid.addWidget(self.roi_x_input, 0, 1)
        roi_grid.addWidget(QtWidgets.QLabel("Y (%):"), 1, 0)
        roi_grid.addWidget(self.roi_y_input, 1, 1)
        roi_grid.addWidget(QtWidgets.QLabel("Ширина (%):"), 2, 0)
        roi_grid.addWidget(self.roi_w_input, 2, 1)
        roi_grid.addWidget(QtWidgets.QLabel("Высота (%):"), 3, 0)
        roi_grid.addWidget(self.roi_h_input, 3, 1)

        roi_row_container = QtWidgets.QWidget()
        roi_row_container.setLayout(roi_grid)
        roi_form.addRow("", roi_row_container)

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

        layout.addLayout(right_panel, 2)

        self._load_general_settings()
        self._reload_channels_list()
        return widget

    def _reload_channels_list(self) -> None:
        channels = self.settings.get_channels()
        current_row = self.channels_list.currentRow()
        self.channels_list.blockSignals(True)
        self.channels_list.clear()
        for channel in channels:
            self.channels_list.addItem(channel.get("name", "Канал"))
        self.channels_list.blockSignals(False)
        if self.channels_list.count():
            target_row = min(current_row if current_row >= 0 else 0, self.channels_list.count() - 1)
            self.channels_list.setCurrentRow(target_row)

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
        if 0 <= index < len(channels):
            channel = channels[index]
            self.channel_name_input.setText(channel.get("name", ""))
            self.channel_source_input.setText(channel.get("source", ""))
            self.best_shots_input.setValue(int(channel.get("best_shots", self.settings.get_best_shots())))
            self.cooldown_input.setValue(int(channel.get("cooldown_seconds", self.settings.get_cooldown_seconds())))
            self.min_conf_input.setValue(float(channel.get("ocr_min_confidence", self.settings.get_min_confidence())))
            self.detection_mode_input.setCurrentIndex(
                max(0, self.detection_mode_input.findData(channel.get("detection_mode", "continuous")))
            )
            self.detector_stride_input.setValue(int(channel.get("detector_frame_stride", 2)))
            self.motion_threshold_input.setValue(float(channel.get("motion_threshold", 0.01)))
            self.motion_stride_input.setValue(int(channel.get("motion_frame_stride", 1)))
            self.motion_activation_frames_input.setValue(int(channel.get("motion_activation_frames", 3)))
            self.motion_release_frames_input.setValue(int(channel.get("motion_release_frames", 6)))

            region = channel.get("region") or {"x": 0, "y": 0, "width": 100, "height": 100}
            self.roi_x_input.setValue(int(region.get("x", 0)))
            self.roi_y_input.setValue(int(region.get("y", 0)))
            self.roi_w_input.setValue(int(region.get("width", 100)))
            self.roi_h_input.setValue(int(region.get("height", 100)))
            self.preview.set_roi(
                {
                    "x": int(region.get("x", 0)),
                    "y": int(region.get("y", 0)),
                    "width": int(region.get("width", 100)),
                    "height": int(region.get("height", 100)),
                }
            )
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
                "region": {"x": 0, "y": 0, "width": 100, "height": 100},
                "detection_mode": "continuous",
                "detector_frame_stride": 2,
                "motion_threshold": 0.01,
                "motion_frame_stride": 1,
                "motion_activation_frames": 3,
                "motion_release_frames": 6,
            }
        )
        self.settings.save_channels(channels)
        self._reload_channels_list()
        self._draw_grid()
        self._start_channels()

    def _remove_channel(self) -> None:
        index = self.channels_list.currentRow()
        channels = self.settings.get_channels()
        if 0 <= index < len(channels):
            channels.pop(index)
            self.settings.save_channels(channels)
            self._reload_channels_list()
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

            region = {
                "x": int(self.roi_x_input.value()),
                "y": int(self.roi_y_input.value()),
                "width": int(self.roi_w_input.value()),
                "height": int(self.roi_h_input.value()),
            }
            region["width"] = min(region["width"], max(1, 100 - region["x"]))
            region["height"] = min(region["height"], max(1, 100 - region["y"]))
            channels[index]["region"] = region
            self.settings.save_channels(channels)
            self._reload_channels_list()
            self._draw_grid()
            self._start_channels()

    def _on_roi_drawn(self, roi: Dict[str, int]) -> None:
        self.roi_x_input.blockSignals(True)
        self.roi_y_input.blockSignals(True)
        self.roi_w_input.blockSignals(True)
        self.roi_h_input.blockSignals(True)
        self.roi_x_input.setValue(roi["x"])
        self.roi_y_input.setValue(roi["y"])
        self.roi_w_input.setValue(roi["width"])
        self.roi_h_input.setValue(roi["height"])
        self.roi_x_input.blockSignals(False)
        self.roi_y_input.blockSignals(False)
        self.roi_w_input.blockSignals(False)
        self.roi_h_input.blockSignals(False)

    def _on_roi_inputs_changed(self) -> None:
        roi = {
            "x": int(self.roi_x_input.value()),
            "y": int(self.roi_y_input.value()),
            "width": int(self.roi_w_input.value()),
            "height": int(self.roi_h_input.value()),
        }
        roi["width"] = min(roi["width"], max(1, 100 - roi["x"]))
        roi["height"] = min(roi["height"], max(1, 100 - roi["y"]))
        self.preview.set_roi(roi)

    def _refresh_preview_frame(self) -> None:
        index = self.channels_list.currentRow()
        channels = self.settings.get_channels()
        if not (0 <= index < len(channels)):
            return
        source = str(channels[index].get("source", ""))
        if not source:
            self.preview.setPixmap(None)
            return
        capture = cv2.VideoCapture(int(source) if source.isnumeric() else source)
        ret, frame = capture.read()
        capture.release()
        if not ret or frame is None:
            self.preview.setPixmap(None)
            return
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb_frame.shape
        bytes_per_line = 3 * width
        q_image = QtGui.QImage(
            rgb_frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888
        ).copy()
        self.preview.setPixmap(QtGui.QPixmap.fromImage(q_image))

    # ------------------ Жизненный цикл ------------------
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self._stop_workers()
        event.accept()
