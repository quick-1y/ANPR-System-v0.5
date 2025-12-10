#!/usr/bin/env python3
# /anpr/ui/main_window.py
import os
from datetime import datetime

import cv2
import psutil
from typing import Dict, List, Optional, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

from anpr.workers.channel_worker import ChannelWorker
from logging_manager import get_logger
from settings_manager import SettingsManager
from storage import EventDatabase

logger = get_logger(__name__)


class ChannelView(QtWidgets.QWidget):
    """Отображает поток канала с подсказками и индикатором движения."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

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

    def set_pixmap(self, pixmap: QtGui.QPixmap) -> None:
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
        self.plate_label = QtWidgets.QLabel("—")
        self.conf_label = QtWidgets.QLabel("—")
        meta_layout.addRow("Дата/Время:", self.time_label)
        meta_layout.addRow("Канал:", self.channel_label)
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
        label.setStyleSheet("background-color: #111; color: #888; border: 1px solid #444;")
        label.setScaledContents(False if keep_aspect else True)
        wrapper.addWidget(label)
        group.display_label = label  # type: ignore[attr-defined]
        return group

    def clear(self) -> None:
        self.time_label.setText("—")
        self.channel_label.setText("—")
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
    GROUP_BOX_STYLE = (
        "QGroupBox { background-color: #2b2b28; color: #f0f0f0; border: 1px solid #383531; padding: 8px; margin-top: 6px; }"
        "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
        "QLabel { color: #f0f0f0; }"
        "QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QDateTimeEdit { background-color: #111; color: #f0f0f0; border: 1px solid #333; padding: 4px; }"
        "QPushButton { background-color: #00ffff; color: #000; border-radius: 4px; padding: 6px 12px; font-weight: 600; }"
        "QPushButton:hover { background-color: #4dfefe; }"
        "QCheckBox { color: #e0e0e0; }"
    )
    TABLE_STYLE = (
        "QHeaderView::section { background-color: rgb(23,25,29); color: white; padding: 6px; }"
        "QTableWidget { background-color: #000; color: lightgray; gridline-color: #333; }"
        "QTableWidget::item { border-bottom: 1px solid #333; }"
        "QTableWidget::item:selected { background-color: #00ffff; color: #000; }"
    )
    LIST_STYLE = "QListWidget { background-color: #111; color: #e0e0e0; border: 1px solid #333; }"

    def __init__(self, settings: Optional[SettingsManager] = None) -> None:
        super().__init__()
        self.setWindowTitle("ANPR Desktop")
        self.resize(1280, 800)

        self.settings = settings or SettingsManager()
        self.db = EventDatabase(self.settings.get_db_path())

        self.channel_workers: List[ChannelWorker] = []
        self.channel_labels: Dict[str, ChannelView] = {}
        self.event_images: Dict[int, Tuple[Optional[QtGui.QImage], Optional[QtGui.QImage]]] = {}
        self.event_cache: Dict[int, Dict] = {}

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setStyleSheet(
            "QTabBar::tab { background: rgb(23,25,29); color: grey; padding: 8px 16px; border: 1px solid #111; }"
            "QTabBar::tab:selected { background: rgb(23,25,29); color: #00ffff; border-bottom: 2px solid #00ffff; }"
            "QTabWidget::pane { border: 1px solid #111; }"
        )
        self.observation_tab = self._build_observation_tab()
        self.search_tab = self._build_search_tab()
        self.settings_tab = self._build_settings_tab()

        self.tabs.addTab(self.observation_tab, "Наблюдение")
        self.tabs.addTab(self.search_tab, "Поиск")
        self.tabs.addTab(self.settings_tab, "Настройки")

        self.setCentralWidget(self.tabs)
        self.setStyleSheet("background-color: #49423d;")
        self._build_status_bar()
        self._start_system_monitoring()
        self._refresh_events_table()
        self._start_channels()

    def _build_status_bar(self) -> None:
        status = self.statusBar()
        status.setStyleSheet("background-color: rgb(23,25,29); color: white; padding: 3px;")
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
        controls.addWidget(QtWidgets.QLabel("Сетка:"))
        self.grid_selector = QtWidgets.QComboBox()
        self.grid_selector.addItems(self.GRID_VARIANTS)
        self.grid_selector.setCurrentText(self.settings.get_grid())
        self.grid_selector.currentTextChanged.connect(self._on_grid_changed)
        controls.addWidget(self.grid_selector)
        controls.addStretch()
        left_column.addLayout(controls)

        self.grid_widget = QtWidgets.QWidget()
        self.grid_layout = QtWidgets.QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(6)
        left_column.addWidget(self.grid_widget, stretch=4)

        layout.addLayout(left_column, stretch=3)

        right_column = QtWidgets.QVBoxLayout()
        details_group = QtWidgets.QGroupBox("Информация о событии")
        details_group.setStyleSheet(
            "QGroupBox { background-color: #000; color: white; border: 1px solid #2e2e2e; padding: 6px; }"
        )
        details_layout = QtWidgets.QVBoxLayout(details_group)
        self.event_detail = EventDetailView()
        details_layout.addWidget(self.event_detail)
        right_column.addWidget(details_group, stretch=3)

        events_group = QtWidgets.QGroupBox("События")
        events_group.setStyleSheet(
            "QGroupBox { background-color: rgb(40,40,40); color: white; border: 1px solid #2e2e2e; padding: 6px; }"
        )
        events_layout = QtWidgets.QVBoxLayout(events_group)
        self.events_table = QtWidgets.QTableWidget(0, 3)
        self.events_table.setHorizontalHeaderLabels(["Дата/Время", "Гос. номер", "Канал"])
        self.events_table.setStyleSheet(self.TABLE_STYLE)
        self.events_table.horizontalHeader().setStretchLastSection(True)
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
    def _prepare_optional_datetime(widget: QtWidgets.QDateTimeEdit) -> None:
        widget.setCalendarPopup(True)
        widget.setDisplayFormat("dd.MM.yyyy HH:mm:ss")
        min_dt = QtCore.QDateTime.fromSecsSinceEpoch(0)
        widget.setMinimumDateTime(min_dt)
        widget.setSpecialValueText("Не выбрано")
        widget.setDateTime(min_dt)

    @staticmethod
    def _get_datetime_value(widget: QtWidgets.QDateTimeEdit) -> Optional[str]:
        if widget.dateTime() == widget.minimumDateTime():
            return None
        return widget.dateTime().toString(QtCore.Qt.ISODate)

    @staticmethod
    def _format_timestamp(value: str) -> str:
        if not value:
            return "—"
        cleaned = value.replace("Z", "+00:00") if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(cleaned)
            return parsed.strftime("%d.%m.%Y %H:%M:%S")
        except ValueError:
            return value

    def _draw_grid(self) -> None:
        for i in reversed(range(self.grid_layout.count())):
            item = self.grid_layout.takeAt(i)
            widget = item.widget()
            if widget:
                widget.setParent(None)

        self.channel_labels.clear()
        channels = self.settings.get_channels()
        rows, cols = map(int, self.grid_selector.currentText().split("x"))
        for col in range(cols):
            self.grid_layout.setColumnStretch(col, 1)
        for row in range(rows):
            self.grid_layout.setRowStretch(row, 1)
        index = 0
        for row in range(rows):
            for col in range(cols):
                label = ChannelView(f"Канал {index+1}")
                if index < len(channels):
                    channel_name = channels[index].get("name", f"Канал {index+1}")
                    self.channel_labels[channel_name] = label
                self.grid_layout.addWidget(label, row, col)
                index += 1

    def _on_grid_changed(self, grid: str) -> None:
        self.settings.save_grid(grid)
        self._draw_grid()

    def _start_channels(self) -> None:
        self._stop_workers()
        self.channel_workers = []
        reconnect_conf = self.settings.get_reconnect()
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
        label = self.channel_labels.get(channel_name)
        if not label:
            return
        target_size = label.video_label.contentsRect().size()
        pixmap = QtGui.QPixmap.fromImage(image).scaled(
            target_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        label.set_pixmap(pixmap)

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

    def _handle_event(self, event: Dict) -> None:
        event_id = int(event.get("id", 0))
        frame_image = event.get("frame_image")
        plate_image = event.get("plate_image")
        if event_id:
            self.event_images[event_id] = (frame_image, plate_image)
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
                self.event_images.pop(stale_id, None)

    def _insert_event_row(self, event: Dict, position: Optional[int] = None) -> None:
        row_index = position if position is not None else self.events_table.rowCount()
        self.events_table.insertRow(row_index)

        timestamp = self._format_timestamp(event.get("timestamp", ""))
        plate = event.get("plate", "—")
        channel = event.get("channel", "—")
        event_id = int(event.get("id") or 0)

        id_item = QtWidgets.QTableWidgetItem(timestamp)
        id_item.setData(QtCore.Qt.UserRole, event_id)
        self.events_table.setItem(row_index, 0, id_item)
        self.events_table.setItem(row_index, 1, QtWidgets.QTableWidgetItem(plate))
        self.events_table.setItem(row_index, 2, QtWidgets.QTableWidgetItem(channel))

    def _trim_events_table(self, max_rows: int = 200) -> None:
        while self.events_table.rowCount() > max_rows:
            last_row = self.events_table.rowCount() - 1
            item = self.events_table.item(last_row, 0)
            event_id = int(item.data(QtCore.Qt.UserRole) or 0) if item else 0
            self.events_table.removeRow(last_row)
            if event_id and event_id in self.event_cache:
                self.event_cache.pop(event_id, None)
            if event_id and event_id in self.event_images:
                self.event_images.pop(event_id, None)

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
        selected = self.events_table.selectedItems()
        if not selected:
            return
        event_id_item = selected[0]
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
            self.event_images[event_id] = (frame_image, plate_image)
        display_event = dict(event) if event else None
        if display_event:
            display_event["timestamp"] = self._format_timestamp(display_event.get("timestamp", ""))
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

        self.search_table = QtWidgets.QTableWidget(0, 5)
        self.search_table.setHorizontalHeaderLabels(
            ["Дата/Время", "Канал", "Номер", "Уверенность", "Источник"]
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
        for row_data in rows:
            row_index = self.search_table.rowCount()
            self.search_table.insertRow(row_index)
            formatted_time = self._format_timestamp(row_data["timestamp"])
            self.search_table.setItem(row_index, 0, QtWidgets.QTableWidgetItem(formatted_time))
            self.search_table.setItem(row_index, 1, QtWidgets.QTableWidgetItem(row_data["channel"]))
            self.search_table.setItem(row_index, 2, QtWidgets.QTableWidgetItem(row_data["plate"]))
            self.search_table.setItem(
                row_index, 3, QtWidgets.QTableWidgetItem(f"{row_data['confidence'] or 0:.2f}")
            )
            self.search_table.setItem(row_index, 4, QtWidgets.QTableWidgetItem(row_data["source"]))

    # ------------------ Настройки ------------------
    def _build_settings_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.settings_nav = QtWidgets.QListWidget()
        self.settings_nav.setFixedWidth(180)
        self.settings_nav.setStyleSheet(self.LIST_STYLE)
        self.settings_nav.addItem("Общие")
        self.settings_nav.addItem("Каналы")
        layout.addWidget(self.settings_nav)

        self.settings_stack = QtWidgets.QStackedWidget()
        self.settings_stack.addWidget(self._build_general_settings_tab())
        self.settings_stack.addWidget(self._build_channel_settings_tab())
        layout.addWidget(self.settings_stack, 1)

        self.settings_nav.currentRowChanged.connect(self.settings_stack.setCurrentIndex)
        self.settings_nav.setCurrentRow(0)
        return widget

    def _build_general_settings_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        widget.setStyleSheet(self.GROUP_BOX_STYLE)

        reconnect_group = QtWidgets.QGroupBox("Автоматическое переподключение")
        reconnect_group.setStyleSheet(self.GROUP_BOX_STYLE)
        reconnect_form = QtWidgets.QFormLayout(reconnect_group)
        self.reconnect_on_loss_checkbox = QtWidgets.QCheckBox("Переподключение при потере сигнала")
        reconnect_form.addRow(self.reconnect_on_loss_checkbox)

        self.frame_timeout_input = QtWidgets.QSpinBox()
        self.frame_timeout_input.setRange(1, 300)
        self.frame_timeout_input.setSuffix(" с")
        self.frame_timeout_input.setToolTip("Сколько секунд ждать кадр перед попыткой переподключения")
        reconnect_form.addRow("Таймаут ожидания кадра:", self.frame_timeout_input)

        self.retry_interval_input = QtWidgets.QSpinBox()
        self.retry_interval_input.setRange(1, 300)
        self.retry_interval_input.setSuffix(" с")
        self.retry_interval_input.setToolTip("Интервал между попытками переподключения при потере сигнала")
        reconnect_form.addRow("Интервал между попытками:", self.retry_interval_input)

        self.periodic_reconnect_checkbox = QtWidgets.QCheckBox("Переподключение по таймеру")
        reconnect_form.addRow(self.periodic_reconnect_checkbox)

        self.periodic_interval_input = QtWidgets.QSpinBox()
        self.periodic_interval_input.setRange(1, 1440)
        self.periodic_interval_input.setSuffix(" мин")
        self.periodic_interval_input.setToolTip("Плановое переподключение каждые N минут")
        reconnect_form.addRow("Интервал переподключения:", self.periodic_interval_input)

        storage_group = QtWidgets.QGroupBox("Хранилище")
        storage_group.setStyleSheet(self.GROUP_BOX_STYLE)
        storage_form = QtWidgets.QFormLayout(storage_group)

        db_row = QtWidgets.QHBoxLayout()
        self.db_dir_input = QtWidgets.QLineEdit()
        browse_db_btn = QtWidgets.QPushButton("Выбрать...")
        browse_db_btn.clicked.connect(self._choose_db_dir)
        db_row.addWidget(self.db_dir_input)
        db_row.addWidget(browse_db_btn)
        db_container = QtWidgets.QWidget()
        db_container.setLayout(db_row)
        storage_form.addRow("Папка БД:", db_container)

        screenshot_row = QtWidgets.QHBoxLayout()
        self.screenshot_dir_input = QtWidgets.QLineEdit()
        browse_screenshot_btn = QtWidgets.QPushButton("Выбрать...")
        browse_screenshot_btn.clicked.connect(self._choose_screenshot_dir)
        screenshot_row.addWidget(self.screenshot_dir_input)
        screenshot_row.addWidget(browse_screenshot_btn)
        screenshot_container = QtWidgets.QWidget()
        screenshot_container.setLayout(screenshot_row)
        storage_form.addRow("Папка для скриншотов:", screenshot_container)

        save_general_btn = QtWidgets.QPushButton("Сохранить общие настройки")
        save_general_btn.clicked.connect(self._save_general_settings)

        layout.addWidget(reconnect_group)
        layout.addWidget(storage_group)
        layout.addWidget(save_general_btn, alignment=QtCore.Qt.AlignLeft)
        layout.addStretch()

        self._load_general_settings()
        return widget

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
        add_btn.clicked.connect(self._add_channel)
        remove_btn = QtWidgets.QPushButton("Удалить")
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

        channel_group = QtWidgets.QGroupBox("Канал")
        channel_group.setStyleSheet(self.GROUP_BOX_STYLE)
        channel_form = QtWidgets.QFormLayout(channel_group)
        self.channel_name_input = QtWidgets.QLineEdit()
        self.channel_source_input = QtWidgets.QLineEdit()
        channel_form.addRow("Название:", self.channel_name_input)
        channel_form.addRow("Источник/RTSP:", self.channel_source_input)
        right_panel.addWidget(channel_group)

        recognition_group = QtWidgets.QGroupBox("Распознавание")
        recognition_group.setStyleSheet(self.GROUP_BOX_STYLE)
        recognition_form = QtWidgets.QFormLayout(recognition_group)
        self.best_shots_input = QtWidgets.QSpinBox()
        self.best_shots_input.setRange(1, 50)
        self.best_shots_input.setToolTip("Количество бестшотов, участвующих в консенсусе трека")
        recognition_form.addRow("Бестшоты на трек:", self.best_shots_input)

        self.cooldown_input = QtWidgets.QSpinBox()
        self.cooldown_input.setRange(0, 3600)
        self.cooldown_input.setToolTip(
            "Интервал (в секундах), в течение которого не создается повторное событие для того же номера"
        )
        recognition_form.addRow("Пауза повтора (сек):", self.cooldown_input)

        self.min_conf_input = QtWidgets.QDoubleSpinBox()
        self.min_conf_input.setRange(0.0, 1.0)
        self.min_conf_input.setSingleStep(0.05)
        self.min_conf_input.setDecimals(2)
        self.min_conf_input.setToolTip(
            "Минимальная уверенность OCR (0-1) для приема результата; ниже — помечается как нечитаемое"
        )
        recognition_form.addRow("Мин. уверенность OCR:", self.min_conf_input)
        right_panel.addWidget(recognition_group)

        motion_group = QtWidgets.QGroupBox("Детектор движения")
        motion_group.setStyleSheet(self.GROUP_BOX_STYLE)
        motion_form = QtWidgets.QFormLayout(motion_group)
        self.detection_mode_input = QtWidgets.QComboBox()
        self.detection_mode_input.addItem("Постоянное", "continuous")
        self.detection_mode_input.addItem("Детектор движения", "motion")
        motion_form.addRow("Обнаружение ТС:", self.detection_mode_input)

        self.detector_stride_input = QtWidgets.QSpinBox()
        self.detector_stride_input.setRange(1, 12)
        self.detector_stride_input.setToolTip(
            "Запускать YOLO на каждом N-м кадре в зоне распознавания, чтобы снизить нагрузку"
        )
        motion_form.addRow("Шаг инференса (кадр):", self.detector_stride_input)

        self.motion_threshold_input = QtWidgets.QDoubleSpinBox()
        self.motion_threshold_input.setRange(0.0, 1.0)
        self.motion_threshold_input.setDecimals(3)
        self.motion_threshold_input.setSingleStep(0.005)
        self.motion_threshold_input.setToolTip("Порог чувствительности по площади изменения внутри ROI")
        motion_form.addRow("Порог движения:", self.motion_threshold_input)

        self.motion_stride_input = QtWidgets.QSpinBox()
        self.motion_stride_input.setRange(1, 30)
        self.motion_stride_input.setToolTip("Обрабатывать каждый N-й кадр для поиска движения")
        motion_form.addRow("Частота анализа (кадр):", self.motion_stride_input)

        self.motion_activation_frames_input = QtWidgets.QSpinBox()
        self.motion_activation_frames_input.setRange(1, 60)
        self.motion_activation_frames_input.setToolTip("Сколько кадров подряд должно быть движение, чтобы включить распознавание")
        motion_form.addRow("Мин. кадров с движением:", self.motion_activation_frames_input)

        self.motion_release_frames_input = QtWidgets.QSpinBox()
        self.motion_release_frames_input.setRange(1, 120)
        self.motion_release_frames_input.setToolTip("Сколько кадров без движения нужно, чтобы остановить распознавание")
        motion_form.addRow("Мин. кадров без движения:", self.motion_release_frames_input)
        right_panel.addWidget(motion_group)

        roi_group = QtWidgets.QGroupBox("Зона распознавания")
        roi_group.setStyleSheet(self.GROUP_BOX_STYLE)
        roi_layout = QtWidgets.QGridLayout()
        self.roi_x_input = QtWidgets.QSpinBox()
        self.roi_x_input.setRange(0, 100)
        self.roi_y_input = QtWidgets.QSpinBox()
        self.roi_y_input.setRange(0, 100)
        self.roi_w_input = QtWidgets.QSpinBox()
        self.roi_w_input.setRange(1, 100)
        self.roi_h_input = QtWidgets.QSpinBox()
        self.roi_h_input.setRange(1, 100)

        for spin in (self.roi_x_input, self.roi_y_input, self.roi_w_input, self.roi_h_input):
            spin.valueChanged.connect(self._on_roi_inputs_changed)

        roi_layout.addWidget(QtWidgets.QLabel("X (%):"), 0, 0)
        roi_layout.addWidget(self.roi_x_input, 0, 1)
        roi_layout.addWidget(QtWidgets.QLabel("Y (%):"), 1, 0)
        roi_layout.addWidget(self.roi_y_input, 1, 1)
        roi_layout.addWidget(QtWidgets.QLabel("Ширина (%):"), 2, 0)
        roi_layout.addWidget(self.roi_w_input, 2, 1)
        roi_layout.addWidget(QtWidgets.QLabel("Высота (%):"), 3, 0)
        roi_layout.addWidget(self.roi_h_input, 3, 1)
        refresh_btn = QtWidgets.QPushButton("Обновить кадр")
        refresh_btn.clicked.connect(self._refresh_preview_frame)
        roi_layout.addWidget(refresh_btn, 4, 0, 1, 2)
        roi_group.setLayout(roi_layout)
        right_panel.addWidget(roi_group)

        save_btn = QtWidgets.QPushButton("Сохранить канал")
        save_btn.clicked.connect(self._save_channel)
        right_panel.addWidget(save_btn)
        right_panel.addStretch()

        layout.addLayout(right_panel, 2)

        self._load_general_settings()
        self._reload_channels_list()
        return widget

    def _reload_channels_list(self) -> None:
        self.channels_list.clear()
        for channel in self.settings.get_channels():
            self.channels_list.addItem(channel.get("name", "Канал"))
        if self.channels_list.count():
            self.channels_list.setCurrentRow(0)

    def _load_general_settings(self) -> None:
        reconnect = self.settings.get_reconnect()
        signal_loss = reconnect.get("signal_loss", {})
        periodic = reconnect.get("periodic", {})
        self.db_dir_input.setText(self.settings.get_db_dir())
        self.screenshot_dir_input.setText(self.settings.get_screenshot_dir())

        self.reconnect_on_loss_checkbox.setChecked(bool(signal_loss.get("enabled", True)))
        self.frame_timeout_input.setValue(int(signal_loss.get("frame_timeout_seconds", 5)))
        self.retry_interval_input.setValue(int(signal_loss.get("retry_interval_seconds", 5)))

        self.periodic_reconnect_checkbox.setChecked(bool(periodic.get("enabled", False)))
        self.periodic_interval_input.setValue(int(periodic.get("interval_minutes", 60)))

    def _choose_screenshot_dir(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Выбор папки для скриншотов")
        if directory:
            self.screenshot_dir_input.setText(directory)

    def _choose_db_dir(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Выбор папки базы данных")
        if directory:
            self.db_dir_input.setText(directory)

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
