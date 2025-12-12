#!/usr/bin/env python3
# /anpr/ui/main_window.py
import os
from datetime import datetime
from pathlib import Path

import cv2
import psutil
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

from anpr.postprocessing.country_config import CountryConfigLoader
from anpr.workers.channel_worker import ChannelWorker
from anpr.infrastructure.logging_manager import get_logger
from anpr.infrastructure.settings_manager import SettingsManager
from anpr.infrastructure.storage import EventDatabase

logger = get_logger(__name__)


class ModernToggleSwitch(QtWidgets.QWidget):
    """Современный переключатель (toggle switch)."""
    
    toggled = QtCore.pyqtSignal(bool)
    
    def __init__(self, parent=None, default_state=False):
        super().__init__(parent)
        self._state = default_state
        self._anim = QtCore.QPropertyAnimation(self, b"thumb_position")
        self._anim.setDuration(150)
        self._thumb_position = 0
        self.setFixedSize(60, 30)
        self._update_thumb_position()
        
    def get_thumb_position(self):
        return self._thumb_position
    
    def set_thumb_position(self, pos):
        self._thumb_position = pos
        self.update()
    
    thumb_position = QtCore.pyqtProperty(int, get_thumb_position, set_thumb_position)
    
    def _update_thumb_position(self):
        self._anim.stop()
        self._anim.setStartValue(self._thumb_position)
        self._anim.setEndValue(30 if self._state else 0)
        self._anim.start()
    
    def mousePressEvent(self, event):
        self.toggle()
        self.toggled.emit(self._state)
    
    def toggle(self):
        self._state = not self._state
        self._update_thumb_position()
    
    def setChecked(self, checked):
        if self._state != checked:
            self._state = checked
            self._update_thumb_position()
    
    def isChecked(self):
        return self._state
    
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # Фон
        bg_color = QtGui.QColor("#4CAF50" if self._state else "#666666")
        painter.setBrush(bg_color)
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawRoundedRect(0, 0, self.width(), self.height(), 15, 15)
        
        # Ползунок
        painter.setBrush(QtGui.QColor("#FFFFFF"))
        painter.drawEllipse(self._thumb_position, 3, 24, 24)


class ModernCardWidget(QtWidgets.QFrame):
    """Карточка в современном стиле."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QtWidgets.QFrame.StyledPanel)
        self.setStyleSheet("""
            ModernCardWidget {
                background-color: #1E1E2E;
                border: 1px solid #2D2D3D;
                border-radius: 8px;
                padding: 16px;
            }
            ModernCardWidget:hover {
                border-color: #00B4D8;
            }
        """)


class SectionHeader(QtWidgets.QWidget):
    """Заголовок раздела с горизонтальной линией."""
    
    def __init__(self, text, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.label = QtWidgets.QLabel(text)
        self.label.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #E0E0E0;
            padding-right: 12px;
        """)
        
        self.line = QtWidgets.QFrame()
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setStyleSheet("background-color: #3A3A4C;")
        self.line.setFixedHeight(1)
        
        layout.addWidget(self.label)
        layout.addWidget(self.line, 1)


class SettingRow(QtWidgets.QWidget):
    """Строка настройки с пояснением."""
    
    def __init__(self, label_text, widget, tooltip="", parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        
        self.label = QtWidgets.QLabel(label_text)
        self.label.setStyleSheet("color: #CCCCCC; font-size: 12px;")
        self.label.setFixedWidth(200)
        
        if tooltip:
            self.label.setToolTip(tooltip)
            widget.setToolTip(tooltip)
        
        layout.addWidget(self.label)
        layout.addWidget(widget, 1)
        
        # Сохраняем виджет как атрибут
        self.widget = widget 

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

    def __init__(self, name: str, pixmap_pool: Optional[PixmapPool]) -> None:
        super().__init__()
        self.name = name
        self._pixmap_pool = pixmap_pool
        self._current_pixmap: Optional[QtGui.QPixmap] = None

        self.video_label = QtWidgets.QLabel("Нет сигнала")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            background-color: #000000;
            color: #AAAAAA;
            border: 2px solid #2D2D3D;
            border-radius: 4px;
            padding: 4px;
            font-size: 11px;
        """)
        self.video_label.setMinimumSize(220, 170)
        self.video_label.setScaledContents(False)
        self.video_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.video_label)

        # Заголовок канала
        self.channel_header = QtWidgets.QLabel(name)
        self.channel_header.setParent(self.video_label)
        self.channel_header.setStyleSheet("""
            background-color: rgba(0, 180, 216, 0.85);
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 11px;
        """)
        self.channel_header.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        
        # Индикатор движения
        self.motion_indicator = QtWidgets.QLabel("⚡")
        self.motion_indicator.setParent(self.video_label)
        self.motion_indicator.setStyleSheet("""
            background-color: rgba(255, 193, 7, 0.9);
            color: #000000;
            padding: 4px 8px;
            border-radius: 12px;
            font-weight: bold;
            font-size: 12px;
        """)
        self.motion_indicator.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.motion_indicator.hide()

        # Последний номер
        self.last_plate = QtWidgets.QLabel("—")
        self.last_plate.setParent(self.video_label)
        self.last_plate.setStyleSheet("""
            background-color: rgba(0, 0, 0, 0.7);
            color: #00E5FF;
            padding: 6px 10px;
            border-radius: 6px;
            font-weight: bold;
            font-size: 12px;
            border-left: 3px solid #00B4D8;
        """)
        self.last_plate.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.last_plate.hide()

        # Статус
        self.status_hint = QtWidgets.QLabel("")
        self.status_hint.setParent(self.video_label)
        self.status_hint.setStyleSheet("""
            background-color: rgba(30, 30, 46, 0.85);
            color: #CCCCCC;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
        """)
        self.status_hint.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.status_hint.hide()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        rect = self.video_label.contentsRect()
        margin = 8
        
        # Позиционируем заголовок
        header_size = self.channel_header.sizeHint()
        self.channel_header.move(rect.left() + margin, rect.top() + margin)
        
        # Позиционируем индикатор движения
        motion_size = self.motion_indicator.sizeHint()
        self.motion_indicator.move(
            rect.right() - motion_size.width() - margin, rect.top() + margin
        )
        
        # Позиционируем номер
        self.last_plate.move(rect.left() + margin, rect.top() + header_size.height() + margin * 2)
        
        # Позиционируем статус
        status_size = self.status_hint.sizeHint()
        self.status_hint.move(
            rect.left() + margin, 
            rect.bottom() - status_size.height() - margin
        )

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
        self.setStyleSheet("""
            background-color: #0F0F1A;
            color: #888888;
            border: 2px solid #2D2D3D;
            border-radius: 4px;
            padding: 8px;
        """)
        self._roi = {"x": 0, "y": 0, "width": 100, "height": 100}
        self._pixmap: Optional[QtGui.QPixmap] = None
        self._rubber_band = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)
        self._rubber_band.setStyleSheet("background-color: rgba(0, 180, 216, 0.3); border: 2px solid #00B4D8;")
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

    def setPixmap(self, pixmap: Optional[QtGui.QPixmap]) -> None:
        self._pixmap = pixmap
        if pixmap is None:
            super().setPixmap(QtGui.QPixmap())
            self.setText("Нет кадра")
            return
        scaled = self._scaled_pixmap(self.size())
        super().setPixmap(scaled)
        self.setText("")

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
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

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        geom = self._image_geometry()
        if geom is None:
            return
        offset, size = geom
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # Рисуем ROI
        roi_rect = QtCore.QRect(
            offset.x() + int(size.width() * self._roi["x"] / 100),
            offset.y() + int(size.height() * self._roi["y"] / 100),
            int(size.width() * self._roi["width"] / 100),
            int(size.height() * self._roi["height"] / 100),
        )
        
        # Внешняя рамка
        pen = QtGui.QPen(QtGui.QColor("#00B4D8"))
        pen.setWidth(3)
        painter.setPen(pen)
        painter.setBrush(QtGui.QColor(0, 180, 216, 20))
        painter.drawRect(roi_rect)
        
        # Внутренняя рамка
        pen = QtGui.QPen(QtGui.QColor("#FFFFFF"))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawRect(roi_rect.adjusted(1, 1, -1, -1))

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
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

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._origin is None:
            return
        rect = QtCore.QRect(self._origin, event.pos()).normalized()
        self._rubber_band.setGeometry(rect)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
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
        self.setStyleSheet("background-color: transparent;")
        
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setSpacing(12)
        
        # Заголовок
        header = QtWidgets.QLabel("Детали события")
        header.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #00B4D8;
            padding-bottom: 8px;
            border-bottom: 2px solid #2D2D3D;
        """)
        main_layout.addWidget(header)
        
        # Карточка с превью кадра
        frame_card = ModernCardWidget()
        frame_layout = QtWidgets.QVBoxLayout(frame_card)
        
        frame_header = QtWidgets.QLabel("📷 Кадр распознавания")
        frame_header.setStyleSheet("font-size: 12px; font-weight: bold; color: #E0E0E0;")
        frame_layout.addWidget(frame_header)
        
        self.frame_preview = QtWidgets.QLabel("Нет изображения")
        self.frame_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.frame_preview.setMinimumHeight(280)
        self.frame_preview.setStyleSheet("""
            background-color: #0F0F1A;
            color: #666666;
            border: 1px solid #2D2D3D;
            border-radius: 4px;
            padding: 8px;
        """)
        self.frame_preview.setScaledContents(False)
        frame_layout.addWidget(self.frame_preview)
        main_layout.addWidget(frame_card)
        
        # Нижняя строка с номером и метаданными
        bottom_row = QtWidgets.QHBoxLayout()
        bottom_row.setSpacing(12)
        
        # Карточка с превью номера
        plate_card = ModernCardWidget()
        plate_card.setMinimumWidth(220)
        plate_layout = QtWidgets.QVBoxLayout(plate_card)
        
        plate_header = QtWidgets.QLabel("🚗 Область номера")
        plate_header.setStyleSheet("font-size: 12px; font-weight: bold; color: #E0E0E0;")
        plate_layout.addWidget(plate_header)
        
        self.plate_preview = QtWidgets.QLabel("Нет изображения")
        self.plate_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.plate_preview.setMinimumSize(200, 140)
        self.plate_preview.setStyleSheet("""
            background-color: #0F0F1A;
            color: #666666;
            border: 1px solid #2D2D3D;
            border-radius: 4px;
            padding: 8px;
        """)
        self.plate_preview.setScaledContents(False)
        plate_layout.addWidget(self.plate_preview)
        bottom_row.addWidget(plate_card)
        
        # Карточка с метаданными
        meta_card = ModernCardWidget()
        meta_card.setMinimumWidth(280)
        meta_layout = QtWidgets.QVBoxLayout(meta_card)
        
        meta_header = QtWidgets.QLabel("📊 Данные распознавания")
        meta_header.setStyleSheet("font-size: 12px; font-weight: bold; color: #E0E0E0;")
        meta_layout.addWidget(meta_header)
        
        # Таблица метаданных
        meta_form = QtWidgets.QFormLayout()
        meta_form.setHorizontalSpacing(16)
        meta_form.setVerticalSpacing(8)
        
        self.time_label = self._create_detail_label()
        self.channel_label = self._create_detail_label()
        self.country_label = self._create_detail_label()
        self.plate_label = self._create_detail_label(style="font-size: 18px; font-weight: bold; color: #00E5FF;")
        self.conf_label = self._create_detail_label()
        
        meta_form.addRow("🕒 Время:", self.time_label)
        meta_form.addRow("📡 Канал:", self.channel_label)
        meta_form.addRow("🌍 Страна:", self.country_label)
        meta_form.addRow("🚘 Номер:", self.plate_label)
        meta_form.addRow("🎯 Уверенность:", self.conf_label)
        
        meta_layout.addLayout(meta_form)
        bottom_row.addWidget(meta_card, 1)
        
        main_layout.addLayout(bottom_row)

    def _create_detail_label(self, style=""):
        label = QtWidgets.QLabel("—")
        label.setStyleSheet(f"""
            color: #CCCCCC;
            font-size: 13px;
            padding: 4px 0px;
            {style}
        """)
        return label

    def clear(self) -> None:
        self.time_label.setText("—")
        self.channel_label.setText("—")
        self.country_label.setText("—")
        self.plate_label.setText("—")
        self.conf_label.setText("—")
        
        self.frame_preview.setPixmap(QtGui.QPixmap())
        self.frame_preview.setText("Нет изображения")
        
        self.plate_preview.setPixmap(QtGui.QPixmap())
        self.plate_preview.setText("Нет изображения")

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
        self.conf_label.setText(f"{float(conf):.2%}" if conf is not None else "—")

        self._set_image(self.frame_preview, frame_image, keep_aspect=True)
        self._set_image(self.plate_preview, plate_image, keep_aspect=True)

    def _set_image(
        self,
        label: QtWidgets.QLabel,
        image: Optional[QtGui.QImage],
        keep_aspect: bool = False,
    ) -> None:
        if image is None:
            label.setPixmap(QtGui.QPixmap())
            label.setText("Нет изображения")
            return
        label.setText("")
        pixmap = QtGui.QPixmap.fromImage(image)
        if keep_aspect:
            pixmap = pixmap.scaled(
                label.size(), 
                QtCore.Qt.KeepAspectRatio, 
                QtCore.Qt.SmoothTransformation
            )
        label.setPixmap(pixmap)


class MainWindow(QtWidgets.QMainWindow):
    """Главное окно приложения ANPR с современным дизайном."""

    GRID_VARIANTS = ["1x1", "1x2", "2x2", "2x3", "3x3"]
    MAX_IMAGE_CACHE = 200
    MAX_IMAGE_CACHE_BYTES = 256 * 1024 * 1024  # 256 MB
    
    # Современная цветовая схема
    COLORS = {
        "primary": "#00B4D8",
        "primary_light": "#4DF8FF",
        "primary_dark": "#0077A6",
        "background": "#0F0F1A",
        "surface": "#1E1E2E",
        "surface_light": "#2D2D3D",
        "text_primary": "#FFFFFF",
        "text_secondary": "#CCCCCC",
        "text_hint": "#888888",
        "success": "#4CAF50",
        "warning": "#FFC107",
        "error": "#F44336",
        "border": "#3A3A4C"
    }

    def __init__(self, settings: Optional[SettingsManager] = None) -> None:
        super().__init__()
        self.setWindowTitle("ANPR Vision Pro")
        self.setWindowIcon(self.style().standardIcon(QtWidgets.QStyle.SP_ComputerIcon))
        self.resize(1440, 900)

        # Центрирование окна
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        self.move(screen.center() - self.rect().center())

        self.settings = settings or SettingsManager()
        self.db = EventDatabase(self.settings.get_db_path())

        self._pixmap_pool = PixmapPool()
        self.channel_workers: List[ChannelWorker] = []
        self.channel_labels: Dict[str, ChannelView] = {}
        self.event_images: "OrderedDict[int, Tuple[Optional[QtGui.QImage], Optional[QtGui.QImage]]]" = OrderedDict()
        self._image_cache_bytes = 0
        self.event_cache: Dict[int, Dict] = {}
        self.flag_cache: Dict[str, Optional[QtGui.QIcon]] = {}
        self.flag_dir = Path(__file__).resolve().parents[2] / "images" / "flags"

        # Применение стилей
        self._apply_styles()
        
        # Создание вкладок
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        
        self.observation_tab = self._build_observation_tab()
        self.search_tab = self._build_search_tab()
        self.settings_tab = self._build_settings_tab()

        self.tabs.addTab(self.observation_tab, "🎥 Наблюдение")
        self.tabs.addTab(self.search_tab, "🔍 Поиск")
        self.tabs.addTab(self.settings_tab, "⚙️ Настройки")

        self.setCentralWidget(self.tabs)
        self._build_status_bar()
        self._start_system_monitoring()
        self._refresh_events_table()
        self._start_channels()

    def _apply_styles(self):
        """Применяет современные стили к приложению."""
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {self.COLORS['background']};
            }}
            
            QTabWidget::pane {{
                border: 1px solid {self.COLORS['border']};
                background-color: {self.COLORS['surface']};
                border-radius: 6px;
                margin-top: 4px;
            }}
            
            QTabBar::tab {{
                background-color: {self.COLORS['surface_light']};
                color: {self.COLORS['text_secondary']};
                padding: 12px 24px;
                margin-right: 2px;
                border: 1px solid {self.COLORS['border']};
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-size: 13px;
                font-weight: 500;
            }}
            
            QTabBar::tab:selected {{
                background-color: {self.COLORS['surface']};
                color: {self.COLORS['primary']};
                border-color: {self.COLORS['primary']};
                border-bottom: 2px solid {self.COLORS['primary']};
                font-weight: 600;
            }}
            
            QTabBar::tab:hover {{
                background-color: {self.COLORS['surface_light']};
                color: {self.COLORS['text_primary']};
            }}
            
            QPushButton {{
                background-color: {self.COLORS['primary']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: 600;
                font-size: 13px;
                min-height: 36px;
            }}
            
            QPushButton:hover {{
                background-color: {self.COLORS['primary_light']};
            }}
            
            QPushButton:pressed {{
                background-color: {self.COLORS['primary_dark']};
            }}
            
            QPushButton:disabled {{
                background-color: {self.COLORS['surface_light']};
                color: {self.COLORS['text_hint']};
            }}
            
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QDateTimeEdit {{
                background-color: {self.COLORS['surface_light']};
                color: {self.COLORS['text_primary']};
                border: 1px solid {self.COLORS['border']};
                border-radius: 4px;
                padding: 8px 12px;
                font-size: 13px;
                selection-background-color: {self.COLORS['primary']};
                min-height: 36px;
            }}
            
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QDateTimeEdit:focus {{
                border: 2px solid {self.COLORS['primary']};
                padding: 7px 11px;
            }}
            
            QComboBox::drop-down {{
                border: none;
                width: 24px;
            }}
            
            QComboBox::down-arrow {{
                image: url(none);
                border-left: 1px solid {self.COLORS['border']};
                padding: 0 8px;
            }}
            
            QCheckBox {{
                color: {self.COLORS['text_secondary']};
                font-size: 13px;
                spacing: 8px;
            }}
            
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border: 2px solid {self.COLORS['border']};
                border-radius: 4px;
            }}
            
            QCheckBox::indicator:checked {{
                background-color: {self.COLORS['primary']};
                border-color: {self.COLORS['primary']};
                image: url('images/check.svg');
            }}
            
            QLabel {{
                color: {self.COLORS['text_secondary']};
                font-size: 13px;
            }}
            
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {self.COLORS['border']};
                border-radius: 8px;
                margin-top: 16px;
                padding-top: 12px;
                background-color: {self.COLORS['surface']};
                color: {self.COLORS['text_primary']};
                font-size: 14px;
            }}
            
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px 0 8px;
                background-color: {self.COLORS['surface']};
            }}
        """)

    def _build_status_bar(self) -> None:
        status = self.statusBar()
        status.setSizeGripEnabled(False)
        
        # Контейнер для статуса
        status_container = QtWidgets.QWidget()
        status_layout = QtWidgets.QHBoxLayout(status_container)
        status_layout.setContentsMargins(12, 4, 12, 4)
        status_layout.setSpacing(20)
        
        # Индикатор системы
        sys_group = QtWidgets.QWidget()
        sys_layout = QtWidgets.QHBoxLayout(sys_group)
        sys_layout.setSpacing(8)
        
        cpu_icon = QtWidgets.QLabel("⚙️")
        self.cpu_label = QtWidgets.QLabel("CPU: —")
        ram_icon = QtWidgets.QLabel("💾")
        self.ram_label = QtWidgets.QLabel("RAM: —")
        
        for widget in [cpu_icon, self.cpu_label, ram_icon, self.ram_label]:
            widget.setStyleSheet("color: #AAAAAA; font-size: 12px;")
            sys_layout.addWidget(widget)
        
        status_layout.addWidget(sys_group)
        status_layout.addStretch()
        
        # Индикатор базы данных
        db_size = self._get_db_size()
        db_label = QtWidgets.QLabel(f"📊 База: {db_size}")
        db_label.setStyleSheet("color: #AAAAAA; font-size: 12px;")
        status_layout.addWidget(db_label)
        
        # Версия приложения
        version_label = QtWidgets.QLabel("ANPR Vision Pro v1.0")
        version_label.setStyleSheet("color: #00B4D8; font-size: 12px; font-weight: bold;")
        status_layout.addWidget(version_label)
        
        status.addPermanentWidget(status_container)

    def _get_db_size(self) -> str:
        """Возвращает размер базы данных в читаемом формате."""
        db_path = Path(self.settings.get_db_path())
        if db_path.exists():
            size = db_path.stat().st_size
            if size < 1024:
                return f"{size} B"
            elif size < 1024 * 1024:
                return f"{size / 1024:.1f} KB"
            else:
                return f"{size / (1024 * 1024):.1f} MB"
        return "0 B"

    def _start_system_monitoring(self) -> None:
        self.stats_timer = QtCore.QTimer(self)
        self.stats_timer.setInterval(2000)  # Обновление каждые 2 секунды
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
        main_layout = QtWidgets.QHBoxLayout(widget)
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # Левая колонка - видео
        left_column = QtWidgets.QVBoxLayout()
        left_column.setSpacing(12)
        
        # Панель управления
        control_card = ModernCardWidget()
        control_layout = QtWidgets.QHBoxLayout(control_card)
        control_layout.setContentsMargins(16, 12, 16, 12)
        
        control_layout.addWidget(QtWidgets.QLabel("Конфигурация сетки:"))
        
        self.grid_selector = QtWidgets.QComboBox()
        self.grid_selector.addItems(self.GRID_VARIANTS)
        self.grid_selector.setCurrentText(self.settings.get_grid())
        self.grid_selector.currentTextChanged.connect(self._on_grid_changed)
        self.grid_selector.setFixedWidth(120)
        control_layout.addWidget(self.grid_selector)
        
        control_layout.addStretch()
        
        # Кнопка обновления
        refresh_btn = QtWidgets.QPushButton("🔄 Обновить потоки")
        refresh_btn.clicked.connect(self._start_channels)
        refresh_btn.setFixedWidth(150)
        control_layout.addWidget(refresh_btn)
        
        left_column.addWidget(control_card)

        # Сетка видео
        self.grid_widget = QtWidgets.QWidget()
        self.grid_layout = QtWidgets.QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(8)
        self.grid_layout.setContentsMargins(4, 4, 4, 4)
        
        left_column.addWidget(self.grid_widget, stretch=1)
        main_layout.addLayout(left_column, stretch=3)

        # Правая колонка - детали и события
        right_column = QtWidgets.QVBoxLayout()
        right_column.setSpacing(16)
        
        # Детали события
        detail_card = ModernCardWidget()
        detail_layout = QtWidgets.QVBoxLayout(detail_card)
        self.event_detail = EventDetailView()
        detail_layout.addWidget(self.event_detail)
        right_column.addWidget(detail_card, stretch=2)
        
        # Список событий
        events_card = ModernCardWidget()
        events_layout = QtWidgets.QVBoxLayout(events_card)
        
        events_header = QtWidgets.QLabel("📋 Последние события")
        events_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #00B4D8;")
        events_layout.addWidget(events_header)
        
        # Панель управления таблицей
        table_controls = QtWidgets.QHBoxLayout()
        
        clear_btn = QtWidgets.QPushButton("🗑️ Очистить")
        clear_btn.setFixedWidth(100)
        clear_btn.clicked.connect(lambda: self.events_table.setRowCount(0))
        table_controls.addWidget(clear_btn)
        
        export_btn = QtWidgets.QPushButton("📤 Экспорт")
        export_btn.setFixedWidth(100)
        export_btn.clicked.connect(self._export_events)
        table_controls.addWidget(export_btn)
        
        table_controls.addStretch()
        
        filter_label = QtWidgets.QLabel("Фильтр:")
        filter_label.setStyleSheet("color: #CCCCCC;")
        table_controls.addWidget(filter_label)
        
        self.events_filter = QtWidgets.QLineEdit()
        self.events_filter.setPlaceholderText("По номеру или каналу...")
        self.events_filter.setFixedWidth(200)
        self.events_filter.textChanged.connect(self._filter_events_table)
        table_controls.addWidget(self.events_filter)
        
        events_layout.addLayout(table_controls)
        
        # Таблица событий
        self.events_table = QtWidgets.QTableWidget(0, 5)
        self.events_table.setHorizontalHeaderLabels(["Время", "Номер", "Страна", "Канал", "Уверенность"])
        self.events_table.horizontalHeader().setStretchLastSection(True)
        self.events_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.events_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.events_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.events_table.verticalHeader().setVisible(False)
        self.events_table.setAlternatingRowColors(True)
        self.events_table.setSortingEnabled(True)
        self.events_table.itemSelectionChanged.connect(self._on_event_selected)
        
        # Настройка колонок
        self.events_table.setColumnWidth(0, 150)  # Время
        self.events_table.setColumnWidth(1, 140)  # Номер
        self.events_table.setColumnWidth(2, 80)   # Страна
        self.events_table.setColumnWidth(3, 100)  # Канал
        self.events_table.setColumnWidth(4, 100)  # Уверенность
        
        events_layout.addWidget(self.events_table)
        right_column.addWidget(events_card, stretch=1)

        main_layout.addLayout(right_column, stretch=2)

        self._draw_grid()
        return widget

    def _export_events(self):
        """Экспорт событий в CSV."""
        from datetime import datetime
        
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Экспорт событий", f"anpr_export_{datetime.now():%Y%m%d_%H%M%S}.csv", "CSV Files (*.csv)"
        )
        
        if filename:
            import csv
            
            with open(filename, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                # Заголовки
                headers = ['Время', 'Номер', 'Страна', 'Канал', 'Уверенность', 'Источник']
                writer.writerow(headers)
                
                # Данные
                for row in range(self.events_table.rowCount()):
                    row_data = []
                    for col in range(self.events_table.columnCount()):
                        item = self.events_table.item(row, col)
                        row_data.append(item.text() if item else "")
                    writer.writerow(row_data)

    def _filter_events_table(self, text):
        """Фильтрация таблицы событий."""
        for row in range(self.events_table.rowCount()):
            match = False
            for col in range(self.events_table.columnCount()):
                item = self.events_table.item(row, col)
                if item and text.lower() in item.text().lower():
                    match = True
                    break
            self.events_table.setRowHidden(row, not match)

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
            return parsed.strftime("%H:%M:%S\n%d.%m.%Y")
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
        
        # Настройка растяжения
        for col in range(cols):
            self.grid_layout.setColumnStretch(col, 1)
        for row in range(rows):
            self.grid_layout.setRowStretch(row, 1)
            
        index = 0
        for row in range(rows):
            for col in range(cols):
                label = ChannelView(f"Канал {index+1}", self._pixmap_pool)
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

        timestamp = self._format_timestamp(event.get("timestamp", ""))
        plate = event.get("plate", "—")
        channel = event.get("channel", "—")
        event_id = int(event.get("id") or 0)
        country_code = event.get("country") or "—"
        confidence = event.get("confidence")

        # Время
        time_item = QtWidgets.QTableWidgetItem(timestamp)
        time_item.setData(QtCore.Qt.UserRole, event_id)
        time_item.setData(QtCore.Qt.UserRole + 1, event.get("timestamp", ""))
        self.events_table.setItem(row_index, 0, time_item)
        
        # Номер
        plate_item = QtWidgets.QTableWidgetItem(plate)
        plate_item.setForeground(QtGui.QColor("#00E5FF"))
        self.events_table.setItem(row_index, 1, plate_item)
        
        # Страна
        country_item = QtWidgets.QTableWidgetItem(country_code if country_code != "—" else "")
        country_icon = self._get_flag_icon(event.get("country"))
        if country_icon:
            country_item.setIcon(country_icon)
        country_item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.events_table.setItem(row_index, 2, country_item)
        
        # Канал
        channel_item = QtWidgets.QTableWidgetItem(channel)
        self.events_table.setItem(row_index, 3, channel_item)
        
        # Уверенность
        conf_text = f"{float(confidence):.2%}" if confidence is not None else "—"
        conf_item = QtWidgets.QTableWidgetItem(conf_text)
        if confidence is not None:
            # Цвет в зависимости от уверенности
            if confidence >= 0.8:
                conf_item.setForeground(QtGui.QColor("#4CAF50"))
            elif confidence >= 0.6:
                conf_item.setForeground(QtGui.QColor("#FFC107"))
            else:
                conf_item.setForeground(QtGui.QColor("#F44336"))
        conf_item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.events_table.setItem(row_index, 4, conf_item)

    def _trim_events_table(self, max_rows: int = 500) -> None:
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
            display_event["timestamp"] = self._format_timestamp(display_event.get("timestamp", ""))
        self.event_detail.set_event(display_event, frame_image, plate_image)

    def _refresh_events_table(self, select_id: Optional[int] = None) -> None:
        rows = self.db.fetch_recent(limit=500)
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
        layout.setSpacing(16)
        layout.setContentsMargins(16, 16, 16, 16)

        # Карточка фильтров
        filters_card = ModernCardWidget()
        filters_layout = QtWidgets.QVBoxLayout(filters_card)
        
        filters_header = QtWidgets.QLabel("🔍 Параметры поиска")
        filters_header.setStyleSheet("font-size: 16px; font-weight: bold; color: #00B4D8;")
        filters_layout.addWidget(filters_header)
        
        # Форма фильтров
        form = QtWidgets.QFormLayout()
        form.setHorizontalSpacing(24)
        form.setVerticalSpacing(12)
        
        self.search_plate = QtWidgets.QLineEdit()
        self.search_plate.setPlaceholderText("Введите номер или его часть...")
        self.search_plate.setMinimumWidth(300)
        
        date_layout = QtWidgets.QHBoxLayout()
        self.search_from = QtWidgets.QDateTimeEdit()
        self._prepare_optional_datetime(self.search_from)
        self.search_to = QtWidgets.QDateTimeEdit()
        self._prepare_optional_datetime(self.search_to)
        date_layout.addWidget(self.search_from)
        date_layout.addWidget(QtWidgets.QLabel("до"))
        date_layout.addWidget(self.search_to)
        date_layout.addStretch()
        
        form.addRow("🚘 Номер:", self.search_plate)
        form.addRow("📅 Период:", date_layout)
        
        filters_layout.addLayout(form)
        
        # Кнопки поиска
        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch()
        
        search_btn = QtWidgets.QPushButton("🔍 Начать поиск")
        search_btn.clicked.connect(self._run_plate_search)
        search_btn.setMinimumWidth(150)
        button_row.addWidget(search_btn)
        
        clear_btn = QtWidgets.QPushButton("🗑️ Очистить")
        clear_btn.clicked.connect(self._clear_search)
        clear_btn.setMinimumWidth(120)
        button_row.addWidget(clear_btn)
        
        filters_layout.addLayout(button_row)
        layout.addWidget(filters_card)

        # Результаты поиска
        results_card = ModernCardWidget()
        results_layout = QtWidgets.QVBoxLayout(results_card)
        
        results_header = QtWidgets.QLabel("📊 Результаты поиска")
        results_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #00B4D8;")
        results_layout.addWidget(results_header)
        
        # Статистика
        self.results_stats = QtWidgets.QLabel("Найдено: 0 записей")
        self.results_stats.setStyleSheet("color: #CCCCCC; font-size: 12px;")
        results_layout.addWidget(self.results_stats)
        
        self.search_table = QtWidgets.QTableWidget(0, 6)
        self.search_table.setHorizontalHeaderLabels(
            ["Время", "Канал", "Страна", "Номер", "Уверенность", "Источник"]
        )
        self.search_table.horizontalHeader().setStretchLastSection(True)
        self.search_table.setAlternatingRowColors(True)
        self.search_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.search_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.search_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.search_table.verticalHeader().setVisible(False)
        self.search_table.setSortingEnabled(True)
        
        # Настройка колонок
        self.search_table.setColumnWidth(0, 150)
        self.search_table.setColumnWidth(1, 100)
        self.search_table.setColumnWidth(2, 80)
        self.search_table.setColumnWidth(3, 140)
        self.search_table.setColumnWidth(4, 100)
        
        results_layout.addWidget(self.search_table)
        layout.addWidget(results_card, stretch=1)

        return widget

    def _clear_search(self):
        """Очистка полей поиска."""
        self.search_plate.clear()
        self.search_from.setDateTime(self.search_from.minimumDateTime())
        self.search_to.setDateTime(self.search_to.minimumDateTime())
        self.search_table.setRowCount(0)
        self.results_stats.setText("Найдено: 0 записей")

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
            time_item = QtWidgets.QTableWidgetItem(formatted_time)
            time_item.setData(QtCore.Qt.UserRole + 1, row_data["timestamp"])
            self.search_table.setItem(row_index, 0, time_item)
            
            self.search_table.setItem(row_index, 1, QtWidgets.QTableWidgetItem(row_data["channel"]))
            
            country_item = QtWidgets.QTableWidgetItem(row_data["country"] or "")
            country_icon = self._get_flag_icon(row_data["country"])
            if country_icon:
                country_item.setIcon(country_icon)
            country_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.search_table.setItem(row_index, 2, country_item)
            
            plate_item = QtWidgets.QTableWidgetItem(row_data["plate"])
            plate_item.setForeground(QtGui.QColor("#00E5FF"))
            self.search_table.setItem(row_index, 3, plate_item)
            
            conf_item = QtWidgets.QTableWidgetItem(f"{row_data['confidence'] or 0:.2%}")
            confidence = row_data.get('confidence', 0)
            if confidence >= 0.8:
                conf_item.setForeground(QtGui.QColor("#4CAF50"))
            elif confidence >= 0.6:
                conf_item.setForeground(QtGui.QColor("#FFC107"))
            else:
                conf_item.setForeground(QtGui.QColor("#F44336"))
            conf_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.search_table.setItem(row_index, 4, conf_item)
            
            self.search_table.setItem(row_index, 5, QtWidgets.QTableWidgetItem(row_data["source"]))
        
        self.results_stats.setText(f"Найдено: {len(rows)} записей")

    # ------------------ Настройки ------------------
    def _build_settings_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Боковая панель навигации
        nav_widget = QtWidgets.QWidget()
        nav_widget.setFixedWidth(220)
        nav_widget.setStyleSheet(f"""
            background-color: {self.COLORS['surface_light']};
            border-right: 1px solid {self.COLORS['border']};
        """)
        
        nav_layout = QtWidgets.QVBoxLayout(nav_widget)
        nav_layout.setSpacing(8)
        nav_layout.setContentsMargins(16, 24, 16, 24)
        
        nav_header = QtWidgets.QLabel("⚙️ Настройки")
        nav_header.setStyleSheet("font-size: 16px; font-weight: bold; color: #00B4D8;")
        nav_layout.addWidget(nav_header)
        
        nav_layout.addSpacing(16)
        
        # Кнопки навигации
        self.nav_buttons = []
        
        nav_items = [
            ("🌐 Общие", "general"),
            ("📡 Каналы", "channels"),
            ("🎯 Распознавание", "recognition"),
            ("💾 Хранилище", "storage"),
            ("🔧 Расширенные", "advanced")
        ]
        
        for text, key in nav_items:
            btn = QtWidgets.QPushButton(text)
            btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton {
                    text-align: left;
                    padding: 12px 16px;
                    border: none;
                    border-radius: 6px;
                    font-size: 13px;
                    font-weight: 500;
                    background-color: transparent;
                    color: #CCCCCC;
                }
                QPushButton:hover {
                    background-color: #2D2D3D;
                }
                QPushButton:checked {
                    background-color: #00B4D8;
                    color: white;
                    font-weight: 600;
                }
            """)
            btn.clicked.connect(lambda checked, k=key: self._show_settings_section(k))
            self.nav_buttons.append((btn, key))
            nav_layout.addWidget(btn)
        
        nav_layout.addStretch()
        
        # Кнопка сохранения
        save_all_btn = QtWidgets.QPushButton("💾 Сохранить все настройки")
        save_all_btn.clicked.connect(self._save_all_settings)
        nav_layout.addWidget(save_all_btn)
        
        main_layout.addWidget(nav_widget)

        # Основная область настроек
        self.settings_stack = QtWidgets.QStackedWidget()
        self.settings_stack.setStyleSheet("background-color: transparent;")
        
        # Создание секций
        self.settings_sections = {}
        self.settings_sections["general"] = self._build_general_settings()
        self.settings_sections["channels"] = self._build_channel_settings()
        self.settings_sections["recognition"] = self._build_recognition_settings()
        self.settings_sections["storage"] = self._build_storage_settings()
        self.settings_sections["advanced"] = self._build_advanced_settings()
        
        for key, widget in self.settings_sections.items():
            self.settings_stack.addWidget(widget)
        
        main_layout.addWidget(self.settings_stack, 1)
        
        # Показываем первую секцию
        self._show_settings_section("general")
        
        return widget

    def _show_settings_section(self, section_key):
        """Показывает выбранную секцию настроек."""
        # Обновляем кнопки навигации
        for btn, key in self.nav_buttons:
            btn.setChecked(key == section_key)
        
        # Показываем соответствующую секцию
        index = list(self.settings_sections.keys()).index(section_key)
        self.settings_stack.setCurrentIndex(index)

def _build_general_settings(self):
    """Общие настройки приложения."""
    widget = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(widget)
    layout.setSpacing(16)
    layout.setContentsMargins(24, 24, 24, 24)
    
    # Основные настройки
    general_card = ModernCardWidget()
    general_layout = QtWidgets.QVBoxLayout(general_card)
    
    general_header = QtWidgets.QLabel("Основные параметры")
    general_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #00B4D8;")
    general_layout.addWidget(general_header)
    
    # Виджет настройки сетки
    grid_combo = QtWidgets.QComboBox()  # <-- Создаем комбобокс отдельно
    grid_combo.addItems(self.GRID_VARIANTS)
    grid_combo.setCurrentText(self.settings.get_grid())
    grid_combo.currentTextChanged.connect(
        lambda text: self.settings.save_grid(text)
    )
    
    grid_row = SettingRow(
        "Сетка отображения:",
        grid_combo,  # <-- Передаем готовый комбобокс
        "Расположение видеопотоков на экране"
    )
    general_layout.addWidget(grid_row)
    
    # Язык интерфейса
    lang_combo = QtWidgets.QComboBox()
    lang_combo.addItems(["Русский", "English", "Español"])
    lang_row = SettingRow(
        "Язык интерфейса:",
        lang_combo,
        "Язык пользовательского интерфейса"
    )
    general_layout.addWidget(lang_row)
    
    # Тема оформления
    theme_combo = QtWidgets.QComboBox()
    theme_combo.addItems(["Темная", "Светлая", "Авто"])
    theme_row = SettingRow(
        "Тема оформления:",
        theme_combo,
        "Цветовая схема приложения"
    )
    general_layout.addWidget(theme_row)
    
    layout.addWidget(general_card)
    
    # Настройки уведомлений
    notify_card = ModernCardWidget()
    notify_layout = QtWidgets.QVBoxLayout(notify_card)
    
    notify_header = QtWidgets.QLabel("🔔 Уведомления")
    notify_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #00B4D8;")
    notify_layout.addWidget(notify_header)
    
    # Переключатели уведомлений
    self.notify_sound = ModernToggleSwitch()
    self.notify_popup = ModernToggleSwitch()
    self.notify_email = ModernToggleSwitch()
    
    notify_layout.addWidget(SettingRow("Звуковые уведомления:", self.notify_sound))
    notify_layout.addWidget(SettingRow("Всплывающие окна:", self.notify_popup))
    notify_layout.addWidget(SettingRow("Email уведомления:", self.notify_email))
    
    layout.addWidget(notify_card)
    
    layout.addStretch()
    return widget

    def _build_recognition_settings(self):
        """Настройки распознавания."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)
        
        # Настройки качества
        quality_card = ModernCardWidget()
        quality_layout = QtWidgets.QVBoxLayout(quality_card)
        
        quality_header = QtWidgets.QLabel("🎯 Качество распознавания")
        quality_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #00B4D8;")
        quality_layout.addWidget(quality_header)
        
        # Минимальная уверенность
        self.min_confidence = QtWidgets.QDoubleSpinBox()
        self.min_confidence.setRange(0.0, 1.0)
        self.min_confidence.setSingleStep(0.05)
        self.min_confidence.setDecimals(2)
        quality_layout.addWidget(SettingRow(
            "Минимальная уверенность:",
            self.min_confidence,
            "Минимальный порог уверенности для принятия результата"
        ))
        
        # Количество бестшотов
        self.best_shots = QtWidgets.QSpinBox()
        self.best_shots.setRange(1, 50)
        quality_layout.addWidget(SettingRow(
            "Бестшотов на трек:",
            self.best_shots,
            "Количество кадров для построения консенсуса"
        ))
        
        # Пауза повтора
        self.cooldown = QtWidgets.QSpinBox()
        self.cooldown.setRange(0, 3600)
        self.cooldown.setSuffix(" сек")
        quality_layout.addWidget(SettingRow(
            "Пауза повтора:",
            self.cooldown,
            "Интервал между повторными событиями для одного номера"
        ))
        
        layout.addWidget(quality_card)
        
        # Настройки стран
        country_card = ModernCardWidget()
        country_layout = QtWidgets.QVBoxLayout(country_card)
        
        country_header = QtWidgets.QLabel("🌍 Страны распознавания")
        country_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #00B4D8;")
        country_layout.addWidget(country_header)
        
        # Каталог шаблонов
        dir_layout = QtWidgets.QHBoxLayout()
        self.country_dir = QtWidgets.QLineEdit()
        browse_btn = QtWidgets.QPushButton("Обзор...")
        browse_btn.clicked.connect(self._choose_country_dir)
        dir_layout.addWidget(self.country_dir, 1)
        dir_layout.addWidget(browse_btn)
        
        country_layout.addWidget(SettingRow(
            "Каталог шаблонов:",
            QtWidgets.QWidget(),
            "Папка с конфигурациями стран"
        ))
        country_layout.itemAt(country_layout.count()-1).widget().setLayout(dir_layout)
        
        # Список стран
        self.country_list = QtWidgets.QListWidget()
        self.country_list.setMinimumHeight(200)
        self.country_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        country_layout.addWidget(self.country_list)
        
        # Кнопки управления странами
        country_buttons = QtWidgets.QHBoxLayout()
        reload_btn = QtWidgets.QPushButton("🔄 Обновить")
        reload_btn.clicked.connect(self._reload_country_templates)
        select_all_btn = QtWidgets.QPushButton("✓ Выбрать все")
        select_all_btn.clicked.connect(lambda: self._select_all_countries(True))
        select_none_btn = QtWidgets.QPushButton("✗ Снять все")
        select_none_btn.clicked.connect(lambda: self._select_all_countries(False))
        
        country_buttons.addWidget(reload_btn)
        country_buttons.addWidget(select_all_btn)
        country_buttons.addWidget(select_none_btn)
        country_buttons.addStretch()
        
        country_layout.addLayout(country_buttons)
        
        layout.addWidget(country_card)
        layout.addStretch()
        return widget

    def _select_all_countries(self, select: bool):
        """Выбрать или снять все страны."""
        for i in range(self.country_list.count()):
            item = self.country_list.item(i)
            if item.flags() & QtCore.Qt.ItemIsUserCheckable:
                item.setCheckState(QtCore.Qt.Checked if select else QtCore.Qt.Unchecked)

    def _build_storage_settings(self):
        """Настройки хранилища."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)
        
        # Настройки базы данных
        db_card = ModernCardWidget()
        db_layout = QtWidgets.QVBoxLayout(db_card)
        
        db_header = QtWidgets.QLabel("💾 База данных")
        db_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #00B4D8;")
        db_layout.addWidget(db_header)
        
        # Путь к БД
        db_path_layout = QtWidgets.QHBoxLayout()
        self.db_path = QtWidgets.QLineEdit()
        db_browse_btn = QtWidgets.QPushButton("Обзор...")
        db_browse_btn.clicked.connect(self._choose_db_dir)
        db_path_layout.addWidget(self.db_path, 1)
        db_path_layout.addWidget(db_browse_btn)
        
        db_layout.addWidget(SettingRow(
            "Папка БД:",
            QtWidgets.QWidget(),
            "Каталог для хранения базы данных"
        ))
        db_layout.itemAt(db_layout.count()-1).widget().setLayout(db_path_layout)
        
        # Очистка старых записей
        self.cleanup_days = QtWidgets.QSpinBox()
        self.cleanup_days.setRange(1, 365)
        self.cleanup_days.setSuffix(" дней")
        db_layout.addWidget(SettingRow(
            "Хранить записи:",
            self.cleanup_days,
            "Автоматически удалять записи старше указанного срока"
        ))
        
        # Кнопка очистки
        cleanup_btn = QtWidgets.QPushButton("🧹 Очистить старые записи")
        cleanup_btn.clicked.connect(self._cleanup_old_records)
        db_layout.addWidget(cleanup_btn)
        
        layout.addWidget(db_card)
        
        # Настройки скриншотов
        ss_card = ModernCardWidget()
        ss_layout = QtWidgets.QVBoxLayout(ss_card)
        
        ss_header = QtWidgets.QLabel("📸 Скриншоты")
        ss_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #00B4D8;")
        ss_layout.addWidget(ss_header)
        
        # Путь к скриншотам
        ss_path_layout = QtWidgets.QHBoxLayout()
        self.ss_path = QtWidgets.QLineEdit()
        ss_browse_btn = QtWidgets.QPushButton("Обзор...")
        ss_browse_btn.clicked.connect(self._choose_screenshot_dir)
        ss_path_layout.addWidget(self.ss_path, 1)
        ss_path_layout.addWidget(ss_browse_btn)
        
        ss_layout.addWidget(SettingRow(
            "Папка скриншотов:",
            QtWidgets.QWidget(),
            "Каталог для сохранения изображений"
        ))
        ss_layout.itemAt(ss_layout.count()-1).widget().setLayout(ss_path_layout)
        
        # Качество сжатия
        self.ss_quality = QtWidgets.QSpinBox()
        self.ss_quality.setRange(1, 100)
        self.ss_quality.setSuffix("%")
        ss_layout.addWidget(SettingRow(
            "Качество JPEG:",
            self.ss_quality,
            "Качество сжатия для сохраняемых изображений"
        ))
        
        layout.addWidget(ss_card)
        layout.addStretch()
        return widget

    def _build_advanced_settings(self):
        """Расширенные настройки."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)
        
        # Настройки переподключения
        recon_card = ModernCardWidget()
        recon_layout = QtWidgets.QVBoxLayout(recon_card)
        
        recon_header = QtWidgets.QLabel("🔌 Переподключение")
        recon_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #00B4D8;")
        recon_layout.addWidget(recon_header)
        
        # Переключатели
        self.reconnect_enabled = ModernToggleSwitch()
        self.reconnect_periodic = ModernToggleSwitch()
        
        recon_layout.addWidget(SettingRow(
            "Автопереподключение:",
            self.reconnect_enabled,
            "Автоматически переподключаться при потере сигнала"
        ))
        
        # Таймауты
        self.reconnect_timeout = QtWidgets.QSpinBox()
        self.reconnect_timeout.setRange(1, 300)
        self.reconnect_timeout.setSuffix(" сек")
        recon_layout.addWidget(SettingRow(
            "Таймаут ожидания:",
            self.reconnect_timeout,
            "Время ожидания кадра перед переподключением"
        ))
        
        self.reconnect_interval = QtWidgets.QSpinBox()
        self.reconnect_interval.setRange(1, 300)
        self.reconnect_interval.setSuffix(" сек")
        recon_layout.addWidget(SettingRow(
            "Интервал попыток:",
            self.reconnect_interval,
            "Пауза между попытками переподключения"
        ))
        
        recon_layout.addWidget(SettingRow(
            "Плановое переподключение:",
            self.reconnect_periodic,
            "Переподключаться по расписанию"
        ))
        
        self.reconnect_period = QtWidgets.QSpinBox()
        self.reconnect_period.setRange(1, 1440)
        self.reconnect_period.setSuffix(" мин")
        recon_layout.addWidget(SettingRow(
            "Период переподключения:",
            self.reconnect_period,
            "Интервал для планового переподключения"
        ))
        
        layout.addWidget(recon_card)
        
        # Настройки производительности
        perf_card = ModernCardWidget()
        perf_layout = QtWidgets.QVBoxLayout(perf_card)
        
        perf_header = QtWidgets.QLabel("⚡ Производительность")
        perf_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #00B4D8;")
        perf_layout.addWidget(perf_header)
        
        # Ограничение потоков
        self.thread_limit = QtWidgets.QSpinBox()
        self.thread_limit.setRange(1, 16)
        perf_layout.addWidget(SettingRow(
            "Макс. потоков:",
            self.thread_limit,
            "Максимальное количество рабочих потоков"
        ))
        
        # Размер кэша
        self.cache_size = QtWidgets.QSpinBox()
        self.cache_size.setRange(10, 1000)
        self.cache_size.setSuffix(" MB")
        perf_layout.addWidget(SettingRow(
            "Размер кэша:",
            self.cache_size,
            "Максимальный размер кэша изображений"
        ))
        
        # Частота обновления
        self.update_freq = QtWidgets.QSpinBox()
        self.update_freq.setRange(1, 60)
        self.update_freq.setSuffix(" FPS")
        perf_layout.addWidget(SettingRow(
            "Частота обновления:",
            self.update_freq,
            "Максимальная частота обновления интерфейса"
        ))
        
        layout.addWidget(perf_card)
        
        # Настройки логирования
        log_card = ModernCardWidget()
        log_layout = QtWidgets.QVBoxLayout(log_card)
        
        log_header = QtWidgets.QLabel("📝 Логирование")
        log_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #00B4D8;")
        log_layout.addWidget(log_header)
        
        # Уровень логирования
        self.log_level = QtWidgets.QComboBox()
        self.log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        log_layout.addWidget(SettingRow(
            "Уровень логирования:",
            self.log_level,
            "Детализация записей в логах"
        ))
        
        # Размер лог-файла
        self.log_size = QtWidgets.QSpinBox()
        self.log_size.setRange(1, 100)
        self.log_size.setSuffix(" MB")
        log_layout.addWidget(SettingRow(
            "Макс. размер лога:",
            self.log_size,
            "Максимальный размер лог-файла"
        ))
        
        # Кнопка просмотра логов
        view_logs_btn = QtWidgets.QPushButton("📁 Открыть папку логов")
        view_logs_btn.clicked.connect(self._open_logs_folder)
        log_layout.addWidget(view_logs_btn)
        
        layout.addWidget(log_card)
        layout.addStretch()
        return widget

    def _open_logs_folder(self):
        """Открывает папку с логами."""
        import subprocess
        import platform
        
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        system = platform.system()
        if system == "Windows":
            subprocess.run(["explorer", str(log_dir)])
        elif system == "Darwin":  # macOS
            subprocess.run(["open", str(log_dir)])
        else:  # Linux
            subprocess.run(["xdg-open", str(log_dir)])

    def _cleanup_old_records(self):
        """Очистка старых записей из БД."""
        days = self.cleanup_days.value()
        reply = QtWidgets.QMessageBox.question(
            self, "Подтверждение",
            f"Удалить все записи старше {days} дней?\nЭта операция необратима.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            try:
                from datetime import datetime, timedelta
                cutoff_date = datetime.now() - timedelta(days=days)
                cutoff_str = cutoff_date.strftime("%Y-%m-%d %H:%M:%S")
                
                # Здесь должна быть логика удаления из БД
                # self.db.delete_older_than(cutoff_str)
                
                QtWidgets.QMessageBox.information(
                    self, "Успех",
                    f"Записи старше {days} дней были удалены."
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Ошибка",
                    f"Не удалось очистить записи: {str(e)}"
                )

    def _choose_screenshot_dir(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Выбор папки для скриншотов")
        if directory:
            self.ss_path.setText(directory)

    def _choose_db_dir(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Выбор папки базы данных")
        if directory:
            self.db_path.setText(directory)

    def _choose_country_dir(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Выбор каталога шаблонов номеров")
        if directory:
            self.country_dir.setText(directory)
            self._reload_country_templates()

    def _reload_country_templates(self, enabled: Optional[List[str]] = None) -> None:
        plate_settings = self.settings.get_plate_settings()
        config_dir = self.country_dir.text().strip() or plate_settings.get("config_dir", "config/countries")
        loader = CountryConfigLoader(config_dir)
        loader.ensure_dir()
        available = loader.available_configs()
        enabled_codes = set(enabled or plate_settings.get("enabled_countries", []))

        self.country_list.clear()
        if not available:
            item = QtWidgets.QListWidgetItem("Конфигурации стран не найдены")
            item.setFlags(QtCore.Qt.NoItemFlags)
            self.country_list.addItem(item)
            return

        for cfg in available:
            item = QtWidgets.QListWidgetItem(f"{cfg['code']} — {cfg['name']}")
            item.setData(QtCore.Qt.UserRole, cfg["code"])
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            item.setCheckState(QtCore.Qt.Checked if cfg["code"] in enabled_codes else QtCore.Qt.Unchecked)
            self.country_list.addItem(item)

    def _collect_enabled_countries(self) -> List[str]:
        codes: List[str] = []
        for idx in range(self.country_list.count()):
            item = self.country_list.item(idx)
            if item and item.flags() & QtCore.Qt.ItemIsUserCheckable and item.checkState() == QtCore.Qt.Checked:
                codes.append(str(item.data(QtCore.Qt.UserRole)))
        return codes

    def _save_all_settings(self) -> None:
        """Сохраняет все настройки."""
        try:
            # Общие настройки
            reconnect = {
                "signal_loss": {
                    "enabled": self.reconnect_enabled.isChecked(),
                    "frame_timeout_seconds": int(self.reconnect_timeout.value()),
                    "retry_interval_seconds": int(self.reconnect_interval.value()),
                },
                "periodic": {
                    "enabled": self.reconnect_periodic.isChecked(),
                    "interval_minutes": int(self.reconnect_period.value()),
                },
            }
            self.settings.save_reconnect(reconnect)
            
            # Настройки хранилища
            db_dir = self.db_path.text().strip() or "data/db"
            os.makedirs(db_dir, exist_ok=True)
            self.settings.save_db_dir(db_dir)
            
            screenshot_dir = self.ss_path.text().strip() or "data/screenshots"
            self.settings.save_screenshot_dir(screenshot_dir)
            os.makedirs(screenshot_dir, exist_ok=True)
            
            # Настройки распознавания
            plate_settings = {
                "config_dir": self.country_dir.text().strip() or "config/countries",
                "enabled_countries": self._collect_enabled_countries(),
            }
            os.makedirs(plate_settings["config_dir"], exist_ok=True)
            self.settings.save_plate_settings(plate_settings)
            
            # Обновляем приложение
            self.db = EventDatabase(self.settings.get_db_path())
            self._refresh_events_table()
            self._start_channels()
            
            QtWidgets.QMessageBox.information(self, "Сохранено", "Все настройки успешно сохранены.")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить настройки: {str(e)}")
            
    def _save_all_settings(self) -> None:
        """Сохраняет все настройки из всех секций."""
        try:
            # 1. Сохраняем общие настройки
            reconnect = {
                "signal_loss": {
                    "enabled": self.reconnect_enabled.isChecked(),
                    "frame_timeout_seconds": int(self.reconnect_timeout.value()),
                    "retry_interval_seconds": int(self.reconnect_interval.value()),
                },
                "periodic": {
                    "enabled": self.reconnect_periodic.isChecked(),
                    "interval_minutes": int(self.reconnect_period.value()),
                },
            }
            self.settings.save_reconnect(reconnect)
            
            # 2. Сохраняем настройки хранилища
            db_dir = self.db_path.text().strip() or "data/db"
            os.makedirs(db_dir, exist_ok=True)
            self.settings.save_db_dir(db_dir)
            
            screenshot_dir = self.ss_path.text().strip() or "data/screenshots"
            self.settings.save_screenshot_dir(screenshot_dir)
            os.makedirs(screenshot_dir, exist_ok=True)
            
            # 3. Сохраняем настройки распознавания
            plate_settings = {
                "config_dir": self.country_dir.text().strip() or "config/countries",
                "enabled_countries": self._collect_enabled_countries(),
            }
            os.makedirs(plate_settings["config_dir"], exist_ok=True)
            self.settings.save_plate_settings(plate_settings)
            
            # 4. Сохраняем сетку отображения
            if hasattr(self, 'grid_selector'):
                self.settings.save_grid(self.grid_selector.currentText())
            
            # 5. Обновляем приложение
            self.db = EventDatabase(self.settings.get_db_path())
            self._refresh_events_table()
            self._start_channels()
            
            # 6. Перерисовываем сетку
            if hasattr(self, 'grid_selector'):
                self._draw_grid()
            
            QtWidgets.QMessageBox.information(
                self, 
                "Сохранено", 
                "Все настройки успешно сохранены и применены."
            )
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, 
                "Ошибка", 
                f"Не удалось сохранить настройки: {str(e)}"
            )
    def _load_general_settings(self) -> None:
        """Загружает настройки в интерфейс."""
        reconnect = self.settings.get_reconnect()
        signal_loss = reconnect.get("signal_loss", {})
        periodic = reconnect.get("periodic", {})
        
        # Загрузка настроек переподключения
        self.reconnect_enabled.setChecked(bool(signal_loss.get("enabled", True)))
        self.reconnect_timeout.setValue(int(signal_loss.get("frame_timeout_seconds", 5)))
        self.reconnect_interval.setValue(int(signal_loss.get("retry_interval_seconds", 5)))
        self.reconnect_periodic.setChecked(bool(periodic.get("enabled", False)))
        self.reconnect_period.setValue(int(periodic.get("interval_minutes", 60)))
        
        # Загрузка настроек хранилища
        self.db_path.setText(self.settings.get_db_dir())
        self.ss_path.setText(self.settings.get_screenshot_dir())
        
        # Загрузка настроек распознавания
        plate_settings = self.settings.get_plate_settings()
        self.country_dir.setText(plate_settings.get("config_dir", "config/countries"))
        self._reload_country_templates(plate_settings.get("enabled_countries", []))
        
        # Загрузка других настроек
        self.min_confidence.setValue(float(self.settings.get_min_confidence()))
        self.best_shots.setValue(int(self.settings.get_best_shots()))
        self.cooldown.setValue(int(self.settings.get_cooldown_seconds()))
        
        # Производительность
        self.cache_size.setValue(int(self.MAX_IMAGE_CACHE_BYTES / (1024 * 1024)))
        
        # Настройки по умолчанию
        self.cleanup_days.setValue(30)
        self.ss_quality.setValue(85)
        self.thread_limit.setValue(4)
        self.update_freq.setValue(30)
        self.log_level.setCurrentText("INFO")
        self.log_size.setValue(10)

    # ------------------ Настройки каналов (старая версия, оставлена для совместимости) ------------------
    def _build_channel_settings_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        widget.setStyleSheet("background-color: transparent;")

        left_panel = QtWidgets.QVBoxLayout()
        left_panel.setSpacing(6)
        self.channels_list = QtWidgets.QListWidget()
        self.channels_list.setFixedWidth(180)
        self.channels_list.setStyleSheet(self._get_list_style())
        self.channels_list.currentRowChanged.connect(self._load_channel_form)
        left_panel.addWidget(self.channels_list)

        list_buttons = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("➕ Добавить")
        add_btn.clicked.connect(self._add_channel)
        remove_btn = QtWidgets.QPushButton("🗑️ Удалить")
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

        channel_group = QtWidgets.QGroupBox("📡 Параметры канала")
        channel_group.setStyleSheet(self._get_group_style())
        channel_form = QtWidgets.QFormLayout(channel_group)
        self.channel_name_input = QtWidgets.QLineEdit()
        self.channel_source_input = QtWidgets.QLineEdit()
        channel_form.addRow("Название:", self.channel_name_input)
        channel_form.addRow("Источник:", self.channel_source_input)
        right_panel.addWidget(channel_group)

        recognition_group = QtWidgets.QGroupBox("🎯 Распознавание")
        recognition_group.setStyleSheet(self._get_group_style())
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
        recognition_form.addRow("Мин. уверенность:", self.min_conf_input)
        right_panel.addWidget(recognition_group)

        motion_group = QtWidgets.QGroupBox("⚡ Детектор движения")
        motion_group.setStyleSheet(self._get_group_style())
        motion_form = QtWidgets.QFormLayout(motion_group)
        self.detection_mode_input = QtWidgets.QComboBox()
        self.detection_mode_input.addItem("Постоянное", "continuous")
        self.detection_mode_input.addItem("Детектор движения", "motion")
        motion_form.addRow("Режим:", self.detection_mode_input)

        self.detector_stride_input = QtWidgets.QSpinBox()
        self.detector_stride_input.setRange(1, 12)
        self.detector_stride_input.setToolTip(
            "Запускать YOLO на каждом N-м кадре в зоне распознавания, чтобы снизить нагрузку"
        )
        motion_form.addRow("Шаг инференса:", self.detector_stride_input)

        self.motion_threshold_input = QtWidgets.QDoubleSpinBox()
        self.motion_threshold_input.setRange(0.0, 1.0)
        self.motion_threshold_input.setDecimals(3)
        self.motion_threshold_input.setSingleStep(0.005)
        self.motion_threshold_input.setToolTip("Порог чувствительности по площади изменения внутри ROI")
        motion_form.addRow("Порог движения:", self.motion_threshold_input)

        self.motion_stride_input = QtWidgets.QSpinBox()
        self.motion_stride_input.setRange(1, 30)
        self.motion_stride_input.setToolTip("Обрабатывать каждый N-й кадр для поиска движения")
        motion_form.addRow("Частота анализа:", self.motion_stride_input)

        self.motion_activation_frames_input = QtWidgets.QSpinBox()
        self.motion_activation_frames_input.setRange(1, 60)
        self.motion_activation_frames_input.setToolTip("Сколько кадров подряд должно быть движение, чтобы включить распознавание")
        motion_form.addRow("Активация (кадры):", self.motion_activation_frames_input)

        self.motion_release_frames_input = QtWidgets.QSpinBox()
        self.motion_release_frames_input.setRange(1, 120)
        self.motion_release_frames_input.setToolTip("Сколько кадров без движения нужно, чтобы остановить распознавание")
        motion_form.addRow("Деактивация (кадры):", self.motion_release_frames_input)
        right_panel.addWidget(motion_group)

        roi_group = QtWidgets.QGroupBox("🎯 Зона распознавания")
        roi_group.setStyleSheet(self._get_group_style())
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
        refresh_btn = QtWidgets.QPushButton("🔄 Обновить кадр")
        refresh_btn.clicked.connect(self._refresh_preview_frame)
        roi_layout.addWidget(refresh_btn, 4, 0, 1, 2)
        roi_group.setLayout(roi_layout)
        right_panel.addWidget(roi_group)

        save_btn = QtWidgets.QPushButton("💾 Сохранить канал")
        save_btn.clicked.connect(self._save_channel)
        right_panel.addWidget(save_btn)
        right_panel.addStretch()

        layout.addLayout(right_panel, 2)

        self._load_general_settings()
        self._reload_channels_list()
        return widget

    def _get_group_style(self):
        return f"""
            QGroupBox {{
                background-color: {self.COLORS['surface']};
                color: {self.COLORS['text_primary']};
                border: 2px solid {self.COLORS['border']};
                border-radius: 8px;
                padding-top: 12px;
                margin-top: 8px;
                font-weight: bold;
                font-size: 13px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px 0 8px;
                background-color: {self.COLORS['surface']};
            }}
        """

    def _get_list_style(self):
        return f"""
            QListWidget {{
                background-color: {self.COLORS['surface_light']};
                color: {self.COLORS['text_primary']};
                border: 1px solid {self.COLORS['border']};
                border-radius: 6px;
                padding: 4px;
            }}
            QListWidget::item {{
                padding: 8px;
                border-radius: 4px;
                margin: 2px;
            }}
            QListWidget::item:selected {{
                background-color: {self.COLORS['primary']};
                color: white;
            }}
            QListWidget::item:hover {{
                background-color: {self.COLORS['surface']};
            }}
        """

    def _reload_channels_list(self) -> None:
        self.channels_list.clear()
        for channel in self.settings.get_channels():
            self.channels_list.addItem(channel.get("name", "Канал"))
        if self.channels_list.count():
            self.channels_list.setCurrentRow(0)

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
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._stop_workers()
        
        # Сохраняем размер и положение окна
        self.settings.save_window_state({
            "geometry": self.saveGeometry().toHex().data().decode(),
            "state": self.saveState().toHex().data().decode()
        })
        
        event.accept()
