#!/usr/bin/env python3
# /anpr/ui/main_window.py
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import psutil
from collections import OrderedDict
from PyQt5 import QtCore, QtGui, QtWidgets

from anpr.postprocessing.country_config import CountryConfigLoader
from anpr.workers.channel_worker import ChannelWorker
from anpr.infrastructure.logging_manager import get_logger
from anpr.infrastructure.settings_manager import SettingsManager
from anpr.infrastructure.storage import EventDatabase

logger = get_logger(__name__)


# ============================================================================
# СТИЛИ И ТЕМА
# ============================================================================

class ModernTheme:
    """Современная тема с неоновыми акцентами и плавными анимациями."""
    
    # Цветовая палитра
    COLORS = {
        # Основные цвета
        "background": "#0f1419",
        "surface": "#1a2029",
        "surface_light": "#252d38",
        "border": "#2a3441",
        
        # Акцентные цвета (неон)
        "primary": "#00e5ff",
        "primary_light": "#4df7ff",
        "primary_dark": "#00a3cc",
        "success": "#00ff9d",
        "warning": "#ffb74d",
        "danger": "#ff5252",
        "info": "#6c8eff",
        
        # Текст
        "text_primary": "#ffffff",
        "text_secondary": "#b0bac5",
        "text_disabled": "#667788",
        
        # Градиенты
        "gradient_primary": "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00e5ff, stop:1 #6c8eff)",
        "gradient_success": "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00ff9d, stop:1 #00e5ff)",
        "gradient_warning": "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ffb74d, stop:1 #ff5252)",
    }
    
    # Стили компонентов
    STYLES = {
        "main_window": f"""
            QMainWindow {{
                background-color: {COLORS["background"]};
                color: {COLORS["text_primary"]};
            }}
        """,
        
        "button_primary": f"""
            QPushButton {{
                background: {COLORS["gradient_primary"]};
                color: {COLORS["text_primary"]};
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-weight: 600;
                font-size: 14px;
                min-height: 40px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4df7ff, stop:1 #8a9eff);
            }}
            QPushButton:pressed {{
                background: {COLORS["primary_dark"]};
            }}
            QPushButton:disabled {{
                background: {COLORS["surface_light"]};
                color: {COLORS["text_disabled"]};
            }}
        """,
        
        "button_secondary": f"""
            QPushButton {{
                background: transparent;
                color: {COLORS["primary"]};
                border: 2px solid {COLORS["primary"]};
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: 600;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background: rgba(0, 229, 255, 0.1);
                border-color: {COLORS["primary_light"]};
                color: {COLORS["primary_light"]};
            }}
        """,
        
        "tab_widget": f"""
            QTabWidget::pane {{
                border: 1px solid {COLORS["border"]};
                border-radius: 12px;
                background: {COLORS["surface"]};
                margin-top: 10px;
            }}
            
            QTabBar::tab {{
                background: transparent;
                color: {COLORS["text_secondary"]};
                padding: 12px 24px;
                margin-right: 4px;
                border: none;
                font-weight: 500;
                font-size: 14px;
                border-radius: 8px 8px 0 0;
            }}
            
            QTabBar::tab:selected {{
                background: {COLORS["surface_light"]};
                color: {COLORS["primary"]};
                font-weight: 600;
            }}
            
            QTabBar::tab:hover:!selected {{
                background: {COLORS["surface_light"]};
                color: {COLORS["text_primary"]};
            }}
            
            QTabBar::tab:first {{
                margin-left: 10px;
            }}
        """,
        
        "table": f"""
            QTableWidget {{
                background: {COLORS["surface"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 8px;
                gridline-color: {COLORS["border"]};
                color: {COLORS["text_primary"]};
                selection-background-color: rgba(0, 229, 255, 0.2);
                selection-color: {COLORS["primary"]};
                font-size: 13px;
            }}
            
            QTableWidget::item {{
                padding: 8px;
                border-bottom: 1px solid {COLORS["border"]};
            }}
            
            QTableWidget::item:selected {{
                background-color: rgba(0, 229, 255, 0.3);
                color: {COLORS["primary_light"]};
            }}
            
            QHeaderView::section {{
                background-color: {COLORS["surface_light"]};
                color: {COLORS["text_primary"]};
                padding: 12px 8px;
                border: none;
                border-right: 1px solid {COLORS["border"]};
                font-weight: 600;
                font-size: 13px;
            }}
            
            QHeaderView::section:last {{
                border-right: none;
            }}
            
            QScrollBar:vertical {{
                background: {COLORS["surface"]};
                width: 10px;
                border-radius: 5px;
            }}
            
            QScrollBar::handle:vertical {{
                background: {COLORS["primary"]};
                border-radius: 5px;
                min-height: 20px;
            }}
            
            QScrollBar::handle:vertical:hover {{
                background: {COLORS["primary_light"]};
            }}
        """,
        
        "group_box": f"""
            QGroupBox {{
                background: {COLORS["surface"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 12px;
                padding: 20px;
                margin-top: 10px;
                color: {COLORS["text_primary"]};
                font-weight: 600;
                font-size: 14px;
            }}
            
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px;
                color: {COLORS["primary"]};
            }}
        """,
        
        "input": f"""
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QDateTimeEdit, QTextEdit {{
                background: {COLORS["surface_light"]};
                border: 2px solid {COLORS["border"]};
                border-radius: 8px;
                padding: 10px 12px;
                color: {COLORS["text_primary"]};
                font-size: 14px;
                selection-background-color: {COLORS["primary"]};
                selection-color: {COLORS["text_primary"]};
            }}
            
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, 
            QComboBox:focus, QDateTimeEdit:focus, QTextEdit:focus {{
                border-color: {COLORS["primary"]};
                background: {COLORS["surface_light"]};
            }}
            
            QComboBox::drop-down {{
                border: none;
                width: 30px;
            }}
            
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid {COLORS["primary"]};
            }}
            
            QComboBox QAbstractItemView {{
                background: {COLORS["surface_light"]};
                border: 1px solid {COLORS["primary"]};
                color: {COLORS["text_primary"]};
                selection-background-color: {COLORS["primary"]};
                selection-color: {COLORS["text_primary"]};
            }}
        """,
        
        "checkbox": f"""
            QCheckBox {{
                color: {COLORS["text_primary"]};
                spacing: 8px;
                font-size: 14px;
            }}
            
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border: 2px solid {COLORS["border"]};
                border-radius: 4px;
            }}
            
            QCheckBox::indicator:checked {{
                background: {COLORS["primary"]};
                border-color: {COLORS["primary"]};
                image: url(:/icons/check.svg);
            }}
            
            QCheckBox::indicator:hover {{
                border-color: {COLORS["primary_light"]};
            }}
        """,
        
        "slider": f"""
            QSlider::groove:horizontal {{
                height: 4px;
                background: {COLORS["border"]};
                border-radius: 2px;
            }}
            
            QSlider::sub-page:horizontal {{
                background: {COLORS["primary"]};
                border-radius: 2px;
            }}
            
            QSlider::handle:horizontal {{
                background: {COLORS["primary"]};
                width: 18px;
                height: 18px;
                margin: -7px 0;
                border-radius: 9px;
            }}
            
            QSlider::handle:horizontal:hover {{
                background: {COLORS["primary_light"]};
                width: 22px;
                height: 22px;
                border-radius: 11px;
            }}
        """,
        
        "list_widget": f"""
            QListWidget {{
                background: {COLORS["surface_light"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 8px;
                color: {COLORS["text_primary"]};
                outline: none;
                font-size: 14px;
            }}
            
            QListWidget::item {{
                padding: 12px 15px;
                border-bottom: 1px solid {COLORS["border"]};
                background: transparent;
            }}
            
            QListWidget::item:selected {{
                background: rgba(0, 229, 255, 0.2);
                color: {COLORS["primary"]};
                border-left: 4px solid {COLORS["primary"]};
            }}
            
            QListWidget::item:hover {{
                background: rgba(255, 255, 255, 0.05);
            }}
        """,
        
        "status_bar": f"""
            QStatusBar {{
                background: {COLORS["surface_light"]};
                color: {COLORS["text_secondary"]};
                border-top: 1px solid {COLORS["border"]};
                font-size: 12px;
            }}
        """,
        
        "card": f"""
            QFrame {{
                background: {COLORS["surface"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 12px;
                padding: 20px;
            }}
        """,
        
        "tooltip": f"""
            QToolTip {{
                background: {COLORS["surface_light"]};
                color: {COLORS["text_primary"]};
                border: 1px solid {COLORS["primary"]};
                border-radius: 6px;
                padding: 8px;
                font-size: 12px;
            }}
        """
    }


class ModernButton(QtWidgets.QPushButton):
    """Современная кнопка с анимацией наведения."""
    
    def __init__(self, text="", parent=None, style="primary"):
        super().__init__(text, parent)
        self.style = style
        self._animation = QtCore.QPropertyAnimation(self, b"geometry")
        self._animation.setDuration(200)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self._apply_style()
    
    def _apply_style(self):
        if self.style == "primary":
            self.setStyleSheet(ModernTheme.STYLES["button_primary"])
        elif self.style == "secondary":
            self.setStyleSheet(ModernTheme.STYLES["button_secondary"])
    
    def enterEvent(self, event):
        self._animate_hover(True)
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        self._animate_hover(False)
        super().leaveEvent(event)
    
    def _animate_hover(self, hover):
        if hover:
            self._animation.stop()
            self._animation.setStartValue(self.geometry())
            self._animation.setEndValue(self.geometry().adjusted(-2, -2, 4, 4))
            self._animation.start()
        else:
            self._animation.stop()
            self._animation.setStartValue(self.geometry())
            self._animation.setEndValue(self.geometry().adjusted(2, 2, -4, -4))
            self._animation.start()


class ModernCard(QtWidgets.QFrame):
    """Карточка для группировки элементов с тенью."""
    
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setStyleSheet(ModernTheme.STYLES["card"])
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setFrameShadow(QtWidgets.QFrame.Raised)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        if title:
            title_label = QtWidgets.QLabel(title)
            title_label.setStyleSheet(f"""
                QLabel {{
                    color: {ModernTheme.COLORS["primary"]};
                    font-weight: 600;
                    font-size: 16px;
                    margin-bottom: 5px;
                }}
            """)
            layout.addWidget(title_label)
        
        self.content_layout = QtWidgets.QVBoxLayout()
        layout.addLayout(self.content_layout)
    
    def add_widget(self, widget):
        self.content_layout.addWidget(widget)
    
    def add_layout(self, layout):
        self.content_layout.addLayout(layout)


class SettingsGroup(QtWidgets.QWidget):
    """Группа настроек с заголовком и описанием."""
    
    def __init__(self, title, description="", parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 20)
        layout.setSpacing(10)
        
        # Заголовок
        title_label = QtWidgets.QLabel(title)
        title_label.setStyleSheet(f"""
            QLabel {{
                color: {ModernTheme.COLORS["primary"]};
                font-weight: 600;
                font-size: 16px;
                padding-bottom: 5px;
                border-bottom: 2px solid {ModernTheme.COLORS["primary"]};
            }}
        """)
        layout.addWidget(title_label)
        
        # Описание
        if description:
            desc_label = QtWidgets.QLabel(description)
            desc_label.setStyleSheet(f"""
                QLabel {{
                    color: {ModernTheme.COLORS["text_secondary"]};
                    font-size: 13px;
                    padding: 5px 0;
                }}
            """)
            desc_label.setWordWrap(True)
            layout.addWidget(desc_label)
        
        # Контейнер для настроек
        self.settings_container = QtWidgets.QWidget()
        self.settings_layout = QtWidgets.QFormLayout(self.settings_container)
        self.settings_layout.setContentsMargins(10, 10, 10, 10)
        self.settings_layout.setHorizontalSpacing(20)
        self.settings_layout.setVerticalSpacing(12)
        
        layout.addWidget(self.settings_container)
    
    def add_setting(self, label, widget, tooltip=""):
        """Добавить настройку с меткой и виджетом."""
        label_widget = QtWidgets.QLabel(label)
        label_widget.setStyleSheet(f"""
            QLabel {{
                color: {ModernTheme.COLORS["text_primary"]};
                font-weight: 500;
                font-size: 14px;
                min-width: 180px;
            }}
        """)
        
        if tooltip:
            label_widget.setToolTip(tooltip)
            widget.setToolTip(tooltip)
        
        self.settings_layout.addRow(label_widget, widget)


# ============================================================================
# ОСНОВНЫЕ КОМПОНЕНТЫ
# ============================================================================

class PixmapPool:
    """Пул для повторного использования QPixmap."""
    
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
    """Виджет отображения видео с современным дизайном."""
    
    def __init__(self, name: str, pixmap_pool: Optional[PixmapPool]) -> None:
        super().__init__()
        self.name = name
        self._pixmap_pool = pixmap_pool
        self._current_pixmap: Optional[QtGui.QPixmap] = None
        
        # Стиль виджета
        self.setStyleSheet(f"""
            QWidget {{
                background: {ModernTheme.COLORS["surface"]};
                border: 2px solid {ModernTheme.COLORS["border"]};
                border-radius: 12px;
            }}
        """)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        
        # Заголовок канала
        header = QtWidgets.QWidget()
        header.setStyleSheet(f"""
            QWidget {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 {ModernTheme.COLORS["surface_light"]}, 
                    stop:1 {ModernTheme.COLORS["surface"]});
                border-radius: 10px 10px 0 0;
                padding: 8px 12px;
            }}
        """)
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(10, 5, 10, 5)
        
        self.name_label = QtWidgets.QLabel(name)
        self.name_label.setStyleSheet(f"""
            QLabel {{
                color: {ModernTheme.COLORS["primary"]};
                font-weight: 600;
                font-size: 14px;
            }}
        """)
        
        self.status_indicator = QtWidgets.QLabel("●")
        self.status_indicator.setStyleSheet(f"""
            QLabel {{
                color: {ModernTheme.COLORS["danger"]};
                font-size: 16px;
                padding-right: 5px;
            }}
        """)
        
        header_layout.addWidget(self.status_indicator)
        header_layout.addWidget(self.name_label)
        header_layout.addStretch()
        
        # Индикаторы
        self.motion_indicator = QtWidgets.QLabel("ДВИЖЕНИЕ")
        self.motion_indicator.setStyleSheet(f"""
            QLabel {{
                background: rgba(255, 82, 82, 0.9);
                color: white;
                font-weight: bold;
                font-size: 11px;
                padding: 4px 8px;
                border-radius: 6px;
                margin-right: 5px;
            }}
        """)
        self.motion_indicator.hide()
        
        self.recognition_indicator = QtWidgets.QLabel("РАСПОЗНАНИЕ")
        self.recognition_indicator.setStyleSheet(f"""
            QLabel {{
                background: rgba(0, 229, 255, 0.9);
                color: white;
                font-weight: bold;
                font-size: 11px;
                padding: 4px 8px;
                border-radius: 6px;
            }}
        """)
        self.recognition_indicator.hide()
        
        header_layout.addWidget(self.motion_indicator)
        header_layout.addWidget(self.recognition_indicator)
        
        layout.addWidget(header)
        
        # Область видео
        self.video_label = QtWidgets.QLabel("⏸ НЕТ СИГНАЛА")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setStyleSheet(f"""
            QLabel {{
                background: {ModernTheme.COLORS["background"]};
                color: {ModernTheme.COLORS["text_disabled"]};
                font-weight: 500;
                font-size: 13px;
                border-radius: 0 0 10px 10px;
            }}
        """)
        self.video_label.setMinimumSize(280, 180)
        self.video_label.setScaledContents(False)
        self.video_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        
        layout.addWidget(self.video_label)
        
        # Панель информации
        self.info_panel = QtWidgets.QWidget()
        self.info_panel.setStyleSheet(f"""
            QWidget {{
                background: rgba(0, 0, 0, 0.7);
                border-radius: 0 0 8px 8px;
                margin: 2px;
            }}
        """)
        self.info_panel.hide()
        
        info_layout = QtWidgets.QHBoxLayout(self.info_panel)
        info_layout.setContentsMargins(10, 6, 10, 6)
        
        self.plate_label = QtWidgets.QLabel("—")
        self.plate_label.setStyleSheet(f"""
            QLabel {{
                color: {ModernTheme.COLORS["success"]};
                font-weight: bold;
                font-size: 13px;
            }}
        """)
        
        self.confidence_label = QtWidgets.QLabel("")
        self.confidence_label.setStyleSheet(f"""
            QLabel {{
                color: {ModernTheme.COLORS["primary"]};
                font-weight: 500;
                font-size: 11px;
                padding: 2px 6px;
                background: rgba(0, 229, 255, 0.2);
                border-radius: 4px;
            }}
        """)
        
        info_layout.addWidget(self.plate_label)
        info_layout.addStretch()
        info_layout.addWidget(self.confidence_label)
        
        self.video_label.setProperty("info_panel", self.info_panel)
        self.info_panel.setParent(self.video_label)
    
    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        rect = self.video_label.contentsRect()
        self.info_panel.setGeometry(
            rect.left(), 
            rect.bottom() - 35, 
            rect.width(), 
            35
        )
    
    def set_pixmap(self, pixmap: QtGui.QPixmap) -> None:
        if self._pixmap_pool and self._current_pixmap is not None:
            self._pixmap_pool.release(self._current_pixmap)
        self._current_pixmap = pixmap
        self.video_label.setPixmap(pixmap)
        self.video_label.setText("")
    
    def set_status(self, active: bool, text: str = "") -> None:
        """Установить статус канала."""
        if active:
            self.status_indicator.setStyleSheet(f"""
                QLabel {{
                    color: {ModernTheme.COLORS["success"]};
                    font-size: 16px;
                    padding-right: 5px;
                }}
            """)
        else:
            self.status_indicator.setStyleSheet(f"""
                QLabel {{
                    color: {ModernTheme.COLORS["danger"]};
                    font-size: 16px;
                    padding-right: 5px;
                }}
            """)
        
        if text:
            self.video_label.setText(f"⏸ {text}")
        else:
            self.video_label.setText("")
    
    def set_motion_active(self, active: bool) -> None:
        self.motion_indicator.setVisible(active)
    
    def set_recognition_active(self, active: bool) -> None:
        self.recognition_indicator.setVisible(active)
    
    def set_plate_info(self, plate: str, confidence: float = 0.0) -> None:
        """Установить информацию о распознанном номере."""
        if plate and plate != "—":
            self.plate_label.setText(plate)
            self.confidence_label.setText(f"{confidence:.1%}")
            self.info_panel.show()
        else:
            self.info_panel.hide()


class EventDetailView(QtWidgets.QWidget):
    """Детальное отображение события с современным дизайном."""
    
    def __init__(self) -> None:
        super().__init__()
        self.setStyleSheet(ModernTheme.STYLES["card"])
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Заголовок
        header = QtWidgets.QLabel("📋 ДЕТАЛИ СОБЫТИЯ")
        header.setStyleSheet(f"""
            QLabel {{
                color: {ModernTheme.COLORS["primary"]};
                font-weight: 600;
                font-size: 16px;
                padding-bottom: 10px;
                border-bottom: 2px solid {ModernTheme.COLORS["primary"]};
            }}
        """)
        layout.addWidget(header)
        
        # Двухколоночный макет
        main_layout = QtWidgets.QHBoxLayout()
        main_layout.setSpacing(20)
        
        # Левая колонка - изображения
        left_column = QtWidgets.QVBoxLayout()
        left_column.setSpacing(15)
        
        self.frame_preview = self._build_image_card("📸 КАДР РАСПОЗНАВАНИЯ")
        self.plate_preview = self._build_image_card("🚗 ОБЛАСТЬ НОМЕРА")
        
        left_column.addWidget(self.frame_preview)
        left_column.addWidget(self.plate_preview)
        
        # Правая колонка - метаданные
        right_column = QtWidgets.QVBoxLayout()
        
        metadata_card = ModernCard("📊 МЕТАДАННЫЕ")
        metadata_layout = QtWidgets.QFormLayout()
        metadata_layout.setContentsMargins(10, 10, 10, 10)
        metadata_layout.setVerticalSpacing(12)
        metadata_layout.setHorizontalSpacing(20)
        
        self.metadata_labels = {}
        fields = [
            ("Дата/время", "timestamp"),
            ("Канал", "channel"),
            ("Гос. номер", "plate"),
            ("Страна", "country"),
            ("Уверенность", "confidence"),
            ("Источник", "source"),
            ("Формат", "format"),
            ("Статус", "validated")
        ]
        
        for label, key in fields:
            label_widget = QtWidgets.QLabel(f"{label}:")
            label_widget.setStyleSheet(f"""
                QLabel {{
                    color: {ModernTheme.COLORS["text_secondary"]};
                    font-weight: 500;
                    font-size: 13px;
                }}
            """)
            
            value_widget = QtWidgets.QLabel("—")
            value_widget.setStyleSheet(f"""
                QLabel {{
                    color: {ModernTheme.COLORS["text_primary"]};
                    font-weight: 500;
                    font-size: 13px;
                    padding: 4px 8px;
                    background: {ModernTheme.COLORS["surface_light"]};
                    border-radius: 6px;
                }}
            """)
            
            metadata_layout.addRow(label_widget, value_widget)
            self.metadata_labels[key] = value_widget
        
        metadata_card.setLayout(metadata_layout)
        right_column.addWidget(metadata_card)
        right_column.addStretch()
        
        main_layout.addLayout(left_column, 2)
        main_layout.addLayout(right_column, 1)
        
        layout.addLayout(main_layout)
    
    def _build_image_card(self, title: str) -> ModernCard:
        """Создать карточку для изображения."""
        card = ModernCard(title)
        
        image_container = QtWidgets.QWidget()
        image_container.setMinimumHeight(180)
        image_container.setStyleSheet(f"""
            QWidget {{
                background: {ModernTheme.COLORS["background"]};
                border-radius: 8px;
            }}
        """)
        
        image_layout = QtWidgets.QVBoxLayout(image_container)
        image_layout.setAlignment(QtCore.Qt.AlignCenter)
        
        self.image_label = QtWidgets.QLabel("🖼 ИЗОБРАЖЕНИЕ НЕДОСТУПНО")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setStyleSheet(f"""
            QLabel {{
                color: {ModernTheme.COLORS["text_disabled"]};
                font-size: 13px;
            }}
        """)
        self.image_label.setProperty("for_card", card)
        
        image_layout.addWidget(self.image_label)
        card.add_widget(image_container)
        
        return card
    
    def set_event(self, event: Optional[Dict] = None) -> None:
        """Установить событие для отображения."""
        if not event:
            # Сброс всех полей
            for label in self.metadata_labels.values():
                label.setText("—")
            
            for card in [self.frame_preview, self.plate_preview]:
                image_label = card.findChild(QtWidgets.QLabel)
                if image_label:
                    image_label.setText("🖼 ИЗОБРАЖЕНИЕ НЕДОСТУПНО")
                    image_label.setPixmap(QtGui.QPixmap())
            return
        
        # Установка метаданных
        fields = {
            "timestamp": self._format_timestamp(event.get("timestamp", "")),
            "channel": event.get("channel", "—"),
            "plate": event.get("plate", "—") or "Не распознан",
            "country": event.get("country", "—"),
            "confidence": f"{event.get('confidence', 0):.1%}" if event.get("confidence") else "—",
            "source": event.get("source", "—"),
            "format": event.get("format", "—"),
            "validated": "✅ ВАЛИДНЫЙ" if event.get("validated") else "❌ НЕВАЛИДНЫЙ"
        }
        
        for key, value in fields.items():
            if key in self.metadata_labels:
                self.metadata_labels[key].setText(value)
                
                # Цветовое кодирование для статуса
                if key == "validated":
                    color = ModernTheme.COLORS["success"] if event.get("validated") else ModernTheme.COLORS["danger"]
                    self.metadata_labels[key].setStyleSheet(f"""
                        QLabel {{
                            color: white;
                            font-weight: 500;
                            font-size: 13px;
                            padding: 4px 8px;
                            background: {color};
                            border-radius: 6px;
                        }}
                    """)
    
    @staticmethod
    def _format_timestamp(value: str) -> str:
        if not value:
            return "—"
        try:
            dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
            return dt.strftime("%d.%m.%Y %H:%M:%S")
        except:
            return value


class SettingsSidebar(QtWidgets.QWidget):
    """Боковая панель навигации по настройкам."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(220)
        self.setStyleSheet(f"""
            QWidget {{
                background: {ModernTheme.COLORS["surface"]};
                border-right: 1px solid {ModernTheme.COLORS["border"]};
            }}
        """)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 20, 0, 20)
        layout.setSpacing(0)
        
        # Заголовок
        title = QtWidgets.QLabel("⚙️ НАСТРОЙКИ")
        title.setStyleSheet(f"""
            QLabel {{
                color: {ModernTheme.COLORS["primary"]};
                font-weight: 600;
                font-size: 16px;
                padding: 15px 20px;
                border-bottom: 1px solid {ModernTheme.COLORS["border"]};
            }}
        """)
        layout.addWidget(title)
        
        # Список категорий
        self.category_list = QtWidgets.QListWidget()
        self.category_list.setStyleSheet(ModernTheme.STYLES["list_widget"])
        self.category_list.setFocusPolicy(QtCore.Qt.NoFocus)
        
        categories = [
            ("🎯 ОБЩИЕ", "Основные настройки приложения"),
            ("📷 КАНАЛЫ", "Настройки видеопотоков"),
            ("🔍 РАСПОЗНАВАНИЕ", "Параметры OCR и детекции"),
            ("🚗 ДЕТЕКЦИЯ ДВИЖЕНИЯ", "Настройки детектора движения"),
            ("💾 ХРАНИЛИЩЕ", "База данных и файлы"),
            ("🎨 ВНЕШНИЙ ВИД", "Темы и интерфейс")
        ]
        
        for icon_text, description in categories:
            item_widget = QtWidgets.QWidget()
            item_layout = QtWidgets.QVBoxLayout(item_widget)
            item_layout.setContentsMargins(15, 12, 15, 12)
            item_layout.setSpacing(4)
            
            text_label = QtWidgets.QLabel(icon_text)
            text_label.setStyleSheet(f"""
                QLabel {{
                    color: {ModernTheme.COLORS["text_primary"]};
                    font-weight: 500;
                    font-size: 14px;
                }}
            """)
            
            desc_label = QtWidgets.QLabel(description)
            desc_label.setStyleSheet(f"""
                QLabel {{
                    color: {ModernTheme.COLORS["text_secondary"]};
                    font-size: 12px;
                }}
            """)
            desc_label.setWordWrap(True)
            
            item_layout.addWidget(text_label)
            item_layout.addWidget(desc_label)
            
            item = QtWidgets.QListWidgetItem(self.category_list)
            item.setSizeHint(item_widget.sizeHint())
            self.category_list.addItem(item)
            self.category_list.setItemWidget(item, item_widget)
        
        layout.addWidget(self.category_list, 1)
        
        # Кнопка сохранения
        save_button = ModernButton("💾 СОХРАНИТЬ ВСЕ", style="primary")
        save_button.clicked.connect(parent._save_all_settings if parent else None)
        layout.addWidget(save_button)


# ============================================================================
# ГЛАВНОЕ ОКНО
# ============================================================================

class MainWindow(QtWidgets.QMainWindow):
    """Главное окно приложения с современным дизайном."""
    
    GRID_VARIANTS = ["1x1", "1x2", "2x2", "2x3", "3x3", "3x4"]
    MAX_IMAGE_CACHE = 200
    MAX_IMAGE_CACHE_BYTES = 256 * 1024 * 1024
    
    def __init__(self, settings: Optional[SettingsManager] = None) -> None:
        super().__init__()
        self.settings = settings or SettingsManager()
        
        # Инициализация UI
        self._init_ui()
        self._setup_storage()
        self._setup_workers()
        
        # Загрузка данных
        self._refresh_events_table()
        self._start_channels()
    
    def _init_ui(self):
        """Инициализация пользовательского интерфейса."""
        self.setWindowTitle("🚗 ANPR DESKTOP v2.0")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet(ModernTheme.STYLES["main_window"])
        
        # Центральный виджет
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # Главный лейаут
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Верхняя панель
        self._create_top_bar(main_layout)
        
        # Основная область с вкладками
        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setStyleSheet(ModernTheme.STYLES["tab_widget"])
        
        # Вкладки
        self._create_monitoring_tab()
        self._create_search_tab()
        self._create_settings_tab()
        
        main_layout.addWidget(self.tab_widget, 1)
        
        # Статус бар
        self._create_status_bar()
        
        # Системный мониторинг
        self._start_system_monitoring()
    
    def _create_top_bar(self, parent_layout):
        """Создать верхнюю панель управления."""
        top_bar = QtWidgets.QWidget()
        top_bar.setMaximumHeight(60)
        
        layout = QtWidgets.QHBoxLayout(top_bar)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Логотип и заголовок
        logo_label = QtWidgets.QLabel("🚗 ANPR DESKTOP")
        logo_label.setStyleSheet(f"""
            QLabel {{
                color: {ModernTheme.COLORS["primary"]};
                font-size: 22px;
                font-weight: 700;
                padding-left: 10px;
            }}
        """)
        
        # Статистика
        stats_widget = QtWidgets.QWidget()
        stats_layout = QtWidgets.QHBoxLayout(stats_widget)
        stats_layout.setSpacing(15)
        
        self.channels_stat = self._create_stat_widget("📷 КАНАЛЫ", "0/0")
        self.events_stat = self._create_stat_widget("📊 СОБЫТИЯ", "0")
        self.fps_stat = self._create_stat_widget("⚡ FPS", "0")
        
        stats_layout.addWidget(self.channels_stat)
        stats_layout.addWidget(self.events_stat)
        stats_layout.addWidget(self.fps_stat)
        stats_layout.addStretch()
        
        # Кнопки управления
        buttons_widget = QtWidgets.QWidget()
        buttons_layout = QtWidgets.QHBoxLayout(buttons_widget)
        buttons_layout.setSpacing(10)
        
        self.start_btn = ModernButton("▶️ ЗАПУСК", style="primary")
        self.stop_btn = ModernButton("⏹ СТОП", style="secondary")
        self.stop_btn.setEnabled(False)
        
        buttons_layout.addWidget(self.start_btn)
        buttons_layout.addWidget(self.stop_btn)
        
        layout.addWidget(logo_label)
        layout.addStretch()
        layout.addWidget(stats_widget)
        layout.addStretch()
        layout.addWidget(buttons_widget)
        
        parent_layout.addWidget(top_bar)
    
    def _create_stat_widget(self, title: str, value: str) -> QtWidgets.QWidget:
        """Создать виджет статистики."""
        widget = QtWidgets.QWidget()
        widget.setStyleSheet(f"""
            QWidget {{
                background: {ModernTheme.COLORS["surface"]};
                border: 1px solid {ModernTheme.COLORS["border"]};
                border-radius: 8px;
                padding: 8px 12px;
            }}
        """)
        
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)
        
        title_label = QtWidgets.QLabel(title)
        title_label.setStyleSheet(f"""
            QLabel {{
                color: {ModernTheme.COLORS["text_secondary"]};
                font-size: 11px;
                font-weight: 500;
            }}
        """)
        
        value_label = QtWidgets.QLabel(value)
        value_label.setStyleSheet(f"""
            QLabel {{
                color: {ModernTheme.COLORS["primary"]};
                font-size: 14px;
                font-weight: 600;
            }}
        """)
        value_label.setProperty("stat_type", title.split()[0].lower())
        
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        
        return widget
    
    def _create_monitoring_tab(self):
        """Создать вкладку мониторинга."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(tab)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(20)
        
        # Левая панель - видео
        video_panel = QtWidgets.QWidget()
        video_layout = QtWidgets.QVBoxLayout(video_panel)
        video_layout.setSpacing(15)
        
        # Панель управления сеткой
        grid_controls = QtWidgets.QWidget()
        grid_layout = QtWidgets.QHBoxLayout(grid_controls)
        
        grid_label = QtWidgets.QLabel("Сетка отображения:")
        grid_label.setStyleSheet(f"""
            QLabel {{
                color: {ModernTheme.COLORS["text_primary"]};
                font-size: 14px;
                font-weight: 500;
            }}
        """)
        
        self.grid_selector = QtWidgets.QComboBox()
        self.grid_selector.addItems(self.GRID_VARIANTS)
        self.grid_selector.setStyleSheet(ModernTheme.STYLES["input"])
        self.grid_selector.setCurrentText(self.settings.get_grid())
        self.grid_selector.currentTextChanged.connect(self._on_grid_changed)
        
        grid_layout.addWidget(grid_label)
        grid_layout.addWidget(self.grid_selector)
        grid_layout.addStretch()
        
        # Виджет сетки
        self.grid_widget = QtWidgets.QWidget()
        self.grid_layout = QtWidgets.QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(10)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        
        video_layout.addWidget(grid_controls)
        video_layout.addWidget(self.grid_widget, 1)
        
        # Правая панель - события
        events_panel = QtWidgets.QWidget()
        events_layout = QtWidgets.QVBoxLayout(events_panel)
        events_layout.setSpacing(15)
        
        # Детали события
        self.event_detail = EventDetailView()
        
        # Таблица событий
        events_card = ModernCard("📈 ПОСЛЕДНИЕ СОБЫТИЯ")
        events_content = QtWidgets.QVBoxLayout()
        
        self.events_table = QtWidgets.QTableWidget(0, 5)
        self.events_table.setHorizontalHeaderLabels([
            "ВРЕМЯ", "НОМЕР", "СТРАНА", "КАНАЛ", "УВЕРЕННОСТЬ"
        ])
        self.events_table.setStyleSheet(ModernTheme.STYLES["table"])
        self.events_table.horizontalHeader().setStretchLastSection(True)
        self.events_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.events_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.events_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.events_table.verticalHeader().setVisible(False)
        self.events_table.itemSelectionChanged.connect(self._on_event_selected)
        
        events_content.addWidget(self.events_table)
        events_card.setLayout(events_content)
        
        events_layout.addWidget(self.event_detail, 2)
        events_layout.addWidget(events_card, 1)
        
        layout.addWidget(video_panel, 2)
        layout.addWidget(events_panel, 1)
        
        self.tab_widget.addTab(tab, "📹 МОНИТОРИНГ")
        self._draw_grid()
    
    def _create_search_tab(self):
        """Создать вкладку поиска."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Карточка фильтров
        filters_card = ModernCard("🔍 ФИЛЬТРЫ ПОИСКА")
        filters_layout = QtWidgets.QFormLayout()
        filters_layout.setContentsMargins(10, 10, 10, 10)
        filters_layout.setVerticalSpacing(15)
        filters_layout.setHorizontalSpacing(20)
        
        # Поле поиска по номеру
        self.search_plate = QtWidgets.QLineEdit()
        self.search_plate.setPlaceholderText("Введите номер или часть номера...")
        self.search_plate.setStyleSheet(ModernTheme.STYLES["input"])
        
        # Диапазон дат
        date_layout = QtWidgets.QHBoxLayout()
        self.search_from = QtWidgets.QDateTimeEdit()
        self.search_to = QtWidgets.QDateTimeEdit()
        
        for widget in [self.search_from, self.search_to]:
            widget.setCalendarPopup(True)
            widget.setDisplayFormat("dd.MM.yyyy HH:mm")
            widget.setStyleSheet(ModernTheme.STYLES["input"])
        
        date_layout.addWidget(QtWidgets.QLabel("С:"))
        date_layout.addWidget(self.search_from)
        date_layout.addWidget(QtWidgets.QLabel("По:"))
        date_layout.addWidget(self.search_to)
        date_layout.addStretch()
        
        # Каналы
        self.channel_filter = QtWidgets.QComboBox()
        self.channel_filter.addItem("Все каналы", None)
        self.channel_filter.setStyleSheet(ModernTheme.STYLES["input"])
        
        filters_layout.addRow("Номер:", self.search_plate)
        filters_layout.addRow("Дата:", date_layout)
        filters_layout.addRow("Канал:", self.channel_filter)
        
        # Кнопка поиска
        search_button = ModernButton("🔍 НАЙТИ", style="primary")
        search_button.clicked.connect(self._run_plate_search)
        
        filters_layout.addRow("", search_button)
        filters_card.setLayout(filters_layout)
        
        # Таблица результатов
        results_card = ModernCard("📋 РЕЗУЛЬТАТЫ")
        results_layout = QtWidgets.QVBoxLayout()
        
        self.search_table = QtWidgets.QTableWidget(0, 6)
        self.search_table.setHorizontalHeaderLabels([
            "ВРЕМЯ", "КАНАЛ", "СТРАНА", "НОМЕР", "УВЕРЕННОСТЬ", "ИСТОЧНИК"
        ])
        self.search_table.setStyleSheet(ModernTheme.STYLES["table"])
        self.search_table.horizontalHeader().setStretchLastSection(True)
        
        results_layout.addWidget(self.search_table)
        results_card.setLayout(results_layout)
        
        layout.addWidget(filters_card)
        layout.addWidget(results_card, 1)
        
        self.tab_widget.addTab(tab, "🔎 ПОИСК")
    
    def _create_settings_tab(self):
        """Создать вкладку настроек с современным дизайном."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Боковая панель
        self.settings_sidebar = SettingsSidebar(self)
        
        # Область настроек
        self.settings_stack = QtWidgets.QStackedWidget()
        
        # Создаем страницы настроек
        self._create_general_settings()
        self._create_channels_settings()
        self._create_recognition_settings()
        self._create_motion_settings()
        self._create_storage_settings()
        self._create_appearance_settings()
        
        # Связываем боковую панель со стеком
        self.settings_sidebar.category_list.currentRowChanged.connect(
            self.settings_stack.setCurrentIndex
        )
        self.settings_sidebar.category_list.setCurrentRow(0)
        
        layout.addWidget(self.settings_sidebar)
        layout.addWidget(self.settings_stack, 1)
        
        self.tab_widget.addTab(tab, "⚙️ НАСТРОЙКИ")
    
    def _create_general_settings(self):
        """Создать страницу общих настроек."""
        widget = QtWidgets.QScrollArea()
        widget.setWidgetResizable(True)
        widget.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollArea > QWidget > QWidget {
                background: transparent;
            }
        """)
        
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # Группа переподключения
        reconnect_group = SettingsGroup(
            "Автоматическое переподключение",
            "Настройки поведения при потере сигнала с камер"
        )
        
        self.reconnect_enabled = QtWidgets.QCheckBox("Включить автопереподключение")
        self.reconnect_enabled.setStyleSheet(ModernTheme.STYLES["checkbox"])
        
        self.frame_timeout = QtWidgets.QSpinBox()
        self.frame_timeout.setRange(1, 300)
        self.frame_timeout.setSuffix(" сек")
        self.frame_timeout.setStyleSheet(ModernTheme.STYLES["input"])
        
        self.retry_interval = QtWidgets.QSpinBox()
        self.retry_interval.setRange(1, 300)
        self.retry_interval.setSuffix(" сек")
        self.retry_interval.setStyleSheet(ModernTheme.STYLES["input"])
        
        reconnect_group.add_setting("Автопереподключение", self.reconnect_enabled)
        reconnect_group.add_setting("Таймаут кадра", self.frame_timeout,
                                   "Время ожидания кадра перед переподключением")
        reconnect_group.add_setting("Интервал повтора", self.retry_interval,
                                   "Интервал между попытками переподключения")
        
        # Группа уведомлений
        notifications_group = SettingsGroup(
            "Уведомления",
            "Настройки оповещений о событиях"
        )
        
        self.notify_new = QtWidgets.QCheckBox("Уведомлять о новых номерах")
        self.notify_error = QtWidgets.QCheckBox("Уведомлять об ошибках")
        self.notify_sound = QtWidgets.QCheckBox("Звуковые уведомления")
        
        for checkbox in [self.notify_new, self.notify_error, self.notify_sound]:
            checkbox.setStyleSheet(ModernTheme.STYLES["checkbox"])
        
        notifications_group.add_setting("Новые номера", self.notify_new)
        notifications_group.add_setting("Ошибки", self.notify_error)
        notifications_group.add_setting("Звук", self.notify_sound)
        
        layout.addWidget(reconnect_group)
        layout.addWidget(notifications_group)
        layout.addStretch()
        
        widget.setWidget(container)
        self.settings_stack.addWidget(widget)
    
    def _create_channels_settings(self):
        """Создать страницу настроек каналов."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # Список каналов
        channels_card = ModernCard("📷 КАНАЛЫ")
        channels_content = QtWidgets.QVBoxLayout()
        
        self.channels_list = QtWidgets.QListWidget()
        self.channels_list.setStyleSheet(ModernTheme.STYLES["list_widget"])
        self.channels_list.currentRowChanged.connect(self._load_channel_form)
        
        buttons_layout = QtWidgets.QHBoxLayout()
        add_btn = ModernButton("➕ ДОБАВИТЬ", style="secondary")
        remove_btn = ModernButton("🗑 УДАЛИТЬ", style="secondary")
        
        add_btn.clicked.connect(self._add_channel)
        remove_btn.clicked.connect(self._remove_channel)
        
        buttons_layout.addWidget(add_btn)
        buttons_layout.addWidget(remove_btn)
        buttons_layout.addStretch()
        
        channels_content.addWidget(self.channels_list)
        channels_content.addLayout(buttons_layout)
        channels_card.setLayout(channels_content)
        
        # Форма редактирования
        form_card = ModernCard("⚙️ РЕДАКТИРОВАНИЕ КАНАЛА")
        form_layout = QtWidgets.QFormLayout()
        form_layout.setContentsMargins(10, 10, 10, 10)
        form_layout.setVerticalSpacing(12)
        form_layout.setHorizontalSpacing(20)
        
        # Поля формы
        fields = [
            ("Название", QtWidgets.QLineEdit()),
            ("Источник", QtWidgets.QLineEdit()),
            ("Бестшоты", QtWidgets.QSpinBox()),
            ("Кулдаун", QtWidgets.QSpinBox()),
            ("Мин. уверенность", QtWidgets.QDoubleSpinBox()),
        ]
        
        for label_text, field in fields:
            label = QtWidgets.QLabel(label_text + ":")
            label.setStyleSheet(f"""
                QLabel {{
                    color: {ModernTheme.COLORS["text_primary"]};
                    font-weight: 500;
                }}
            """)
            
            if isinstance(field, QtWidgets.QSpinBox):
                field.setRange(1, 50)
            elif isinstance(field, QtWidgets.QDoubleSpinBox):
                field.setRange(0.0, 1.0)
                field.setSingleStep(0.05)
                field.setDecimals(2)
            
            field.setStyleSheet(ModernTheme.STYLES["input"])
            form_layout.addRow(label, field)
            setattr(self, f"channel_{label_text.split()[0].lower()}", field)
        
        # Кнопка сохранения
        save_btn = ModernButton("💾 СОХРАНИТЬ КАНАЛ", style="primary")
        save_btn.clicked.connect(self._save_channel)
        
        form_layout.addRow("", save_btn)
        form_card.setLayout(form_layout)
        
        layout.addWidget(channels_card, 1)
        layout.addWidget(form_card, 1)
        
        self.settings_stack.addWidget(widget)
        self._reload_channels_list()
    
    def _create_recognition_settings(self):
        """Создать страницу настроек распознавания."""
        widget = QtWidgets.QScrollArea()
        widget.setWidgetResizable(True)
        
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # Настройки OCR
        ocr_group = SettingsGroup(
            "Настройки OCR",
            "Параметры оптического распознавания символов"
        )
        
        self.ocr_min_confidence = QtWidgets.QDoubleSpinBox()
        self.ocr_min_confidence.setRange(0.0, 1.0)
        self.ocr_min_confidence.setSingleStep(0.05)
        self.ocr_min_confidence.setDecimals(2)
        self.ocr_min_confidence.setStyleSheet(ModernTheme.STYLES["input"])
        
        self.ocr_best_shots = QtWidgets.QSpinBox()
        self.ocr_best_shots.setRange(1, 10)
        self.ocr_best_shots.setStyleSheet(ModernTheme.STYLES["input"])
        
        ocr_group.add_setting("Минимальная уверенность", self.ocr_min_confidence,
                             "Минимальная уверенность для принятия результата")
        ocr_group.add_setting("Бестшотов на трек", self.ocr_best_shots,
                             "Количество кадров для консенсуса")
        
        # Настройки детекции
        detection_group = SettingsGroup(
            "Настройки детекции",
            "Параметры обнаружения автомобилей"
        )
        
        self.detector_stride = QtWidgets.QSpinBox()
        self.detector_stride.setRange(1, 10)
        self.detector_stride.setStyleSheet(ModernTheme.STYLES["input"])
        
        self.detection_mode = QtWidgets.QComboBox()
        self.detection_mode.addItems(["Постоянная", "По движению"])
        self.detection_mode.setStyleSheet(ModernTheme.STYLES["input"])
        
        detection_group.add_setting("Шаг детекции", self.detector_stride,
                                   "Обрабатывать каждый N-й кадр")
        detection_group.add_setting("Режим работы", self.detection_mode,
                                   "Способ активации детекции")
        
        # Настройки стран
        countries_group = SettingsGroup(
            "Настройки стран",
            "Конфигурация форматов номерных знаков"
        )
        
        self.country_config_dir = QtWidgets.QLineEdit()
        self.country_config_dir.setStyleSheet(ModernTheme.STYLES["input"])
        
        self.country_templates = QtWidgets.QListWidget()
        self.country_templates.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.country_templates.setStyleSheet(ModernTheme.STYLES["list_widget"])
        self.country_templates.setMaximumHeight(150)
        
        countries_group.add_setting("Каталог шаблонов", self.country_config_dir,
                                   "Путь к файлам конфигураций стран")
        countries_group.add_setting("Активные страны", self.country_templates,
                                   "Выберите страны для распознавания")
        
        layout.addWidget(ocr_group)
        layout.addWidget(detection_group)
        layout.addWidget(countries_group)
        layout.addStretch()
        
        widget.setWidget(container)
        self.settings_stack.addWidget(widget)
    
    def _create_motion_settings(self):
        """Создать страницу настроек детекции движения."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        motion_group = SettingsGroup(
            "Детектор движения",
            "Настройки обнаружения движения в кадре"
        )
        
        # Параметры чувствительности
        self.motion_threshold = QtWidgets.QDoubleSpinBox()
        self.motion_threshold.setRange(0.0, 1.0)
        self.motion_threshold.setSingleStep(0.01)
        self.motion_threshold.setDecimals(3)
        self.motion_threshold.setStyleSheet(ModernTheme.STYLES["input"])
        
        self.motion_stride = QtWidgets.QSpinBox()
        self.motion_stride.setRange(1, 30)
        self.motion_stride.setStyleSheet(ModernTheme.STYLES["input"])
        
        # Параметры стабилизации
        self.activation_frames = QtWidgets.QSpinBox()
        self.activation_frames.setRange(1, 60)
        self.activation_frames.setStyleSheet(ModernTheme.STYLES["input"])
        
        self.release_frames = QtWidgets.QSpinBox()
        self.release_frames.setRange(1, 120)
        self.release_frames.setStyleSheet(ModernTheme.STYLES["input"])
        
        motion_group.add_setting("Порог чувствительности", self.motion_threshold,
                                "Минимальная доля изменений для детекции движения")
        motion_group.add_setting("Частота анализа", self.motion_stride,
                                "Анализировать каждый N-й кадр")
        motion_group.add_setting("Кадров для активации", self.activation_frames,
                                "Минимальное количество кадров с движением")
        motion_group.add_setting("Кадров для деактивации", self.release_frames,
                                "Кадров без движения для отключения")
        
        layout.addWidget(motion_group)
        layout.addStretch()
        
        self.settings_stack.addWidget(widget)
    
    def _create_storage_settings(self):
        """Создать страницу настроек хранилища."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # Настройки базы данных
        db_group = SettingsGroup(
            "База данных",
            "Параметры хранения событий"
        )
        
        self.db_path = QtWidgets.QLineEdit()
        self.db_path.setStyleSheet(ModernTheme.STYLES["input"])
        
        db_browse_btn = ModernButton("📁 ВЫБРАТЬ", style="secondary")
        db_browse_btn.clicked.connect(self._choose_db_dir)
        
        db_layout = QtWidgets.QHBoxLayout()
        db_layout.addWidget(self.db_path, 1)
        db_layout.addWidget(db_browse_btn)
        
        db_group.add_setting("Путь к БД", db_layout, "Расположение файла базы данных")
        
        # Настройки скриншотов
        screenshots_group = SettingsGroup(
            "Скриншоты",
            "Настройки сохранения изображений"
        )
        
        self.screenshots_dir = QtWidgets.QLineEdit()
        self.screenshots_dir.setStyleSheet(ModernTheme.STYLES["input"])
        
        screenshots_browse_btn = ModernButton("📁 ВЫБРАТЬ", style="secondary")
        screenshots_browse_btn.clicked.connect(self._choose_screenshot_dir)
        
        screenshots_layout = QtWidgets.QHBoxLayout()
        screenshots_layout.addWidget(self.screenshots_dir, 1)
        screenshots_layout.addWidget(screenshots_browse_btn)
        
        self.save_screenshots = QtWidgets.QCheckBox("Сохранять скриншоты")
        self.save_screenshots.setStyleSheet(ModernTheme.STYLES["checkbox"])
        
        screenshots_group.add_setting("Папка скриншотов", screenshots_layout)
        screenshots_group.add_setting("Сохранение", self.save_screenshots)
        
        layout.addWidget(db_group)
        layout.addWidget(screenshots_group)
        layout.addStretch()
        
        self.settings_stack.addWidget(widget)
    
    def _create_appearance_settings(self):
        """Создать страницу настроек внешнего вида."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # Настройки темы
        theme_group = SettingsGroup(
            "Тема оформления",
            "Настройки внешнего вида интерфейса"
        )
        
        self.theme_selector = QtWidgets.QComboBox()
        self.theme_selector.addItems(["Тёмная (неоновая)", "Тёмная", "Светлая"])
        self.theme_selector.setStyleSheet(ModernTheme.STYLES["input"])
        
        self.scaling = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scaling.setRange(80, 150)
        self.scaling.setValue(100)
        self.scaling.setStyleSheet(ModernTheme.STYLES["slider"])
        
        scaling_layout = QtWidgets.QHBoxLayout()
        scaling_layout.addWidget(QtWidgets.QLabel("80%"))
        scaling_layout.addWidget(self.scaling, 1)
        scaling_layout.addWidget(QtWidgets.QLabel("150%"))
        
        theme_group.add_setting("Тема", self.theme_selector)
        theme_group.add_setting("Масштаб", scaling_layout)
        
        # Настройки отображения
        display_group = SettingsGroup(
            "Отображение",
            "Настройки показа информации"
        )
        
        self.show_fps = QtWidgets.QCheckBox("Показывать FPS")
        self.show_stats = QtWidgets.QCheckBox("Показывать статистику")
        self.show_hints = QtWidgets.QCheckBox("Показывать подсказки")
        
        for checkbox in [self.show_fps, self.show_stats, self.show_hints]:
            checkbox.setStyleSheet(ModernTheme.STYLES["checkbox"])
        
        display_group.add_setting("FPS", self.show_fps)
        display_group.add_setting("Статистика", self.show_stats)
        display_group.add_setting("Подсказки", self.show_hints)
        
        layout.addWidget(theme_group)
        layout.addWidget(display_group)
        layout.addStretch()
        
        self.settings_stack.addWidget(widget)
    
    def _create_status_bar(self):
        """Создать статус бар."""
        status_bar = QtWidgets.QStatusBar()
        status_bar.setStyleSheet(ModernTheme.STYLES["status_bar"])
        
        # Индикатор системы
        self.system_status = QtWidgets.QLabel("✅ СИСТЕМА АКТИВНА")
        self.system_status.setStyleSheet(f"""
            QLabel {{
                color: {ModernTheme.COLORS["success"]};
                font-weight: 500;
                padding: 0 10px;
            }}
        """)
        
        # Память
        self.memory_status = QtWidgets.QLabel("💾 ПАМЯТЬ: —")
        self.memory_status.setStyleSheet(f"""
            QLabel {{
                color: {ModernTheme.COLORS["text_secondary"]};
                font-weight: 500;
                padding: 0 10px;
                border-left: 1px solid {ModernTheme.COLORS["border"]};
            }}
        """)
        
        # CPU
        self.cpu_status = QtWidgets.QLabel("⚡ CPU: —")
        self.cpu_status.setStyleSheet(f"""
            QLabel {{
                color: {ModernTheme.COLORS["text_secondary"]};
                font-weight: 500;
                padding: 0 10px;
                border-left: 1px solid {ModernTheme.COLORS["border"]};
            }}
        """)
        
        # Время работы
        self.uptime_status = QtWidgets.QLabel("🕐 ВРЕМЯ: 00:00:00")
        self.uptime_status.setStyleSheet(f"""
            QLabel {{
                color: {ModernTheme.COLORS["text_secondary"]};
                font-weight: 500;
                padding: 0 10px;
                border-left: 1px solid {ModernTheme.COLORS["border"]};
            }}
        """)
        
        status_bar.addWidget(self.system_status)
        status_bar.addWidget(self.memory_status)
        status_bar.addWidget(self.cpu_status)
        status_bar.addWidget(self.uptime_status)
        
        self.setStatusBar(status_bar)
    
    # ============================================================================
    # МЕТОДЫ РАБОТЫ С ДАННЫМИ (остаются без изменений)
    # ============================================================================
    
    def _setup_storage(self):
        """Инициализировать хранилище."""
        self.db = EventDatabase(self.settings.get_db_path())
        self._pixmap_pool = PixmapPool()
        self.channel_labels = {}
        self.event_images = OrderedDict()
        self._image_cache_bytes = 0
        self.event_cache = {}
        self.flag_cache = {}
        self.flag_dir = Path(__file__).resolve().parents[2] / "images" / "flags"
    
    def _setup_workers(self):
        """Инициализировать воркеры."""
        self.channel_workers = []
    
    def _draw_grid(self):
        """Отрисовать сетку каналов."""
        # Очистка существующей сетки
        for i in reversed(range(self.grid_layout.count())):
            item = self.grid_layout.takeAt(i)
            if item.widget():
                item.widget().deleteLater()
        
        self.channel_labels.clear()
        channels = self.settings.get_channels()
        
        # Определение размеров сетки
        rows, cols = map(int, self.grid_selector.currentText().split("x"))
        
        # Создание каналов
        index = 0
        for row in range(rows):
            for col in range(cols):
                if index < len(channels):
                    channel_name = channels[index].get("name", f"Канал {index+1}")
                    label = ChannelView(channel_name, self._pixmap_pool)
                    self.channel_labels[channel_name] = label
                else:
                    label = ChannelView(f"Канал {index+1}", self._pixmap_pool)
                    label.set_status(False, "НЕ НАСТРОЕН")
                
                self.grid_layout.addWidget(label, row, col)
                index += 1
    
    def _start_system_monitoring(self):
        """Запустить мониторинг системы."""
        self.uptime_start = datetime.now()
        self.stats_timer = QtCore.QTimer(self)
        self.stats_timer.setInterval(2000)
        self.stats_timer.timeout.connect(self._update_system_stats)
        self.stats_timer.start()
        self._update_system_stats()
    
    def _update_system_stats(self):
        """Обновить статистику системы."""
        # CPU и память
        cpu_percent = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()
        
        self.cpu_status.setText(f"⚡ CPU: {cpu_percent:.0f}%")
        self.memory_status.setText(f"💾 ПАМЯТЬ: {ram.percent:.0f}% ({ram.used//1024//1024}MB)")
        
        # Время работы
        uptime = datetime.now() - self.uptime_start
        hours, remainder = divmod(uptime.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.uptime_status.setText(f"🕐 ВРЕМЯ: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        # Обновление статистики на верхней панели
        if hasattr(self, 'channels_stat'):
            active = len([w for w in self.channel_workers if w.isRunning()])
            total = len(self.settings.get_channels())
            self.channels_stat.findChild(QtWidgets.QLabel, "channels").setText(f"{active}/{total}")
        
        if hasattr(self, 'events_stat'):
            event_count = len(self.event_cache)
            self.events_stat.findChild(QtWidgets.QLabel, "events").setText(str(event_count))
    
    # ============================================================================
    # ОСТАЛЬНЫЕ МЕТОДЫ (адаптированные под новый дизайн)
    # ============================================================================
    
    def _on_grid_changed(self, grid: str) -> None:
        """Обработчик изменения сетки."""
        self.settings.save_grid(grid)
        self._draw_grid()
    
    def _start_channels(self) -> None:
        """Запустить каналы."""
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
                    label.set_status(False, "НЕТ ИСТОЧНИКА")
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
        
        self.system_status.setText("✅ СИСТЕМА АКТИВНА")
        self.system_status.setStyleSheet(f"""
            QLabel {{
                color: {ModernTheme.COLORS["success"]};
                font-weight: 500;
                padding: 0 10px;
            }}
        """)
    
    def _stop_workers(self) -> None:
        """Остановить воркеры."""
        for worker in self.channel_workers:
            worker.stop()
            worker.wait(1000)
        self.channel_workers = []
        
        self.system_status.setText("⏸ СИСТЕМА ОСТАНОВЛЕНА")
        self.system_status.setStyleSheet(f"""
            QLabel {{
                color: {ModernTheme.COLORS["warning"]};
                font-weight: 500;
                padding: 0 10px;
            }}
        """)
    
    def _update_frame(self, channel_name: str, image: QtGui.QImage) -> None:
        """Обновить кадр на видеопанели."""
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
        label.set_status(True)
    
    def _handle_event(self, event: Dict) -> None:
        """Обработать событие распознавания."""
        event_id = int(event.get("id", 0))
        
        # Обновление информации на канале
        channel_name = event.get("channel", "")
        if channel_name in self.channel_labels:
            label = self.channel_labels[channel_name]
            label.set_recognition_active(True)
            label.set_plate_info(event.get("plate", ""), event.get("confidence", 0))
            
            # Сбросить индикатор через 3 секунды
            QtCore.QTimer.singleShot(3000, lambda: label.set_recognition_active(False))
        
        # Добавление в таблицу
        self._insert_event_row(event)
        self._update_event_stats()
    
    def _handle_status(self, channel: str, status: str) -> None:
        """Обработать статус канала."""
        label = self.channel_labels.get(channel)
        if not label:
            return
        
        if "движ" in status.lower():
            label.set_motion_active(True)
            QtCore.QTimer.singleShot(2000, lambda: label.set_motion_active(False))
        
        if "нет сигнала" in status.lower() or "ошибка" in status.lower():
            label.set_status(False, status)
        else:
            label.set_status(True, status)
    
    def _insert_event_row(self, event: Dict) -> None:
        """Добавить строку в таблицу событий."""
        row = self.events_table.rowCount()
        self.events_table.insertRow(row)
        
        # Форматирование данных
        timestamp = self._format_timestamp(event.get("timestamp", ""))
        plate = event.get("plate", "—") or "Не распознан"
        country = event.get("country", "—")
        channel = event.get("channel", "—")
        confidence = f"{event.get('confidence', 0):.1%}"
        
        # Установка данных
        items = [
            QtWidgets.QTableWidgetItem(timestamp),
            QtWidgets.QTableWidgetItem(plate),
            QtWidgets.QTableWidgetItem(country),
            QtWidgets.QTableWidgetItem(channel),
            QtWidgets.QTableWidgetItem(confidence),
        ]
        
        # Цветовое кодирование уверенности
        if event.get("confidence", 0) > 0.9:
            items[4].setForeground(QtGui.QColor(ModernTheme.COLORS["success"]))
        elif event.get("confidence", 0) > 0.7:
            items[4].setForeground(QtGui.QColor(ModernTheme.COLORS["warning"]))
        else:
            items[4].setForeground(QtGui.QColor(ModernTheme.COLORS["danger"]))
        
        for col, item in enumerate(items):
            item.setData(QtCore.Qt.UserRole, event.get("id"))
            self.events_table.setItem(row, col, item)
        
        # Ограничение количества строк
        if self.events_table.rowCount() > 100:
            self.events_table.removeRow(0)
    
    def _on_event_selected(self):
        """Обработчик выбора события в таблице."""
        selected = self.events_table.selectedItems()
        if not selected:
            return
        
        event_id = selected[0].data(QtCore.Qt.UserRole)
        if not event_id:
            return
        
        # Поиск события в кеше
        event = self.event_cache.get(event_id)
        if event:
            self.event_detail.set_event(event)
    
    def _run_plate_search(self):
        """Выполнить поиск по номеру."""
        # TODO: Реализовать поиск
        pass
    
    def _reload_channels_list(self):
        """Обновить список каналов."""
        self.channels_list.clear()
        for channel in self.settings.get_channels():
            self.channels_list.addItem(channel.get("name", "Канал"))
        
        if self.channels_list.count():
            self.channels_list.setCurrentRow(0)
    
    def _load_channel_form(self, index: int):
        """Загрузить форму канала."""
        channels = self.settings.get_channels()
        if 0 <= index < len(channels):
            channel = channels[index]
            # TODO: Заполнить поля формы
            pass
    
    def _add_channel(self):
        """Добавить новый канал."""
        # TODO: Реализовать добавление канала
        pass
    
    def _remove_channel(self):
        """Удалить выбранный канал."""
        index = self.channels_list.currentRow()
        if index >= 0:
            # TODO: Реализовать удаление канала
            pass
    
    def _save_channel(self):
        """Сохранить изменения канала."""
        # TODO: Реализовать сохранение канала
        pass
    
    def _save_all_settings(self):
        """Сохранить все настройки."""
        # TODO: Реализовать сохранение всех настроек
        pass
    
    def _choose_db_dir(self):
        """Выбрать директорию для базы данных."""
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Выбор папки базы данных"
        )
        if directory:
            self.db_path.setText(directory)
    
    def _choose_screenshot_dir(self):
        """Выбрать директорию для скриншотов."""
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Выбор папки для скриншотов"
        )
        if directory:
            self.screenshots_dir.setText(directory)
    
    def _refresh_events_table(self):
        """Обновить таблицу событий."""
        # TODO: Загрузить события из БД
        pass
    
    def _update_event_stats(self):
        """Обновить статистику событий."""
        if hasattr(self, 'events_stat'):
            event_count = len(self.event_cache)
            self.events_stat.findChild(QtWidgets.QLabel, "events").setText(str(event_count))
    
    @staticmethod
    def _format_timestamp(value: str) -> str:
        """Форматировать timestamp."""
        if not value:
            return "—"
        try:
            dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
            return dt.strftime("%H:%M:%S")
        except:
            return value
    
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Обработчик закрытия окна."""
        self._stop_workers()
        event.accept()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
