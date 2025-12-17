#!/usr/bin/env python3
"""Централизованный фасад для конфигурации приложения."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List, Optional

import torch

from anpr.infrastructure.logging_manager import get_logger
from anpr.infrastructure.settings_manager import SettingsManager

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelPaths:
    yolo_path: str
    ocr_path: str
    device: torch.device


@dataclass(frozen=True)
class OCRConstants:
    img_height: int
    img_width: int
    alphabet: str


@dataclass(frozen=True)
class DetectionThresholds:
    detection_confidence: float
    ocr_confidence: float


@dataclass(frozen=True)
class ChannelConfig:
    id: int
    name: str
    source: str
    best_shots: int
    cooldown_seconds: int
    ocr_min_confidence: float
    detection_mode: str
    detector_frame_stride: int
    motion_threshold: float
    motion_frame_stride: int
    motion_activation_frames: int
    motion_release_frames: int
    region: Dict[str, Any]
    debug: Dict[str, Any]

    @classmethod
    def from_dict(cls, channel: Dict[str, Any]) -> "ChannelConfig":
        return cls(
            id=int(channel.get("id", 0) or 0),
            name=str(channel.get("name", "")),
            source=str(channel.get("source", "0")),
            best_shots=int(channel.get("best_shots", 3)),
            cooldown_seconds=int(channel.get("cooldown_seconds", 5)),
            ocr_min_confidence=float(channel.get("ocr_min_confidence", 0.6)),
            detection_mode=str(channel.get("detection_mode", "continuous")),
            detector_frame_stride=int(channel.get("detector_frame_stride", 2)),
            motion_threshold=float(channel.get("motion_threshold", 0.01)),
            motion_frame_stride=int(channel.get("motion_frame_stride", 1)),
            motion_activation_frames=int(channel.get("motion_activation_frames", 3)),
            motion_release_frames=int(channel.get("motion_release_frames", 6)),
            region=dict(channel.get("region", {})),
            debug=dict(channel.get("debug", {})),
        )


class Config:
    """Singleton-фасад для потокобезопасного доступа к настройкам."""

    _instance: Optional["Config"] = None
    _instance_lock: Lock = Lock()

    def __new__(cls, settings_manager: Optional[SettingsManager] = None) -> "Config":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init(settings_manager)
        return cls._instance

    def _init(self, settings_manager: Optional[SettingsManager]) -> None:
        self._settings = settings_manager or SettingsManager()
        self._cache: Dict[str, Any] = {}
        self._cache_lock = Lock()

    @classmethod
    def instance(cls) -> "Config":
        return cls()

    # ---- Централизованные статические методы доступа ----
    @classmethod
    def models(cls) -> ModelPaths:
        instance = cls()
        return instance._get_cached("models", instance._build_models)

    @classmethod
    def ocr_constants(cls) -> OCRConstants:
        instance = cls()
        return instance._get_cached("ocr_constants", instance._build_ocr_constants)

    @classmethod
    def thresholds(cls) -> DetectionThresholds:
        instance = cls()
        return instance._get_cached("thresholds", instance._build_thresholds)

    @classmethod
    def channels(cls) -> List[ChannelConfig]:
        instance = cls()
        return instance._get_cached("channels", instance._build_channels)

    # ---- Методы обновления ----
    @classmethod
    def refresh(cls) -> None:
        instance = cls()
        instance._settings.refresh()
        with instance._cache_lock:
            instance._cache.clear()
        logger.info("Конфигурация обновлена из %s", instance._settings.path)

    # ---- Методы делегирования для UI и сервисов ----
    @classmethod
    def get_channels(cls) -> List[Dict[str, Any]]:
        return cls()._settings.get_channels()

    @classmethod
    def save_channels(cls, channels: List[Dict[str, Any]]) -> None:
        cls()._settings.save_channels(channels)
        cls()._invalidate("channels")

    @classmethod
    def get_grid(cls) -> str:
        return cls()._settings.get_grid()

    @classmethod
    def save_grid(cls, grid: str) -> None:
        cls()._settings.save_grid(grid)

    @classmethod
    def get_reconnect(cls) -> Dict[str, Any]:
        return cls()._settings.get_reconnect()

    @classmethod
    def save_reconnect(cls, reconnect_conf: Dict[str, Any]) -> None:
        cls()._settings.save_reconnect(reconnect_conf)

    @classmethod
    def get_db_dir(cls) -> str:
        return cls()._settings.get_db_dir()

    @classmethod
    def get_database_file(cls) -> str:
        return cls()._settings.get_database_file()

    @classmethod
    def get_db_path(cls) -> str:
        return cls()._settings.get_db_path()

    @classmethod
    def save_db_dir(cls, path: str) -> None:
        cls()._settings.save_db_dir(path)

    @classmethod
    def save_screenshot_dir(cls, path: str) -> None:
        cls()._settings.save_screenshot_dir(path)

    @classmethod
    def get_screenshot_dir(cls) -> str:
        return cls()._settings.get_screenshot_dir()

    @classmethod
    def get_time_settings(cls) -> Dict[str, Any]:
        return cls()._settings.get_time_settings()

    @classmethod
    def save_time_settings(cls, time_settings: Dict[str, Any]) -> None:
        cls()._settings.save_time_settings(time_settings)

    @classmethod
    def get_timezone(cls) -> str:
        return cls()._settings.get_timezone()

    @classmethod
    def get_time_offset_minutes(cls) -> int:
        return cls()._settings.get_time_offset_minutes()

    @classmethod
    def get_best_shots(cls) -> int:
        return cls()._settings.get_best_shots()

    @classmethod
    def save_best_shots(cls, best_shots: int) -> None:
        cls()._settings.save_best_shots(best_shots)

    @classmethod
    def get_cooldown_seconds(cls) -> int:
        return cls()._settings.get_cooldown_seconds()

    @classmethod
    def save_cooldown_seconds(cls, cooldown: int) -> None:
        cls()._settings.save_cooldown_seconds(cooldown)

    @classmethod
    def get_min_confidence(cls) -> float:
        return cls()._settings.get_min_confidence()

    @classmethod
    def save_min_confidence(cls, min_conf: float) -> None:
        cls()._settings.save_min_confidence(min_conf)

    @classmethod
    def get_plate_settings(cls) -> Dict[str, Any]:
        return cls()._settings.get_plate_settings()

    @classmethod
    def save_plate_settings(cls, plate_settings: Dict[str, Any]) -> None:
        cls()._settings.save_plate_settings(plate_settings)

    @classmethod
    def get_logging_config(cls) -> Dict[str, Any]:
        return cls()._settings.get_logging_config()

    @classmethod
    def get_models_config(cls) -> Dict[str, Any]:
        return cls()._settings.get_models()

    @classmethod
    def save_models_config(cls, models: Dict[str, Any]) -> None:
        cls()._settings.save_models(models)
        cls()._invalidate("models")

    @classmethod
    def get_ocr_constants_config(cls) -> Dict[str, Any]:
        return cls()._settings.get_ocr_constants()

    @classmethod
    def save_ocr_constants_config(cls, ocr_constants: Dict[str, Any]) -> None:
        cls()._settings.save_ocr_constants(ocr_constants)
        cls()._invalidate("ocr_constants")

    @classmethod
    def get_thresholds_config(cls) -> Dict[str, Any]:
        return cls()._settings.get_thresholds()

    @classmethod
    def save_thresholds_config(cls, thresholds: Dict[str, Any]) -> None:
        cls()._settings.save_thresholds(thresholds)
        cls()._invalidate("thresholds")

    # ---- Вспомогательные методы ----
    def _invalidate(self, key: str) -> None:
        with self._cache_lock:
            self._cache.pop(key, None)

    def _get_cached(self, key: str, builder) -> Any:
        with self._cache_lock:
            if key not in self._cache:
                self._cache[key] = builder()
        return self._cache[key]

    def _build_models(self) -> ModelPaths:
        models = self._settings.get_models()
        device_raw = models.get("device", "cpu")
        device = torch.device(str(device_raw))
        return ModelPaths(
            yolo_path=str(models.get("yolo_path", "models/yolo/best.pt")),
            ocr_path=str(models.get("ocr_path", "models/ocr_crnn/crnn_ocr_model_int8_fx.pth")),
            device=device,
        )

    def _build_ocr_constants(self) -> OCRConstants:
        constants = self._settings.get_ocr_constants()
        return OCRConstants(
            img_height=int(constants.get("img_height", 32)),
            img_width=int(constants.get("img_width", 128)),
            alphabet=str(constants.get("alphabet", "")),
        )

    def _build_thresholds(self) -> DetectionThresholds:
        thresholds = self._settings.get_thresholds()
        return DetectionThresholds(
            detection_confidence=float(thresholds.get("detection_confidence", 0.5)),
            ocr_confidence=float(thresholds.get("ocr_confidence", 0.6)),
        )

    def _build_channels(self) -> List[ChannelConfig]:
        channels = self._settings.get_channels()
        return [ChannelConfig.from_dict(ch) for ch in channels]
