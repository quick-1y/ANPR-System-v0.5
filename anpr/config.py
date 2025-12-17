# /anpr/config.py
"""Централизованная точка доступа к настройкам приложения.

Config выступает фасадом над SettingsManager и предоставляет единообразные
пути к моделям и параметры инференса. Все модули работают через этот класс,
избегая прямого чтения ``settings.json``.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from anpr.infrastructure.settings_manager import SettingsManager


class Config:
    """Синглтон, предоставляющий доступ к конфигурации приложения."""

    _instance: "Config | None" = None

    def __new__(cls) -> "Config":  # noqa: D401 - стандартный паттерн синглтона
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._settings = SettingsManager()
        return cls._instance

    # ------------------------- Модель и инференс -------------------------
    @property
    def model_paths(self) -> Dict[str, str]:
        return self._settings.get_model_settings()

    @property
    def yolo_model_path(self) -> str:
        return str(self.model_paths.get("yolo_model_path", ""))

    @property
    def ocr_model_path(self) -> str:
        return str(self.model_paths.get("ocr_model_path", ""))

    @property
    def device(self) -> torch.device:
        device_name = self.model_paths.get("device") or "cpu"
        return torch.device(device_name)

    @property
    def ocr_config(self) -> Dict[str, Any]:
        return self._settings.get_ocr_settings()

    @property
    def ocr_height(self) -> int:
        return int(self.ocr_config.get("img_height", 32))

    @property
    def ocr_width(self) -> int:
        return int(self.ocr_config.get("img_width", 128))

    @property
    def ocr_alphabet(self) -> str:
        return str(self.ocr_config.get("alphabet", ""))

    @property
    def ocr_confidence_threshold(self) -> float:
        return float(self.ocr_config.get("confidence_threshold", 0.6))

    @property
    def detector_config(self) -> Dict[str, Any]:
        return self._settings.get_detector_settings()

    @property
    def detection_confidence_threshold(self) -> float:
        return float(self.detector_config.get("confidence_threshold", 0.5))

    # --------------------------- Делегаты UI -----------------------------
    def __getattr__(self, name: str):
        """Делегирует неизвестные атрибуты во внутренний SettingsManager."""

        if hasattr(self._settings, name):
            return getattr(self._settings, name)
        raise AttributeError(name)


__all__ = ["Config"]
