# /anpr/config.py
"""Централизованные пути и пороги для моделей ANPR.

Вынесение конфигурации в отдельный модуль позволяет использовать единые
значения в детекторе, OCR и пайплайне без прямой зависимости модулей друг от
друга.
"""

from __future__ import annotations

import torch


class ModelConfig:
    """Пути к моделям и базовые параметры распознавания."""

    YOLO_MODEL_PATH: str = "models/yolo/best.pt"
    OCR_MODEL_PATH: str = "models/ocr_crnn/crnn_ocr_model_int8_fx.pth"

    OCR_IMG_HEIGHT: int = 32
    OCR_IMG_WIDTH: int = 128
    OCR_ALPHABET: str = "0123456789ABCEHKMOPTXY"
    OCR_CONFIDENCE_THRESHOLD: float = 0.6

    DETECTION_CONFIDENCE_THRESHOLD: float = 0.5

    TRACK_BEST_SHOTS: int = 3

    DEVICE: torch.device = torch.device("cpu")

