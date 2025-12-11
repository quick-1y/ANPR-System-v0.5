#!/usr/bin/env python3
# /anpr/pipeline/factory.py
from __future__ import annotations

from typing import Tuple
import threading

from anpr.config import ModelConfig
from anpr.detection.yolo_detector import YOLODetector
from anpr.pipeline.anpr_pipeline import ANPRPipeline
from anpr.recognition.crnn_recognizer import CRNNRecognizer
from anpr.plate_postprocessing import PlatePostProcessor


_RECOGNIZER_LOCK = threading.Lock()
_RECOGNIZER_SINGLETON: CRNNRecognizer | None = None
_POSTPROCESSOR_SINGLETON: PlatePostProcessor | None = None


def _get_shared_recognizer() -> CRNNRecognizer:
    """Lazily initializes a single OCR recognizer instance for all pipelines.

    CRNN quantization with ``prepare_fx`` is not thread-safe, so creating the
    recognizer concurrently for multiple channels can crash. By guarding
    initialization with a lock and reusing the instance across pipelines, we
    avoid the race while keeping inference stateless and reusable.
    """

    global _RECOGNIZER_SINGLETON

    if _RECOGNIZER_SINGLETON is None:
        with _RECOGNIZER_LOCK:
            if _RECOGNIZER_SINGLETON is None:
                _RECOGNIZER_SINGLETON = CRNNRecognizer(
                    ModelConfig.OCR_MODEL_PATH, ModelConfig.DEVICE
                )
    return _RECOGNIZER_SINGLETON


def _get_postprocessor(plate_config: dict | None) -> PlatePostProcessor:
    global _POSTPROCESSOR_SINGLETON
    config_dir = plate_config.get("config_dir") if plate_config else "anpr/plate_postprocessing/configs"
    enabled_countries = plate_config.get("enabled_countries") if plate_config else None
    if _POSTPROCESSOR_SINGLETON is None:
        with _RECOGNIZER_LOCK:
            if _POSTPROCESSOR_SINGLETON is None:
                _POSTPROCESSOR_SINGLETON = PlatePostProcessor(
                    config_dir=config_dir, enabled_countries=enabled_countries
                )
    return _POSTPROCESSOR_SINGLETON


def build_components(
    best_shots: int,
    cooldown_seconds: int,
    min_confidence: float,
    plate_config: dict | None = None,
) -> Tuple[ANPRPipeline, YOLODetector]:
    """Создаёт независимые компоненты пайплайна (детектор, OCR и агрегация)."""

    detector = YOLODetector(ModelConfig.YOLO_MODEL_PATH, ModelConfig.DEVICE)
    recognizer = _get_shared_recognizer()
    postprocessor = _get_postprocessor(plate_config)
    pipeline = ANPRPipeline(
        recognizer,
        best_shots,
        cooldown_seconds,
        min_confidence=min_confidence,
        postprocessor=postprocessor,
    )
    return pipeline, detector
