#!/usr/bin/env python3
# /anpr/pipeline/factory.py
from __future__ import annotations

from typing import Tuple
import threading

from anpr.config import ModelConfig
from anpr.detection.yolo_detector import YOLODetector
from anpr.pipeline.anpr_pipeline import ANPRPipeline
from anpr.recognition.crnn_recognizer import CRNNRecognizer
from anpr.validation import PlatePostProcessor, ValidatorConfig


_RECOGNIZER_LOCK = threading.Lock()
_RECOGNIZER_SINGLETON: CRNNRecognizer | None = None
_VALIDATORS: dict[tuple[str, tuple[str, ...]], PlatePostProcessor] = {}


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


def _get_postprocessor(config: ValidatorConfig) -> PlatePostProcessor:
    key = (str(config.config_dir), tuple(config.enabled_countries))
    if key not in _VALIDATORS:
        _VALIDATORS[key] = PlatePostProcessor(config)
    return _VALIDATORS[key]


def build_components(
    best_shots: int,
    cooldown_seconds: int,
    min_confidence: float,
    validator_config: ValidatorConfig,
) -> Tuple[ANPRPipeline, YOLODetector]:
    """Создаёт независимые компоненты пайплайна (детектор, OCR и агрегация)."""

    detector = YOLODetector(ModelConfig.YOLO_MODEL_PATH, ModelConfig.DEVICE)
    recognizer = _get_shared_recognizer()
    postprocessor = _get_postprocessor(validator_config)
    pipeline = ANPRPipeline(
        recognizer,
        postprocessor,
        best_shots,
        cooldown_seconds,
        min_confidence=min_confidence,
    )
    return pipeline, detector
