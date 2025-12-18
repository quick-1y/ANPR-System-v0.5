#!/usr/bin/env python3
# /anpr/pipeline/factory.py
from __future__ import annotations

import os
from typing import Dict, Tuple
import threading

from anpr.config import Config
from anpr.detection.yolo_detector import YOLODetector
from anpr.pipeline.anpr_pipeline import ANPRPipeline
from anpr.postprocessing.country_config import CountryConfigLoader
from anpr.postprocessing.validator import PlatePostProcessor
from anpr.recognition.crnn_recognizer import CRNNRecognizer


_RECOGNIZER_LOCK = threading.Lock()
_RECOGNIZER_SINGLETON: CRNNRecognizer | None = None


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
                config = Config()
                _RECOGNIZER_SINGLETON = CRNNRecognizer(
                    config.ocr_model_path, config.device
                )
    return _RECOGNIZER_SINGLETON


def _build_postprocessor(config: Dict[str, object]) -> PlatePostProcessor:
    config_dir = str(config.get("config_dir") or "config/countries")
    enabled_countries = config.get("enabled_countries")
    loader = CountryConfigLoader(os.path.abspath(config_dir))
    loader.ensure_dir()
    return PlatePostProcessor(loader, enabled_countries)


def build_components(
    best_shots: int,
    cooldown_seconds: int,
    min_confidence: float,
    plate_config: Dict[str, object] | None = None,
    direction_config: Dict[str, object] | None = None,
) -> Tuple[ANPRPipeline, YOLODetector]:
    """Создаёт независимые компоненты пайплайна (детектор, OCR и агрегация)."""

    config = Config()
    detector = YOLODetector(config.yolo_model_path, config.device)
    recognizer = _get_shared_recognizer()
    postprocessor = _build_postprocessor(plate_config or {})
    pipeline = ANPRPipeline(
        recognizer,
        best_shots,
        cooldown_seconds,
        min_confidence=min_confidence,
        postprocessor=postprocessor,
        direction_config=direction_config,
    )
    return pipeline, detector
