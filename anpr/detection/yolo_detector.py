# /anpr/detection/yolo_detector.py
"""Обертка для детектора номерных знаков YOLO."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from ultralytics import YOLO

from anpr.config import ModelConfig
from logging_manager import get_logger

logger = get_logger(__name__)


class YOLODetector:
    """Детектор с безопасным откатом к обычной детекции при ошибках трекера."""

    def __init__(self, model_path: str, device) -> None:
        self.model = YOLO(model_path)
        self.model.to(device)
        self.device = device
        self._tracking_supported = True
        logger.info("Детектор YOLO успешно загружен (model=%s, device=%s)", model_path, device)

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        detections = self.model.predict(frame, verbose=False, device=self.device)
        results: List[Dict[str, Any]] = []
        for det in detections[0].boxes.data:
            x1, y1, x2, y2, conf, _ = det.cpu().numpy()
            if conf >= ModelConfig.DETECTION_CONFIDENCE_THRESHOLD:
                results.append({"bbox": [int(x1), int(y1), int(x2), int(y2)], "confidence": float(conf)})
        return results

    def _track_internal(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        detections = self.model.track(frame, persist=True, verbose=False, device=self.device)
        results: List[Dict[str, Any]] = []
        if detections[0].boxes.id is None:
            return results

        track_ids = detections[0].boxes.id.int().cpu().tolist()
        boxes = detections[0].boxes.xyxy.cpu().numpy()
        confs = detections[0].boxes.conf.cpu().numpy()

        for box, track_id, conf in zip(boxes, track_ids, confs):
            if conf >= ModelConfig.DETECTION_CONFIDENCE_THRESHOLD:
                results.append(
                    {
                        "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                        "confidence": float(conf),
                        "track_id": track_id,
                    }
                )
        return results

    def track(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if not self._tracking_supported:
            return self.detect(frame)

        try:
            return self._track_internal(frame)
        except ModuleNotFoundError:
            self._tracking_supported = False
            logger.warning("Отключаем трекинг YOLO: отсутствуют зависимости")
            return self.detect(frame)
        except Exception:
            self._tracking_supported = False
            logger.exception("Отключаем трекинг YOLO из-за ошибки, переключаемся на detect")
            return self.detect(frame)

