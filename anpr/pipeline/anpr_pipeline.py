# /anpr/pipeline/anpr_pipeline.py
"""Пайплайн объединяющий детекцию и OCR."""

from __future__ import annotations

import time
from collections import Counter
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from anpr.config import Config
from anpr.postprocessing.validator import PlatePostProcessor
from anpr.recognition.crnn_recognizer import CRNNRecognizer


class TrackAggregator:
    """Агрегирует результаты распознавания в рамках одного трека."""

    def __init__(self, best_shots: int):
        self.best_shots = max(1, best_shots)
        self.track_texts: Dict[int, List[str]] = {}
        self.last_emitted: Dict[int, str] = {}

    def add_result(self, track_id: int, text: str) -> str:
        if not text:
            return ""

        bucket = self.track_texts.setdefault(track_id, [])
        bucket.append(text)
        if len(bucket) > self.best_shots:
            bucket.pop(0)

        counts = Counter(bucket)
        consensus, freq = counts.most_common(1)[0]
        quorum = max(1, (self.best_shots + 1) // 2)
        has_quorum = len(bucket) >= self.best_shots and freq >= quorum
        if has_quorum and self.last_emitted.get(track_id) != consensus:
            self.last_emitted[track_id] = consensus
            return consensus
        return ""

    def clear_last(self, track_id: int) -> None:
        self.last_emitted.pop(track_id, None)


class ANPRPipeline:
    """Основной класс распознавания."""

    def __init__(
        self,
        recognizer: CRNNRecognizer,
        best_shots: int,
        cooldown_seconds: int = 0,
        min_confidence: float = Config.thresholds().ocr_confidence,
        postprocessor: Optional[PlatePostProcessor] = None,
    ) -> None:
        self.recognizer = recognizer
        self.aggregator = TrackAggregator(best_shots)
        self.cooldown_seconds = max(0, cooldown_seconds)
        self.min_confidence = max(0.0, min(1.0, min_confidence))
        self._last_seen: Dict[str, float] = {}
        self.postprocessor = postprocessor

    def _on_cooldown(self, plate: str) -> bool:
        last_seen = self._last_seen.get(plate)
        if last_seen is None:
            return False
        return (time.monotonic() - last_seen) < self.cooldown_seconds

    def _touch_plate(self, plate: str) -> None:
        self._last_seen[plate] = time.monotonic()

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        if maxWidth <= 0 or maxHeight <= 0:
            return image
        dst = np.array(
            [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32"
        )
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    def _preprocess_plate(self, plate_image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return plate_image
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                return self._four_point_transform(plate_image, approx.reshape(4, 2))
        return plate_image

    def process_frame(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        plate_inputs: List[np.ndarray] = []
        detection_indices: List[int] = []

        for idx, detection in enumerate(detections):
            x1, y1, x2, y2 = detection["bbox"]
            roi = frame[y1:y2, x1:x2]

            if roi.size > 0:
                processed_plate = self._preprocess_plate(roi)

                if processed_plate.size > 0:
                    plate_inputs.append(processed_plate)
                    detection_indices.append(idx)

        batch_results = self.recognizer.recognize_batch(plate_inputs)

        for detection_idx, (current_text, confidence) in zip(detection_indices, batch_results):
            detection = detections[detection_idx]

            if confidence < self.min_confidence:
                detection["text"] = "Нечитаемо"
                detection["unreadable"] = True
                detection["confidence"] = confidence
                continue

            if "track_id" in detection:
                detection["text"] = self.aggregator.add_result(detection["track_id"], current_text)
            else:
                detection["text"] = current_text

            detection["confidence"] = confidence

            if self.postprocessor and detection.get("text"):
                processed = self.postprocessor.process(detection["text"])
                detection["original_text"] = detection.get("text")
                if processed.is_valid:
                    detection["text"] = processed.plate
                elif processed.plate:
                    detection["text"] = processed.plate or detection.get("text")
                else:
                    detection["text"] = ""
                detection["country"] = processed.country
                detection["format"] = processed.format_name
                detection["validated"] = processed.is_valid
                if not detection["text"] and "track_id" in detection:
                    self.aggregator.clear_last(detection["track_id"])

            if self.cooldown_seconds > 0 and detection.get("text"):
                if self._on_cooldown(detection["text"]):
                    detection["text"] = ""
                else:
                    self._touch_plate(detection["text"])
        return detections

