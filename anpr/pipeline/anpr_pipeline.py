# /anpr/pipeline/anpr_pipeline.py
"""Пайплайн объединяющий детекцию и OCR."""

from __future__ import annotations

import time
from collections import Counter, deque
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
        self.track_texts: Dict[int, List[tuple[str, float]]] = {}
        self.last_emitted: Dict[int, str] = {}

    def add_result(self, track_id: int, text: str, confidence: float) -> str:
        if not text:
            return ""

        bucket = self.track_texts.setdefault(track_id, [])
        bucket.append((text, max(0.0, float(confidence))))
        if len(bucket) > self.best_shots:
            bucket.pop(0)

        weights: Dict[str, float] = {}
        counts: Counter[str] = Counter()
        total_weight = 0.0
        for entry_text, entry_confidence in bucket:
            weights[entry_text] = weights.get(entry_text, 0.0) + entry_confidence
            counts[entry_text] += 1
            total_weight += entry_confidence

        if not weights or total_weight <= 0:
            return ""

        consensus = max(weights, key=lambda value: (weights[value], counts[value]))
        consensus_weight = weights[consensus]
        quorum = max(1, (self.best_shots + 1) // 2)
        has_quorum = len(bucket) >= self.best_shots and counts[consensus] >= quorum
        has_weighted_majority = consensus_weight >= total_weight * 0.5
        if has_quorum and self.last_emitted.get(track_id) != consensus:
            if not has_weighted_majority:
                return ""
            self.last_emitted[track_id] = consensus
            return consensus
        return ""

    def clear_last(self, track_id: int) -> None:
        self.last_emitted.pop(track_id, None)


class TrackDirectionEstimator:
    """Оценивает направление движения по истории рамок номера."""

    APPROACHING = "APPROACHING"
    RECEDING = "RECEDING"
    UNKNOWN = "UNKNOWN"

    def __init__(
        self,
        history_size: int = 12,
        min_track_length: int = 3,
        smoothing_window: int = 5,
        confidence_threshold: float = 0.55,
        jitter_pixels: float = 1.0,
        min_area_change_ratio: float = 0.02,
    ) -> None:
        self.history_size = max(1, history_size)
        self.min_track_length = max(1, min_track_length)
        self.smoothing_window = max(1, smoothing_window)
        self.confidence_threshold = max(0.0, min(1.0, confidence_threshold))
        self.jitter_pixels = max(0.0, jitter_pixels)
        self.min_area_change_ratio = max(0.0, min_area_change_ratio)
        self._history: Dict[int, deque[tuple[float, float]]] = {}

    @classmethod
    def from_config(cls, config: Dict[str, float | int]) -> "TrackDirectionEstimator":
        return cls(
            history_size=int(config.get("history_size", 12)),
            min_track_length=int(config.get("min_track_length", 3)),
            smoothing_window=int(config.get("smoothing_window", 5)),
            confidence_threshold=float(config.get("confidence_threshold", 0.55)),
            jitter_pixels=float(config.get("jitter_pixels", 1.0)),
            min_area_change_ratio=float(config.get("min_area_change_ratio", 0.02)),
        )

    def _filtered(self, deltas: np.ndarray, threshold: float) -> np.ndarray:
        if deltas.size == 0:
            return deltas
        mask = np.abs(deltas) >= threshold
        return deltas[mask]

    def _recent_trend(self, values: np.ndarray) -> float:
        if values.size == 0:
            return 0.0
        window = values[-self.smoothing_window :]
        return float(window.mean())

    def _votes(self, vertical_deltas: np.ndarray, area_deltas: np.ndarray, current_area: float) -> list[int]:
        votes: list[int] = []
        filtered_vertical = self._filtered(vertical_deltas, self.jitter_pixels)
        area_threshold = max(self.min_area_change_ratio * max(current_area, 1.0), 1.0)
        filtered_area = self._filtered(area_deltas, area_threshold)

        for delta in filtered_vertical:
            votes.append(1 if delta > 0 else -1)
        for delta in filtered_area:
            votes.append(1 if delta > 0 else -1)
        return votes

    def _confidence(self, score: float, vote_count: int) -> float:
        if vote_count == 0:
            return 0.0
        normalized = np.tanh(abs(score))
        density = min(1.0, vote_count / max(1, self.min_track_length))
        return float(normalized * density)

    def update(self, track_id: int, bbox: list[int]) -> Dict[str, str]:
        if not bbox or len(bbox) != 4:
            return {"direction": self.UNKNOWN}

        width = max(1.0, float(bbox[2] - bbox[0]))
        height = max(1.0, float(bbox[3] - bbox[1]))
        center_y = (float(bbox[1]) + float(bbox[3])) / 2.0
        area = width * height

        history = self._history.setdefault(track_id, deque(maxlen=self.history_size))
        history.append((center_y, area))

        if len(history) < self.min_track_length:
            return {"direction": self.UNKNOWN}

        centers = np.array([item[0] for item in history], dtype=float)
        areas = np.array([item[1] for item in history], dtype=float)
        vertical_deltas = np.diff(centers)
        area_deltas = np.diff(areas)

        trend_vertical = self._recent_trend(vertical_deltas)
        trend_area = self._recent_trend(area_deltas)

        votes = self._votes(vertical_deltas, area_deltas, areas[-1])
        if not votes:
            return {"direction": self.UNKNOWN}

        score = float(np.mean(votes)) * 0.7 + (np.sign(trend_area) if trend_area != 0 else 0.0) * 0.3
        confidence = self._confidence(score, len(votes))

        if confidence < self.confidence_threshold:
            return {"direction": self.UNKNOWN}

        direction = self.APPROACHING if score >= 0 else self.RECEDING
        return {"direction": direction}


class ANPRPipeline:
    """Основной класс распознавания."""

    def __init__(
        self,
        recognizer: CRNNRecognizer,
        best_shots: int,
        cooldown_seconds: int = 0,
        min_confidence: float = Config().ocr_confidence_threshold,
        postprocessor: Optional[PlatePostProcessor] = None,
        direction_config: Optional[Dict[str, float | int]] = None,
    ) -> None:
        self.recognizer = recognizer
        self.aggregator = TrackAggregator(best_shots)
        self.cooldown_seconds = max(0, cooldown_seconds)
        self.min_confidence = max(0.0, min(1.0, min_confidence))
        self._last_seen: Dict[str, float] = {}
        self.postprocessor = postprocessor
        self.direction_estimator = TrackDirectionEstimator.from_config(direction_config or {})

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

    def _rotate_bound(self, image: np.ndarray, angle: float) -> np.ndarray:
        (height, width) = image.shape[:2]
        if height == 0 or width == 0:
            return image
        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        matrix[0, 2] += (new_width / 2.0) - center[0]
        matrix[1, 2] += (new_height / 2.0) - center[1]
        return cv2.warpAffine(image, matrix, (new_width, new_height))

    def _detect_plate_quadrilateral(self, binary: np.ndarray) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        image_area = float(binary.shape[0] * binary.shape[1])
        min_area = image_area * 0.1
        candidates = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in candidates[:10]:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                rect = cv2.boundingRect(approx)
                width = rect[2]
                height = rect[3]
                if width == 0 or height == 0:
                    continue
                aspect_ratio = width / float(height)
                if 1.3 <= aspect_ratio <= 7.0:
                    return approx.reshape(4, 2)

        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        width = rect[1][0]
        height = rect[1][1]
        if width == 0 or height == 0:
            return None
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio < 1.3 or aspect_ratio > 7.0:
            return None
        if cv2.contourArea(largest) < min_area:
            return None
        return box.astype("float32")

    def _estimate_skew_angle(self, gray: np.ndarray, binary: np.ndarray) -> tuple[float, float]:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        edges = cv2.dilate(edges, None, iterations=1)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=0.4 * gray.shape[1], maxLineGap=15)
        if lines is not None:
            angles = []
            weights = []
            for x1, y1, x2, y2 in lines[:, 0]:
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    continue
                angle = np.degrees(np.arctan2(dy, dx))
                if angle < -90:
                    angle += 180
                if angle > 90:
                    angle -= 180
                if abs(angle) > 45:
                    continue
                length = np.hypot(dx, dy)
                angles.append(angle)
                weights.append(length)
            if angles:
                angles_array = np.array(angles)
                weights_array = np.array(weights)
                median_angle = float(np.average(angles_array, weights=weights_array))
                spread = float(np.std(angles_array))
                count_score = min(1.0, len(angles_array) / 6.0)
                spread_score = max(0.0, 1.0 - (spread / 15.0))
                confidence = count_score * spread_score
                return median_angle, confidence

        contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0, 0.0
        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        width, height = rect[1]
        if width == 0 or height == 0:
            return 0.0, 0.0
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio < 1.3 or aspect_ratio > 7.0:
            return 0.0, 0.0
        angle = rect[2]
        if width < height:
            angle = angle + 90
        return float(angle), 0.35

    def _preprocess_plate(self, plate_image: np.ndarray) -> np.ndarray:
        if plate_image.size == 0:
            return plate_image
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

        quadrilateral = self._detect_plate_quadrilateral(cleaned)
        if quadrilateral is not None:
            return self._four_point_transform(plate_image, quadrilateral)

        angle, confidence = self._estimate_skew_angle(blurred, cleaned)
        if confidence < 0.35:
            return plate_image
        if abs(angle) < 5:
            return plate_image
        if abs(angle) > 45:
            return plate_image
        return self._rotate_bound(plate_image, angle)

    def process_frame(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        plate_inputs: List[np.ndarray] = []
        detection_indices: List[int] = []

        for idx, detection in enumerate(detections):
            if self.direction_estimator and detection.get("track_id") is not None:
                direction_info = self.direction_estimator.update(int(detection["track_id"]), detection.get("bbox", []))
                detection.update(direction_info)
            else:
                detection.setdefault("direction", TrackDirectionEstimator.UNKNOWN)

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
                detection["text"] = self.aggregator.add_result(detection["track_id"], current_text, confidence)
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
