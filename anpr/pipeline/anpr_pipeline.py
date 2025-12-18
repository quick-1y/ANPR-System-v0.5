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
        self._bbox_history: Dict[int, deque[tuple[float, float, float, float]]] = {}
        self._last_direction: Dict[int, str] = {}

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

    def _remember_direction(self, track_id: int, direction: str) -> Dict[str, str]:
        self._last_direction[track_id] = direction
        return {"direction": direction}

    def update(self, track_id: int, bbox: list[int]) -> Dict[str, str]:
        if not bbox or len(bbox) != 4:
            return self._remember_direction(track_id, self.UNKNOWN)

        width = max(1.0, float(bbox[2] - bbox[0]))
        height = max(1.0, float(bbox[3] - bbox[1]))
        center_y = (float(bbox[1]) + float(bbox[3])) / 2.0
        area = width * height

        history = self._history.setdefault(track_id, deque(maxlen=self.history_size))
        history.append((center_y, area))

        bbox_history = self._bbox_history.setdefault(track_id, deque(maxlen=self.history_size))
        bbox_history.append((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])))

        if len(history) < self.min_track_length:
            return self._remember_direction(track_id, self.UNKNOWN)

        centers = np.array([item[0] for item in history], dtype=float)
        areas = np.array([item[1] for item in history], dtype=float)
        vertical_deltas = np.diff(centers)
        area_deltas = np.diff(areas)

        trend_vertical = self._recent_trend(vertical_deltas)
        trend_area = self._recent_trend(area_deltas)

        votes = self._votes(vertical_deltas, area_deltas, areas[-1])
        if not votes:
            return self._remember_direction(track_id, self.UNKNOWN)

        score = float(np.mean(votes)) * 0.7 + (np.sign(trend_area) if trend_area != 0 else 0.0) * 0.3
        confidence = self._confidence(score, len(votes))

        if confidence < self.confidence_threshold:
            return self._remember_direction(track_id, self.UNKNOWN)

        direction = self.APPROACHING if score >= 0 else self.RECEDING
        return self._remember_direction(track_id, direction)

    def _color_for_direction(self, direction: str) -> tuple[int, int, int]:
        if direction == self.APPROACHING:
            return 0, 180, 0
        if direction == self.RECEDING:
            return 0, 0, 200
        return 0, 200, 200

    def render_debug(self, frame: np.ndarray, thickness: int = 2) -> np.ndarray:
        """Наносит историю движения треков на кадр для отладки направления."""

        if frame is None:
            return frame

        overlay = frame.copy()

        for track_id, boxes in self._bbox_history.items():
            if len(boxes) < 2:
                continue

            centers = [((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0) for bbox in boxes]
            points = np.array(centers, dtype=int)
            direction = self._last_direction.get(track_id, self.UNKNOWN)
            color = self._color_for_direction(direction)

            cv2.polylines(overlay, [points], isClosed=False, color=color, thickness=thickness)
            cv2.circle(overlay, tuple(points[-1]), radius=max(2, thickness + 1), color=color, thickness=-1)

            if len(points) >= 2:
                cv2.arrowedLine(
                    overlay,
                    tuple(points[-2]),
                    tuple(points[-1]),
                    color=color,
                    thickness=max(1, thickness - 1),
                    tipLength=0.4,
                )

            label = f"{track_id}: {direction}"
            text_origin = (points[-1][0] + 6, points[-1][1] + 6)
            cv2.putText(
                overlay,
                label,
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                max(1, thickness - 1),
                cv2.LINE_AA,
            )

        return overlay


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

    def render_direction_debug(self, frame: np.ndarray) -> np.ndarray:
        """Отрисовывает историю направления треков на кадре (для отладки)."""

        if not self.direction_estimator:
            return frame

        return self.direction_estimator.render_debug(frame)

