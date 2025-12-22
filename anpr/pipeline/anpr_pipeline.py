# /anpr/pipeline/anpr_pipeline.py
"""Пайплайн объединяющий детекцию и OCR."""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from anpr.config import Config
from anpr.postprocessing.validator import PlatePostProcessor
from anpr.recognition.crnn_recognizer import CRNNRecognizer


@dataclass
class BestShotCandidate:
    score: float
    bbox: list[int]
    plate_image: np.ndarray
    frame: np.ndarray
    detection_confidence: float
    direction: str


class BestShotCollector:
    """Собирает лучшие кадры по треку на основе качества."""

    def __init__(self, max_shots: int, stale_seconds: float) -> None:
        self.max_shots = max(1, max_shots)
        self.stale_seconds = max(0.1, stale_seconds)
        self._candidates: Dict[int, List[BestShotCandidate]] = {}
        self._last_seen: Dict[int, float] = {}

    def update_seen(self, track_id: int, now: float) -> None:
        self._last_seen[track_id] = now

    def should_store(self, track_id: int, score: float) -> bool:
        bucket = self._candidates.get(track_id, [])
        if len(bucket) < self.max_shots:
            return True
        return score > min(bucket, key=lambda candidate: candidate.score).score

    def add_candidate(self, track_id: int, candidate: BestShotCandidate) -> None:
        bucket = self._candidates.setdefault(track_id, [])
        bucket.append(candidate)
        bucket.sort(key=lambda item: item.score, reverse=True)
        if len(bucket) > self.max_shots:
            bucket.pop()

    def pop_ready_tracks(self, now: float) -> Dict[int, List[BestShotCandidate]]:
        ready: Dict[int, List[BestShotCandidate]] = {}
        for track_id, candidates in list(self._candidates.items()):
            last_seen = self._last_seen.get(track_id, 0.0)
            if len(candidates) >= self.max_shots or (now - last_seen) >= self.stale_seconds:
                ready[track_id] = candidates
                self._candidates.pop(track_id, None)
        return ready

    def cleanup_stale_tracks(self, now: float) -> List[int]:
        stale_tracks = [
            track_id
            for track_id, last_seen in list(self._last_seen.items())
            if (now - last_seen) >= self.stale_seconds
        ]
        for track_id in stale_tracks:
            self._last_seen.pop(track_id, None)
            self._candidates.pop(track_id, None)
        return stale_tracks


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
        track_stale_seconds: float = 1.0,
    ) -> None:
        self.recognizer = recognizer
        self.best_shots = max(1, best_shots)
        self._bestshot_collector = BestShotCollector(self.best_shots, track_stale_seconds)
        self._completed_tracks: Dict[int, str] = {}
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

    def _score_quality(self, roi: np.ndarray, detection_confidence: float, frame_shape: tuple[int, int, int]) -> float:
        if roi.size == 0:
            return 0.0

        height, width = roi.shape[:2]
        frame_area = float(frame_shape[0] * frame_shape[1]) if frame_shape else 1.0
        area_ratio = (width * height) / max(frame_area, 1.0)
        size_target = 0.04
        size_score = min(1.0, area_ratio / size_target) if size_target > 0 else 0.0

        aspect_ratio = width / max(height, 1)
        expected_ratio = 4.0
        aspect_score = max(0.0, 1.0 - abs(aspect_ratio - expected_ratio) / expected_ratio)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        focus_measure = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness_score = min(1.0, focus_measure / 150.0)

        brightness = float(gray.mean())
        brightness_score = 1.0 - abs(brightness - 127.5) / 127.5
        brightness_score = max(0.0, min(1.0, brightness_score))

        detection_score = max(0.0, min(1.0, detection_confidence))

        return (
            0.35 * detection_score
            + 0.25 * sharpness_score
            + 0.2 * size_score
            + 0.1 * aspect_score
            + 0.1 * brightness_score
        )

    def _aggregate_text(self, results: list[tuple[str, float]]) -> tuple[str, float]:
        filtered = [(text, max(0.0, conf)) for text, conf in results if text]
        if not filtered:
            return "", 0.0

        weights: Dict[str, float] = {}
        counts: Counter[str] = Counter()
        total_weight = 0.0
        for entry_text, entry_confidence in filtered:
            weights[entry_text] = weights.get(entry_text, 0.0) + entry_confidence
            counts[entry_text] += 1
            total_weight += entry_confidence

        if not weights or total_weight <= 0:
            return "", 0.0

        consensus = max(weights, key=lambda value: (weights[value], counts[value]))
        quorum = max(1, min(len(filtered), self.best_shots) // 2 + 1)
        has_quorum = counts[consensus] >= quorum
        has_weighted_majority = weights[consensus] >= total_weight * 0.5
        if not has_quorum or not has_weighted_majority:
            return "", 0.0

        average_confidence = sum(
            conf for text, conf in filtered if text == consensus
        ) / max(1, counts[consensus])
        return consensus, average_confidence

    def process_frame(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        now = time.monotonic()
        results: List[Dict[str, Any]] = []
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
            track_id = detection.get("track_id")
            detection_confidence = float(detection.get("confidence", 0.0))

            if track_id is None:
                if roi.size > 0:
                    processed_plate = self._preprocess_plate(roi)
                    if processed_plate.size > 0:
                        plate_inputs.append(processed_plate)
                        detection_indices.append(idx)
                continue

            track_id_int = int(track_id)
            self._bestshot_collector.update_seen(track_id_int, now)
            if track_id_int in self._completed_tracks:
                continue

            quality_score = self._score_quality(roi, detection_confidence, frame.shape)
            if not self._bestshot_collector.should_store(track_id_int, quality_score):
                continue

            processed_plate = self._preprocess_plate(roi)
            if processed_plate.size == 0:
                continue

            candidate = BestShotCandidate(
                score=quality_score,
                bbox=[int(x1), int(y1), int(x2), int(y2)],
                plate_image=processed_plate,
                frame=frame.copy(),
                detection_confidence=detection_confidence,
                direction=str(detection.get("direction") or TrackDirectionEstimator.UNKNOWN),
            )
            self._bestshot_collector.add_candidate(track_id_int, candidate)

        if plate_inputs:
            batch_results = self.recognizer.recognize_batch(plate_inputs)
            for detection_idx, (current_text, confidence) in zip(detection_indices, batch_results):
                detection = detections[detection_idx]
                if confidence < self.min_confidence:
                    detection["text"] = "Нечитаемо"
                    detection["unreadable"] = True
                    detection["confidence"] = confidence
                    results.append(detection)
                    continue

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

                if self.cooldown_seconds > 0 and detection.get("text"):
                    if self._on_cooldown(detection["text"]):
                        detection["text"] = ""
                    else:
                        self._touch_plate(detection["text"])

                results.append(detection)

        ready_tracks = self._bestshot_collector.pop_ready_tracks(now)
        if ready_tracks:
            track_inputs: List[np.ndarray] = []
            mapping: List[tuple[int, BestShotCandidate]] = []
            for track_id, candidates in ready_tracks.items():
                for candidate in candidates:
                    track_inputs.append(candidate.plate_image)
                    mapping.append((track_id, candidate))

            ocr_results = self.recognizer.recognize_batch(track_inputs) if track_inputs else []
            track_results: Dict[int, List[tuple[str, float, BestShotCandidate]]] = {}
            for (track_id, candidate), (text, confidence) in zip(mapping, ocr_results):
                track_results.setdefault(track_id, []).append((text, confidence, candidate))

            for track_id, items in track_results.items():
                consensus, avg_conf = self._aggregate_text(
                    [(text, confidence) for text, confidence, _ in items if confidence >= self.min_confidence]
                )
                best_candidate = max(items, key=lambda item: item[2].score)[2]

                if not consensus:
                    continue

                result = {
                    "track_id": track_id,
                    "text": consensus,
                    "confidence": avg_conf,
                    "bbox": best_candidate.bbox,
                    "frame": best_candidate.frame,
                    "direction": best_candidate.direction,
                }

                if self.postprocessor and result.get("text"):
                    processed = self.postprocessor.process(result["text"])
                    result["original_text"] = result.get("text")
                    if processed.is_valid:
                        result["text"] = processed.plate
                    elif processed.plate:
                        result["text"] = processed.plate or result.get("text")
                    else:
                        result["text"] = ""
                    result["country"] = processed.country
                    result["format"] = processed.format_name
                    result["validated"] = processed.is_valid

                if self.cooldown_seconds > 0 and result.get("text"):
                    if self._on_cooldown(result["text"]):
                        result["text"] = ""
                    else:
                        self._touch_plate(result["text"])

                if result.get("text"):
                    self._completed_tracks[track_id] = result["text"]
                    results.append(result)

        stale_tracks = self._bestshot_collector.cleanup_stale_tracks(now)
        for track_id in stale_tracks:
            self._completed_tracks.pop(track_id, None)

        return results
