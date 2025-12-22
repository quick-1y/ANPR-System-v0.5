# /anpr/pipeline/plate_skew_correction.py
"""Коррекция угла и перспективы номерного знака."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple

import cv2
import numpy as np

from anpr.config import Config


@dataclass(frozen=True)
class SkewDetection:
    angle: float
    confidence: float
    quad: Optional[np.ndarray] = None


class PlateSkewCorrector:
    """Определяет угол наклона и корректирует перспективу номера."""

    def __init__(
        self,
        min_angle: float = 5.0,
        max_angle: float = 45.0,
        confidence_threshold: float = 0.7,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        config = Config()
        self.min_angle = abs(min_angle)
        self.max_angle = abs(max_angle)
        self.confidence_threshold = max(0.0, min(1.0, confidence_threshold))
        self.target_size = target_size or (config.ocr_width, config.ocr_height)

    def correct(self, plate_image: np.ndarray) -> np.ndarray:
        if plate_image.size == 0:
            return plate_image

        gray, edges, binary = self._preprocess(plate_image)
        strategies: Iterable[Callable[[np.ndarray, np.ndarray, np.ndarray], Optional[SkewDetection]]] = [
            self._detect_by_hough_lines,
            self._detect_by_contours,
            self._detect_by_text_baseline,
            self._detect_by_corners,
        ]

        for strategy in strategies:
            detection = strategy(gray, edges, binary)
            if detection and detection.confidence >= self.confidence_threshold:
                corrected = self._apply_correction(plate_image, detection)
                return self._postprocess(corrected)

        fallback = self._fallback_detection(gray, edges, binary)
        if fallback:
            corrected = self._apply_correction(plate_image, fallback)
            return self._postprocess(corrected)

        return self._postprocess(plate_image)

    def _preprocess(self, plate_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        block_size = max(11, (gray.shape[1] // 8) | 1)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            7,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
        edges = cv2.Canny(cleaned, 50, 150)
        return gray, edges, cleaned

    def _detect_by_hough_lines(
        self, gray: np.ndarray, edges: np.ndarray, binary: np.ndarray
    ) -> Optional[SkewDetection]:
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=60, minLineLength=gray.shape[1] // 3, maxLineGap=12)
        if lines is None:
            return None

        angles = []
        lengths = []
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            angle = np.degrees(np.arctan2(dy, dx))
            angle = self._normalize_angle(angle)
            if abs(angle) > self.max_angle:
                continue
            length = float(np.hypot(dx, dy))
            angles.append(angle)
            lengths.append(length)

        if not angles:
            return None

        angles_array = np.array(angles, dtype=np.float32)
        median_angle = float(np.median(angles_array))
        deviation = float(np.median(np.abs(angles_array - median_angle)))
        length_score = min(1.0, float(np.sum(lengths)) / (gray.shape[1] * 2))
        consistency = max(0.0, 1.0 - deviation / 10.0)
        confidence = min(1.0, length_score * 0.6 + consistency * 0.4)
        return SkewDetection(angle=median_angle, confidence=confidence)

    def _detect_by_contours(
        self, gray: np.ndarray, edges: np.ndarray, binary: np.ndarray
    ) -> Optional[SkewDetection]:
        contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best_detection: Optional[SkewDetection] = None
        image_area = float(gray.shape[0] * gray.shape[1])

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < image_area * 0.1:
                continue
            rect = cv2.minAreaRect(contour)
            (width, height) = rect[1]
            if width <= 0 or height <= 0:
                continue
            aspect_ratio = max(width, height) / min(width, height)
            if not 1.6 <= aspect_ratio <= 6.0:
                continue
            angle = self._rect_angle(rect)
            if abs(angle) > self.max_angle:
                continue
            rectangularity = area / (width * height)
            confidence = min(1.0, (area / image_area) * 2.0) * min(1.0, rectangularity)

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            quad = approx.reshape(4, 2) if len(approx) == 4 else None

            if best_detection is None or confidence > best_detection.confidence:
                best_detection = SkewDetection(angle=angle, confidence=confidence, quad=quad)

        return best_detection

    def _detect_by_text_baseline(
        self, gray: np.ndarray, edges: np.ndarray, binary: np.ndarray
    ) -> Optional[SkewDetection]:
        points = np.column_stack(np.where(edges > 0))
        if points.shape[0] < 150:
            return None
        points = points[:, ::-1]
        line = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy = float(line[0]), float(line[1])
        angle = self._normalize_angle(np.degrees(np.arctan2(vy, vx)))
        if abs(angle) > self.max_angle:
            return None
        confidence = min(1.0, points.shape[0] / (gray.shape[0] * gray.shape[1] * 0.2))
        return SkewDetection(angle=angle, confidence=confidence)

    def _detect_by_corners(
        self, gray: np.ndarray, edges: np.ndarray, binary: np.ndarray
    ) -> Optional[SkewDetection]:
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=5)
        if corners is None:
            return None
        points = corners.reshape(-1, 2)
        if points.shape[0] < 4:
            return None
        rect = cv2.minAreaRect(points)
        angle = self._rect_angle(rect)
        if abs(angle) > self.max_angle:
            return None
        confidence = min(1.0, points.shape[0] / 50.0)
        return SkewDetection(angle=angle, confidence=confidence)

    def _fallback_detection(
        self, gray: np.ndarray, edges: np.ndarray, binary: np.ndarray
    ) -> Optional[SkewDetection]:
        detection = self._detect_by_contours(gray, edges, binary)
        if detection:
            return detection
        detection = self._detect_by_hough_lines(gray, edges, binary)
        if detection:
            return detection
        return None

    def _apply_correction(self, plate_image: np.ndarray, detection: SkewDetection) -> np.ndarray:
        if detection.quad is not None and detection.quad.shape == (4, 2):
            warped = self._four_point_transform(plate_image, detection.quad)
            return warped

        if abs(detection.angle) < self.min_angle:
            return plate_image

        return self._rotate_image(plate_image, detection.angle)

    def _postprocess(self, plate_image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        normalized = cv2.equalizeHist(gray)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(normalized)
        resized = self._resize_with_padding(enhanced, self.target_size)
        return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

    def _resize_with_padding(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        target_width, target_height = target_size
        height, width = image.shape[:2]
        if width == 0 or height == 0:
            return image
        scale = min(target_width / width, target_height / height)
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        pad_left = (target_width - new_width) // 2
        pad_right = target_width - new_width - pad_left
        pad_top = (target_height - new_height) // 2
        pad_bottom = target_height - new_height - pad_top
        return cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        height, width = image.shape[:2]
        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

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
        width_a = np.hypot(br[0] - bl[0], br[1] - bl[1])
        width_b = np.hypot(tr[0] - tl[0], tr[1] - tl[1])
        max_width = max(int(width_a), int(width_b))
        height_a = np.hypot(tr[0] - br[0], tr[1] - br[1])
        height_b = np.hypot(tl[0] - bl[0], tl[1] - bl[1])
        max_height = max(int(height_a), int(height_b))
        if max_width <= 0 or max_height <= 0:
            return image
        dst = np.array(
            [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype="float32"
        )
        transform = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, transform, (max_width, max_height), flags=cv2.INTER_CUBIC)

    def _normalize_angle(self, angle: float) -> float:
        angle = (angle + 180.0) % 180.0 - 90.0
        if angle > 45.0:
            angle -= 90.0
        elif angle < -45.0:
            angle += 90.0
        return float(angle)

    def _rect_angle(self, rect: Tuple[Tuple[float, float], Tuple[float, float], float]) -> float:
        width, height = rect[1]
        angle = rect[2]
        if width < height:
            angle += 90.0
        return self._normalize_angle(angle)
