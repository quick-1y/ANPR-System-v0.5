#!/usr/bin/env python3
# /anpr/workers/channel_worker.py
import asyncio
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import cv2
from PyQt5 import QtCore, QtGui

from anpr.detection.motion_detector import MotionDetector, MotionDetectorConfig
from anpr.pipeline.factory import build_components
from logging_manager import get_logger
from storage import AsyncEventDatabase

logger = get_logger(__name__)


@dataclass
class Region:
    """Прямоугольная область в процентах относительно кадра."""

    x: int = 0
    y: int = 0
    width: int = 100
    height: int = 100

    def clamp(self) -> "Region":
        self.x = max(0, min(100, int(self.x)))
        self.y = max(0, min(100, int(self.y)))
        self.width = max(1, min(100 - self.x, int(self.width)))
        self.height = max(1, min(100 - self.y, int(self.height)))
        return self

    def to_rect(self, frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        height, width, _ = frame_shape
        x2_pct = min(100, self.x + self.width)
        y2_pct = min(100, self.y + self.height)

        x1 = int(width * self.x / 100)
        y1 = int(height * self.y / 100)
        x2 = max(x1 + 1, int(width * x2_pct / 100))
        y2 = max(y1 + 1, int(height * y2_pct / 100))
        return x1, y1, x2, y2


@dataclass
class ReconnectPolicy:
    """Политика переподключения канала."""

    enabled: bool
    frame_timeout_seconds: float
    retry_interval_seconds: float
    periodic_enabled: bool
    periodic_reconnect_seconds: float

    @classmethod
    def from_dict(cls, config: Optional[Dict[str, Any]]) -> "ReconnectPolicy":
        reconnect_conf = config or {}
        signal_loss_conf = reconnect_conf.get("signal_loss", {})
        periodic_conf = reconnect_conf.get("periodic", {})
        return cls(
            enabled=bool(signal_loss_conf.get("enabled", False)),
            frame_timeout_seconds=float(signal_loss_conf.get("frame_timeout_seconds", 5)),
            retry_interval_seconds=float(signal_loss_conf.get("retry_interval_seconds", 5)),
            periodic_enabled=bool(periodic_conf.get("enabled", False)),
            periodic_reconnect_seconds=float(periodic_conf.get("interval_minutes", 0)) * 60,
        )


@dataclass
class ChannelRuntimeConfig:
    """Нормализованная конфигурация канала."""

    name: str
    source: str
    best_shots: int
    cooldown_seconds: int
    min_confidence: float
    detector_frame_stride: int
    detection_mode: str
    motion_threshold: float
    motion_frame_stride: int
    motion_activation_frames: int
    motion_release_frames: int
    region: Region

    @classmethod
    def from_dict(cls, channel_conf: Dict[str, Any]) -> "ChannelRuntimeConfig":
        return cls(
            name=channel_conf.get("name", "Канал"),
            source=str(channel_conf.get("source", "0")),
            best_shots=int(channel_conf.get("best_shots", 3)),
            cooldown_seconds=int(channel_conf.get("cooldown_seconds", 5)),
            min_confidence=float(channel_conf.get("ocr_min_confidence", 0.6)),
            detector_frame_stride=max(1, int(channel_conf.get("detector_frame_stride", 2))),
            detection_mode=channel_conf.get("detection_mode", "continuous"),
            motion_threshold=float(channel_conf.get("motion_threshold", 0.01)),
            motion_frame_stride=int(channel_conf.get("motion_frame_stride", 1)),
            motion_activation_frames=int(channel_conf.get("motion_activation_frames", 3)),
            motion_release_frames=int(channel_conf.get("motion_release_frames", 6)),
            region=Region(**(channel_conf.get("region") or {})).clamp(),
        )


class InferenceLimiter:
    """Пропускает лишние кадры для инференса детектора."""

    def __init__(self, stride: int) -> None:
        self.stride = max(1, stride)
        self._counter = 0

    def allow(self) -> bool:
        should_run = self._counter == 0
        self._counter = (self._counter + 1) % self.stride
        return should_run


class ChannelWorker(QtCore.QThread):
    """Background worker that captures frames, runs ANPR pipeline and emits UI events."""

    frame_ready = QtCore.pyqtSignal(str, QtGui.QImage)
    event_ready = QtCore.pyqtSignal(dict)
    status_ready = QtCore.pyqtSignal(str, str)

    def __init__(
        self,
        channel_conf: Dict,
        db_path: str,
        screenshot_dir: str,
        reconnect_conf: Optional[Dict[str, Any]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.config = ChannelRuntimeConfig.from_dict(channel_conf)
        self.reconnect_policy = ReconnectPolicy.from_dict(reconnect_conf)
        self.db_path = db_path
        self.screenshot_dir = screenshot_dir
        os.makedirs(self.screenshot_dir, exist_ok=True)
        self._running = True

        motion_config = MotionDetectorConfig(
            threshold=self.config.motion_threshold,
            frame_stride=self.config.motion_frame_stride,
            activation_frames=self.config.motion_activation_frames,
            release_frames=self.config.motion_release_frames,
        )
        self.motion_detector = MotionDetector(motion_config)
        self._inference_limiter = InferenceLimiter(self.config.detector_frame_stride)

    def _open_capture(self, source: str) -> Optional[cv2.VideoCapture]:
        capture = cv2.VideoCapture(int(source) if source.isnumeric() else source)
        if not capture.isOpened():
            return None
        return capture

    async def _open_with_retries(self, source: str, channel_name: str) -> Optional[cv2.VideoCapture]:
        """Подключает источник с учетом настроек переподключения."""

        while self._running:
            capture = await asyncio.to_thread(self._open_capture, source)
            if capture is not None:
                self.status_ready.emit(channel_name, "")
                return capture

            if not self.reconnect_policy.enabled:
                self.status_ready.emit(channel_name, "Нет сигнала")
                return None

            self.status_ready.emit(
                channel_name,
                f"Нет сигнала, повтор через {int(self.reconnect_policy.retry_interval_seconds)}с",
            )
            await asyncio.sleep(max(0.1, self.reconnect_policy.retry_interval_seconds))
        return None

    def _build_pipeline(self) -> Tuple[object, object]:
        return build_components(
            self.config.best_shots, self.config.cooldown_seconds, self.config.min_confidence
        )

    def _extract_region(self, frame: cv2.Mat) -> Tuple[cv2.Mat, Tuple[int, int, int, int]]:
        x1, y1, x2, y2 = self.config.region.to_rect(frame.shape)
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

    def _motion_detected(self, roi_frame: cv2.Mat) -> bool:
        if self.config.detection_mode != "motion":
            return True

        return self.motion_detector.update(roi_frame)

    @staticmethod
    def _offset_detections(detections: list[dict], roi_rect: Tuple[int, int, int, int]) -> list[dict]:
        x1, y1, _, _ = roi_rect
        adjusted: list[dict] = []
        for det in detections:
            box = det.get("bbox")
            if not box:
                continue
            det_copy = det.copy()
            det_copy["bbox"] = [int(box[0] + x1), int(box[1] + y1), int(box[2] + x1), int(box[3] + y1)]
            adjusted.append(det_copy)
        return adjusted

    @staticmethod
    def _to_qimage(frame: cv2.Mat) -> Optional[QtGui.QImage]:
        if frame is None or frame.size == 0:
            return None
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        bytes_per_line = channels * width
        return QtGui.QImage(
            rgb_frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888
        ).copy()

    @staticmethod
    def _sanitize_for_filename(value: str) -> str:
        normalized = value.replace(os.sep, "_")
        safe_chars = [c if c.isalnum() or c in ("-", "_") else "_" for c in normalized]
        return "".join(safe_chars) or "event"

    def _build_screenshot_paths(self, channel_name: str, plate: str) -> Tuple[str, str]:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        channel_safe = self._sanitize_for_filename(channel_name)
        plate_safe = self._sanitize_for_filename(plate or "plate")
        uid = uuid.uuid4().hex[:8]
        base = f"{timestamp}_{channel_safe}_{plate_safe}_{uid}"
        return (
            os.path.join(self.screenshot_dir, f"{base}_frame.jpg"),
            os.path.join(self.screenshot_dir, f"{base}_plate.jpg"),
        )

    def _save_bgr_image(self, path: str, image: Optional[cv2.Mat]) -> Optional[str]:
        if image is None or image.size == 0:
            return None
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            if cv2.imwrite(path, image):
                return path
        except Exception:  # noqa: BLE001
            logger.exception("Не удалось сохранить скриншот по пути %s", path)
        return None

    async def _process_events(
        self,
        storage: AsyncEventDatabase,
        source: str,
        results: list[dict],
        channel_name: str,
        frame: cv2.Mat,
    ) -> None:
        for res in results:
            if res.get("unreadable"):
                logger.debug(
                    "Канал %s: номер помечен как нечитаемый (confidence=%.2f)",
                    channel_name,
                    res.get("confidence", 0.0),
                )
                continue
            if res.get("text"):
                event = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "channel": channel_name,
                    "plate": res.get("text", ""),
                    "confidence": res.get("confidence", 0.0),
                    "source": source,
                }
                x1, y1, x2, y2 = res.get("bbox", (0, 0, 0, 0))
                plate_crop = frame[y1:y2, x1:x2] if frame is not None else None
                frame_path, plate_path = self._build_screenshot_paths(channel_name, event["plate"])
                event["frame_path"] = self._save_bgr_image(frame_path, frame)
                event["plate_path"] = self._save_bgr_image(plate_path, plate_crop)
                event["frame_image"] = self._to_qimage(frame)
                event["plate_image"] = self._to_qimage(plate_crop) if plate_crop is not None else None
                event["id"] = await storage.insert_event_async(
                    channel=event["channel"],
                    plate=event["plate"],
                    confidence=event["confidence"],
                    source=event["source"],
                    timestamp=event["timestamp"],
                    frame_path=event.get("frame_path"),
                    plate_path=event.get("plate_path"),
                )
                self.event_ready.emit(event)
                logger.info(
                    "Канал %s: зафиксирован номер %s (conf=%.2f, track=%s)",
                    event["channel"],
                    event["plate"],
                    event["confidence"],
                    res.get("track_id", "-"),
                )

    async def _loop(self) -> None:
        pipeline, detector = await asyncio.to_thread(self._build_pipeline)
        storage = AsyncEventDatabase(self.db_path)

        source = self.config.source
        channel_name = self.config.name
        capture = await self._open_with_retries(source, self.config.name)
        if capture is None:
            logger.warning("Не удалось открыть источник %s для канала %s", source, self.config)
            return
        logger.info("Канал %s запущен (источник=%s)", channel_name, source)
        waiting_for_motion = False
        last_frame_ts = time.monotonic()
        last_reconnect_ts = last_frame_ts
        while self._running:
            now = time.monotonic()
            if (
                self.reconnect_policy.periodic_enabled
                and self.reconnect_policy.periodic_reconnect_seconds > 0
                and now - last_reconnect_ts >= self.reconnect_policy.periodic_reconnect_seconds
            ):
                self.status_ready.emit(channel_name, "Плановое переподключение...")
                capture.release()
                capture = await self._open_with_retries(source, channel_name)
                if capture is None:
                    logger.warning("Переподключение не удалось для канала %s", channel_name)
                    break
                last_reconnect_ts = time.monotonic()
                last_frame_ts = last_reconnect_ts
                continue

            ret, frame = await asyncio.to_thread(capture.read)
            if not ret or frame is None:
                if not self.reconnect_policy.enabled:
                    self.status_ready.emit(channel_name, "Поток остановлен")
                    logger.warning("Поток остановлен для канала %s", channel_name)
                    break

                if time.monotonic() - last_frame_ts < self.reconnect_policy.frame_timeout_seconds:
                    await asyncio.sleep(0.05)
                    continue

                self.status_ready.emit(channel_name, "Потеря сигнала, переподключение...")
                logger.warning("Потеря сигнала на канале %s, выполняем переподключение", channel_name)
                capture.release()
                capture = await self._open_with_retries(source, channel_name)
                if capture is None:
                    logger.warning("Переподключение не удалось для канала %s", channel_name)
                    break
                last_reconnect_ts = time.monotonic()
                last_frame_ts = last_reconnect_ts
                continue

            last_frame_ts = time.monotonic()

            roi_frame, roi_rect = self._extract_region(frame)
            motion_detected = self._motion_detected(roi_frame)

            if not motion_detected:
                if not waiting_for_motion and self.config.detection_mode == "motion":
                    self.status_ready.emit(channel_name, "Ожидание движения")
                waiting_for_motion = True
            else:
                if waiting_for_motion:
                    self.status_ready.emit(channel_name, "Движение обнаружено")
                waiting_for_motion = False
                if self._inference_limiter.allow():
                    detections = await asyncio.to_thread(detector.track, roi_frame)
                    detections = self._offset_detections(detections, roi_rect)
                    results = await asyncio.to_thread(pipeline.process_frame, frame, detections)
                    await self._process_events(storage, source, results, channel_name, frame)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_frame.shape
            bytes_per_line = 3 * width
            # Копируем буфер, чтобы предотвратить обращение Qt к уже освобожденной памяти
            # во время перерисовок окна.
            q_image = QtGui.QImage(
                rgb_frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888
            ).copy()
            self.frame_ready.emit(channel_name, q_image)

        capture.release()

    def run(self) -> None:
        try:
            asyncio.run(self._loop())
        except Exception as exc:  # noqa: BLE001
            self.status_ready.emit(self.config.name, f"Ошибка: {exc}")
            logger.exception("Канал %s аварийно остановлен", self.config.name)

    def stop(self) -> None:
        self._running = False
