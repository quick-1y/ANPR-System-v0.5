#!/usr/bin/env python3
# /anpr/workers/channel_worker.py
import asyncio
import json
import os
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
import threading
import atexit
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui

from anpr.detection.motion_detector import MotionDetector, MotionDetectorConfig
from anpr.pipeline.factory import build_components
from anpr.infrastructure.logging_manager import get_logger
from anpr.infrastructure.storage import AsyncEventDatabase

logger = get_logger(__name__)


@dataclass
class Region:
    """Произвольная область кадра, заданная точками."""

    points: List[Tuple[float, float]]
    unit: str = "px"

    @classmethod
    def from_dict(cls, region_conf: Optional[Dict[str, Any]]) -> "Region":
        if not region_conf:
            return cls(points=[], unit="px")

        unit = str(region_conf.get("unit", "px")).lower()
        raw_points = region_conf.get("points")
        if raw_points:
            points = [
                (float(p.get("x", 0)), float(p.get("y", 0)))
                for p in raw_points
                if isinstance(p, dict)
            ]
            return cls(points=points, unit=unit if unit in ("px", "percent") else "px")

        x = float(region_conf.get("x", 0))
        y = float(region_conf.get("y", 0))
        width = float(region_conf.get("width", 100))
        height = float(region_conf.get("height", 100))
        rect_points = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
        return cls(points=rect_points, unit="percent")

    def _clamp_points(self, points: List[Tuple[int, int]], width: int, height: int) -> List[Tuple[int, int]]:
        if not points:
            return []
        clamped: List[Tuple[int, int]] = []
        for x, y in points:
            clamped.append(
                (
                    max(0, min(width - 1, int(round(x)))),
                    max(0, min(height - 1, int(round(y)))),
                )
            )
        return clamped

    def polygon_points(self, frame_shape: Tuple[int, int, int]) -> List[Tuple[int, int]]:
        height, width, _ = frame_shape
        if not self.points:
            return [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]

        if self.unit == "percent":
            scaled = [
                (width * x / 100.0, height * y / 100.0)
                for (x, y) in self.points
            ]
        else:
            scaled = self.points

        return self._clamp_points([(int(x), int(y)) for x, y in scaled], width, height)

    def bounding_rect(self, frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        polygon = self.polygon_points(frame_shape)
        if not polygon:
            height, width, _ = frame_shape
            return 0, 0, width, height

        xs, ys = zip(*polygon)
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        return x1, y1, x2 + 1, y2 + 1

    def is_full_frame(self) -> bool:
        return not self.points


@dataclass
class DebugOptions:
    show_detection_boxes: bool = False
    show_ocr_text: bool = False

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "DebugOptions":
        debug_conf = data or {}
        return cls(
            show_detection_boxes=bool(debug_conf.get("show_detection_boxes", False)),
            show_ocr_text=bool(debug_conf.get("show_ocr_text", False)),
        )


@dataclass
class PlateSize:
    """Порог размеров номера в пикселях."""

    width: int = 0
    height: int = 0

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PlateSize":
        size = data or {}
        try:
            width = max(0, int(size.get("width", 0)))
            height = max(0, int(size.get("height", 0)))
        except (TypeError, ValueError):
            width, height = 0, 0
        return cls(width=width, height=height)

    def as_tuple(self) -> Tuple[int, int]:
        return self.width, self.height


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
    debug: DebugOptions
    min_plate_size: PlateSize
    max_plate_size: PlateSize

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
            region=Region.from_dict(channel_conf.get("region")),
            debug=DebugOptions.from_dict(channel_conf.get("debug")),
            min_plate_size=PlateSize.from_dict(channel_conf.get("min_plate_size")),
            max_plate_size=PlateSize.from_dict(channel_conf.get("max_plate_size")),
        )


# Хранилище для ProcessPoolExecutor по типу конфигурации
_EXECUTORS: dict[str, ProcessPoolExecutor] = {}
_EXECUTOR_LOCK = threading.Lock()


def _config_fingerprint(config: dict) -> str:
    """Детерминированный отпечаток для разделения экзекьюторов."""
    return json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _get_executor_for_config(config: dict) -> ProcessPoolExecutor:
    """Возвращает ProcessPoolExecutor для конкретной конфигурации."""
    key = _config_fingerprint(config)
    with _EXECUTOR_LOCK:
        executor = _EXECUTORS.get(key)
        if executor is None:
            executor = ProcessPoolExecutor(max_workers=1)
            _EXECUTORS[key] = executor
    return executor


def _shutdown_executors() -> None:
    """Завершает все экзекьюторы при выходе из программы."""
    with _EXECUTOR_LOCK:
        for executor in _EXECUTORS.values():
            executor.shutdown(cancel_futures=True)
        _EXECUTORS.clear()


atexit.register(_shutdown_executors)


def _offset_detections_process(
    detections: list[dict], roi_rect: Tuple[int, int, int, int]
) -> list[dict]:
    """Смещает координаты детекций относительно ROI."""
    x1, y1, _, _ = roi_rect
    adjusted: list[dict] = []
    for det in detections:
        box = det.get("bbox")
        if not box:
            continue
        det_copy = det.copy()
        det_copy["bbox"] = [
            int(box[0] + x1),
            int(box[1] + y1),
            int(box[2] + x1),
            int(box[3] + y1),
        ]
        adjusted.append(det_copy)
    return adjusted


def _filter_by_size(
    detections: list[dict], min_size: Tuple[int, int], max_size: Tuple[int, int]
) -> list[dict]:
    """Отсекает детекции, не попадающие в диапазон размеров."""

    min_w, min_h = (max(0, int(min_size[0])), max(0, int(min_size[1])))
    max_w, max_h = (max(0, int(max_size[0])), max(0, int(max_size[1])))

    filtered: list[dict] = []
    for det in detections:
        box = det.get("bbox")
        if not box or len(box) != 4:
            continue

        width = int(box[2]) - int(box[0])
        height = int(box[3]) - int(box[1])

        if min_w and width < min_w:
            continue
        if min_h and height < min_h:
            continue
        if max_w and width > max_w:
            continue
        if max_h and height > max_h:
            continue

        filtered.append(det)

    return filtered


def _run_inference_task(
    frame: cv2.Mat,
    roi_frame: cv2.Mat,
    roi_rect: Tuple[int, int, int, int],
    config: dict,
) -> Tuple[list[dict], list[dict]]:
    """
    Выполняет инференс в отдельном процессе.
    Модели кэшируются локально в каждом процессе.
    """
    
    # Локальный кэш моделей для этого процесса
    if not hasattr(_run_inference_task, "_local_cache"):
        _run_inference_task._local_cache = {}
    
    # Создаем ключ для кэширования
    key = _config_fingerprint(config)
    
    # Получаем или создаем модели в этом процессе
    if key not in _run_inference_task._local_cache:
        logger.debug(f"Создание моделей inference для конфига {key[:32]}...")
        pipeline, detector = build_components(
            config["best_shots"],
            config["cooldown_seconds"],
            config["min_confidence"],
            config.get("plate_config", {})
        )
        _run_inference_task._local_cache[key] = (pipeline, detector)
    
    pipeline, detector = _run_inference_task._local_cache[key]

    # Выполняем детекцию и распознавание
    detections = detector.track(roi_frame)
    detections = _filter_by_size(
        detections,
        (int(config.get("min_plate_width", 0)), int(config.get("min_plate_height", 0))),
        (int(config.get("max_plate_width", 0)), int(config.get("max_plate_height", 0))),
    )
    detections = _offset_detections_process(detections, roi_rect)
    results = pipeline.process_frame(frame, detections)

    return detections, results


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
        plate_config: Optional[Dict[str, Any]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.config = ChannelRuntimeConfig.from_dict(channel_conf)
        self.reconnect_policy = ReconnectPolicy.from_dict(reconnect_conf)
        self.db_path = db_path
        self.screenshot_dir = screenshot_dir
        os.makedirs(self.screenshot_dir, exist_ok=True)
        self._running = True
        self.plate_config = plate_config or {}

        motion_config = MotionDetectorConfig(
            threshold=self.config.motion_threshold,
            frame_stride=self.config.motion_frame_stride,
            activation_frames=self.config.motion_activation_frames,
            release_frames=self.config.motion_release_frames,
        )
        self.motion_detector = MotionDetector(motion_config)
        self._inference_limiter = InferenceLimiter(self.config.detector_frame_stride)
        self._inference_task: Optional[asyncio.Task] = None
        self._last_debug: Dict[str, list] = {"detections": [], "results": []}

    def _open_capture(self, source: str) -> Optional[cv2.VideoCapture]:
        """Открывает видеопоток с учетом типа источника."""
        try:
            capture = cv2.VideoCapture(int(source) if source.isnumeric() else source)
            if not capture.isOpened():
                logger.warning(f"Не удалось открыть источник: {source}")
                return None
            return capture
        except Exception as e:
            logger.error(f"Ошибка открытия источника {source}: {e}")
            return None

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

    def _inference_config(self) -> dict:
        """Возвращает конфигурацию для inference."""
        plate_config = dict(self.plate_config)
        if plate_config.get("config_dir"):
            plate_config["config_dir"] = os.path.abspath(str(plate_config.get("config_dir")))

        return {
            "best_shots": self.config.best_shots,
            "cooldown_seconds": self.config.cooldown_seconds,
            "min_confidence": self.config.min_confidence,
            "plate_config": plate_config,
            "min_plate_width": self.config.min_plate_size.width,
            "min_plate_height": self.config.min_plate_size.height,
            "max_plate_width": self.config.max_plate_size.width,
            "max_plate_height": self.config.max_plate_size.height,
        }

    def _extract_region(self, frame: cv2.Mat) -> Tuple[cv2.Mat, Tuple[int, int, int, int]]:
        """Извлекает ROI из кадра с учетом произвольной формы."""
        polygon = self.config.region.polygon_points(frame.shape)
        x1, y1, x2, y2 = self.config.region.bounding_rect(frame.shape)
        roi_frame = frame[y1:y2, x1:x2]

        if not self.config.region.is_full_frame() and roi_frame.size:
            local_polygon = np.array([[(x - x1), (y - y1)] for x, y in polygon], dtype=np.int32)
            mask = np.zeros((roi_frame.shape[0], roi_frame.shape[1]), dtype=np.uint8)
            cv2.fillPoly(mask, [local_polygon], 255)
            roi_frame = cv2.bitwise_and(roi_frame, roi_frame, mask=mask)

        return roi_frame, (x1, y1, x2, y2)

    def _motion_detected(self, roi_frame: cv2.Mat) -> bool:
        """Проверяет наличие движения в ROI."""
        if self.config.detection_mode != "motion":
            return True

        return self.motion_detector.update(roi_frame)

    async def _run_inference(
        self, frame: cv2.Mat, roi_frame: cv2.Mat, roi_rect: Tuple[int, int, int, int]
    ) -> Tuple[list[dict], list[dict]]:
        """Запускает инференс в отдельном процессе."""
        loop = asyncio.get_running_loop()
        config = self._inference_config()
        
        return await loop.run_in_executor(
            _get_executor_for_config(config),
            _run_inference_task,
            frame,
            roi_frame,
            roi_rect,
            config,
        )

    @staticmethod
    def _to_qimage(frame: Optional[cv2.Mat], *, is_rgb: bool = False) -> Optional[QtGui.QImage]:
        """Конвертирует OpenCV Mat в QImage."""
        if frame is None or frame.size == 0:
            return None
        
        rgb_frame = frame if is_rgb else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        bytes_per_line = channels * width
        
        # Важно создать копию данных, чтобы избежать проблем с памятью
        return QtGui.QImage(
            rgb_frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888
        ).copy()

    @staticmethod
    def _sanitize_for_filename(value: str) -> str:
        """Очищает строку для использования в имени файла."""
        normalized = value.replace(os.sep, "_")
        safe_chars = [c if c.isalnum() or c in ("-", "_") else "_" for c in normalized]
        return "".join(safe_chars) or "event"

    def _build_screenshot_paths(self, channel_name: str, plate: str) -> Tuple[str, str]:
        """Генерирует пути для сохранения скриншотов."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        channel_safe = self._sanitize_for_filename(channel_name)
        plate_safe = self._sanitize_for_filename(plate or "plate")
        uid = uuid.uuid4().hex[:8]
        base = f"{timestamp}_{channel_safe}_{plate_safe}_{uid}"
        
        return (
            os.path.join(self.screenshot_dir, f"{base}_frame.jpg"),
            os.path.join(self.screenshot_dir, f"{base}_plate.jpg"),
        )

    @staticmethod
    def _draw_label(frame: cv2.Mat, text: str, origin: Tuple[int, int]) -> None:
        if not text:
            return
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        text_size, _ = cv2.getTextSize(text, font, scale, thickness)
        x, y = origin
        x = max(0, x)
        y = max(text_size[1] + 4, y)
        cv2.rectangle(
            frame,
            (x, y - text_size[1] - 6),
            (x + text_size[0] + 8, y + 4),
            (0, 0, 0),
            -1,
        )
        cv2.putText(frame, text, (x + 4, y - 2), font, scale, (0, 255, 0), thickness)

    def _draw_size_guides(self, frame: cv2.Mat) -> None:
        """Отображает пороги min/max размера номера."""

        roi_rect = self.config.region.bounding_rect(frame.shape)
        anchor_x, anchor_y, _, _ = roi_rect
        frame_h, frame_w, _ = frame.shape

        def _draw_box(size: PlateSize, color: Tuple[int, int, int]) -> None:
            if size.width <= 0 or size.height <= 0:
                return

            x2 = min(frame_w - 1, anchor_x + size.width)
            y2 = min(frame_h - 1, anchor_y + size.height)
            cv2.rectangle(frame, (anchor_x, anchor_y), (x2, y2), color, 1, lineType=cv2.LINE_AA)

        _draw_box(self.config.min_plate_size, (0, 200, 0))
        _draw_box(self.config.max_plate_size, (200, 80, 80))

    def _draw_debug_info(self, frame: cv2.Mat) -> None:
        if not (self.config.debug.show_detection_boxes or self.config.debug.show_ocr_text):
            return

        detections = self._last_debug.get("detections", [])
        results = self._last_debug.get("results", [])

        if self.config.debug.show_detection_boxes:
            for det in detections:
                bbox = det.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 200, 0), 2)
                if self.config.debug.show_ocr_text:
                    label = det.get("text") or ""
                    self._draw_label(frame, label, (bbox[0], bbox[1] - 6))

        if self.config.debug.show_ocr_text:
            for res in results:
                text = res.get("text")
                bbox = res.get("bbox") or res.get("plate_bbox")
                if not text or not bbox or len(bbox) != 4:
                    continue
                self._draw_label(frame, str(text), (bbox[0], bbox[1] - 6))

    def _save_bgr_image(self, path: str, image: Optional[cv2.Mat]) -> Optional[str]:
        """Сохраняет BGR изображение на диск."""
        if image is None or image.size == 0:
            return None
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            if cv2.imwrite(path, image):
                return path
        except Exception as e:
            logger.exception("Не удалось сохранить скриншот по пути %s: %s", path, e)
        
        return None

    async def _process_events(
        self,
        storage: AsyncEventDatabase,
        source: str,
        results: list[dict],
        channel_name: str,
        frame: cv2.Mat,
        rgb_frame: Optional[cv2.Mat] = None,
    ) -> None:
        """Обрабатывает результаты распознавания и сохраняет события."""
        for res in results:
            if res.get("unreadable"):
                logger.debug(
                    "Канал %s: номер помечен как нечитаемый (confidence=%.2f)",
                    channel_name,
                    res.get("confidence", 0.0),
                )
                continue
            
            if not res.get("text"):
                continue
            
            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "channel": channel_name,
                "plate": res.get("text", ""),
                "country": res.get("country"),
                "confidence": res.get("confidence", 0.0),
                "source": source,
            }
            
            # Извлекаем область номера для скриншота
            x1, y1, x2, y2 = res.get("bbox", (0, 0, 0, 0))
            plate_crop = frame[y1:y2, x1:x2] if frame is not None else None
            
            # Сохраняем скриншоты
            frame_path, plate_path = self._build_screenshot_paths(channel_name, event["plate"])
            event["frame_path"] = self._save_bgr_image(frame_path, frame)
            event["plate_path"] = self._save_bgr_image(plate_path, plate_crop)
            
            # Готовим изображения для UI
            event["frame_image"] = self._to_qimage(rgb_frame, is_rgb=True)
            event["plate_image"] = self._to_qimage(plate_crop) if plate_crop is not None else None
            
            # Сохраняем в БД
            event["id"] = await storage.insert_event_async(
                channel=event["channel"],
                plate=event["plate"],
                country=event.get("country"),
                confidence=event["confidence"],
                source=event["source"],
                timestamp=event["timestamp"],
                frame_path=event.get("frame_path"),
                plate_path=event.get("plate_path"),
            )
            
            # Отправляем событие в UI
            self.event_ready.emit(event)
            logger.info(
                "Канал %s: зафиксирован номер %s (conf=%.2f, track=%s)",
                event["channel"],
                event["plate"],
                event["confidence"],
                res.get("track_id", "-"),
            )

    async def _inference_and_process(
        self,
        storage: AsyncEventDatabase,
        source: str,
        channel_name: str,
        frame: cv2.Mat,
        roi_frame: cv2.Mat,
        roi_rect: Tuple[int, int, int, int],
        rgb_frame: cv2.Mat,
    ) -> None:
        """Выполняет инференс и обработку результатов."""
        try:
            detections, results = await self._run_inference(frame, roi_frame, roi_rect)
            self._last_debug = {"detections": detections, "results": results}
            await self._process_events(storage, source, results, channel_name, frame, rgb_frame)
        except Exception as e:
            logger.exception("Ошибка инференса для канала %s: %s", channel_name, e)

    async def _loop(self) -> None:
        """Основной цикл обработки канала."""
        storage = AsyncEventDatabase(self.db_path)

        source = self.config.source
        channel_name = self.config.name
        
        # Подключаемся к источнику
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
            
            # Плановое переподключение
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

            # Чтение кадра
            ret, frame = await asyncio.to_thread(capture.read)
            
            # Проверка потери сигнала
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

            # Обработка кадра
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

                # Запуск инференса с учетом stride
                if self._inference_limiter.allow():
                    if self._inference_task is None or self._inference_task.done():
                        self._inference_task = asyncio.create_task(
                            self._inference_and_process(
                                storage,
                                source,
                                channel_name,
                                frame.copy(),
                                roi_frame.copy(),
                                roi_rect,
                                rgb_frame.copy(),
                            )
                        )
                    else:
                        logger.debug(
                            "Канал %s: пропуск инференса, предыдущая задача еще выполняется",
                            channel_name,
                        )

            # Отправка кадра в UI
            display_frame = rgb_frame.copy()
            self._draw_size_guides(display_frame)
            if self.config.debug.show_detection_boxes or self.config.debug.show_ocr_text:
                self._draw_debug_info(display_frame)

            height, width, channel = display_frame.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(
                display_frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888
            ).copy()

            self.frame_ready.emit(channel_name, q_image)

        # Завершение работы
        capture.release()

        if self._inference_task is not None:
            try:
                await asyncio.wait_for(self._inference_task, timeout=1)
            except Exception as e:
                logger.warning("Задача инференса для канала %s не завершена корректно: %s", 
                             channel_name, e)

    def run(self) -> None:
        """Запуск потока."""
        try:
            asyncio.run(self._loop())
        except Exception as exc:
            self.status_ready.emit(self.config.name, f"Ошибка: {exc}")
            logger.exception("Канал %s аварийно остановлен", self.config.name)

    def stop(self) -> None:
        """Остановка потока."""
        self._running = False
