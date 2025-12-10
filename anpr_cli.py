# /anpr_cli.py
"""CLI-обертка для запуска ANPR из командной строки.

Файл сохраняет прежнюю точку входа, но делегирует детекцию, OCR и пайплайн
выделенным модулам для слабой связности.
"""

from __future__ import annotations

import argparse
import os
from typing import List

import cv2

from anpr.config import ModelConfig
from anpr.detection.yolo_detector import YOLODetector
from anpr.pipeline.anpr_pipeline import ANPRPipeline, Visualizer
from anpr.recognition.crnn_recognizer import CRNNRecognizer
from logging_manager import LoggingManager, get_logger

logger = get_logger(__name__)


def _process_video(pipeline: ANPRPipeline, detector: YOLODetector, source_path: str) -> None:
    cap = cv2.VideoCapture(int(source_path) if source_path.isnumeric() else source_path)
    if not cap.isOpened():
        raise IOError("Ошибка открытия видеопотока")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.track(frame)
        results = pipeline.process_frame(frame, detections)
        frame = Visualizer.draw_results(frame, results)
        cv2.imshow("ANPR Result", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def _process_image(pipeline: ANPRPipeline, detector: YOLODetector, source_path: str) -> None:
    frame = cv2.imread(source_path)
    if frame is None:
        raise IOError("Ошибка чтения изображения")

    detections = detector.detect(frame)
    results = pipeline.process_frame(frame, detections)

    print(f"\nНа изображении '{os.path.basename(source_path)}' распознаны номера:")
    if not results:
        print("- Номера не найдены.")
    for res in results:
        print(f"- {res.get('text', '')} (уверенность детектора: {res.get('confidence', 0.0):.2f})")

    frame = Visualizer.draw_results(frame, results)
    cv2.imshow("ANPR Result", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_source(pipeline: ANPRPipeline, detector: YOLODetector, source_path: str) -> None:
    is_video = source_path.endswith((".mp4", ".avi", ".mov")) or source_path.isnumeric()
    if is_video:
        _process_video(pipeline, detector, source_path)
    else:
        _process_image(pipeline, detector, source_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Распознавание автомобильных номеров.")
    parser.add_argument(
        "--source",
        required=True,
        help="Путь к изображению, видеофайлу или ID веб-камеры (например, '0').",
    )
    args = parser.parse_args()

    try:
        LoggingManager()
        detector = YOLODetector(ModelConfig.YOLO_MODEL_PATH, ModelConfig.DEVICE)
        recognizer = CRNNRecognizer(ModelConfig.OCR_MODEL_PATH, ModelConfig.DEVICE)
        pipeline = ANPRPipeline(recognizer, ModelConfig.TRACK_BEST_SHOTS)
        process_source(pipeline, detector, args.source)
    except (IOError, FileNotFoundError) as exc:
        logger.error("Критическая ошибка: %s", exc)
        print(f"Критическая ошибка: {exc}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Непредвиденная ошибка")
        print(f"Непредвиденная ошибка: {exc}")


if __name__ == "__main__":
    main()

