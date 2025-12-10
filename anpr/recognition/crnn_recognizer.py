# /anpr/recognition/crnn_recognizer.py
"""Обертка для квантованной CRNN-модели."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.ao.quantization import QConfigMapping
from torchvision import transforms

from anpr.config import ModelConfig
from anpr.recognition.crnn import CRNN
from logging_manager import get_logger

logger = get_logger(__name__)


class CRNNRecognizer:
    """Подготовка, загрузка и инференс CRNN."""

    def __init__(self, model_path: str, device: torch.device) -> None:
        self.device = device
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Grayscale(),
                transforms.Resize((ModelConfig.OCR_IMG_HEIGHT, ModelConfig.OCR_IMG_WIDTH)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        self.int_to_char: Dict[int, str] = {i + 1: char for i, char in enumerate(ModelConfig.OCR_ALPHABET)}
        self.int_to_char[0] = ""

        num_classes = len(ModelConfig.OCR_ALPHABET) + 1

        model_to_load = CRNN(num_classes).eval()
        qconfig_mapping = QConfigMapping().set_global(torch.ao.quantization.get_default_qconfig("fbgemm"))
        example_inputs = (torch.randn(1, 1, ModelConfig.OCR_IMG_HEIGHT, ModelConfig.OCR_IMG_WIDTH),)
        model_prepared = quantize_fx.prepare_fx(model_to_load, qconfig_mapping, example_inputs)
        model_quantized = quantize_fx.convert_fx(model_prepared)

        model_quantized.load_state_dict(torch.load(model_path, map_location=device))
        self.model = model_quantized
        logger.info("Распознаватель OCR (INT8) успешно загружен (model=%s, device=%s)", model_path, device)

    @torch.no_grad()
    def recognize(self, plate_image) -> Tuple[str, float]:
        preprocessed_plate = self.transform(plate_image).unsqueeze(0).to(self.device)
        preds = self.model(preprocessed_plate)
        return self._decode_with_confidence(preds)

    def _decode_with_confidence(self, log_probs: torch.Tensor) -> Tuple[str, float]:
        probs = log_probs.permute(1, 0, 2)[0]
        time_steps = probs.size(0)

        decoded_chars: List[str] = []
        char_confidences: List[float] = []
        last_char_idx = 0

        for t in range(time_steps):
            timestep_log_probs = probs[t]
            char_idx = int(torch.argmax(timestep_log_probs).item())
            char_conf = float(torch.exp(torch.max(timestep_log_probs)).item())

            if char_idx != 0 and char_idx != last_char_idx:
                decoded_chars.append(self.int_to_char.get(char_idx, ""))
                char_confidences.append(char_conf)

            last_char_idx = char_idx

        text = "".join(decoded_chars)
        if not char_confidences:
            return text, 0.0

        avg_confidence = sum(char_confidences) / len(char_confidences)
        return text, avg_confidence

