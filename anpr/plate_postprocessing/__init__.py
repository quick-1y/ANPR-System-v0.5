"""Валидация и постобработка распознанных госномеров."""

from .validator import PlatePostProcessor, PlateValidationResult

__all__ = ["PlatePostProcessor", "PlateValidationResult"]
