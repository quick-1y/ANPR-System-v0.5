"""Модуль валидации и постобработки номерных знаков."""
from anpr.validation.base import CountrySpec, PlateValidationResult, ValidatorConfig
from anpr.validation.validator import CountryRegistry, PlatePostProcessor

__all__ = [
    "CountrySpec",
    "PlateValidationResult",
    "ValidatorConfig",
    "CountryRegistry",
    "PlatePostProcessor",
]
