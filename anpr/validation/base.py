"""Базовые структуры и модели для валидации автомобильных номеров."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Pattern
import re


@dataclass
class LicensePlateFormat:
    name: str
    pattern: str
    regex: Pattern[str] = field(init=False)

    def __post_init__(self) -> None:
        self.regex = re.compile(self.pattern)

    def matches(self, value: str) -> bool:
        return bool(self.regex.match(value))


@dataclass
class CorrectionRules:
    digit_to_letter: Dict[str, str] = field(default_factory=dict)
    common_mistakes: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class CountrySpec:
    name: str
    code: str
    priority: int = 100
    license_plate_formats: List[LicensePlateFormat] = field(default_factory=list)
    valid_letters: str = ""
    valid_digits: str = "0123456789"
    corrections: CorrectionRules = field(default_factory=CorrectionRules)
    stop_words: List[str] = field(default_factory=list)
    invalid_sequences: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict) -> "CountrySpec":
        formats = [
            LicensePlateFormat(name=f.get("name", ""), pattern=f.get("regex", ""))
            for f in data.get("license_plate_formats", [])
            if f.get("regex")
        ]
        corrections = CorrectionRules(**(data.get("corrections") or {}))
        valid_chars = data.get("valid_characters", {})
        return cls(
            name=data.get("name", ""),
            code=data.get("code", ""),
            priority=int(data.get("priority", 100)),
            license_plate_formats=formats,
            valid_letters=valid_chars.get("letters", ""),
            valid_digits=valid_chars.get("digits", "0123456789"),
            corrections=corrections,
            stop_words=[word.upper() for word in data.get("stop_words", [])],
            invalid_sequences=[seq.upper() for seq in data.get("invalid_sequences", [])],
        )

    def allows(self, value: str) -> bool:
        if self.valid_letters:
            allowed = set(self.valid_letters + self.valid_digits)
            return all(ch in allowed for ch in value)
        return True

    def match_format(self, value: str) -> Optional[LicensePlateFormat]:
        for fmt in self.license_plate_formats:
            if fmt.matches(value):
                return fmt
        return None


@dataclass
class PlateValidationResult:
    is_valid: bool
    normalized: str
    country_code: Optional[str] = None
    country_name: Optional[str] = None
    format_name: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class ValidatorConfig:
    config_dir: Path
    enabled_countries: List[str]

    @classmethod
    def from_dict(cls, data: Dict) -> "ValidatorConfig":
        config_dir = Path(data.get("config_dir", "anpr/validation/configs"))
        enabled_countries = [code.upper() for code in data.get("enabled_countries", [])]
        return cls(config_dir=config_dir, enabled_countries=enabled_countries)


def sort_specs(specs: Iterable[CountrySpec]) -> List[CountrySpec]:
    """Сортирует спецификации по приоритету и коду страны."""
    return sorted(specs, key=lambda spec: (spec.priority, spec.code))
