from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

from .registry import CountryRegistry, CountryRule


@dataclass
class ProcessedPlate:
    value: str
    country_code: Optional[str] = None
    country_name: Optional[str] = None
    format_name: Optional[str] = None
    valid: bool = False
    raw_value: Optional[str] = None


class PlatePostProcessor:
    """Валидатор госномеров с поддержкой плагинов стран."""

    DEFAULT_STOP_WORDS = {"TEST", "SAMPLE"}

    def __init__(self, registry: CountryRegistry, allowed_countries: Optional[Iterable[str]] = None) -> None:
        self.registry = registry
        allowed_set = {c.upper() for c in allowed_countries} if allowed_countries else set()
        self.allowed_countries = allowed_set or set(registry.available_codes())

    def process(self, text: str) -> ProcessedPlate:
        cleaned = self._normalize(text)
        for country in self._iter_countries():
            candidate = self._apply_country_corrections(cleaned, country)
            if not self._is_allowed_characters(candidate, country):
                continue
            if self._is_blocked(candidate, country):
                continue
            matched_format = self._match_format(candidate, country)
            if matched_format:
                return ProcessedPlate(
                    value=candidate,
                    country_code=country.code,
                    country_name=country.name,
                    format_name=matched_format,
                    valid=True,
                    raw_value=text,
                )
        return ProcessedPlate(value=cleaned, valid=False, raw_value=text)

    def _iter_countries(self) -> List[CountryRule]:
        return [c for c in self.registry.countries if c.code in self.allowed_countries]

    def _normalize(self, text: str) -> str:
        return re.sub(r"[\s\-]", "", text).upper()

    def _apply_country_corrections(self, value: str, country: CountryRule) -> str:
        normalized = value
        # Специальные замены для конкретных ошибок распознавания
        for correction in country.corrections.common_mistakes:
            src = str(correction.get("from", "")).upper()
            dst = str(correction.get("to", "")).upper()
            normalized = normalized.replace(src, dst)

        # Цифры -> буквы
        for digit, letter in country.corrections.digit_to_letter.items():
            normalized = normalized.replace(str(digit), str(letter).upper())

        # Латиница -> кириллица (только для стран с кириллическим алфавитом)
        if country.alphabet.lower() == "cyrillic":
            for latin, cyr in country.corrections.latin_to_cyrillic.items():
                normalized = normalized.replace(latin.upper(), cyr.upper())

        return normalized

    def _is_allowed_characters(self, value: str, country: CountryRule) -> bool:
        if not value:
            return False
        letters = country.valid_letters or ""
        digits = country.valid_digits or "0123456789"
        allowed = set(letters + digits)
        return all(ch in allowed for ch in value)

    def _is_blocked(self, value: str, country: CountryRule) -> bool:
        # Стоп-слова
        upper_value = value.upper()
        if upper_value in self.DEFAULT_STOP_WORDS:
            return True
        if upper_value in {w.upper() for w in country.stop_words}:
            return True

        # Простые проверки последовательностей
        if re.search(r"(.)\1{3,}", value):
            return True
        if re.search(r"(0123|1234|2345|3456|4567|5678|6789)", value):
            return True
        for seq in country.invalid_sequences:
            if seq and seq.upper() in upper_value:
                return True
        return False

    def _match_format(self, value: str, country: CountryRule) -> Optional[str]:
        for plate_format in country.formats:
            if not plate_format.regex:
                continue
            if re.match(plate_format.regex, value):
                return plate_format.name
        return None

