#/anpr/postprocessing/validator.py
"""Постпроцессинг и валидация номеров с поддержкой плагинов стран."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

from .country_config import CountryConfig, CountryConfigLoader


@dataclass
class PlatePostprocessResult:
    original: str
    normalized: str
    plate: str
    country: Optional[str]
    is_valid: bool
    format_name: Optional[str] = None


class PlatePostProcessor:
    """Выполняет коррекцию и валидацию номеров после OCR."""

    DEFAULT_DIGIT_TO_LETTER = {
        "0": "O",
        "1": "I",
        "2": "Z",
        "3": "E",
        "4": "A",
        "5": "S",
        "6": "G",
        "7": "T",
        "8": "B",
    }
    DEFAULT_LETTER_TO_DIGIT = {value: key for key, value in DEFAULT_DIGIT_TO_LETTER.items()}

    def __init__(self, config_loader: CountryConfigLoader, enabled_countries: Optional[Iterable[str]] = None) -> None:
        self.loader = config_loader
        self.countries: List[CountryConfig] = self.loader.load(enabled_countries)

    @staticmethod
    def _normalize(raw: str) -> str:
        cleaned = re.sub(r"[^0-9A-Za-zА-ЯЁ]+", "", raw or "")
        normalized = cleaned.upper().replace("Ё", "Е")
        return normalized

    def _apply_corrections(self, text: str, country: CountryConfig) -> str:
        corrected = text
        for mistake in country.corrections.common_mistakes:
            src = mistake.get("from", "")
            dst = mistake.get("to", "")
            if src:
                corrected = corrected.replace(src, dst)
        digit_to_letter = {**self.DEFAULT_DIGIT_TO_LETTER, **country.corrections.digit_to_letter}
        letter_to_digit = {**self.DEFAULT_LETTER_TO_DIGIT, **country.corrections.letter_to_digit}
        for src, dst in digit_to_letter.items():
            corrected = corrected.replace(src, dst)
        for src, dst in letter_to_digit.items():
            corrected = corrected.replace(src, dst)
        return corrected

    def _valid_characters(self, text: str, country: CountryConfig) -> bool:
        allowed = set(country.valid_digits + country.valid_letters)
        return all(ch in allowed for ch in text)

    @staticmethod
    def _contains_invalid_sequences(text: str, sequences: List[str]) -> bool:
        return any(seq and seq in text for seq in sequences)

    def _match_country(self, text: str, country: CountryConfig) -> Optional[str]:
        for fmt in country.formats:
            if fmt.pattern.fullmatch(text):
                return fmt.name
        return None

    def _check_stop_words(self, text: str, stop_words: List[str]) -> bool:
        return any(text == stop_word for stop_word in stop_words)

    def _score_candidate(self, candidate: str, country: CountryConfig) -> float:
        score = 0.0
        frequency_weight = country.frequency_dictionary.entries.get(candidate)
        if frequency_weight:
            score += frequency_weight
        for rule in country.language_model.rules:
            if rule.pattern.fullmatch(candidate):
                score += rule.weight
        return score

    def _variants(self, normalized: str, country: CountryConfig, apply_corrections: bool = True) -> List[str]:
        variants = [normalized]
        if apply_corrections:
            corrected = self._apply_corrections(normalized, country)
            if corrected and corrected not in variants:
                variants.append(corrected)
        return variants

    def correct_text(self, raw_text: str) -> str:
        normalized = self._normalize(raw_text)
        if not self.countries:
            return normalized

        best_candidate = normalized
        best_score = max(self._score_candidate(normalized, country) for country in self.countries)

        for country in self.countries:
            corrected = self._apply_corrections(normalized, country)
            if not corrected:
                continue
            score = self._score_candidate(corrected, country)
            if score > best_score:
                best_score = score
                best_candidate = corrected

        if best_score > 0:
            return best_candidate

        primary_country = self.countries[0]
        corrected = self._apply_corrections(normalized, primary_country)
        return corrected or normalized

    def process(self, raw_text: str, apply_corrections: bool = True) -> PlatePostprocessResult:
        normalized = self._normalize(raw_text)
        if not self.countries:
            return PlatePostprocessResult(raw_text, normalized, normalized, None, True, None)

        for country in self.countries:
            for candidate in self._variants(normalized, country, apply_corrections=apply_corrections):
                if not candidate:
                    continue

                if self._check_stop_words(candidate, country.stop_words):
                    return PlatePostprocessResult(raw_text, normalized, "", country.code, False, None)

                if self._contains_invalid_sequences(candidate, country.invalid_sequences):
                    continue

                if country.valid_letters and not self._valid_characters(candidate, country):
                    continue

                format_name = self._match_country(candidate, country)
                if format_name:
                    return PlatePostprocessResult(raw_text, normalized, candidate, country.code, True, format_name)

        return PlatePostprocessResult(raw_text, normalized, normalized, None, False, None)
