"""Правила постобработки результатов OCR."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from .base import CountryProfile
from .registry import DEFAULT_CONFIG_DIR, filter_profiles, load_country_profiles


@dataclass(frozen=True)
class PlateValidationResult:
    raw_text: str
    normalized_text: str
    country_code: Optional[str]
    country_name: Optional[str]
    plate_format: Optional[str]
    is_valid: bool
    reason: Optional[str] = None
    debug_log: List[str] | None = None


class PlatePostProcessor:
    """Валидирует и исправляет результаты OCR по конфигурациям стран."""

    LATIN_TO_CYRILLIC: Dict[str, str] = {
        "A": "А",
        "B": "В",
        "C": "С",
        "E": "Е",
        "H": "Н",
        "K": "К",
        "M": "М",
        "O": "О",
        "P": "Р",
        "T": "Т",
        "X": "Х",
        "Y": "У",
    }
    DIGIT_TO_LETTER: Dict[str, str] = {"0": "О", "1": "Т", "3": "З", "5": "S", "6": "Б", "8": "В"}

    def __init__(self, config_dir: str | None = None) -> None:
        self.profiles = load_country_profiles(config_dir or DEFAULT_CONFIG_DIR)
        self._stop_words = {"TEST", "SAMPLE"}

    @staticmethod
    def _sanitize(text: str, strip_chars: str) -> str:
        pattern = f"[{' '.join(re.escape(ch) for ch in strip_chars)}\\s]"
        return re.sub(pattern, "", text).upper()

    def _replace_latin_to_cyrillic(self, text: str) -> str:
        return "".join(self.LATIN_TO_CYRILLIC.get(ch, ch) for ch in text)

    def _apply_corrections(self, text: str, profile: CountryProfile) -> str:
        corrected = text
        for rule in profile.corrections:
            corrected = corrected.replace(rule.src, rule.dst)
        if profile.uses_cyrillic:
            corrected = self._replace_latin_to_cyrillic(corrected)
            corrected = "".join(self.DIGIT_TO_LETTER.get(ch, ch) for ch in corrected)
        return corrected

    def _has_illegal_symbols(self, text: str, profile: CountryProfile) -> bool:
        valid = set(profile.letters + profile.digits)
        return any(ch not in valid for ch in text)

    def _is_repetitive(self, text: str, profile: CountryProfile) -> bool:
        if len(text) >= profile.max_repeated and len(set(text)) == 1:
            return True
        if text.isdigit() and text in ("0123", "1234", "2345", "3456", "4567", "5678", "6789"):
            return True
        return False

    def process(self, text: str, allowed_countries: Optional[List[str]] = None) -> PlateValidationResult:
        steps: List[str] = []
        cleaned_text = text.strip()
        steps.append(f"input: '{text}'")
        if not cleaned_text:
            steps.append("reject: empty input after stripping")
            return PlateValidationResult("", "", None, None, None, False, reason="empty", debug_log=steps)

        profiles = filter_profiles(self.profiles, allowed_countries)
        for profile in profiles:
            candidate = self._sanitize(cleaned_text, profile.strip_chars)
            steps.append(
                f"{profile.code}: sanitize -> '{candidate}' (strip='{profile.strip_chars}')"
            )
            candidate = self._apply_corrections(candidate, profile)
            steps.append(f"{profile.code}: corrections -> '{candidate}'")

            if candidate.upper() in self._stop_words or candidate.upper() in profile.stop_words:
                steps.append(f"{profile.code}: reject stop-word '{candidate}'")
                return PlateValidationResult(cleaned_text, candidate, None, None, None, False, "stop_word", steps)
            if self._is_repetitive(candidate, profile):
                steps.append(f"{profile.code}: reject repetitive sequence '{candidate}'")
                return PlateValidationResult(cleaned_text, candidate, None, None, None, False, "repetitive", steps)
            if self._has_illegal_symbols(candidate, profile):
                steps.append(f"{profile.code}: skip due to illegal symbols in '{candidate}'")
                continue

            for fmt in profile.formats:
                if fmt.regex.match(candidate):
                    steps.append(f"{profile.code}: matched format '{fmt.name}' with '{candidate}'")
                    return PlateValidationResult(
                        raw_text=cleaned_text,
                        normalized_text=candidate,
                        country_code=profile.code,
                        country_name=profile.name,
                        plate_format=fmt.name,
                        is_valid=True,
                        debug_log=steps,
                    )

        steps.append("reject: no profile matched")
        return PlateValidationResult(cleaned_text, cleaned_text, None, None, None, False, "no_match", steps)
