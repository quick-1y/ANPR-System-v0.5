"""Правила постобработки и валидации распознанных госномеров."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Pattern, Sequence

import yaml


@dataclass(frozen=True)
class CountryFormat:
    name: str
    regex: Pattern[str]


@dataclass(frozen=True)
class CorrectionRules:
    digit_to_letter: Dict[str, str] = field(default_factory=dict)
    letter_to_digit: Dict[str, str] = field(default_factory=dict)
    common_mistakes: List[tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class CountryProfile:
    name: str
    code: str
    priority: int
    formats: List[CountryFormat]
    valid_letters: str
    valid_digits: str
    corrections: CorrectionRules
    stop_words: List[str] = field(default_factory=list)
    invalid_sequences: List[str] = field(default_factory=list)
    min_length: Optional[int] = None
    max_length: Optional[int] = None


@dataclass(frozen=True)
class PlateValidationResult:
    raw_text: str
    normalized_text: str
    corrected_text: str
    country_code: Optional[str]
    country_name: Optional[str]
    format_name: Optional[str]
    is_valid: bool
    reason: Optional[str]


class PlatePostProcessor:
    """Загружает профили стран и применяет постобработку/валидацию номеров."""

    def __init__(
        self,
        config_dir: str,
        enabled_countries: Optional[Sequence[str]] = None,
    ) -> None:
        self.config_dir = config_dir
        self.enabled_countries = {code.upper() for code in enabled_countries or []}
        self.profiles: List[CountryProfile] = self._load_profiles()

    @staticmethod
    def normalize_ocr_text(text: str) -> str:
        normalized = re.sub(r"[\s\-_]", "", text or "")
        normalized = normalized.replace("|", "I").upper()
        return normalized

    def process(self, text: str) -> PlateValidationResult:
        normalized = self.normalize_ocr_text(text)
        best_result: Optional[PlateValidationResult] = None

        if not self.profiles:
            return PlateValidationResult(
                raw_text=text,
                normalized_text=normalized,
                corrected_text=normalized,
                country_code=None,
                country_name=None,
                format_name=None,
                is_valid=True,
                reason=None,
            )

        for profile in self.profiles:
            candidate = self._apply_corrections(normalized, profile)
            result = self._validate_candidate(normalized, candidate, profile)
            if best_result is None or self._is_better_result(result, best_result):
                best_result = result
            if result.is_valid:
                break

        if best_result is None:
            return PlateValidationResult(
                raw_text=text,
                normalized_text=normalized,
                corrected_text=normalized,
                country_code=None,
                country_name=None,
                format_name=None,
                is_valid=False,
                reason="Нет доступных профилей",
            )
        return best_result

    def _is_better_result(
        self, candidate: PlateValidationResult, current: PlateValidationResult
    ) -> bool:
        if candidate.is_valid and not current.is_valid:
            return True
        if candidate.is_valid and current.is_valid:
            return False
        # Если оба невалидны, выбираем тот, у которого есть найденная страна
        if candidate.country_code and not current.country_code:
            return True
        return False

    def _apply_corrections(self, text: str, profile: CountryProfile) -> str:
        corrected = text
        if profile.corrections.digit_to_letter:
            corrected = corrected.translate(str.maketrans(profile.corrections.digit_to_letter))
        if profile.corrections.letter_to_digit:
            corrected = corrected.translate(str.maketrans(profile.corrections.letter_to_digit))
        for wrong, right in profile.corrections.common_mistakes:
            corrected = corrected.replace(wrong.upper(), right.upper())
        return corrected

    def _validate_candidate(
        self, normalized: str, candidate: str, profile: CountryProfile
    ) -> PlateValidationResult:
        reason = None
        format_name = None

        if not candidate:
            reason = "Пустая строка"
        elif profile.stop_words and candidate in profile.stop_words:
            reason = "Стоп-слово"
        elif self._is_repeating_sequence(candidate):
            reason = "Повторяющаяся последовательность"
        elif profile.invalid_sequences and any(seq in candidate for seq in profile.invalid_sequences):
            reason = "Невалидная последовательность"
        elif profile.min_length and len(candidate) < profile.min_length:
            reason = "Слишком короткий номер"
        elif profile.max_length and len(candidate) > profile.max_length:
            reason = "Слишком длинный номер"
        elif not self._is_valid_charset(candidate, profile):
            reason = "Недопустимые символы"
        else:
            format_name = self._match_format(candidate, profile)
            if format_name is None:
                reason = "Не соответствует формату"

        return PlateValidationResult(
            raw_text=normalized,
            normalized_text=candidate,
            corrected_text=candidate if reason is None else "",
            country_code=profile.code,
            country_name=profile.name,
            format_name=format_name,
            is_valid=reason is None,
            reason=reason,
        )

    @staticmethod
    def _is_repeating_sequence(value: str) -> bool:
        if len(value) < 4:
            return False
        if len(set(value)) == 1:
            return True
        digit_runs = {"0123", "1234", "2345", "3456", "4567", "5678", "6789"}
        return value.isdigit() and value in digit_runs

    @staticmethod
    def _is_valid_charset(candidate: str, profile: CountryProfile) -> bool:
        allowed = set(profile.valid_letters + profile.valid_digits)
        return all(ch in allowed for ch in candidate)

    @staticmethod
    def _match_format(candidate: str, profile: CountryProfile) -> Optional[str]:
        for fmt in profile.formats:
            if fmt.regex.match(candidate):
                return fmt.name
        return None

    def _load_profiles(self) -> List[CountryProfile]:
        profiles: List[CountryProfile] = []
        if not os.path.isdir(self.config_dir):
            return profiles

        for filename in sorted(os.listdir(self.config_dir)):
            if not filename.lower().endswith(('.yml', '.yaml')):
                continue
            path = os.path.join(self.config_dir, filename)
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            code = str(data.get("code", "")).upper()
            if self.enabled_countries and code not in self.enabled_countries:
                continue
            profiles.append(self._profile_from_dict(data))

        profiles.sort(key=lambda p: p.priority)
        return profiles

    def _profile_from_dict(self, data: Dict[str, object]) -> CountryProfile:
        formats = [
            CountryFormat(name=item.get("name", "default"), regex=re.compile(item.get("regex", "^$")))
            for item in data.get("license_plate_formats", [])
            if isinstance(item, dict)
        ]
        corrections_block = data.get("corrections", {}) or {}
        corrections = CorrectionRules(
            digit_to_letter={k: v for k, v in (corrections_block.get("digit_to_letter") or {}).items()},
            letter_to_digit={k: v for k, v in (corrections_block.get("letter_to_digit") or {}).items()},
            common_mistakes=[
                (item.get("from", ""), item.get("to", ""))
                for item in corrections_block.get("common_mistakes", [])
                if isinstance(item, dict) and item.get("from") and item.get("to")
            ],
        )
        return CountryProfile(
            name=data.get("name", "Unknown"),
            code=str(data.get("code", "XX")).upper(),
            priority=int(data.get("priority", 100)),
            formats=formats,
            valid_letters=str(data.get("valid_characters", {}).get("letters", "")).upper(),
            valid_digits=str(data.get("valid_characters", {}).get("digits", "")),
            corrections=corrections,
            stop_words=[w.upper() for w in data.get("stop_words", [])],
            invalid_sequences=[s.upper() for s in data.get("invalid_sequences", [])],
            min_length=int(data["min_length"]) if data.get("min_length") is not None else None,
            max_length=int(data["max_length"]) if data.get("max_length") is not None else None,
        )

    @classmethod
    def available_countries(cls, config_dir: str) -> List[CountryProfile]:
        processor = cls(config_dir=config_dir, enabled_countries=None)
        return processor.profiles

