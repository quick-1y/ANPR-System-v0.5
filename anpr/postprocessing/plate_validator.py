from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import yaml


@dataclass(frozen=True)
class PlateFormat:
    name: str
    regex: re.Pattern[str]


@dataclass(frozen=True)
class CountryRules:
    code: str
    name: str
    priority: int
    formats: List[PlateFormat]
    valid_letters: set[str]
    valid_digits: set[str]
    digit_to_letter: Dict[str, str]
    common_mistakes: List[tuple[str, str]]
    stop_words: List[str]
    invalid_sequences: List[str]


@dataclass(frozen=True)
class ValidationResult:
    plate: str
    country: str
    format_name: str
    corrected: bool


class PlateValidator:
    """Загружает правила стран и применяет базовую валидацию/коррекцию."""

    def __init__(self, config_dir: str, allowed_countries: Optional[Sequence[str]] = None) -> None:
        self.config_dir = config_dir
        self.allowed_countries = {c.upper() for c in (allowed_countries or [])}
        self._rules = self._load_rules()

    @staticmethod
    def discover_codes(config_dir: str) -> Dict[str, str]:
        codes: Dict[str, str] = {}
        for filename in os.listdir(config_dir):
            if not filename.endswith(".yaml"):
                continue
            path = os.path.join(config_dir, filename)
            with open(path, "r", encoding="utf-8") as fp:
                data = yaml.safe_load(fp) or {}
                code = str(data.get("code", "")).upper()
                name = str(data.get("name", code or filename.replace(".yaml", "")))
                if code:
                    codes[code] = name
        return codes

    def _load_rules(self) -> List[CountryRules]:
        rules: List[CountryRules] = []
        for filename in os.listdir(self.config_dir):
            if not filename.endswith(".yaml"):
                continue
            path = os.path.join(self.config_dir, filename)
            with open(path, "r", encoding="utf-8") as fp:
                cfg = yaml.safe_load(fp) or {}
            code = str(cfg.get("code", "")).upper()
            if not code or (self.allowed_countries and code not in self.allowed_countries):
                continue
            formats_cfg = cfg.get("license_plate_formats", []) or []
            formats = [
                PlateFormat(item.get("name", "default"), re.compile(item.get("regex", "")))
                for item in formats_cfg
                if item.get("regex")
            ]
            valid_characters = cfg.get("valid_characters", {}) or {}
            digit_to_letter = (cfg.get("corrections", {}) or {}).get("digit_to_letter", {})
            common_mistakes_cfg = (cfg.get("corrections", {}) or {}).get("common_mistakes", [])
            rules.append(
                CountryRules(
                    code=code,
                    name=cfg.get("name", code),
                    priority=int(cfg.get("priority", 100)),
                    formats=formats,
                    valid_letters=set(str(valid_characters.get("letters", ""))),
                    valid_digits=set(str(valid_characters.get("digits", "0123456789"))),
                    digit_to_letter={k: str(v) for k, v in digit_to_letter.items()},
                    common_mistakes=[(str(item.get("from", "")), str(item.get("to", ""))) for item in common_mistakes_cfg],
                    stop_words=[str(word).upper() for word in cfg.get("stop_words", [])],
                    invalid_sequences=[str(seq).upper() for seq in cfg.get("invalid_sequences", [])],
                )
            )
        return sorted(rules, key=lambda r: r.priority)

    @staticmethod
    def _strip_separators(plate: str) -> str:
        return re.sub(r"[\s\-]", "", plate)

    def _apply_corrections(self, raw: str, rules: CountryRules) -> tuple[str, bool]:
        normalized = self._strip_separators(raw.upper())
        corrected = False
        # Базовая нормализация похожих символов (кириллица -> латиница),
        # чтобы не отбрасывать корректные номера из-за различий алфавитов OCR.
        base_translation = str.maketrans(
            {
                "А": "A",
                "В": "B",
                "Е": "E",
                "К": "K",
                "М": "M",
                "Н": "H",
                "О": "O",
                "Р": "P",
                "С": "C",
                "Т": "T",
                "У": "Y",
                "Х": "X",
            }
        )
        translated = normalized.translate(base_translation)
        corrected = corrected or translated != normalized
        normalized = translated
        if rules.digit_to_letter:
            translation = str.maketrans({k: v for k, v in rules.digit_to_letter.items()})
            translated = normalized.translate(translation)
            corrected = corrected or translated != normalized
            normalized = translated
        for src, dst in rules.common_mistakes:
            if not src:
                continue
            new_value = normalized.replace(src.upper(), dst.upper())
            corrected = corrected or new_value != normalized
            normalized = new_value
        normalized = re.sub(r"[^A-ZА-Я0-9]", "", normalized)
        return normalized, corrected

    @staticmethod
    def _has_only_allowed_chars(candidate: str, rules: CountryRules) -> bool:
        if not candidate:
            return False
        return all((c in rules.valid_letters) or (c in rules.valid_digits) for c in candidate)

    @staticmethod
    def _looks_like_sequence(candidate: str, rules: CountryRules) -> bool:
        upper = candidate.upper()
        if upper in rules.stop_words:
            return True
        if upper in rules.invalid_sequences:
            return True
        if len(set(upper)) == 1 and len(upper) >= 3:
            return True
        if upper.isdigit() and len(upper) >= 3:
            step = ord(upper[1]) - ord(upper[0])
            if step in (1, -1) and all((ord(upper[i + 1]) - ord(upper[i]) == step) for i in range(len(upper) - 1)):
                return True
        return False

    def validate(self, plate: str) -> Optional[ValidationResult]:
        for rules in self._rules:
            candidate, corrected = self._apply_corrections(plate, rules)
            if not candidate or self._looks_like_sequence(candidate, rules):
                continue
            if rules.valid_letters and rules.valid_digits and not self._has_only_allowed_chars(candidate, rules):
                continue
            for fmt in rules.formats:
                if fmt.regex.fullmatch(candidate):
                    return ValidationResult(plate=candidate, country=rules.code, format_name=fmt.name, corrected=corrected)
        return None


class PlatePostProcessor:
    """Оборачивает валидацию для использования в пайплайне."""

    def __init__(self, validator: PlateValidator) -> None:
        self.validator = validator

    def process(self, plate: str) -> Optional[ValidationResult]:
        if not plate:
            return None
        return self.validator.validate(plate)
