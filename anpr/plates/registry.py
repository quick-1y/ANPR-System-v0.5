from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml


@dataclass(order=True)
class PlateFormat:
    priority: int
    name: str
    regex: str
    description: str = ""


@dataclass
class CorrectionRules:
    digit_to_letter: Dict[str, str] = field(default_factory=dict)
    latin_to_cyrillic: Dict[str, str] = field(default_factory=dict)
    common_mistakes: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class CountryRule:
    code: str
    name: str
    priority: int
    alphabet: str
    formats: List[PlateFormat]
    valid_letters: str
    valid_digits: str
    corrections: CorrectionRules
    stop_words: List[str]
    invalid_sequences: List[str]


class CountryRegistry:
    """Загружает набор правил из YAML и предоставляет доступ по коду страны."""

    def __init__(self, config_dir: str) -> None:
        self.config_dir = config_dir
        self._countries: Dict[str, CountryRule] = {}
        self._load()

    @property
    def countries(self) -> List[CountryRule]:
        return sorted(self._countries.values(), key=lambda c: c.priority)

    def get(self, code: str) -> Optional[CountryRule]:
        return self._countries.get(code.upper())

    def available_codes(self) -> List[str]:
        return [country.code for country in self.countries]

    def _load(self) -> None:
        for path in glob.glob(os.path.join(self.config_dir, "*.yaml")):
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                code = str(data.get("code", "")).upper()
                if not code:
                    continue
                self._countries[code] = self._parse_country(data)

    def _parse_country(self, data: Dict[str, object]) -> CountryRule:
        formats = [
            PlateFormat(
                priority=index,
                name=item.get("name", f"format_{index}"),
                regex=item.get("regex", ""),
                description=item.get("description", ""),
            )
            for index, item in enumerate(data.get("license_plate_formats", []) or [])
        ]

        corrections_data = data.get("corrections", {}) or {}
        corrections = CorrectionRules(
            digit_to_letter={k: str(v) for k, v in (corrections_data.get("digit_to_letter") or {}).items()},
            latin_to_cyrillic={
                k.upper(): str(v).upper() for k, v in (corrections_data.get("latin_to_cyrillic") or {}).items()
            },
            common_mistakes=list(corrections_data.get("common_mistakes") or []),
        )

        valid_characters = data.get("valid_characters", {}) or {}

        return CountryRule(
            code=str(data.get("code", "")).upper(),
            name=data.get("name", ""),
            priority=int(data.get("priority", 0)),
            alphabet=str(data.get("alphabet", "cyrillic")),
            formats=formats,
            valid_letters=str(valid_characters.get("letters", "")),
            valid_digits=str(valid_characters.get("digits", "0123456789")),
            corrections=corrections,
            stop_words=list(data.get("stop_words") or []),
            invalid_sequences=list(data.get("invalid_sequences") or []),
        )

