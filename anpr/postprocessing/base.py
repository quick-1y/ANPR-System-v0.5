"""Базовые структуры для валидации госномеров."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Pattern
import re


@dataclass(frozen=True)
class CorrectionRule:
    src: str
    dst: str

    @classmethod
    def from_dict(cls, raw: Dict[str, str]) -> "CorrectionRule":
        return cls(src=str(raw.get("from", "")), dst=str(raw.get("to", "")))


@dataclass(frozen=True)
class PlateFormat:
    name: str
    regex: Pattern[str]
    description: str = ""

    @classmethod
    def from_dict(cls, raw: Dict[str, str]) -> "PlateFormat":
        pattern = raw.get("regex") or ""
        return cls(
            name=str(raw.get("name", "default")),
            regex=re.compile(pattern, re.IGNORECASE),
            description=str(raw.get("description", "")),
        )


@dataclass(frozen=True)
class CountryProfile:
    name: str
    code: str
    priority: int
    formats: List[PlateFormat]
    letters: str
    digits: str
    uses_cyrillic: bool
    corrections: List[CorrectionRule]
    strip_chars: str
    stop_words: List[str]
    max_repeated: int

    @classmethod
    def from_config(cls, path: Path, data: Dict[str, object]) -> "CountryProfile":
        formats = [PlateFormat.from_dict(item) for item in data.get("license_plate_formats", [])]
        corrections = [CorrectionRule.from_dict(item) for item in data.get("corrections", {}).get("common_mistakes", [])]
        letters = str(data.get("valid_characters", {}).get("letters", ""))
        digits = str(data.get("valid_characters", {}).get("digits", "0123456789"))
        uses_cyrillic = bool(data.get("uses_cyrillic", False))
        strip_chars = str(data.get("strip_characters", "- _"))
        stop_words = [str(w).upper() for w in data.get("stop_words", [])]
        max_repeated = int(data.get("max_repeated", 4))

        return cls(
            name=str(data.get("name", path.stem.title())),
            code=str(data.get("code", path.stem.upper())),
            priority=int(data.get("priority", 100)),
            formats=formats,
            letters=letters,
            digits=digits,
            uses_cyrillic=uses_cyrillic,
            corrections=corrections,
            strip_chars=strip_chars,
            stop_words=stop_words,
            max_repeated=max(2, max_repeated),
        )
