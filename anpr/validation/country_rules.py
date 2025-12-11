from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml


@dataclass
class LicensePlateFormat:
    name: str
    regex: str
    _compiled: re.Pattern | None = field(init=False, default=None)

    def compiled(self) -> re.Pattern:
        if self._compiled is None:
            self._compiled = re.compile(self.regex)
        return self._compiled

    def matches(self, plate: str) -> bool:
        return bool(self.compiled().match(plate))


@dataclass
class CorrectionRules:
    digit_to_letter: Dict[str, str]
    letter_to_digit: Dict[str, str]
    common_mistakes: List[Dict[str, str]]

    @classmethod
    def from_dict(cls, data: Dict) -> "CorrectionRules":
        return cls(
            digit_to_letter={str(k): str(v) for k, v in (data.get("digit_to_letter") or {}).items()},
            letter_to_digit={str(k): str(v) for k, v in (data.get("letter_to_digit") or {}).items()},
            common_mistakes=[{"from": str(item.get("from", "")), "to": str(item.get("to", ""))} for item in data.get("common_mistakes", [])],
        )

    def apply(self, text: str) -> str:
        mapping = {**self.digit_to_letter, **self.letter_to_digit}
        corrected = text.translate(str.maketrans(mapping))
        for mistake in self.common_mistakes:
            src = mistake.get("from", "")
            dst = mistake.get("to", "")
            if src:
                corrected = corrected.replace(src, dst)
        return corrected


@dataclass
class CountryRule:
    name: str
    code: str
    priority: int
    formats: List[LicensePlateFormat]
    valid_letters: str
    valid_digits: str
    corrections: CorrectionRules
    stop_words: List[str]
    invalid_sequences: List[str]
    min_length: int
    max_length: int

    @classmethod
    def from_dict(cls, data: Dict) -> "CountryRule":
        formats = [LicensePlateFormat(f.get("name", ""), f.get("regex", "")) for f in data.get("license_plate_formats", [])]
        corrections = CorrectionRules.from_dict(data.get("corrections") or {})
        valid_chars = data.get("valid_characters") or {}
        stop_words = [str(word).upper() for word in data.get("stop_words", [])]
        invalid_sequences = [str(seq).upper() for seq in data.get("invalid_sequences", [])]
        min_len = int(data.get("min_length" or 0) or 0)
        max_len = int(data.get("max_length" or 16) or 16)
        return cls(
            name=str(data.get("name", "")),
            code=str(data.get("code", "")).upper(),
            priority=int(data.get("priority", 0)),
            formats=formats,
            valid_letters=str(valid_chars.get("letters", "")).upper(),
            valid_digits=str(valid_chars.get("digits", "")),
            corrections=corrections,
            stop_words=stop_words,
            invalid_sequences=invalid_sequences,
            min_length=min_len,
            max_length=max_len,
        )

    def normalize(self, text: str) -> str:
        normalized = text.upper()
        for ch in (" ", "-", "_", "\t", "\n"):
            normalized = normalized.replace(ch, "")
        return normalized

    def apply_corrections(self, text: str) -> str:
        corrected = self.corrections.apply(text)
        allowed_chars = set(self.valid_letters + self.valid_digits)
        filtered = "".join(ch for ch in corrected if ch in allowed_chars)
        return filtered

    def is_stop_word(self, text: str) -> bool:
        normalized = text.upper()
        return normalized in self.stop_words

    def has_invalid_sequence(self, text: str) -> bool:
        upper = text.upper()
        return any(seq in upper for seq in self.invalid_sequences)

    def matches(self, text: str) -> bool:
        if not text or len(text) < self.min_length or len(text) > self.max_length:
            return False
        return any(fmt.matches(text) for fmt in self.formats)


def load_country_rules(config_dir: str, enabled_codes: Optional[Iterable[str]] = None) -> List[CountryRule]:
    directory = Path(config_dir)
    if not directory.exists():
        return []

    enabled_set = {code.upper() for code in enabled_codes} if enabled_codes else None
    rules: List[CountryRule] = []

    for path in sorted(directory.glob("*.yaml")):
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        code = str(data.get("code", "")).upper()
        if enabled_set is not None and code not in enabled_set:
            continue
        rules.append(CountryRule.from_dict(data))

    rules.sort(key=lambda r: (-r.priority, r.name))
    return rules


__all__ = ["CountryRule", "LicensePlateFormat", "CorrectionRules", "load_country_rules"]
