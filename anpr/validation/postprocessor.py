from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from anpr.validation.country_rules import CountryRule, load_country_rules


@dataclass
class PlateValidationResult:
    plate: str
    country_code: Optional[str]
    country_name: Optional[str]
    is_valid: bool


class PlatePostProcessor:
    """Валидация и коррекция распознанных номеров на основе конфигов стран."""

    def __init__(self, config_dir: str, enabled_countries: Optional[Iterable[str]] = None) -> None:
        self.config_dir = config_dir
        self.enabled_countries = [code.upper() for code in enabled_countries] if enabled_countries else None
        self._rules: List[CountryRule] = load_country_rules(config_dir, self.enabled_countries)

    def reload(self) -> None:
        self._rules = load_country_rules(self.config_dir, self.enabled_countries)

    @property
    def available_countries(self) -> List[str]:
        return [rule.code for rule in self._rules]

    def process(self, raw_text: str) -> PlateValidationResult:
        normalized = raw_text.upper().replace(" ", "").replace("-", "").replace("_", "")

        for rule in self._rules:
            candidate = rule.apply_corrections(rule.normalize(normalized))
            if not candidate:
                continue
            if rule.is_stop_word(candidate) or rule.has_invalid_sequence(candidate):
                continue
            if not rule.matches(candidate):
                continue
            return PlateValidationResult(
                plate=candidate,
                country_code=rule.code,
                country_name=rule.name,
                is_valid=True,
            )

        return PlateValidationResult(plate=normalized, country_code=None, country_name=None, is_valid=False)


__all__ = ["PlatePostProcessor", "PlateValidationResult"]
