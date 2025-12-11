"""Постобработка и валидация результатов OCR для номерных знаков."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

from anpr.validation.base import (
    CountrySpec,
    PlateValidationResult,
    ValidatorConfig,
    sort_specs,
)


class CountryRegistry:
    """Реестр поддерживаемых стран и правил коррекции."""

    def __init__(self, specs: Iterable[CountrySpec]):
        self._specs = sort_specs(specs)
        self._by_code = {spec.code.upper(): spec for spec in self._specs}

    @classmethod
    def load_from_dir(cls, path: Path, enabled: Optional[List[str]] = None) -> "CountryRegistry":
        enabled_codes = {code.upper() for code in enabled or []}
        specs: List[CountrySpec] = []
        for file in sorted(path.glob("*.yaml")):
            try:
                with file.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                spec = CountrySpec.from_dict(data)
                if enabled_codes and spec.code.upper() not in enabled_codes:
                    continue
                specs.append(spec)
            except Exception:  # noqa: BLE001
                continue
        return cls(specs)

    def get(self, code: str) -> Optional[CountrySpec]:
        return self._by_code.get(code.upper())

    def all(self) -> List[CountrySpec]:
        return self._specs

    def to_metadata(self) -> List[Dict[str, str]]:
        return [
            {"name": spec.name, "code": spec.code, "priority": str(spec.priority)}
            for spec in self._specs
        ]


class PlatePostProcessor:
    """Обрабатывает номер: нормализует, корректирует, валидирует."""

    def __init__(self, validator_config: ValidatorConfig) -> None:
        self.config = validator_config
        self.registry = CountryRegistry.load_from_dir(
            self.config.config_dir, enabled=self.config.enabled_countries
        )

    def _normalize_common(self, raw: str, corrections: CountrySpec | None = None) -> str:
        text = (raw or "").strip().upper()
        text = text.replace(" ", "").replace("-", "")
        if corrections is None:
            for spec in self.registry.all():
                text = self._apply_corrections(text, spec)
            return text
        return self._apply_corrections(text, corrections)

    @staticmethod
    def _apply_corrections(text: str, spec: CountrySpec) -> str:
        corrected = text
        if spec.corrections.digit_to_letter:
            for digit, letter in spec.corrections.digit_to_letter.items():
                corrected = corrected.replace(digit, letter)
        for mistake in spec.corrections.common_mistakes:
            src = mistake.get("from")
            dst = mistake.get("to")
            if src and dst:
                corrected = corrected.replace(src, dst)
        return corrected

    def normalize_candidate(self, raw: str) -> str:
        """Лёгкая нормализация для голосования по треку."""
        return self._normalize_common(raw)

    def validate(self, raw: str) -> PlateValidationResult:
        normalized = self._normalize_common(raw)
        for spec in self.registry.all():
            candidate = self._normalize_common(normalized, corrections=spec)
            if spec.stop_words and candidate in spec.stop_words:
                return PlateValidationResult(
                    is_valid=False,
                    normalized=candidate,
                    country_code=spec.code,
                    country_name=spec.name,
                    reason="Стоп-слово",
                )
            if any(seq and seq in candidate for seq in spec.invalid_sequences):
                return PlateValidationResult(
                    is_valid=False,
                    normalized=candidate,
                    country_code=spec.code,
                    country_name=spec.name,
                    reason="Недопустимая последовательность",
                )
            if not spec.allows(candidate):
                continue
            fmt = spec.match_format(candidate)
            if fmt:
                return PlateValidationResult(
                    is_valid=True,
                    normalized=candidate,
                    country_code=spec.code,
                    country_name=spec.name,
                    format_name=fmt.name,
                )
        return PlateValidationResult(
            is_valid=False,
            normalized=normalized,
            reason="Не соответствует формату",
        )

    def dump_metadata(self, path: Path) -> None:
        """Сохраняет актуальный каталог стран (для UI)."""
        meta = self.registry.to_metadata()
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
