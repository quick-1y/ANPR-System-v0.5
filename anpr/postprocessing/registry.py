"""Загрузка конфигураций стран для валидации госномеров."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import yaml

from .base import CountryProfile

DEFAULT_CONFIG_DIR = Path(__file__).with_suffix("").parent / "configs"


def load_country_profiles(config_dir: Path | str = DEFAULT_CONFIG_DIR) -> List[CountryProfile]:
    base = Path(config_dir)
    profiles: List[CountryProfile] = []
    for yaml_path in sorted(base.glob("*.yaml")):
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            profiles.append(CountryProfile.from_config(yaml_path, data))
        except Exception as exc:  # noqa: BLE001
            # Ошибки загрузки одной страны не должны останавливать приложение
            print(f"Не удалось загрузить профиль страны {yaml_path.name}: {exc}")
    profiles.sort(key=lambda p: p.priority)
    return profiles


def list_country_profiles(config_dir: Path | str = DEFAULT_CONFIG_DIR) -> List[Dict[str, str]]:
    return [
        {"code": profile.code, "name": profile.name}
        for profile in load_country_profiles(config_dir)
    ]


def filter_profiles(profiles: Iterable[CountryProfile], allowed: List[str] | None) -> List[CountryProfile]:
    if not allowed:
        return list(profiles)
    allowed_set = {code.upper() for code in allowed}
    return [profile for profile in profiles if profile.code.upper() in allowed_set]
