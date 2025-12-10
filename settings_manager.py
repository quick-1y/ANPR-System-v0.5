#!/usr/bin/env python3
# /settings_manager.py
import json
import os
from typing import Any, Dict, List


class SettingsManager:
    """Управляет конфигурацией приложения и каналами."""

    def __init__(self, path: str = "settings.json") -> None:
        self.path = path
        self.settings = self._load()

    def _default(self) -> Dict[str, Any]:
        return {
            "grid": "2x2",
            "channels": [
                {
                    "id": 1,
                    "name": "Канал 1",
                    "source": "0",
                    "best_shots": 3,
                    "cooldown_seconds": 5,
                    "ocr_min_confidence": 0.6,
                    "region": {"x": 0, "y": 0, "width": 100, "height": 100},
                    "detection_mode": "continuous",
                    "detector_frame_stride": 2,
                    "motion_threshold": 0.01,
                    "motion_frame_stride": 1,
                    "motion_activation_frames": 3,
                    "motion_release_frames": 6,
                },
            ],
            "reconnect": {
                "signal_loss": {
                    "enabled": True,
                    "frame_timeout_seconds": 5,
                    "retry_interval_seconds": 5,
                },
                "periodic": {"enabled": False, "interval_minutes": 60},
            },
            "storage": {
                "db_dir": "data/db",
                "database_file": "anpr.db",
                "screenshots_dir": "data/screenshots",
            },
            "tracking": {
                "best_shots": 3,
                "cooldown_seconds": 5,
                "ocr_min_confidence": 0.6,
            },
            "logging": {
                "level": "INFO",
                "file": "data/app.log",
                "max_bytes": 1048576,
                "backup_count": 5,
            },
        }

    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            defaults = self._default()
            self._save(defaults)
            return defaults
        with open(self.path, "r", encoding="utf-8") as f:
            return self._upgrade(json.load(f))

    def _upgrade(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Обновляет существующие настройки, добавляя недостающие поля."""

        changed = False
        tracking_defaults = data.get("tracking", {})
        reconnect_defaults = self._reconnect_defaults()
        storage_defaults = self._storage_defaults()
        for channel in data.get("channels", []):
            if self._fill_channel_defaults(channel, tracking_defaults):
                changed = True

        if self._fill_reconnect_defaults(data, reconnect_defaults):
            changed = True

        if self._fill_storage_defaults(data, storage_defaults):
            changed = True

        if changed:
            self._save(data)
        return data

    @staticmethod
    def _channel_defaults(tracking_defaults: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "best_shots": int(tracking_defaults.get("best_shots", 3)),
            "cooldown_seconds": int(tracking_defaults.get("cooldown_seconds", 5)),
            "ocr_min_confidence": float(tracking_defaults.get("ocr_min_confidence", 0.6)),
            "region": {"x": 0, "y": 0, "width": 100, "height": 100},
            "detection_mode": "continuous",
            "detector_frame_stride": 2,
            "motion_threshold": 0.01,
            "motion_frame_stride": 1,
            "motion_activation_frames": 3,
            "motion_release_frames": 6,
        }

    @staticmethod
    def _reconnect_defaults() -> Dict[str, Any]:
        return {
            "signal_loss": {
                "enabled": True,
                "frame_timeout_seconds": 5,
                "retry_interval_seconds": 5,
            },
            "periodic": {"enabled": False, "interval_minutes": 60},
        }

    @staticmethod
    def _storage_defaults() -> Dict[str, Any]:
        return {
            "db_dir": "data/db",
            "database_file": "anpr.db",
            "screenshots_dir": "data/screenshots",
        }

    def _fill_channel_defaults(self, channel: Dict[str, Any], tracking_defaults: Dict[str, Any]) -> bool:
        defaults = self._channel_defaults(tracking_defaults)
        changed = False
        for key, value in defaults.items():
            if key not in channel:
                # Сохраняем только отсутствующие ключи, не перезаписывая пользовательские значения.
                channel[key] = value
                changed = True
        return changed

    def _fill_reconnect_defaults(self, data: Dict[str, Any], defaults: Dict[str, Any]) -> bool:
        if "reconnect" not in data:
            data["reconnect"] = defaults
            return True

        changed = False
        reconnect_section = data.get("reconnect", {})
        for key, default_value in defaults.items():
            if key not in reconnect_section:
                reconnect_section[key] = default_value
                changed = True
            elif isinstance(default_value, dict):
                for sub_key, sub_val in default_value.items():
                    if sub_key not in reconnect_section[key]:
                        reconnect_section[key][sub_key] = sub_val
                        changed = True
        data["reconnect"] = reconnect_section
        return changed

    def _fill_storage_defaults(self, data: Dict[str, Any], defaults: Dict[str, Any]) -> bool:
        if "storage" not in data:
            data["storage"] = defaults
            return True

        changed = False
        storage = data.get("storage", {})

        # Миграция со старого ключа events_db -> db_dir + database_file
        if "events_db" in storage:
            legacy_path = storage.get("events_db", "data/events.db")
            storage.setdefault("db_dir", os.path.dirname(legacy_path) or ".")
            storage.setdefault("database_file", os.path.basename(legacy_path) or "anpr.db")
            changed = True

        for key, val in defaults.items():
            if key not in storage:
                storage[key] = val
                changed = True
        data["storage"] = storage
        return changed

    def _save(self, data: Dict[str, Any]) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_channels(self) -> List[Dict[str, Any]]:
        channels = self.settings.get("channels", [])
        tracking_defaults = self.settings.get("tracking", {})
        changed = False
        for channel in channels:
            if self._fill_channel_defaults(channel, tracking_defaults):
                changed = True

        if changed:
            self.save_channels(channels)
        return channels

    def save_channels(self, channels: List[Dict[str, Any]]) -> None:
        self.settings["channels"] = channels
        self._save(self.settings)

    def get_grid(self) -> str:
        return self.settings.get("grid", "2x2")

    def save_grid(self, grid: str) -> None:
        self.settings["grid"] = grid
        self._save(self.settings)

    def get_reconnect(self) -> Dict[str, Any]:
        if self._fill_reconnect_defaults(self.settings, self._reconnect_defaults()):
            self._save(self.settings)
        return self.settings.get("reconnect", {})

    def save_reconnect(self, reconnect_conf: Dict[str, Any]) -> None:
        self.settings["reconnect"] = reconnect_conf
        self._save(self.settings)

    def get_db_dir(self) -> str:
        storage = self.settings.get("storage", {})
        return storage.get("db_dir", "data/db")

    def get_database_file(self) -> str:
        storage = self.settings.get("storage", {})
        return storage.get("database_file", "anpr.db")

    def get_db_path(self) -> str:
        directory = self.get_db_dir()
        filename = self.get_database_file()
        return os.path.join(directory, filename)

    def save_db_dir(self, path: str) -> None:
        storage = self.settings.get("storage", {})
        storage["db_dir"] = path
        self.settings["storage"] = storage
        self._save(self.settings)

    def save_screenshot_dir(self, path: str) -> None:
        storage = self.settings.get("storage", {})
        storage["screenshots_dir"] = path
        self.settings["storage"] = storage
        self._save(self.settings)

    def get_screenshot_dir(self) -> str:
        storage = self.settings.get("storage", {})
        return storage.get("screenshots_dir", "data/screenshots")

    def get_best_shots(self) -> int:
        tracking = self.settings.get("tracking", {})
        return int(tracking.get("best_shots", 3))

    def save_best_shots(self, best_shots: int) -> None:
        tracking = self.settings.get("tracking", {})
        tracking["best_shots"] = int(best_shots)
        self.settings["tracking"] = tracking
        self._save(self.settings)

    def get_cooldown_seconds(self) -> int:
        tracking = self.settings.get("tracking", {})
        return int(tracking.get("cooldown_seconds", 5))

    def save_cooldown_seconds(self, cooldown: int) -> None:
        tracking = self.settings.get("tracking", {})
        tracking["cooldown_seconds"] = int(cooldown)
        self.settings["tracking"] = tracking
        self._save(self.settings)

    def get_min_confidence(self) -> float:
        tracking = self.settings.get("tracking", {})
        return float(tracking.get("ocr_min_confidence", 0.6))

    def save_min_confidence(self, min_conf: float) -> None:
        tracking = self.settings.get("tracking", {})
        tracking["ocr_min_confidence"] = float(min_conf)
        self.settings["tracking"] = tracking
        self._save(self.settings)

    def get_logging_config(self) -> Dict[str, Any]:
        return self.settings.get("logging", {})

    def refresh(self) -> None:
        self.settings = self._load()

    def update_channel(self, channel_id: int, data: Dict[str, Any]) -> None:
        channels = self.get_channels()
        for idx, channel in enumerate(channels):
            if channel.get("id") == channel_id:
                channels[idx].update(data)
                break
        else:
            channels.append(data)
        self.save_channels(channels)
