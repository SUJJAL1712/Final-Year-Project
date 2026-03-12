"""Centralized configuration loader."""

from datetime import datetime
from pathlib import Path
from functools import lru_cache
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"

# Dynamic analysis date range — always extends to today's date
ANALYSIS_START = "2015-01-01"
ANALYSIS_END = datetime.now().strftime("%Y-%m-%d")


class Config:
    """Loads and provides access to all YAML configuration files."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def __init__(self):
        if not self._loaded:
            self.settings = self._load_yaml("settings.yaml")
            self.markets = self._load_yaml("markets.yaml")
            self.taxonomy = self._load_yaml("events_taxonomy.yaml")
            self._loaded = True

    @staticmethod
    def _load_yaml(filename: str) -> dict:
        filepath = CONFIG_DIR / filename
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @property
    def time_range(self) -> dict:
        return self.settings["time_range"]

    @property
    def dc_config(self) -> dict:
        return self.settings["directional_changes"]

    @property
    def llm_config(self) -> dict:
        return self.settings["llm"]

    @property
    def analysis_config(self) -> dict:
        return self.settings["analysis"]

    @property
    def model_config(self) -> dict:
        return self.settings["model"]

    def get_market(self, market_id: str) -> dict:
        return self.markets["markets"][market_id]

    def get_all_symbols(self, market_id: str) -> list[str]:
        """Get all trackable symbols for a market."""
        market = self.get_market(market_id)
        symbols = []
        for idx in market.get("indices", []):
            symbols.append(idx["symbol"])
        for etf in market.get("sector_etfs", {}).values():
            symbols.append(etf)
        for stock in market.get("key_stocks", []):
            symbols.append(stock["symbol"])
        return symbols

    def get_event_categories(self) -> dict:
        return self.taxonomy["event_categories"]

    def get_reference_events(self) -> list[dict]:
        return self.taxonomy["reference_events"]


@lru_cache()
def get_config() -> Config:
    return Config()
