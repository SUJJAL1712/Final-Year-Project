"""
Tariff event tracking and database.

Maintains a structured database of major tariff events affecting
the US-India-China triangle. Combines manual curation of known events
with LLM-assisted extraction from news.
"""

import pandas as pd
from loguru import logger

from src.utils.config import DATA_DIR


class TariffEventTracker:
    """Tracks and manages tariff-related events."""

    def __init__(self):
        self.events_dir = DATA_DIR / "events"
        self.events_dir.mkdir(parents=True, exist_ok=True)

    def get_curated_tariff_events(self) -> pd.DataFrame:
        """
        Return a curated database of major tariff events (2015-2025).

        This is the ground truth dataset built from verified public records.
        Each event is tagged with affected markets, sectors, and severity.
        """
        events = [
            # === US-China Trade War: Opening Salvos (2018) ===
            {"date": "2018-01-22", "event": "Trump imposes tariffs on solar panels and washing machines",
             "category": "tariff_imposition", "severity": 5, "source_country": "US", "target_country": "China",
             "affected_sectors": ["solar", "consumer_goods"], "tariff_rate": "30%",
             "markets_affected": ["US", "China"]},
            {"date": "2018-03-01", "event": "Trump announces 25% steel and 10% aluminum tariffs globally",
             "category": "tariff_imposition", "severity": 8, "source_country": "US", "target_country": "Global",
             "affected_sectors": ["steel", "aluminum", "industrials"], "tariff_rate": "25%/10%",
             "markets_affected": ["US", "China", "India"]},
            {"date": "2018-03-22", "event": "Trump signs memo targeting $50B of Chinese imports",
             "category": "tariff_imposition", "severity": 8, "source_country": "US", "target_country": "China",
             "affected_sectors": ["technology", "industrials"], "tariff_rate": "25%",
             "markets_affected": ["US", "China"]},
            {"date": "2018-04-02", "event": "China retaliates with tariffs on $3B of US goods",
             "category": "retaliatory_tariff", "severity": 7, "source_country": "China", "target_country": "US",
             "affected_sectors": ["agriculture", "steel", "aluminum"], "tariff_rate": "15-25%",
             "markets_affected": ["US", "China"]},
            {"date": "2018-07-06", "event": "US-China tariffs take effect on $34B goods each",
             "category": "retaliatory_tariff", "severity": 9, "source_country": "US", "target_country": "China",
             "affected_sectors": ["technology", "agriculture", "automotive"], "tariff_rate": "25%",
             "markets_affected": ["US", "China", "India"]},
            {"date": "2018-09-24", "event": "US imposes 10% tariffs on $200B of Chinese goods",
             "category": "tariff_imposition", "severity": 9, "source_country": "US", "target_country": "China",
             "affected_sectors": ["consumer_goods", "technology", "manufacturing"], "tariff_rate": "10%",
             "markets_affected": ["US", "China", "India"]},

            # === Escalation (2019) ===
            {"date": "2019-05-10", "event": "US raises tariffs to 25% on $200B of Chinese goods",
             "category": "tariff_increase", "severity": 9, "source_country": "US", "target_country": "China",
             "affected_sectors": ["consumer_goods", "technology"], "tariff_rate": "25%",
             "markets_affected": ["US", "China", "India"]},
            {"date": "2019-05-13", "event": "China retaliates with tariffs on $60B of US goods",
             "category": "retaliatory_tariff", "severity": 8, "source_country": "China", "target_country": "US",
             "affected_sectors": ["agriculture", "energy", "chemicals"], "tariff_rate": "5-25%",
             "markets_affected": ["US", "China"]},
            {"date": "2019-08-01", "event": "Trump announces 10% tariff on remaining $300B Chinese goods",
             "category": "tariff_imposition", "severity": 8, "source_country": "US", "target_country": "China",
             "affected_sectors": ["consumer_electronics", "apparel", "toys"], "tariff_rate": "10%",
             "markets_affected": ["US", "China"]},
            {"date": "2019-08-23", "event": "China announces new tariffs on $75B of US goods",
             "category": "retaliatory_tariff", "severity": 8, "source_country": "China", "target_country": "US",
             "affected_sectors": ["agriculture", "automotive", "energy"], "tariff_rate": "5-10%",
             "markets_affected": ["US", "China"]},

            # === Phase One Deal (2019-2020) ===
            {"date": "2019-12-13", "event": "US-China Phase One deal announced",
             "category": "trade_deal_signed", "severity": 8, "source_country": "US", "target_country": "China",
             "affected_sectors": ["agriculture", "technology", "financial"], "tariff_rate": "reduction",
             "markets_affected": ["US", "China", "India"]},
            {"date": "2020-01-15", "event": "Phase One trade deal formally signed",
             "category": "trade_deal_signed", "severity": 9, "source_country": "US", "target_country": "China",
             "affected_sectors": ["agriculture", "energy", "manufacturing"], "tariff_rate": "partial rollback",
             "markets_affected": ["US", "China", "India"]},

            # === India-specific tariff events ===
            {"date": "2018-06-21", "event": "India raises tariffs on 28 US products in retaliation",
             "category": "retaliatory_tariff", "severity": 5, "source_country": "India", "target_country": "US",
             "affected_sectors": ["agriculture", "steel"], "tariff_rate": "20-120%",
             "markets_affected": ["US", "India"]},
            {"date": "2019-06-05", "event": "US ends India preferential trade status (GSP)",
             "category": "tariff_imposition", "severity": 6, "source_country": "US", "target_country": "India",
             "affected_sectors": ["gems", "agriculture", "chemicals"], "tariff_rate": "GSP removal",
             "markets_affected": ["US", "India"]},
            {"date": "2019-06-16", "event": "India retaliatory tariffs on 28 US goods take effect",
             "category": "retaliatory_tariff", "severity": 5, "source_country": "India", "target_country": "US",
             "affected_sectors": ["agriculture", "chemicals"], "tariff_rate": "various",
             "markets_affected": ["US", "India"]},

            # === Tech War / Sanctions (2020-2023) ===
            {"date": "2020-05-15", "event": "US blocks Huawei from global chip supply",
             "category": "export_ban", "severity": 9, "source_country": "US", "target_country": "China",
             "affected_sectors": ["semiconductors", "telecom"], "tariff_rate": "ban",
             "markets_affected": ["US", "China"]},
            {"date": "2022-08-09", "event": "US CHIPS and Science Act signed",
             "category": "tech_investment_ban", "severity": 7, "source_country": "US", "target_country": "China",
             "affected_sectors": ["semiconductors"], "tariff_rate": "subsidy+restriction",
             "markets_affected": ["US", "China", "India"]},
            {"date": "2022-10-07", "event": "US imposes sweeping chip export controls on China",
             "category": "chip_export_restriction", "severity": 10, "source_country": "US", "target_country": "China",
             "affected_sectors": ["semiconductors", "AI", "technology"], "tariff_rate": "ban",
             "markets_affected": ["US", "China"]},
            {"date": "2023-10-17", "event": "US tightens China chip export controls further",
             "category": "chip_export_restriction", "severity": 8, "source_country": "US", "target_country": "China",
             "affected_sectors": ["semiconductors", "AI"], "tariff_rate": "expanded ban",
             "markets_affected": ["US", "China"]},

            # === Trump 2.0 Era (2025) ===
            {"date": "2025-02-01", "event": "Trump imposes 25% tariff on Canada/Mexico, 10% on China",
             "category": "tariff_imposition", "severity": 9, "source_country": "US", "target_country": "China",
             "affected_sectors": ["consumer_goods", "automotive", "agriculture"], "tariff_rate": "10-25%",
             "markets_affected": ["US", "China", "India"]},
            {"date": "2025-02-04", "event": "China retaliates with tariffs on US goods",
             "category": "retaliatory_tariff", "severity": 8, "source_country": "China", "target_country": "US",
             "affected_sectors": ["agriculture", "energy"], "tariff_rate": "10-15%",
             "markets_affected": ["US", "China"]},
            {"date": "2025-03-04", "event": "US raises China tariffs to 20%, adds tariffs on Canada/Mexico",
             "category": "tariff_increase", "severity": 9, "source_country": "US", "target_country": "China",
             "affected_sectors": ["broad_economy"], "tariff_rate": "20%",
             "markets_affected": ["US", "China", "India"]},
        ]

        df = pd.DataFrame(events)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Save to events directory
        df.to_csv(self.events_dir / "tariff_events.csv", index=False)
        logger.info(f"Loaded {len(df)} curated tariff events")
        return df

    def get_events_for_market(self, market_id: str) -> pd.DataFrame:
        """Get tariff events that affect a specific market."""
        df = self.get_curated_tariff_events()
        mask = df["markets_affected"].apply(lambda x: market_id in x)
        return df[mask].reset_index(drop=True)

    def get_events_in_range(self, start: str, end: str) -> pd.DataFrame:
        """Get tariff events within a date range."""
        df = self.get_curated_tariff_events()
        mask = (df["date"] >= start) & (df["date"] <= end)
        return df[mask].reset_index(drop=True)

    def get_bilateral_events(
        self, country_a: str, country_b: str
    ) -> pd.DataFrame:
        """Get events between two specific countries."""
        df = self.get_curated_tariff_events()
        mask = (
            ((df["source_country"] == country_a) & (df["target_country"] == country_b))
            | ((df["source_country"] == country_b) & (df["target_country"] == country_a))
        )
        return df[mask].reset_index(drop=True)
