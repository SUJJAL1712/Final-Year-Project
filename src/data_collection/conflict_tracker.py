"""
Regional conflict event tracking and database.

Maintains a structured database of major regional conflicts and geopolitical
events that impact global markets. Focuses on events relevant to the
US-India-China triangle and broader global market impact.
"""

import pandas as pd
from loguru import logger

from src.utils.config import DATA_DIR


class ConflictEventTracker:
    """Tracks and manages conflict-related geopolitical events."""

    def __init__(self):
        self.events_dir = DATA_DIR / "events"
        self.events_dir.mkdir(parents=True, exist_ok=True)

    def get_curated_conflict_events(self) -> pd.DataFrame:
        """
        Return a curated database of major conflict events (2015-2025).
        """
        events = [
            # === South China Sea Tensions ===
            {"date": "2016-07-12", "event": "International tribunal rules against China on South China Sea",
             "category": "territorial_dispute", "severity": 6,
             "primary_countries": ["China", "Philippines"],
             "affected_markets": ["China", "US"],
             "affected_sectors": ["shipping", "energy", "defense"]},

            # === India-Pakistan Tensions ===
            {"date": "2016-09-29", "event": "India surgical strikes across LoC in Pakistan",
             "category": "military_action", "severity": 7,
             "primary_countries": ["India", "Pakistan"],
             "affected_markets": ["India"],
             "affected_sectors": ["defense", "financials"]},
            {"date": "2019-02-26", "event": "India Balakot air strikes in Pakistan",
             "category": "conflict_escalation", "severity": 8,
             "primary_countries": ["India", "Pakistan"],
             "affected_markets": ["India"],
             "affected_sectors": ["defense", "airlines", "financials"]},
            {"date": "2019-02-27", "event": "Pakistan retaliates, Indian pilot captured",
             "category": "conflict_escalation", "severity": 8,
             "primary_countries": ["India", "Pakistan"],
             "affected_markets": ["India"],
             "affected_sectors": ["defense", "airlines", "financials"]},

            # === India-China Border Conflict ===
            {"date": "2020-05-05", "event": "India-China troops clash at Pangong Tso, Ladakh",
             "category": "border_skirmish", "severity": 5,
             "primary_countries": ["India", "China"],
             "affected_markets": ["India", "China"],
             "affected_sectors": ["defense", "technology"]},
            {"date": "2020-06-15", "event": "Galwan Valley deadly clash, 20 Indian soldiers killed",
             "category": "border_skirmish", "severity": 9,
             "primary_countries": ["India", "China"],
             "affected_markets": ["India", "China"],
             "affected_sectors": ["defense", "technology", "consumer_goods"]},
            {"date": "2020-06-29", "event": "India bans 59 Chinese apps including TikTok",
             "category": "sanctions_imposed", "severity": 7,
             "primary_countries": ["India", "China"],
             "affected_markets": ["India", "China"],
             "affected_sectors": ["technology", "internet"]},
            {"date": "2021-02-11", "event": "India-China agree to disengage at Pangong Tso",
             "category": "conflict_deescalation", "severity": 6,
             "primary_countries": ["India", "China"],
             "affected_markets": ["India", "China"],
             "affected_sectors": ["defense"]},

            # === US-Iran Tensions ===
            {"date": "2020-01-03", "event": "US kills Iranian General Soleimani",
             "category": "conflict_escalation", "severity": 9,
             "primary_countries": ["US", "Iran"],
             "affected_markets": ["US", "India", "China"],
             "affected_sectors": ["energy", "defense", "airlines"]},
            {"date": "2020-01-08", "event": "Iran retaliates with missile strikes on US bases in Iraq",
             "category": "conflict_escalation", "severity": 8,
             "primary_countries": ["US", "Iran"],
             "affected_markets": ["US", "India", "China"],
             "affected_sectors": ["energy", "defense"]},

            # === Russia-Ukraine War ===
            {"date": "2022-02-24", "event": "Russia invades Ukraine",
             "category": "conflict_outbreak", "severity": 10,
             "primary_countries": ["Russia", "Ukraine"],
             "affected_markets": ["US", "India", "China"],
             "affected_sectors": ["energy", "agriculture", "defense", "financials", "shipping"]},
            {"date": "2022-03-08", "event": "US bans Russian oil imports",
             "category": "sanctions_imposed", "severity": 8,
             "primary_countries": ["US", "Russia"],
             "affected_markets": ["US", "India", "China"],
             "affected_sectors": ["energy"]},
            {"date": "2022-06-27", "event": "G7 agrees to Russian oil price cap",
             "category": "sanctions_imposed", "severity": 7,
             "primary_countries": ["US", "Russia"],
             "affected_markets": ["US", "India", "China"],
             "affected_sectors": ["energy"]},
            {"date": "2022-09-21", "event": "Russia announces partial mobilization",
             "category": "conflict_escalation", "severity": 8,
             "primary_countries": ["Russia", "Ukraine"],
             "affected_markets": ["US", "India", "China"],
             "affected_sectors": ["energy", "defense", "agriculture"]},

            # === Taiwan Strait Crisis ===
            {"date": "2022-08-02", "event": "Pelosi visits Taiwan, China launches military drills",
             "category": "conflict_escalation", "severity": 9,
             "primary_countries": ["US", "China"],
             "affected_markets": ["US", "China", "India"],
             "affected_sectors": ["semiconductors", "technology", "defense", "shipping"]},

            # === Israel-Hamas War ===
            {"date": "2023-10-07", "event": "Hamas attacks Israel",
             "category": "conflict_outbreak", "severity": 9,
             "primary_countries": ["Israel", "Palestine"],
             "affected_markets": ["US", "India"],
             "affected_sectors": ["energy", "defense", "airlines"]},
            {"date": "2023-10-27", "event": "Israel begins ground offensive in Gaza",
             "category": "conflict_escalation", "severity": 8,
             "primary_countries": ["Israel", "Palestine"],
             "affected_markets": ["US", "India"],
             "affected_sectors": ["energy", "defense"]},

            # === Red Sea / Houthi Crisis ===
            {"date": "2023-11-19", "event": "Houthis seize Galaxy Leader cargo ship in Red Sea",
             "category": "shipping_disruption", "severity": 7,
             "primary_countries": ["Yemen"],
             "affected_markets": ["US", "India", "China"],
             "affected_sectors": ["shipping", "energy", "consumer_goods"]},
            {"date": "2024-01-11", "event": "US-UK launch strikes against Houthi targets in Yemen",
             "category": "conflict_escalation", "severity": 8,
             "primary_countries": ["US", "Yemen"],
             "affected_markets": ["US", "India", "China"],
             "affected_sectors": ["shipping", "energy", "defense"]},
            {"date": "2024-02-19", "event": "Houthi attacks force major shipping reroutes around Africa",
             "category": "shipping_disruption", "severity": 8,
             "primary_countries": ["Yemen"],
             "affected_markets": ["US", "India", "China"],
             "affected_sectors": ["shipping", "energy", "consumer_goods", "manufacturing"]},

            # === India-Pakistan 2025 ===
            {"date": "2025-04-22", "event": "Pahalgam terror attack in Indian Kashmir",
             "category": "conflict_escalation", "severity": 7,
             "primary_countries": ["India", "Pakistan"],
             "affected_markets": ["India"],
             "affected_sectors": ["defense", "financials", "tourism"]},
        ]

        df = pd.DataFrame(events)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        df.to_csv(self.events_dir / "conflict_events.csv", index=False)
        logger.info(f"Loaded {len(df)} curated conflict events")
        return df

    def get_combined_geopolitical_events(self) -> pd.DataFrame:
        """
        Combine tariff and conflict events into a unified timeline.
        This is the master event database for the entire analysis.
        """
        from src.data_collection.tariff_tracker import TariffEventTracker

        tariff_events = TariffEventTracker().get_curated_tariff_events()
        conflict_events = self.get_curated_conflict_events()

        # Standardize columns
        tariff_events["event_type"] = "tariff"
        conflict_events["event_type"] = "conflict"

        # Align columns
        tariff_std = tariff_events[["date", "event", "category", "severity", "event_type"]].copy()
        tariff_std["markets_affected"] = tariff_events["markets_affected"]

        conflict_std = conflict_events[["date", "event", "category", "severity", "event_type"]].copy()
        conflict_std["markets_affected"] = conflict_events["affected_markets"]

        combined = pd.concat([tariff_std, conflict_std], ignore_index=True)
        combined = combined.sort_values("date").reset_index(drop=True)

        combined.to_csv(self.events_dir / "all_geopolitical_events.csv", index=False)
        logger.info(f"Combined event database: {len(combined)} events")
        return combined

    def get_events_around_date(
        self, target_date: str, window_days: int = 7
    ) -> pd.DataFrame:
        """Get all events within a window around a specific date."""
        combined = self.get_combined_geopolitical_events()
        target = pd.Timestamp(target_date)
        start = target - pd.Timedelta(days=window_days)
        end = target + pd.Timedelta(days=window_days)
        mask = (combined["date"] >= start) & (combined["date"] <= end)
        return combined[mask]

    def get_market_specific_timeline(self, market_id: str) -> pd.DataFrame:
        """Get chronological event timeline for a specific market."""
        combined = self.get_combined_geopolitical_events()
        mask = combined["markets_affected"].apply(
            lambda x: market_id in x if isinstance(x, list) else market_id in str(x)
        )
        return combined[mask].reset_index(drop=True)
