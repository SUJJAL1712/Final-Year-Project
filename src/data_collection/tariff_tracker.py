"""
Tariff event tracking from real GDELT news data.

Fetches actual historical tariff news from the GDELT Project API (free, no key)
and classifies them into structured events. When an Anthropic API key is
available, uses Claude for high-quality classification; otherwise falls back
to keyword-based classification.

NO hardcoded events — all data sourced from real news articles.
"""

import os
import re

import pandas as pd
from loguru import logger

from src.utils.config import DATA_DIR
from src.data_collection.gdelt_fetcher import GDELTFetcher


class TariffEventTracker:
    """Tracks tariff events from real GDELT news + optional LLM classification."""

    # Keyword → category mapping for fallback classification (order matters)
    KEYWORD_CATEGORIES = {
        "retaliatory": "retaliatory_tariff",
        "retaliat": "retaliatory_tariff",
        "counter-tariff": "retaliatory_tariff",
        "trade war escalat": "trade_war_escalation",
        "trade war": "trade_war_escalation",
        "trade deal": "trade_deal_signed",
        "trade agreement": "trade_deal_signed",
        "phase one": "trade_deal_signed",
        "phase 1": "trade_deal_signed",
        "tariff increase": "tariff_increase",
        "raise tariff": "tariff_increase",
        "raises tariff": "tariff_increase",
        "hike tariff": "tariff_increase",
        "tariff cut": "tariff_reduction",
        "tariff reduc": "tariff_reduction",
        "tariff remov": "tariff_removal",
        "lift tariff": "tariff_removal",
        "tariff threat": "tariff_threat",
        "threaten tariff": "tariff_threat",
        "tariff negotiat": "tariff_negotiation",
        "trade talk": "tariff_negotiation",
        "trade negotiat": "tariff_negotiation",
        "export ban": "export_ban",
        "export control": "chip_export_restriction",
        "chip ban": "chip_export_restriction",
        "semiconductor ban": "chip_export_restriction",
        "chips act": "chip_export_restriction",
        "sanction": "sanctions_imposed",
        "entity list": "entity_list_addition",
        "blacklist": "entity_list_addition",
        "import duty": "tariff_imposition",
        "tariff": "tariff_imposition",  # default catch-all (keep last)
    }

    COUNTRY_PATTERNS = {
        "US": [
            r"\bUS\b", r"\bU\.S\.", r"United States", r"America",
            r"Washington", r"Trump", r"Biden",
        ],
        "China": [
            r"\bChina\b", r"Beijing", r"Chinese", r"Xi Jinping", r"\bPRC\b",
        ],
        "India": [
            r"\bIndia\b", r"New Delhi", r"Modi", r"Indian",
        ],
    }

    SEVERITY_MAP = {
        "tariff_imposition": 7,
        "tariff_increase": 7,
        "tariff_reduction": 5,
        "tariff_removal": 6,
        "retaliatory_tariff": 8,
        "tariff_threat": 4,
        "tariff_negotiation": 3,
        "trade_war_escalation": 8,
        "trade_war_deescalation": 5,
        "trade_deal_signed": 7,
        "trade_deal_collapsed": 8,
        "export_ban": 8,
        "chip_export_restriction": 9,
        "sanctions_imposed": 8,
        "sanctions_lifted": 6,
        "entity_list_addition": 7,
    }

    def __init__(self):
        self.events_dir = DATA_DIR / "events"
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.events_dir / "tariff_events.csv"
        self.gdelt = GDELTFetcher()

    def get_tariff_events(
        self,
        start_date: str = "2015-01-01",
        end_date: str = "2025-12-31",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Get classified tariff events from real GDELT news data.

        Pipeline:
        1. Fetch tariff news from GDELT API (free, no key required)
        2. Classify via LLM (if API key available) or keyword matching
        3. Deduplicate into distinct events (weekly grouping)
        4. Cache to disk for fast reload
        """
        if self.cache_path.exists() and not force_refresh:
            logger.info(f"Loading cached tariff events from {self.cache_path}")
            df = pd.read_csv(self.cache_path, parse_dates=["date"])
            return df

        # Step 1: Fetch from GDELT
        logger.info("Fetching tariff news from GDELT...")
        raw_news = self.gdelt.fetch_tariff_news(start_date, end_date)

        if raw_news.empty:
            logger.warning("No tariff news from GDELT")
            return pd.DataFrame()

        logger.info(f"GDELT returned {len(raw_news)} tariff articles")

        # Step 2: Classify
        if os.getenv("ANTHROPIC_API_KEY"):
            logger.info("Using LLM classification (Claude API)...")
            events = self._classify_with_llm(raw_news)
        else:
            logger.info("No ANTHROPIC_API_KEY — using keyword classification...")
            events = self._classify_with_keywords(raw_news)

        if events.empty:
            logger.warning("No events after classification")
            return pd.DataFrame()

        # Step 3: Deduplicate
        events = self._deduplicate_events(events)
        events["event_type"] = "tariff"

        # Step 4: Cache
        events.to_csv(self.cache_path, index=False)
        logger.info(f"Saved {len(events)} tariff events to cache")

        return events

    def _classify_with_llm(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Classify articles using LLM batch classification."""
        from src.llm_pipeline.event_classifier import EventClassifier

        classifier = EventClassifier()
        articles = news_df.copy()

        # Ensure required columns for classify_batch
        if "content" not in articles.columns:
            articles["content"] = articles["title"]

        classified = classifier.classify_batch(articles)

        if classified.empty:
            return pd.DataFrame()

        events = pd.DataFrame()
        events["date"] = pd.to_datetime(
            classified.get("date", pd.Series(dtype=str)), errors="coerce"
        )
        events["event"] = classified.get("title", "")
        events["category"] = classified.get(
            "event_subcategory", "tariff_imposition"
        )
        events["severity"] = (
            pd.to_numeric(classified.get("severity", 5), errors="coerce")
            .fillna(5)
            .astype(int)
        )
        events["source_country"] = classified.get("source_country", "")
        events["target_country"] = classified.get("target_country", "")

        # Handle list-type columns safely
        if "affected_sectors" in classified.columns:
            events["affected_sectors"] = classified["affected_sectors"].apply(
                lambda x: x if isinstance(x, list) else []
            )
        else:
            events["affected_sectors"] = [[] for _ in range(len(events))]

        if "countries_involved" in classified.columns:
            events["markets_affected"] = classified["countries_involved"].apply(
                lambda x: x if isinstance(x, list) else ["US", "China"]
            )
        else:
            events["markets_affected"] = [
                ["US", "China"] for _ in range(len(events))
            ]

        events["confidence"] = (
            pd.to_numeric(classified.get("confidence", 0.5), errors="coerce")
            .fillna(0.5)
        )

        # Filter out non-relevant classifications
        events = events[events["category"] != "not_relevant"]
        events = events.dropna(subset=["date"])
        return events

    def _classify_with_keywords(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Fallback: classify articles using keyword matching."""
        records = []

        for _, row in news_df.iterrows():
            title = str(row.get("title", ""))
            title_lower = title.lower()

            # Determine category by keyword matching (first match wins)
            category = "tariff_imposition"
            for keyword, cat in self.KEYWORD_CATEGORIES.items():
                if keyword.lower() in title_lower:
                    category = cat
                    break

            # Extract countries mentioned
            countries = []
            source = ""
            target = ""
            for country, patterns in self.COUNTRY_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, title, re.IGNORECASE):
                        countries.append(country)
                        break

            if len(countries) >= 2:
                source, target = countries[0], countries[1]
            elif len(countries) == 1:
                source = countries[0]

            severity = self.SEVERITY_MAP.get(category, 5)

            # Adjust severity based on GDELT tone
            tone = row.get("tone", 0)
            if isinstance(tone, (int, float)) and abs(tone) > 5:
                severity = min(10, severity + 1)

            records.append({
                "date": pd.to_datetime(row.get("published_at"), errors="coerce"),
                "event": title,
                "category": category,
                "severity": severity,
                "source_country": source,
                "target_country": target,
                "affected_sectors": [],
                "markets_affected": countries if countries else ["US", "China"],
                "confidence": 0.6,
            })

        return pd.DataFrame(records).dropna(subset=["date"])

    def _deduplicate_events(self, events: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate by keeping highest-severity event per category per week."""
        if events.empty:
            return events

        events = events.sort_values("date")
        events["_week"] = events["date"].dt.isocalendar().week.astype(int)
        events["_year"] = events["date"].dt.year

        deduped = (
            events.sort_values("severity", ascending=False)
            .drop_duplicates(subset=["_year", "_week", "category"], keep="first")
            .drop(columns=["_week", "_year"])
            .sort_values("date")
            .reset_index(drop=True)
        )

        return deduped

    # ---- Backward-compatible aliases ----

    def get_curated_tariff_events(self) -> pd.DataFrame:
        """Backward-compatible alias for get_tariff_events."""
        return self.get_tariff_events()

    def get_events_for_market(self, market_id: str) -> pd.DataFrame:
        """Get tariff events that affect a specific market."""
        df = self.get_tariff_events()
        if df.empty:
            return df
        mask = df["markets_affected"].apply(
            lambda x: market_id in x if isinstance(x, list) else market_id in str(x)
        )
        return df[mask].reset_index(drop=True)

    def get_events_in_range(self, start: str, end: str) -> pd.DataFrame:
        """Get tariff events within a date range."""
        df = self.get_tariff_events()
        if df.empty:
            return df
        mask = (df["date"] >= start) & (df["date"] <= end)
        return df[mask].reset_index(drop=True)

    def get_bilateral_events(self, country_a: str, country_b: str) -> pd.DataFrame:
        """Get events between two specific countries."""
        df = self.get_tariff_events()
        if df.empty:
            return df
        mask = (
            ((df["source_country"] == country_a) & (df["target_country"] == country_b))
            | ((df["source_country"] == country_b) & (df["target_country"] == country_a))
        )
        return df[mask].reset_index(drop=True)
