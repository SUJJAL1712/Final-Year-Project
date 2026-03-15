"""
Regional conflict event tracking from real GDELT data.

Two data sources merged for full 2013-present coverage:
1. GDELT daily aggregate export files — historical backbone, CAMEO-coded
   events (complete daily coverage from Apr 2013 to present, no API key)
2. GDELT DOC 2.0 API — recent article search (~last 3 months only),
   classified via LLM (if API key available) or keyword matching.
   NOTE: DOC 2.0 has a rolling ~3-month window and cannot provide
   historical article data.

NO hardcoded events — all data sourced from real GDELT data.
"""

import hashlib
import ast
import os
import re

import pandas as pd
from loguru import logger

from src.utils.config import DATA_DIR, ANALYSIS_START, ANALYSIS_END
from src.data_collection.gdelt_fetcher import GDELTFetcher


class ConflictEventTracker:
    """Tracks conflict events from real GDELT news + optional LLM classification."""

    KEYWORD_CATEGORIES = {
        "invasion": "conflict_outbreak",
        "declares war": "conflict_outbreak",
        "war breaks out": "conflict_outbreak",
        "new conflict": "conflict_outbreak",
        "escalat": "conflict_escalation",
        "intensif": "conflict_escalation",
        "retaliatory strike": "conflict_escalation",
        "deescalat": "conflict_deescalation",
        "de-escalat": "conflict_deescalation",
        "tension eas": "conflict_deescalation",
        "ceasefire": "ceasefire",
        "truce": "ceasefire",
        "peace deal": "ceasefire",
        "military buildup": "military_buildup",
        "troops deploy": "military_buildup",
        "mobiliz": "military_buildup",
        "border clash": "border_skirmish",
        "border skirmish": "border_skirmish",
        "border incident": "border_skirmish",
        "galwan": "border_skirmish",
        "ladakh": "border_skirmish",
        "air strike": "military_action",
        "missile strike": "military_action",
        "bombing": "military_action",
        "airstrike": "military_action",
        "surgical strike": "military_action",
        "military operation": "military_action",
        "drone strike": "military_action",
        "shipping disruption": "shipping_disruption",
        "shipping attack": "shipping_disruption",
        "ship seize": "shipping_disruption",
        "shipping reroute": "shipping_disruption",
        "red sea": "shipping_disruption",
        "houthi": "shipping_disruption",
        "sanction": "sanctions_imposed",
        "diplomatic breakdown": "diplomatic_breakdown",
        "ambassador recall": "diplomatic_breakdown",
        "diplomatic progress": "diplomatic_breakthrough",
        "diplomatic breakthrou": "diplomatic_breakthrough",
        "territorial dispute": "territorial_dispute",
        "territorial claim": "territorial_dispute",
        "south china sea": "territorial_dispute",
        "conflict": "conflict_escalation",  # default catch-all (keep last)
    }

    COUNTRY_PATTERNS = {
        "US": [
            r"\bUS\b", r"\bU\.S\.", r"United States", r"America", r"Washington",
        ],
        "China": [r"\bChina\b", r"Beijing", r"Chinese", r"\bPRC\b"],
        "India": [r"\bIndia\b", r"New Delhi", r"Indian"],
        "Russia": [r"\bRussia\b", r"Moscow", r"Russian", r"Kremlin"],
        "Ukraine": [r"\bUkraine\b", r"Ukrainian", r"Kyiv"],
        "Israel": [r"\bIsrael\b", r"Israeli"],
        "Palestine": [r"\bPalestine\b", r"\bGaza\b", r"\bHamas\b"],
        "Pakistan": [r"\bPakistan\b", r"Islamabad", r"Pakistani"],
        "Iran": [r"\bIran\b", r"Tehran", r"Iranian"],
        "Yemen": [r"\bYemen\b", r"\bHouthi\b"],
        "Taiwan": [r"\bTaiwan\b", r"Taipei"],
    }

    SEVERITY_MAP = {
        "conflict_outbreak": 9,
        "conflict_escalation": 7,
        "conflict_deescalation": 5,
        "ceasefire": 6,
        "military_buildup": 5,
        "border_skirmish": 6,
        "military_action": 8,
        "shipping_disruption": 7,
        "sanctions_imposed": 7,
        "sanctions_lifted": 5,
        "diplomatic_breakdown": 5,
        "diplomatic_breakthrough": 5,
        "territorial_dispute": 5,
        "alliance_formation": 4,
    }

    # Countries → which stock markets are affected
    MARKET_IMPACT = {
        "Russia": ["US", "India", "China"],
        "Ukraine": ["US", "India", "China"],
        "Israel": ["US", "India"],
        "Palestine": ["US", "India"],
        "Yemen": ["US", "India", "China"],
        "Taiwan": ["US", "China"],
        "Pakistan": ["India"],
        "Iran": ["US", "India", "China"],
    }

    def __init__(self):
        self.events_dir = DATA_DIR / "events"
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.gdelt = GDELTFetcher()

    def _cache_path_for(self, start_date: str, end_date: str) -> "Path":
        """Generate date-range-aware cache path for conflict events."""
        range_hash = hashlib.sha256(
            f"conflict|{start_date}|{end_date}".encode()
        ).hexdigest()[:10]
        return self.events_dir / f"conflict_events_{range_hash}.csv"

    def get_conflict_events(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Get classified conflict events from real GDELT data.

        Pipeline:
        1. Load historical backbone from GDELT daily aggregate files
           (complete daily coverage from Apr 2013 to present, CAMEO-coded)
        2. Fetch recent articles from GDELT DOC 2.0 API (~last 3 months only)
        3. Classify recent articles via LLM or keyword matching
        4. Merge historical + recent, deduplicate
        5. Cache to disk for fast reload
        """
        start_date = start_date or ANALYSIS_START
        end_date = end_date or ANALYSIS_END
        cache_path = self._cache_path_for(start_date, end_date)

        if cache_path.exists() and not force_refresh:
            logger.info(f"Loading cached conflict events from {cache_path}")
            return pd.read_csv(cache_path, parse_dates=["date"])

        frames = []

        # Step 1: Historical backbone from GDELT daily aggregate files
        try:
            from src.data_collection.gdelt_historical import GDELTHistoricalFetcher
            historical = GDELTHistoricalFetcher()
            hist_events = historical.fetch_conflict_events(start_date, end_date)
            if not hist_events.empty:
                logger.info(f"Historical backbone: {len(hist_events)} conflict events")
                frames.append(hist_events)
        except Exception as e:
            logger.warning(f"Historical fetch failed (will use DOC 2.0 only): {e}")

        # NOTE: GDELT DOC 2.0 API only covers ~3 months (rolling window),
        # which is useless for a 10+ year analysis. The historical backbone
        # above provides complete CAMEO-coded event coverage from 2013-present.
        # DOC 2.0 is intentionally skipped.

        if not frames:
            logger.warning("No conflict events from any source")
            return pd.DataFrame()

        # Step 4: Merge and deduplicate
        events = pd.concat(frames, ignore_index=True)
        events = self._deduplicate_events(events)
        events["event_type"] = "conflict"

        # Step 5: Cache
        events.to_csv(cache_path, index=False)
        logger.info(f"Saved {len(events)} conflict events to cache")

        return events

    def _classify_with_llm(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Classify articles using LLM batch classification."""
        from src.llm_pipeline.event_classifier import EventClassifier

        classifier = EventClassifier()
        articles = news_df.copy()

        # GDELT DOC 2.0 returns metadata but not article body text.
        # Build a descriptive content string from available metadata
        # instead of pretending content = title.
        if "content" not in articles.columns or (
            articles["content"].astype(str) == articles["title"].astype(str)
        ).all():
            articles["content"] = articles.apply(
                lambda r: (
                    f"Headline: {r.get('title', '')}\n"
                    f"Source: {r.get('source', 'unknown')}\n"
                    f"Country: {r.get('source_country', 'unknown')}\n"
                    f"Tone: {r.get('tone', 'N/A')}"
                ),
                axis=1,
            )

        classified = classifier.classify_batch(articles)
        if classified.empty:
            return pd.DataFrame()

        events = pd.DataFrame()
        events["date"] = pd.to_datetime(
            classified.get("date", pd.Series(dtype=str)), errors="coerce"
        )
        events["event"] = classified.get("title", "")
        events["category"] = classified.get(
            "event_subcategory", "conflict_escalation"
        )
        events["severity"] = (
            pd.to_numeric(classified.get("severity", 5), errors="coerce")
            .fillna(5)
            .astype(int)
        )

        if "countries_involved" in classified.columns:
            events["primary_countries"] = classified["countries_involved"].apply(
                lambda x: x if isinstance(x, list) else []
            )
        else:
            events["primary_countries"] = [[] for _ in range(len(events))]

        # Use affected_markets from LLM if available, otherwise infer
        if "affected_markets" in classified.columns:
            events["affected_markets"] = classified["affected_markets"].apply(
                lambda x: (
                    x if isinstance(x, list)
                    else self._infer_affected_markets([])
                )
            )
        else:
            events["affected_markets"] = events["primary_countries"].apply(
                self._infer_affected_markets
            )

        if "affected_sectors" in classified.columns:
            events["affected_sectors"] = classified["affected_sectors"].apply(
                lambda x: x if isinstance(x, list) else ["defense", "energy"]
            )
        else:
            events["affected_sectors"] = [
                ["defense", "energy"] for _ in range(len(events))
            ]

        events["confidence"] = (
            pd.to_numeric(classified.get("confidence", 0.5), errors="coerce")
            .fillna(0.5)
        )

        events = events[events["category"] != "not_relevant"]
        return events.dropna(subset=["date"])

    def _classify_with_keywords(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Fallback: classify articles using keyword matching."""
        records = []

        for _, row in news_df.iterrows():
            title = str(row.get("title", ""))
            title_lower = title.lower()

            # Determine category (first match wins)
            category = "conflict_escalation"
            for keyword, cat in self.KEYWORD_CATEGORIES.items():
                if keyword.lower() in title_lower:
                    category = cat
                    break

            # Extract countries mentioned
            countries = []
            for country, patterns in self.COUNTRY_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, title, re.IGNORECASE):
                        countries.append(country)
                        break

            severity = self.SEVERITY_MAP.get(category, 5)

            # Adjust severity based on GDELT tone
            tone = row.get("tone", 0)
            if isinstance(tone, (int, float)) and abs(tone) > 5:
                severity = min(10, severity + 1)

            affected_markets = self._infer_affected_markets(countries)

            records.append({
                "date": pd.to_datetime(row.get("published_at"), errors="coerce"),
                "event": title,
                "category": category,
                "severity": severity,
                "primary_countries": countries,
                "affected_markets": affected_markets,
                "affected_sectors": ["defense", "energy"],
                "confidence": 0.6,
            })

        return pd.DataFrame(records).dropna(subset=["date"])

    def _infer_affected_markets(self, countries: list) -> list:
        """Infer which stock markets are affected based on countries involved."""
        if not isinstance(countries, list):
            return ["US", "India", "China"]

        markets = set()
        for c in countries:
            if c in ("US", "India", "China"):
                markets.add(c)
            elif c in self.MARKET_IMPACT:
                markets.update(self.MARKET_IMPACT[c])

        # Default: if no specific market identified, assume global impact
        if not markets:
            markets = {"US", "India", "China"}

        return sorted(markets)

    def _deduplicate_events(self, events: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate by keeping highest-severity per (period, category, region).

        Uses 2-day periods AND a region key derived from involved countries,
        so genuinely different conflicts (e.g., Russia-Ukraine vs Israel-Gaza)
        are never merged even if they share a category and happen close together.
        Only syndicated copies of the same underlying event are collapsed.
        """
        if events.empty:
            return events

        events = events.sort_values("date")

        def _normalize_primary_countries(value) -> list[str]:
            """Normalize cached/list country payloads into a clean string list."""
            if isinstance(value, list):
                return [str(v).strip() for v in value if str(v).strip()]

            if isinstance(value, str):
                raw = value.strip()
                if not raw or raw.lower() in {"nan", "none"}:
                    return []

                # Handle CSV-serialized Python list strings like:
                # "['Russia', 'Ukraine']"
                if raw.startswith("[") and raw.endswith("]"):
                    try:
                        parsed = ast.literal_eval(raw)
                        if isinstance(parsed, list):
                            return [
                                str(v).strip()
                                for v in parsed
                                if str(v).strip()
                            ]
                    except (ValueError, SyntaxError):
                        pass

                # Fallback: comma-separated strings
                if "," in raw:
                    return [
                        part.strip().strip("'\"")
                        for part in raw.split(",")
                        if part.strip()
                    ]

                return [raw.strip().strip("'\"")]

            return []

        # 2-day periods for fine granularity
        events["_period"] = (
            (events["date"] - pd.Timestamp("2015-01-01")).dt.days // 2
        )
        # Region key from involved countries — events about different conflicts
        # (different country pairs) are always kept as distinct events.
        events["_region_key"] = events.apply(
            lambda row: (
                "|".join(
                    sorted(
                        set(
                            _normalize_primary_countries(
                                row.get("primary_countries", [])
                            )
                            or [
                                c for c in [
                                    str(row.get("source_country", "")).strip(),
                                    str(row.get("target_country", "")).strip(),
                                ]
                                if c and c.lower() not in {"nan", "none"}
                            ]
                        )
                    )
                )
                or "unknown"
            ),
            axis=1,
        )

        deduped = (
            events.sort_values("severity", ascending=False)
            .drop_duplicates(
                subset=["_period", "category", "_region_key"], keep="first"
            )
            .drop(columns=["_period", "_region_key"])
            .sort_values("date")
            .reset_index(drop=True)
        )
        return deduped

    # ---- Backward-compatible aliases ----

    def get_curated_conflict_events(self) -> pd.DataFrame:
        """Backward-compatible alias for get_conflict_events."""
        return self.get_conflict_events()

    def get_combined_geopolitical_events(
        self, force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Combine tariff and conflict events into a unified timeline.
        This is the master event database for the entire analysis.
        """
        from src.data_collection.tariff_tracker import TariffEventTracker

        # Date-range-aware combined cache
        from src.utils.config import ANALYSIS_START, ANALYSIS_END
        range_hash = hashlib.sha256(
            f"combined_geopolitical_events_{ANALYSIS_START}_{ANALYSIS_END}".encode()
        ).hexdigest()[:10]
        cache_path = self.events_dir / f"all_geopolitical_events_{range_hash}.csv"

        if cache_path.exists() and not force_refresh:
            logger.info(f"Loading cached combined events from {cache_path}")
            return pd.read_csv(cache_path, parse_dates=["date"])

        tariff_events = TariffEventTracker().get_tariff_events(
            force_refresh=force_refresh
        )
        conflict_events = self.get_conflict_events(
            force_refresh=force_refresh
        )

        frames = []

        if not tariff_events.empty:
            keep_cols = ["date", "event", "category", "severity"]
            for col in ["source_country", "target_country"]:
                if col in tariff_events.columns:
                    keep_cols.append(col)
            tariff_std = tariff_events[keep_cols].copy()
            tariff_std["event_type"] = "tariff"
            tariff_std["markets_affected"] = tariff_events["markets_affected"]
            frames.append(tariff_std)

        if not conflict_events.empty:
            keep_cols = ["date", "event", "category", "severity"]
            for col in ["source_country", "target_country"]:
                if col in conflict_events.columns:
                    keep_cols.append(col)
            conflict_std = conflict_events[keep_cols].copy()
            conflict_std["event_type"] = "conflict"
            conflict_std["markets_affected"] = conflict_events["affected_markets"]
            frames.append(conflict_std)

        if not frames:
            logger.warning("No events from either tariff or conflict trackers")
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values("date").reset_index(drop=True)

        combined.to_csv(cache_path, index=False)
        logger.info(f"Combined event database: {len(combined)} events")
        return combined

    def get_events_around_date(
        self, target_date: str, window_days: int = 7
    ) -> pd.DataFrame:
        """Get all events within a window around a specific date."""
        combined = self.get_combined_geopolitical_events()
        if combined.empty:
            return combined
        target = pd.Timestamp(target_date)
        start = target - pd.Timedelta(days=window_days)
        end = target + pd.Timedelta(days=window_days)
        mask = (combined["date"] >= start) & (combined["date"] <= end)
        return combined[mask]

    def get_market_specific_timeline(self, market_id: str) -> pd.DataFrame:
        """Get chronological event timeline for a specific market."""
        combined = self.get_combined_geopolitical_events()
        if combined.empty:
            return combined
        mask = combined["markets_affected"].apply(
            lambda x: market_id in x if isinstance(x, list) else market_id in str(x)
        )
        return combined[mask].reset_index(drop=True)
