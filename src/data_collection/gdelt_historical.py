"""
GDELT historical event fetcher using daily aggregate export files.

Downloads COMPLETE daily event data from GDELT's public archive at:
    http://data.gdeltproject.org/events/YYYYMMDD.export.CSV.zip

Each daily file contains ALL events coded on that day — no sampling.
These files use the GDELT 1.0 export format (58 tab-separated columns)
with the same CAMEO coding system used by GDELT 2.0. The first 35 core
columns (actors, event codes, GoldsteinScale, tone, mention counts) are
identical between the two formats. These daily files have been published
every day since April 1, 2013 and continue to be updated alongside
GDELT 2.0.

Data coverage:
- Daily files available: 2013-04-01 to present (updated daily)
- This project uses: max(ANALYSIS_START, 2013-04-01) to ANALYSIS_END

NOT to be confused with:
- GDELT Events 2.0 15-minute export files (96 files/day; each event
  appears in only ONE file, so downloading a subset gives incomplete
  coverage)
- GDELT DOC 2.0 API (article search with ~3-month rolling window,
  NOT suitable for historical data)

No API key required — GDELT data is completely free and open.
"""

import io
import zipfile
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

import requests
import pandas as pd
from loguru import logger

from src.utils.config import DATA_DIR, ANALYSIS_START, ANALYSIS_END

# ------------------------------------------------------------------ #
# GDELT daily aggregate export file base URL
# Each file contains ALL events coded on that day in a single CSV.
# Format: GDELT 1.0 (58 columns, tab-separated, no header row).
# ------------------------------------------------------------------ #
GDELT_DAILY_BASE = "http://data.gdeltproject.org/events/"

# Earliest available daily file
GDELT_DAILY_START = datetime(2013, 4, 1)

# Column names for GDELT 1.0 daily export format (58 columns).
# The first 35 columns (through AvgTone) are identical to GDELT 2.0.
# Columns 36-55 are geo fields (1.0 has 20 geo cols; 2.0 has 23 — the
# three extra ADM2Code fields in 2.0 are the only structural difference).
# Columns 56-57 are DATEADDED and SOURCEURL.
GDELT_DAILY_COLUMNS = [
    "GlobalEventID", "Day", "MonthYear", "Year", "FractionDate",
    # Actor 1 (10 fields)
    "Actor1Code", "Actor1Name", "Actor1CountryCode",
    "Actor1KnownGroupCode", "Actor1EthnicCode",
    "Actor1Religion1Code", "Actor1Religion2Code",
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
    # Actor 2 (10 fields)
    "Actor2Code", "Actor2Name", "Actor2CountryCode",
    "Actor2KnownGroupCode", "Actor2EthnicCode",
    "Actor2Religion1Code", "Actor2Religion2Code",
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    # Event attributes (10 fields)
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode",
    "QuadClass", "GoldsteinScale", "NumMentions", "NumSources",
    "NumArticles", "AvgTone",
    # Actor 1 geo (7 fields — no ADM2Code in 1.0)
    "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code", "Actor1Geo_Lat", "Actor1Geo_Long",
    "Actor1Geo_FeatureID",
    # Actor 2 geo (7 fields)
    "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code", "Actor2Geo_Lat", "Actor2Geo_Long",
    "Actor2Geo_FeatureID",
    # Action geo (7 fields)
    "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code", "ActionGeo_Lat", "ActionGeo_Long",
    "ActionGeo_FeatureID",
    # Metadata
    "DATEADDED", "SOURCEURL",
]

assert len(GDELT_DAILY_COLUMNS) == 58, (
    f"Expected 58 columns for GDELT 1.0 format, got {len(GDELT_DAILY_COLUMNS)}"
)

# Columns we actually need (saves memory — skip geo and metadata)
USE_COLS = [
    "GlobalEventID", "Day",
    "Actor1Name", "Actor1CountryCode",
    "Actor2Name", "Actor2CountryCode",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode",
    "QuadClass", "GoldsteinScale", "NumMentions", "NumSources",
    "NumArticles", "AvgTone",
    "SOURCEURL",
]
USE_COL_INDICES = [GDELT_DAILY_COLUMNS.index(c) for c in USE_COLS]

# Target countries (GDELT uses FIPS 10-4 / ISO 3166 three-letter codes)
TARGET_COUNTRIES = {
    "USA", "CHN", "IND", "RUS", "UKR", "ISR", "PSE",
    "PAK", "IRN", "TWN", "YEM", "JPN", "KOR",
}

# Map GDELT 3-letter codes to our project's country names
COUNTRY_MAP = {
    "USA": "US", "CHN": "China", "IND": "India",
    "RUS": "Russia", "UKR": "Ukraine", "ISR": "Israel",
    "PSE": "Palestine", "PAK": "Pakistan", "IRN": "Iran",
    "TWN": "Taiwan", "YEM": "Yemen", "JPN": "Japan",
    "KOR": "Korea", "VNM": "Vietnam", "MEX": "Mexico",
    "CAN": "Canada", "GBR": "UK",
}

# Country -> which stock markets it affects
MARKET_IMPACT = {
    "Russia": ["US", "India", "China"],
    "Ukraine": ["US", "India", "China"],
    "Israel": ["US", "India"],
    "Palestine": ["US", "India"],
    "Yemen": ["US", "India", "China"],
    "Taiwan": ["US", "China"],
    "Pakistan": ["India"],
    "Iran": ["US", "India", "China"],
    "Japan": ["US", "China"],
    "Korea": ["US", "China"],
}


class GDELTHistoricalFetcher:
    """
    Fetches COMPLETE daily event data from GDELT daily aggregate files.

    Each daily file (one per day, 2013-04-01 to present) contains ALL
    events coded on that day.  This is fundamentally different from the
    GDELT 2.0 15-minute export files, where each event appears in only
    one of 96 daily files — downloading a subset of those would give
    incomplete coverage.

    Pipeline:
      1. Download daily ZIP files (one per day, ~10-50 KB each).
      2. Filter to root events involving target countries with >= 3 mentions.
      3. Classify via CAMEO codes into tariff / conflict categories.
      4. Map GoldsteinScale to 1-10 severity.
      5. Infer affected stock markets from actor countries.
      6. Deduplicate (keep highest-severity per 2-day window + category + country pair).
      7. Cache per-day filtered CSVs and a combined processed cache.

    First run downloads ~4,000+ files (one per day). Each file is cached
    individually so interrupted runs resume where they left off.
    """

    # ----------------------------------------------------------------
    # CAMEO EventCode -> tariff category mapping
    # ----------------------------------------------------------------
    TARIFF_CODES = {
        # ---- Policy actions (kept) ----
        # These represent actual tariff/trade policy changes that move markets.
        #
        # Easing sanctions/tariffs
        "0841": "sanctions_lifted",
        "0842": "tariff_reduction",
        # Demands / threats / restrictions
        "104": "tariff_threat",
        "131": "tariff_threat",
        "1311": "tariff_threat",
        "1312": "tariff_threat",
        "1313": "tariff_threat",
        "1231": "trade_deal_collapsed",
        "124": "trade_deal_collapsed",
        "160": "trade_war_escalation",
        "161": "tariff_increase",
        "1611": "tariff_increase",
        "162": "tariff_imposition",
        "163": "export_ban",
        "1631": "export_ban",
        "1632": "export_ban",
        "164": "sanctions_imposed",
        "1641": "sanctions_imposed",
        "1642": "sanctions_imposed",
        #
        # ---- Routine diplomatic codes REMOVED ----
        # 031* (express intent to cooperate) — routine diplomacy, ~25K events
        # 040/042/043 (consult/negotiate)    — routine diplomacy, ~64K events
        # 057/061*/071* (economic cooperation) — routine diplomacy
        # These overwhelm actual policy events (108K total vs 17K impactful)
        # and produce near-constant tariff_event_ratio features.
    }

    # ----------------------------------------------------------------
    # CAMEO EventCode -> conflict category mapping
    # ----------------------------------------------------------------
    CONFLICT_CODES = {
        "145": "military_buildup",
        "150": "military_buildup",
        "151": "military_buildup",
        "152": "military_buildup",
        "153": "military_buildup",
        "170": "conflict_escalation",
        "171": "conflict_escalation",
        "172": "sanctions_imposed",
        "173": "conflict_escalation",
        "174": "conflict_escalation",
        "175": "conflict_escalation",
        "180": "military_action",
        "181": "conflict_escalation",
        "182": "border_skirmish",
        "183": "conflict_outbreak",
        "190": "military_action",
        "191": "shipping_disruption",
        "192": "territorial_dispute",
        "193": "conflict_escalation",
        "194": "conflict_escalation",
        "195": "military_action",
        "196": "military_action",
        "200": "conflict_outbreak",
        "201": "conflict_outbreak",
        "202": "conflict_outbreak",
    }

    # Conflict root codes (fallback when specific code not found)
    CONFLICT_ROOT_CODES = {
        "15": "military_buildup",
        "17": "conflict_escalation",
        "18": "military_action",
        "19": "conflict_escalation",
        "20": "conflict_outbreak",
    }

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or (DATA_DIR / "raw" / "gdelt_events")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._download_cache = self.cache_dir / "daily"
        self._download_cache.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------

    @staticmethod
    def build_daily_url(date: datetime) -> str:
        """Build URL for a GDELT daily aggregate export file.

        These files contain ALL events coded on the given day in a single
        CSV.  URL pattern: http://data.gdeltproject.org/events/YYYYMMDD.export.CSV.zip
        """
        return f"{GDELT_DAILY_BASE}{date.strftime('%Y%m%d')}.export.CSV.zip"

    def _download_day(self, date: datetime) -> pd.DataFrame | None:
        """Download and filter a single day's complete GDELT daily file.

        Each daily file contains every event coded on that day (unlike
        the 15-minute files which partition events across 96 files).
        Returns filtered DataFrame or None on failure.
        """
        date_str = date.strftime("%Y%m%d")
        cache_path = self._download_cache / f"{date_str}.csv"

        # Return cached result if available
        if cache_path.exists():
            try:
                # Cache files are written after mixed-type numeric cleanup, but
                # pandas may still infer mixed dtypes on reload. Force string
                # to keep parsing stable and avoid noisy DtypeWarning spam.
                df = pd.read_csv(cache_path, dtype=str, low_memory=False)
                # Restore numeric dtypes expected by downstream processing.
                if "NumMentions" in df.columns:
                    df["NumMentions"] = pd.to_numeric(
                        df["NumMentions"], errors="coerce"
                    ).fillna(0)
                for col in ["GoldsteinScale", "AvgTone", "NumSources", "NumArticles"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                if not df.empty:
                    return df
            except Exception:
                pass  # re-download if corrupt

        url = self.build_daily_url(date)
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 404:
                logger.debug(f"No daily file for {date_str}")
                return None
            resp.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    # Read only the columns we need.
                    # Daily files have 58 columns (GDELT 1.0 format).
                    # We pass names for all 58 so pandas can index correctly,
                    # then usecols selects just what we need.
                    df = pd.read_csv(
                        f,
                        sep="\t",
                        header=None,
                        names=GDELT_DAILY_COLUMNS,
                        usecols=USE_COL_INDICES,
                        dtype=str,
                        on_bad_lines="skip",
                    )

            if df.empty:
                return None

            # Filter: root events only (avoids duplicated sub-events)
            df = df[df["IsRootEvent"] == "1"].copy()

            # Filter: at least one actor from target countries
            mask = (
                df["Actor1CountryCode"].isin(TARGET_COUNTRIES)
                | df["Actor2CountryCode"].isin(TARGET_COUNTRIES)
            )
            df = df[mask].copy()

            # Filter: significant events (mentioned in >= 3 sources)
            df["NumMentions"] = pd.to_numeric(
                df["NumMentions"], errors="coerce"
            ).fillna(0)
            df = df[df["NumMentions"] >= 3].copy()

            # Convert numeric columns
            for col in ["GoldsteinScale", "AvgTone", "NumSources", "NumArticles"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            if not df.empty:
                df.to_csv(cache_path, index=False)
                return df
            return None

        except requests.exceptions.RequestException as e:
            logger.debug(f"Network error for {date_str}: {e}")
            return None
        except Exception as e:
            logger.debug(f"Error processing {url}: {e}")
            return None

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify_event(self, event_code: str, event_root: str):
        """Classify a GDELT event into our tariff/conflict categories.

        Checks full CAMEO code first, then 3-digit base code, then root.
        Returns (event_type, category) or ("", "") if not relevant.
        """
        # Check specific tariff codes (full code, then base/3-digit)
        if event_code in self.TARIFF_CODES:
            return "tariff", self.TARIFF_CODES[event_code]
        base = event_code[:3] if len(event_code) >= 3 else event_code
        if base in self.TARIFF_CODES:
            return "tariff", self.TARIFF_CODES[base]

        # Check specific conflict codes
        if event_code in self.CONFLICT_CODES:
            return "conflict", self.CONFLICT_CODES[event_code]
        if base in self.CONFLICT_CODES:
            return "conflict", self.CONFLICT_CODES[base]

        # Fallback: conflict root codes
        if event_root in self.CONFLICT_ROOT_CODES:
            return "conflict", self.CONFLICT_ROOT_CODES[event_root]

        return "", ""

    @staticmethod
    def _goldstein_to_severity(goldstein: float, event_type: str) -> int:
        """Map GoldsteinScale (-10 to +10) to our severity scale (1-10)."""
        if pd.isna(goldstein):
            return 5

        if event_type == "conflict":
            # More negative = more severe
            if goldstein <= -9:
                return 10
            elif goldstein <= -7:
                return 9
            elif goldstein <= -5:
                return 8
            elif goldstein <= -3:
                return 7
            elif goldstein <= -1:
                return 6
            elif goldstein <= 1:
                return 5
            else:
                return 4
        else:
            # Tariff: severity from magnitude of impact
            mag = abs(goldstein)
            if mag >= 9:
                return 10
            elif mag >= 7:
                return 8
            elif mag >= 5:
                return 7
            elif mag >= 3:
                return 6
            elif mag >= 1:
                return 5
            else:
                return 4

    def _infer_markets(self, actor1: str, actor2: str) -> list[str]:
        """Infer affected stock markets from actor country names."""
        markets = set()
        for c in [actor1, actor2]:
            if c in ("US", "India", "China"):
                markets.add(c)
            elif c in MARKET_IMPACT:
                markets.update(MARKET_IMPACT[c])
        if not markets:
            markets = {"US", "India", "China"}
        return sorted(markets)

    # ------------------------------------------------------------------
    # Processing pipeline
    # ------------------------------------------------------------------

    # Pre-built lookup tables for vectorized classification.
    # Combine tariff + conflict codes into a single dict mapping
    # CAMEO code → (event_type, category).  Built once at class level.
    _CODE_LOOKUP: dict[str, tuple[str, str]] = {}
    for _code, _cat in TARIFF_CODES.items():
        _CODE_LOOKUP[_code] = ("tariff", _cat)
    for _code, _cat in CONFLICT_CODES.items():
        _CODE_LOOKUP[_code] = ("conflict", _cat)
    _ROOT_LOOKUP: dict[str, tuple[str, str]] = {}
    for _code, _cat in CONFLICT_ROOT_CODES.items():
        _ROOT_LOOKUP[_code] = ("conflict", _cat)

    def _process_raw(self, raw: pd.DataFrame, event_type_filter: str = "all") -> pd.DataFrame:
        """Process raw GDELT events into our standard event format.

        Fully vectorized — uses pandas map/apply on columns instead of
        iterrows(), which is ~100-200x faster on large DataFrames.
        """
        if raw.empty:
            return pd.DataFrame()

        df = raw.copy()

        # --- 1. Classify events (vectorized) ---
        codes = df["EventCode"].astype(str)
        roots = df["EventRootCode"].astype(str)
        bases = codes.str[:3]

        # Cascading lookup: full code → 3-digit base → root code
        classified = codes.map(self._CODE_LOOKUP)
        miss = classified.isna()
        if miss.any():
            classified[miss] = bases[miss].map(self._CODE_LOOKUP)
        miss = classified.isna()
        if miss.any():
            classified[miss] = roots[miss].map(self._ROOT_LOOKUP)

        # Drop unclassified rows
        miss = classified.isna()
        if miss.all():
            return pd.DataFrame()
        df = df[~miss].copy()
        classified = classified[~miss]

        # Unpack (event_type, category) tuples
        df["event_type"] = classified.apply(lambda x: x[0])
        df["category"] = classified.apply(lambda x: x[1])

        # Filter by event type if requested
        if event_type_filter != "all":
            df = df[df["event_type"] == event_type_filter]
            if df.empty:
                return pd.DataFrame()

        # --- 2. Map country codes (vectorized) ---
        a1_codes = df["Actor1CountryCode"].astype(str)
        a2_codes = df["Actor2CountryCode"].astype(str)
        df["source_country"] = a1_codes.map(COUNTRY_MAP).fillna(a1_codes)
        df["target_country"] = a2_codes.map(COUNTRY_MAP).fillna(a2_codes)

        a1_names = df["Actor1Name"].astype(str)
        a2_names = df["Actor2Name"].astype(str)
        # Use actor name if available, else mapped country
        a1_display = a1_names.where(
            (a1_names != "") & (a1_names != "nan"), df["source_country"]
        )
        a2_display = a2_names.where(
            (a2_names != "") & (a2_names != "nan"), df["target_country"]
        )

        # --- 3. Severity from GoldsteinScale (vectorized with np.select) ---
        gs = pd.to_numeric(df["GoldsteinScale"], errors="coerce")
        mag = gs.abs()
        is_conflict = df["event_type"] == "conflict"

        # Conflict severity: more negative = more severe
        conflict_sev = np.select(
            [gs <= -9, gs <= -7, gs <= -5, gs <= -3, gs <= -1, gs <= 1],
            [10, 9, 8, 7, 6, 5],
            default=4,
        )
        # Tariff severity: magnitude of impact
        tariff_sev = np.select(
            [mag >= 9, mag >= 7, mag >= 5, mag >= 3, mag >= 1],
            [10, 8, 7, 6, 5],
            default=4,
        )
        df["severity"] = np.where(is_conflict, conflict_sev, tariff_sev)
        df.loc[gs.isna(), "severity"] = 5

        # --- 4. Build output columns (vectorized) ---
        df["date"] = pd.to_datetime(df["Day"].astype(str), format="%Y%m%d", errors="coerce")
        cat_display = df["category"].str.replace("_", " ", regex=False)
        df["event"] = a1_display + " → " + a2_display + ": " + cat_display

        # Markets affected (vectorized via helper on each row — fast enough
        # since we only call it on classified rows, not 100M raw rows)
        def _markets_for_row(src, tgt):
            markets = set()
            for c in [src, tgt]:
                if c in ("US", "India", "China"):
                    markets.add(c)
                elif c in MARKET_IMPACT:
                    markets.update(MARKET_IMPACT[c])
            return sorted(markets) if markets else ["US", "India", "China"]

        markets_series = [
            _markets_for_row(s, t)
            for s, t in zip(df["source_country"], df["target_country"])
        ]

        def _countries_for_row(src, tgt):
            return sorted({c for c in [src, tgt] if c and c != "nan"})

        countries_series = [
            _countries_for_row(s, t)
            for s, t in zip(df["source_country"], df["target_country"])
        ]

        result = pd.DataFrame({
            "date": df["date"].values,
            "event": df["event"].values,
            "category": df["category"].values,
            "severity": df["severity"].values,
            "event_type": df["event_type"].values,
            "source_country": df["source_country"].values,
            "target_country": df["target_country"].values,
            "primary_countries": countries_series,
            "markets_affected": markets_series,
            "affected_markets": markets_series,
            "affected_sectors": [[] for _ in range(len(df))],
            "tone": pd.to_numeric(df["AvgTone"], errors="coerce").fillna(0).values,
            "num_mentions": pd.to_numeric(df["NumMentions"], errors="coerce").fillna(1).values,
            "confidence": 0.8,
            "data_source": "gdelt_daily",
        })

        return result.dropna(subset=["date"])

    @staticmethod
    def _dedup_chunk(df: pd.DataFrame) -> pd.DataFrame:
        """Light per-day dedup: one event per (event_type, category, country-pair)
        keeping highest severity.  Runs inside the download loop so the
        accumulated DataFrame stays small."""
        if df.empty or len(df) <= 1:
            return df
        df = df.sort_values("severity", ascending=False)
        return df.drop_duplicates(
            subset=["event_type", "category", "source_country", "target_country"],
            keep="first",
        ).reset_index(drop=True)

    def _deduplicate(self, events: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate: keep highest-severity per (2-day window, category, country-pair).

        Works in monthly chunks to keep memory usage bounded even for
        multi-million-row DataFrames.
        """
        if events.empty:
            return events

        events = events.sort_values(
            ["date", "severity"], ascending=[True, False]
        ).reset_index(drop=True)

        events["_period"] = (
            (events["date"] - pd.Timestamp("2015-01-01")).dt.days // 2
        )
        events["_ckey"] = (
            events["source_country"].astype(str)
            + "|"
            + events["target_country"].astype(str)
        )
        events = events.drop_duplicates(
            subset=["event_type", "_period", "category", "_ckey"], keep="first"
        )
        events = events.drop(columns=["_period", "_ckey"])
        return events.sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_historical_events(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        event_type: str = "all",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch historical events from GDELT daily aggregate export files.

        Downloads one file per day (each containing ALL events for that
        day), filters to target countries and relevant CAMEO codes, maps
        to our event categories, and caches results.

        Args:
            start_date: Start of range (default: ANALYSIS_START).
            end_date: End of range (default: ANALYSIS_END).
            event_type: "all", "tariff", or "conflict".
            force_refresh: Re-download everything (ignores combined cache,
                           but per-day download caches are still used).
        """
        start_date = start_date or ANALYSIS_START
        end_date = end_date or ANALYSIS_END

        # Check combined processed cache
        cache_key = hashlib.sha256(
            f"daily|{event_type}|{start_date}|{end_date}".encode()
        ).hexdigest()[:12]
        cache_path = self.cache_dir / f"historical_{cache_key}.csv"

        if cache_path.exists() and not force_refresh:
            logger.info(f"Loading cached historical events from {cache_path}")
            df = pd.read_csv(cache_path, parse_dates=["date"])
            logger.info(f"Loaded {len(df)} historical events")
            return df

        # Build date range
        start = datetime.strptime(start_date[:10], "%Y-%m-%d")
        end = datetime.strptime(end_date[:10], "%Y-%m-%d")

        # Clamp to GDELT daily file availability (2013-04-01 to present)
        start = max(start, GDELT_DAILY_START)
        # Don't try to download future dates
        today = datetime.now()
        end = min(end, today)

        dates = []
        current = start
        while current <= end:
            dates.append(current)
            current += timedelta(days=1)

        logger.info(
            f"Fetching GDELT daily files for {len(dates)} days "
            f"({start.date()} to {end.date()})..."
        )
        logger.info("First run may take 15-30 minutes. Per-day files are cached for resume.")

        # Download concurrently and process each day incrementally to avoid
        # building a massive multi-year raw DataFrame in memory.
        processed_chunks = []
        completed = 0
        errors = 0
        day_files_with_data = 0
        raw_filtered_rows = 0

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(self._download_day, d): d for d in dates}

            for future in as_completed(futures):
                day = futures[future]
                completed += 1
                if completed % 200 == 0:
                    logger.info(
                        f"Progress: {completed}/{len(dates)} days "
                        f"({completed * 100 // len(dates)}%)"
                    )
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        processed = self._process_raw(df, event_type)
                        day_files_with_data += 1
                        raw_filtered_rows += len(df)
                        if not processed.empty:
                            # Per-day dedup: collapse same-day duplicates
                            # (same category + country pair → keep highest
                            # severity).  This prevents the accumulated
                            # DataFrame from growing to millions of rows
                            # and OOM-ing during the final cross-day dedup.
                            processed = self._dedup_chunk(processed)
                            processed_chunks.append(processed)
                except Exception as e:
                    errors += 1
                    logger.debug(f"Failed processing day {day.strftime('%Y-%m-%d')}: {e}")

        logger.info(f"Downloaded {day_files_with_data} day-files with data ({errors} errors)")

        if day_files_with_data == 0:
            logger.warning("No historical events found")
            empty = pd.DataFrame()
            empty.to_csv(cache_path, index=False)
            return empty

        if not processed_chunks:
            logger.warning("No relevant classified events found after filtering")
            empty = pd.DataFrame()
            empty.to_csv(cache_path, index=False)
            return empty

        logger.info(f"Raw filtered events: {raw_filtered_rows}")
        events = pd.concat(processed_chunks, ignore_index=True)
        logger.info(f"Classified events: {len(events)}")

        # Deduplicate
        events = self._deduplicate(events)
        logger.info(f"After dedup: {len(events)}")

        # Cache
        events.to_csv(cache_path, index=False)
        logger.info(f"Cached {len(events)} historical events to {cache_path}")

        return events

    def fetch_tariff_events(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch tariff-specific historical events.

        Internally fetches ALL event types in a single pass (cached),
        then filters to tariff.  This avoids reading 19 GB of per-day
        CSVs twice when collect_data.py calls both fetch_tariff_events
        and fetch_conflict_events sequentially.
        """
        all_events = self.fetch_historical_events(
            start_date, end_date, "all", force_refresh
        )
        if all_events.empty:
            return all_events
        return all_events[
            all_events["event_type"] == "tariff"
        ].reset_index(drop=True)

    def fetch_conflict_events(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch conflict-specific historical events.

        Loads from the same "all" cache populated by fetch_tariff_events,
        so the second call is instant.
        """
        all_events = self.fetch_historical_events(
            start_date, end_date, "all", force_refresh
        )
        if all_events.empty:
            return all_events
        return all_events[
            all_events["event_type"] == "conflict"
        ].reset_index(drop=True)
