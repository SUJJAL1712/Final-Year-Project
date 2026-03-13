"""
GDELT Events 2.0 historical data fetcher.

Downloads structured event data from GDELT's publicly available export files.
These contain CAMEO-coded events (who did what to whom) with actor countries,
severity (GoldsteinScale), and tone — covering February 2015 to present with
FULL historical coverage.

Unlike the GDELT DOC 2.0 API (which has a ~3-month rolling window for article
search), these export files are permanently available. This module provides the
historical backbone for event data from 2015 onward.

No API key required — GDELT data is completely free and open.
"""

import io
import zipfile
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd
from loguru import logger

from src.utils.config import DATA_DIR, ANALYSIS_START, ANALYSIS_END

# GDELT Events 2.0 export file base URL
GDELT_EVENTS_BASE = "http://data.gdeltproject.org/gdeltv2/"

# Column names for GDELT Events 2.0 export format (61 columns, tab-separated, no header)
GDELT_COLUMNS = [
    "GlobalEventID", "Day", "MonthYear", "Year", "FractionDate",
    "Actor1Code", "Actor1Name", "Actor1CountryCode",
    "Actor1KnownGroupCode", "Actor1EthnicCode",
    "Actor1Religion1Code", "Actor1Religion2Code",
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
    "Actor2Code", "Actor2Name", "Actor2CountryCode",
    "Actor2KnownGroupCode", "Actor2EthnicCode",
    "Actor2Religion1Code", "Actor2Religion2Code",
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode", "QuadClass",
    "GoldsteinScale", "NumMentions", "NumSources", "NumArticles", "AvgTone",
    "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code", "Actor1Geo_ADM2Code",
    "Actor1Geo_Lat", "Actor1Geo_Long", "Actor1Geo_FeatureID",
    "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code", "Actor2Geo_ADM2Code",
    "Actor2Geo_Lat", "Actor2Geo_Long", "Actor2Geo_FeatureID",
    "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code", "ActionGeo_ADM2Code",
    "ActionGeo_Lat", "ActionGeo_Long", "ActionGeo_FeatureID",
    "DATEADDED", "SOURCEURL",
]

# Columns we actually read (saves memory)
USE_COLS = [
    "GlobalEventID", "Day",
    "Actor1Name", "Actor1CountryCode",
    "Actor2Name", "Actor2CountryCode",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode", "QuadClass",
    "GoldsteinScale", "NumMentions", "NumSources", "NumArticles", "AvgTone",
    "SOURCEURL",
]
USE_COL_INDICES = [GDELT_COLUMNS.index(c) for c in USE_COLS]

# Target countries (GDELT 3-letter ISO codes)
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

# Country → which stock markets it affects
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
    Fetches structured event data from GDELT Events 2.0 export files.

    Downloads one file per day (with fallback hours), filters to relevant
    events involving target countries, maps CAMEO codes to our tariff/conflict
    categories, and caches all results locally.

    First run downloads ~4,000 files (one per day from Feb 2015 to present),
    totaling ~400MB–1GB. Each file is cached individually so interrupted runs
    resume where they left off. Subsequent runs only download new days.
    """

    # CAMEO EventCode → tariff category mapping
    TARIFF_CODES = {
        # Economic cooperation / trade deals
        "031": "trade_deal_signed",
        "0311": "trade_deal_signed",
        "0331": "trade_deal_signed",
        "0341": "trade_deal_signed",
        "0356": "trade_deal_signed",
        "0561": "trade_deal_signed",
        "057": "trade_deal_signed",
        "061": "trade_deal_signed",
        "0611": "trade_deal_signed",
        "071": "trade_deal_signed",
        # Negotiations / consultations
        "040": "tariff_negotiation",
        "042": "tariff_negotiation",
        "043": "tariff_negotiation",
        "1031": "tariff_negotiation",
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
    }

    # CAMEO EventCode → conflict category mapping
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
    def _build_url(date: datetime, hour: int = 12) -> str:
        """Build URL for a GDELT Events 2.0 export file."""
        ts = date.strftime(f"%Y%m%d{hour:02d}0000")
        return f"{GDELT_EVENTS_BASE}{ts}.export.CSV.zip"

    def _download_day(self, date: datetime) -> pd.DataFrame | None:
        """Download and filter a single day's GDELT export file.

        Tries multiple hours (noon, midnight, 6am, 6pm) in case one is missing.
        Returns filtered DataFrame or None.
        """
        date_str = date.strftime("%Y%m%d")
        cache_path = self._download_cache / f"{date_str}.csv"

        # Cached already?
        if cache_path.exists():
            try:
                df = pd.read_csv(cache_path)
                if not df.empty:
                    return df
            except Exception:
                pass  # re-download if corrupt

        for hour in [12, 0, 6, 18]:
            url = self._build_url(date, hour)
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 404:
                    continue
                resp.raise_for_status()

                with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                    csv_name = z.namelist()[0]
                    with z.open(csv_name) as f:
                        df = pd.read_csv(
                            f,
                            sep="\t",
                            header=None,
                            names=GDELT_COLUMNS,
                            usecols=USE_COL_INDICES,
                            dtype=str,
                            on_bad_lines="skip",
                        )

                if df.empty:
                    continue

                # Filter: root events only
                df = df[df["IsRootEvent"] == "1"]

                # Filter: at least one actor from target countries
                mask = (
                    df["Actor1CountryCode"].isin(TARGET_COUNTRIES)
                    | df["Actor2CountryCode"].isin(TARGET_COUNTRIES)
                )
                df = df[mask]

                # Filter: significant events
                df["NumMentions"] = pd.to_numeric(
                    df["NumMentions"], errors="coerce"
                ).fillna(0)
                df = df[df["NumMentions"] >= 3]

                # Convert numerics
                for col in ["GoldsteinScale", "AvgTone", "NumSources", "NumArticles"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                if not df.empty:
                    df.to_csv(cache_path, index=False)
                    return df
                return None

            except requests.exceptions.RequestException:
                continue
            except Exception as e:
                logger.debug(f"Error processing {url}: {e}")
                continue

        return None

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify_event(self, event_code: str, event_root: str):
        """Classify a GDELT event into our tariff/conflict categories.

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

    def _process_raw(self, raw: pd.DataFrame, event_type_filter: str = "all") -> pd.DataFrame:
        """Process raw GDELT events into our standard event format."""
        records = []

        for _, row in raw.iterrows():
            code = str(row.get("EventCode", ""))
            root = str(row.get("EventRootCode", ""))

            etype, category = self._classify_event(code, root)
            if not etype:
                continue
            if event_type_filter != "all" and etype != event_type_filter:
                continue

            # Map country codes
            a1_code = str(row.get("Actor1CountryCode", ""))
            a2_code = str(row.get("Actor2CountryCode", ""))
            actor1 = COUNTRY_MAP.get(a1_code, a1_code)
            actor2 = COUNTRY_MAP.get(a2_code, a2_code)
            a1_name = str(row.get("Actor1Name", "") or actor1)
            a2_name = str(row.get("Actor2Name", "") or actor2)

            goldstein = row.get("GoldsteinScale", 0)
            severity = self._goldstein_to_severity(goldstein, etype)
            markets = self._infer_markets(actor1, actor2)
            countries = sorted({c for c in [actor1, actor2] if c})

            records.append({
                "date": pd.to_datetime(str(row.get("Day", "")), format="%Y%m%d", errors="coerce"),
                "event": f"{a1_name} \u2192 {a2_name}: {category.replace('_', ' ')}",
                "category": category,
                "severity": severity,
                "event_type": etype,
                "source_country": actor1,
                "target_country": actor2,
                "primary_countries": countries,
                "markets_affected": markets,
                "affected_markets": markets,
                "affected_sectors": [],
                "tone": row.get("AvgTone", 0),
                "num_mentions": row.get("NumMentions", 1),
                "confidence": 0.8,
                "data_source": "gdelt_events_v2",
            })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records).dropna(subset=["date"])
        return df

    def _deduplicate(self, events: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate: keep highest-severity per (2-day window, category, country-pair)."""
        if events.empty:
            return events

        events = events.sort_values(["date", "severity"], ascending=[True, False])
        events["_period"] = (
            (events["date"] - pd.Timestamp("2015-01-01")).dt.days // 2
        )
        events["_ckey"] = events.apply(
            lambda r: f"{r['source_country']}|{r['target_country']}", axis=1
        )
        events = events.drop_duplicates(
            subset=["_period", "category", "_ckey"], keep="first"
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
        Fetch historical events from GDELT Events 2.0 export files.

        Downloads one file per day, filters to target countries and relevant
        CAMEO codes, maps to our event categories, and caches results.

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
            f"hist|{event_type}|{start_date}|{end_date}".encode()
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
        # GDELT Events 2.0 started Feb 18, 2015
        gdelt_start = datetime(2015, 2, 18)
        start = max(start, gdelt_start)
        # Don't try to download future dates
        today = datetime.now()
        end = min(end, today)

        dates = []
        current = start
        while current <= end:
            dates.append(current)
            current += timedelta(days=1)

        logger.info(
            f"Fetching GDELT Events 2.0 files for {len(dates)} days "
            f"({start.date()} to {end.date()})..."
        )
        logger.info("First run may take 15-30 minutes. Per-day files are cached for resume.")

        # Download concurrently
        all_dfs = []
        completed = 0
        errors = 0

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(self._download_day, d): d for d in dates}

            for future in as_completed(futures):
                completed += 1
                if completed % 200 == 0:
                    logger.info(
                        f"Progress: {completed}/{len(dates)} days "
                        f"({completed * 100 // len(dates)}%)"
                    )
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        all_dfs.append(df)
                except Exception:
                    errors += 1

        logger.info(f"Downloaded {len(all_dfs)} day-files with data ({errors} errors)")

        if not all_dfs:
            logger.warning("No historical events found")
            empty = pd.DataFrame()
            empty.to_csv(cache_path, index=False)
            return empty

        # Combine raw events and process
        raw = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Raw filtered events: {len(raw)}")

        events = self._process_raw(raw, event_type)
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
        """Fetch tariff-specific historical events."""
        return self.fetch_historical_events(start_date, end_date, "tariff", force_refresh)

    def fetch_conflict_events(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch conflict-specific historical events."""
        return self.fetch_historical_events(start_date, end_date, "conflict", force_refresh)
