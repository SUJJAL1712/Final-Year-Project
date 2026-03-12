"""
GDELT Project API client for fetching real news articles.

The GDELT Project (https://www.gdeltproject.org/) monitors the world's news
media and provides free API access to search articles via the DOC 2.0 API.

IMPORTANT LIMITATION: The GDELT DOC 2.0 API has a rolling lookback window
of approximately 3 months. Queries for dates older than ~3 months return
empty results. To build historical coverage:
  - Run this periodically so each quarter's data is cached locally before
    it falls out of the API window.
  - Locally cached results from previous runs are always preserved.
  - Per-query caches are keyed by (query, start_date, end_date).

This is the PRIMARY data source for geopolitical event news in this project.
No API key is required — GDELT is completely free.
"""

import time
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd
from loguru import logger

from src.utils.config import DATA_DIR


# GDELT DOC 2.0 API endpoint
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# Search queries for tariff-related news
TARIFF_QUERIES = [
    '"tariff" ("United States" OR "US") ("China" OR "Chinese")',
    '"trade war" ("United States" OR "US") China',
    '"tariff" India ("United States" OR "US")',
    '"import duty" OR "export ban" China',
    '"retaliatory tariff" OR "counter-tariff"',
    '"trade deal" ("United States" OR "US") China',
    '"CHIPS Act" OR "chip export" OR "semiconductor ban"',
    '"sanctions" ("United States" OR "US") China',
    '"tariff" "trade policy" ("India" OR "China")',
    '"entity list" OR "export control" China technology',
]

# Search queries for conflict-related news
CONFLICT_QUERIES = [
    '"military conflict" OR "armed conflict" ("India" OR "China" OR "US")',
    '"Russia" "Ukraine" "war" OR "invasion"',
    '"Israel" ("Hamas" OR "Gaza") conflict',
    '"India" "China" ("border" OR "Galwan" OR "Ladakh")',
    '"Red Sea" ("Houthi" OR "shipping")',
    '"Taiwan" ("strait" OR "military" OR "tension") China',
    '"India" "Pakistan" ("military" OR "strike" OR "border")',
    '"Iran" ("United States" OR "US") ("military" OR "strike")',
    '"sanctions" Russia ("oil" OR "energy")',
    '"South China Sea" ("dispute" OR "military")',
]


class GDELTFetcher:
    """
    Fetches real news from the GDELT Project DOC 2.0 API.

    Features:
    - No API key required (free and open)
    - Rolling ~3-month lookback window (older queries return empty)
    - Results cached per-query to disk — preserves historical data
    - Rate limiting to respect GDELT's fair-use guidelines
    - Quarterly date chunks for fine-grained caching
    """

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or (DATA_DIR / "raw" / "gdelt")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request_time = 0.0
        self._min_interval = 2.0  # seconds between requests

    def _rate_limit(self):
        """Respect GDELT's rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def _cache_key(self, query: str, start: str, end: str) -> str:
        """Generate a deterministic cache key for a query + date range."""
        raw = f"{query}|{start}|{end}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _get_cached(self, cache_file: Path) -> pd.DataFrame | None:
        """Load cached results from disk."""
        if cache_file.exists():
            try:
                df = pd.read_csv(cache_file, parse_dates=["published_at"])
                if not df.empty:
                    return df
            except Exception:
                pass
        return None

    def _query_gdelt(
        self,
        query: str,
        start_date: str = "2015-01-01",
        end_date: str = "2025-12-31",
        max_records: int = 250,
        mode: str = "artlist",
    ) -> list[dict]:
        """
        Query the GDELT DOC 2.0 API.

        Args:
            query: GDELT search query string
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_records: Max records to return (up to 250 per request)
            mode: 'artlist' for article list, 'timeline' for timeline

        Returns:
            List of article dicts with title, url, published_at, source, tone
        """
        self._rate_limit()

        # GDELT expects dates as YYYYMMDDHHMMSS
        start_fmt = datetime.strptime(start_date[:10], "%Y-%m-%d").strftime(
            "%Y%m%d%H%M%S"
        )
        end_fmt = datetime.strptime(end_date[:10], "%Y-%m-%d").strftime(
            "%Y%m%d%H%M%S"
        )

        params = {
            "query": query,
            "mode": mode,
            "maxrecords": str(max_records),
            "startdatetime": start_fmt,
            "enddatetime": end_fmt,
            "format": "json",
            "sort": "datedesc",
        }

        try:
            resp = requests.get(GDELT_DOC_API, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.JSONDecodeError:
            logger.warning(f"GDELT returned non-JSON for query: {query[:50]}...")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"GDELT API error: {e}")
            return []

        articles = []
        for article in data.get("articles", []):
            articles.append({
                "title": article.get("title", "").strip(),
                "url": article.get("url", ""),
                "published_at": article.get("seendate", ""),
                "source": article.get("domain", ""),
                "tone": article.get("tone", 0),
                "language": article.get("language", "English"),
                "source_country": article.get("sourcecountry", ""),
            })

        return articles

    def _fetch_chunked(
        self,
        queries: list[str],
        start_date: str = "2015-01-01",
        end_date: str = "2025-12-31",
        cache_name: str = "articles",
    ) -> pd.DataFrame:
        """
        Fetch articles across multiple queries and quarterly date chunks.

        GDELT DOC 2.0 API has a ~3-month rolling lookback window, so older
        queries return empty from the API. However, per-query results are
        cached locally (keyed by query+dates), so data from previous runs
        is preserved. We use quarterly chunks for fine-grained caching.
        """
        # Combined cache key includes date range to prevent stale cross-range data
        range_hash = hashlib.sha256(
            f"{cache_name}|{start_date}|{end_date}".encode()
        ).hexdigest()[:12]
        cache_file = self.cache_dir / f"{cache_name}_{range_hash}_combined.csv"

        cached = self._get_cached(cache_file)
        if cached is not None:
            logger.info(
                f"Loaded {len(cached)} cached {cache_name} articles from disk"
            )
            return cached

        all_articles = []
        empty_api_chunks = 0
        total_api_chunks = 0

        # Quarterly date boundaries
        QUARTERS = [
            ("01-01", "03-31"),
            ("04-01", "06-30"),
            ("07-01", "09-30"),
            ("10-01", "12-31"),
        ]

        start_year = max(int(start_date[:4]), 2015)
        end_year = min(int(end_date[:4]), 2026)

        for year in range(start_year, end_year + 1):
            for q_start_md, q_end_md in QUARTERS:
                chunk_start = f"{year}-{q_start_md}"
                chunk_end = f"{year}-{q_end_md}"

                # Skip chunks fully outside the requested range
                if chunk_end < start_date[:10] or chunk_start > end_date[:10]:
                    continue

                for query in queries:
                    key = self._cache_key(query, chunk_start, chunk_end)
                    query_cache = self.cache_dir / f"{cache_name}_{key}.json"

                    # Check per-query cache first
                    if query_cache.exists():
                        try:
                            with open(query_cache, "r", encoding="utf-8") as f:
                                cached_articles = json.load(f)
                            all_articles.extend(cached_articles)
                            continue
                        except Exception:
                            pass

                    # Query GDELT API
                    total_api_chunks += 1
                    articles = self._query_gdelt(
                        query, chunk_start, chunk_end, max_records=250
                    )

                    # Add query metadata
                    for a in articles:
                        a["query"] = query[:60]

                    if not articles:
                        empty_api_chunks += 1
                    else:
                        logger.info(
                            f"GDELT {chunk_start[:7]} [{query[:40]}...]: "
                            f"{len(articles)} articles"
                        )

                    # Cache per-query results (even empty to avoid re-querying)
                    try:
                        with open(query_cache, "w", encoding="utf-8") as f:
                            json.dump(articles, f, default=str)
                    except Exception as e:
                        logger.warning(f"Failed to cache query results: {e}")

                    all_articles.extend(articles)

        if total_api_chunks > 0 and empty_api_chunks > 0:
            logger.info(
                f"GDELT DOC 2.0: {empty_api_chunks}/{total_api_chunks} API "
                f"query-chunks returned empty (expected for dates older than "
                f"~3 months — the API has a rolling lookback window)"
            )

        if not all_articles:
            logger.warning(f"No {cache_name} articles found from GDELT")
            return pd.DataFrame()

        df = pd.DataFrame(all_articles)

        # Parse dates
        df["published_at"] = pd.to_datetime(
            df["published_at"], format="mixed", errors="coerce"
        )

        # Deduplicate by title
        df = df.drop_duplicates(subset=["title"], keep="first")

        # Filter to English articles only
        if "language" in df.columns:
            df = df[
                df["language"].str.lower().str.contains("english", na=True)
            ]

        # Sort by date
        df = df.sort_values("published_at").reset_index(drop=True)

        # Save combined cache
        df.to_csv(cache_file, index=False)
        logger.info(f"Cached {len(df)} {cache_name} articles to disk")

        return df

    def fetch_tariff_news(
        self,
        start_date: str = "2015-01-01",
        end_date: str = "2025-12-31",
    ) -> pd.DataFrame:
        """
        Fetch tariff-related news articles from GDELT.

        Returns DataFrame with columns:
        title, url, published_at, source, tone, query
        """
        return self._fetch_chunked(
            queries=TARIFF_QUERIES,
            start_date=start_date,
            end_date=end_date,
            cache_name="tariff",
        )

    def fetch_conflict_news(
        self,
        start_date: str = "2015-01-01",
        end_date: str = "2025-12-31",
    ) -> pd.DataFrame:
        """
        Fetch conflict-related news articles from GDELT.

        Returns DataFrame with columns:
        title, url, published_at, source, tone, query
        """
        return self._fetch_chunked(
            queries=CONFLICT_QUERIES,
            start_date=start_date,
            end_date=end_date,
            cache_name="conflict",
        )

    def fetch_all_news(
        self,
        start_date: str = "2015-01-01",
        end_date: str = "2025-12-31",
    ) -> pd.DataFrame:
        """Fetch both tariff and conflict news, combined and deduplicated."""
        tariff = self.fetch_tariff_news(start_date, end_date)
        conflict = self.fetch_conflict_news(start_date, end_date)

        frames = []
        if not tariff.empty:
            frames.append(tariff)
        if not conflict.empty:
            frames.append(conflict)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["title"], keep="first")
        combined = combined.sort_values("published_at").reset_index(drop=True)

        return combined
