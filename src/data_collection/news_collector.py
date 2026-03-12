"""
News data collection for geopolitical event analysis.

Collects news articles related to tariffs, trade wars, and regional conflicts
from multiple sources. The news text is later processed by the LLM pipeline
for structured event extraction.
"""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
import feedparser
import pandas as pd
from loguru import logger

from src.utils.config import DATA_DIR


class NewsCollector:
    """Collects news articles from multiple sources."""

    # RSS feeds for geopolitical and trade news (free, no API key needed)
    RSS_FEEDS = {
        "reuters_world": "https://feeds.reuters.com/Reuters/worldNews",
        "reuters_business": "https://feeds.reuters.com/reuters/businessNews",
        "bbc_world": "http://feeds.bbci.co.uk/news/world/rss.xml",
        "bbc_business": "http://feeds.bbci.co.uk/news/business/rss.xml",
        "ft_world": "https://www.ft.com/world?format=rss",
        "aljazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    }

    # Search keywords for targeted collection
    TARIFF_KEYWORDS = [
        "tariff", "trade war", "import duty", "export ban",
        "trade deal", "trade agreement", "trade sanctions",
        "customs duty", "protectionism", "free trade",
        "retaliatory tariff", "trade deficit", "WTO",
    ]

    CONFLICT_KEYWORDS = [
        "military conflict", "armed conflict", "border clash",
        "war", "ceasefire", "sanctions", "invasion",
        "territorial dispute", "military buildup",
        "missile strike", "naval blockade",
    ]

    COUNTRY_KEYWORDS = {
        "US": ["United States", "US", "America", "Washington", "Trump", "Biden"],
        "China": ["China", "Beijing", "Xi Jinping", "Chinese", "PRC"],
        "India": ["India", "New Delhi", "Modi", "Indian"],
    }

    def __init__(self):
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.data_dir = DATA_DIR / "raw" / "news"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def collect_from_newsapi(
        self,
        query: str,
        from_date: str | None = None,
        to_date: str | None = None,
        page_size: int = 100,
    ) -> list[dict]:
        """
        Collect articles from NewsAPI.org (requires free API key).

        Args:
            query: Search query
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            page_size: Number of results per page
        """
        if not self.news_api_key:
            logger.warning("NEWS_API_KEY not set, skipping NewsAPI collection")
            return []

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "apiKey": self.news_api_key,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": min(page_size, 100),
        }
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            articles = []
            for article in data.get("articles", []):
                articles.append({
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "content": article.get("content", ""),
                    "source": article.get("source", {}).get("name", ""),
                    "published_at": article.get("publishedAt", ""),
                    "url": article.get("url", ""),
                    "query": query,
                    "collector": "newsapi",
                })

            logger.info(f"NewsAPI: collected {len(articles)} articles for '{query}'")
            return articles

        except Exception as e:
            logger.error(f"NewsAPI error: {e}")
            return []

    def collect_from_rss(self) -> list[dict]:
        """Collect latest articles from RSS feeds."""
        articles = []

        for feed_name, feed_url in self.RSS_FEEDS.items():
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries:
                    text = f"{entry.get('title', '')} {entry.get('summary', '')}"

                    # Filter for relevant content
                    is_relevant = any(
                        kw.lower() in text.lower()
                        for kw in self.TARIFF_KEYWORDS + self.CONFLICT_KEYWORDS
                    )

                    if is_relevant:
                        articles.append({
                            "title": entry.get("title", ""),
                            "description": entry.get("summary", ""),
                            "content": entry.get("summary", ""),
                            "source": feed_name,
                            "published_at": entry.get("published", ""),
                            "url": entry.get("link", ""),
                            "query": "rss_feed",
                            "collector": "rss",
                        })

                logger.info(f"RSS {feed_name}: found {len(articles)} relevant articles")

            except Exception as e:
                logger.error(f"RSS feed {feed_name} error: {e}")

        return articles

    def collect_tariff_news(
        self, from_date: str | None = None, to_date: str | None = None
    ) -> pd.DataFrame:
        """
        Collect all tariff-related news articles from multiple sources.

        Sources:
        1. GDELT API (free, no key, covers 2015+)
        2. NewsAPI (requires NEWS_API_KEY for recent articles)
        3. RSS feeds (free, latest articles only)
        """
        all_articles = []

        # Source 1: GDELT (primary, covers full historical range)
        try:
            from src.data_collection.gdelt_fetcher import GDELTFetcher
            gdelt = GDELTFetcher()
            gdelt_news = gdelt.fetch_tariff_news(
                start_date=from_date or "2015-01-01",
                end_date=to_date or "2025-12-31",
            )
            if not gdelt_news.empty:
                for _, row in gdelt_news.iterrows():
                    all_articles.append({
                        "title": row.get("title", ""),
                        "description": "",
                        "content": row.get("title", ""),
                        "source": row.get("source", ""),
                        "published_at": row.get("published_at", ""),
                        "url": row.get("url", ""),
                        "query": row.get("query", "gdelt"),
                        "collector": "gdelt",
                    })
                logger.info(f"GDELT: {len(gdelt_news)} tariff articles")
        except Exception as e:
            logger.error(f"GDELT tariff collection failed: {e}")

        # Source 2: NewsAPI (if key available)
        queries = [
            "US China tariff trade war",
            "India tariff trade policy",
            "import duty increase",
            "retaliatory tariffs",
            "trade war escalation",
        ]
        for query in queries:
            articles = self.collect_from_newsapi(query, from_date, to_date)
            all_articles.extend(articles)
            time.sleep(1)

        # Source 3: RSS feeds
        rss_articles = self.collect_from_rss()
        tariff_rss = [
            a for a in rss_articles
            if any(kw in a.get("title", "").lower() for kw in self.TARIFF_KEYWORDS)
        ]
        all_articles.extend(tariff_rss)

        df = pd.DataFrame(all_articles)
        if not df.empty:
            df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
            df = df.drop_duplicates(subset=["title"]).sort_values("published_at")
            df.to_csv(self.data_dir / "tariff_news.csv", index=False)

        logger.info(f"Total tariff news collected: {len(df)}")
        return df

    def collect_conflict_news(
        self, from_date: str | None = None, to_date: str | None = None
    ) -> pd.DataFrame:
        """
        Collect all conflict-related news articles from multiple sources.

        Sources:
        1. GDELT API (free, no key, covers 2015+)
        2. NewsAPI (requires NEWS_API_KEY for recent articles)
        3. RSS feeds (free, latest articles only)
        """
        all_articles = []

        # Source 1: GDELT (primary)
        try:
            from src.data_collection.gdelt_fetcher import GDELTFetcher
            gdelt = GDELTFetcher()
            gdelt_news = gdelt.fetch_conflict_news(
                start_date=from_date or "2015-01-01",
                end_date=to_date or "2025-12-31",
            )
            if not gdelt_news.empty:
                for _, row in gdelt_news.iterrows():
                    all_articles.append({
                        "title": row.get("title", ""),
                        "description": "",
                        "content": row.get("title", ""),
                        "source": row.get("source", ""),
                        "published_at": row.get("published_at", ""),
                        "url": row.get("url", ""),
                        "query": row.get("query", "gdelt"),
                        "collector": "gdelt",
                    })
                logger.info(f"GDELT: {len(gdelt_news)} conflict articles")
        except Exception as e:
            logger.error(f"GDELT conflict collection failed: {e}")

        # Source 2: NewsAPI
        queries = [
            "military conflict regional war",
            "Russia Ukraine war impact",
            "Israel Hamas conflict",
            "India China border dispute Galwan",
            "Red Sea shipping crisis Houthi",
            "Taiwan strait tension",
        ]
        for query in queries:
            articles = self.collect_from_newsapi(query, from_date, to_date)
            all_articles.extend(articles)
            time.sleep(1)

        # Source 3: RSS feeds
        rss_articles = self.collect_from_rss()
        conflict_rss = [
            a for a in rss_articles
            if any(kw in a.get("title", "").lower() for kw in self.CONFLICT_KEYWORDS)
        ]
        all_articles.extend(conflict_rss)

        df = pd.DataFrame(all_articles)
        if not df.empty:
            df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
            df = df.drop_duplicates(subset=["title"]).sort_values("published_at")
            df.to_csv(self.data_dir / "conflict_news.csv", index=False)

        logger.info(f"Total conflict news collected: {len(df)}")
        return df

    def build_event_news_corpus(self, events: list[dict]) -> pd.DataFrame:
        """
        For each known event, collect surrounding news articles.
        This creates the corpus for LLM analysis.

        Args:
            events: List of event dicts with 'date' and 'event' keys
        """
        all_articles = []

        for event in events:
            event_date = pd.Timestamp(event["date"])
            from_date = (event_date - timedelta(days=3)).strftime("%Y-%m-%d")
            to_date = (event_date + timedelta(days=3)).strftime("%Y-%m-%d")

            articles = self.collect_from_newsapi(
                event["event"], from_date, to_date
            )
            for a in articles:
                a["reference_event"] = event["event"]
                a["reference_date"] = event["date"]
            all_articles.extend(articles)
            time.sleep(1)

        df = pd.DataFrame(all_articles)
        if not df.empty:
            df.to_csv(self.data_dir / "event_news_corpus.csv", index=False)

        return df

    def load_cached_news(self, category: str = "tariff") -> pd.DataFrame | None:
        """Load previously collected news from disk."""
        path = self.data_dir / f"{category}_news.csv"
        if path.exists():
            return pd.read_csv(path, parse_dates=["published_at"])
        return None
