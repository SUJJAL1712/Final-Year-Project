"""
Data collection pipeline script.

Fetches all required data from real online sources:
1. Stock prices from Yahoo Finance (via yfinance)
2. Historical events from GDELT daily aggregate export files (2013-present)
3. Recent geopolitical news from GDELT DOC 2.0 API (~last 3 months only)
4. RSS feeds from major news outlets
5. Classifies events via LLM (if API key available) or keywords

Usage:
    python scripts/collect_data.py --all
    python scripts/collect_data.py --stocks
    python scripts/collect_data.py --historical   (download GDELT daily files)
    python scripts/collect_data.py --news
    python scripts/collect_data.py --events
    python scripts/collect_data.py --sentiment    (requires ANTHROPIC_API_KEY)
"""

import os
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from src.utils.logger import setup_logger
from src.data_collection.stock_fetcher import StockDataFetcher
from src.data_collection.news_collector import NewsCollector
from src.data_collection.tariff_tracker import TariffEventTracker
from src.data_collection.conflict_tracker import ConflictEventTracker
from src.data_collection.gdelt_fetcher import GDELTFetcher


def collect_historical_events():
    """Download GDELT daily aggregate files for full historical coverage.

    This is the historical backbone — CAMEO-coded events with actor countries,
    GoldsteinScale severity, and tone. Each daily file contains ALL events
    coded on that day (not a sample). Downloads ~4,000+ files (one per day
    from 2013/2015 to present) on first run, then only new days on subsequent runs.
    """
    logger.info("=== Collecting Historical Events (GDELT daily files) ===")
    from src.data_collection.gdelt_historical import GDELTHistoricalFetcher

    fetcher = GDELTHistoricalFetcher()

    tariff_events = fetcher.fetch_tariff_events()
    logger.info(f"Historical tariff events: {len(tariff_events)}")

    conflict_events = fetcher.fetch_conflict_events()
    logger.info(f"Historical conflict events: {len(conflict_events)}")


def collect_stocks():
    """Fetch all stock data for US, India, China markets."""
    logger.info("=== Collecting Stock Data ===")
    fetcher = StockDataFetcher()

    # Main indices
    logger.info("Fetching main indices...")
    index_prices = fetcher.get_index_prices()
    logger.info(f"Fetched {len(index_prices)} index series")

    # Sector ETFs (US)
    logger.info("Fetching US sector ETFs...")
    sector_data = fetcher.get_sector_data("US")
    logger.info(f"Fetched {len(sector_data)} sector series")

    # Key stocks
    for market in ["US", "India", "China"]:
        logger.info(f"Fetching {market} key stocks...")
        data = fetcher.fetch_market(market)
        logger.info(f"Fetched {len(data)} symbols for {market}")


def collect_gdelt_news():
    """Fetch geopolitical news from GDELT Project API (free, no API key)."""
    logger.info("=== Collecting News from GDELT ===")
    gdelt = GDELTFetcher()

    tariff_news = gdelt.fetch_tariff_news()
    logger.info(f"GDELT tariff news: {len(tariff_news)} articles")

    conflict_news = gdelt.fetch_conflict_news()
    logger.info(f"GDELT conflict news: {len(conflict_news)} articles")

    combined = gdelt.fetch_all_news()
    logger.info(f"GDELT total news: {len(combined)} articles")


def collect_rss_news():
    """Collect latest news from RSS feeds (free, no API key).

    Persists articles to CSV so the sentiment pipeline can reload them.
    """
    import pandas as pd

    logger.info("=== Collecting RSS News ===")
    collector = NewsCollector()

    rss_articles = collector.collect_from_rss()
    logger.info(f"RSS feeds: {len(rss_articles)} relevant articles")

    # Save to CSV so run_sentiment_analysis() can pick them up
    if rss_articles:
        rss_df = pd.DataFrame(rss_articles)
        rss_df["published_at"] = pd.to_datetime(
            rss_df["published_at"], errors="coerce"
        )
        rss_path = collector.data_dir / "rss_articles.csv"
        rss_df.to_csv(rss_path, index=False)
        logger.info(f"Saved {len(rss_df)} RSS articles to {rss_path}")

    return rss_articles


def collect_newsapi_news():
    """Collect news from NewsAPI only (requires NEWS_API_KEY).

    Uses only the NewsAPI-specific endpoint to avoid re-fetching
    GDELT/RSS data that build_events() already handles.
    """
    if not os.getenv("NEWS_API_KEY"):
        logger.info(
            "NEWS_API_KEY not set — skipping NewsAPI collection. "
            "Set the key for additional news sources."
        )
        return

    import pandas as pd

    logger.info("=== Collecting NewsAPI News ===")
    collector = NewsCollector()

    queries = [
        "US China tariff trade war",
        "India tariff trade policy",
        "retaliatory tariffs",
        "military conflict regional war",
        "Russia Ukraine war impact",
        "Israel Hamas conflict",
    ]

    all_articles = []
    for query in queries:
        articles = collector.collect_from_newsapi(query)
        all_articles.extend(articles)
        time.sleep(1)

    if all_articles:
        df = pd.DataFrame(all_articles)
        df.to_csv(collector.data_dir / "newsapi_articles.csv", index=False)
        logger.info(f"NewsAPI: collected {len(df)} articles total")


def build_events():
    """
    Build event databases from GDELT news.

    Uses LLM classification if ANTHROPIC_API_KEY is set,
    otherwise falls back to keyword-based classification.
    """
    logger.info("=== Building Event Database from GDELT ===")

    tariff_tracker = TariffEventTracker()
    tariff_events = tariff_tracker.get_tariff_events()
    logger.info(f"Tariff events: {len(tariff_events)}")

    conflict_tracker = ConflictEventTracker()
    conflict_events = conflict_tracker.get_conflict_events()
    logger.info(f"Conflict events: {len(conflict_events)}")

    combined = conflict_tracker.get_combined_geopolitical_events()
    logger.info(f"Combined events: {len(combined)}")


def run_sentiment_analysis():
    """
    Run LLM sentiment analysis on collected news (requires ANTHROPIC_API_KEY).
    Results are cached to disk.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.info(
            "ANTHROPIC_API_KEY not set — skipping sentiment analysis. "
            "Set the key to enable LLM features."
        )
        return

    logger.info("=== Running LLM Sentiment Analysis ===")

    import pandas as pd
    from src.utils.config import DATA_DIR
    from src.llm_pipeline.sentiment_analyzer import FinancialSentimentAnalyzer

    results_dir = DATA_DIR.parent / "results"
    results_dir.mkdir(exist_ok=True)
    cache_path = results_dir / "daily_sentiment.csv"

    # Check cache staleness (consistent with run_analysis.py)
    if cache_path.exists():
        try:
            cached = pd.read_csv(cache_path, parse_dates=["date"], index_col="date")
            if not cached.empty:
                cache_max = cached.index.max()
                age_days = (pd.Timestamp.now() - cache_max).days
                if age_days < 90:
                    logger.info(
                        f"Sentiment cache fresh ({len(cached)} days, "
                        f"latest: {cache_max.date()}), skipping"
                    )
                    return
                logger.info(
                    f"Sentiment cache stale ({age_days} days), refreshing..."
                )
        except Exception as e:
            logger.warning(f"Failed to read sentiment cache: {e}")

    # Use historical GDELT events (108K+ spanning 2013-present) instead of
    # the DOC 2.0 API which only covers ~3 months and is rate-limited.
    # Load already-cached tariff + conflict events from --historical/--events.
    from src.data_collection.gdelt_historical import GDELTHistoricalFetcher

    hist = GDELTHistoricalFetcher()
    tariff_hist = hist.fetch_tariff_events()
    conflict_hist = hist.fetch_conflict_events()
    frames = [df for df in [tariff_hist, conflict_hist] if not df.empty]

    if frames:
        all_hist = pd.concat(frames, ignore_index=True)
        news = pd.DataFrame({
            "title": all_hist["event"],
            "published_at": all_hist["date"],
            "source": "gdelt_historical",
        })
        logger.info(f"Loaded {len(news)} historical events")
        # Sample to control API cost: keep up to 15 highest-severity events
        # per day (total, not per country). With Haiku this costs ~$1.50.
        if "severity" in all_hist.columns:
            sampled = all_hist.sort_values("severity", ascending=False)
            sampled = sampled.groupby(sampled["date"].dt.date).head(15)
            news = pd.DataFrame({
                "title": sampled["event"],
                "published_at": sampled["date"],
                "source": "gdelt_historical",
            })
        logger.info(f"Sampled {len(news)} events for sentiment analysis (top 15/day by severity)")
    else:
        news = pd.DataFrame()

    # Also load any cached NewsAPI/RSS articles for richer coverage
    news_dir = DATA_DIR / "raw" / "news"
    for cached_file in [
        "tariff_news.csv",
        "conflict_news.csv",
        "newsapi_articles.csv",
        "rss_articles.csv",
    ]:
        cached_path = news_dir / cached_file
        if cached_path.exists():
            try:
                extra = pd.read_csv(cached_path, parse_dates=["published_at"])
                if not extra.empty and "title" in extra.columns:
                    news = pd.concat([news, extra], ignore_index=True)
                    logger.info(f"Added {len(extra)} articles from {cached_file}")
            except Exception as e:
                logger.warning(f"Failed to load {cached_file}: {e}")

    # Deduplicate across sources by (title, date) — same headline on
    # different dates may be distinct events, only collapse same-day copies.
    if not news.empty:
        news["published_at"] = pd.to_datetime(
            news["published_at"], errors="coerce"
        )
        news["_pub_date"] = news["published_at"].dt.date
        news = news.drop_duplicates(
            subset=["title", "_pub_date"], keep="first"
        )
        news = news.drop(columns=["_pub_date"])

    if news.empty:
        logger.warning("No news for sentiment analysis")
        return

    analyzer = FinancialSentimentAnalyzer()
    logger.info(f"Analyzing sentiment for {len(news)} articles...")
    sentiment_results = analyzer.analyze_batch(news)

    if not sentiment_results.empty:
        daily = analyzer.compute_daily_sentiment(sentiment_results)
        if not daily.empty:
            daily.to_csv(cache_path)
            logger.info(f"Saved daily sentiment ({len(daily)} days)")


def main():
    parser = argparse.ArgumentParser(description="Data Collection Pipeline")
    parser.add_argument(
        "--all", action="store_true", help="Collect all data"
    )
    parser.add_argument(
        "--stocks", action="store_true", help="Collect stock data only"
    )
    parser.add_argument(
        "--historical",
        action="store_true",
        help="Download GDELT daily aggregate files (full historical coverage)",
    )
    parser.add_argument(
        "--news", action="store_true", help="Collect news data (GDELT + RSS)"
    )
    parser.add_argument(
        "--events", action="store_true", help="Build event database"
    )
    parser.add_argument(
        "--sentiment",
        action="store_true",
        help="Run LLM sentiment (requires ANTHROPIC_API_KEY)",
    )
    args = parser.parse_args()

    setup_logger("INFO")

    if args.all or not any([
        args.stocks, args.historical, args.news, args.events, args.sentiment
    ]):
        # Full pipeline: historical events first, then build_events() merges
        # historical backbone with recent DOC 2.0 articles.
        collect_stocks()
        collect_historical_events()
        build_events()
        collect_rss_news()
        collect_newsapi_news()
        run_sentiment_analysis()
    else:
        if args.stocks:
            collect_stocks()
        if args.historical:
            collect_historical_events()
        if args.events:
            build_events()
        if args.news:
            collect_gdelt_news()
            collect_rss_news()
            collect_newsapi_news()
        if args.sentiment:
            run_sentiment_analysis()

    logger.info("Data collection complete!")


if __name__ == "__main__":
    main()
