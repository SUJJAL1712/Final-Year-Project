"""
Data collection pipeline script.

Fetches all required data from real online sources:
1. Stock prices from Yahoo Finance (via yfinance)
2. Geopolitical news from GDELT Project API (free, no key)
3. RSS feeds from major news outlets
4. Classifies events via LLM (if API key available) or keywords

Usage:
    python scripts/collect_data.py --all
    python scripts/collect_data.py --stocks
    python scripts/collect_data.py --news
    python scripts/collect_data.py --events
    python scripts/collect_data.py --sentiment  (requires ANTHROPIC_API_KEY)
"""

import os
import sys
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
    """Collect latest news from RSS feeds (free, no API key)."""
    logger.info("=== Collecting RSS News ===")
    collector = NewsCollector()

    rss_articles = collector.collect_from_rss()
    logger.info(f"RSS feeds: {len(rss_articles)} relevant articles")

    return rss_articles


def collect_newsapi_news():
    """Collect news from NewsAPI (requires NEWS_API_KEY)."""
    if not os.getenv("NEWS_API_KEY"):
        logger.info(
            "NEWS_API_KEY not set — skipping NewsAPI collection. "
            "Set the key for additional news sources."
        )
        return

    logger.info("=== Collecting NewsAPI News ===")
    collector = NewsCollector()

    tariff_news = collector.collect_tariff_news()
    logger.info(f"NewsAPI tariff news: {len(tariff_news)} articles")

    conflict_news = collector.collect_conflict_news()
    logger.info(f"NewsAPI conflict news: {len(conflict_news)} articles")


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

    from src.utils.config import DATA_DIR
    from src.llm_pipeline.sentiment_analyzer import FinancialSentimentAnalyzer

    results_dir = DATA_DIR.parent / "results"
    results_dir.mkdir(exist_ok=True)
    cache_path = results_dir / "daily_sentiment.csv"

    if cache_path.exists():
        logger.info(f"Sentiment cache exists at {cache_path}")
        return

    # Load GDELT news
    gdelt = GDELTFetcher()
    news = gdelt.fetch_all_news()

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

    if args.all or not any([args.stocks, args.news, args.events, args.sentiment]):
        # Full pipeline: build_events() fetches from GDELT internally
        # and classifies events. No need to call collect_gdelt_news()
        # separately — GDELT results are cached per-query, so subsequent
        # calls won't re-hit the API, but we avoid redundant processing.
        collect_stocks()
        build_events()
        collect_rss_news()
        collect_newsapi_news()
        run_sentiment_analysis()
    else:
        if args.events:
            build_events()
        if args.stocks:
            collect_stocks()
        if args.news:
            collect_gdelt_news()
            collect_rss_news()
            collect_newsapi_news()
        if args.sentiment:
            run_sentiment_analysis()

    logger.info("Data collection complete!")


if __name__ == "__main__":
    main()
