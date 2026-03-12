"""
Data collection pipeline script.

Usage:
    python scripts/collect_data.py --all
    python scripts/collect_data.py --stocks
    python scripts/collect_data.py --news
    python scripts/collect_data.py --events
"""

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


def collect_news():
    """Collect news articles for LLM analysis."""
    logger.info("=== Collecting News Data ===")
    collector = NewsCollector()

    tariff_news = collector.collect_tariff_news()
    logger.info(f"Tariff news: {len(tariff_news)} articles")

    conflict_news = collector.collect_conflict_news()
    logger.info(f"Conflict news: {len(conflict_news)} articles")


def collect_events():
    """Build curated event databases."""
    logger.info("=== Building Event Database ===")

    tariff_tracker = TariffEventTracker()
    tariff_events = tariff_tracker.get_curated_tariff_events()
    logger.info(f"Tariff events: {len(tariff_events)}")

    conflict_tracker = ConflictEventTracker()
    conflict_events = conflict_tracker.get_curated_conflict_events()
    logger.info(f"Conflict events: {len(conflict_events)}")

    combined = conflict_tracker.get_combined_geopolitical_events()
    logger.info(f"Combined events: {len(combined)}")


def main():
    parser = argparse.ArgumentParser(description="Data Collection Pipeline")
    parser.add_argument("--all", action="store_true", help="Collect all data")
    parser.add_argument("--stocks", action="store_true", help="Collect stock data only")
    parser.add_argument("--news", action="store_true", help="Collect news data only")
    parser.add_argument("--events", action="store_true", help="Build event database only")
    args = parser.parse_args()

    setup_logger("INFO")

    if args.all or not any([args.stocks, args.news, args.events]):
        collect_events()
        collect_stocks()
        collect_news()
    else:
        if args.events:
            collect_events()
        if args.stocks:
            collect_stocks()
        if args.news:
            collect_news()

    logger.info("Data collection complete!")


if __name__ == "__main__":
    main()
