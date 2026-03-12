"""
Full analysis pipeline script.

Runs the complete analysis workflow:
1. DC analysis across all markets
2. LLM sentiment analysis (if --skip-llm is not passed)
3. Event study analysis
4. Cross-market contagion analysis
5. Granger causality
6. Feature engineering and model training
7. Ablation study

Usage:
    python scripts/run_analysis.py
    python scripts/run_analysis.py --market US
    python scripts/run_analysis.py --skip-llm  (run without LLM calls)
"""

import os
import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from loguru import logger

from src.utils.logger import setup_logger
from src.utils.config import DATA_DIR
from src.data_collection.stock_fetcher import StockDataFetcher
from src.data_collection.conflict_tracker import ConflictEventTracker
from src.data_collection.gdelt_fetcher import GDELTFetcher
from src.directional_changes.dc_algorithm import (
    DirectionalChangeDetector,
    MultiThresholdDC,
)
from src.directional_changes.intrinsic_time import IntrinsicTimeAnalyzer
from src.analysis.event_study import EventStudyAnalyzer
from src.analysis.dc_event_correlation import DCEventCorrelator
from src.analysis.cross_market_contagion import CrossMarketContagionAnalyzer
from src.analysis.granger_causality import GrangerCausalityAnalyzer
from src.models.feature_engineering import FeatureEngineer
from src.models.hybrid_model import HybridDCLLMPredictor
from src.models.baselines import BaselineModels
from src.utils.config import ANALYSIS_START, ANALYSIS_END
from src.utils.helpers import compute_returns


RESULTS_DIR = DATA_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_dc_analysis(prices: pd.Series, market_name: str):
    """Run DC analysis for a single market."""
    logger.info(f"=== DC Analysis: {market_name} ===")

    # Multi-threshold analysis
    multi_dc = MultiThresholdDC()
    scaling = multi_dc.scaling_law_data(prices)
    scaling.to_csv(RESULTS_DIR / f"scaling_laws_{market_name}.csv", index=False)
    logger.info(f"Scaling laws computed for {market_name}")

    # DC events at primary threshold
    detector = DirectionalChangeDetector(0.02)
    summaries = detector.detect(prices)
    dc_df = detector.to_dataframe(summaries)
    dc_df.to_csv(RESULTS_DIR / f"dc_events_{market_name}.csv", index=False)
    logger.info(f"DC events: {len(dc_df)} at theta=0.02")

    # Intrinsic time
    analyzer = IntrinsicTimeAnalyzer(0.02)
    bursts = analyzer.detect_activity_bursts(prices)
    bursts.to_csv(
        RESULTS_DIR / f"activity_bursts_{market_name}.csv", index=False
    )
    logger.info(f"Activity bursts: {len(bursts)}")

    return dc_df


def run_llm_analysis(events_df: pd.DataFrame, skip_llm: bool) -> pd.DataFrame:
    """
    Run LLM sentiment analysis on event-related news.

    Returns a DataFrame of daily sentiment per market (or empty if skipped).
    """
    if skip_llm:
        logger.info("--skip-llm passed: skipping LLM analysis")
        return pd.DataFrame()

    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.warning(
            "ANTHROPIC_API_KEY not set — skipping LLM sentiment analysis. "
            "Set the key to enable LLM features."
        )
        return pd.DataFrame()

    from src.llm_pipeline.sentiment_analyzer import FinancialSentimentAnalyzer

    logger.info("=== LLM Sentiment Analysis ===")
    cache_path = RESULTS_DIR / "daily_sentiment.csv"

    if cache_path.exists():
        try:
            cached = pd.read_csv(cache_path, parse_dates=["date"], index_col="date")
            if not cached.empty:
                cache_max = cached.index.max()
                age_days = (pd.Timestamp.now() - cache_max).days
                if age_days < 90:
                    logger.info(
                        f"Loading cached sentiment from {cache_path} "
                        f"({len(cached)} days, latest: {cache_max.date()})"
                    )
                    return cached
                logger.info(
                    f"Sentiment cache stale (latest: {cache_max.date()}, "
                    f"{age_days} days old), refreshing..."
                )
        except Exception as e:
            logger.warning(f"Failed to read sentiment cache: {e}")

    # Use GDELT news articles for sentiment analysis
    gdelt = GDELTFetcher()
    news = gdelt.fetch_all_news()

    # Also load any cached NewsAPI/RSS articles for richer coverage
    news_dir = DATA_DIR / "raw" / "news"
    for cached_file in ["tariff_news.csv", "conflict_news.csv", "newsapi_articles.csv"]:
        cached_path = news_dir / cached_file
        if cached_path.exists():
            try:
                extra = pd.read_csv(cached_path, parse_dates=["published_at"])
                if not extra.empty and "title" in extra.columns:
                    news = pd.concat([news, extra], ignore_index=True)
                    logger.info(f"Added {len(extra)} articles from {cached_file}")
            except Exception as e:
                logger.warning(f"Failed to load {cached_file}: {e}")

    # Deduplicate across sources
    if not news.empty:
        news = news.drop_duplicates(subset=["title"], keep="first")

    if news.empty:
        logger.warning("No news articles for sentiment analysis")
        return pd.DataFrame()

    analyzer = FinancialSentimentAnalyzer()

    # Use batch mode for efficiency
    logger.info(f"Analyzing sentiment for {len(news)} articles...")
    sentiment_results = analyzer.analyze_batch(news)

    if sentiment_results.empty:
        return pd.DataFrame()

    # Aggregate to daily sentiment
    daily_sentiment = analyzer.compute_daily_sentiment(sentiment_results)

    if not daily_sentiment.empty:
        daily_sentiment.to_csv(cache_path)
        logger.info(f"Saved daily sentiment ({len(daily_sentiment)} days)")

    return daily_sentiment


def run_event_study(
    prices: pd.Series, events_df: pd.DataFrame, market_name: str
):
    """Run event study analysis."""
    logger.info(f"=== Event Study: {market_name} ===")

    analyzer = EventStudyAnalyzer()
    event_dates = events_df["date"].astype(str).tolist()

    if not event_dates:
        logger.warning(f"No events for {market_name}, skipping event study")
        return pd.DataFrame()

    # Individual event results
    results = analyzer.multi_event_study(prices, event_dates)
    results.to_csv(
        RESULTS_DIR / f"event_study_{market_name}.csv", index=False
    )

    # Average CAR
    acar = analyzer.average_car(prices, event_dates)
    if "error" not in acar:
        logger.info(
            f"ACAR: {acar['acar'].iloc[-1]:.6f}, "
            f"n={acar['n_events']}, "
            f"significant events={results['significant'].sum()}"
        )

    return results


def run_dc_event_correlation(
    prices: pd.Series, events_df: pd.DataFrame, market_name: str
):
    """Test DC-geopolitical event correlation."""
    logger.info(f"=== DC-Event Correlation: {market_name} ===")

    correlator = DCEventCorrelator(0.02)
    event_dates = events_df["date"].astype(str).tolist()

    if not event_dates:
        logger.warning(f"No events for {market_name}, skipping correlation")
        return {}

    coincidence = correlator.temporal_coincidence(prices, event_dates)
    logger.info(
        f"Enrichment ratio: {coincidence['enrichment_ratio']:.2f}x "
        f"(p={coincidence['p_value']:.4f})"
    )

    magnitude = correlator.dc_magnitude_around_events(prices, event_dates)
    logger.info(
        f"Near-event avg magnitude: {magnitude['near_event_avg_magnitude']:.6f}"
    )
    logger.info(
        f"Normal avg magnitude: {magnitude['normal_avg_magnitude']:.6f}"
    )

    return {"coincidence": coincidence, "magnitude": magnitude}


def run_cross_market_contagion(
    market_prices: dict[str, pd.Series],
    events_df: pd.DataFrame,
):
    """Run cross-market contagion analysis."""
    logger.info("=== Cross-Market Contagion Analysis ===")

    contagion = CrossMarketContagionAnalyzer()

    # Compute returns
    market_returns = {}
    for name, prices in market_prices.items():
        ret = compute_returns(prices)
        ret.index = ret.index.normalize()
        market_returns[name] = ret

    # Spillover matrix
    spillover = contagion.compute_spillover_matrix(market_returns)
    spillover_df = pd.DataFrame(spillover).T
    spillover_df.to_csv(RESULTS_DIR / "spillover_matrix.csv")
    logger.info(f"Spillover matrix:\n{spillover_df}")

    # Multi-event contagion
    if not events_df.empty:
        contagion_results = contagion.multi_event_contagion(
            market_prices, events_df
        )
        if not contagion_results.empty:
            contagion_results.to_csv(
                RESULTS_DIR / "contagion_results.csv", index=False
            )
            logger.info(f"Contagion: {len(contagion_results)} events analyzed")

    # India trade diversion test (US-China bilateral tariff events only)
    if "US" in market_prices and "India" in market_prices and "China" in market_prices:
        # Filter to US-China bilateral tariff events specifically
        tariff_events = pd.DataFrame()
        if "event_type" in events_df.columns:
            tariff_all = events_df[events_df["event_type"] == "tariff"]
            if not tariff_all.empty and "markets_affected" in tariff_all.columns:
                # Keep only events that affect BOTH US and China
                tariff_events = tariff_all[
                    tariff_all["markets_affected"].apply(
                        lambda x: "US" in str(x) and "China" in str(x)
                    )
                ]

        if not tariff_events.empty:
            diversion = contagion.india_trade_diversion_test(
                market_prices["US"],
                market_prices["India"],
                market_prices["China"],
                tariff_events["date"].astype(str).tolist(),
            )
            logger.info(
                f"India trade diversion: "
                f"benefit={diversion.get('india_benefits_from_us_china_tension')}, "
                f"correlation={diversion.get('us_china_spread_india_correlation', 'N/A')}"
            )
            # Save (excluding DataFrame in 'reactions' key)
            diversion_save = {
                k: v for k, v in diversion.items() if k != "reactions"
            }
            with open(RESULTS_DIR / "india_trade_diversion.json", "w") as f:
                json.dump(diversion_save, f, indent=2, default=str)


def run_model_training(
    prices: pd.Series,
    events_df: pd.DataFrame,
    market_name: str,
    sentiment_df: pd.DataFrame | None = None,
):
    """Train hybrid model and run ablation study."""
    logger.info(f"=== Model Training: {market_name} ===")

    market_id = market_name.lower()
    engineer = FeatureEngineer(dc_threshold=0.02)
    features = engineer.build_unified_features(
        prices,
        events_df=events_df,
        sentiment_df=sentiment_df,
        market_id=market_id,
    )
    labels = engineer.build_labels(prices, horizon=5, method="direction")

    logger.info(f"Features: {features.shape}")
    logger.info(f"Labels: {labels.value_counts().to_dict()}")

    # Baselines
    baselines = BaselineModels()
    baseline_results = baselines.run_all_baselines(prices, labels)
    baseline_results.to_csv(
        RESULTS_DIR / f"baselines_{market_name}.csv", index=False
    )

    # Ablation study
    model = HybridDCLLMPredictor()
    ablation = model.run_ablation_study(features, labels, "xgboost")
    ablation.to_csv(
        RESULTS_DIR / f"ablation_{market_name}.csv", index=False
    )

    # Model comparison
    comparison = model.model_comparison(features, labels)
    comparison.to_csv(
        RESULTS_DIR / f"model_comparison_{market_name}.csv", index=False
    )

    # Final predictions
    predictions = model.get_final_predictions(features, labels, "xgboost")
    predictions.to_csv(RESULTS_DIR / f"predictions_{market_name}.csv")

    return ablation, comparison


def main():
    parser = argparse.ArgumentParser(description="Run Full Analysis Pipeline")
    parser.add_argument(
        "--market",
        choices=["US", "India", "China", "all"],
        default="all",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM-dependent analysis (no API key needed)",
    )
    args = parser.parse_args()

    setup_logger("INFO")
    logger.info("Starting analysis pipeline...")

    # Load events from GDELT (real data, not hardcoded)
    fetcher = StockDataFetcher()
    events_tracker = ConflictEventTracker()
    events = events_tracker.get_combined_geopolitical_events()
    logger.info(f"Loaded {len(events)} geopolitical events from GDELT")

    # LLM sentiment analysis (skipped if --skip-llm or no API key)
    sentiment_df = run_llm_analysis(events, skip_llm=args.skip_llm)
    if not sentiment_df.empty:
        logger.info(f"LLM sentiment available: {len(sentiment_df)} days")
    else:
        logger.info("No LLM sentiment data (--skip-llm or no API key)")

    symbol_map = {"US": "^GSPC", "India": "^NSEI", "China": "^HSI"}
    markets = (
        [args.market] if args.market != "all" else ["US", "India", "China"]
    )

    all_prices = {}

    for market in markets:
        symbol = symbol_map[market]
        logger.info(f"\n{'='*60}")
        logger.info(f"ANALYZING: {market} ({symbol})")
        logger.info(f"{'='*60}")

        df = fetcher.fetch_symbol(symbol, ANALYSIS_START, ANALYSIS_END)
        if df.empty:
            logger.warning(f"No data for {market}, skipping")
            continue

        prices = df["close"]
        all_prices[market] = prices

        # Get market-specific events
        market_events = events[
            events["markets_affected"].apply(
                lambda x, m=market: m in str(x)
            )
        ] if not events.empty else pd.DataFrame()

        # Run analyses
        run_dc_analysis(prices, market)
        run_event_study(prices, market_events, market)
        run_dc_event_correlation(prices, market_events, market)
        run_model_training(
            prices, market_events, market,
            sentiment_df=sentiment_df if not sentiment_df.empty else None,
        )

    # Cross-market analysis (requires all 3 markets)
    if args.market == "all" and len(all_prices) >= 2:
        logger.info("\n=== Cross-Market Analysis ===")

        # Granger causality
        returns = {}
        for market, prices in all_prices.items():
            ret = compute_returns(prices)
            ret.index = ret.index.normalize()
            returns[market] = ret

        gc = GrangerCausalityAnalyzer()
        gc_results = gc.cross_market_granger(returns)
        gc_results.to_csv(
            RESULTS_DIR / "granger_causality.csv", index=False
        )
        logger.info(f"Granger causality:\n{gc_results}")

        # Cross-market contagion
        run_cross_market_contagion(all_prices, events)

    logger.info("\nAnalysis pipeline complete! Results saved to results/")


if __name__ == "__main__":
    main()
