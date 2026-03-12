"""
Full analysis pipeline script.

Runs the complete analysis workflow:
1. DC analysis across all markets
2. Event study analysis
3. Cross-market contagion analysis
4. Feature engineering and model training
5. Ablation study

Usage:
    python scripts/run_analysis.py
    python scripts/run_analysis.py --market US
    python scripts/run_analysis.py --skip-llm  (run without LLM calls)
"""

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
from src.directional_changes.dc_algorithm import DirectionalChangeDetector, MultiThresholdDC
from src.directional_changes.intrinsic_time import IntrinsicTimeAnalyzer
from src.analysis.event_study import EventStudyAnalyzer
from src.analysis.dc_event_correlation import DCEventCorrelator
from src.analysis.cross_market_contagion import CrossMarketContagionAnalyzer
from src.analysis.granger_causality import GrangerCausalityAnalyzer
from src.models.feature_engineering import FeatureEngineer
from src.models.hybrid_model import HybridDCLLMPredictor
from src.models.baselines import BaselineModels
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
    bursts.to_csv(RESULTS_DIR / f"activity_bursts_{market_name}.csv", index=False)
    logger.info(f"Activity bursts: {len(bursts)}")

    return dc_df


def run_event_study(prices: pd.Series, events_df: pd.DataFrame, market_name: str):
    """Run event study analysis."""
    logger.info(f"=== Event Study: {market_name} ===")

    analyzer = EventStudyAnalyzer()
    event_dates = events_df["date"].astype(str).tolist()

    # Individual event results
    results = analyzer.multi_event_study(prices, event_dates)
    results.to_csv(RESULTS_DIR / f"event_study_{market_name}.csv", index=False)

    # Average CAR
    acar = analyzer.average_car(prices, event_dates)
    if "error" not in acar:
        logger.info(
            f"ACAR: {acar['acar'].iloc[-1]:.6f}, "
            f"n={acar['n_events']}, "
            f"significant events={results['significant'].sum()}"
        )

    return results


def run_dc_event_correlation(prices: pd.Series, events_df: pd.DataFrame, market_name: str):
    """Test DC-geopolitical event correlation."""
    logger.info(f"=== DC-Event Correlation: {market_name} ===")

    correlator = DCEventCorrelator(0.02)
    event_dates = events_df["date"].astype(str).tolist()

    coincidence = correlator.temporal_coincidence(prices, event_dates)
    logger.info(f"Enrichment ratio: {coincidence['enrichment_ratio']:.2f}x (p={coincidence['p_value']:.4f})")

    magnitude = correlator.dc_magnitude_around_events(prices, event_dates)
    logger.info(f"Near-event avg magnitude: {magnitude['near_event_avg_magnitude']:.6f}")
    logger.info(f"Normal avg magnitude: {magnitude['normal_avg_magnitude']:.6f}")

    return {"coincidence": coincidence, "magnitude": magnitude}


def run_model_training(prices: pd.Series, events_df: pd.DataFrame, market_name: str):
    """Train hybrid model and run ablation study."""
    logger.info(f"=== Model Training: {market_name} ===")

    engineer = FeatureEngineer(dc_threshold=0.02)
    features = engineer.build_unified_features(prices, events_df=events_df)
    labels = engineer.build_labels(prices, horizon=5, method="direction")

    logger.info(f"Features: {features.shape}")
    logger.info(f"Labels: {labels.value_counts().to_dict()}")

    # Baselines
    baselines = BaselineModels()
    baseline_results = baselines.run_all_baselines(prices, labels)
    baseline_results.to_csv(RESULTS_DIR / f"baselines_{market_name}.csv", index=False)

    # Ablation study
    model = HybridDCLLMPredictor()
    ablation = model.run_ablation_study(features, labels, "xgboost")
    ablation.to_csv(RESULTS_DIR / f"ablation_{market_name}.csv", index=False)

    # Model comparison
    comparison = model.model_comparison(features, labels)
    comparison.to_csv(RESULTS_DIR / f"model_comparison_{market_name}.csv", index=False)

    # Final predictions
    predictions = model.get_final_predictions(features, labels, "xgboost")
    predictions.to_csv(RESULTS_DIR / f"predictions_{market_name}.csv")

    return ablation, comparison


def main():
    parser = argparse.ArgumentParser(description="Run Full Analysis Pipeline")
    parser.add_argument("--market", choices=["US", "India", "China", "all"], default="all")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM-dependent analysis")
    args = parser.parse_args()

    setup_logger("INFO")
    logger.info("Starting analysis pipeline...")

    fetcher = StockDataFetcher()
    events_tracker = ConflictEventTracker()
    events = events_tracker.get_combined_geopolitical_events()

    symbol_map = {"US": "^GSPC", "India": "^NSEI", "China": "^HSI"}
    markets = [args.market] if args.market != "all" else ["US", "India", "China"]

    for market in markets:
        symbol = symbol_map[market]
        logger.info(f"\n{'='*60}")
        logger.info(f"ANALYZING: {market} ({symbol})")
        logger.info(f"{'='*60}")

        df = fetcher.fetch_symbol(symbol, "2015-01-01", "2025-12-31")
        if df.empty:
            logger.warning(f"No data for {market}, skipping")
            continue

        prices = df["close"]

        # Get market-specific events
        market_events = events[
            events["markets_affected"].apply(
                lambda x: market in str(x)
            )
        ]

        # Run analyses
        run_dc_analysis(prices, market)
        run_event_study(prices, market_events, market)
        run_dc_event_correlation(prices, market_events, market)
        run_model_training(prices, market_events, market)

    # Cross-market analysis
    if args.market == "all":
        logger.info("\n=== Cross-Market Analysis ===")
        returns = {}
        for market, symbol in symbol_map.items():
            df = fetcher.fetch_symbol(symbol)
            if not df.empty:
                ret = compute_returns(df["close"])
                ret.index = ret.index.normalize()
                returns[market] = ret

        if len(returns) >= 2:
            gc = GrangerCausalityAnalyzer()
            gc_results = gc.cross_market_granger(returns)
            gc_results.to_csv(RESULTS_DIR / "granger_causality.csv", index=False)
            logger.info(f"Granger causality:\n{gc_results}")

    logger.info("\nAnalysis pipeline complete! Results saved to results/")


if __name__ == "__main__":
    main()
