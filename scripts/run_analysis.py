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

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.logger import setup_logger
from src.utils.config import DATA_DIR
from src.data_collection.stock_fetcher import StockDataFetcher
from src.data_collection.conflict_tracker import ConflictEventTracker
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


# Sector ETFs for vulnerability analysis
SECTOR_ETFS = {
    "Technology": "XLK",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Consumer_Discretionary": "XLY",
    "Consumer_Staples": "XLP",
    "Defense": "ITA",
    "Utilities": "XLU",
}


def run_sector_analysis(results_dir: Path):
    """
    Run sector vulnerability analysis independently.

    Downloads sector ETF prices, loads existing geopolitical events,
    and computes sector-specific sensitivity to each event type.
    """
    from src.analysis.sector_vulnerability import SectorVulnerabilityAnalyzer

    fetcher = StockDataFetcher()
    events_tracker = ConflictEventTracker()
    events = events_tracker.get_combined_geopolitical_events()

    if events.empty:
        logger.warning("No events loaded — cannot run sector analysis")
        return

    logger.info(f"Loaded {len(events)} events for sector analysis")

    # Download sector ETF prices
    sector_prices = {}
    for sector_name, symbol in SECTOR_ETFS.items():
        df = fetcher.fetch_symbol(symbol, ANALYSIS_START, ANALYSIS_END)
        if not df.empty:
            prices = df["close"]
            if hasattr(prices.index, "tz") and prices.index.tz is not None:
                prices.index = prices.index.tz_localize(None)
            sector_prices[sector_name] = prices
            logger.info(f"  {sector_name} ({symbol}): {len(prices)} days")
        else:
            logger.warning(f"  {sector_name} ({symbol}): no data")

    if len(sector_prices) < 3:
        logger.warning("Too few sectors loaded, skipping sector analysis")
        return

    analyzer = SectorVulnerabilityAnalyzer()

    # Deduplicate: keep only the highest-severity event per day per type
    # Without this, 1.27M events = 12M iterations = hours of runtime
    events_deduped = (
        events.sort_values("severity", ascending=False)
        .groupby([events["date"].dt.date, "event_type"])
        .first()
        .reset_index()
    )
    logger.info(f"Deduplicated to {len(events_deduped)} unique date-type pairs (from {len(events)} raw events)")
    event_dates = pd.to_datetime(events_deduped["date"]).tolist()
    event_types = events_deduped["event_type"].tolist() if "event_type" in events_deduped.columns else ["unknown"] * len(events_deduped)

    # 1. Sector sensitivity matrix
    logger.info("Computing sector sensitivity to event types...")
    sensitivity = analyzer.compute_sector_sensitivity(
        sector_prices, event_dates, event_types, window_days=5
    )
    sensitivity.to_csv(results_dir / "sector_sensitivity.csv", index=False)
    logger.info(f"Sector sensitivity: {len(sensitivity)} sector-event combinations")

    # 2. Vulnerability heatmap data (pivot from sensitivity — no recomputation)
    if not sensitivity.empty:
        heatmap = sensitivity.pivot_table(
            index="sector", columns="event_type",
            values="avg_abnormal_return", aggfunc="mean",
        )
        heatmap.to_csv(results_dir / "sector_vulnerability_heatmap.csv")
        logger.info(f"Vulnerability heatmap:\n{heatmap.round(4)}")
    else:
        heatmap = pd.DataFrame()

    # 3. Natural hedges
    logger.info("Identifying natural hedges during geopolitical events...")
    hedge_results = analyzer.identify_hedges(
        sector_prices, event_dates, window_days=5
    )
    if hedge_results["hedges"]:
        hedge_df = pd.DataFrame(hedge_results["hedges"])
        hedge_df.to_csv(results_dir / "sector_hedges.csv", index=False)
        logger.info(f"Found {len(hedge_df)} natural hedge pairs")
        for h in hedge_results["hedges"][:5]:
            logger.info(f"  {h['sector_a']} ↔ {h['sector_b']}: corr={h['correlation']:.3f}")
    else:
        logger.info("No strong natural hedges found (all correlations > -0.3)")

    hedge_results["correlation_matrix"].to_csv(results_dir / "sector_event_correlations.csv")

    # 4. Regime analysis (high-severity events only)
    logger.info("Running sector regime analysis for high-severity events...")
    regime_df = analyzer.sector_regime_analysis(sector_prices, events)
    if not regime_df.empty:
        regime_df.to_csv(results_dir / "sector_regime_shifts.csv", index=False)
        n_shifted = regime_df["regime_changed"].sum()
        logger.info(f"Regime shifts detected: {n_shifted}/{len(regime_df)} sector-event pairs")
    else:
        logger.info("No high-severity events for regime analysis")

    # --- Save figures ---
    import plotly.io as pio
    from src.visualization.heatmaps import HeatmapVisualizer
    viz = HeatmapVisualizer()
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Figure 1: Sector vulnerability heatmap
    if not heatmap.empty:
        fig = viz.plot_sector_vulnerability_heatmap(heatmap)
        try:
            pio.write_image(fig, str(fig_dir / "sector_vulnerability_heatmap.png"), scale=2)
            logger.info(f"  Saved: sector_vulnerability_heatmap.png")
        except Exception as e:
            fig.write_html(str(fig_dir / "sector_vulnerability_heatmap.html"))
            logger.info(f"  Saved as HTML (install kaleido for PNG): sector_vulnerability_heatmap.html")

    # Figure 2: Sector correlation during events
    corr_matrix = hedge_results["correlation_matrix"]
    fig = viz.plot_cross_market_correlation_heatmap(
        corr_matrix, title="Sector Correlation During Geopolitical Events"
    )
    try:
        pio.write_image(fig, str(fig_dir / "sector_event_correlations.png"), scale=2)
        logger.info(f"  Saved: sector_event_correlations.png")
    except Exception as e:
        fig.write_html(str(fig_dir / "sector_event_correlations.html"))
        logger.info(f"  Saved as HTML: sector_event_correlations.html")

    # Figure 3: Regime shift bar chart (% of sectors that shifted per event)
    if not regime_df.empty:
        import plotly.graph_objects as go_fig
        shift_by_sector = regime_df.groupby("sector")["regime_changed"].mean() * 100
        shift_by_sector = shift_by_sector.sort_values(ascending=True)

        fig = go_fig.Figure(go_fig.Bar(
            x=shift_by_sector.values,
            y=shift_by_sector.index,
            orientation="h",
            marker_color=["#e53935" if v > 50 else "#1976d2" for v in shift_by_sector.values],
        ))
        fig.update_layout(
            title="Sector Regime Shift Rate After High-Severity Events",
            xaxis_title="% of Events Causing Regime Shift",
            template="plotly_white",
            height=500, width=800,
        )
        try:
            pio.write_image(fig, str(fig_dir / "sector_regime_shifts.png"), scale=2)
            logger.info(f"  Saved: sector_regime_shifts.png")
        except Exception as e:
            fig.write_html(str(fig_dir / "sector_regime_shifts.html"))
            logger.info(f"  Saved as HTML: sector_regime_shifts.html")

    logger.info("Sector vulnerability analysis complete!")


def _generate_sector_figures(results_dir: Path):
    """Generate sector figures from pre-existing CSVs. No data fetching needed."""
    import plotly.io as pio
    import plotly.graph_objects as go_fig
    from src.visualization.heatmaps import HeatmapVisualizer

    viz = HeatmapVisualizer()
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    saved = []

    # 1. Vulnerability heatmap
    heatmap_path = results_dir / "sector_vulnerability_heatmap.csv"
    if heatmap_path.exists():
        heatmap = pd.read_csv(heatmap_path, index_col=0)
        fig = viz.plot_sector_vulnerability_heatmap(heatmap)
        try:
            pio.write_image(fig, str(fig_dir / "sector_vulnerability_heatmap.png"), scale=2)
            saved.append("sector_vulnerability_heatmap.png")
        except Exception:
            fig.write_html(str(fig_dir / "sector_vulnerability_heatmap.html"))
            saved.append("sector_vulnerability_heatmap.html")
    else:
        logger.warning("sector_vulnerability_heatmap.csv not found — run --sectors first")

    # 2. Sector correlation during events
    corr_path = results_dir / "sector_event_correlations.csv"
    if corr_path.exists():
        corr_matrix = pd.read_csv(corr_path, index_col=0)
        fig = viz.plot_cross_market_correlation_heatmap(
            corr_matrix, title="Sector Correlation During Geopolitical Events"
        )
        try:
            pio.write_image(fig, str(fig_dir / "sector_event_correlations.png"), scale=2)
            saved.append("sector_event_correlations.png")
        except Exception:
            fig.write_html(str(fig_dir / "sector_event_correlations.html"))
            saved.append("sector_event_correlations.html")
    else:
        logger.warning("sector_event_correlations.csv not found — run --sectors first")

    # 3. Regime shift bar chart
    regime_path = results_dir / "sector_regime_shifts.csv"
    if regime_path.exists():
        regime_df = pd.read_csv(regime_path)
        shift_by_sector = regime_df.groupby("sector")["regime_changed"].mean() * 100
        shift_by_sector = shift_by_sector.sort_values(ascending=True)

        fig = go_fig.Figure(go_fig.Bar(
            x=shift_by_sector.values,
            y=shift_by_sector.index,
            orientation="h",
            marker_color=["#e53935" if v > 50 else "#1976d2" for v in shift_by_sector.values],
        ))
        fig.update_layout(
            title="Sector Regime Shift Rate After High-Severity Events",
            xaxis_title="% of Events Causing Regime Shift",
            template="plotly_white",
            height=500, width=800,
        )
        try:
            pio.write_image(fig, str(fig_dir / "sector_regime_shifts.png"), scale=2)
            saved.append("sector_regime_shifts.png")
        except Exception:
            fig.write_html(str(fig_dir / "sector_regime_shifts.html"))
            saved.append("sector_regime_shifts.html")
    else:
        logger.warning("sector_regime_shifts.csv not found — run --sectors first")

    for name in saved:
        logger.info(f"  → {name}")


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
    prices: pd.Series,
    events_df: pd.DataFrame,
    market_name: str,
    market_returns: pd.Series | None = None,
):
    """Run event study analysis."""
    logger.info(f"=== Event Study: {market_name} ===")

    analyzer = EventStudyAnalyzer()
    event_dates = sorted(events_df["date"].astype(str).unique().tolist())

    if not event_dates:
        logger.warning(f"No events for {market_name}, skipping event study")
        return pd.DataFrame()

    # Individual event results
    results = analyzer.multi_event_study(prices, event_dates, market_returns)
    results.to_csv(
        RESULTS_DIR / f"event_study_{market_name}.csv", index=False
    )

    # Average CAR
    acar = analyzer.average_car(prices, event_dates, market_returns)
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
    event_dates = sorted(events_df["date"].astype(str).unique().tolist())

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

    # Multi-event contagion — deduplicate to highest-severity per day
    if not events_df.empty:
        high_sev = events_df[events_df["severity"] >= 7].copy()
        if not high_sev.empty:
            high_sev["_date_key"] = pd.to_datetime(high_sev["date"]).dt.normalize()
            high_sev = (
                high_sev.sort_values("severity", ascending=False)
                .drop_duplicates(subset=["_date_key"], keep="first")
                .drop(columns=["_date_key"])
            )
            logger.info(f"Contagion: {len(high_sev)} high-severity events (from {len(events_df)} total)")
        else:
            high_sev = events_df.head(0)  # empty with same columns
        contagion_results = contagion.multi_event_contagion(
            market_prices, high_sev
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
                sorted(tariff_events["date"].astype(str).unique().tolist()),
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
    labels = engineer.build_labels(prices, horizon=5, method="direction_vol")

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


# ======================================================================
# Research-grade extensions (additive — called after existing analyses)
# ======================================================================


def run_extended_event_study(
    prices: pd.Series,
    events_df: pd.DataFrame,
    market_name: str,
    market_returns: pd.Series | None = None,
):
    """Extended event study: BH correction, Cohen's d, placebo test, window sensitivity."""
    logger.info(f"=== Extended Event Study: {market_name} ===")

    analyzer = EventStudyAnalyzer()
    event_dates = sorted(events_df["date"].astype(str).unique().tolist())

    if not event_dates:
        logger.warning(f"No events for {market_name}, skipping extended event study")
        return

    # BH-corrected multi-event study with effect sizes
    extended = analyzer.multi_event_study_extended(prices, event_dates, market_returns)
    if not extended.empty:
        extended.to_csv(
            RESULTS_DIR / f"event_study_extended_{market_name}.csv", index=False
        )

    # Placebo test
    placebo = analyzer.placebo_test(prices, event_dates, n_placebos=500)
    if "error" not in placebo:
        pd.DataFrame([placebo]).to_csv(
            RESULTS_DIR / f"placebo_test_{market_name}.csv", index=False
        )

    # Window sensitivity
    window_sens = analyzer.window_sensitivity(prices, event_dates, market_returns=market_returns)
    if not window_sens.empty:
        window_sens.to_csv(
            RESULTS_DIR / f"event_window_sensitivity_{market_name}.csv", index=False
        )


def run_extended_model_training(
    prices: pd.Series,
    events_df: pd.DataFrame,
    market_name: str,
    sentiment_df: pd.DataFrame | None = None,
):
    """Extended model training: CIs, ablation, DM test, temporal holdout,
    rolling eval, calibration, learning curves."""
    logger.info(f"=== Extended Model Training: {market_name} ===")

    from src.models.feature_engineering import FeatureEngineer

    market_id = market_name.lower()
    engineer = FeatureEngineer(dc_threshold=0.02)
    features = engineer.build_unified_features(
        prices,
        events_df=events_df,
        sentiment_df=sentiment_df,
        market_id=market_id,
    )
    labels = engineer.build_labels(prices, horizon=5, method="direction_vol")

    model = HybridDCLLMPredictor()

    # Extended ablation with CIs
    ablation_ext = model.run_ablation_study_extended(features, labels, "xgboost")
    ablation_ext.to_csv(
        RESULTS_DIR / f"ablation_extended_{market_name}.csv", index=False
    )

    # Extended model comparison with DM test
    comparison_ext = model.model_comparison_extended(features, labels)
    comparison_ext.to_csv(
        RESULTS_DIR / f"model_comparison_extended_{market_name}.csv", index=False
    )

    # Temporal holdout (train ≤ 2022, test 2023+)
    holdout = model.temporal_holdout_evaluation(features, labels)
    pd.DataFrame([{k: v for k, v in holdout.items() if k != "classification_report"}]).to_csv(
        RESULTS_DIR / f"temporal_holdout_{market_name}.csv", index=False
    )

    # Rolling window evaluation
    rolling = model.rolling_evaluation(features, labels)
    if not rolling.empty:
        rolling.to_csv(
            RESULTS_DIR / f"rolling_eval_{market_name}.csv", index=False
        )

    # Calibration analysis
    cal = model.calibration_analysis(features, labels)
    if "error" not in cal:
        pd.DataFrame([{"ece": cal["ece"], "n_test": cal["n_test"]}]).to_csv(
            RESULTS_DIR / f"calibration_{market_name}.csv", index=False
        )

    # Learning curves
    lc = model.learning_curves(features, labels)
    if not lc.empty:
        lc.to_csv(
            RESULTS_DIR / f"learning_curves_{market_name}.csv", index=False
        )


def run_dc_sensitivity(
    prices: pd.Series,
    events_df: pd.DataFrame,
    market_name: str,
    sentiment_df: pd.DataFrame | None = None,
):
    """Test DC features at multiple thresholds to show robustness."""
    logger.info(f"=== DC Threshold Sensitivity: {market_name} ===")

    from src.models.feature_engineering import FeatureEngineer

    market_id = market_name.lower()
    results = []

    for theta in [0.01, 0.02, 0.05]:
        engineer = FeatureEngineer(dc_threshold=theta)
        features = engineer.build_unified_features(
            prices,
            events_df=events_df,
            sentiment_df=sentiment_df,
            market_id=market_id,
        )
        labels = engineer.build_labels(prices, horizon=5, method="direction_vol")

        model = HybridDCLLMPredictor()
        model.define_feature_groups(features)
        dc_cols = model.feature_groups.get("dc_only", [])
        valid_cols = [c for c in dc_cols if c in features.columns]

        if not valid_cols:
            continue

        result = model.train_and_evaluate_extended(features, labels, valid_cols, "xgboost")
        if "error" not in result:
            results.append({
                "theta": theta,
                "n_features": result["n_features"],
                "mean_accuracy": result["mean_accuracy"],
                "ci_lower": result["ci_lower_accuracy"],
                "ci_upper": result["ci_upper_accuracy"],
                "mean_f1": result["mean_f1"],
            })

    df = pd.DataFrame(results)
    if not df.empty:
        df.to_csv(RESULTS_DIR / f"dc_sensitivity_{market_name}.csv", index=False)
        logger.info(f"DC sensitivity:\n{df.to_string()}")


def run_regime_analysis(
    prices: pd.Series,
    events_df: pd.DataFrame,
    market_name: str,
    sentiment_df: pd.DataFrame | None = None,
):
    """Subsample analysis: pre-COVID vs post-COVID, tariff vs conflict."""
    logger.info(f"=== Regime Analysis: {market_name} ===")

    from src.models.feature_engineering import FeatureEngineer

    market_id = market_name.lower()
    engineer = FeatureEngineer(dc_threshold=0.02)
    features = engineer.build_unified_features(
        prices,
        events_df=events_df,
        sentiment_df=sentiment_df,
        market_id=market_id,
    )
    labels = engineer.build_labels(prices, horizon=5, method="direction_vol")

    results = []
    cutoff = pd.Timestamp("2020-01-01")

    for regime_name, mask in [
        ("pre_covid", features.index <= cutoff),
        ("post_covid", features.index > cutoff),
        ("full_period", pd.Series(True, index=features.index)),
    ]:
        feat_sub = features[mask]
        lab_sub = labels.loc[feat_sub.index.intersection(labels.index)]

        if len(feat_sub) < 100:
            continue

        model = HybridDCLLMPredictor()
        result = model.train_and_evaluate_extended(feat_sub, lab_sub, model_name="xgboost")
        if "error" not in result:
            results.append({
                "regime": regime_name,
                "n_samples": len(feat_sub),
                "mean_accuracy": result["mean_accuracy"],
                "ci_lower": result["ci_lower_accuracy"],
                "ci_upper": result["ci_upper_accuracy"],
                "mean_f1": result["mean_f1"],
            })

    # Also split event study by event type
    if not events_df.empty and "event_type" in events_df.columns:
        analyzer = EventStudyAnalyzer()
        for etype in ["tariff", "conflict"]:
            sub_events = events_df[events_df["event_type"] == etype]
            if len(sub_events) < 5:
                continue
            event_dates = sorted(sub_events["date"].astype(str).unique().tolist())
            acar = analyzer.average_car(prices, event_dates)
            if "error" not in acar:
                results.append({
                    "regime": f"event_type_{etype}",
                    "n_samples": acar["n_events"],
                    "mean_accuracy": np.nan,
                    "acar": float(acar["acar"].iloc[-1]),
                    "ci_lower": acar.get("acar_ci_lower", np.nan),
                    "ci_upper": acar.get("acar_ci_upper", np.nan),
                    "mean_f1": np.nan,
                })

    df = pd.DataFrame(results)
    if not df.empty:
        df.to_csv(RESULTS_DIR / f"regime_analysis_{market_name}.csv", index=False)
        logger.info(f"Regime analysis:\n{df.to_string()}")


def run_multi_horizon_analysis(
    prices: pd.Series,
    events_df: pd.DataFrame,
    market_name: str,
    sentiment_df: pd.DataFrame | None = None,
):
    """Test model vs persistence across multiple prediction horizons.

    Uses volatility-scaled thresholds so each horizon gets an
    appropriately sized neutral zone. This reveals whether model
    advantage emerges at longer horizons where persistence is weaker.
    """
    logger.info(f"=== Multi-Horizon Analysis: {market_name} ===")
    from src.models.feature_engineering import FeatureEngineer

    market_id = market_name.lower()
    results = []

    for horizon in [1, 3, 5, 10]:
        logger.info(f"  Horizon {horizon}d ...")
        engineer = FeatureEngineer(dc_threshold=0.02)
        features = engineer.build_unified_features(
            prices,
            events_df=events_df,
            sentiment_df=sentiment_df,
            market_id=market_id,
        )
        labels = engineer.build_labels(prices, horizon=horizon, method="direction_vol")

        # Align
        common = features.index.intersection(labels.dropna().index)
        if len(common) < 100:
            logger.warning(f"  Only {len(common)} samples for {horizon}d, skipping")
            continue

        feat = features.loc[common]
        lab = labels.loc[common]

        # Persistence baseline
        baselines = BaselineModels()
        persistence = baselines.persistence_baseline(lab)

        # Best model (xgboost)
        model = HybridDCLLMPredictor()
        model_result = model.train_and_evaluate(
            feat, lab, feat.columns.tolist(), "xgboost"
        )
        if "error" in model_result:
            logger.warning(f"  Model error at {horizon}d: {model_result['error']}")
            continue

        lift = model_result["mean_accuracy"] - persistence["mean_accuracy"]
        results.append({
            "horizon": horizon,
            "persistence_accuracy": persistence["mean_accuracy"],
            "persistence_f1": persistence.get("mean_f1", np.nan),
            "model_accuracy": model_result["mean_accuracy"],
            "model_f1": model_result["mean_f1"],
            "lift_accuracy": lift,
            "lift_f1": model_result["mean_f1"] - persistence.get("mean_f1", 0),
            "n_samples": len(lab),
            "n_up": int((lab == 2).sum()),
            "n_neutral": int((lab == 1).sum()),
            "n_down": int((lab == 0).sum()),
        })
        logger.info(
            f"  {horizon}d: model={model_result['mean_accuracy']:.4f} "
            f"persistence={persistence['mean_accuracy']:.4f} lift={lift:+.4f}"
        )

    df = pd.DataFrame(results)
    if not df.empty:
        df.to_csv(RESULTS_DIR / f"multi_horizon_{market_name}.csv", index=False)
        logger.info(f"Multi-horizon results:\n{df.to_string()}")

    return df


def save_key_figures(all_prices: dict, results_dir: Path):
    """Save publication-quality figures for key findings.

    Only saves visualizations that convey meaningful research insights —
    not decorative charts.
    """
    import plotly.graph_objects as go
    import plotly.io as pio

    fig_dir = results_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    saved = []

    # --- 1. Multi-horizon: model vs persistence across horizons ---
    for market in all_prices:
        mh_path = results_dir / f"multi_horizon_{market}.csv"
        if not mh_path.exists():
            continue
        mh = pd.read_csv(mh_path)
        if mh.empty:
            continue

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"{h}d" for h in mh["horizon"]],
            y=mh["persistence_accuracy"],
            name="Persistence Baseline",
            marker_color="#90a4ae",
        ))
        fig.add_trace(go.Bar(
            x=[f"{h}d" for h in mh["horizon"]],
            y=mh["model_accuracy"],
            name="Hybrid DC-LLM Model",
            marker_color="#1976d2",
        ))
        # Add lift annotation
        for i, row in mh.iterrows():
            lift = row["lift_accuracy"]
            color = "#4caf50" if lift > 0 else "#f44336"
            fig.add_annotation(
                x=f"{int(row['horizon'])}d",
                y=max(row["model_accuracy"], row["persistence_accuracy"]) + 0.01,
                text=f"{lift:+.2%}",
                showarrow=False,
                font=dict(color=color, size=12, family="Arial Black"),
            )

        fig.update_layout(
            title=f"Model vs Persistence Across Horizons — {market}",
            yaxis_title="Accuracy",
            xaxis_title="Prediction Horizon",
            barmode="group",
            template="plotly_white",
            height=500, width=800,
            legend=dict(x=0.02, y=0.98),
        )
        path = fig_dir / f"multi_horizon_{market}.png"
        pio.write_image(fig, str(path), scale=2)
        saved.append(path.name)

    # --- 2. Label distribution comparison: fixed vs vol-scaled ---
    for market, prices in all_prices.items():
        from src.models.feature_engineering import FeatureEngineer
        engineer = FeatureEngineer(dc_threshold=0.02)
        fixed_labels = engineer.build_labels(prices, horizon=5, method="direction")
        vol_labels = engineer.build_labels(prices, horizon=5, method="direction_vol")

        fig = go.Figure()
        for labels_data, name, colors in [
            (fixed_labels, "Fixed ±0.5%", ["#ef5350", "#78909c", "#66bb6a"]),
            (vol_labels, "Vol-Scaled", ["#e53935", "#546e7a", "#43a047"]),
        ]:
            counts = labels_data.value_counts().sort_index()
            total = counts.sum()
            fig.add_trace(go.Bar(
                x=["Down", "Neutral", "Up"],
                y=[counts.get(0, 0) / total, counts.get(1, 0) / total, counts.get(2, 0) / total],
                name=name,
                text=[f"{counts.get(c, 0)}" for c in [0, 1, 2]],
                textposition="outside",
            ))

        fig.update_layout(
            title=f"Label Distribution: Fixed vs Volatility-Scaled — {market}",
            yaxis_title="Proportion",
            xaxis_title="Label Class",
            barmode="group",
            template="plotly_white",
            height=450, width=700,
            yaxis=dict(tickformat=".0%"),
        )
        path = fig_dir / f"label_distribution_{market}.png"
        pio.write_image(fig, str(path), scale=2)
        saved.append(path.name)

    # --- 3. Ablation study: feature group contribution ---
    for market in all_prices:
        abl_path = results_dir / f"ablation_{market}.csv"
        if not abl_path.exists():
            continue
        abl = pd.read_csv(abl_path)
        if abl.empty or "mean_accuracy" not in abl.columns:
            continue

        abl_sorted = abl.sort_values("mean_accuracy", ascending=True)
        colors = ["#1976d2" if "all" in g else "#546e7a"
                  for g in abl_sorted["feature_group"]]

        fig = go.Figure(go.Bar(
            x=abl_sorted["mean_accuracy"],
            y=abl_sorted["feature_group"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.3f}" for v in abl_sorted["mean_accuracy"]],
            textposition="outside",
        ))
        fig.update_layout(
            title=f"Feature Group Ablation Study — {market}",
            xaxis_title="Mean Accuracy (CV)",
            template="plotly_white",
            height=400, width=700,
            margin=dict(l=140),
        )
        path = fig_dir / f"ablation_{market}.png"
        pio.write_image(fig, str(path), scale=2)
        saved.append(path.name)

    # --- 4. Granger causality heatmap ---
    gc_path = results_dir / "granger_causality.csv"
    if gc_path.exists():
        gc = pd.read_csv(gc_path)
        if not gc.empty and "p_value" in gc.columns:
            markets_list = sorted(set(gc["source"].tolist() + gc["target"].tolist()))
            n = len(markets_list)
            p_matrix = pd.DataFrame(1.0, index=markets_list, columns=markets_list)
            for _, row in gc.iterrows():
                p_matrix.loc[row["source"], row["target"]] = row["p_value"]

            # -log10(p) for visual clarity (higher = more significant)
            import plotly.figure_factory as ff
            log_p = -np.log10(p_matrix.values.clip(min=1e-50))
            np.fill_diagonal(log_p, 0)

            fig = go.Figure(go.Heatmap(
                z=log_p,
                x=markets_list,
                y=markets_list,
                colorscale="YlOrRd",
                text=[[f"p={p_matrix.iloc[i, j]:.2e}" for j in range(n)] for i in range(n)],
                texttemplate="%{text}",
                colorbar=dict(title="-log₁₀(p)"),
            ))
            fig.update_layout(
                title="Granger Causality: Cross-Market Lead-Lag (p-values)",
                xaxis_title="Target Market (affected)",
                yaxis_title="Source Market (leads)",
                template="plotly_white",
                height=500, width=600,
            )
            path = fig_dir / "granger_causality_heatmap.png"
            pio.write_image(fig, str(path), scale=2)
            saved.append(path.name)

    # --- 5. Rolling evaluation: performance over time ---
    for market in all_prices:
        roll_path = results_dir / f"rolling_eval_{market}.csv"
        if not roll_path.exists():
            continue
        roll = pd.read_csv(roll_path)
        if roll.empty or "accuracy" not in roll.columns:
            continue

        fig = go.Figure()
        if "test_start" in roll.columns:
            x_vals = pd.to_datetime(roll["test_start"])
        else:
            x_vals = list(range(len(roll)))

        fig.add_trace(go.Scatter(
            x=x_vals, y=roll["accuracy"],
            mode="lines+markers",
            name="Accuracy",
            line=dict(color="#1976d2", width=2),
        ))
        if "f1" in roll.columns:
            fig.add_trace(go.Scatter(
                x=x_vals, y=roll["f1"],
                mode="lines+markers",
                name="F1",
                line=dict(color="#26a69a", width=2),
            ))

        fig.update_layout(
            title=f"Rolling Window Model Performance — {market}",
            yaxis_title="Score",
            xaxis_title="Test Window Start",
            template="plotly_white",
            height=400, width=800,
        )
        path = fig_dir / f"rolling_performance_{market}.png"
        pio.write_image(fig, str(path), scale=2)
        saved.append(path.name)

    # --- 6. Model comparison with baselines ---
    for market in all_prices:
        base_path = results_dir / f"baselines_{market}.csv"
        comp_path = results_dir / f"model_comparison_{market}.csv"
        if not base_path.exists() or not comp_path.exists():
            continue
        baselines_df = pd.read_csv(base_path)
        comp_df = pd.read_csv(comp_path)
        if baselines_df.empty or comp_df.empty:
            continue

        # Combine into one chart
        combined = pd.concat([
            baselines_df[["model", "mean_accuracy"]],
            comp_df[["model", "mean_accuracy"]],
        ], ignore_index=True).sort_values("mean_accuracy", ascending=True)

        colors = []
        for m in combined["model"]:
            if m == "persistence":
                colors.append("#f44336")  # red for key baseline
            elif m in ["xgboost", "lightgbm", "random_forest", "gradient_boosting"]:
                colors.append("#1976d2")  # blue for ML models
            else:
                colors.append("#90a4ae")  # grey for other baselines

        fig = go.Figure(go.Bar(
            x=combined["mean_accuracy"],
            y=combined["model"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.3f}" for v in combined["mean_accuracy"]],
            textposition="outside",
        ))
        fig.update_layout(
            title=f"All Models vs Baselines — {market}",
            xaxis_title="Mean Accuracy",
            template="plotly_white",
            height=450, width=750,
            margin=dict(l=150),
        )
        path = fig_dir / f"models_vs_baselines_{market}.png"
        pio.write_image(fig, str(path), scale=2)
        saved.append(path.name)

    # --- 7. DC Scaling Laws (log-log verification) ---
    for market in all_prices:
        sl_path = results_dir / f"scaling_laws_{market}.csv"
        if not sl_path.exists():
            continue
        sl = pd.read_csv(sl_path)
        if sl.empty or "threshold" not in sl.columns:
            continue

        from plotly.subplots import make_subplots
        metrics = [
            ("num_dc_events", "# DC Events", "#1976d2"),
            ("avg_overshoot_ratio", "Avg Overshoot Ratio", "#e65100"),
            ("avg_dc_duration_days", "Avg DC Duration (days)", "#2e7d32"),
        ]
        available = [(col, title, c) for col, title, c in metrics if col in sl.columns]
        if not available:
            continue

        fig = make_subplots(
            rows=1, cols=len(available),
            subplot_titles=[t for _, t, _ in available],
        )
        for i, (col, title, color) in enumerate(available, 1):
            fig.add_trace(go.Scatter(
                x=np.log10(sl["threshold"]),
                y=np.log10(sl[col].clip(lower=1e-10)),
                mode="lines+markers",
                name=title,
                line=dict(color=color, width=2),
                marker=dict(size=8),
            ), row=1, col=i)
            fig.update_xaxes(title_text="log₁₀(θ)", row=1, col=i)
            fig.update_yaxes(title_text=f"log₁₀({col})", row=1, col=i)

        fig.update_layout(
            title=f"DC Scaling Laws Verification — {market}",
            template="plotly_white",
            height=400, width=300 * len(available),
            showlegend=False,
        )
        path = fig_dir / f"scaling_laws_{market}.png"
        pio.write_image(fig, str(path), scale=2)
        saved.append(path.name)

    # --- 8. Event Study: ACAR with confidence bands ---
    for market in all_prices:
        es_path = results_dir / f"event_study_extended_{market}.csv"
        if not es_path.exists():
            es_path = results_dir / f"event_study_{market}.csv"
        if not es_path.exists():
            continue
        es = pd.read_csv(es_path)
        if es.empty:
            continue

        # Try to build ACAR from individual events
        car_cols = [c for c in es.columns if c.startswith("car_day_") or c == "total_car"]
        if "total_car" in es.columns:
            fig = go.Figure()
            # Histogram of CARs
            fig.add_trace(go.Histogram(
                x=es["total_car"],
                nbinsx=30,
                name="CAR Distribution",
                marker_color="#1976d2",
                opacity=0.7,
            ))
            mean_car = es["total_car"].mean()
            fig.add_vline(x=mean_car, line_dash="dash", line_color="red",
                          annotation_text=f"Mean CAR: {mean_car:.4f}")
            fig.add_vline(x=0, line_dash="solid", line_color="grey")

            sig_count = es["significant"].sum() if "significant" in es.columns else "N/A"
            fig.update_layout(
                title=f"Event Study: Distribution of CARs — {market} (n={len(es)}, significant={sig_count})",
                xaxis_title="Cumulative Abnormal Return",
                yaxis_title="Count",
                template="plotly_white",
                height=450, width=700,
            )
            path = fig_dir / f"event_study_car_dist_{market}.png"
            pio.write_image(fig, str(path), scale=2)
            saved.append(path.name)

        # Effect sizes if available
        if "cohens_d" in es.columns:
            fig = go.Figure()
            d_vals = es["cohens_d"].dropna()
            fig.add_trace(go.Histogram(
                x=d_vals, nbinsx=25,
                marker_color="#e65100", opacity=0.7,
                name="Cohen's d",
            ))
            fig.add_vline(x=0.2, line_dash="dash", line_color="#4caf50",
                          annotation_text="Small (0.2)")
            fig.add_vline(x=0.5, line_dash="dash", line_color="#ff9800",
                          annotation_text="Medium (0.5)")
            fig.add_vline(x=0.8, line_dash="dash", line_color="#f44336",
                          annotation_text="Large (0.8)")
            fig.update_layout(
                title=f"Event Study: Effect Sizes (Cohen's d) — {market}",
                xaxis_title="Cohen's d",
                yaxis_title="Count",
                template="plotly_white",
                height=400, width=700,
            )
            path = fig_dir / f"effect_sizes_{market}.png"
            pio.write_image(fig, str(path), scale=2)
            saved.append(path.name)

    # --- 9. Placebo test: real vs placebo distribution ---
    for market in all_prices:
        pl_path = results_dir / f"placebo_test_{market}.csv"
        if not pl_path.exists():
            continue
        pl = pd.read_csv(pl_path)
        if pl.empty:
            continue

        if "placebo_cars" in pl.columns:
            placebo_vals = pl["placebo_cars"].dropna()
        elif "placebo_mean" in pl.columns and "placebo_std" in pl.columns:
            # Simulate distribution from summary stats
            pmean = pl["placebo_mean"].iloc[0]
            pstd = pl["placebo_std"].iloc[0]
            placebo_vals = np.random.normal(pmean, pstd, 500)
        else:
            continue

        real_acar = pl["real_acar"].iloc[0] if "real_acar" in pl.columns else None
        emp_p = pl["empirical_p_value"].iloc[0] if "empirical_p_value" in pl.columns else None

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=placebo_vals, nbinsx=30,
            marker_color="#90a4ae", opacity=0.7,
            name="Placebo CARs",
        ))
        if real_acar is not None:
            fig.add_vline(x=real_acar, line_dash="solid", line_color="#f44336",
                          annotation_text=f"Real ACAR: {real_acar:.4f}")

        title = f"Placebo Test — {market}"
        if emp_p is not None:
            title += f" (p={emp_p:.4f})"
        fig.update_layout(
            title=title,
            xaxis_title="Average CAR",
            yaxis_title="Count (500 placebos)",
            template="plotly_white",
            height=400, width=700,
        )
        path = fig_dir / f"placebo_test_{market}.png"
        pio.write_image(fig, str(path), scale=2)
        saved.append(path.name)

    # --- 10. DC Sensitivity: accuracy vs threshold ---
    for market in all_prices:
        dcs_path = results_dir / f"dc_sensitivity_{market}.csv"
        if not dcs_path.exists():
            continue
        dcs = pd.read_csv(dcs_path)
        if dcs.empty or "theta" not in dcs.columns:
            continue

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dcs["theta"], y=dcs["mean_accuracy"],
            mode="lines+markers",
            name="Accuracy",
            line=dict(color="#1976d2", width=3),
            marker=dict(size=10),
            error_y=dict(
                type="data",
                symmetric=False,
                array=(dcs["ci_upper"] - dcs["mean_accuracy"]).tolist(),
                arrayminus=(dcs["mean_accuracy"] - dcs["ci_lower"]).tolist(),
            ) if "ci_lower" in dcs.columns and "ci_upper" in dcs.columns else None,
        ))
        fig.update_layout(
            title=f"DC Feature Sensitivity to Threshold — {market}",
            xaxis_title="DC Threshold (θ)",
            yaxis_title="Accuracy (with 95% CI)",
            template="plotly_white",
            height=400, width=600,
            xaxis=dict(type="log"),
        )
        path = fig_dir / f"dc_sensitivity_{market}.png"
        pio.write_image(fig, str(path), scale=2)
        saved.append(path.name)

    # --- 11. Learning curves ---
    for market in all_prices:
        lc_path = results_dir / f"learning_curves_{market}.csv"
        if not lc_path.exists():
            continue
        lc = pd.read_csv(lc_path)
        if lc.empty or "accuracy" not in lc.columns:
            continue

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=lc["fraction"] if "fraction" in lc.columns else list(range(len(lc))),
            y=lc["accuracy"],
            mode="lines+markers",
            name="Accuracy",
            line=dict(color="#1976d2", width=3),
            marker=dict(size=10),
        ))
        if "f1" in lc.columns:
            fig.add_trace(go.Scatter(
                x=lc["fraction"] if "fraction" in lc.columns else list(range(len(lc))),
                y=lc["f1"],
                mode="lines+markers",
                name="F1",
                line=dict(color="#26a69a", width=3),
                marker=dict(size=10),
            ))
        fig.update_layout(
            title=f"Learning Curves — {market}",
            xaxis_title="Training Data Fraction",
            yaxis_title="Score",
            template="plotly_white",
            height=400, width=600,
            xaxis=dict(tickformat=".0%") if "fraction" in lc.columns else {},
        )
        path = fig_dir / f"learning_curves_{market}.png"
        pio.write_image(fig, str(path), scale=2)
        saved.append(path.name)

    # --- 12. Regime analysis: pre/post COVID + event types ---
    for market in all_prices:
        reg_path = results_dir / f"regime_analysis_{market}.csv"
        if not reg_path.exists():
            continue
        reg = pd.read_csv(reg_path)
        if reg.empty:
            continue

        # Split into period regimes and event regimes
        period_regimes = reg[~reg["regime"].str.startswith("event_type_")]
        event_regimes = reg[reg["regime"].str.startswith("event_type_")]

        if not period_regimes.empty and "mean_accuracy" in period_regimes.columns:
            fig = go.Figure()
            pr = period_regimes.dropna(subset=["mean_accuracy"])
            if not pr.empty:
                fig.add_trace(go.Bar(
                    x=pr["regime"],
                    y=pr["mean_accuracy"],
                    marker_color=["#1976d2", "#e65100", "#2e7d32"][:len(pr)],
                    text=[f"{v:.3f}" for v in pr["mean_accuracy"]],
                    textposition="outside",
                ))
                fig.update_layout(
                    title=f"Model Accuracy by Regime — {market}",
                    yaxis_title="Accuracy",
                    template="plotly_white",
                    height=400, width=600,
                )
                path = fig_dir / f"regime_analysis_{market}.png"
                pio.write_image(fig, str(path), scale=2)
                saved.append(path.name)

    # --- 13. Spillover matrix heatmap ---
    spill_path = results_dir / "spillover_matrix.csv"
    if spill_path.exists():
        spill = pd.read_csv(spill_path)
        if not spill.empty:
            # Try to reshape into a matrix
            if "source" in spill.columns and "target" in spill.columns and "mean_correlation" in spill.columns:
                markets_list = sorted(set(spill["source"].tolist() + spill["target"].tolist()))
                n = len(markets_list)
                corr_matrix = pd.DataFrame(0.0, index=markets_list, columns=markets_list)
                for _, row in spill.iterrows():
                    corr_matrix.loc[row["source"], row["target"]] = row["mean_correlation"]

                fig = go.Figure(go.Heatmap(
                    z=corr_matrix.values,
                    x=markets_list, y=markets_list,
                    colorscale="RdBu_r",
                    zmid=0,
                    text=[[f"{corr_matrix.iloc[i, j]:.3f}" for j in range(n)] for i in range(n)],
                    texttemplate="%{text}",
                    colorbar=dict(title="Correlation"),
                ))
                fig.update_layout(
                    title="Cross-Market Spillover Correlation Matrix",
                    template="plotly_white",
                    height=500, width=600,
                )
                path = fig_dir / "spillover_heatmap.png"
                pio.write_image(fig, str(path), scale=2)
                saved.append(path.name)

    # --- 14. Temporal holdout: CV vs out-of-sample ---
    for market in all_prices:
        th_path = results_dir / f"temporal_holdout_{market}.csv"
        comp_path = results_dir / f"model_comparison_{market}.csv"
        if not th_path.exists():
            continue
        th = pd.read_csv(th_path)
        if th.empty or "accuracy" not in th.columns:
            continue

        cv_acc = None
        if comp_path.exists():
            comp = pd.read_csv(comp_path)
            if not comp.empty and "mean_accuracy" in comp.columns:
                cv_acc = comp["mean_accuracy"].max()

        fig = go.Figure()
        bars = ["Out-of-Sample\n(2023+)"]
        vals = [th["accuracy"].iloc[0]]
        colors = ["#e65100"]

        if cv_acc is not None:
            bars.insert(0, "Cross-Validation\n(2015-2025)")
            vals.insert(0, cv_acc)
            colors.insert(0, "#1976d2")

        fig.add_trace(go.Bar(
            x=bars, y=vals,
            marker_color=colors,
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
        ))

        if cv_acc and th["accuracy"].iloc[0]:
            gap = cv_acc - th["accuracy"].iloc[0]
            fig.add_annotation(
                x=0.5, y=max(vals) + 0.02,
                text=f"Gap: {gap:+.3f}" + (" ⚠️ overfitting" if gap > 0.03 else " ✓ stable"),
                showarrow=False,
                font=dict(size=13),
                xref="paper",
            )

        fig.update_layout(
            title=f"CV vs Out-of-Sample Performance — {market}",
            yaxis_title="Accuracy",
            template="plotly_white",
            height=400, width=500,
        )
        path = fig_dir / f"temporal_holdout_{market}.png"
        pio.write_image(fig, str(path), scale=2)
        saved.append(path.name)

    # --- 15. Extended Granger with BH correction ---
    gc_ext_path = results_dir / "granger_causality_extended.csv"
    if gc_ext_path.exists():
        gc_ext = pd.read_csv(gc_ext_path)
        if not gc_ext.empty and "p_value_adjusted" in gc_ext.columns:
            markets_list = sorted(set(gc_ext["source"].tolist() + gc_ext["target"].tolist()))
            n = len(markets_list)
            p_matrix = pd.DataFrame(1.0, index=markets_list, columns=markets_list)
            sig_matrix = pd.DataFrame("", index=markets_list, columns=markets_list)
            for _, row in gc_ext.iterrows():
                p_adj = row["p_value_adjusted"]
                p_matrix.loc[row["source"], row["target"]] = p_adj
                sig_matrix.loc[row["source"], row["target"]] = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "ns"

            log_p = -np.log10(p_matrix.values.clip(min=1e-50))
            np.fill_diagonal(log_p, 0)

            text = [[f"p={p_matrix.iloc[i, j]:.2e}\n{sig_matrix.iloc[i, j]}"
                     for j in range(n)] for i in range(n)]

            fig = go.Figure(go.Heatmap(
                z=log_p, x=markets_list, y=markets_list,
                colorscale="YlOrRd",
                text=text, texttemplate="%{text}",
                colorbar=dict(title="-log₁₀(p_adj)"),
            ))
            fig.update_layout(
                title="Granger Causality (BH-Corrected): Cross-Market Lead-Lag",
                xaxis_title="Target (affected)",
                yaxis_title="Source (leads)",
                template="plotly_white",
                height=500, width=600,
            )
            path = fig_dir / "granger_bh_corrected.png"
            pio.write_image(fig, str(path), scale=2)
            saved.append(path.name)

    logger.info(f"Saved {len(saved)} figures to {fig_dir}/")
    for name in saved:
        logger.info(f"  → {name}")


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
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Only regenerate figures from existing CSV results (fast)",
    )
    parser.add_argument(
        "--sectors",
        action="store_true",
        help="Run only sector vulnerability analysis (fast, ~2 min)",
    )
    parser.add_argument(
        "--sector-figures",
        action="store_true",
        help="Generate sector figures from existing sector CSVs (instant)",
    )
    args = parser.parse_args()

    setup_logger("INFO")

    if args.figures_only:
        logger.info("Regenerating figures from existing results...")
        fetcher = StockDataFetcher()
        symbol_map = {"US": "^GSPC", "India": "^NSEI", "China": "^HSI"}
        all_prices = {}
        for market in ["US", "India", "China"]:
            df = fetcher.fetch_symbol(symbol_map[market], ANALYSIS_START, ANALYSIS_END)
            if not df.empty:
                prices = df["close"]
                if hasattr(prices.index, "tz") and prices.index.tz is not None:
                    prices.index = prices.index.tz_localize(None)
                all_prices[market] = prices
        save_key_figures(all_prices, RESULTS_DIR)
        logger.info("Done — figures saved to results/figures/")
        return

    if args.sectors:
        logger.info("Running sector vulnerability analysis only...")
        run_sector_analysis(RESULTS_DIR)
        logger.info("Done — sector results saved to results/")
        return

    if args.sector_figures:
        logger.info("Generating sector figures from existing CSVs...")
        _generate_sector_figures(RESULTS_DIR)
        logger.info("Done — sector figures saved to results/figures/")
        return

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
        if hasattr(prices.index, "tz") and prices.index.tz is not None:
            prices.index = prices.index.tz_localize(None)
        all_prices[market] = prices

        # Get market-specific events (filter out invalid pre-2015 dates)
        market_events = events[
            events["markets_affected"].apply(
                lambda x, m=market: m in str(x)
            )
        ] if not events.empty else pd.DataFrame()
        if not market_events.empty:
            # Skip first 90 days to ensure sufficient estimation window for event study
            min_date = str(pd.Timestamp(ANALYSIS_START) + pd.Timedelta(days=90))[:10]
            market_events = market_events[market_events["date"] >= min_date]
            logger.info(f"Market events for {market}: {len(market_events)} (unique dates: {market_events['date'].nunique()})")

        # For non-US markets, use US returns as benchmark (market model)
        us_benchmark = None
        if market != "US" and "US" in all_prices:
            us_benchmark = compute_returns(all_prices["US"])
            us_benchmark.index = us_benchmark.index.normalize()

        # Run analyses
        run_dc_analysis(prices, market)
        run_event_study(prices, market_events, market, market_returns=us_benchmark)
        run_dc_event_correlation(prices, market_events, market)
        run_model_training(
            prices, market_events, market,
            sentiment_df=sentiment_df if not sentiment_df.empty else None,
        )

        # Research-grade extensions
        sent = sentiment_df if not sentiment_df.empty else None
        run_extended_event_study(prices, market_events, market, market_returns=us_benchmark)
        run_extended_model_training(prices, market_events, market, sentiment_df=sent)
        run_dc_sensitivity(prices, market_events, market, sentiment_df=sent)
        run_regime_analysis(prices, market_events, market, sentiment_df=sent)
        run_multi_horizon_analysis(prices, market_events, market, sentiment_df=sent)

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

        # Extended Granger: AIC lag selection + BH correction
        gc_ext = gc.cross_market_granger_extended(returns)
        gc_ext.to_csv(
            RESULTS_DIR / "granger_causality_extended.csv", index=False
        )
        logger.info(f"Granger (AIC + BH):\n{gc_ext}")

        # Cross-market contagion
        run_cross_market_contagion(all_prices, events)

    # Sector vulnerability analysis
    logger.info("\n=== Sector Vulnerability Analysis ===")
    try:
        run_sector_analysis(RESULTS_DIR)
    except Exception as e:
        logger.warning(f"Sector analysis failed (non-critical): {e}")

    # Save key figures for thesis / presentation
    try:
        save_key_figures(all_prices, RESULTS_DIR)
    except Exception as e:
        logger.warning(f"Figure saving failed (non-critical): {e}")

    logger.info("\nAnalysis pipeline complete! Results saved to results/")


if __name__ == "__main__":
    main()
