"""
Page 5: Prediction Model & Results
Hybrid DC-LLM model performance, ablation study, and predictions.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd

from src.data_collection.stock_fetcher import StockDataFetcher
from src.data_collection.conflict_tracker import ConflictEventTracker
from src.models.feature_engineering import FeatureEngineer
from src.models.hybrid_model import HybridDCLLMPredictor
from src.models.baselines import BaselineModels
from src.visualization.dc_plots import DCVisualizer
from src.visualization.heatmaps import HeatmapVisualizer
from src.utils.config import DATA_DIR

st.set_page_config(page_title="Predictions", layout="wide")
st.title("Hybrid DC-LLM Prediction Model")

st.markdown("""
This page demonstrates the **hybrid model** that combines:
- **Directional Changes features** (intrinsic time, DC frequency, magnitude)
- **LLM-derived features** (geopolitical sentiment, event classification)
- **Traditional market features** (returns, volatility, momentum)

The **ablation study** shows the contribution of each feature group.
""")

# Sidebar
st.sidebar.header("Model Settings")
market_symbol = st.sidebar.selectbox(
    "Target Market",
    ["^GSPC", "^NSEI", "^HSI"],
    format_func=lambda x: {
        "^GSPC": "US (S&P 500)",
        "^NSEI": "India (NIFTY)",
        "^HSI": "China (HSI)",
    }[x],
)
model_choice = st.sidebar.selectbox(
    "Model",
    ["xgboost", "lightgbm", "random_forest", "gradient_boosting"],
)
prediction_horizon = st.sidebar.selectbox(
    "Prediction Horizon (days)", [5, 10, 20]
)
dc_threshold = st.sidebar.slider("DC Threshold", 0.01, 0.10, 0.02, 0.005)


# Map symbols to market IDs for sentiment
SYMBOL_TO_MARKET = {"^GSPC": "us", "^NSEI": "india", "^HSI": "china"}


def _load_sentiment_cache() -> pd.DataFrame | None:
    """Load cached daily sentiment from results directory if available."""
    cache_path = DATA_DIR.parent / "results" / "daily_sentiment.csv"
    if cache_path.exists():
        try:
            df = pd.read_csv(cache_path, parse_dates=["date"], index_col="date")
            return df
        except Exception:
            return None
    return None


@st.cache_data(ttl=3600)
def load_and_build_features(sym, threshold, horizon):
    fetcher = StockDataFetcher()
    from src.utils.config import ANALYSIS_START, ANALYSIS_END
    df = fetcher.fetch_symbol(sym, ANALYSIS_START, ANALYSIS_END)
    if df.empty:
        return None, None, None

    prices = df["close"]
    volume = df.get("volume")

    # Load events (from GDELT, not hardcoded)
    tracker = ConflictEventTracker()
    all_events = tracker.get_combined_geopolitical_events()

    # Filter events to the target market to avoid feature contamination
    market_id = SYMBOL_TO_MARKET.get(sym, "us")
    market_name = {"us": "US", "india": "India", "china": "China"}.get(
        market_id, market_id.capitalize()
    )

    if not all_events.empty and "markets_affected" in all_events.columns:
        events = all_events[
            all_events["markets_affected"].apply(
                lambda x: market_name in str(x)
            )
        ].reset_index(drop=True)
    else:
        events = all_events

    # Load cached LLM sentiment if available
    sentiment_df = _load_sentiment_cache()

    # Build features with market-filtered events
    engineer = FeatureEngineer(threshold)
    features = engineer.build_unified_features(
        prices,
        volume=volume,
        events_df=events,
        sentiment_df=sentiment_df,
        market_id=market_id,
    )
    labels = engineer.build_labels(prices, horizon=horizon, method="direction")

    return features, labels, prices


# Build features
with st.spinner("Building feature matrix..."):
    try:
        result = load_and_build_features(
            market_symbol, dc_threshold, prediction_horizon
        )
        features, labels, prices = result
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

if features is None or labels is None:
    st.warning("Could not build features. Check data availability.")
    st.stop()

# Show LLM feature status
has_llm = any("llm_" in c for c in features.columns)
if has_llm:
    st.info("LLM sentiment features are included in the model.")
else:
    st.warning(
        "LLM sentiment features not available. "
        "Run `python scripts/collect_data.py --sentiment` to generate them."
    )

st.success(
    f"Feature matrix: {features.shape[0]} samples x {features.shape[1]} features"
)

tab1, tab2, tab3 = st.tabs(
    ["Ablation Study", "Model Comparison", "Predictions"]
)

with tab1:
    st.subheader("Ablation Study: Feature Group Contribution")
    st.markdown(
        "Demonstrates that combining DC + LLM features outperforms either alone."
    )

    if st.button("Run Ablation Study", type="primary"):
        with st.spinner("Running ablation study (this may take a minute)..."):
            model = HybridDCLLMPredictor()
            ablation = model.run_ablation_study(features, labels, model_choice)

        if not ablation.empty:
            viz = DCVisualizer()
            fig = viz.plot_ablation_results(ablation)
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(ablation.round(4), use_container_width=True)

with tab2:
    st.subheader("Model Comparison")

    if st.button("Compare All Models"):
        with st.spinner("Evaluating models..."):
            model = HybridDCLLMPredictor()
            comparison = model.model_comparison(features, labels)

        st.dataframe(comparison.round(4), use_container_width=True)

        # Baselines using actual prices (not placeholder)
        st.subheader("Baseline Comparison")
        if prices is not None:
            baselines = BaselineModels()
            baseline_results = baselines.run_all_baselines(prices, labels)
            st.dataframe(baseline_results.round(4), use_container_width=True)

with tab3:
    st.subheader("Final Predictions")

    if st.button("Generate Predictions"):
        with st.spinner("Training final model..."):
            model = HybridDCLLMPredictor()
            predictions = model.get_final_predictions(
                features, labels, model_choice
            )

        if not predictions.empty:
            acc = (predictions["correct"]).mean()
            st.metric("Test Accuracy", f"{acc:.2%}")

            # Feature importance
            st.subheader("Top Features")
            result = model.train_and_evaluate(
                features, labels, model_name=model_choice
            )
            if "top_features" in result:
                heatmap_viz = HeatmapVisualizer()
                fig = heatmap_viz.plot_feature_importance(
                    result["top_features"]
                )
                st.plotly_chart(fig, use_container_width=True)

            # Prediction timeline
            st.subheader("Prediction Results Over Time")
            st.dataframe(
                predictions.tail(50),
                use_container_width=True,
            )
