"""
Page 5: Prediction Model & Results
Hybrid DC-LLM model performance, ablation study, and predictions.
"""

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
    format_func=lambda x: {"^GSPC": "US (S&P 500)", "^NSEI": "India (NIFTY)", "^HSI": "China (HSI)"}[x],
)
model_choice = st.sidebar.selectbox(
    "Model",
    ["xgboost", "lightgbm", "random_forest", "gradient_boosting"],
)
prediction_horizon = st.sidebar.selectbox("Prediction Horizon (days)", [5, 10, 20])
dc_threshold = st.sidebar.slider("DC Threshold", 0.01, 0.10, 0.02, 0.005)


@st.cache_data(ttl=3600)
def load_and_build_features(sym, threshold, horizon):
    fetcher = StockDataFetcher()
    df = fetcher.fetch_symbol(sym, "2015-01-01", "2025-12-31")
    if df.empty:
        return None, None

    prices = df["close"]
    volume = df.get("volume")

    # Load events
    tracker = ConflictEventTracker()
    events = tracker.get_combined_geopolitical_events()

    # Build features
    engineer = FeatureEngineer(threshold)
    features = engineer.build_unified_features(
        prices, volume=volume, events_df=events
    )
    labels = engineer.build_labels(prices, horizon=horizon, method="direction")

    return features, labels


# Build features
with st.spinner("Building feature matrix..."):
    try:
        features, labels = load_and_build_features(
            market_symbol, dc_threshold, prediction_horizon
        )
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

if features is None or labels is None:
    st.warning("Could not build features. Check data availability.")
    st.stop()

st.success(f"Feature matrix: {features.shape[0]} samples x {features.shape[1]} features")

tab1, tab2, tab3 = st.tabs(["Ablation Study", "Model Comparison", "Predictions"])

with tab1:
    st.subheader("Ablation Study: Feature Group Contribution")
    st.markdown("Demonstrates that combining DC + LLM features outperforms either alone.")

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

        # Baselines
        st.subheader("Baseline Comparison")
        baselines = BaselineModels()
        prices_series = features.index.to_series()  # placeholder
        baseline_results = baselines.run_all_baselines(
            pd.Series(range(len(labels)), index=labels.index),
            labels,
        )
        st.dataframe(baseline_results.round(4), use_container_width=True)

with tab3:
    st.subheader("Final Predictions")

    if st.button("Generate Predictions"):
        with st.spinner("Training final model..."):
            model = HybridDCLLMPredictor()
            predictions = model.get_final_predictions(features, labels, model_choice)

        if not predictions.empty:
            acc = (predictions["correct"]).mean()
            st.metric("Test Accuracy", f"{acc:.2%}")

            # Feature importance
            st.subheader("Top Features")
            result = model.train_and_evaluate(features, labels, model_name=model_choice)
            if "top_features" in result:
                heatmap_viz = HeatmapVisualizer()
                fig = heatmap_viz.plot_feature_importance(result["top_features"])
                st.plotly_chart(fig, use_container_width=True)

            # Prediction timeline
            st.subheader("Prediction Results Over Time")
            st.dataframe(
                predictions.tail(50),
                use_container_width=True,
            )
