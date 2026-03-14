"""
Page 5: Prediction Model & Results
Hybrid DC-LLM model performance, ablation study (with CIs),
model comparison, feature importance, and predictions.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

RESULTS_DIR = DATA_DIR.parent / "results"

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
SYMBOL_TO_NAME = {"^GSPC": "US", "^NSEI": "India", "^HSI": "China"}
market_name = SYMBOL_TO_NAME[market_symbol]


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
    mkt_name = {"us": "US", "india": "India", "china": "China"}.get(
        market_id, market_id.capitalize()
    )

    if not all_events.empty and "markets_affected" in all_events.columns:
        events = all_events[
            all_events["markets_affected"].apply(
                lambda x: mkt_name in str(x)
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

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Ablation Study",
    "Extended Ablation (with CIs)",
    "Model Comparison",
    "Baselines",
    "Predictions",
])

with tab1:
    st.subheader("Ablation Study: Feature Group Contribution")
    st.markdown(
        "Demonstrates that combining DC + LLM features outperforms either alone."
    )

    # Try precomputed results first
    ablation_path = RESULTS_DIR / f"ablation_{market_name}.csv"
    ablation_data = None

    if ablation_path.exists():
        try:
            ablation_data = pd.read_csv(ablation_path)
        except Exception:
            pass

    if ablation_data is not None and not ablation_data.empty:
        viz = DCVisualizer()
        fig = viz.plot_ablation_results(ablation_data)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(ablation_data.round(4), use_container_width=True)
    elif st.button("Run Ablation Study", type="primary"):
        with st.spinner("Running ablation study (this may take a minute)..."):
            model = HybridDCLLMPredictor()
            ablation_data = model.run_ablation_study(features, labels, model_choice)

        if not ablation_data.empty:
            viz = DCVisualizer()
            fig = viz.plot_ablation_results(ablation_data)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(ablation_data.round(4), use_container_width=True)

with tab2:
    st.subheader("Extended Ablation with 95% Confidence Intervals")
    st.markdown("""
    Adds **t-distribution CIs** to each ablation variant, so you can see
    whether the accuracy difference between feature groups is statistically
    meaningful (overlapping CIs = not significant).
    """)

    ext_ablation_path = RESULTS_DIR / f"ablation_extended_{market_name}.csv"
    ext_ablation = None

    if ext_ablation_path.exists():
        try:
            ext_ablation = pd.read_csv(ext_ablation_path)
        except Exception:
            pass

    if ext_ablation is not None and not ext_ablation.empty:
        fig = go.Figure()

        # Accuracy bars with CI error bars
        has_ci = "ci_lower_accuracy" in ext_ablation.columns and "ci_upper_accuracy" in ext_ablation.columns
        error_y_args = {}
        if has_ci:
            error_y_args = dict(
                type="data",
                symmetric=False,
                array=(ext_ablation["ci_upper_accuracy"] - ext_ablation["mean_accuracy"]).tolist(),
                arrayminus=(ext_ablation["mean_accuracy"] - ext_ablation["ci_lower_accuracy"]).tolist(),
            )

        fig.add_trace(go.Bar(
            x=ext_ablation["feature_group"],
            y=ext_ablation["mean_accuracy"],
            error_y=error_y_args if error_y_args else None,
            name="Accuracy (95% CI)",
            marker_color="#1976d2",
        ))

        if "mean_f1" in ext_ablation.columns:
            f1_error = {}
            if "ci_lower_f1" in ext_ablation.columns and "ci_upper_f1" in ext_ablation.columns:
                f1_error = dict(
                    type="data",
                    symmetric=False,
                    array=(ext_ablation["ci_upper_f1"] - ext_ablation["mean_f1"]).tolist(),
                    arrayminus=(ext_ablation["mean_f1"] - ext_ablation["ci_lower_f1"]).tolist(),
                )
            fig.add_trace(go.Bar(
                x=ext_ablation["feature_group"],
                y=ext_ablation["mean_f1"],
                error_y=f1_error if f1_error else None,
                name="F1 (95% CI)",
                marker_color="#26a69a",
            ))

        fig.update_layout(
            title=f"Ablation Study with CIs: {market_name}",
            xaxis_title="Feature Group",
            yaxis_title="Score",
            barmode="group",
            template="plotly_white",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(ext_ablation.round(4), use_container_width=True)
    else:
        st.info("Run `python scripts/run_analysis.py` to generate extended ablation results.")

with tab3:
    st.subheader("Model Comparison")

    comp_path = RESULTS_DIR / f"model_comparison_{market_name}.csv"
    comp_data = None

    if comp_path.exists():
        try:
            comp_data = pd.read_csv(comp_path)
        except Exception:
            pass

    if comp_data is not None and not comp_data.empty:
        st.dataframe(comp_data.round(4), use_container_width=True)
    elif st.button("Compare All Models"):
        with st.spinner("Evaluating models..."):
            model = HybridDCLLMPredictor()
            comp_data = model.model_comparison(features, labels)
        st.dataframe(comp_data.round(4), use_container_width=True)

with tab4:
    st.subheader("Baseline Comparison")
    st.markdown("""
    Simple baselines the hybrid model must beat to demonstrate value:
    random, majority class, momentum, mean reversion, persistence, buy-and-hold.
    """)

    baseline_path = RESULTS_DIR / f"baselines_{market_name}.csv"
    baseline_data = None

    if baseline_path.exists():
        try:
            baseline_data = pd.read_csv(baseline_path)
        except Exception:
            pass

    if baseline_data is not None and not baseline_data.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=baseline_data["model"],
            y=baseline_data["mean_accuracy"],
            error_y=dict(type="data", array=baseline_data["std_accuracy"]) if "std_accuracy" in baseline_data.columns else None,
            name="Accuracy",
            marker_color="#607d8b",
        ))

        if "mean_f1" in baseline_data.columns:
            fig.add_trace(go.Bar(
                x=baseline_data["model"],
                y=baseline_data["mean_f1"],
                name="F1 Score",
                marker_color="#78909c",
            ))

        # Add hybrid model accuracy for comparison
        if comp_data is not None and not comp_data.empty:
            best_model = comp_data.loc[comp_data["mean_accuracy"].idxmax()]
            fig.add_hline(
                y=best_model["mean_accuracy"],
                line_dash="dash",
                line_color="#c62828",
                annotation_text=f"Best hybrid: {best_model['mean_accuracy']:.4f}",
            )

        fig.update_layout(
            title=f"Baseline Performance: {market_name}",
            xaxis_title="Model",
            yaxis_title="Score",
            barmode="group",
            template="plotly_white",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(baseline_data.round(4), use_container_width=True)
    else:
        if st.button("Run Baselines"):
            with st.spinner("Running baselines..."):
                baselines = BaselineModels()
                baseline_data = baselines.run_all_baselines(prices, labels)
            st.dataframe(baseline_data.round(4), use_container_width=True)

with tab5:
    st.subheader("Final Predictions")

    # Try precomputed predictions
    pred_path = RESULTS_DIR / f"predictions_{market_name}.csv"
    predictions = None

    if pred_path.exists():
        try:
            predictions = pd.read_csv(pred_path, index_col=0, parse_dates=True)
        except Exception:
            pass

    if predictions is not None and not predictions.empty:
        if "correct" in predictions.columns:
            acc = predictions["correct"].mean()
            st.metric("Test Accuracy", f"{acc:.2%}")

        # Feature importance from precomputed results
        st.subheader("Top Features")
        model = HybridDCLLMPredictor()
        try:
            result = model.train_and_evaluate(
                features, labels, model_name=model_choice
            )
            if "top_features" in result:
                heatmap_viz = HeatmapVisualizer()
                fig = heatmap_viz.plot_feature_importance(
                    result["top_features"]
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

        # Prediction timeline
        st.subheader("Prediction Results Over Time")
        n_show = min(50, len(predictions))
        st.dataframe(
            predictions.tail(n_show),
            use_container_width=True,
        )
    elif st.button("Generate Predictions"):
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
            n_show = min(50, len(predictions))
            st.dataframe(
                predictions.tail(n_show),
                use_container_width=True,
            )
