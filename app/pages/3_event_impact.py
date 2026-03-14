"""
Page 3: Event Impact Analysis
Event study results, LLM analysis, DC-event correlation,
and extended analysis with BH correction and effect sizes.
"""

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
from src.analysis.event_study import EventStudyAnalyzer
from src.analysis.dc_event_correlation import DCEventCorrelator
from src.visualization.event_timeline import EventTimelineVisualizer
from src.visualization.heatmaps import HeatmapVisualizer
from src.utils.config import DATA_DIR

st.set_page_config(page_title="Event Impact", layout="wide")
st.title("Event Impact Analysis")

# Sidebar
st.sidebar.header("Analysis Settings")
market = st.sidebar.selectbox("Market", ["US (S&P 500)", "India (NIFTY 50)", "China (HSI)"])
event_type = st.sidebar.selectbox("Event Type", ["All", "Tariff Events", "Conflict Events"])
dc_threshold = st.sidebar.slider("DC Threshold", 0.01, 0.10, 0.02, 0.005)

symbol_map = {
    "US (S&P 500)": "^GSPC",
    "India (NIFTY 50)": "^NSEI",
    "China (HSI)": "^HSI",
}
market_name_map = {
    "US (S&P 500)": "US",
    "India (NIFTY 50)": "India",
    "China (HSI)": "China",
}
symbol = symbol_map[market]
market_name = market_name_map[market]

RESULTS_DIR = DATA_DIR.parent / "results"


@st.cache_data(ttl=3600)
def load_data(sym):
    fetcher = StockDataFetcher()
    from src.utils.config import ANALYSIS_START, ANALYSIS_END
    df = fetcher.fetch_symbol(sym, ANALYSIS_START, ANALYSIS_END)
    return df


@st.cache_data
def load_all_events():
    tracker = ConflictEventTracker()
    return tracker.get_combined_geopolitical_events()


with st.spinner("Loading data..."):
    try:
        df = load_data(symbol)
        events_df = load_all_events()
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

if df.empty:
    st.warning("No price data available.")
    st.stop()

prices = df["close"]

# Filter events by type
if not events_df.empty and "event_type" in events_df.columns:
    if event_type == "Tariff Events":
        events_df = events_df[events_df["event_type"] == "tariff"]
    elif event_type == "Conflict Events":
        events_df = events_df[events_df["event_type"] == "conflict"]

# Filter events by market
if not events_df.empty and "markets_affected" in events_df.columns:
    events_df = events_df[
        events_df["markets_affected"].apply(
            lambda x: market_name in str(x)
        )
    ].reset_index(drop=True)

event_dates = events_df["date"].astype(str).tolist() if not events_df.empty else []

# Tabs — now with extended analysis and sentiment
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Event Study",
    "Extended Results (BH + Effect Sizes)",
    "LLM Sentiment",
    "DC-Event Correlation",
    "Event Details",
])

with tab1:
    st.subheader("Event Study: Abnormal Returns")

    if len(event_dates) > 0:
        analyzer = EventStudyAnalyzer()
        acar_result = analyzer.average_car(prices, event_dates)

        if "error" not in acar_result:
            viz = EventTimelineVisualizer()
            fig = viz.plot_event_study_results(
                acar_result["acar"],
                acar_result["acar_std"],
                acar_result["n_events"],
                f"Average CAR: {market}",
            )
            st.plotly_chart(fig, use_container_width=True)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Events Analyzed", acar_result["n_events"])
            acar_val = acar_result["acar"].iloc[-1]
            col2.metric("Total ACAR", f"{acar_val:.4f}")
            acar_series = acar_result["acar"]
            col3.metric(
                "Event Day ACAR",
                f"{acar_series.iloc[10]:.4f}" if len(acar_series) > 10 else "N/A",
            )
            # Show CI if available
            if "acar_ci_lower" in acar_result and "acar_ci_upper" in acar_result:
                ci_low = acar_result["acar_ci_lower"]
                ci_high = acar_result["acar_ci_upper"]
                col4.metric("95% CI", f"[{ci_low:.4f}, {ci_high:.4f}]")

            # Individual event results
            multi_results = analyzer.multi_event_study(prices, event_dates)
            if not multi_results.empty:
                with st.expander("Individual Event Results"):
                    st.dataframe(
                        multi_results.sort_values("event_date", ascending=False),
                        use_container_width=True,
                    )
        else:
            st.warning("Insufficient data for event study.")
    else:
        st.info(f"No events to analyze for {market} with current filters.")

with tab2:
    st.subheader("Extended Event Study: BH-Corrected p-values & Effect Sizes")
    st.markdown("""
    - **Benjamini-Hochberg correction**: controls false discovery rate when testing many events
    - **Cohen's d**: standardized effect size (|d| > 0.2 = small, > 0.5 = medium, > 0.8 = large)
    """)

    # Try loading precomputed extended results
    ext_path = RESULTS_DIR / f"event_study_extended_{market_name}.csv"
    ext_data = None

    if ext_path.exists():
        try:
            ext_data = pd.read_csv(ext_path)
        except Exception:
            pass

    if ext_data is not None and not ext_data.empty:
        col1, col2, col3, col4 = st.columns(4)
        n_sig_raw = ext_data["significant"].sum() if "significant" in ext_data.columns else 0
        n_sig_adj = ext_data["significant_adjusted"].sum() if "significant_adjusted" in ext_data.columns else 0
        col1.metric("Events", len(ext_data))
        col2.metric("Significant (raw)", n_sig_raw)
        col3.metric("Significant (BH-adjusted)", n_sig_adj)
        avg_d = ext_data["cohens_d"].abs().mean() if "cohens_d" in ext_data.columns else 0
        col4.metric("Mean |Cohen's d|", f"{avg_d:.3f}")

        # Effect size distribution
        if "cohens_d" in ext_data.columns:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=ext_data["cohens_d"],
                nbinsx=30,
                marker_color="#1976d2",
                name="Cohen's d",
            ))
            fig.add_vline(x=0.2, line_dash="dash", line_color="#26a69a",
                          annotation_text="Small (0.2)")
            fig.add_vline(x=0.5, line_dash="dash", line_color="#ff9800",
                          annotation_text="Medium (0.5)")
            fig.add_vline(x=0.8, line_dash="dash", line_color="#c62828",
                          annotation_text="Large (0.8)")
            fig.add_vline(x=-0.2, line_dash="dash", line_color="#26a69a")
            fig.add_vline(x=-0.5, line_dash="dash", line_color="#ff9800")
            fig.add_vline(x=-0.8, line_dash="dash", line_color="#c62828")

            fig.update_layout(
                title=f"Effect Size Distribution: {market}",
                xaxis_title="Cohen's d",
                yaxis_title="Count",
                template="plotly_white",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Comparison: raw vs BH-adjusted significance
        if "p_value" in ext_data.columns and "p_value_adjusted" in ext_data.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ext_data["p_value"],
                y=ext_data["p_value_adjusted"],
                mode="markers",
                marker=dict(
                    size=8,
                    color=ext_data["cohens_d"].abs() if "cohens_d" in ext_data.columns else "#1976d2",
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="|Cohen's d|"),
                ),
                text=ext_data["event_date"].astype(str) if "event_date" in ext_data.columns else None,
                name="Events",
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color="grey"),
                name="y = x",
                showlegend=False,
            ))
            fig.add_hline(y=0.05, line_dash="dot", line_color="#c62828",
                          annotation_text="alpha = 0.05")
            fig.add_vline(x=0.05, line_dash="dot", line_color="#c62828")

            fig.update_layout(
                title="Raw vs BH-Adjusted p-values",
                xaxis_title="Raw p-value",
                yaxis_title="BH-adjusted p-value",
                template="plotly_white",
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Full Extended Results Table"):
            st.dataframe(ext_data.round(4), use_container_width=True)
    else:
        if len(event_dates) > 0:
            if st.button("Run Extended Event Study", type="primary"):
                with st.spinner("Running BH correction, Cohen's d..."):
                    try:
                        analyzer = EventStudyAnalyzer()
                        extended = analyzer.multi_event_study_extended(prices, event_dates)
                        if not extended.empty:
                            extended.to_csv(ext_path, index=False)
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.info("Run `python scripts/run_analysis.py` or click above to generate extended results.")
        else:
            st.info(f"No events for {market} with current filters.")

with tab3:
    st.subheader("LLM Sentiment Analysis")

    sentiment_path = RESULTS_DIR / "daily_sentiment.csv"
    if sentiment_path.exists():
        try:
            sentiment_df = pd.read_csv(sentiment_path, parse_dates=["date"], index_col="date")

            if not sentiment_df.empty:
                viz = EventTimelineVisualizer()
                fig = viz.plot_sentiment_timeline(sentiment_df)
                st.plotly_chart(fig, use_container_width=True)

                # Market-specific sentiment stats
                market_id = market_name.lower()
                col_name = f"{market_id}_sentiment_mean"
                if col_name in sentiment_df.columns:
                    sent_series = sentiment_df[col_name].dropna()
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Days Covered", len(sent_series))
                    col2.metric("Mean Sentiment", f"{sent_series.mean():.3f}")
                    col3.metric("Min Sentiment", f"{sent_series.min():.3f}")
                    col4.metric("Max Sentiment", f"{sent_series.max():.3f}")

                with st.expander("Sentiment Data"):
                    st.dataframe(sentiment_df.tail(50), use_container_width=True)
            else:
                st.info("Sentiment data file is empty.")
        except Exception as e:
            st.warning(f"Error loading sentiment data: {e}")
    else:
        st.info(
            "No LLM sentiment data available. "
            "Run `python scripts/collect_data.py --sentiment` to generate it."
        )

with tab4:
    st.subheader("DC-Event Temporal Coincidence")

    correlator = DCEventCorrelator(dc_threshold)

    if event_dates:
        coincidence = correlator.temporal_coincidence(prices, event_dates)

        col1, col2, col3 = st.columns(3)
        col1.metric("DC Events Near Geo Events", coincidence["coincident_dc_events"])
        col2.metric("Expected (Random)", f"{coincidence['expected_dc_events']:.1f}")
        col3.metric("Enrichment Ratio", f"{coincidence['enrichment_ratio']:.2f}x")

        if coincidence["significant"]:
            st.success(f"Significant clustering (p={coincidence['p_value']:.4f})")
        else:
            st.info(f"Not significant (p={coincidence['p_value']:.4f})")

        # Magnitude comparison
        st.subheader("DC Magnitude: Event vs Normal Periods")
        magnitude_comp = correlator.dc_magnitude_around_events(prices, event_dates)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Near Events Avg |Magnitude|",
                       f"{magnitude_comp['near_event_avg_magnitude']:.4f}")
            st.metric("Normal Period Avg |Magnitude|",
                       f"{magnitude_comp['normal_avg_magnitude']:.4f}")
        with col2:
            st.metric("Near Events Avg OS Ratio",
                       f"{magnitude_comp['near_event_avg_os_ratio']:.3f}")
            st.metric("Normal Period Avg OS Ratio",
                       f"{magnitude_comp['normal_avg_os_ratio']:.3f}")

        # Magnitude comparison bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Near Events", "Normal Periods"],
            y=[magnitude_comp["near_event_avg_magnitude"],
               magnitude_comp["normal_avg_magnitude"]],
            marker_color=["#c62828", "#607d8b"],
            name="Avg |Magnitude|",
        ))
        fig.update_layout(
            title="DC Magnitude: Event vs Normal Periods",
            yaxis_title="Average |DC Magnitude|",
            template="plotly_white",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("Event Database")
    if not events_df.empty:
        display_cols = ["date", "event", "category", "severity", "event_type", "markets_affected"]
        available_cols = [c for c in display_cols if c in events_df.columns]
        st.dataframe(
            events_df[available_cols].sort_values("date", ascending=False),
            use_container_width=True,
            height=500,
        )
    else:
        st.info("No events available for this selection.")
