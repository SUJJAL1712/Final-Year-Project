"""
Page 3: Event Impact Analysis
Event study results, LLM analysis, and DC-event correlation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd

from src.data_collection.stock_fetcher import StockDataFetcher
from src.data_collection.tariff_tracker import TariffEventTracker
from src.data_collection.conflict_tracker import ConflictEventTracker
from src.analysis.event_study import EventStudyAnalyzer
from src.analysis.dc_event_correlation import DCEventCorrelator
from src.visualization.event_timeline import EventTimelineVisualizer
from src.visualization.heatmaps import HeatmapVisualizer

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
symbol = symbol_map[market]


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

# Filter events
if event_type == "Tariff Events":
    events_df = events_df[events_df["event_type"] == "tariff"]
elif event_type == "Conflict Events":
    events_df = events_df[events_df["event_type"] == "conflict"]

event_dates = events_df["date"].astype(str).tolist()

# Tabs
tab1, tab2, tab3 = st.tabs(["Event Study", "DC-Event Correlation", "Event Details"])

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

            col1, col2, col3 = st.columns(3)
            col1.metric("Events Analyzed", acar_result["n_events"])
            acar_val = acar_result["acar"].iloc[-1]
            col2.metric("Total ACAR", f"{acar_val:.4f}")
            col3.metric("Event Day ACAR", f"{acar_result['acar'].iloc[10]:.4f}" if len(acar_result['acar']) > 10 else "N/A")

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
        st.info("No events to analyze for this selection.")

with tab2:
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

with tab3:
    st.subheader("Event Database")
    if not events_df.empty:
        st.dataframe(
            events_df.sort_values("date", ascending=False),
            use_container_width=True,
            height=500,
        )
