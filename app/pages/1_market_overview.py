"""
Page 1: Market Overview
Displays cross-market comparison, event timeline, and basic statistics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd

from src.data_collection.stock_fetcher import StockDataFetcher
from src.data_collection.conflict_tracker import ConflictEventTracker
from src.visualization.event_timeline import EventTimelineVisualizer
from src.visualization.cross_market import CrossMarketVisualizer
from src.utils.config import ANALYSIS_START

st.set_page_config(page_title="Market Overview", layout="wide")
st.title("Market Overview: US-India-China Triangle")


@st.cache_data(ttl=3600)
def load_index_data():
    fetcher = StockDataFetcher()
    return fetcher.get_index_prices()


@st.cache_data
def load_events():
    tracker = ConflictEventTracker()
    return tracker.get_combined_geopolitical_events()


# Sidebar filters
st.sidebar.header("Filters")
start_date = st.sidebar.date_input("Start Date", pd.Timestamp(ANALYSIS_START))
end_date = st.sidebar.date_input("End Date", pd.Timestamp.now())
min_severity = st.sidebar.slider("Minimum Event Severity", 1, 10, 5)

# Load data
with st.spinner("Loading market data..."):
    try:
        index_prices = load_index_data()
        events_df = load_events()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Make sure you've run the data collection pipeline first.")
        st.stop()

if events_df.empty:
    st.warning("No event data available. Run `python scripts/collect_data.py --all` first.")
    events_filtered = pd.DataFrame()
else:
    # Guard: ensure required columns exist before filtering
    has_severity = "severity" in events_df.columns
    has_date = "date" in events_df.columns

    if has_date and has_severity:
        events_filtered = events_df[
            (events_df["date"] >= str(start_date))
            & (events_df["date"] <= str(end_date))
            & (events_df["severity"] >= min_severity)
        ]
    elif has_date:
        events_filtered = events_df[
            (events_df["date"] >= str(start_date))
            & (events_df["date"] <= str(end_date))
        ]
    else:
        events_filtered = events_df

# Display event timeline
st.subheader("Event Timeline")
if index_prices:
    viz = CrossMarketVisualizer()
    display_prices = {}
    for key, prices in index_prices.items():
        prices = prices[(prices.index >= str(start_date)) & (prices.index <= str(end_date))]
        if not prices.empty:
            market = key.split("_")[0] if "_" in key else key
            if market not in display_prices:
                display_prices[market] = prices

    fig = viz.plot_normalized_comparison(
        display_prices,
        events_filtered if not events_filtered.empty else None,
    )
    st.plotly_chart(fig, use_container_width=True)

# Event statistics
st.subheader("Event Statistics")
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Events", len(events_filtered) if not events_filtered.empty else 0)
    if not events_filtered.empty and "event_type" in events_filtered.columns:
        tariff_count = (events_filtered["event_type"] == "tariff").sum()
        conflict_count = (events_filtered["event_type"] == "conflict").sum()
        st.metric("Tariff Events", tariff_count)
        st.metric("Conflict Events", conflict_count)

with col2:
    if not events_filtered.empty and "severity" in events_filtered.columns:
        st.metric("Avg Severity", f"{events_filtered['severity'].mean():.1f}")
        st.metric("Max Severity", events_filtered["severity"].max())

# Event table
st.subheader("Event Database")
if not events_filtered.empty:
    display_cols = ["date", "event", "category", "severity", "event_type", "markets_affected"]
    available_cols = [c for c in display_cols if c in events_filtered.columns]
    st.dataframe(
        events_filtered[available_cols].sort_values("date", ascending=False),
        use_container_width=True,
        height=400,
    )
else:
    st.info("No events match the current filters.")
