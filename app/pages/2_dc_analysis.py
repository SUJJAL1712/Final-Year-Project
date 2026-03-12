"""
Page 2: Directional Changes Analysis
Interactive DC detection, scaling law verification, and intrinsic time analysis.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd

from src.data_collection.stock_fetcher import StockDataFetcher
from src.directional_changes.dc_algorithm import DirectionalChangeDetector, MultiThresholdDC
from src.directional_changes.intrinsic_time import IntrinsicTimeAnalyzer
from src.visualization.dc_plots import DCVisualizer

st.set_page_config(page_title="DC Analysis", layout="wide")
st.title("Directional Changes Analysis")

st.markdown("""
**Directional Changes (DC)** transform price time series into a sequence of
alternating upward/downward trends based on a threshold. This captures the
market's *intrinsic time* rather than clock time.
""")

# Sidebar controls
st.sidebar.header("DC Parameters")
symbol = st.sidebar.selectbox(
    "Select Symbol",
    ["^GSPC", "^IXIC", "^NSEI", "^BSESN", "^HSI", "000001.SS",
     "AAPL", "TCS.NS", "BABA"],
    index=0,
)
threshold = st.sidebar.slider(
    "DC Threshold (theta)",
    min_value=0.005,
    max_value=0.15,
    value=0.02,
    step=0.005,
    format="%.3f",
    help="Minimum price reversal to trigger a DC event",
)
start_date = st.sidebar.date_input("Start Date", pd.Timestamp("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.Timestamp.now())


@st.cache_data(ttl=3600)
def fetch_prices(sym, start, end):
    fetcher = StockDataFetcher()
    df = fetcher.fetch_symbol(sym, str(start), str(end))
    return df


# Load data
with st.spinner(f"Fetching {symbol} data..."):
    try:
        df = fetch_prices(symbol, start_date, end_date)
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

if df.empty:
    st.warning("No data available for this symbol/period.")
    st.stop()

prices = df["close"]
viz = DCVisualizer()

# Tab layout
tab1, tab2, tab3, tab4 = st.tabs([
    "DC Events", "Scaling Laws", "Intrinsic Time", "Statistics",
])

with tab1:
    st.subheader(f"DC Events (theta = {threshold:.1%})")
    detector = DirectionalChangeDetector(threshold)
    summaries = detector.detect(prices)
    dc_df = detector.to_dataframe(summaries)

    col1, col2, col3 = st.columns(3)
    col1.metric("DC Events Detected", len(summaries))
    col2.metric("Avg Magnitude", f"{dc_df['dc_magnitude'].abs().mean():.2%}" if not dc_df.empty else "N/A")
    col3.metric("Avg OS/DC Ratio", f"{dc_df['overshoot_ratio'].mean():.2f}" if not dc_df.empty else "N/A")

    fig = viz.plot_dc_events_on_price(prices, dc_df, f"{symbol} with DC Events")
    st.plotly_chart(fig, use_container_width=True)

    if not dc_df.empty:
        with st.expander("DC Events Table"):
            st.dataframe(dc_df, use_container_width=True)

with tab2:
    st.subheader("DC Scaling Law Verification")
    st.markdown("""
    **Scaling laws** are empirical regularities of DC:
    - Number of DC events scales as θ^(-2) (coastline length)
    - OS/DC ratio is approximately constant across thresholds
    - DC duration scales as θ^2
    """)

    multi_dc = MultiThresholdDC()
    scaling_df = multi_dc.scaling_law_data(prices)

    if not scaling_df.empty:
        fig = viz.plot_scaling_laws(scaling_df)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Scaling Law Data"):
            st.dataframe(scaling_df.round(4), use_container_width=True)

with tab3:
    st.subheader("Intrinsic Time & Time Compression")
    analyzer = IntrinsicTimeAnalyzer(threshold)

    intrinsic_time = analyzer.compute_intrinsic_time(prices)
    tcf = analyzer.time_compression_factor(prices, window_days=30)

    fig = viz.plot_intrinsic_time(prices, intrinsic_time, tcf)
    st.plotly_chart(fig, use_container_width=True)

    # Activity bursts
    bursts = analyzer.detect_activity_bursts(prices, tcf_threshold=2.0)
    if not bursts.empty:
        st.subheader("Detected Activity Bursts")
        st.markdown("Periods where intrinsic time runs >2x faster than normal:")
        st.dataframe(bursts, use_container_width=True)

with tab4:
    st.subheader("DC Statistics Summary")
    if not dc_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Upturn Events:**")
            upturns = dc_df[dc_df["dc_direction"] == "UPTURN"]
            st.write(f"Count: {len(upturns)}")
            st.write(f"Avg magnitude: {upturns['dc_magnitude'].mean():.4f}")
            st.write(f"Avg duration: {upturns['dc_duration_days'].mean():.1f} days")

        with col2:
            st.markdown("**Downturn Events:**")
            downturns = dc_df[dc_df["dc_direction"] == "DOWNTURN"]
            st.write(f"Count: {len(downturns)}")
            st.write(f"Avg magnitude: {downturns['dc_magnitude'].mean():.4f}")
            st.write(f"Avg duration: {downturns['dc_duration_days'].mean():.1f} days")
