"""
Page 4: Cross-Market Analysis
US-India-China contagion, Granger causality, and India trade diversion.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd

from src.data_collection.stock_fetcher import StockDataFetcher
from src.data_collection.conflict_tracker import ConflictEventTracker
from src.data_collection.tariff_tracker import TariffEventTracker
from src.analysis.cross_market_contagion import CrossMarketContagionAnalyzer
from src.analysis.granger_causality import GrangerCausalityAnalyzer
from src.visualization.cross_market import CrossMarketVisualizer
from src.visualization.heatmaps import HeatmapVisualizer
from src.utils.helpers import compute_returns

st.set_page_config(page_title="Cross-Market Analysis", layout="wide")
st.title("Cross-Market Analysis: US-India-China Triangle")


@st.cache_data(ttl=3600)
def load_index_data():
    fetcher = StockDataFetcher()
    return fetcher.get_index_prices()


@st.cache_data
def load_tariff_events():
    tracker = TariffEventTracker()
    return tracker.get_curated_tariff_events()


with st.spinner("Loading cross-market data..."):
    try:
        prices = load_index_data()
        tariff_events = load_tariff_events()
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

# Build display_prices dict (one index per market)
display_prices = {}
for key, p in prices.items():
    market = key.split("_")[0] if "_" in key else key
    if market not in display_prices:
        display_prices[market] = p

tab1, tab2, tab3 = st.tabs([
    "Cross-Market Comparison",
    "Granger Causality",
    "India Trade Diversion",
])

with tab1:
    st.subheader("Normalized Market Comparison")
    viz = CrossMarketVisualizer()

    if display_prices:
        fig = viz.plot_normalized_comparison(display_prices)
        st.plotly_chart(fig, use_container_width=True)

    # Cross-market returns correlation
    st.subheader("Return Correlation Matrix")
    returns = {}
    for name, p in display_prices.items():
        ret = compute_returns(p)
        ret.index = ret.index.normalize()
        returns[name] = ret

    returns_df = pd.DataFrame(returns).dropna()
    if not returns_df.empty:
        corr = returns_df.corr()
        heatmap_viz = HeatmapVisualizer()
        fig = heatmap_viz.plot_cross_market_correlation_heatmap(corr)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Granger Causality Tests")
    st.markdown("Tests whether one market's returns help predict another's.")

    returns = {}
    for name, p in prices.items():
        ret = compute_returns(p)
        ret.index = ret.index.normalize()
        returns[name] = ret

    if len(returns) >= 2:
        with st.spinner("Running Granger causality tests..."):
            gc = GrangerCausalityAnalyzer()
            gc_results = gc.cross_market_granger(returns)

        if not gc_results.empty:
            st.dataframe(gc_results, use_container_width=True)

            heatmap_viz = HeatmapVisualizer()
            fig = heatmap_viz.plot_granger_causality_heatmap(gc_results)
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("India Trade Diversion Hypothesis")
    st.markdown("""
    **Hypothesis:** When the US imposes tariffs on China, India benefits
    from trade diversion — manufacturers shift production to India to
    avoid Chinese tariffs.

    We test this by analyzing India's market returns during US-China
    bilateral tariff events.
    """)

    # Get US-China bilateral events (guard column existence)
    us_china_events = pd.DataFrame()
    if (
        not tariff_events.empty
        and "source_country" in tariff_events.columns
        and "target_country" in tariff_events.columns
    ):
        us_china_events = tariff_events[
            (tariff_events["source_country"].isin(["US", "China"]))
            & (tariff_events["target_country"].isin(["US", "China"]))
        ]

    if not us_china_events.empty and len(display_prices) >= 3:
        analyzer = CrossMarketContagionAnalyzer()

        us_prices = display_prices.get("US")
        india_prices = display_prices.get("India")
        china_prices = display_prices.get("China")

        if us_prices is not None and india_prices is not None and china_prices is not None:
            event_dates = us_china_events["date"].astype(str).tolist()

            with st.spinner("Analyzing India trade diversion..."):
                result = analyzer.india_trade_diversion_test(
                    us_prices, india_prices, china_prices, event_dates
                )

            if "error" not in result:
                col1, col2, col3 = st.columns(3)
                col1.metric("Events Analyzed", result["total_events_analyzed"])
                col2.metric("India Avg Return (All Events)",
                            f"{result['india_avg_return_all_events']:.4f}")
                col3.metric("US-China Spread Correlation",
                            f"{result['us_china_spread_india_correlation']:.3f}")

                benefits = result.get("india_benefits_from_us_china_tension")
                if benefits is True:
                    st.success("India shows POSITIVE returns when both US and China decline — trade diversion hypothesis supported!")
                elif benefits is False:
                    st.warning("India also declines — trade diversion not clearly supported.")
                else:
                    st.info("Insufficient data to test.")

                # Plot
                viz = CrossMarketVisualizer()
                fig = viz.plot_india_trade_diversion(result)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Analysis error: {result['error']}")
        else:
            st.info("Need price data for all 3 markets (US, India, China).")
    else:
        st.info("Need US-China bilateral tariff events and price data for all 3 markets.")
