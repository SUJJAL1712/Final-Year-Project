"""
Page 4: Cross-Market Analysis
US-India-China contagion, Granger causality (standard + AIC/BH-extended),
impulse response, and India trade diversion.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.data_collection.stock_fetcher import StockDataFetcher
from src.data_collection.conflict_tracker import ConflictEventTracker
from src.data_collection.tariff_tracker import TariffEventTracker
from src.analysis.cross_market_contagion import CrossMarketContagionAnalyzer
from src.analysis.granger_causality import GrangerCausalityAnalyzer
from src.visualization.cross_market import CrossMarketVisualizer
from src.visualization.heatmaps import HeatmapVisualizer
from src.utils.helpers import compute_returns
from src.utils.config import DATA_DIR

st.set_page_config(page_title="Cross-Market Analysis", layout="wide")
st.title("Cross-Market Analysis: US-India-China Triangle")

RESULTS_DIR = DATA_DIR.parent / "results"


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

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Cross-Market Comparison",
    "Granger Causality",
    "Extended Granger (AIC + BH)",
    "Impulse Response",
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

        # Spillover matrix from precomputed results
        spillover_path = RESULTS_DIR / "spillover_matrix.csv"
        if spillover_path.exists():
            try:
                spillover = pd.read_csv(spillover_path, index_col=0)
                st.subheader("Return Spillover Matrix")
                st.dataframe(spillover.round(4), use_container_width=True)
            except Exception:
                pass

with tab2:
    st.subheader("Granger Causality Tests (Standard)")
    st.markdown("Tests whether one market's returns help predict another's using min p-value lag selection.")

    returns = {}
    for name, p in prices.items():
        ret = compute_returns(p)
        ret.index = ret.index.normalize()
        returns[name] = ret

    # Try precomputed first
    gc_path = RESULTS_DIR / "granger_causality.csv"
    gc_results = None

    if gc_path.exists():
        try:
            gc_results = pd.read_csv(gc_path)
        except Exception:
            pass

    if gc_results is None and len(returns) >= 2:
        if st.button("Run Granger Causality Tests", key="gc_standard"):
            with st.spinner("Running Granger causality tests..."):
                gc = GrangerCausalityAnalyzer()
                gc_results = gc.cross_market_granger(returns)

    if gc_results is not None and not gc_results.empty:
        st.dataframe(gc_results.round(4), use_container_width=True)

        heatmap_viz = HeatmapVisualizer()
        fig = heatmap_viz.plot_granger_causality_heatmap(gc_results)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Extended Granger Causality (AIC Lag + BH Correction)")
    st.markdown("""
    **Improvements over standard test:**
    - **AIC lag selection**: Uses VAR information criterion instead of cherry-picking the lag with lowest p-value
    - **Benjamini-Hochberg correction**: Controls false discovery rate across multiple market pairs
    """)

    gc_ext_path = RESULTS_DIR / "granger_causality_extended.csv"
    gc_ext = None

    if gc_ext_path.exists():
        try:
            gc_ext = pd.read_csv(gc_ext_path)
        except Exception:
            pass

    if gc_ext is not None and not gc_ext.empty:
        # Summary metrics
        n_sig_raw = gc_ext["granger_causes"].sum() if "granger_causes" in gc_ext.columns else 0
        n_sig_adj = gc_ext["significant_adjusted"].sum() if "significant_adjusted" in gc_ext.columns else 0
        n_pairs = len(gc_ext)

        col1, col2, col3 = st.columns(3)
        col1.metric("Market Pairs Tested", n_pairs)
        col2.metric("Significant (AIC)", n_sig_raw)
        col3.metric("Significant (BH-adjusted)", n_sig_adj)

        # Side-by-side comparison: AIC p-value vs min-p p-value
        if "aic_p_value" in gc_ext.columns and "best_p_value_minp" in gc_ext.columns:
            fig = go.Figure()

            labels = gc_ext.apply(lambda r: f"{r['cause']} -> {r['effect']}", axis=1)

            fig.add_trace(go.Bar(
                x=labels,
                y=gc_ext["aic_p_value"],
                name="AIC-selected lag",
                marker_color="#1976d2",
            ))
            fig.add_trace(go.Bar(
                x=labels,
                y=gc_ext["best_p_value_minp"],
                name="Min p-value lag",
                marker_color="#e65100",
            ))

            fig.add_hline(y=0.05, line_dash="dash", line_color="#c62828",
                          annotation_text="alpha = 0.05")

            fig.update_layout(
                title="Granger p-values: AIC vs Min-p Lag Selection",
                yaxis_title="p-value",
                barmode="group",
                template="plotly_white",
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Extended heatmap using BH-adjusted p-values
        if "aic_p_adjusted" in gc_ext.columns:
            st.subheader("BH-Adjusted Granger Causality Heatmap")
            pivot = gc_ext.pivot(
                index="cause", columns="effect", values="aic_p_adjusted"
            ).fillna(1.0)

            significance = -np.log10(pivot.values.clip(min=1e-10))

            fig = go.Figure(data=go.Heatmap(
                z=significance,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale="YlOrRd",
                text=np.round(pivot.values, 4),
                texttemplate="p=%{text:.4f}",
                textfont={"size": 11},
                colorbar=dict(title="-log10(p_adj)"),
            ))

            fig.update_layout(
                title="BH-Adjusted Granger Causality: Row -> Column",
                xaxis_title="Effect (Target)",
                yaxis_title="Cause (Source)",
                template="plotly_white",
                height=450, width=600,
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Full Extended Granger Results"):
            st.dataframe(gc_ext.round(4), use_container_width=True)
    else:
        st.info("Run `python scripts/run_analysis.py` to generate extended Granger results.")

with tab4:
    st.subheader("VAR Impulse Response Functions")
    st.markdown("""
    Shows how a one-standard-deviation shock in one market propagates to others
    over time (20 trading days). Computed from a Vector Autoregression model.
    """)

    returns = {}
    for name, p in display_prices.items():
        ret = compute_returns(p)
        ret.index = ret.index.normalize()
        returns[name] = ret

    if len(returns) >= 2:
        if st.button("Compute Impulse Response", key="irf"):
            with st.spinner("Fitting VAR model..."):
                gc = GrangerCausalityAnalyzer()
                irf_result = gc.var_impulse_response(returns)

                if "error" not in irf_result:
                    viz = CrossMarketVisualizer()
                    fig = viz.plot_impulse_response(
                        irf_result["irf_data"],
                        irf_result["market_names"],
                        irf_result["periods"],
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    col1, col2 = st.columns(2)
                    col1.metric("VAR AIC", f"{irf_result['aic']:.2f}")
                    col2.metric("VAR BIC", f"{irf_result['bic']:.2f}")
                else:
                    st.warning(f"VAR model failed: {irf_result['error']}")
    else:
        st.info("Need at least 2 markets for impulse response analysis.")

with tab5:
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
