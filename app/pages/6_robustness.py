"""
Page 6: Robustness & Sensitivity Analysis
Placebo tests, window sensitivity, DC threshold sensitivity, and regime analysis.
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
from src.utils.config import ANALYSIS_START, ANALYSIS_END, DATA_DIR

st.set_page_config(page_title="Robustness Analysis", layout="wide")
st.title("Robustness & Sensitivity Analysis")

st.markdown("""
Tests whether the main findings hold under different conditions.
These checks are essential for research credibility — results that are
sensitive to arbitrary choices (window size, threshold, time period) should
be treated with caution.
""")

RESULTS_DIR = DATA_DIR.parent / "results"

# Sidebar
st.sidebar.header("Settings")
market = st.sidebar.selectbox(
    "Market",
    ["US", "India", "China"],
    format_func=lambda x: {"US": "US (S&P 500)", "India": "India (NIFTY 50)", "China": "China (HSI)"}[x],
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Placebo Test",
    "Window Sensitivity",
    "DC Threshold Sensitivity",
    "Regime Analysis",
])

# ---------------------------------------------------------------------------
# Tab 1: Placebo Test
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Placebo Test: Are Event Study Results Spurious?")
    st.markdown("""
    Runs the event study on **500 random non-event dates**. If the real ACAR
    falls outside the placebo distribution, the event effect is unlikely to
    be due to chance.
    """)

    # Try loading precomputed results
    placebo_path = RESULTS_DIR / f"placebo_test_{market}.csv"
    placebo_data = None

    if placebo_path.exists():
        try:
            placebo_data = pd.read_csv(placebo_path)
        except Exception:
            pass

    if placebo_data is not None and not placebo_data.empty:
        row = placebo_data.iloc[0]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Real ACAR", f"{row['real_acar']:.6f}")
        col2.metric("Placebo Mean", f"{row['placebo_mean']:.6f}")
        col3.metric("Empirical p-value", f"{row['empirical_p_value']:.4f}")
        col4.metric("Placebos Run", int(row['n_placebos']))

        if row['empirical_p_value'] < 0.05:
            st.success(
                f"Real ACAR ({row['real_acar']:.6f}) is **outside** the placebo "
                f"distribution (p={row['empirical_p_value']:.4f}) — event effect is robust."
            )
        else:
            st.warning(
                f"Real ACAR ({row['real_acar']:.6f}) falls **within** the placebo "
                f"distribution (p={row['empirical_p_value']:.4f}) — cannot rule out chance."
            )

        # Visualization: normal distribution of placebos with real ACAR marked
        fig = go.Figure()
        mean_p = row['placebo_mean']
        std_p = row['placebo_std']
        x_range = np.linspace(mean_p - 4 * std_p, mean_p + 4 * std_p, 200)
        y_normal = (1 / (std_p * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean_p) / std_p) ** 2)

        fig.add_trace(go.Scatter(
            x=x_range, y=y_normal, mode="lines", name="Placebo Distribution",
            line=dict(color="#78909c", width=2), fill="tozeroy",
            fillcolor="rgba(120,144,156,0.2)",
        ))
        fig.add_vline(x=row['real_acar'], line_dash="solid", line_color="#c62828", line_width=3,
                      annotation_text=f"Real ACAR = {row['real_acar']:.6f}",
                      annotation_font_color="#c62828")
        fig.add_vline(x=row['placebo_5th'], line_dash="dash", line_color="#607d8b",
                      annotation_text="5th pctl")
        fig.add_vline(x=row['placebo_95th'], line_dash="dash", line_color="#607d8b",
                      annotation_text="95th pctl")

        fig.update_layout(
            title=f"Placebo Test: {market}",
            xaxis_title="ACAR",
            yaxis_title="Density",
            template="plotly_white",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        if st.button("Run Placebo Test (may take a few minutes)", type="primary", key="placebo"):
            with st.spinner("Running 500 placebo event studies..."):
                try:
                    symbol_map = {"US": "^GSPC", "India": "^NSEI", "China": "^HSI"}
                    fetcher = StockDataFetcher()
                    df = fetcher.fetch_symbol(symbol_map[market], ANALYSIS_START, ANALYSIS_END)
                    prices = df["close"]

                    tracker = ConflictEventTracker()
                    events = tracker.get_combined_geopolitical_events()
                    market_events = events[
                        events["markets_affected"].apply(lambda x: market in str(x))
                    ] if not events.empty and "markets_affected" in events.columns else pd.DataFrame()

                    event_dates = market_events["date"].astype(str).tolist()
                    analyzer = EventStudyAnalyzer()
                    result = analyzer.placebo_test(prices, event_dates, n_placebos=500)

                    if "error" not in result:
                        pd.DataFrame([result]).to_csv(placebo_path, index=False)
                        st.rerun()
                    else:
                        st.error(f"Placebo test failed: {result['error']}")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info("Run `python scripts/run_analysis.py` to generate placebo test results, or click the button above.")

# ---------------------------------------------------------------------------
# Tab 2: Window Sensitivity
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Event Window Sensitivity")
    st.markdown("""
    Tests whether the ACAR result changes significantly when the post-event
    window is varied (5, 10, 15, 20, 30 days). Stable results across windows
    indicate robust findings.
    """)

    ws_path = RESULTS_DIR / f"event_window_sensitivity_{market}.csv"
    ws_data = None

    if ws_path.exists():
        try:
            ws_data = pd.read_csv(ws_path)
        except Exception:
            pass

    if ws_data is not None and not ws_data.empty:
        # Plot ACAR vs window with CI
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=ws_data["post_window"],
            y=ws_data["acar"],
            mode="lines+markers",
            name="ACAR",
            line=dict(color="#1976d2", width=3),
            marker=dict(size=10),
        ))

        if "acar_ci_lower" in ws_data.columns and "acar_ci_upper" in ws_data.columns:
            fig.add_trace(go.Scatter(
                x=list(ws_data["post_window"]) + list(ws_data["post_window"][::-1]),
                y=list(ws_data["acar_ci_upper"]) + list(ws_data["acar_ci_lower"][::-1]),
                fill="toself",
                fillcolor="rgba(25,118,210,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="95% CI",
            ))

        fig.add_hline(y=0, line_dash="dash", line_color="grey")

        fig.update_layout(
            title=f"ACAR Sensitivity to Event Window: {market}",
            xaxis_title="Post-Event Window (days)",
            yaxis_title="ACAR",
            template="plotly_white",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(ws_data.round(6), use_container_width=True)

        # Stability assessment
        acar_range = ws_data["acar"].max() - ws_data["acar"].min()
        acar_mean = ws_data["acar"].abs().mean()
        if acar_mean > 0:
            cv = acar_range / acar_mean
            if cv < 0.5:
                st.success(f"ACAR is **stable** across windows (CV = {cv:.2f})")
            else:
                st.warning(f"ACAR varies across windows (CV = {cv:.2f}) — results may be window-dependent")
    else:
        st.info("Run `python scripts/run_analysis.py` to generate window sensitivity results.")

# ---------------------------------------------------------------------------
# Tab 3: DC Threshold Sensitivity
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("DC Threshold Sensitivity")
    st.markdown("""
    Tests whether the DC-only model's predictive power holds across different
    DC thresholds (theta = 0.01, 0.02, 0.05). If accuracy drops sharply at
    certain thresholds, the DC signal may be threshold-dependent.
    """)

    dc_path = RESULTS_DIR / f"dc_sensitivity_{market}.csv"
    dc_data = None

    if dc_path.exists():
        try:
            dc_data = pd.read_csv(dc_path)
        except Exception:
            pass

    if dc_data is not None and not dc_data.empty:
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=dc_data["theta"].astype(str),
            y=dc_data["mean_accuracy"],
            error_y=dict(
                type="data",
                symmetric=False,
                array=(dc_data["ci_upper"] - dc_data["mean_accuracy"]).tolist(),
                arrayminus=(dc_data["mean_accuracy"] - dc_data["ci_lower"]).tolist(),
            ),
            name="Accuracy",
            marker_color="#1976d2",
        ))

        fig.add_trace(go.Bar(
            x=dc_data["theta"].astype(str),
            y=dc_data["mean_f1"],
            name="F1 Score",
            marker_color="#26a69a",
        ))

        fig.update_layout(
            title=f"DC Threshold Sensitivity: {market}",
            xaxis_title="DC Threshold (theta)",
            yaxis_title="Score",
            barmode="group",
            template="plotly_white",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(dc_data.round(4), use_container_width=True)

        # Stability check
        acc_range = dc_data["mean_accuracy"].max() - dc_data["mean_accuracy"].min()
        if acc_range < 0.03:
            st.success(f"Accuracy is **stable** across thresholds (range = {acc_range:.4f})")
        else:
            st.warning(f"Accuracy varies across thresholds (range = {acc_range:.4f})")
    else:
        st.info("Run `python scripts/run_analysis.py` to generate DC sensitivity results.")

# ---------------------------------------------------------------------------
# Tab 4: Regime Analysis
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Regime Analysis: Pre/Post-COVID & Event Types")
    st.markdown("""
    Tests whether model performance and event impact differ across time periods
    (pre-COVID vs post-COVID) and event types (tariff vs conflict).
    """)

    regime_path = RESULTS_DIR / f"regime_analysis_{market}.csv"
    regime_data = None

    if regime_path.exists():
        try:
            regime_data = pd.read_csv(regime_path)
        except Exception:
            pass

    if regime_data is not None and not regime_data.empty:
        # Split into model regimes and event type regimes
        model_regimes = regime_data[~regime_data["regime"].str.startswith("event_type")]
        event_regimes = regime_data[regime_data["regime"].str.startswith("event_type")]

        if not model_regimes.empty:
            st.markdown("#### Model Performance by Regime")

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=model_regimes["regime"],
                y=model_regimes["mean_accuracy"],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=(model_regimes["ci_upper"] - model_regimes["mean_accuracy"]).tolist(),
                    arrayminus=(model_regimes["mean_accuracy"] - model_regimes["ci_lower"]).tolist(),
                ) if "ci_upper" in model_regimes.columns else None,
                name="Accuracy",
                marker_color=["#1976d2", "#e65100", "#607d8b"][:len(model_regimes)],
                text=[f"n={int(n)}" for n in model_regimes["n_samples"]],
                textposition="outside",
            ))

            fig.update_layout(
                title=f"Model Accuracy by Regime: {market}",
                xaxis_title="Regime",
                yaxis_title="Accuracy",
                template="plotly_white",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        if not event_regimes.empty:
            st.markdown("#### Event Impact by Type (ACAR)")

            fig = go.Figure()
            labels = event_regimes["regime"].str.replace("event_type_", "").str.capitalize()
            fig.add_trace(go.Bar(
                x=labels,
                y=event_regimes["mean_accuracy"],  # This is ACAR for event types
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=(event_regimes["ci_upper"] - event_regimes["mean_accuracy"]).tolist(),
                    arrayminus=(event_regimes["mean_accuracy"] - event_regimes["ci_lower"]).tolist(),
                ) if "ci_upper" in event_regimes.columns and not event_regimes["ci_upper"].isna().all() else None,
                name="ACAR",
                marker_color=["#ff9800", "#f44336"][:len(event_regimes)],
                text=[f"n={int(n)}" for n in event_regimes["n_samples"]],
                textposition="outside",
            ))

            fig.add_hline(y=0, line_dash="dash", line_color="grey")

            fig.update_layout(
                title=f"Average CAR by Event Type: {market}",
                xaxis_title="Event Type",
                yaxis_title="ACAR",
                template="plotly_white",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(regime_data.round(6), use_container_width=True)
    else:
        st.info("Run `python scripts/run_analysis.py` to generate regime analysis results.")
