"""
Streamlit Dashboard: Geopolitical Impact Analyzer

Main entry point for the interactive dashboard.
Run with: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(
    page_title="Geopolitical Impact Analyzer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    st.title("Geopolitical Impact on Stock Prices")
    st.markdown(
        "**Analyzing Tariffs & Regional Conflicts through Directional Changes and LLMs**"
    )
    st.markdown("---")

    st.markdown("""
    ### Project Overview

    This dashboard provides an interactive analysis of how tariffs and regional
    conflicts impact stock prices across the **US-India-China triangle**.

    **Key Components:**

    | Module | Description |
    |--------|-------------|
    | **Market Overview** | Cross-market price comparison and event timeline |
    | **DC Analysis** | Directional Changes detection, scaling laws, and intrinsic time |
    | **Event Impact** | Event study (ACAR), BH-corrected p-values, Cohen's d effect sizes, LLM sentiment |
    | **Cross-Market** | Contagion analysis, Granger causality (standard + AIC/BH), impulse response, India trade diversion |
    | **Predictions** | Hybrid DC-LLM model, ablation study with CIs, baselines comparison |
    | **Robustness** | Placebo tests, window sensitivity, DC threshold sensitivity, regime analysis |
    | **Model Deep Dive** | Temporal holdout, rolling evaluation, calibration (ECE), learning curves, Diebold-Mariano test |

    **Data Sources:**
    - Stock prices from **Yahoo Finance** (full 2015-present)
    - Historical events from **GDELT daily aggregate** export files (CAMEO-coded, Apr 2013-present)
    - Recent articles from **GDELT DOC 2.0 API** (~last 3 months rolling window only)
    - RSS feeds and NewsAPI for real-time coverage
    - LLM sentiment from **Claude** (Anthropic API)

    **Navigate using the sidebar pages.**
    """)

    # Quick stats — load actual event count
    event_count = "N/A"
    try:
        from src.data_collection.conflict_tracker import ConflictEventTracker
        events = ConflictEventTracker().get_combined_geopolitical_events()
        event_count = str(len(events))
    except Exception:
        pass

    from src.utils.config import ANALYSIS_START, ANALYSIS_END
    start_year = ANALYSIS_START[:4]
    end_year = ANALYSIS_END[:4]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Markets Analyzed", "3", "US, India, China")
    with col2:
        st.metric("Time Period", f"{start_year}-{end_year}", f"{int(end_year) - int(start_year)} years")
    with col3:
        st.metric("Geopolitical Events", event_count, "Tariffs + Conflicts")
    with col4:
        st.metric("Dashboard Pages", "7", "5 core + 2 research-grade")

    st.markdown("---")

    # Research-grade analysis summary
    st.markdown("""
    ### Research-Grade Extensions

    This dashboard includes comprehensive statistical rigor checks:

    | Analysis | Purpose | Status |
    |----------|---------|--------|
    | **BH Correction** | Controls false discovery rate in multi-event testing | Event Impact > Extended Results |
    | **Cohen's d** | Standardized effect sizes for event impact | Event Impact > Extended Results |
    | **AIC Lag Selection** | Proper Granger lag via VAR, not min p-value | Cross-Market > Extended Granger |
    | **Temporal Holdout** | Train 2015-2022, test 2023+ (no lookahead) | Model Deep Dive > Temporal Holdout |
    | **Rolling Evaluation** | 3-year rolling window, quarterly test | Model Deep Dive > Rolling Evaluation |
    | **Placebo Test** | 500 random non-event dates to validate ACAR | Robustness > Placebo Test |
    | **Diebold-Mariano** | Statistical significance of model accuracy gap | Model Deep Dive > Model Comparison |
    | **Calibration (ECE)** | Are predicted probabilities meaningful? | Model Deep Dive > Model Calibration |
    | **Learning Curves** | Does more data help? | Model Deep Dive > Learning Curves |
    | **DC Sensitivity** | Results stable across DC thresholds? | Robustness > DC Threshold Sensitivity |
    | **Window Sensitivity** | Results stable across event windows? | Robustness > Window Sensitivity |
    | **Regime Analysis** | Pre/post-COVID, tariff vs conflict | Robustness > Regime Analysis |
    """)


if __name__ == "__main__":
    main()
