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
    | **DC Analysis** | Directional Changes detection and intrinsic time |
    | **Event Impact** | Event study results and LLM sentiment analysis |
    | **Cross-Market** | Contagion analysis and India trade diversion |
    | **Predictions** | Hybrid DC-LLM model results and ablation study |

    **Data Sources:**
    - Stock prices from **Yahoo Finance** (full 2015-present)
    - Historical events from **GDELT daily aggregate** export files (CAMEO-coded, Apr 2013-present)
    - Recent articles from **GDELT DOC 2.0 API** (~last 3 months rolling window only)
    - RSS feeds and NewsAPI for real-time coverage

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

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Markets Analyzed", "3", "US, India, China")
    with col2:
        st.metric("Time Period", f"{start_year}-{end_year}", f"{int(end_year) - int(start_year)} years")
    with col3:
        st.metric("Geopolitical Events", event_count, "Tariffs + Conflicts")


if __name__ == "__main__":
    main()
