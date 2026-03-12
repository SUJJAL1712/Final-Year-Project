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

    **Navigate using the sidebar pages.**
    """)

    # Quick stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Markets Analyzed", "3", "US, India, China")
    with col2:
        st.metric("Time Period", "2015-2025", "10 years")
    with col3:
        st.metric("Geopolitical Events", "40+", "Tariffs + Conflicts")


if __name__ == "__main__":
    main()
