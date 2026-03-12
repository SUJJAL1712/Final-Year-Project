# Geopolitical Impact Analyzer

**Analyzing the Potential Impact of Tariffs and Regional Conflicts on Stock Prices through Directional Changes and LLMs**

*Final Year Engineering Project*

---

## Overview

This project investigates how geopolitical events — tariffs, trade wars, and regional conflicts — impact stock prices across the **US-India-China triangle**. It combines two novel approaches:

1. **Directional Changes (DC)**: A threshold-based event detection framework that captures the market's *intrinsic time*, revealing the true dynamics of price movements during geopolitical stress
2. **LLM Analysis (Claude)**: Using large language models to classify geopolitical events, extract market-specific sentiment, and assess cross-market transmission channels

### Key Innovations

- **DC-LLM Hybrid Model**: Combines DC-derived features (intrinsic time, magnitude, frequency) with LLM-derived features (sentiment, event classification) for market direction prediction
- **Cross-Market Contagion via Intrinsic Time**: Measures how fast geopolitical shocks propagate from one market to another using DC's time compression factor
- **India Trade Diversion Hypothesis**: Tests whether India benefits when US imposes tariffs on China (trade diversion effect)
- **Ablation Study**: Demonstrates that the combined DC+LLM approach outperforms either component alone

## Project Structure

```
├── config/                          # Configuration files
│   ├── settings.yaml               # Global settings
│   ├── markets.yaml                # US/India/China market config
│   └── events_taxonomy.yaml        # Event classification taxonomy
├── src/
│   ├── data_collection/            # Data fetching modules
│   │   ├── stock_fetcher.py       # Yahoo Finance market data
│   │   ├── news_collector.py      # News API & RSS collection
│   │   ├── tariff_tracker.py      # Curated tariff event database
│   │   └── conflict_tracker.py    # Curated conflict event database
│   ├── directional_changes/        # DC framework
│   │   ├── dc_algorithm.py        # Core DC detection algorithm
│   │   ├── dc_features.py        # DC-based feature extraction
│   │   └── intrinsic_time.py     # Intrinsic time & cross-market TCF
│   ├── llm_pipeline/              # Claude API analysis
│   │   ├── client.py             # API client with caching
│   │   ├── event_classifier.py   # Geopolitical event classification
│   │   ├── sentiment_analyzer.py # Market-specific sentiment
│   │   ├── impact_assessor.py    # Market impact assessment
│   │   └── prompts/              # Structured prompt templates
│   ├── analysis/                   # Quantitative analysis
│   │   ├── event_study.py        # Event study methodology (CAR)
│   │   ├── dc_event_correlation.py # DC-geopolitical correlation
│   │   ├── cross_market_contagion.py # US-India-China contagion
│   │   ├── sector_vulnerability.py   # Sector-level analysis
│   │   └── granger_causality.py  # Granger causality & VAR
│   ├── models/                     # Prediction models
│   │   ├── feature_engineering.py # Unified feature matrix
│   │   ├── hybrid_model.py       # DC-LLM hybrid predictor
│   │   └── baselines.py          # Baseline models
│   └── visualization/             # Plotly visualizations
├── app/                            # Streamlit dashboard
│   ├── streamlit_app.py           # Main dashboard
│   └── pages/                     # Dashboard pages
├── notebooks/                      # Jupyter analysis notebooks
├── scripts/                        # Pipeline scripts
├── tests/                          # Test suite
└── data/                           # Data storage (gitignored)
```

## Quick Start

### 1. Setup Environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY
```

### 3. Collect Data

```bash
python scripts/collect_data.py --all
```

### 4. Run Analysis

```bash
python scripts/run_analysis.py
```

### 5. Launch Dashboard

```bash
streamlit run app/streamlit_app.py
```

### 6. Run Tests

```bash
pytest tests/ -v
```

## Methodology

### Directional Changes Framework

Unlike traditional fixed-interval analysis, DC detects when prices reverse direction by a threshold θ. This captures the market's *intrinsic time*:

- **DC Event**: Confirmed when price reverses ≥ θ from the last extreme
- **Overshoot**: Price continuation after DC confirmation until next reversal
- **Scaling Laws**: DC events follow power laws (N ∝ θ⁻², Duration ∝ θ²)
- **Time Compression Factor**: Measures how "eventful" a period is — spikes during geopolitical crises

### LLM Analysis Pipeline

Uses Claude to transform unstructured news into structured signals:

1. **Event Classification**: Maps news to a taxonomy of 30+ event subcategories
2. **Market-Specific Sentiment**: Separate sentiment scores for US, India, China
3. **Impact Assessment**: Estimates magnitude, timing, and transmission channels
4. **DC Attribution**: Links detected market movements to their geopolitical causes

### Hybrid Prediction Model

Combines three feature streams:
- **DC Features**: Direction, magnitude, frequency, overshoot ratio, acceleration
- **LLM Features**: Sentiment, event severity, impact estimates
- **Market Features**: Returns, volatility, momentum, cross-market signals

Ablation study demonstrates the value of each component.

## Key Research Questions

1. Do DC events cluster around tariff announcements and conflict escalations?
2. Are DC magnitudes larger during geopolitical events vs. normal periods?
3. How fast do shocks propagate across the US-India-China triangle?
4. Does India benefit from US-China trade tensions (trade diversion)?
5. Does the DC+LLM hybrid outperform either approach alone?

## Technologies

- **Python 3.10+** with pandas, numpy, scipy, scikit-learn
- **Anthropic Claude API** for LLM analysis
- **XGBoost/LightGBM** for prediction models
- **Plotly** for interactive visualizations
- **Streamlit** for the dashboard
- **yfinance** for market data
