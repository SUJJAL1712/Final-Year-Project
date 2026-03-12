"""Tests for analysis modules."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.event_study import EventStudyAnalyzer
from src.analysis.dc_event_correlation import DCEventCorrelator
from src.utils.helpers import compute_returns


def _make_price_series(n: int = 500, seed: int = 42) -> pd.Series:
    """
    Create a realistic price series with controlled randomness.

    Uses a fixed seed for deterministic, reproducible test results.
    """
    rng = np.random.RandomState(seed)
    daily_returns = rng.normal(0, 0.015, n)
    prices = 100 * np.cumprod(1 + daily_returns)
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(prices, index=dates, name="close")


class TestEventStudyAnalyzer:
    def test_single_event(self):
        prices = _make_price_series(500)
        analyzer = EventStudyAnalyzer()
        result = analyzer.single_event_study(prices, "2020-06-15")

        assert "error" not in result
        assert "abnormal_returns" in result
        assert "cumulative_abnormal_returns" in result
        assert "p_value" in result
        assert 0 <= result["p_value"] <= 1

    def test_multi_event(self):
        prices = _make_price_series(500)
        dates = ["2020-03-15", "2020-06-15", "2020-09-15"]
        analyzer = EventStudyAnalyzer()
        results = analyzer.multi_event_study(prices, dates)

        assert not results.empty
        assert "event_day_ar" in results.columns
        assert "p_value" in results.columns

    def test_average_car(self):
        prices = _make_price_series(500)
        dates = ["2020-03-15", "2020-06-15", "2020-09-15"]
        analyzer = EventStudyAnalyzer()
        result = analyzer.average_car(prices, dates)

        assert "error" not in result
        assert "acar" in result
        assert result["n_events"] > 0

    def test_insufficient_data(self):
        prices = pd.Series(
            [100, 101],
            index=pd.bdate_range("2020-01-01", periods=2),
        )
        analyzer = EventStudyAnalyzer()
        result = analyzer.single_event_study(prices, "2020-01-01")
        assert "error" in result


class TestDCEventCorrelator:
    def test_temporal_coincidence(self):
        prices = _make_price_series(500)
        event_dates = ["2020-03-15", "2020-06-15", "2020-09-15"]

        correlator = DCEventCorrelator(0.02)
        result = correlator.temporal_coincidence(prices, event_dates)

        assert "coincident_dc_events" in result
        assert "expected_dc_events" in result
        assert "enrichment_ratio" in result
        assert "p_value" in result

    def test_magnitude_comparison(self):
        prices = _make_price_series(500)
        event_dates = ["2020-03-15", "2020-06-15"]

        correlator = DCEventCorrelator(0.02)
        result = correlator.dc_magnitude_around_events(prices, event_dates)

        assert "near_event_avg_magnitude" in result
        assert "normal_avg_magnitude" in result
