"""Tests for the Directional Changes algorithm."""

import numpy as np
import pandas as pd
import pytest

from src.directional_changes.dc_algorithm import (
    DirectionalChangeDetector,
    MultiThresholdDC,
    TrendDirection,
)
from src.directional_changes.dc_features import DCFeatureExtractor
from src.directional_changes.intrinsic_time import IntrinsicTimeAnalyzer


def _make_price_series(values: list[float], start_date: str = "2020-01-01") -> pd.Series:
    """Helper to create a price series from a list of values."""
    dates = pd.bdate_range(start=start_date, periods=len(values))
    return pd.Series(values, index=dates, name="close")


class TestDirectionalChangeDetector:
    def test_simple_upturn(self):
        """A clear upward move of >2% should detect an upturn DC."""
        prices = _make_price_series([100, 99, 98, 97, 100, 103, 105])
        detector = DirectionalChangeDetector(0.02)
        summaries = detector.detect(prices)
        assert len(summaries) > 0

    def test_simple_downturn(self):
        """A clear downward move should detect a downturn DC."""
        prices = _make_price_series([100, 101, 102, 103, 100, 97, 95])
        detector = DirectionalChangeDetector(0.02)
        summaries = detector.detect(prices)
        assert len(summaries) > 0

    def test_no_dc_below_threshold(self):
        """Price movements below threshold should not trigger DC events."""
        # Prices that move less than 1%
        prices = _make_price_series([100, 100.5, 100.3, 100.8, 100.2, 100.6])
        detector = DirectionalChangeDetector(0.02)
        summaries = detector.detect(prices)
        assert len(summaries) == 0

    def test_alternating_directions(self):
        """DC events should alternate between upturn and downturn."""
        prices = _make_price_series([
            100, 95, 90, 85, 90, 95, 100, 105,  # Down then up
            100, 95, 90, 95, 100, 105, 110,      # Down then up
        ])
        detector = DirectionalChangeDetector(0.05)
        summaries = detector.detect(prices)

        if len(summaries) >= 2:
            directions = [s.dc_event.direction for s in summaries]
            for i in range(1, len(directions)):
                assert directions[i] != directions[i - 1], "Consecutive DCs should alternate"

    def test_threshold_validation(self):
        """Threshold must be positive."""
        with pytest.raises(ValueError):
            DirectionalChangeDetector(0)
        with pytest.raises(ValueError):
            DirectionalChangeDetector(-0.01)

    def test_to_dataframe(self):
        """to_dataframe should return proper columns."""
        prices = _make_price_series([100, 90, 95, 105, 95, 80, 90, 105, 95])
        detector = DirectionalChangeDetector(0.05)
        summaries = detector.detect(prices)
        df = detector.to_dataframe(summaries)

        if not df.empty:
            expected_cols = [
                "dc_confirm_time", "dc_confirm_price", "dc_direction",
                "dc_magnitude", "dc_duration_days", "overshoot_ratio",
            ]
            for col in expected_cols:
                assert col in df.columns, f"Missing column: {col}"

    def test_empty_series(self):
        """Empty or too-short series should return empty results."""
        detector = DirectionalChangeDetector(0.02)
        assert detector.detect(pd.Series(dtype=float)) == []
        assert detector.detect(pd.Series([100], index=pd.bdate_range("2020-01-01", periods=1))) == []


class TestMultiThresholdDC:
    def test_scaling_law_data(self):
        """Multi-threshold should return scaling law statistics."""
        np.random.seed(42)
        prices = _make_price_series(
            (100 * np.cumprod(1 + np.random.normal(0, 0.015, 500))).tolist()
        )
        multi_dc = MultiThresholdDC([0.01, 0.02, 0.05])
        scaling = multi_dc.scaling_law_data(prices)

        assert len(scaling) > 0
        assert "threshold" in scaling.columns
        assert "num_dc_events" in scaling.columns
        # Smaller thresholds should have more events
        if len(scaling) >= 2:
            assert scaling.iloc[0]["num_dc_events"] >= scaling.iloc[-1]["num_dc_events"]


class TestDCFeatureExtractor:
    def test_feature_extraction(self):
        """Feature extraction should produce features for each date."""
        np.random.seed(42)
        prices = _make_price_series(
            (100 * np.cumprod(1 + np.random.normal(0, 0.02, 200))).tolist()
        )
        extractor = DCFeatureExtractor(0.02)
        features = extractor.extract_features(prices)

        assert len(features) == len(prices)
        assert "current_direction" in features.columns
        assert "dc_frequency" in features.columns


class TestIntrinsicTimeAnalyzer:
    def test_intrinsic_time_monotonic(self):
        """Intrinsic time should be non-decreasing."""
        np.random.seed(42)
        prices = _make_price_series(
            (100 * np.cumprod(1 + np.random.normal(0, 0.02, 200))).tolist()
        )
        analyzer = IntrinsicTimeAnalyzer(0.02)
        it = analyzer.compute_intrinsic_time(prices)

        diffs = it.diff().dropna()
        assert (diffs >= -1e-10).all(), "Intrinsic time should be non-decreasing"

    def test_tcf_positive(self):
        """Time compression factor should be positive."""
        np.random.seed(42)
        prices = _make_price_series(
            (100 * np.cumprod(1 + np.random.normal(0, 0.02, 200))).tolist()
        )
        analyzer = IntrinsicTimeAnalyzer(0.02)
        tcf = analyzer.time_compression_factor(prices)

        assert (tcf.dropna() >= 0).all(), "TCF should be non-negative"
