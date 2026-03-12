"""
Correlation analysis between DC events and geopolitical events.

The core research question: Do Directional Change events coincide with
tariff announcements and conflict escalations? If so, DC provides a
quantitative signal that can be combined with LLM-classified events
for prediction.
"""

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger

from src.directional_changes.dc_algorithm import DirectionalChangeDetector
from src.utils.helpers import compute_returns


class DCEventCorrelator:
    """
    Analyzes the correlation between DC events and geopolitical events.

    Key analyses:
    1. Temporal coincidence: Do DC events cluster around geopolitical events?
    2. Magnitude analysis: Are DC events larger when driven by geopolitics?
    3. Asymmetry: Do tariffs cause more downturns than upturns?
    4. Overshoot behavior: Does the DC scaling law break during geopolitical events?
    """

    def __init__(self, threshold: float = 0.02):
        self.threshold = threshold
        self.detector = DirectionalChangeDetector(threshold)

    def temporal_coincidence(
        self,
        prices: pd.Series,
        event_dates: list[str],
        window_days: int = 3,
    ) -> dict:
        """
        Test if DC events are more likely to occur near geopolitical events.

        Uses a permutation test: if DC events cluster around event dates
        more than expected by chance, the geopolitical events are driving
        market movements detectable by DC.
        """
        summaries = self.detector.detect(prices)
        dc_df = self.detector.to_dataframe(summaries).copy()
        dc_dates = pd.to_datetime(dc_df["dc_confirm_time"])
        event_dates = [pd.Timestamp(d) for d in event_dates]

        # Count DC events near geopolitical events
        coincident_count = 0
        for event_date in event_dates:
            window_start = event_date - pd.Timedelta(days=window_days)
            window_end = event_date + pd.Timedelta(days=window_days)
            matches = dc_dates[(dc_dates >= window_start) & (dc_dates <= window_end)]
            coincident_count += len(matches)

        # Expected count under null hypothesis (uniform distribution)
        total_dc = len(dc_dates)
        total_days = (prices.index[-1] - prices.index[0]).days
        event_window_fraction = len(event_dates) * (2 * window_days + 1) / total_days
        expected_count = total_dc * event_window_fraction

        # Poisson test
        p_value = 1 - stats.poisson.cdf(coincident_count - 1, expected_count)

        return {
            "coincident_dc_events": coincident_count,
            "expected_dc_events": expected_count,
            "enrichment_ratio": coincident_count / max(expected_count, 1e-6),
            "p_value": p_value,
            "significant": p_value < 0.05,
            "total_dc_events": total_dc,
            "total_geo_events": len(event_dates),
        }

    def dc_magnitude_around_events(
        self,
        prices: pd.Series,
        event_dates: list[str],
        window_days: int = 5,
    ) -> pd.DataFrame:
        """
        Compare DC event magnitudes near geopolitical events vs. normal periods.
        """
        summaries = self.detector.detect(prices)
        dc_df = self.detector.to_dataframe(summaries).copy()
        dc_df["dc_confirm_time"] = pd.to_datetime(dc_df["dc_confirm_time"])
        event_timestamps = [pd.Timestamp(d) for d in event_dates]

        # Tag DC events as "near_event" or "normal"
        def is_near_event(dc_date):
            for ev_date in event_timestamps:
                if abs((dc_date - ev_date).days) <= window_days:
                    return True
            return False

        dc_df["near_geopolitical_event"] = dc_df["dc_confirm_time"].apply(is_near_event)

        near = dc_df[dc_df["near_geopolitical_event"]]
        normal = dc_df[~dc_df["near_geopolitical_event"]]

        comparison = {
            "near_event_count": len(near),
            "normal_count": len(normal),
            "near_event_avg_magnitude": near["dc_magnitude"].abs().mean(),
            "normal_avg_magnitude": normal["dc_magnitude"].abs().mean(),
            "near_event_avg_os_ratio": near["overshoot_ratio"].mean(),
            "normal_avg_os_ratio": normal["overshoot_ratio"].mean(),
            "near_event_avg_duration": near["dc_duration_days"].mean(),
            "normal_avg_duration": normal["dc_duration_days"].mean(),
        }

        # Statistical test: are magnitudes different?
        if len(near) > 2 and len(normal) > 2:
            t_stat, p_value = stats.mannwhitneyu(
                near["dc_magnitude"].abs(),
                normal["dc_magnitude"].abs(),
                alternative="greater",
            )
            comparison["magnitude_test_p_value"] = p_value
            comparison["magnitudes_significantly_larger"] = p_value < 0.05
        else:
            comparison["magnitude_test_p_value"] = None
            comparison["magnitudes_significantly_larger"] = None

        return comparison

    def directional_bias_analysis(
        self,
        prices: pd.Series,
        event_dates: list[str],
        event_types: list[str] | None = None,
        window_days: int = 3,
    ) -> dict:
        """
        Analyze if geopolitical events create directional bias in DC events.

        e.g., Tariff impositions should trigger more downturn DC events
        while trade deals should trigger more upturn DC events.
        """
        summaries = self.detector.detect(prices)
        dc_df = self.detector.to_dataframe(summaries).copy()
        dc_df["dc_confirm_time"] = pd.to_datetime(dc_df["dc_confirm_time"])

        results = {}

        for i, event_date in enumerate(event_dates):
            event_ts = pd.Timestamp(event_date)
            window_start = event_ts - pd.Timedelta(days=window_days)
            window_end = event_ts + pd.Timedelta(days=window_days)

            nearby_dc = dc_df[
                (dc_df["dc_confirm_time"] >= window_start)
                & (dc_df["dc_confirm_time"] <= window_end)
            ]

            if len(nearby_dc) > 0:
                upturn_pct = (nearby_dc["dc_direction"] == "UPTURN").mean()
                event_type = event_types[i] if event_types and i < len(event_types) else "unknown"

                results[str(event_date)] = {
                    "event_type": event_type,
                    "dc_events_nearby": len(nearby_dc),
                    "upturn_pct": upturn_pct,
                    "downturn_pct": 1 - upturn_pct,
                    "avg_magnitude": nearby_dc["dc_magnitude"].abs().mean(),
                    "dominant_direction": "UPTURN" if upturn_pct > 0.5 else "DOWNTURN",
                }

        return results

    def scaling_law_deviation(
        self,
        prices: pd.Series,
        event_dates: list[str],
        window_days: int = 30,
    ) -> dict:
        """
        Test if DC scaling laws break during geopolitical stress periods.

        The DC scaling law states that OS/DC ratio is approximately constant.
        During extreme geopolitical events, this ratio may deviate,
        indicating a market regime change.
        """
        from src.directional_changes.dc_algorithm import MultiThresholdDC

        multi_dc = MultiThresholdDC()

        # Full period scaling law
        full_scaling = multi_dc.scaling_law_data(prices)

        # Geopolitical period scaling law
        event_timestamps = [pd.Timestamp(d) for d in event_dates]
        geo_mask = pd.Series(False, index=prices.index)
        for ev in event_timestamps:
            window_start = ev - pd.Timedelta(days=window_days)
            window_end = ev + pd.Timedelta(days=window_days)
            geo_mask |= (prices.index >= window_start) & (prices.index <= window_end)

        geo_prices = prices[geo_mask]
        normal_prices = prices[~geo_mask]

        geo_scaling = multi_dc.scaling_law_data(geo_prices) if len(geo_prices) > 50 else None
        normal_scaling = multi_dc.scaling_law_data(normal_prices) if len(normal_prices) > 50 else None

        return {
            "full_period": full_scaling,
            "geopolitical_periods": geo_scaling,
            "normal_periods": normal_scaling,
            "deviates": geo_scaling is not None and normal_scaling is not None,
        }
