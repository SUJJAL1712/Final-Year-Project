"""
Feature extraction from Directional Change events.

Transforms raw DC events into numerical features suitable for ML models.
These features capture the market's intrinsic dynamics and can be combined
with LLM-derived geopolitical features for prediction.
"""

import numpy as np
import pandas as pd

from src.directional_changes.dc_algorithm import (
    DCSummary,
    DirectionalChangeDetector,
    TrendDirection,
)


class DCFeatureExtractor:
    """Extracts time-series features from DC events."""

    def __init__(self, threshold: float = 0.02):
        self.threshold = threshold
        self.detector = DirectionalChangeDetector(threshold)

    def extract_features(
        self, prices: pd.Series, window: int = 10
    ) -> pd.DataFrame:
        """
        Extract DC-based features aligned to the original time index.

        For each date, computes features based on recent DC events within
        the lookback window.

        Args:
            prices: Price series with datetime index
            window: Number of recent DC events to consider

        Returns:
            DataFrame with DC features indexed by date
        """
        summaries = self.detector.detect(prices)
        if not summaries:
            return pd.DataFrame(index=prices.index)

        dc_df = self.detector.to_dataframe(summaries)
        features = self._compute_rolling_features(dc_df, prices.index, window)
        return features

    def _compute_rolling_features(
        self, dc_df: pd.DataFrame, dates: pd.DatetimeIndex, window: int
    ) -> pd.DataFrame:
        """Compute rolling features from DC events for each date."""
        records = []

        # Pre-sort DC events and use searchsorted for O(N log E) instead of O(N*E)
        dc_df = dc_df.sort_values("dc_confirm_time").reset_index(drop=True)
        dc_confirm_times = dc_df["dc_confirm_time"].values

        for date in dates:
            # Binary search for number of DC events up to this date
            n_past = int(np.searchsorted(dc_confirm_times, date, side="right"))

            if n_past < 2:
                records.append(self._empty_features(date))
                continue

            past_events = dc_df.iloc[:n_past]
            recent = past_events.iloc[-window:]
            rec = {"date": date}

            # Current market state
            last_event = past_events.iloc[-1]
            rec["current_direction"] = 1 if last_event["dc_direction"] == "UPTURN" else -1
            rec["last_dc_magnitude"] = last_event["dc_magnitude"]
            rec["last_os_ratio"] = last_event["overshoot_ratio"]

            # Time since last DC event (in days)
            rec["days_since_last_dc"] = (date - last_event["dc_confirm_time"]).days

            # Recent DC statistics
            rec["avg_dc_magnitude"] = recent["dc_magnitude"].abs().mean()
            rec["std_dc_magnitude"] = recent["dc_magnitude"].abs().std()
            rec["avg_os_ratio"] = recent["overshoot_ratio"].mean()
            rec["avg_dc_duration"] = recent["dc_duration_days"].mean()

            # Trend strength: ratio of upturns to total events
            upturn_count = (recent["dc_direction"] == "UPTURN").sum()
            rec["upturn_ratio"] = upturn_count / len(recent)

            # Volatility proxy: DC event frequency (more events = more volatile)
            if len(recent) >= 2:
                time_span = (recent.iloc[-1]["dc_confirm_time"] - recent.iloc[0]["dc_confirm_time"]).days
                rec["dc_frequency"] = len(recent) / max(time_span, 1)
            else:
                rec["dc_frequency"] = 0.0

            # Asymmetry: compare avg upturn magnitude vs downturn magnitude
            upturns = recent[recent["dc_direction"] == "UPTURN"]["dc_magnitude"]
            downturns = recent[recent["dc_direction"] == "DOWNTURN"]["dc_magnitude"]
            rec["magnitude_asymmetry"] = (
                upturns.abs().mean() - downturns.abs().mean()
                if len(upturns) > 0 and len(downturns) > 0
                else 0.0
            )

            # Acceleration: is the DC duration getting shorter? (market speeding up)
            if len(recent) >= 3:
                durations = recent["dc_duration_days"].values
                rec["duration_trend"] = np.polyfit(range(len(durations)), durations, 1)[0]
            else:
                rec["duration_trend"] = 0.0

            # Overshoot surprise: how much does recent OS deviate from DC scaling law
            rec["os_surprise"] = last_event["overshoot_ratio"] - recent["overshoot_ratio"].mean()

            records.append(rec)

        features_df = pd.DataFrame(records).set_index("date")
        return features_df

    @staticmethod
    def _empty_features(date) -> dict:
        """Return empty feature dict when not enough DC events."""
        return {
            "date": date,
            "current_direction": 0,
            "last_dc_magnitude": 0.0,
            "last_os_ratio": 0.0,
            "days_since_last_dc": np.nan,
            "avg_dc_magnitude": 0.0,
            "std_dc_magnitude": 0.0,
            "avg_os_ratio": 0.0,
            "avg_dc_duration": 0.0,
            "upturn_ratio": 0.5,
            "dc_frequency": 0.0,
            "magnitude_asymmetry": 0.0,
            "duration_trend": 0.0,
            "os_surprise": 0.0,
        }

    def compute_dc_based_labels(
        self, prices: pd.Series, horizon: int = 5
    ) -> pd.Series:
        """
        Create prediction labels based on next DC event direction.

        For each date, the label is:
        +1 if the next DC event is an UPTURN
        -1 if the next DC event is a DOWNTURN
         0 if no DC event within horizon

        This is the prediction target for the hybrid model.
        """
        summaries = self.detector.detect(prices)
        dc_df = self.detector.to_dataframe(summaries)

        labels = pd.Series(0, index=prices.index, name="dc_label")

        for _, event in dc_df.iterrows():
            confirm_date = event["dc_confirm_time"]
            direction = 1 if event["dc_direction"] == "UPTURN" else -1

            # Label the days leading up to this DC event
            lookback_start = confirm_date - pd.Timedelta(days=horizon)
            mask = (labels.index >= lookback_start) & (labels.index < confirm_date)
            labels[mask] = direction

        return labels
