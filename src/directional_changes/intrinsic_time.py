"""
Intrinsic Time Analysis.

Intrinsic time is a market-driven clock where time advances based on
directional changes rather than physical time. This captures the
"eventfulness" of different market periods — a day with a 5% crash
is much more eventful (many DC ticks) than a quiet day.

Key insight: When tariffs or conflicts hit, intrinsic time accelerates,
revealing the market's heightened reaction before physical-time
indicators might show significance.
"""

import numpy as np
import pandas as pd

from src.directional_changes.dc_algorithm import (
    DCSummary,
    DirectionalChangeDetector,
    MultiThresholdDC,
)


class IntrinsicTimeAnalyzer:
    """Analyzes market dynamics through the lens of intrinsic time."""

    def __init__(self, threshold: float = 0.02):
        self.threshold = threshold
        self.detector = DirectionalChangeDetector(threshold)

    def compute_intrinsic_time(self, prices: pd.Series) -> pd.Series:
        """
        Map physical time to intrinsic time.

        Each DC event increments intrinsic time by 1. Between DC events,
        intrinsic time is interpolated linearly. This creates a non-uniform
        time mapping where volatile periods are "stretched" and quiet
        periods are "compressed".

        Returns:
            Series mapping each date to its intrinsic time value.
        """
        summaries = self.detector.detect(prices)
        if not summaries:
            return pd.Series(0, index=prices.index, name="intrinsic_time")

        it = pd.Series(np.nan, index=prices.index, name="intrinsic_time")

        # Mark DC events with integer intrinsic time
        for i, s in enumerate(summaries):
            it[s.dc_event.confirm_time] = float(i + 1)

        # Set boundaries
        it.iloc[0] = 0.0

        # Interpolate between DC events
        it = it.interpolate(method="time")
        it = it.ffill().bfill()

        return it

    def time_compression_factor(
        self, prices: pd.Series, window_days: int = 30
    ) -> pd.Series:
        """
        Compute the time compression factor (TCF) over rolling windows.

        TCF = (number of DC events in window) / (expected DC events in window)

        TCF > 1: Market is more eventful than average (accelerated intrinsic time)
        TCF < 1: Market is calmer than average (decelerated intrinsic time)

        This is a key signal: when tariffs are announced, TCF should spike
        in affected markets, and the speed of contagion can be measured by
        comparing TCF spikes across US, India, and China.
        """
        summaries = self.detector.detect(prices)
        if not summaries:
            return pd.Series(1.0, index=prices.index, name="tcf")

        dc_df = self.detector.to_dataframe(summaries)
        dc_dates = dc_df["dc_confirm_time"]

        # Overall average event frequency
        total_days = (prices.index[-1] - prices.index[0]).days
        total_events = len(summaries)
        avg_events_per_day = total_events / max(total_days, 1)
        expected_in_window = avg_events_per_day * window_days

        # Rolling count of DC events using searchsorted — O(N log E) instead of O(N*E)
        dc_dates_sorted = np.sort(dc_dates.values)
        dates_arr = prices.index.values
        window_starts = dates_arr - np.timedelta64(window_days, "D")

        right_idx = np.searchsorted(dc_dates_sorted, dates_arr, side="right")
        left_idx = np.searchsorted(dc_dates_sorted, window_starts, side="left")
        counts = right_idx - left_idx

        tcf = pd.Series(
            counts / max(expected_in_window, 1e-6),
            index=prices.index,
            name="tcf",
        )

        return tcf

    def detect_activity_bursts(
        self, prices: pd.Series, tcf_threshold: float = 2.0, window_days: int = 30
    ) -> pd.DataFrame:
        """
        Detect periods of unusually high market activity.

        These bursts correspond to periods where intrinsic time is running
        much faster than physical time — likely driven by major events
        like tariff announcements or conflict escalations.

        Args:
            prices: Price series
            tcf_threshold: Minimum TCF to qualify as a burst
            window_days: Window for TCF computation

        Returns:
            DataFrame of burst periods with start, end, peak TCF
        """
        tcf = self.time_compression_factor(prices, window_days)
        is_burst = tcf >= tcf_threshold

        bursts = []
        in_burst = False
        burst_start = None

        for date, active in is_burst.items():
            if active and not in_burst:
                in_burst = True
                burst_start = date
            elif not active and in_burst:
                in_burst = False
                burst_segment = tcf[burst_start:date]
                bursts.append({
                    "start": burst_start,
                    "end": date,
                    "duration_days": (date - burst_start).days,
                    "peak_tcf": burst_segment.max(),
                    "avg_tcf": burst_segment.mean(),
                    "price_at_start": prices.get(burst_start, np.nan),
                    "price_at_end": prices.get(date, np.nan),
                })

        if in_burst:
            burst_segment = tcf[burst_start:]
            bursts.append({
                "start": burst_start,
                "end": prices.index[-1],
                "duration_days": (prices.index[-1] - burst_start).days,
                "peak_tcf": burst_segment.max(),
                "avg_tcf": burst_segment.mean(),
                "price_at_start": prices.get(burst_start, np.nan),
                "price_at_end": prices.iloc[-1],
            })

        return pd.DataFrame(bursts)


class CrossMarketIntrinsicTime:
    """
    Compare intrinsic time dynamics across markets.

    Key research question: When a tariff is announced targeting China,
    how quickly does intrinsic time accelerate in:
    1. US markets (immediate)
    2. Chinese markets (immediate to delayed)
    3. Indian markets (delayed, potentially opposite direction)

    The lag in intrinsic time acceleration reveals contagion speed.
    """

    def __init__(self, threshold: float = 0.02):
        self.analyzer = IntrinsicTimeAnalyzer(threshold)

    def cross_market_tcf(
        self, market_prices: dict[str, pd.Series], window_days: int = 30
    ) -> pd.DataFrame:
        """
        Compute TCF for multiple markets and align them.

        Args:
            market_prices: Dict mapping market name to price series
            window_days: Rolling window for TCF

        Returns:
            DataFrame with TCF columns for each market
        """
        tcf_dict = {}
        for name, prices in market_prices.items():
            tcf_dict[f"tcf_{name}"] = self.analyzer.time_compression_factor(
                prices, window_days
            )

        df = pd.DataFrame(tcf_dict)
        return df.dropna()

    def contagion_lag(
        self,
        source_prices: pd.Series,
        target_prices: pd.Series,
        max_lag_days: int = 10,
        window_days: int = 30,
    ) -> dict:
        """
        Estimate contagion lag between two markets using cross-correlation
        of their TCF series.

        Returns:
            Dict with optimal lag, correlation at that lag, and full lag profile
        """
        source_tcf = self.analyzer.time_compression_factor(source_prices, window_days)
        target_tcf = self.analyzer.time_compression_factor(target_prices, window_days)

        # Align series
        common_idx = source_tcf.index.intersection(target_tcf.index)
        s = source_tcf.reindex(common_idx).dropna()
        t = target_tcf.reindex(common_idx).dropna()

        # Cross-correlation at different lags
        lags = range(-max_lag_days, max_lag_days + 1)
        correlations = {}
        for lag in lags:
            shifted_t = t.shift(lag)
            valid = pd.concat([s, shifted_t], axis=1).dropna()
            if len(valid) > 10:
                correlations[lag] = valid.iloc[:, 0].corr(valid.iloc[:, 1])
            else:
                correlations[lag] = np.nan

        optimal_lag = max(correlations, key=lambda k: correlations.get(k, -1))

        return {
            "optimal_lag_days": optimal_lag,
            "max_correlation": correlations[optimal_lag],
            "lag_profile": correlations,
        }
