"""
Core Directional Changes (DC) Algorithm Implementation.

The DC framework transforms a price time series into a sequence of alternating
upward and downward trends based on a threshold theta. Unlike fixed-interval
sampling, DC captures the market's intrinsic time — events are defined by
price movements, not clock ticks.

Key concepts:
- DC Event: A confirmed trend reversal (price moved >= theta from the last extreme)
- Overshoot (OS): The continuation of price movement after a DC event,
  until the next DC event confirms a reversal
- Intrinsic Time: A clock that ticks on DC events, not on calendar time

Reference: Glattfelder, Dupuis, Olsen (2011) "Patterns in high-frequency FX data"
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


class TrendDirection(Enum):
    UPTURN = 1
    DOWNTURN = -1


@dataclass
class DCEvent:
    """Represents a single Directional Change event."""
    # When the DC was confirmed
    confirm_time: pd.Timestamp
    confirm_price: float
    # The extreme point that preceded this DC
    extreme_time: pd.Timestamp
    extreme_price: float
    # The starting point of the DC move
    start_time: pd.Timestamp
    start_price: float
    # Direction of this DC (upturn = price went up to confirm)
    direction: TrendDirection
    # Threshold that triggered this DC
    threshold: float

    @property
    def dc_magnitude(self) -> float:
        """Price change during the DC move (from extreme to confirmation)."""
        return (self.confirm_price - self.extreme_price) / self.extreme_price

    @property
    def dc_duration(self) -> pd.Timedelta:
        """Time duration of the DC event."""
        return self.confirm_time - self.extreme_time

    @property
    def dc_duration_days(self) -> float:
        return self.dc_duration.total_seconds() / 86400


@dataclass
class OvershootEvent:
    """Represents the overshoot period following a DC event."""
    # Start = DC confirmation point
    start_time: pd.Timestamp
    start_price: float
    # End = next extreme point (or latest point if ongoing)
    end_time: pd.Timestamp
    end_price: float
    # The most extreme price reached during overshoot
    extreme_time: pd.Timestamp
    extreme_price: float
    direction: TrendDirection
    threshold: float

    @property
    def os_magnitude(self) -> float:
        """Price change during the overshoot."""
        return (self.extreme_price - self.start_price) / self.start_price

    @property
    def os_duration(self) -> pd.Timedelta:
        return self.end_time - self.start_time

    @property
    def os_duration_days(self) -> float:
        return self.os_duration.total_seconds() / 86400


@dataclass
class DCSummary:
    """Complete DC-OS pair forming one full trend segment."""
    dc_event: DCEvent
    overshoot: Optional[OvershootEvent] = None

    @property
    def total_magnitude(self) -> float:
        """Total price move = DC + OS."""
        os_mag = self.overshoot.os_magnitude if self.overshoot else 0.0
        return self.dc_event.dc_magnitude + os_mag

    @property
    def overshoot_ratio(self) -> float:
        """Ratio of overshoot to DC magnitude. Scaling law: OS/DC ≈ constant."""
        if abs(self.dc_event.dc_magnitude) < 1e-10:
            return 0.0
        os_mag = abs(self.overshoot.os_magnitude) if self.overshoot else 0.0
        return os_mag / abs(self.dc_event.dc_magnitude)


class DirectionalChangeDetector:
    """
    Detects Directional Change events in a price time series.

    The algorithm works as follows:
    1. Start tracking from the first price observation
    2. Track the running extreme (high for downturns, low for upturns)
    3. When price reverses from the extreme by at least theta, confirm a DC event
    4. After confirmation, track the overshoot until the next DC event

    Args:
        threshold: The DC threshold (theta). A value of 0.02 means 2% moves.
    """

    def __init__(self, threshold: float = 0.02):
        if threshold <= 0:
            raise ValueError("Threshold must be positive")
        self.threshold = threshold

    def detect(self, prices: pd.Series) -> list[DCSummary]:
        """
        Run DC detection on a price series.

        Args:
            prices: Time-indexed price series (e.g., daily close prices)

        Returns:
            List of DCSummary objects, each containing a DC event and its overshoot
        """
        if len(prices) < 2:
            return []

        prices = prices.dropna()
        dc_events: list[DCEvent] = []

        # Initialize: determine first trend by looking at first two distinct prices
        idx = prices.index
        p = prices.values

        # Find initial direction
        current_direction = None
        extreme_idx = 0
        extreme_price = p[0]
        extreme_time = idx[0]

        for i in range(1, len(p)):
            up_change = (p[i] - extreme_price) / extreme_price
            down_change = (extreme_price - p[i]) / extreme_price

            if up_change >= self.threshold and current_direction != TrendDirection.UPTURN:
                # Confirm upturn DC
                current_direction = TrendDirection.UPTURN
                dc_events.append(DCEvent(
                    confirm_time=idx[i],
                    confirm_price=p[i],
                    extreme_time=extreme_time,
                    extreme_price=extreme_price,
                    start_time=idx[0] if len(dc_events) == 0 else dc_events[-1].confirm_time,
                    start_price=p[0] if len(dc_events) == 0 else dc_events[-1].confirm_price,
                    direction=TrendDirection.UPTURN,
                    threshold=self.threshold,
                ))
                extreme_price = p[i]
                extreme_time = idx[i]
                extreme_idx = i

            elif down_change >= self.threshold and current_direction != TrendDirection.DOWNTURN:
                # Confirm downturn DC
                current_direction = TrendDirection.DOWNTURN
                dc_events.append(DCEvent(
                    confirm_time=idx[i],
                    confirm_price=p[i],
                    extreme_time=extreme_time,
                    extreme_price=extreme_price,
                    start_time=idx[0] if len(dc_events) == 0 else dc_events[-1].confirm_time,
                    start_price=p[0] if len(dc_events) == 0 else dc_events[-1].confirm_price,
                    direction=TrendDirection.DOWNTURN,
                    threshold=self.threshold,
                ))
                extreme_price = p[i]
                extreme_time = idx[i]
                extreme_idx = i

            else:
                # Update running extreme
                if current_direction == TrendDirection.UPTURN or current_direction is None:
                    if p[i] > extreme_price:
                        extreme_price = p[i]
                        extreme_time = idx[i]
                        extreme_idx = i
                if current_direction == TrendDirection.DOWNTURN or current_direction is None:
                    if p[i] < extreme_price:
                        extreme_price = p[i]
                        extreme_time = idx[i]
                        extreme_idx = i

        # Build DC + Overshoot summaries
        summaries = []
        for i, dc in enumerate(dc_events):
            # Overshoot runs from DC confirmation to the extreme before next DC
            if i + 1 < len(dc_events):
                next_dc = dc_events[i + 1]
                # Find the extreme between this DC confirm and next DC confirm
                mask = (prices.index > dc.confirm_time) & (prices.index <= next_dc.extreme_time)
                segment = prices[mask]

                if len(segment) > 0:
                    if dc.direction == TrendDirection.UPTURN:
                        os_extreme_idx = segment.idxmax()
                        os_extreme_price = segment.max()
                    else:
                        os_extreme_idx = segment.idxmin()
                        os_extreme_price = segment.min()

                    overshoot = OvershootEvent(
                        start_time=dc.confirm_time,
                        start_price=dc.confirm_price,
                        end_time=next_dc.extreme_time,
                        end_price=prices[next_dc.extreme_time],
                        extreme_time=os_extreme_idx,
                        extreme_price=os_extreme_price,
                        direction=dc.direction,
                        threshold=self.threshold,
                    )
                else:
                    overshoot = None
            else:
                overshoot = None

            summaries.append(DCSummary(dc_event=dc, overshoot=overshoot))

        return summaries

    def to_dataframe(self, summaries: list[DCSummary]) -> pd.DataFrame:
        """Convert DC summaries to a DataFrame for analysis."""
        records = []
        for s in summaries:
            rec = {
                "dc_confirm_time": s.dc_event.confirm_time,
                "dc_confirm_price": s.dc_event.confirm_price,
                "dc_extreme_time": s.dc_event.extreme_time,
                "dc_extreme_price": s.dc_event.extreme_price,
                "dc_direction": s.dc_event.direction.name,
                "dc_magnitude": s.dc_event.dc_magnitude,
                "dc_duration_days": s.dc_event.dc_duration_days,
                "threshold": self.threshold,
            }
            if s.overshoot:
                rec["os_magnitude"] = s.overshoot.os_magnitude
                rec["os_duration_days"] = s.overshoot.os_duration_days
                rec["os_extreme_price"] = s.overshoot.extreme_price
                rec["overshoot_ratio"] = s.overshoot_ratio
            else:
                rec["os_magnitude"] = 0.0
                rec["os_duration_days"] = 0.0
                rec["os_extreme_price"] = s.dc_event.confirm_price
                rec["overshoot_ratio"] = 0.0

            rec["total_magnitude"] = s.total_magnitude
            records.append(rec)

        return pd.DataFrame(records)


class MultiThresholdDC:
    """
    Run DC detection at multiple thresholds simultaneously.

    This provides a multi-resolution view of the market:
    - Small thresholds capture short-term noise and microstructure
    - Large thresholds capture major trends and regime changes
    - The relationship between scales reveals market complexity
    """

    def __init__(self, thresholds: list[float] | None = None):
        self.thresholds = thresholds or [0.005, 0.01, 0.02, 0.05, 0.10]
        self.detectors = {t: DirectionalChangeDetector(t) for t in self.thresholds}

    def detect_all(self, prices: pd.Series) -> dict[float, list[DCSummary]]:
        """Run DC detection at all thresholds."""
        return {t: det.detect(prices) for t, det in self.detectors.items()}

    def scaling_law_data(self, prices: pd.Series) -> pd.DataFrame:
        """
        Compute scaling law statistics across thresholds.

        The DC scaling laws state:
        - Number of DC events ∝ theta^(-2) (coastline length)
        - OS/DC ratio ≈ constant across thresholds
        - DC duration ∝ theta^2

        These are key empirical findings that validate market microstructure.
        """
        results = self.detect_all(prices)
        records = []

        for threshold, summaries in results.items():
            if not summaries:
                continue
            df = DirectionalChangeDetector(threshold).to_dataframe(summaries)
            records.append({
                "threshold": threshold,
                "num_dc_events": len(summaries),
                "avg_dc_magnitude": df["dc_magnitude"].abs().mean(),
                "avg_os_magnitude": df["os_magnitude"].abs().mean(),
                "avg_overshoot_ratio": df["overshoot_ratio"].mean(),
                "avg_dc_duration_days": df["dc_duration_days"].mean(),
                "avg_total_magnitude": df["total_magnitude"].abs().mean(),
                "std_dc_magnitude": df["dc_magnitude"].abs().std(),
                "coastline": df["dc_magnitude"].abs().sum(),
            })

        return pd.DataFrame(records)
