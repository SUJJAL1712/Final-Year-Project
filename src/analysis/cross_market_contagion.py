"""
Cross-market contagion analysis for the US-India-China triangle.

Measures how geopolitical shocks transmit across markets, with special
attention to:
1. US → China: Direct tariff impact
2. US → India: Ally impact (often positive trade diversion)
3. China → India: Competitive dynamics
4. Speed of transmission (using DC intrinsic time)
"""

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger

from src.directional_changes.intrinsic_time import CrossMarketIntrinsicTime
from src.utils.helpers import compute_returns, align_timeseries


class CrossMarketContagionAnalyzer:
    """
    Analyzes how geopolitical events propagate across US, India, and China markets.
    """

    def __init__(self, dc_threshold: float = 0.02):
        self.dc_threshold = dc_threshold
        self.cross_it = CrossMarketIntrinsicTime(dc_threshold)

    def compute_spillover_matrix(
        self,
        market_returns: dict[str, pd.Series],
        window: int = 60,
    ) -> pd.DataFrame:
        """
        Compute rolling spillover matrix between markets.

        Uses variance decomposition to measure how much of each market's
        variance is explained by shocks from other markets.
        """
        # Align all series
        aligned = {}
        common_index = None
        for name, ret in market_returns.items():
            ret.index = ret.index.normalize()
            if common_index is None:
                common_index = ret.index
            else:
                common_index = common_index.intersection(ret.index)

        for name, ret in market_returns.items():
            aligned[name] = ret.reindex(common_index).dropna()

        df = pd.DataFrame(aligned).dropna()
        markets = list(df.columns)

        # Rolling correlation matrix as proxy for spillover
        rolling_corr = df.rolling(window).corr()

        # Extract average cross-correlations
        avg_spillover = {}
        for i, m1 in enumerate(markets):
            for j, m2 in enumerate(markets):
                if i != j:
                    key = f"{m1}_to_{m2}"
                    # Get rolling correlation between m1 and m2
                    corrs = []
                    for date in df.index[window:]:
                        try:
                            corr_matrix = df.loc[:date].tail(window).corr()
                            corrs.append(corr_matrix.loc[m1, m2])
                        except (KeyError, ValueError):
                            pass
                    if corrs:
                        avg_spillover[key] = {
                            "mean_correlation": np.mean(corrs),
                            "std_correlation": np.std(corrs),
                            "max_correlation": np.max(corrs),
                        }

        return avg_spillover

    def event_contagion_analysis(
        self,
        market_prices: dict[str, pd.Series],
        event_date: str,
        source_market: str,
        window_days: int = 10,
    ) -> dict:
        """
        Analyze how a shock in the source market propagates to other markets.

        Measures:
        - Lag: How many days until other markets react
        - Magnitude: How much of the original shock is transmitted
        - Direction: Same or opposite direction
        """
        event_ts = pd.Timestamp(event_date)
        results = {"event_date": event_date, "source_market": source_market}

        # Get returns around event
        source_returns = compute_returns(market_prices[source_market])
        source_returns.index = source_returns.index.normalize()

        # Source market reaction
        event_window = source_returns[
            (source_returns.index >= event_ts - pd.Timedelta(days=1))
            & (source_returns.index <= event_ts + pd.Timedelta(days=window_days))
        ]
        results["source_cum_return"] = event_window.sum() if len(event_window) > 0 else 0

        # Analyze contagion to other markets
        for market_name, prices in market_prices.items():
            if market_name == source_market:
                continue

            target_returns = compute_returns(prices)
            target_returns.index = target_returns.index.normalize()

            target_window = target_returns[
                (target_returns.index >= event_ts - pd.Timedelta(days=1))
                & (target_returns.index <= event_ts + pd.Timedelta(days=window_days))
            ]

            if len(target_window) == 0:
                continue

            # Find the day of maximum absolute return (peak reaction)
            peak_idx = target_window.abs().idxmax()
            peak_lag = (peak_idx - event_ts).days

            # Cumulative return over window
            cum_return = target_window.sum()

            # Transmission ratio
            transmission = cum_return / results["source_cum_return"] if results["source_cum_return"] != 0 else 0

            results[f"{market_name}_cum_return"] = cum_return
            results[f"{market_name}_peak_lag_days"] = peak_lag
            results[f"{market_name}_peak_return"] = target_window.loc[peak_idx]
            results[f"{market_name}_transmission_ratio"] = transmission
            results[f"{market_name}_same_direction"] = (
                np.sign(cum_return) == np.sign(results["source_cum_return"])
            )

        return results

    def multi_event_contagion(
        self,
        market_prices: dict[str, pd.Series],
        events_df: pd.DataFrame,
        window_days: int = 10,
    ) -> pd.DataFrame:
        """
        Run contagion analysis across all events and aggregate patterns.
        """
        results = []

        for _, event in events_df.iterrows():
            # Determine source market from event
            markets_affected = event.get("markets_affected", [])
            if isinstance(markets_affected, str):
                import ast
                try:
                    markets_affected = ast.literal_eval(markets_affected)
                except (ValueError, SyntaxError):
                    # Fallback: parse comma-separated string
                    markets_affected = [
                        m.strip().strip("'\"")
                        for m in markets_affected.strip("[]").split(",")
                        if m.strip()
                    ]

            source_country = event.get("source_country", "")
            source_market = source_country if source_country in market_prices else (
                markets_affected[0] if markets_affected else None
            )

            if source_market and source_market in market_prices:
                result = self.event_contagion_analysis(
                    market_prices, str(event["date"]), source_market, window_days
                )
                result["event"] = event.get("event", "")
                result["category"] = event.get("category", "")
                result["severity"] = event.get("severity", 0)
                results.append(result)

        return pd.DataFrame(results)

    def india_trade_diversion_test(
        self,
        us_prices: pd.Series,
        india_prices: pd.Series,
        china_prices: pd.Series,
        us_china_event_dates: list[str],
        window_days: int = 20,
    ) -> dict:
        """
        Test the India Trade Diversion Hypothesis:
        When US imposes tariffs on China, does India benefit?

        We test if India's returns are positive when both US and China
        experience negative reactions to bilateral tariff events.
        """
        us_ret = compute_returns(us_prices)
        india_ret = compute_returns(india_prices)
        china_ret = compute_returns(china_prices)

        us_ret.index = us_ret.index.normalize()
        india_ret.index = india_ret.index.normalize()
        china_ret.index = china_ret.index.normalize()

        india_reactions = []
        for date_str in us_china_event_dates:
            event_ts = pd.Timestamp(date_str)
            window_end = event_ts + pd.Timedelta(days=window_days)

            india_window = india_ret[
                (india_ret.index >= event_ts) & (india_ret.index <= window_end)
            ]
            us_window = us_ret[
                (us_ret.index >= event_ts) & (us_ret.index <= window_end)
            ]
            china_window = china_ret[
                (china_ret.index >= event_ts) & (china_ret.index <= window_end)
            ]

            if len(india_window) > 0:
                india_reactions.append({
                    "event_date": date_str,
                    "india_cum_return": india_window.sum(),
                    "us_cum_return": us_window.sum() if len(us_window) > 0 else 0,
                    "china_cum_return": china_window.sum() if len(china_window) > 0 else 0,
                })

        if not india_reactions:
            return {"error": "no_valid_events"}

        reactions_df = pd.DataFrame(india_reactions)

        # Test: Is India return positive when US-China both negative?
        both_negative = reactions_df[
            (reactions_df["us_cum_return"] < 0) & (reactions_df["china_cum_return"] < 0)
        ]

        india_positive_when_both_negative = (
            both_negative["india_cum_return"].mean() > 0
            if len(both_negative) > 0 else None
        )

        # Correlation between US-China tension and India benefit
        us_china_spread = reactions_df["us_cum_return"] - reactions_df["china_cum_return"]
        corr_with_india = us_china_spread.corr(reactions_df["india_cum_return"])

        return {
            "total_events_analyzed": len(reactions_df),
            "both_us_china_negative_count": len(both_negative),
            "india_avg_return_when_both_negative": (
                both_negative["india_cum_return"].mean() if len(both_negative) > 0 else None
            ),
            "india_benefits_from_us_china_tension": india_positive_when_both_negative,
            "us_china_spread_india_correlation": corr_with_india,
            "india_avg_return_all_events": reactions_df["india_cum_return"].mean(),
            "reactions": reactions_df,
        }

    def dc_contagion_speed(
        self, market_prices: dict[str, pd.Series], window_days: int = 30
    ) -> dict:
        """
        Use DC intrinsic time to measure contagion speed between markets.
        """
        return self.cross_it.cross_market_tcf(market_prices, window_days)
