"""
Event Study Methodology for measuring geopolitical event impact.

The event study approach measures Abnormal Returns (AR) around geopolitical
events. By comparing actual returns to expected returns (estimated from a
pre-event estimation window), we isolate the event's specific market impact.

This is combined with DC analysis: DC events that coincide with geopolitical
events are analyzed for their abnormal characteristics.
"""

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger

from src.utils.config import get_config
from src.utils.helpers import compute_returns


class EventStudyAnalyzer:
    """
    Conducts event studies to measure the impact of geopolitical events
    on stock prices.

    Methodology:
    1. Define event window [-pre, +post] around event date
    2. Estimate normal returns from estimation window [-estimation, -pre-1]
    3. Compute abnormal returns = actual - expected
    4. Compute cumulative abnormal returns (CAR)
    5. Test statistical significance
    """

    def __init__(self):
        config = get_config().analysis_config
        self.pre_window = config["event_window"]["pre"]
        self.post_window = config["event_window"]["post"]
        self.estimation_window = config["estimation_window"]
        self.significance_level = config["significance_level"]

    def single_event_study(
        self,
        prices: pd.Series,
        event_date: str | pd.Timestamp,
        market_returns: pd.Series | None = None,
    ) -> dict:
        """
        Conduct an event study for a single event.

        Args:
            prices: Price series of the asset
            event_date: Date of the event
            market_returns: Market index returns for market model
                           (if None, uses mean-adjusted model)

        Returns:
            Dict with abnormal returns, CAR, and test statistics
        """
        event_date = pd.Timestamp(event_date)
        returns = compute_returns(prices, method="simple")

        # Find event date in the index (or nearest trading day)
        if event_date not in returns.index:
            nearest = returns.index[returns.index.get_indexer([event_date], method="nearest")[0]]
            logger.info(f"Event date {event_date} not in index, using {nearest}")
            event_date = nearest

        event_idx = returns.index.get_loc(event_date)

        # Define windows
        est_start = max(0, event_idx - self.pre_window - self.estimation_window)
        est_end = event_idx - self.pre_window
        event_start = max(0, event_idx - self.pre_window)
        event_end = min(len(returns), event_idx + self.post_window + 1)

        if est_end <= est_start or event_end <= event_start:
            logger.warning(f"Insufficient data for event study at {event_date}")
            return {"error": "insufficient_data"}

        estimation_returns = returns.iloc[est_start:est_end]
        event_returns = returns.iloc[event_start:event_end]

        # Estimate normal returns (market model or mean-adjusted)
        if market_returns is not None:
            expected_returns = self._market_model(
                estimation_returns, market_returns, event_returns
            )
        else:
            expected_returns = self._mean_adjusted_model(
                estimation_returns, event_returns
            )

        # Compute abnormal returns
        ar = event_returns - expected_returns
        car = ar.cumsum()

        # Statistical tests
        ar_std = estimation_returns.std()
        t_stat_ar = ar / ar_std if ar_std > 0 else ar * 0
        car_std = ar_std * np.sqrt(len(ar))
        t_stat_car = car.iloc[-1] / car_std if car_std > 0 else 0

        # P-value for CAR
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat_car), df=len(estimation_returns) - 1))

        # Key metrics
        event_day_ar = ar.iloc[self.pre_window] if self.pre_window < len(ar) else 0
        total_car = car.iloc[-1]
        pre_event_car = car.iloc[:self.pre_window].iloc[-1] if self.pre_window > 0 else 0
        post_event_car = total_car - pre_event_car

        return {
            "event_date": event_date,
            "abnormal_returns": ar,
            "cumulative_abnormal_returns": car,
            "event_day_ar": event_day_ar,
            "total_car": total_car,
            "pre_event_car": pre_event_car,
            "post_event_car": post_event_car,
            "t_statistic": t_stat_car,
            "p_value": p_value,
            "significant": p_value < self.significance_level,
            "estimation_std": ar_std,
        }

    def _market_model(
        self,
        est_returns: pd.Series,
        market_returns: pd.Series,
        event_returns: pd.Series,
    ) -> pd.Series:
        """Estimate expected returns using the market model: R_i = alpha + beta * R_m."""
        # Align estimation period
        common = est_returns.index.intersection(market_returns.index)
        y = est_returns.reindex(common).dropna()
        x = market_returns.reindex(common).dropna()
        common = y.index.intersection(x.index)
        y, x = y[common], x[common]

        if len(y) < 10:
            return self._mean_adjusted_model(est_returns, event_returns)

        # OLS regression
        x_with_const = np.column_stack([np.ones(len(x)), x.values])
        beta = np.linalg.lstsq(x_with_const, y.values, rcond=None)[0]
        alpha, beta_m = beta[0], beta[1]

        # Predict expected returns during event window
        event_market = market_returns.reindex(event_returns.index).fillna(0)
        expected = alpha + beta_m * event_market
        return expected

    def _mean_adjusted_model(
        self, est_returns: pd.Series, event_returns: pd.Series
    ) -> pd.Series:
        """Simple mean-adjusted model: expected return = mean of estimation period."""
        mean_return = est_returns.mean()
        return pd.Series(mean_return, index=event_returns.index)

    def multi_event_study(
        self,
        prices: pd.Series,
        event_dates: list[str],
        market_returns: pd.Series | None = None,
    ) -> pd.DataFrame:
        """
        Run event studies for multiple events and aggregate results.

        Returns:
            DataFrame with results for each event
        """
        results = []

        for date in event_dates:
            result = self.single_event_study(prices, date, market_returns)
            if "error" not in result:
                results.append({
                    "event_date": result["event_date"],
                    "event_day_ar": result["event_day_ar"],
                    "total_car": result["total_car"],
                    "pre_event_car": result["pre_event_car"],
                    "post_event_car": result["post_event_car"],
                    "t_statistic": result["t_statistic"],
                    "p_value": result["p_value"],
                    "significant": result["significant"],
                })

        df = pd.DataFrame(results)
        if not df.empty:
            logger.info(
                f"Event study: {df['significant'].sum()}/{len(df)} events significant"
            )
        return df

    def average_car(
        self,
        prices: pd.Series,
        event_dates: list[str],
        market_returns: pd.Series | None = None,
    ) -> dict:
        """
        Compute Average Cumulative Abnormal Returns (ACAR) across events.

        This is the primary aggregate measure of event impact.
        """
        all_cars = []

        for date in event_dates:
            result = self.single_event_study(prices, date, market_returns)
            if "error" not in result:
                car = result["cumulative_abnormal_returns"]
                # Normalize index to relative days
                car.index = range(-self.pre_window, len(car) - self.pre_window)
                all_cars.append(car)

        if not all_cars:
            return {"error": "no_valid_events"}

        car_matrix = pd.DataFrame(all_cars)
        acar = car_matrix.mean()
        acar_std = car_matrix.std() / np.sqrt(len(all_cars))

        # Cross-sectional t-test for ACAR
        t_stat = acar / acar_std
        p_values = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(all_cars) - 1))

        return {
            "acar": acar,
            "acar_std": acar_std,
            "t_statistics": t_stat,
            "p_values": p_values,
            "n_events": len(all_cars),
            "car_matrix": car_matrix,
        }
