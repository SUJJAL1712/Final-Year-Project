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
from src.utils.helpers import compute_returns, adjust_pvalues


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
        seen_snapped_dates: set[pd.Timestamp] = set()

        for date in event_dates:
            result = self.single_event_study(prices, date, market_returns)
            if "error" not in result:
                snapped = result["event_date"]
                if snapped in seen_snapped_dates:
                    continue
                seen_snapped_dates.add(snapped)
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

        # CI on final ACAR value
        final_acar = float(acar.iloc[-1])
        final_acar_se = float(acar_std.iloc[-1])
        n_events = len(all_cars)
        ci_half = stats.t.ppf(0.975, df=n_events - 1) * final_acar_se
        acar_ci_lower = final_acar - ci_half
        acar_ci_upper = final_acar + ci_half

        return {
            "acar": acar,
            "acar_std": acar_std,
            "t_statistics": t_stat,
            "p_values": p_values,
            "n_events": n_events,
            "car_matrix": car_matrix,
            "acar_ci_lower": acar_ci_lower,
            "acar_ci_upper": acar_ci_upper,
        }

    # ------------------------------------------------------------------
    # Research-grade extensions
    # ------------------------------------------------------------------

    def multi_event_study_extended(
        self,
        prices: pd.Series,
        event_dates: list[str],
        market_returns: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Like multi_event_study but adds BH-corrected p-values and Cohen's d."""
        results = []
        seen_snapped_dates: set[pd.Timestamp] = set()

        for date in event_dates:
            result = self.single_event_study(prices, date, market_returns)
            if "error" not in result:
                snapped = result["event_date"]
                if snapped in seen_snapped_dates:
                    continue
                seen_snapped_dates.add(snapped)
                # Cohen's d: effect size = CAR / (estimation_std * sqrt(window_length))
                est_std = result.get("estimation_std", 0)
                event_window_length = self.pre_window + self.post_window + 1
                cohens_d = (
                    result["total_car"] / (est_std * np.sqrt(event_window_length))
                    if est_std > 0 else 0.0
                )
                results.append({
                    "event_date": result["event_date"],
                    "event_day_ar": result["event_day_ar"],
                    "total_car": result["total_car"],
                    "pre_event_car": result["pre_event_car"],
                    "post_event_car": result["post_event_car"],
                    "t_statistic": result["t_statistic"],
                    "p_value": result["p_value"],
                    "significant": result["significant"],
                    "cohens_d": cohens_d,
                })

        df = pd.DataFrame(results)
        if not df.empty and len(df) > 1:
            df["p_value_adjusted"] = adjust_pvalues(df["p_value"].values)
            df["significant_adjusted"] = df["p_value_adjusted"] < self.significance_level
            logger.info(
                f"Event study (extended): {df['significant'].sum()}/{len(df)} raw significant, "
                f"{df['significant_adjusted'].sum()}/{len(df)} after BH correction"
            )
        return df

    def placebo_test(
        self,
        prices: pd.Series,
        event_dates: list[str],
        n_placebos: int = 500,
        seed: int = 42,
    ) -> dict:
        """Placebo test: run event study on random non-event dates.

        If the real ACAR is outside the placebo distribution, the event
        effect is unlikely to be spurious.

        Returns:
            Dict with placebo_cars (distribution), real_acar, empirical_p_value.
        """
        returns = compute_returns(prices, method="simple")
        rng = np.random.RandomState(seed)

        # Real ACAR
        real = self.average_car(prices, event_dates)
        if "error" in real:
            return {"error": "no_valid_events"}
        real_acar = float(real["acar"].iloc[-1])

        # Build exclusion set: real event dates ± 20 days
        event_ts = pd.to_datetime(event_dates)
        exclude = set()
        for d in event_ts:
            for offset in range(-20, 21):
                exclude.add(d + pd.Timedelta(days=offset))

        # Available dates for placebo
        available = [d for d in returns.index if d not in exclude]
        if len(available) < n_placebos:
            n_placebos = len(available)

        # Run placebo event studies
        placebo_cars = []
        n_events_per_placebo = min(len(event_dates), 50)  # cap for speed

        for _ in range(n_placebos):
            placebo_dates = rng.choice(available, size=n_events_per_placebo, replace=False)
            placebo_dates = [str(d.date()) for d in placebo_dates]
            result = self.average_car(prices, placebo_dates)
            if "error" not in result:
                placebo_cars.append(float(result["acar"].iloc[-1]))

        if not placebo_cars:
            return {"error": "no_valid_placebos"}

        placebo_arr = np.array(placebo_cars)
        # Empirical p-value: fraction of placebos with |ACAR| >= |real|
        empirical_p = float(np.mean(np.abs(placebo_arr) >= abs(real_acar)))

        logger.info(
            f"Placebo test: real ACAR={real_acar:.6f}, "
            f"placebo mean={np.mean(placebo_arr):.6f}, "
            f"empirical p={empirical_p:.4f}"
        )

        return {
            "real_acar": real_acar,
            "placebo_mean": float(np.mean(placebo_arr)),
            "placebo_std": float(np.std(placebo_arr)),
            "placebo_5th": float(np.percentile(placebo_arr, 5)),
            "placebo_95th": float(np.percentile(placebo_arr, 95)),
            "empirical_p_value": empirical_p,
            "n_placebos": len(placebo_cars),
            "n_events_per_placebo": n_events_per_placebo,
        }

    def window_sensitivity(
        self,
        prices: pd.Series,
        event_dates: list[str],
        post_windows: list[int] | None = None,
        market_returns: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Test ACAR sensitivity to event window size.

        Runs average_car at different post-event windows and collects
        the final ACAR value + CI + p-value for each.
        """
        post_windows = post_windows or [5, 10, 15, 20, 30]
        original_post = self.post_window
        results = []

        for pw in post_windows:
            self.post_window = pw
            acar_result = self.average_car(prices, event_dates, market_returns)
            if "error" not in acar_result:
                final_acar = float(acar_result["acar"].iloc[-1])
                results.append({
                    "post_window": pw,
                    "acar": final_acar,
                    "acar_ci_lower": acar_result.get("acar_ci_lower", np.nan),
                    "acar_ci_upper": acar_result.get("acar_ci_upper", np.nan),
                    "n_events": acar_result["n_events"],
                })

        self.post_window = original_post  # restore
        df = pd.DataFrame(results)
        if not df.empty:
            logger.info(f"Window sensitivity:\n{df.to_string()}")
        return df
