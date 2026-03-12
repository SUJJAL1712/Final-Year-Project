"""
Granger Causality analysis between geopolitical signals and market movements.

Tests whether LLM-derived geopolitical signals (sentiment, event severity)
Granger-cause stock market movements, and vice versa. Also tests
cross-market Granger causality to establish lead-lag relationships
between US, India, and China markets.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.api import VAR
from loguru import logger

from src.utils.config import get_config


class GrangerCausalityAnalyzer:
    """
    Tests Granger causality between:
    1. Geopolitical signals → Market returns
    2. Market A returns → Market B returns
    3. DC features → Market returns
    """

    def __init__(self):
        config = get_config().analysis_config
        self.max_lags = config["granger_max_lags"]
        self.significance_level = config["significance_level"]

    def test_stationarity(self, series: pd.Series) -> dict:
        """Augmented Dickey-Fuller test for stationarity."""
        series = series.dropna()
        if len(series) < 20:
            return {"stationary": False, "error": "insufficient_data"}

        result = adfuller(series, autolag="AIC")
        return {
            "adf_statistic": result[0],
            "p_value": result[1],
            "lags_used": result[2],
            "stationary": result[1] < self.significance_level,
        }

    def pairwise_granger_test(
        self,
        cause: pd.Series,
        effect: pd.Series,
        max_lags: int | None = None,
    ) -> dict:
        """
        Test if 'cause' Granger-causes 'effect'.

        Returns results for each lag, plus the best lag.
        """
        max_lags = max_lags or self.max_lags

        # Align and prepare data
        df = pd.DataFrame({"cause": cause, "effect": effect}).dropna()

        if len(df) < max_lags * 3:
            return {"error": "insufficient_data", "n_obs": len(df)}

        # Check stationarity
        for col in ["cause", "effect"]:
            stat = self.test_stationarity(df[col])
            if not stat.get("stationary", False):
                logger.info(f"{col} is non-stationary, differencing")
                df[col] = df[col].diff().dropna()
                df = df.dropna()

        try:
            # Granger test (effect is the dependent variable)
            results = grangercausalitytests(
                df[["effect", "cause"]], maxlag=max_lags, verbose=False
            )

            lag_results = {}
            best_lag = None
            best_p_value = 1.0

            for lag in range(1, max_lags + 1):
                if lag in results:
                    tests = results[lag][0]
                    f_stat = tests["ssr_ftest"][0]
                    p_value = tests["ssr_ftest"][1]
                    lag_results[lag] = {
                        "f_statistic": f_stat,
                        "p_value": p_value,
                        "significant": p_value < self.significance_level,
                    }
                    if p_value < best_p_value:
                        best_p_value = p_value
                        best_lag = lag

            return {
                "granger_causes": best_p_value < self.significance_level,
                "best_lag": best_lag,
                "best_p_value": best_p_value,
                "lag_results": lag_results,
                "n_observations": len(df),
            }

        except Exception as e:
            logger.error(f"Granger test failed: {e}")
            return {"error": str(e)}

    def cross_market_granger(
        self, market_returns: dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        Test Granger causality between all market pairs.

        Key questions:
        - Does US market Granger-cause India/China? (likely yes)
        - Does China Granger-cause India? (competitive effect)
        - Does India Granger-cause anything? (emerging market signal)
        """
        results = []

        markets = list(market_returns.keys())
        for cause_market in markets:
            for effect_market in markets:
                if cause_market == effect_market:
                    continue

                logger.info(f"Testing: {cause_market} → {effect_market}")
                result = self.pairwise_granger_test(
                    market_returns[cause_market],
                    market_returns[effect_market],
                )

                results.append({
                    "cause": cause_market,
                    "effect": effect_market,
                    "granger_causes": result.get("granger_causes", False),
                    "best_lag": result.get("best_lag"),
                    "best_p_value": result.get("best_p_value"),
                    "n_observations": result.get("n_observations"),
                })

        return pd.DataFrame(results)

    def geopolitical_signal_granger(
        self,
        signal: pd.Series,
        market_returns: dict[str, pd.Series],
        signal_name: str = "geopolitical_signal",
    ) -> pd.DataFrame:
        """
        Test if a geopolitical signal (e.g., LLM sentiment, event severity)
        Granger-causes market returns.
        """
        results = []

        for market_name, returns in market_returns.items():
            logger.info(f"Testing: {signal_name} → {market_name}")
            result = self.pairwise_granger_test(signal, returns)

            results.append({
                "signal": signal_name,
                "market": market_name,
                "granger_causes": result.get("granger_causes", False),
                "best_lag": result.get("best_lag"),
                "best_p_value": result.get("best_p_value"),
            })

            # Also test reverse: market → signal (feedback effect)
            reverse = self.pairwise_granger_test(returns, signal)
            results.append({
                "signal": market_name,
                "market": signal_name,
                "granger_causes": reverse.get("granger_causes", False),
                "best_lag": reverse.get("best_lag"),
                "best_p_value": reverse.get("best_p_value"),
            })

        return pd.DataFrame(results)

    def var_impulse_response(
        self,
        market_returns: dict[str, pd.Series],
        periods: int = 20,
        lags: int = 5,
    ) -> dict:
        """
        Compute impulse response functions using Vector Autoregression.

        Shows how a shock in one market propagates to others over time.
        """
        df = pd.DataFrame(market_returns).dropna()

        if len(df) < lags * 5:
            return {"error": "insufficient_data"}

        try:
            model = VAR(df)
            fitted = model.fit(lags)
            irf = fitted.irf(periods)

            return {
                "irf_data": irf.irfs,
                "market_names": list(df.columns),
                "periods": periods,
                "aic": fitted.aic,
                "bic": fitted.bic,
                "summary": str(fitted.summary()),
            }

        except Exception as e:
            logger.error(f"VAR model failed: {e}")
            return {"error": str(e)}
