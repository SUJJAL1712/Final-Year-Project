"""Shared utility functions."""

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import numpy as np


def trading_days_between(start: str, end: str) -> int:
    """Count trading days between two dates (approximate, excludes weekends)."""
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    return len(pd.bdate_range(start_dt, end_dt))


def normalize_series(series: pd.Series) -> pd.Series:
    """Min-max normalize a series to [0, 1]."""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(0.5, index=series.index)
    return (series - min_val) / (max_val - min_val)


def compute_returns(prices: pd.Series, method: str = "log") -> pd.Series:
    """Compute returns from a price series.

    Args:
        prices: Price series
        method: 'log' for log returns, 'simple' for arithmetic returns
    """
    if method == "log":
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()


def rolling_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """Compute rolling annualized volatility."""
    return returns.rolling(window).std() * np.sqrt(252)


def align_timeseries(*series: pd.Series) -> list[pd.Series]:
    """Align multiple time series to their common date range."""
    common_index = series[0].index
    for s in series[1:]:
        common_index = common_index.intersection(s.index)
    return [s.reindex(common_index) for s in series]


def chunk_list(lst: list, chunk_size: int) -> list[list]:
    """Split a list into chunks of given size."""
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def adjust_pvalues(
    p_values: list[float] | np.ndarray,
    method: str = "fdr_bh",
    alpha: float = 0.05,
) -> np.ndarray:
    """Apply multiple hypothesis correction to a list of p-values.

    Args:
        p_values: Raw p-values from multiple tests.
        method: Correction method — 'fdr_bh' (Benjamini-Hochberg, default),
                'bonferroni', 'holm', 'fdr_by', etc.
        alpha: Family-wise error rate.

    Returns:
        Array of adjusted p-values (same length as input).
    """
    from statsmodels.stats.multitest import multipletests

    p_arr = np.asarray(p_values, dtype=float)
    # Handle edge cases (NaN, empty)
    if len(p_arr) == 0:
        return p_arr
    valid = ~np.isnan(p_arr)
    if valid.sum() == 0:
        return p_arr

    adjusted = np.full_like(p_arr, np.nan)
    _, adj, _, _ = multipletests(p_arr[valid], alpha=alpha, method=method)
    adjusted[valid] = adj
    return adjusted
