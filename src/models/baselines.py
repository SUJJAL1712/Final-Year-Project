"""
Baseline models for comparison.

Any prediction model's value must be demonstrated against baselines.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from loguru import logger


class BaselineModels:
    """Simple baselines to benchmark the hybrid model against."""

    @staticmethod
    def random_baseline(labels: pd.Series, n_trials: int = 100) -> dict:
        """Random prediction baseline."""
        classes = labels.unique()
        accuracies = []
        for _ in range(n_trials):
            random_preds = np.random.choice(classes, size=len(labels))
            accuracies.append(accuracy_score(labels, random_preds))

        return {
            "model": "random",
            "mean_accuracy": np.mean(accuracies),
            "std_accuracy": np.std(accuracies),
        }

    @staticmethod
    def majority_class_baseline(labels: pd.Series) -> dict:
        """Always predict the most common class."""
        majority = labels.mode()[0]
        preds = pd.Series(majority, index=labels.index)
        return {
            "model": "majority_class",
            "mean_accuracy": accuracy_score(labels, preds),
            "std_accuracy": 0.0,
            "majority_class": majority,
            "majority_pct": (labels == majority).mean(),
        }

    @staticmethod
    def momentum_baseline(returns: pd.Series, labels: pd.Series, lookback: int = 5) -> dict:
        """
        Momentum baseline: predict direction based on recent return trend.
        If last N days were positive, predict up; if negative, predict down.
        """
        momentum = returns.rolling(lookback).sum()

        preds = pd.Series(0, index=labels.index)
        preds[momentum > 0.005] = 1
        preds[momentum < -0.005] = -1

        common = preds.index.intersection(labels.index)
        preds = preds[common]
        y = labels[common]

        return {
            "model": f"momentum_{lookback}d",
            "mean_accuracy": accuracy_score(y, preds),
            "mean_f1": f1_score(y, preds, average="weighted", zero_division=0),
        }

    @staticmethod
    def mean_reversion_baseline(returns: pd.Series, labels: pd.Series, lookback: int = 5) -> dict:
        """
        Mean reversion baseline: predict opposite of recent trend.
        """
        momentum = returns.rolling(lookback).sum()

        preds = pd.Series(0, index=labels.index)
        preds[momentum > 0.005] = -1  # Predict reversal
        preds[momentum < -0.005] = 1

        common = preds.index.intersection(labels.index)
        preds = preds[common]
        y = labels[common]

        return {
            "model": f"mean_reversion_{lookback}d",
            "mean_accuracy": accuracy_score(y, preds),
            "mean_f1": f1_score(y, preds, average="weighted", zero_division=0),
        }

    @staticmethod
    def persistence_baseline(labels: pd.Series) -> dict:
        """Persistence baseline: predict yesterday's direction."""
        preds = labels.shift(1).dropna()
        y = labels.loc[preds.index]
        return {
            "model": "persistence",
            "mean_accuracy": accuracy_score(y, preds),
            "mean_f1": f1_score(y, preds, average="weighted", zero_division=0),
        }

    @staticmethod
    def buy_and_hold_baseline(labels: pd.Series) -> dict:
        """Buy-and-hold baseline: always predict +1 (up)."""
        preds = pd.Series(1, index=labels.index)
        return {
            "model": "buy_and_hold",
            "mean_accuracy": accuracy_score(labels, preds),
            "mean_f1": f1_score(labels, preds, average="weighted", zero_division=0),
        }

    @staticmethod
    def bootstrap_ci(
        labels: pd.Series, preds: pd.Series, n_boot: int = 1000, seed: int = 42
    ) -> tuple[float, float]:
        """Bootstrap 95% CI on accuracy."""
        rng = np.random.RandomState(seed)
        accs = []
        n = len(labels)
        for _ in range(n_boot):
            idx = rng.randint(0, n, size=n)
            accs.append(accuracy_score(labels.iloc[idx], preds.iloc[idx]))
        return (float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5)))

    def run_all_baselines(
        self, prices: pd.Series, labels: pd.Series
    ) -> pd.DataFrame:
        """Run all baselines and return comparison table."""
        returns = prices.pct_change().dropna()

        results = [
            self.random_baseline(labels),
            self.majority_class_baseline(labels),
            self.momentum_baseline(returns, labels, 5),
            self.momentum_baseline(returns, labels, 20),
            self.mean_reversion_baseline(returns, labels, 5),
            self.persistence_baseline(labels),
            self.buy_and_hold_baseline(labels),
        ]

        df = pd.DataFrame(results)
        logger.info(f"\nBaseline Results:\n{df.to_string()}")
        return df
