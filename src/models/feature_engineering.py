"""
Feature engineering: combining DC features, LLM features, and market features.

This module builds the unified feature matrix that merges:
1. DC-based features (intrinsic time, magnitude, frequency)
2. LLM-based features (sentiment, event classification, impact estimates)
3. Traditional market features (returns, volatility, volume)
4. Cross-market features (correlation, spillover signals)
"""

import numpy as np
import pandas as pd
from loguru import logger

from src.directional_changes.dc_features import DCFeatureExtractor
from src.utils.helpers import compute_returns, rolling_volatility


class FeatureEngineer:
    """Builds the unified feature matrix for the hybrid prediction model."""

    def __init__(self, dc_threshold: float = 0.02):
        self.dc_threshold = dc_threshold
        self.dc_extractor = DCFeatureExtractor(dc_threshold)

    def build_market_features(
        self, prices: pd.Series, volume: pd.Series | None = None
    ) -> pd.DataFrame:
        """
        Build traditional market features from price/volume data.
        """
        returns = compute_returns(prices, method="simple")
        log_returns = compute_returns(prices, method="log")

        features = pd.DataFrame(index=prices.index)

        # Returns at different horizons
        features["return_1d"] = prices.pct_change(1)
        features["return_5d"] = prices.pct_change(5)
        features["return_20d"] = prices.pct_change(20)

        # Volatility
        features["volatility_5d"] = returns.rolling(5).std() * np.sqrt(252)
        features["volatility_20d"] = returns.rolling(20).std() * np.sqrt(252)
        features["volatility_60d"] = returns.rolling(60).std() * np.sqrt(252)

        # Volatility ratio (short/long) - increasing means stress
        features["vol_ratio"] = features["volatility_5d"] / features["volatility_60d"]

        # Momentum indicators
        features["sma_20"] = prices.rolling(20).mean()
        features["sma_50"] = prices.rolling(50).mean()
        features["price_to_sma20"] = prices / features["sma_20"]
        features["sma20_to_sma50"] = features["sma_20"] / features["sma_50"]

        # Drawdown from recent high
        rolling_max = prices.rolling(60).max()
        features["drawdown"] = (prices - rolling_max) / rolling_max

        # Volume features
        if volume is not None:
            features["volume_ratio"] = volume / volume.rolling(20).mean()
            features["volume_volatility"] = volume.rolling(20).std() / volume.rolling(20).mean()

        # Return distribution features
        features["skewness_20d"] = returns.rolling(20).skew()
        features["kurtosis_20d"] = returns.rolling(20).kurt()

        return features

    def build_dc_features(self, prices: pd.Series) -> pd.DataFrame:
        """Build features from Directional Changes analysis."""
        return self.dc_extractor.extract_features(prices, window=10)

    def build_cross_market_features(
        self,
        target_returns: pd.Series,
        other_market_returns: dict[str, pd.Series],
    ) -> pd.DataFrame:
        """
        Build features capturing cross-market dynamics.

        For the target market, compute features based on
        other markets' recent behavior.
        """
        features = pd.DataFrame(index=target_returns.index)

        for name, other_ret in other_market_returns.items():
            other_ret.index = other_ret.index.normalize()
            aligned = other_ret.reindex(target_returns.index)

            # Other market's recent return
            features[f"{name}_return_1d"] = aligned
            features[f"{name}_return_5d"] = aligned.rolling(5).sum()

            # Rolling correlation with target
            features[f"{name}_corr_20d"] = (
                target_returns.rolling(20).corr(aligned)
            )

            # Lead signal: other market's return as predictor
            features[f"{name}_lag1"] = aligned.shift(1)
            features[f"{name}_lag2"] = aligned.shift(2)

        return features

    def build_event_features(
        self,
        dates: pd.DatetimeIndex,
        events_df: pd.DataFrame,
        window_days: int = 30,
    ) -> pd.DataFrame:
        """
        Build features from the geopolitical event database.

        For each date, computes:
        - Days since last event
        - Recent event severity sum
        - Event type composition (% tariff vs conflict)
        """
        features = pd.DataFrame(index=dates)

        events_df = events_df.copy()
        events_df["date"] = pd.to_datetime(events_df["date"])

        for date in dates:
            # Events in lookback window
            window_start = date - pd.Timedelta(days=window_days)
            recent = events_df[
                (events_df["date"] >= window_start) & (events_df["date"] <= date)
            ]

            features.loc[date, "event_count_30d"] = len(recent)
            features.loc[date, "severity_sum_30d"] = recent["severity"].sum() if len(recent) > 0 else 0
            features.loc[date, "avg_severity_30d"] = recent["severity"].mean() if len(recent) > 0 else 0
            features.loc[date, "max_severity_30d"] = recent["severity"].max() if len(recent) > 0 else 0

            # Days since last event
            past_events = events_df[events_df["date"] <= date]
            if len(past_events) > 0:
                features.loc[date, "days_since_event"] = (date - past_events["date"].max()).days
            else:
                features.loc[date, "days_since_event"] = 999

            # Event type composition
            if len(recent) > 0 and "event_type" in recent.columns:
                features.loc[date, "tariff_event_ratio"] = (
                    (recent["event_type"] == "tariff").sum() / len(recent)
                )
                features.loc[date, "conflict_event_ratio"] = (
                    (recent["event_type"] == "conflict").sum() / len(recent)
                )
            else:
                features.loc[date, "tariff_event_ratio"] = 0
                features.loc[date, "conflict_event_ratio"] = 0

        return features

    def build_sentiment_features(
        self,
        dates: pd.DatetimeIndex,
        sentiment_df: pd.DataFrame,
        market_id: str = "us",
    ) -> pd.DataFrame:
        """Build features from LLM sentiment analysis."""
        features = pd.DataFrame(index=dates)

        if sentiment_df.empty:
            return features

        sentiment_df.index = pd.to_datetime(sentiment_df.index)

        col = f"{market_id}_sentiment_mean"
        if col in sentiment_df.columns:
            aligned = sentiment_df[col].reindex(dates).ffill()
            features["llm_sentiment"] = aligned
            features["llm_sentiment_5d_ma"] = aligned.rolling(5).mean()
            features["llm_sentiment_momentum"] = aligned - aligned.shift(5)
        else:
            features["llm_sentiment"] = 0
            features["llm_sentiment_5d_ma"] = 0
            features["llm_sentiment_momentum"] = 0

        return features

    def build_unified_features(
        self,
        prices: pd.Series,
        volume: pd.Series | None = None,
        other_market_returns: dict[str, pd.Series] | None = None,
        events_df: pd.DataFrame | None = None,
        sentiment_df: pd.DataFrame | None = None,
        market_id: str = "us",
    ) -> pd.DataFrame:
        """
        Build the complete unified feature matrix.

        This is the main entry point that combines all feature sources.
        """
        logger.info("Building unified feature matrix...")

        # Market features
        market_feat = self.build_market_features(prices, volume)

        # DC features
        dc_feat = self.build_dc_features(prices)

        # Combine
        features = market_feat.join(dc_feat, how="left")

        # Cross-market features
        if other_market_returns:
            target_returns = compute_returns(prices, method="simple")
            target_returns.index = target_returns.index.normalize()
            cross_feat = self.build_cross_market_features(
                target_returns, other_market_returns
            )
            features = features.join(cross_feat, how="left")

        # Event features
        if events_df is not None and not events_df.empty:
            event_feat = self.build_event_features(features.index, events_df)
            features = features.join(event_feat, how="left")

        # Sentiment features
        if sentiment_df is not None and not sentiment_df.empty:
            sent_feat = self.build_sentiment_features(
                features.index, sentiment_df, market_id
            )
            features = features.join(sent_feat, how="left")

        # Forward fill and drop initial NaN rows
        features = features.ffill().dropna(how="all")

        logger.info(f"Feature matrix: {features.shape[0]} rows x {features.shape[1]} columns")
        return features

    def build_labels(
        self,
        prices: pd.Series,
        horizon: int = 5,
        method: str = "direction",
    ) -> pd.Series:
        """
        Build prediction labels.

        Args:
            prices: Price series
            horizon: Forward-looking horizon in days
            method: 'direction' (fixed ±0.5% thresholds),
                     'direction_vol' (volatility-scaled thresholds),
                     'return' (continuous), or 'dc' (next DC direction)
        """
        if method == "direction":
            future_return = prices.pct_change(horizon).shift(-horizon)
            labels = pd.Series(1, index=prices.index)  # 1 = neutral
            labels[future_return > 0.005] = 2   # Up more than 0.5%
            labels[future_return < -0.005] = 0  # Down more than 0.5%
            return labels.rename("label")

        elif method == "direction_vol":
            # Volatility-scaled thresholds: neutral zone = 1σ × √horizon
            # In low-vol periods the threshold shrinks, creating more
            # directional labels. In high-vol periods it widens, filtering
            # out noise. This prevents persistence from being trivially
            # strong just because fixed thresholds are too narrow.
            daily_returns = prices.pct_change(1)
            daily_vol = daily_returns.rolling(20).std()
            threshold = daily_vol * np.sqrt(horizon)
            # Fallback for first 20 rows where vol is NaN; floor to prevent
            # near-zero thresholds in extremely calm periods
            threshold = threshold.fillna(0.005).clip(lower=0.001)

            future_return = prices.pct_change(horizon).shift(-horizon)
            labels = pd.Series(1, index=prices.index)  # 1 = neutral
            labels[future_return > threshold] = 2       # up
            labels[future_return < -threshold] = 0      # down
            return labels.rename("label")

        elif method == "return":
            return prices.pct_change(horizon).shift(-horizon).rename("label")

        elif method == "dc":
            return self.dc_extractor.compute_dc_based_labels(prices, horizon)

        else:
            raise ValueError(f"Unknown label method: {method}")
