"""
LLM-powered financial sentiment analysis.

Unlike generic sentiment analysis, this module produces market-specific
sentiment scores. The same event can be bullish for one market and
bearish for another (e.g., US tariffs on China may benefit India).
"""

import pandas as pd
import numpy as np
from loguru import logger

from src.llm_pipeline.client import ClaudeClient
from src.llm_pipeline.prompts.sentiment import (
    SENTIMENT_SYSTEM_PROMPT,
    SENTIMENT_ANALYSIS_PROMPT,
    BATCH_SENTIMENT_PROMPT,
)
from src.utils.helpers import chunk_list


class FinancialSentimentAnalyzer:
    """
    Produces market-specific sentiment scores for geopolitical news.

    Key innovation: Instead of a single sentiment score, this produces
    separate scores for US, India, and China markets, capturing the
    asymmetric impact of events across the triangle.
    """

    def __init__(self):
        self.client = ClaudeClient()

    def analyze_article(self, title: str, content: str, date: str) -> dict:
        """
        Analyze sentiment of a single article for each market.

        Returns:
            Dict with flat per-market sentiment columns:
            us_sentiment, india_sentiment, china_sentiment
            plus overall_sentiment, volatility_expectation, confidence
        """
        prompt = SENTIMENT_ANALYSIS_PROMPT.format(
            title=title,
            date=date,
            content=content[:2000],
        )

        result = self.client.query_json(prompt, system=SENTIMENT_SYSTEM_PROMPT)

        # Flatten nested market_specific_sentiment into top-level columns
        # so compute_daily_sentiment can find us_sentiment, india_sentiment, etc.
        mss = result.get("market_specific_sentiment", {})
        if isinstance(mss, dict):
            for market_key, data in mss.items():
                flat_key = market_key.lower().replace(" ", "_") + "_sentiment"
                if isinstance(data, dict):
                    result[flat_key] = data.get("sentiment", 0.0)
                elif isinstance(data, (int, float)):
                    result[flat_key] = data
            # Remove nested dict to keep DataFrame columns clean
            result.pop("market_specific_sentiment", None)

        result["title"] = title
        result["date"] = date
        return result

    def analyze_articles_individually(
        self, articles_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Analyze each article individually (higher quality than batch).

        Use this when you need detailed per-article sentiment with reasoning.
        Falls back to batch mode if article count is high.
        """
        if len(articles_df) > 50:
            logger.info(
                f"{len(articles_df)} articles — switching to batch mode for efficiency"
            )
            return self.analyze_batch(articles_df)

        results = []
        for i, (_, row) in enumerate(articles_df.iterrows()):
            logger.info(f"Analyzing article {i + 1}/{len(articles_df)}")
            result = self.analyze_article(
                title=row.get("title", ""),
                content=row.get("content", row.get("title", "")),
                date=str(row.get("published_at", row.get("date", ""))),
            )
            results.append(result)

        return pd.DataFrame(results)

    def analyze_batch(self, articles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze sentiment for a batch of articles.

        Returns DataFrame with per-market sentiment columns.
        """
        results = []
        chunks = chunk_list(articles_df.to_dict("records"), 15)

        for chunk_idx, chunk in enumerate(chunks):
            logger.info(f"Sentiment batch {chunk_idx + 1}/{len(chunks)}")

            headlines = "\n".join(
                f"{i}. [{a.get('published_at', 'N/A')}] {a['title']}"
                for i, a in enumerate(chunk)
            )

            prompt = BATCH_SENTIMENT_PROMPT.format(
                count=len(chunk),
                headlines=headlines,
            )

            batch_result = self.client.query_json(prompt, system=SENTIMENT_SYSTEM_PROMPT)

            if isinstance(batch_result, list):
                for item in batch_result:
                    idx = item.get("index", 0)
                    if idx < len(chunk):
                        item["title"] = chunk[idx]["title"]
                        item["date"] = chunk[idx].get("published_at", "")
                    results.append(item)
            else:
                logger.warning(
                    f"LLM batch sentiment returned non-list (error?): {str(batch_result)[:200]}"
                )

        return pd.DataFrame(results)

    def compute_daily_sentiment(
        self, classified_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Aggregate article-level sentiment into daily market sentiment scores.

        Produces a daily time series of sentiment for each market,
        weighted by article confidence and severity.
        """
        if classified_df.empty:
            return pd.DataFrame()

        # Ensure date column
        if "date" in classified_df.columns:
            classified_df["date"] = pd.to_datetime(classified_df["date"], errors="coerce")
        else:
            return pd.DataFrame()

        classified_df = classified_df.dropna(subset=["date"])

        daily_records = []
        for date, group in classified_df.groupby(classified_df["date"].dt.date):
            record = {"date": pd.Timestamp(date)}

            for market in ["us", "india", "china"]:
                col = f"{market}_sentiment"
                if col in group.columns:
                    values = pd.to_numeric(group[col], errors="coerce").dropna()
                    record[f"{market}_sentiment_mean"] = values.mean() if len(values) > 0 else 0.0
                    record[f"{market}_sentiment_std"] = values.std() if len(values) > 1 else 0.0
                    record[f"{market}_article_count"] = len(values)
                else:
                    record[f"{market}_sentiment_mean"] = 0.0
                    record[f"{market}_sentiment_std"] = 0.0
                    record[f"{market}_article_count"] = 0

            daily_records.append(record)

        df = pd.DataFrame(daily_records).set_index("date").sort_index()
        return df

    def compute_sentiment_divergence(
        self, daily_sentiment: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute sentiment divergence between market pairs.

        When US-China sentiment diverges sharply (US positive, China negative),
        it signals a tariff/trade war event. When India diverges from both,
        it may signal trade diversion opportunities.
        """
        div = pd.DataFrame(index=daily_sentiment.index)

        if all(c in daily_sentiment.columns for c in ["us_sentiment_mean", "china_sentiment_mean"]):
            div["us_china_divergence"] = (
                daily_sentiment["us_sentiment_mean"] - daily_sentiment["china_sentiment_mean"]
            )

        if all(c in daily_sentiment.columns for c in ["us_sentiment_mean", "india_sentiment_mean"]):
            div["us_india_divergence"] = (
                daily_sentiment["us_sentiment_mean"] - daily_sentiment["india_sentiment_mean"]
            )

        if all(c in daily_sentiment.columns for c in ["india_sentiment_mean", "china_sentiment_mean"]):
            div["india_china_divergence"] = (
                daily_sentiment["india_sentiment_mean"] - daily_sentiment["china_sentiment_mean"]
            )

        # India as beneficiary signal: positive when India benefits from US-China conflict
        if "us_china_divergence" in div.columns and "india_sentiment_mean" in daily_sentiment.columns:
            div["india_beneficiary_signal"] = (
                daily_sentiment["india_sentiment_mean"]
                * abs(div["us_china_divergence"])
            )

        return div
