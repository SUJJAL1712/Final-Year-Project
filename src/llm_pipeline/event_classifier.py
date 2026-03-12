"""
LLM-powered geopolitical event classifier.

Uses Claude to classify news articles into structured event categories
from the taxonomy. This transforms unstructured news text into
machine-readable event signals for the prediction model.
"""

import pandas as pd
from loguru import logger

from src.llm_pipeline.client import ClaudeClient
from src.llm_pipeline.prompts.classification import (
    SYSTEM_PROMPT,
    EVENT_CLASSIFICATION_PROMPT,
    BATCH_CLASSIFICATION_PROMPT,
)
from src.utils.helpers import chunk_list


class EventClassifier:
    """Classifies news articles into geopolitical event categories using Claude."""

    def __init__(self):
        self.client = ClaudeClient()

    def classify_article(self, title: str, content: str, date: str) -> dict:
        """
        Classify a single news article.

        Returns:
            Dict with event category, severity, affected markets, etc.
        """
        prompt = EVENT_CLASSIFICATION_PROMPT.format(
            title=title,
            date=date,
            content=content[:2000],  # Truncate to save tokens
        )

        result = self.client.query_json(prompt, system=SYSTEM_PROMPT)

        if "error" not in result:
            result["original_title"] = title
            result["date"] = date

        return result

    def classify_batch(self, articles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify a batch of articles efficiently.

        Args:
            articles_df: DataFrame with 'title', 'content', 'published_at' columns

        Returns:
            DataFrame with classification results
        """
        results = []

        # Process in chunks for efficiency
        chunks = chunk_list(articles_df.to_dict("records"), 10)

        for chunk_idx, chunk in enumerate(chunks):
            logger.info(f"Classifying batch {chunk_idx + 1}/{len(chunks)}")

            headlines = "\n".join(
                f"{i}. [{a.get('published_at', 'N/A')}] {a['title']}"
                for i, a in enumerate(chunk)
            )

            prompt = BATCH_CLASSIFICATION_PROMPT.format(
                count=len(chunk),
                headlines=headlines,
            )

            batch_result = self.client.query_json(prompt, system=SYSTEM_PROMPT)

            if isinstance(batch_result, list):
                for item in batch_result:
                    idx = item.get("headline_index", 0)
                    if idx < len(chunk):
                        item["title"] = chunk[idx]["title"]
                        item["date"] = chunk[idx].get("published_at", "")
                    results.append(item)
            else:
                logger.warning(f"Unexpected batch result format: {type(batch_result)}")

        return pd.DataFrame(results)

    def classify_known_events(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich known curated events with LLM-generated analysis.

        Takes the curated event database and adds LLM insights about
        expected market impact, affected sectors, and cross-market dynamics.
        """
        enriched = []

        for _, event in events_df.iterrows():
            logger.info(f"Enriching event: {event['event'][:60]}...")

            result = self.classify_article(
                title=event["event"],
                content=event["event"],
                date=str(event["date"]),
            )

            # Merge original data with LLM classification
            merged = {**event.to_dict(), **result}
            merged["llm_classified"] = True
            enriched.append(merged)

        return pd.DataFrame(enriched)

    def is_geopolitical_event(self, title: str, content: str = "") -> bool:
        """Quick check if an article describes a geopolitical event."""
        keywords = [
            "tariff", "trade war", "sanctions", "conflict", "military",
            "invasion", "ceasefire", "export ban", "import duty",
            "trade deal", "border clash", "diplomatic",
        ]
        text = f"{title} {content}".lower()
        return any(kw in text for kw in keywords)
