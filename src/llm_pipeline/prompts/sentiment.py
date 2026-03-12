"""Prompt templates for financial sentiment analysis."""

SENTIMENT_SYSTEM_PROMPT = """You are a financial sentiment analyst specializing in
geopolitical risk assessment. You evaluate how news events will affect market
sentiment across different countries and sectors. Always respond in valid JSON."""


SENTIMENT_ANALYSIS_PROMPT = """Analyze the financial market sentiment of the following news article.

ARTICLE:
Title: {title}
Date: {date}
Content: {content}

Evaluate sentiment from the perspective of each affected market.

Respond with this JSON structure:
{{
    "overall_sentiment": <float from -1.0 (very bearish) to +1.0 (very bullish)>,
    "market_specific_sentiment": {{
        "US": {{
            "sentiment": <-1.0 to 1.0>,
            "reasoning": "<brief explanation>"
        }},
        "India": {{
            "sentiment": <-1.0 to 1.0>,
            "reasoning": "<brief explanation>"
        }},
        "China": {{
            "sentiment": <-1.0 to 1.0>,
            "reasoning": "<brief explanation>"
        }}
    }},
    "sector_sentiment": {{
        "<sector_name>": <-1.0 to 1.0>
    }},
    "volatility_expectation": "<low|medium|high|extreme>",
    "duration_of_impact": "<short_term|medium_term|long_term>",
    "confidence": <0.0 to 1.0>
}}

Consider:
1. Direct impact on mentioned countries/sectors
2. Second-order effects (trade diversion, supply chain shifts)
3. Whether India might benefit from US-China tensions (trade diversion)
4. Market expectations vs. actual news (surprise factor)"""


BATCH_SENTIMENT_PROMPT = """Rate the financial sentiment of these {count} headlines on a scale
from -1.0 (very bearish) to +1.0 (very bullish) for each market.

HEADLINES:
{headlines}

Respond with a JSON array:
[
    {{
        "index": 0,
        "us_sentiment": <-1.0 to 1.0>,
        "india_sentiment": <-1.0 to 1.0>,
        "china_sentiment": <-1.0 to 1.0>,
        "volatility": "<low|medium|high>"
    }},
    ...
]"""
