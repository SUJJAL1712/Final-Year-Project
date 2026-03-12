"""
LLM-powered market impact assessment.

Uses Claude to estimate the likely market impact of geopolitical events,
including magnitude, timing, and cross-market transmission channels.
This bridges the LLM analysis with the DC-based quantitative framework.
"""

import pandas as pd
from loguru import logger

from src.llm_pipeline.client import ClaudeClient
from src.llm_pipeline.prompts.impact import (
    IMPACT_SYSTEM_PROMPT,
    IMPACT_ASSESSMENT_PROMPT,
    DC_EVENT_ATTRIBUTION_PROMPT,
)


class MarketImpactAssessor:
    """
    Assesses market impact of geopolitical events using Claude.

    Two key capabilities:
    1. Forward-looking: Given an event, predict market impact
    2. Backward-looking: Given a DC event, attribute it to a cause
    """

    def __init__(self):
        self.client = ClaudeClient()

    def assess_event_impact(
        self,
        event: dict,
        market_context: str = "",
        recent_events: str = "",
    ) -> dict:
        """
        Predict the market impact of a geopolitical event.

        Args:
            event: Dict with 'date', 'event', 'category', 'severity'
            market_context: Description of recent market conditions
            recent_events: Description of other recent events for context

        Returns:
            Structured impact assessment
        """
        prompt = IMPACT_ASSESSMENT_PROMPT.format(
            date=event.get("date", "Unknown"),
            event=event.get("event", "Unknown event"),
            category=event.get("category", "Unknown"),
            severity=event.get("severity", 5),
            market_context=market_context or "No specific context provided",
            recent_events=recent_events or "No recent events noted",
        )

        result = self.client.query_json(prompt, system=IMPACT_SYSTEM_PROMPT)
        result["event_date"] = event.get("date")
        result["event_description"] = event.get("event")
        return result

    def assess_all_events(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Assess impact for all events in a DataFrame."""
        assessments = []

        for i, (_, event) in enumerate(events_df.iterrows()):
            logger.info(f"Assessing event {i+1}/{len(events_df)}: {event['event'][:50]}...")

            # Provide context from nearby events
            event_date = pd.Timestamp(event["date"])
            nearby = events_df[
                (events_df["date"] >= event_date - pd.Timedelta(days=30))
                & (events_df["date"] < event_date)
            ]
            recent_str = "; ".join(nearby["event"].tolist()[-3:]) if len(nearby) > 0 else ""

            assessment = self.assess_event_impact(
                event.to_dict(),
                recent_events=recent_str,
            )
            assessments.append(assessment)

        return pd.DataFrame(assessments)

    def attribute_dc_event(
        self,
        market: str,
        symbol: str,
        date: str,
        direction: str,
        magnitude: float,
        duration_days: float,
        threshold: float,
        headlines: list[str],
        window_start: str = "",
        window_end: str = "",
    ) -> dict:
        """
        Attribute a Directional Change event to its likely geopolitical cause.

        This is the key bridge between DC detection and LLM analysis:
        when we detect a significant market move via DC, we ask Claude
        to analyze surrounding news and determine whether it was caused
        by a tariff/conflict event.

        Args:
            market: Market identifier (US, India, China)
            symbol: Ticker symbol
            date: Date of DC confirmation
            direction: UPTURN or DOWNTURN
            magnitude: DC magnitude (e.g., 0.05 for 5%)
            duration_days: Duration of the DC move
            threshold: DC threshold used
            headlines: News headlines from the surrounding period
            window_start: Start of news window
            window_end: End of news window
        """
        headlines_str = "\n".join(
            f"{i}. {h}" for i, h in enumerate(headlines)
        )

        direction_word = "upward" if direction == "UPTURN" else "downward"

        prompt = DC_EVENT_ATTRIBUTION_PROMPT.format(
            market=market,
            symbol=symbol,
            date=date,
            direction=direction,
            direction_word=direction_word,
            magnitude=magnitude,
            duration_days=duration_days,
            threshold=threshold,
            headlines=headlines_str or "No headlines available",
            window_start=window_start or "N/A",
            window_end=window_end or "N/A",
        )

        return self.client.query_json(prompt, system=IMPACT_SYSTEM_PROMPT)

    def build_impact_features(self, assessments_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract numerical features from impact assessments for the ML model.

        Converts the structured LLM output into a feature matrix
        that can be combined with DC features.
        """
        features = []

        for _, row in assessments_df.iterrows():
            feat = {"date": row.get("event_date")}

            # Market impact estimates
            mag = row.get("magnitude_estimate", {})
            if isinstance(mag, dict):
                feat["llm_us_impact"] = mag.get("US_market_impact_pct", 0)
                feat["llm_india_impact"] = mag.get("India_market_impact_pct", 0)
                feat["llm_china_impact"] = mag.get("China_market_impact_pct", 0)
            else:
                feat["llm_us_impact"] = 0
                feat["llm_india_impact"] = 0
                feat["llm_china_impact"] = 0

            # Timing estimates
            timing = row.get("timing", {})
            if isinstance(timing, dict):
                feat["llm_reaction_days"] = timing.get("immediate_reaction_days", 2)
                feat["llm_full_impact_days"] = timing.get("full_impact_days", 10)
            else:
                feat["llm_reaction_days"] = 2
                feat["llm_full_impact_days"] = 10

            # Confidence
            feat["llm_confidence"] = row.get("confidence", 0.5)

            # India beneficiary flag
            india_analysis = row.get("india_specific_analysis", {})
            if isinstance(india_analysis, dict):
                feat["india_trade_diversion"] = 1 if india_analysis.get("trade_diversion_benefit") else 0
                net = india_analysis.get("net_impact", "neutral")
                feat["india_net_impact"] = {"positive": 1, "negative": -1, "neutral": 0}.get(net, 0)
            else:
                feat["india_trade_diversion"] = 0
                feat["india_net_impact"] = 0

            # Transmission channel count
            channels = row.get("transmission_channels", [])
            feat["num_transmission_channels"] = len(channels) if isinstance(channels, list) else 0

            features.append(feat)

        return pd.DataFrame(features)
