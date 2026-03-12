"""Prompt templates for market impact assessment."""

IMPACT_SYSTEM_PROMPT = """You are a quantitative analyst specializing in event-driven
market impact assessment. You estimate the likely magnitude and timing of market
reactions to geopolitical events, with specific expertise in the US-India-China
economic triangle. Always respond in valid JSON."""


IMPACT_ASSESSMENT_PROMPT = """Assess the potential market impact of the following geopolitical event.

EVENT:
Date: {date}
Description: {event}
Category: {category}
Severity: {severity}/10

CONTEXT:
Recent market conditions: {market_context}
Related recent events: {recent_events}

Provide a detailed impact assessment:
{{
    "impact_summary": "<one paragraph assessment>",
    "magnitude_estimate": {{
        "US_market_impact_pct": <estimated % move, negative for decline>,
        "India_market_impact_pct": <estimated % move>,
        "China_market_impact_pct": <estimated % move>
    }},
    "timing": {{
        "immediate_reaction_days": <1-3>,
        "full_impact_days": <5-30>,
        "recovery_expected_days": <10-90 or null if permanent shift>
    }},
    "transmission_channels": [
        {{
            "channel": "<e.g., trade_flow, sentiment_contagion, supply_chain>",
            "from_market": "<source market>",
            "to_market": "<target market>",
            "lag_days": <estimated lag>,
            "strength": "<strong|moderate|weak>"
        }}
    ],
    "sector_impacts": [
        {{
            "sector": "<sector name>",
            "direction": "<positive|negative>",
            "magnitude": "<large|moderate|small>",
            "reasoning": "<brief>"
        }}
    ],
    "india_specific_analysis": {{
        "trade_diversion_benefit": <true|false>,
        "supply_chain_opportunity": "<description or null>",
        "net_impact": "<positive|negative|neutral>",
        "reasoning": "<brief>"
    }},
    "confidence": <0.0-1.0>,
    "key_risks": ["<risk1>", "<risk2>"]
}}"""


DC_EVENT_ATTRIBUTION_PROMPT = """A significant Directional Change (DC) event was detected in the market.

DC EVENT DETAILS:
Market: {market}
Symbol: {symbol}
Date: {date}
Direction: {direction} (price moved {direction_word})
Magnitude: {magnitude:.2%}
Duration: {duration_days:.1f} days
Threshold: {threshold:.1%}

NEWS HEADLINES FROM THE SURROUNDING PERIOD ({window_start} to {window_end}):
{headlines}

Analyze what likely caused this directional change. Respond with:
{{
    "most_likely_cause": "<brief description>",
    "cause_category": "<tariff|conflict|monetary_policy|earnings|other>",
    "related_headlines": [<indices of most relevant headlines>],
    "geopolitical_component_pct": <0-100, how much of this move is geopolitically driven>,
    "confidence": <0.0-1.0>,
    "cross_market_implications": {{
        "US": "<expected reaction if not the source>",
        "India": "<expected reaction>",
        "China": "<expected reaction>"
    }}
}}"""
