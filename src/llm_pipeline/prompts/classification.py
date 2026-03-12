"""Prompt templates for event classification."""

SYSTEM_PROMPT = """You are an expert geopolitical and financial analyst specializing in
the impact of tariffs, trade wars, and regional conflicts on global stock markets.
You analyze events with specific focus on the US-India-China triangle.
Always respond in valid JSON format."""


EVENT_CLASSIFICATION_PROMPT = """Analyze the following news article and classify the geopolitical event.

ARTICLE:
Title: {title}
Date: {date}
Content: {content}

Classify this event using the following taxonomy:

TARIFF EVENTS: tariff_imposition, tariff_increase, tariff_reduction, tariff_removal,
retaliatory_tariff, tariff_threat, tariff_negotiation

TRADE WAR EVENTS: trade_war_escalation, trade_war_deescalation, trade_deal_signed,
trade_deal_collapsed, export_ban, sanctions_imposed, sanctions_lifted

TECH WAR EVENTS: chip_export_restriction, entity_list_addition, tech_investment_ban,
data_regulation

MILITARY CONFLICT EVENTS: conflict_outbreak, conflict_escalation, conflict_deescalation,
ceasefire, military_buildup, border_skirmish, military_action

GEOPOLITICAL TENSION: diplomatic_breakdown, diplomatic_breakthrough, alliance_formation,
territorial_dispute

SUPPLY CHAIN: shipping_disruption, supply_chain_diversification, critical_resource_restriction

Respond with this JSON structure:
{{
    "event_category": "<main category>",
    "event_subcategory": "<specific subcategory from above>",
    "severity": <1-10 integer>,
    "countries_involved": ["<country1>", "<country2>"],
    "source_country": "<country initiating action>",
    "target_country": "<country targeted by action>",
    "affected_markets": ["US", "India", "China"],
    "affected_sectors": ["<sector1>", "<sector2>"],
    "expected_market_direction": "<positive|negative|mixed|uncertain>",
    "confidence": <0.0-1.0>,
    "summary": "<one sentence summary of the event's market relevance>"
}}"""


BATCH_CLASSIFICATION_PROMPT = """Analyze the following {count} news headlines and classify each one.

HEADLINES:
{headlines}

For each headline, provide a classification. Respond with a JSON array:
[
    {{
        "headline_index": 0,
        "event_subcategory": "<category>",
        "severity": <1-10>,
        "countries_involved": ["<country>"],
        "expected_market_direction": "<positive|negative|mixed|uncertain>",
        "confidence": <0.0-1.0>
    }},
    ...
]

Only classify headlines that are clearly about tariffs, trade conflicts, military conflicts,
or geopolitical tensions. For irrelevant headlines, set category to "not_relevant"."""
