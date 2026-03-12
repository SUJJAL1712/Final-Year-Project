"""
Event timeline visualization.

Creates interactive timelines showing geopolitical events alongside
market price movements.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class EventTimelineVisualizer:
    """Visualizes geopolitical events on market price timelines."""

    EVENT_COLORS = {
        "tariff": "#ff9800",
        "conflict": "#f44336",
        "trade_deal_signed": "#4caf50",
        "sanctions": "#9c27b0",
        "tech_war": "#2196f3",
        "default": "#607d8b",
    }

    def plot_event_timeline(
        self,
        market_prices: dict[str, pd.Series],
        events_df: pd.DataFrame,
        title: str = "Geopolitical Events & Market Impact",
    ) -> go.Figure:
        """
        Multi-panel timeline: each market's price with event annotations.
        """
        n_markets = len(market_prices)
        fig = make_subplots(
            rows=n_markets, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=list(market_prices.keys()),
        )

        colors = ["#1976d2", "#e65100", "#2e7d32"]

        for i, (market_name, prices) in enumerate(market_prices.items()):
            # Normalize to 100
            normalized = (prices / prices.iloc[0]) * 100

            fig.add_trace(
                go.Scatter(
                    x=normalized.index, y=normalized.values,
                    name=market_name,
                    line=dict(color=colors[i % len(colors)], width=2),
                ),
                row=i + 1, col=1,
            )

            # Add event markers
            for _, event in events_df.iterrows():
                event_date = pd.Timestamp(event["date"])
                if event_date < prices.index[0] or event_date > prices.index[-1]:
                    continue

                # Determine event color
                category = str(event.get("category", ""))
                event_type = str(event.get("event_type", ""))
                color = self.EVENT_COLORS.get(event_type, self.EVENT_COLORS["default"])

                severity = event.get("severity", 5)
                size = max(6, min(16, severity * 1.5))

                fig.add_trace(
                    go.Scatter(
                        x=[event_date],
                        y=[normalized.asof(event_date) if not pd.isna(normalized.asof(event_date)) else 100],
                        mode="markers",
                        marker=dict(color=color, size=size, symbol="diamond",
                                    line=dict(color="white", width=1)),
                        name=event.get("event", "")[:30],
                        hovertext=f"{event.get('event', '')}<br>Severity: {severity}",
                        showlegend=False,
                    ),
                    row=i + 1, col=1,
                )

        fig.update_layout(
            height=300 * n_markets,
            template="plotly_white",
            title=title,
            hovermode="x unified",
        )
        return fig

    def plot_event_study_results(
        self,
        acar: pd.Series,
        acar_std: pd.Series,
        n_events: int,
        title: str = "Average Cumulative Abnormal Returns",
    ) -> go.Figure:
        """Plot ACAR with confidence intervals."""
        fig = go.Figure()

        x = acar.index.tolist()

        # Confidence band
        upper = (acar + 1.96 * acar_std).values
        lower = (acar - 1.96 * acar_std).values

        fig.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=list(upper) + list(lower[::-1]),
                fill="toself",
                fillcolor="rgba(25, 118, 210, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="95% CI",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x, y=acar.values,
                mode="lines+markers",
                name=f"ACAR (n={n_events})",
                line=dict(color="#1976d2", width=3),
                marker=dict(size=4),
            )
        )

        fig.add_vline(x=0, line_dash="dash", line_color="grey",
                      annotation_text="Event Day")
        fig.add_hline(y=0, line_dash="solid", line_color="grey")

        fig.update_layout(
            title=title,
            xaxis_title="Days Relative to Event",
            yaxis_title="Cumulative Abnormal Return",
            template="plotly_white",
            height=500,
        )
        return fig

    def plot_sentiment_timeline(
        self,
        daily_sentiment: pd.DataFrame,
        events_df: pd.DataFrame | None = None,
    ) -> go.Figure:
        """Plot LLM sentiment across markets over time."""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=["Market-Specific Sentiment", "Sentiment Divergence"],
        )

        market_colors = {"us": "#1976d2", "india": "#e65100", "china": "#c62828"}

        for market, color in market_colors.items():
            col = f"{market}_sentiment_mean"
            if col in daily_sentiment.columns:
                fig.add_trace(
                    go.Scatter(
                        x=daily_sentiment.index,
                        y=daily_sentiment[col].rolling(5).mean(),
                        name=f"{market.upper()} Sentiment",
                        line=dict(color=color, width=2),
                    ),
                    row=1, col=1,
                )

        # Sentiment divergence
        if "us_sentiment_mean" in daily_sentiment.columns and "china_sentiment_mean" in daily_sentiment.columns:
            divergence = daily_sentiment["us_sentiment_mean"] - daily_sentiment["china_sentiment_mean"]
            fig.add_trace(
                go.Scatter(
                    x=daily_sentiment.index,
                    y=divergence.rolling(5).mean(),
                    name="US-China Divergence",
                    line=dict(color="#9c27b0", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(156, 39, 176, 0.1)",
                ),
                row=2, col=1,
            )

        fig.add_hline(y=0, line_dash="dash", line_color="grey", row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="grey", row=2, col=1)

        fig.update_layout(
            height=600,
            template="plotly_white",
            title="LLM Sentiment Analysis Over Time",
        )
        return fig
