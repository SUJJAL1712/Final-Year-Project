"""
Cross-market visualization: US-India-China triangle analysis plots.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class CrossMarketVisualizer:
    """Visualizes cross-market dynamics in the US-India-China triangle."""

    MARKET_COLORS = {
        "US": "#1976d2",
        "India": "#e65100",
        "China": "#c62828",
    }

    def plot_normalized_comparison(
        self,
        market_prices: dict[str, pd.Series],
        events_df: pd.DataFrame | None = None,
        title: str = "US-India-China Market Comparison (Normalized)",
    ) -> go.Figure:
        """Plot normalized (base=100) price comparison across markets."""
        fig = go.Figure()

        for market, prices in market_prices.items():
            normalized = (prices / prices.iloc[0]) * 100
            color = self.MARKET_COLORS.get(market, "#607d8b")
            fig.add_trace(
                go.Scatter(
                    x=normalized.index, y=normalized.values,
                    name=market, line=dict(color=color, width=2),
                )
            )

        if events_df is not None and not events_df.empty:
            top_events = events_df.nlargest(20, "severity") if "severity" in events_df.columns else events_df.head(20)
            for _, event in top_events.iterrows():
                event_dt = pd.Timestamp(event["date"])
                fig.add_shape(
                    type="line",
                    x0=event_dt, x1=event_dt,
                    y0=0, y1=1,
                    yref="paper",
                    line=dict(dash="dash", color="rgba(0,0,0,0.3)", width=1),
                )

        fig.update_layout(
            title=title,
            yaxis_title="Normalized Price (Base=100)",
            template="plotly_white",
            height=500,
            hovermode="x unified",
        )
        return fig

    def plot_contagion_analysis(
        self, contagion_results: pd.DataFrame
    ) -> go.Figure:
        """Visualize cross-market contagion patterns."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Transmission Ratio", "Peak Lag (Days)"],
        )

        markets = ["India", "China"]

        for market in markets:
            ratio_col = f"{market}_transmission_ratio"
            lag_col = f"{market}_peak_lag_days"

            if ratio_col in contagion_results.columns:
                color = self.MARKET_COLORS.get(market, "#607d8b")

                fig.add_trace(
                    go.Box(
                        y=contagion_results[ratio_col].dropna(),
                        name=f"→ {market}",
                        marker_color=color,
                    ),
                    row=1, col=1,
                )

            if lag_col in contagion_results.columns:
                fig.add_trace(
                    go.Box(
                        y=contagion_results[lag_col].dropna(),
                        name=f"→ {market}",
                        marker_color=color,
                    ),
                    row=1, col=2,
                )

        fig.update_layout(
            title="Cross-Market Contagion from US",
            template="plotly_white",
            height=450,
        )
        return fig

    def plot_india_trade_diversion(self, results: dict) -> go.Figure:
        """Visualize India's trade diversion benefit from US-China tensions."""
        reactions = results.get("reactions", pd.DataFrame())
        if reactions.empty:
            return go.Figure()

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "Market Reactions to US-China Events",
                "India as Beneficiary Analysis",
            ],
        )

        # Scatter: US vs China returns with India as color
        fig.add_trace(
            go.Scatter(
                x=reactions["us_cum_return"],
                y=reactions["china_cum_return"],
                mode="markers",
                marker=dict(
                    color=reactions["india_cum_return"],
                    colorscale="RdYlGn",
                    size=10,
                    showscale=True,
                    colorbar=dict(title="India Return", x=0.45),
                ),
                text=reactions["event_date"],
                name="Events",
            ),
            row=1, col=1,
        )

        # Bar: India return when both US and China are negative
        markets = ["us", "india", "china"]
        avg_returns = [reactions[f"{m}_cum_return"].mean() for m in markets]

        fig.add_trace(
            go.Bar(
                x=["US", "India", "China"],
                y=avg_returns,
                marker_color=[
                    self.MARKET_COLORS["US"],
                    self.MARKET_COLORS["India"],
                    self.MARKET_COLORS["China"],
                ],
                name="Avg Return",
            ),
            row=1, col=2,
        )

        fig.update_xaxes(title_text="US Cumulative Return", row=1, col=1)
        fig.update_yaxes(title_text="China Cumulative Return", row=1, col=1)

        fig.update_layout(
            title="India Trade Diversion Hypothesis",
            template="plotly_white",
            height=450,
        )
        return fig

    def plot_impulse_response(
        self,
        irf_data: np.ndarray,
        market_names: list[str],
        periods: int = 20,
    ) -> go.Figure:
        """Plot VAR impulse response functions."""
        n_vars = len(market_names)
        fig = make_subplots(
            rows=n_vars, cols=n_vars,
            subplot_titles=[
                f"Shock: {src} → Response: {tgt}"
                for src in market_names
                for tgt in market_names
            ],
        )

        x = list(range(periods + 1))

        for i, shock in enumerate(market_names):
            for j, response in enumerate(market_names):
                color = self.MARKET_COLORS.get(response, "#607d8b")
                fig.add_trace(
                    go.Scatter(
                        x=x, y=irf_data[:, j, i],
                        mode="lines",
                        line=dict(color=color, width=2),
                        showlegend=False,
                    ),
                    row=j + 1, col=i + 1,
                )
                fig.add_hline(y=0, line_dash="dash", line_color="grey",
                              row=j + 1, col=i + 1)

        fig.update_layout(
            title="Impulse Response Functions",
            template="plotly_white",
            height=250 * n_vars,
            width=300 * n_vars,
        )
        return fig
