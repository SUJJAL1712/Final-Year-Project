"""
Heatmap visualizations for sector vulnerability and cross-market analysis.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


class HeatmapVisualizer:
    """Creates heatmap visualizations."""

    def plot_sector_vulnerability_heatmap(
        self,
        vulnerability_matrix: pd.DataFrame,
        title: str = "Sector Vulnerability to Geopolitical Events",
    ) -> go.Figure:
        """
        Heatmap showing how each sector responds to different event types.
        Red = negative impact, Green = positive impact.
        """
        fig = go.Figure(
            data=go.Heatmap(
                z=vulnerability_matrix.values,
                x=vulnerability_matrix.columns,
                y=vulnerability_matrix.index,
                colorscale=[
                    [0, "#c62828"],      # Strong negative
                    [0.3, "#ef5350"],    # Negative
                    [0.5, "#ffffff"],    # Neutral
                    [0.7, "#66bb6a"],   # Positive
                    [1, "#2e7d32"],      # Strong positive
                ],
                zmid=0,
                text=np.round(vulnerability_matrix.values * 100, 2),
                texttemplate="%{text:.2f}%",
                textfont={"size": 10},
                colorbar=dict(title="Abnormal Return"),
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Event Type",
            yaxis_title="Sector",
            template="plotly_white",
            height=600,
            width=900,
        )
        return fig

    def plot_cross_market_correlation_heatmap(
        self,
        correlation_matrix: pd.DataFrame,
        title: str = "Cross-Market Correlation During Events",
    ) -> go.Figure:
        """Correlation heatmap between markets during geopolitical events."""
        fig = go.Figure(
            data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale="RdBu_r",
                zmid=0,
                text=np.round(correlation_matrix.values, 3),
                texttemplate="%{text:.3f}",
                textfont={"size": 12},
            )
        )

        fig.update_layout(
            title=title,
            template="plotly_white",
            height=500,
            width=600,
        )
        return fig

    def plot_granger_causality_heatmap(
        self, granger_df: pd.DataFrame
    ) -> go.Figure:
        """Visualize Granger causality test results as directed heatmap."""
        # Create pivot: rows = cause, columns = effect
        pivot = granger_df.pivot(
            index="cause", columns="effect", values="best_p_value"
        ).fillna(1.0)

        # Transform p-values: -log10(p) for visual significance
        significance = -np.log10(pivot.values.clip(min=1e-10))

        fig = go.Figure(
            data=go.Heatmap(
                z=significance,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale="YlOrRd",
                text=np.round(pivot.values, 4),
                texttemplate="p=%{text:.4f}",
                textfont={"size": 11},
                colorbar=dict(title="-log10(p-value)"),
            )
        )

        fig.update_layout(
            title="Granger Causality: Row → Column<br>(darker = stronger causality)",
            xaxis_title="Effect (Target Market)",
            yaxis_title="Cause (Source)",
            template="plotly_white",
            height=500,
            width=600,
        )
        return fig

    def plot_feature_importance(
        self,
        importance: pd.Series,
        top_n: int = 20,
        title: str = "Top Feature Importances",
    ) -> go.Figure:
        """Horizontal bar chart of feature importances."""
        top = importance.head(top_n).sort_values()

        # Color by feature type
        colors = []
        for feat in top.index:
            if any(kw in feat for kw in ["dc_", "upturn", "os_", "duration_trend"]):
                colors.append("#1976d2")  # DC features
            elif any(kw in feat for kw in ["llm_", "sentiment"]):
                colors.append("#e65100")  # LLM features
            elif any(kw in feat for kw in ["event_", "severity", "tariff", "conflict"]):
                colors.append("#9c27b0")  # Event features
            else:
                colors.append("#607d8b")  # Market features

        fig = go.Figure(
            go.Bar(
                x=top.values,
                y=top.index,
                orientation="h",
                marker_color=colors,
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Importance",
            template="plotly_white",
            height=max(400, top_n * 25),
        )
        return fig
