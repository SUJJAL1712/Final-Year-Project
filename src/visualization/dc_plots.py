"""
Directional Changes visualization.

Creates publication-quality plots for DC analysis results.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class DCVisualizer:
    """Visualizes Directional Changes analysis results."""

    COLORS = {
        "upturn": "#26a69a",     # Green
        "downturn": "#ef5350",   # Red
        "neutral": "#78909c",    # Grey
        "price": "#1976d2",      # Blue
    }

    def plot_dc_events_on_price(
        self,
        prices: pd.Series,
        dc_df: pd.DataFrame,
        title: str = "Price with Directional Changes",
        geopolitical_events: pd.DataFrame | None = None,
    ) -> go.Figure:
        """
        Plot price chart annotated with DC events and optional geopolitical events.
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
            subplot_titles=[title, "DC Magnitude"],
        )

        # Price line
        fig.add_trace(
            go.Scatter(
                x=prices.index, y=prices.values,
                mode="lines", name="Price",
                line=dict(color=self.COLORS["price"], width=1.5),
            ),
            row=1, col=1,
        )

        if not dc_df.empty:
            # DC event markers
            for direction, color, symbol in [
                ("UPTURN", self.COLORS["upturn"], "triangle-up"),
                ("DOWNTURN", self.COLORS["downturn"], "triangle-down"),
            ]:
                mask = dc_df["dc_direction"] == direction
                subset = dc_df[mask]
                fig.add_trace(
                    go.Scatter(
                        x=subset["dc_confirm_time"],
                        y=subset["dc_confirm_price"],
                        mode="markers",
                        name=f"DC {direction.lower()}",
                        marker=dict(color=color, size=8, symbol=symbol),
                    ),
                    row=1, col=1,
                )

            # DC magnitude bar chart
            colors = [
                self.COLORS["upturn"] if d == "UPTURN" else self.COLORS["downturn"]
                for d in dc_df["dc_direction"]
            ]
            fig.add_trace(
                go.Bar(
                    x=dc_df["dc_confirm_time"],
                    y=dc_df["dc_magnitude"],
                    name="DC Magnitude",
                    marker_color=colors,
                ),
                row=2, col=1,
            )

        # Geopolitical event annotations
        if geopolitical_events is not None and not geopolitical_events.empty:
            for _, event in geopolitical_events.iterrows():
                event_date = pd.Timestamp(event["date"])
                if event_date >= prices.index[0] and event_date <= prices.index[-1]:
                    fig.add_vline(
                        x=event_date,
                        line_dash="dash",
                        line_color="rgba(255, 152, 0, 0.5)",
                        row=1, col=1,
                    )
                    fig.add_annotation(
                        x=event_date,
                        y=prices.max(),
                        text=event["event"][:40],
                        showarrow=True,
                        arrowhead=2,
                        font=dict(size=8),
                        row=1, col=1,
                    )

        fig.update_layout(
            height=700,
            template="plotly_white",
            showlegend=True,
            hovermode="x unified",
        )
        return fig

    def plot_scaling_laws(self, scaling_df: pd.DataFrame) -> go.Figure:
        """Plot DC scaling law verification: log-log plots."""
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=[
                "Coastline Length vs Threshold",
                "OS/DC Ratio vs Threshold",
                "DC Duration vs Threshold",
            ],
        )

        # 1. Coastline (num events) vs threshold
        fig.add_trace(
            go.Scatter(
                x=np.log10(scaling_df["threshold"]),
                y=np.log10(scaling_df["num_dc_events"]),
                mode="markers+lines",
                name="Coastline",
                marker=dict(size=10, color=self.COLORS["price"]),
            ),
            row=1, col=1,
        )

        # 2. OS/DC ratio vs threshold (should be ~constant)
        fig.add_trace(
            go.Scatter(
                x=np.log10(scaling_df["threshold"]),
                y=scaling_df["avg_overshoot_ratio"],
                mode="markers+lines",
                name="OS/DC Ratio",
                marker=dict(size=10, color=self.COLORS["upturn"]),
            ),
            row=1, col=2,
        )

        # 3. DC duration vs threshold
        fig.add_trace(
            go.Scatter(
                x=np.log10(scaling_df["threshold"]),
                y=np.log10(scaling_df["avg_dc_duration_days"].clip(lower=0.01)),
                mode="markers+lines",
                name="DC Duration",
                marker=dict(size=10, color=self.COLORS["downturn"]),
            ),
            row=1, col=3,
        )

        fig.update_xaxes(title_text="log10(threshold)", row=1, col=1)
        fig.update_xaxes(title_text="log10(threshold)", row=1, col=2)
        fig.update_xaxes(title_text="log10(threshold)", row=1, col=3)
        fig.update_yaxes(title_text="log10(# DC events)", row=1, col=1)
        fig.update_yaxes(title_text="OS/DC Ratio", row=1, col=2)
        fig.update_yaxes(title_text="log10(days)", row=1, col=3)

        fig.update_layout(
            height=400,
            template="plotly_white",
            title="DC Scaling Laws Verification",
            showlegend=False,
        )
        return fig

    def plot_intrinsic_time(
        self,
        prices: pd.Series,
        intrinsic_time: pd.Series,
        tcf: pd.Series,
    ) -> go.Figure:
        """Plot intrinsic time and time compression factor."""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=["Price", "Intrinsic Time", "Time Compression Factor"],
        )

        fig.add_trace(
            go.Scatter(x=prices.index, y=prices.values, name="Price",
                       line=dict(color=self.COLORS["price"])),
            row=1, col=1,
        )

        fig.add_trace(
            go.Scatter(x=intrinsic_time.index, y=intrinsic_time.values,
                       name="Intrinsic Time", line=dict(color=self.COLORS["upturn"])),
            row=2, col=1,
        )

        fig.add_trace(
            go.Scatter(x=tcf.index, y=tcf.values, name="TCF",
                       line=dict(color=self.COLORS["downturn"])),
            row=3, col=1,
        )
        fig.add_hline(y=1.0, line_dash="dash", line_color="grey", row=3, col=1)
        fig.add_hline(y=2.0, line_dash="dot", line_color="orange",
                      annotation_text="Burst threshold", row=3, col=1)

        fig.update_layout(height=800, template="plotly_white", hovermode="x unified")
        return fig

    def plot_ablation_results(self, ablation_df: pd.DataFrame) -> go.Figure:
        """Plot ablation study results as a bar chart."""
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=ablation_df["feature_group"],
                y=ablation_df["mean_accuracy"],
                error_y=dict(type="data", array=ablation_df["std_accuracy"]),
                name="Accuracy",
                marker_color=self.COLORS["price"],
            )
        )

        fig.add_trace(
            go.Bar(
                x=ablation_df["feature_group"],
                y=ablation_df["mean_f1"],
                error_y=dict(type="data", array=ablation_df["std_f1"]),
                name="F1 Score",
                marker_color=self.COLORS["upturn"],
            )
        )

        fig.update_layout(
            title="Ablation Study: Feature Group Contribution",
            xaxis_title="Feature Group",
            yaxis_title="Score",
            barmode="group",
            template="plotly_white",
            height=500,
        )
        return fig
