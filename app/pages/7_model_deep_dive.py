"""
Page 7: Model Deep Dive
Temporal holdout, rolling evaluation, calibration, learning curves,
Diebold-Mariano test, and extended model comparison.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils.config import DATA_DIR

st.set_page_config(page_title="Model Deep Dive", layout="wide")
st.title("Model Deep Dive: Research-Grade Evaluation")

st.markdown("""
Beyond cross-validation accuracy, these analyses test whether the hybrid model
is genuinely useful: Does it work on **unseen future data**? Is it **stable
over time**? Are its probability estimates **calibrated**? Does it beat
baselines by a **statistically significant** margin?
""")

RESULTS_DIR = DATA_DIR.parent / "results"

# Sidebar
st.sidebar.header("Settings")
market = st.sidebar.selectbox(
    "Market",
    ["US", "India", "China"],
    format_func=lambda x: {"US": "US (S&P 500)", "India": "India (NIFTY 50)", "China": "China (HSI)"}[x],
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Temporal Holdout",
    "Rolling Evaluation",
    "Model Calibration",
    "Learning Curves",
    "Model Comparison (DM Test)",
])

# ---------------------------------------------------------------------------
# Tab 1: Temporal Holdout
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Temporal Holdout: Train 2015-2022, Test 2023+")
    st.markdown("""
    The most honest evaluation: train on historical data, test on **genuinely
    unseen future data**. Cross-validation can leak temporal information; this
    test does not.
    """)

    holdout_path = RESULTS_DIR / f"temporal_holdout_{market}.csv"
    holdout_data = None

    if holdout_path.exists():
        try:
            holdout_data = pd.read_csv(holdout_path)
        except Exception:
            pass

    if holdout_data is not None and not holdout_data.empty:
        row = holdout_data.iloc[0]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Test Accuracy", f"{row.get('test_accuracy', 0):.4f}")
        col2.metric("Test F1", f"{row.get('test_f1', 0):.4f}")
        col3.metric("Train Samples", int(row.get('n_train', 0)))
        col4.metric("Test Samples", int(row.get('n_test', 0)))

        # Show train/test split dates if available
        if "train_end" in row:
            st.info(f"Train period: {row.get('train_start', '2015')} to {row['train_end']} | "
                    f"Test period: {row.get('test_start', '2023')} to {row.get('test_end', 'present')}")

        # Compare with cross-validation accuracy
        ablation_path = RESULTS_DIR / f"ablation_{market}.csv"
        if ablation_path.exists():
            try:
                ablation = pd.read_csv(ablation_path)
                if not ablation.empty:
                    full_model = ablation[ablation["feature_group"] == "full_model"]
                    if not full_model.empty:
                        cv_acc = full_model.iloc[0]["mean_accuracy"]
                        holdout_acc = row.get("test_accuracy", 0)
                        gap = cv_acc - holdout_acc
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=["Cross-Validation", "Temporal Holdout"],
                            y=[cv_acc, holdout_acc],
                            marker_color=["#1976d2", "#e65100"],
                            text=[f"{cv_acc:.4f}", f"{holdout_acc:.4f}"],
                            textposition="outside",
                        ))
                        fig.update_layout(
                            title=f"CV vs Temporal Holdout Accuracy: {market}",
                            yaxis_title="Accuracy",
                            template="plotly_white",
                            height=400,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        if gap > 0.05:
                            st.warning(f"Gap of {gap:.4f} between CV and holdout — possible overfitting to historical patterns")
                        else:
                            st.success(f"Small gap ({gap:.4f}) — model generalizes well to unseen future data")
            except Exception:
                pass
    else:
        st.info("Run `python scripts/run_analysis.py` to generate temporal holdout results.")

# ---------------------------------------------------------------------------
# Tab 2: Rolling Evaluation
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Rolling Window Evaluation")
    st.markdown("""
    Tests model stability over time using a **3-year rolling training window**
    with **quarterly test periods**. Reveals whether the model degrades during
    specific market regimes.
    """)

    rolling_path = RESULTS_DIR / f"rolling_eval_{market}.csv"
    rolling_data = None

    if rolling_path.exists():
        try:
            rolling_data = pd.read_csv(rolling_path)
        except Exception:
            pass

    if rolling_data is not None and not rolling_data.empty:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=["Accuracy Over Time", "F1 Score Over Time"],
        )

        x_labels = rolling_data.get("test_start", rolling_data.index).astype(str)

        fig.add_trace(go.Scatter(
            x=x_labels,
            y=rolling_data["accuracy"],
            mode="lines+markers",
            name="Accuracy",
            line=dict(color="#1976d2", width=2),
            marker=dict(size=6),
        ), row=1, col=1)

        # Mean accuracy line
        mean_acc = rolling_data["accuracy"].mean()
        fig.add_hline(y=mean_acc, line_dash="dash", line_color="#78909c",
                      annotation_text=f"Mean = {mean_acc:.4f}", row=1, col=1)

        if "f1" in rolling_data.columns:
            fig.add_trace(go.Scatter(
                x=x_labels,
                y=rolling_data["f1"],
                mode="lines+markers",
                name="F1",
                line=dict(color="#26a69a", width=2),
                marker=dict(size=6),
            ), row=2, col=1)

            mean_f1 = rolling_data["f1"].mean()
            fig.add_hline(y=mean_f1, line_dash="dash", line_color="#78909c",
                          annotation_text=f"Mean = {mean_f1:.4f}", row=2, col=1)

        fig.update_layout(
            title=f"Rolling Window Performance: {market}",
            template="plotly_white",
            height=600,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean Accuracy", f"{rolling_data['accuracy'].mean():.4f}")
        col2.metric("Std Accuracy", f"{rolling_data['accuracy'].std():.4f}")
        col3.metric("Min Accuracy", f"{rolling_data['accuracy'].min():.4f}")
        col4.metric("Windows", len(rolling_data))

        with st.expander("Rolling Window Details"):
            st.dataframe(rolling_data.round(4), use_container_width=True)
    else:
        st.info("Run `python scripts/run_analysis.py` to generate rolling evaluation results.")

# ---------------------------------------------------------------------------
# Tab 3: Model Calibration
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Model Calibration: Are Probabilities Meaningful?")
    st.markdown("""
    A well-calibrated model's predicted probability of 0.7 should correspond to
    ~70% actual accuracy. The **Expected Calibration Error (ECE)** measures the
    average gap between predicted confidence and actual accuracy.
    """)

    cal_path = RESULTS_DIR / f"calibration_{market}.csv"
    cal_data = None

    if cal_path.exists():
        try:
            cal_data = pd.read_csv(cal_path)
        except Exception:
            pass

    if cal_data is not None and not cal_data.empty:
        row = cal_data.iloc[0]

        col1, col2 = st.columns(2)
        ece_val = row.get("ece", 0)
        col1.metric("Expected Calibration Error", f"{ece_val:.4f}")
        col2.metric("Test Samples", int(row.get("n_test", 0)))

        if ece_val < 0.05:
            st.success(f"ECE = {ece_val:.4f} — model is **well calibrated**")
        elif ece_val < 0.10:
            st.info(f"ECE = {ece_val:.4f} — model is **reasonably calibrated**")
        else:
            st.warning(f"ECE = {ece_val:.4f} — model is **poorly calibrated**, probabilities are unreliable")

        # Reliability diagram (perfect calibration line)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Perfect Calibration",
            line=dict(color="#78909c", dash="dash", width=2),
        ))

        fig.update_layout(
            title=f"Calibration Summary: {market} (ECE = {ece_val:.4f})",
            xaxis_title="Predicted Probability",
            yaxis_title="Actual Accuracy",
            template="plotly_white",
            height=400,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run `python scripts/run_analysis.py` to generate calibration results.")

# ---------------------------------------------------------------------------
# Tab 4: Learning Curves
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Learning Curves: Does More Data Help?")
    st.markdown("""
    Trains the model on increasing fractions of the data (25%, 50%, 75%, 100%).
    - **Rising curve**: more data helps, consider collecting more
    - **Flat curve**: model has saturated, adding data won't help
    - **Gap between train/test**: overfitting risk
    """)

    lc_path = RESULTS_DIR / f"learning_curves_{market}.csv"
    lc_data = None

    if lc_path.exists():
        try:
            lc_data = pd.read_csv(lc_path)
        except Exception:
            pass

    if lc_data is not None and not lc_data.empty:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=lc_data["fraction"],
            y=lc_data["accuracy"],
            mode="lines+markers",
            name="Accuracy",
            line=dict(color="#1976d2", width=3),
            marker=dict(size=10),
        ))

        if "f1" in lc_data.columns:
            fig.add_trace(go.Scatter(
                x=lc_data["fraction"],
                y=lc_data["f1"],
                mode="lines+markers",
                name="F1 Score",
                line=dict(color="#26a69a", width=3),
                marker=dict(size=10),
            ))

        fig.update_layout(
            title=f"Learning Curves: {market}",
            xaxis_title="Training Data Fraction",
            yaxis_title="Score",
            template="plotly_white",
            height=450,
            xaxis=dict(tickvals=[0.25, 0.5, 0.75, 1.0],
                       ticktext=["25%", "50%", "75%", "100%"]),
        )
        st.plotly_chart(fig, use_container_width=True)

        if "n_samples" in lc_data.columns:
            st.dataframe(lc_data.round(4), use_container_width=True)

        # Interpretation
        if len(lc_data) >= 2:
            improvement = lc_data["accuracy"].iloc[-1] - lc_data["accuracy"].iloc[0]
            if improvement > 0.02:
                st.info(f"Accuracy improves by {improvement:.4f} with more data — additional data may further help")
            else:
                st.success(f"Accuracy is largely stable ({improvement:.4f} change) — model has saturated")
    else:
        st.info("Run `python scripts/run_analysis.py` to generate learning curve results.")

# ---------------------------------------------------------------------------
# Tab 5: Model Comparison with DM Test
# ---------------------------------------------------------------------------
with tab5:
    st.subheader("Model Comparison: Diebold-Mariano Test")
    st.markdown("""
    The **Diebold-Mariano test** determines whether the accuracy difference
    between two models is **statistically significant** (not just random noise).
    Without this, a 0.5% accuracy lead could be meaningless.
    """)

    comp_path = RESULTS_DIR / f"model_comparison_extended_{market}.csv"
    comp_data = None

    if comp_path.exists():
        try:
            comp_data = pd.read_csv(comp_path)
        except Exception:
            pass

    if comp_data is not None and not comp_data.empty:
        # Sort by accuracy
        comp_sorted = comp_data.sort_values("mean_accuracy", ascending=False)

        fig = go.Figure()

        colors = ["#1976d2", "#26a69a", "#e65100", "#9c27b0"]
        for i, (_, row) in enumerate(comp_sorted.iterrows()):
            error_y_args = {}
            if "ci_lower_accuracy" in row and "ci_upper_accuracy" in row:
                if pd.notna(row["ci_lower_accuracy"]) and pd.notna(row["ci_upper_accuracy"]):
                    error_y_args = dict(
                        type="data",
                        symmetric=False,
                        array=[row["ci_upper_accuracy"] - row["mean_accuracy"]],
                        arrayminus=[row["mean_accuracy"] - row["ci_lower_accuracy"]],
                    )

            fig.add_trace(go.Bar(
                x=[row["model"]],
                y=[row["mean_accuracy"]],
                error_y=error_y_args if error_y_args else None,
                name=row["model"],
                marker_color=colors[i % len(colors)],
                text=[f"{row['mean_accuracy']:.4f}"],
                textposition="outside",
            ))

        fig.update_layout(
            title=f"Model Comparison with 95% CIs: {market}",
            yaxis_title="Accuracy",
            template="plotly_white",
            height=450,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # DM test results
        if "dm_p_value_vs_best" in comp_data.columns:
            st.markdown("#### Diebold-Mariano p-values (vs best model)")
            best_model = comp_sorted.iloc[0]["model"]
            st.markdown(f"**Best model:** {best_model}")

            for _, row in comp_sorted.iterrows():
                if row["model"] == best_model:
                    continue
                dm_p = row.get("dm_p_value_vs_best", np.nan)
                if pd.notna(dm_p):
                    if dm_p < 0.05:
                        st.markdown(f"- **{row['model']}**: p = {dm_p:.4f} — best model is significantly better")
                    else:
                        st.markdown(f"- **{row['model']}**: p = {dm_p:.4f} — difference is NOT significant")

        st.dataframe(comp_sorted.round(4), use_container_width=True)
    else:
        # Fall back to standard model comparison
        comp_basic_path = RESULTS_DIR / f"model_comparison_{market}.csv"
        if comp_basic_path.exists():
            try:
                comp_basic = pd.read_csv(comp_basic_path)
                if not comp_basic.empty:
                    st.info("Extended comparison with DM test not available. Showing basic comparison.")
                    st.dataframe(comp_basic.round(4), use_container_width=True)
            except Exception:
                pass
        else:
            st.info("Run `python scripts/run_analysis.py` to generate model comparison results.")
