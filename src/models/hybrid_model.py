"""
Hybrid DC-LLM Prediction Model.

The core innovation: combining Directional Changes features with
LLM-derived geopolitical signals for market direction prediction.

Architecture:
1. DC features capture the market's intrinsic dynamics
2. LLM features capture geopolitical context and sentiment
3. An ensemble model combines both signal sources
4. Ablation studies demonstrate the value of each component
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from loguru import logger

from src.utils.config import get_config


class HybridDCLLMPredictor:
    """
    Hybrid model that combines DC and LLM features for market prediction.

    Supports ablation studies to measure the contribution of each
    feature source (DC-only, LLM-only, Market-only, Combined).
    """

    def __init__(self):
        config = get_config().model_config
        self.test_size = config["test_size"]
        self.random_state = config["random_state"]
        self.cv_folds = config["cv_folds"]
        self.importance_threshold = config["importance_threshold"]

        self.models = self._build_models()
        self.scaler = StandardScaler()
        self.feature_groups = {}
        self.results = {}

    def _build_models(self) -> dict:
        """Initialize the model ensemble."""
        return {
            "random_forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=5,
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "xgboost": XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric="mlogloss",
            ),
            "lightgbm": LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbose=-1,
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=self.random_state,
            ),
        }

    def define_feature_groups(self, features: pd.DataFrame) -> dict[str, list[str]]:
        """
        Categorize features into groups for ablation studies.
        """
        all_cols = features.columns.tolist()

        dc_cols = [c for c in all_cols if any(
            kw in c for kw in ["dc_", "current_direction", "last_dc", "last_os",
                               "upturn_ratio", "os_surprise", "magnitude_asymmetry",
                               "duration_trend", "dc_frequency"]
        )]

        llm_cols = [c for c in all_cols if any(
            kw in c for kw in ["llm_", "sentiment"]
        )]

        event_cols = [c for c in all_cols if any(
            kw in c for kw in ["event_count", "severity", "days_since_event",
                               "tariff_event", "conflict_event", "india_trade",
                               "india_net"]
        )]

        market_cols = [c for c in all_cols if any(
            kw in c for kw in ["return_", "volatility_", "vol_ratio", "sma_",
                               "price_to_", "drawdown", "volume_", "skewness_",
                               "kurtosis_"]
        )]

        cross_market_cols = [c for c in all_cols if any(
            kw in c for kw in ["_corr_", "_lag", "US_", "India_", "China_",
                               "us_", "india_", "china_"]
        ) and c not in llm_cols]

        self.feature_groups = {
            "dc_only": dc_cols,
            "llm_only": llm_cols + event_cols,
            "market_only": market_cols,
            "cross_market": cross_market_cols,
            "dc_market": dc_cols + market_cols,
            "dc_llm": dc_cols + llm_cols + event_cols,
            "all_features": all_cols,
        }

        logger.info("Feature groups defined:")
        for group, cols in self.feature_groups.items():
            logger.info(f"  {group}: {len(cols)} features")

        return self.feature_groups

    def train_and_evaluate(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        feature_subset: list[str] | None = None,
        model_name: str = "xgboost",
    ) -> dict:
        """
        Train and evaluate a model using time-series cross-validation.
        """
        # Use feature subset or all
        cols = feature_subset or features.columns.tolist()
        X = features[cols].copy()
        y = labels.copy()

        # Align and clean
        common = X.index.intersection(y.index)
        X = X.loc[common].dropna()
        y = y.loc[X.index]

        # Remove zero-variance features
        X = X.loc[:, X.std() > 0]

        if len(X) < 100:
            return {"error": "insufficient_data", "n_samples": len(X)}

        # Time-series split
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        model = self.models[model_name]

        fold_results = []
        feature_importances = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index,
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index,
            )

            # Train
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            # Evaluate
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            fold_results.append({
                "fold": fold,
                "accuracy": acc,
                "f1_score": f1,
                "n_train": len(X_train),
                "n_test": len(X_test),
            })

            # Feature importance
            if hasattr(model, "feature_importances_"):
                imp = pd.Series(model.feature_importances_, index=X.columns)
                feature_importances.append(imp)

        # Aggregate results
        results_df = pd.DataFrame(fold_results)
        avg_importance = pd.concat(feature_importances).groupby(level=0).mean().sort_values(ascending=False) if feature_importances else pd.Series()

        return {
            "model_name": model_name,
            "n_features": len(cols),
            "feature_names": cols,
            "mean_accuracy": results_df["accuracy"].mean(),
            "std_accuracy": results_df["accuracy"].std(),
            "mean_f1": results_df["f1_score"].mean(),
            "std_f1": results_df["f1_score"].std(),
            "fold_results": results_df,
            "feature_importance": avg_importance,
            "top_features": avg_importance.head(20),
        }

    def run_ablation_study(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        model_name: str = "xgboost",
    ) -> pd.DataFrame:
        """
        Run ablation study: train with different feature subsets to
        demonstrate the contribution of DC and LLM features.

        This is the key evidence for the project's thesis: that combining
        DC analysis with LLM-derived geopolitical signals outperforms
        either approach alone.
        """
        if not self.feature_groups:
            self.define_feature_groups(features)

        ablation_results = []

        for group_name, feature_cols in self.feature_groups.items():
            # Filter to features that actually exist
            valid_cols = [c for c in feature_cols if c in features.columns]
            if not valid_cols:
                logger.warning(f"No valid features for group: {group_name}")
                continue

            logger.info(f"Ablation: {group_name} ({len(valid_cols)} features)")
            result = self.train_and_evaluate(
                features, labels, valid_cols, model_name
            )

            if "error" not in result:
                ablation_results.append({
                    "feature_group": group_name,
                    "n_features": result["n_features"],
                    "mean_accuracy": result["mean_accuracy"],
                    "std_accuracy": result["std_accuracy"],
                    "mean_f1": result["mean_f1"],
                    "std_f1": result["std_f1"],
                })

        df = pd.DataFrame(ablation_results).sort_values("mean_accuracy", ascending=False)
        self.results["ablation"] = df

        logger.info("\n=== Ablation Study Results ===")
        logger.info(f"\n{df.to_string()}")

        return df

    def model_comparison(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> pd.DataFrame:
        """
        Compare all models using the full feature set.
        """
        comparison_results = []

        for model_name in self.models:
            logger.info(f"Evaluating model: {model_name}")
            result = self.train_and_evaluate(features, labels, model_name=model_name)

            if "error" not in result:
                comparison_results.append({
                    "model": model_name,
                    "mean_accuracy": result["mean_accuracy"],
                    "std_accuracy": result["std_accuracy"],
                    "mean_f1": result["mean_f1"],
                    "std_f1": result["std_f1"],
                })

        df = pd.DataFrame(comparison_results).sort_values("mean_accuracy", ascending=False)
        self.results["model_comparison"] = df
        return df

    def get_final_predictions(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        model_name: str = "xgboost",
    ) -> pd.DataFrame:
        """
        Train on all data except the last test_size fraction,
        then generate predictions for the test period.
        """
        X = features.dropna()
        y = labels.loc[X.index]

        split_idx = int(len(X) * (1 - self.test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = self.models[model_name]
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled) if hasattr(model, "predict_proba") else None

        results = pd.DataFrame({
            "actual": y_test,
            "predicted": y_pred,
        }, index=X_test.index)

        if y_proba is not None:
            for i, cls in enumerate(model.classes_):
                results[f"prob_{cls}"] = y_proba[:, i]

        results["correct"] = results["actual"] == results["predicted"]

        logger.info(f"\nFinal Predictions ({model_name}):")
        logger.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        logger.info(f"\n{classification_report(y_test, y_pred)}")

        return results

    # ------------------------------------------------------------------
    # Research-grade extensions (additive — existing methods unchanged)
    # ------------------------------------------------------------------

    @staticmethod
    def _confidence_interval(values: np.ndarray, confidence: float = 0.95) -> tuple:
        """Compute CI using t-distribution (appropriate for small n_folds)."""
        n = len(values)
        if n < 2:
            return (np.nan, np.nan)
        mean = np.mean(values)
        sem = np.std(values, ddof=1) / np.sqrt(n)
        ci = scipy_stats.t.interval(confidence, df=n - 1, loc=mean, scale=sem)
        return (ci[0], ci[1])

    def train_and_evaluate_extended(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        feature_subset: list[str] | None = None,
        model_name: str = "xgboost",
    ) -> dict:
        """Like train_and_evaluate but adds CIs, importance stability,
        and per-observation errors for Diebold-Mariano tests.

        Returns the same keys as train_and_evaluate plus:
        - ci_lower_accuracy, ci_upper_accuracy
        - ci_lower_f1, ci_upper_f1
        - importance_stability (Spearman rank corr across folds)
        - top10_overlap (fraction of top-10 features stable across folds)
        - fold_errors: list of (y_test, y_pred) per fold for DM test
        """
        cols = feature_subset or features.columns.tolist()
        X = features[cols].copy()
        y = labels.copy()

        common = X.index.intersection(y.index)
        X = X.loc[common].dropna()
        y = y.loc[X.index]
        X = X.loc[:, X.std() > 0]

        if len(X) < 100:
            return {"error": "insufficient_data", "n_samples": len(X)}

        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        model = self.models[model_name]

        fold_results = []
        feature_importances = []
        fold_errors = []  # (y_true, y_pred) per fold for DM test

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            scaler = StandardScaler()
            X_train_s = pd.DataFrame(
                scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
            )
            X_test_s = pd.DataFrame(
                scaler.transform(X_test), columns=X_test.columns, index=X_test.index
            )

            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            fold_results.append({"fold": fold, "accuracy": acc, "f1_score": f1,
                                 "n_train": len(X_train), "n_test": len(X_test)})
            fold_errors.append((y_test.values, y_pred))

            if hasattr(model, "feature_importances_"):
                imp = pd.Series(model.feature_importances_, index=X.columns)
                feature_importances.append(imp)

        results_df = pd.DataFrame(fold_results)
        accs = results_df["accuracy"].values
        f1s = results_df["f1_score"].values

        avg_importance = (
            pd.concat(feature_importances).groupby(level=0).mean().sort_values(ascending=False)
            if feature_importances else pd.Series(dtype=float)
        )

        # --- Confidence intervals ---
        ci_acc = self._confidence_interval(accs)
        ci_f1 = self._confidence_interval(f1s)

        # --- Importance stability ---
        importance_stability = np.nan
        top10_overlap = np.nan
        if len(feature_importances) >= 2:
            from scipy.stats import spearmanr
            ranks = [imp.rank(ascending=False) for imp in feature_importances]
            corrs = []
            for i in range(len(ranks)):
                for j in range(i + 1, len(ranks)):
                    aligned = pd.DataFrame({"a": ranks[i], "b": ranks[j]}).dropna()
                    if len(aligned) > 2:
                        c, _ = spearmanr(aligned["a"], aligned["b"])
                        corrs.append(c)
            importance_stability = float(np.mean(corrs)) if corrs else np.nan

            # Top-10 overlap: fraction of folds where each top-10 feature appears in top-10
            top10_sets = [set(imp.nlargest(10).index) for imp in feature_importances]
            all_top10 = set.union(*top10_sets)
            overlap_scores = []
            for feat in all_top10:
                frac = sum(1 for s in top10_sets if feat in s) / len(top10_sets)
                overlap_scores.append(frac)
            top10_overlap = float(np.mean(overlap_scores))

        return {
            "model_name": model_name,
            "n_features": len(X.columns),
            "feature_names": list(X.columns),
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "ci_lower_accuracy": ci_acc[0],
            "ci_upper_accuracy": ci_acc[1],
            "mean_f1": float(np.mean(f1s)),
            "std_f1": float(np.std(f1s)),
            "ci_lower_f1": ci_f1[0],
            "ci_upper_f1": ci_f1[1],
            "fold_results": results_df,
            "feature_importance": avg_importance,
            "top_features": avg_importance.head(20),
            "importance_stability": importance_stability,
            "top10_overlap": top10_overlap,
            "fold_errors": fold_errors,
        }

    def run_ablation_study_extended(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        model_name: str = "xgboost",
    ) -> pd.DataFrame:
        """Ablation study with confidence intervals and importance stability."""
        if not self.feature_groups:
            self.define_feature_groups(features)

        ablation_results = []

        for group_name, feature_cols in self.feature_groups.items():
            valid_cols = [c for c in feature_cols if c in features.columns]
            if not valid_cols:
                continue

            logger.info(f"Ablation (extended): {group_name} ({len(valid_cols)} features)")
            result = self.train_and_evaluate_extended(features, labels, valid_cols, model_name)

            if "error" not in result:
                ablation_results.append({
                    "feature_group": group_name,
                    "n_features": result["n_features"],
                    "mean_accuracy": result["mean_accuracy"],
                    "std_accuracy": result["std_accuracy"],
                    "ci_lower_accuracy": result["ci_lower_accuracy"],
                    "ci_upper_accuracy": result["ci_upper_accuracy"],
                    "mean_f1": result["mean_f1"],
                    "std_f1": result["std_f1"],
                    "ci_lower_f1": result["ci_lower_f1"],
                    "ci_upper_f1": result["ci_upper_f1"],
                    "importance_stability": result.get("importance_stability", np.nan),
                    "top10_overlap": result.get("top10_overlap", np.nan),
                })

        df = pd.DataFrame(ablation_results).sort_values("mean_accuracy", ascending=False)
        self.results["ablation_extended"] = df

        logger.info("\n=== Extended Ablation Study Results ===")
        logger.info(f"\n{df.to_string()}")
        return df

    def model_comparison_extended(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> pd.DataFrame:
        """Model comparison with CIs and Diebold-Mariano test."""
        comparison_results = []
        all_fold_errors = {}

        for model_name in self.models:
            logger.info(f"Evaluating (extended): {model_name}")
            result = self.train_and_evaluate_extended(features, labels, model_name=model_name)

            if "error" not in result:
                comparison_results.append({
                    "model": model_name,
                    "mean_accuracy": result["mean_accuracy"],
                    "std_accuracy": result["std_accuracy"],
                    "ci_lower_accuracy": result["ci_lower_accuracy"],
                    "ci_upper_accuracy": result["ci_upper_accuracy"],
                    "mean_f1": result["mean_f1"],
                    "std_f1": result["std_f1"],
                    "ci_lower_f1": result["ci_lower_f1"],
                    "ci_upper_f1": result["ci_upper_f1"],
                })
                all_fold_errors[model_name] = result["fold_errors"]

        df = pd.DataFrame(comparison_results).sort_values("mean_accuracy", ascending=False)

        # Diebold-Mariano test: best model vs each other
        if len(df) >= 2 and all_fold_errors:
            best_model = df.iloc[0]["model"]
            dm_pvalues = []
            for _, row in df.iterrows():
                m = row["model"]
                if m == best_model:
                    dm_pvalues.append(np.nan)
                else:
                    p = self.diebold_mariano_test(
                        all_fold_errors[best_model], all_fold_errors[m]
                    )
                    dm_pvalues.append(p)
            df["dm_p_value_vs_best"] = dm_pvalues

        self.results["model_comparison_extended"] = df
        return df

    @staticmethod
    def diebold_mariano_test(
        errors_a: list[tuple], errors_b: list[tuple], h: int = 1
    ) -> float:
        """Diebold-Mariano test for equal predictive accuracy.

        Args:
            errors_a, errors_b: Lists of (y_true, y_pred) per fold.
            h: Forecast horizon (for HAC variance correction).

        Returns:
            Two-sided p-value. Small p → models differ significantly.
        """
        # Compute squared error differences across all folds
        d_all = []
        for (yt_a, yp_a), (yt_b, yp_b) in zip(errors_a, errors_b):
            min_len = min(len(yt_a), len(yt_b))
            e_a = (yt_a[:min_len] != yp_a[:min_len]).astype(float)  # 0-1 loss
            e_b = (yt_b[:min_len] != yp_b[:min_len]).astype(float)
            d_all.extend(e_a - e_b)

        d = np.array(d_all)
        if len(d) < 10 or np.std(d) == 0:
            return 1.0

        n = len(d)
        d_bar = np.mean(d)
        # HAC variance (Newey-West with h-1 lags)
        gamma_0 = np.var(d, ddof=1)
        gamma_sum = 0
        for k in range(1, h):
            gamma_k = np.cov(d[k:], d[:-k])[0, 1] if len(d) > k else 0
            gamma_sum += 2 * gamma_k
        var_d = (gamma_0 + gamma_sum) / n
        if var_d <= 0:
            return 1.0

        dm_stat = d_bar / np.sqrt(var_d)
        p_value = 2 * (1 - scipy_stats.norm.cdf(abs(dm_stat)))
        return float(p_value)

    def temporal_holdout_evaluation(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        train_end: str = "2022-12-31",
        model_name: str = "xgboost",
    ) -> dict:
        """Train on data up to train_end, test on everything after.

        This is a strict out-of-sample test on a meaningful temporal
        boundary (post-COVID, post-Ukraine-invasion regime).
        """
        X = features.dropna()
        y = labels.loc[X.index]
        X = X.loc[:, X.std() > 0]

        cutoff = pd.Timestamp(train_end)
        train_mask = X.index <= cutoff
        test_mask = X.index > cutoff

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        if len(X_train) < 50 or len(X_test) < 20:
            return {"error": "insufficient_data",
                    "n_train": len(X_train), "n_test": len(X_test)}

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = self.models[model_name]
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        logger.info(f"Temporal holdout (train ≤ {train_end}): "
                    f"acc={acc:.4f}, F1={f1:.4f}, n_test={len(X_test)}")

        return {
            "model_name": model_name,
            "train_end": train_end,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "accuracy": acc,
            "f1": f1,
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            ),
        }

    def rolling_evaluation(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        train_window: int = 756,
        step: int = 63,
        model_name: str = "xgboost",
    ) -> pd.DataFrame:
        """Rolling-window evaluation: train on a fixed window, test on next quarter.

        Reveals whether model performance is stable over time or degrades.

        Args:
            train_window: Number of observations in train set (~3 years).
            step: Step size between windows (~1 quarter).
        """
        X = features.dropna()
        y = labels.loc[X.index]
        X = X.loc[:, X.std() > 0]

        results = []
        start = 0
        while start + train_window + step <= len(X):
            X_train = X.iloc[start:start + train_window]
            X_test = X.iloc[start + train_window:start + train_window + step]
            y_train = y.iloc[start:start + train_window]
            y_test = y.iloc[start + train_window:start + train_window + step]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_train)
            X_te_s = scaler.transform(X_test)

            model = self.models[model_name]
            model.fit(X_tr_s, y_train)
            y_pred = model.predict(X_te_s)

            results.append({
                "window_start": X_train.index[0],
                "window_end": X_train.index[-1],
                "test_start": X_test.index[0],
                "test_end": X_test.index[-1],
                "accuracy": accuracy_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                "n_test": len(X_test),
            })
            start += step

        df = pd.DataFrame(results)
        if not df.empty:
            logger.info(f"Rolling eval: {len(df)} windows, "
                        f"mean acc={df['accuracy'].mean():.4f} "
                        f"(std={df['accuracy'].std():.4f})")
        return df

    def calibration_analysis(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        model_name: str = "xgboost",
        n_bins: int = 10,
    ) -> dict:
        """Compute reliability diagram data and Expected Calibration Error.

        Uses the same train/test split as get_final_predictions.
        """
        X = features.dropna()
        y = labels.loc[X.index]

        split_idx = int(len(X) * (1 - self.test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)

        model = self.models[model_name]
        model.fit(X_tr_s, y_train)

        if not hasattr(model, "predict_proba"):
            return {"error": "model_has_no_predict_proba"}

        y_proba = model.predict_proba(X_te_s)
        classes = model.classes_

        # Per-class calibration
        calibration = {}
        ece_total = 0.0
        for i, cls in enumerate(classes):
            probs = y_proba[:, i]
            actuals = (y_test.values == cls).astype(float)
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_centers = []
            bin_accs = []
            bin_counts = []
            for b in range(n_bins):
                mask = (probs >= bin_edges[b]) & (probs < bin_edges[b + 1])
                if mask.sum() > 0:
                    bin_centers.append((bin_edges[b] + bin_edges[b + 1]) / 2)
                    bin_accs.append(actuals[mask].mean())
                    bin_counts.append(int(mask.sum()))
                    ece_total += mask.sum() * abs(actuals[mask].mean() - probs[mask].mean())

            calibration[f"class_{cls}"] = {
                "bin_centers": bin_centers,
                "bin_accuracies": bin_accs,
                "bin_counts": bin_counts,
            }

        ece = ece_total / len(y_test) if len(y_test) > 0 else 0.0
        logger.info(f"Expected Calibration Error (ECE): {ece:.4f}")

        return {"ece": ece, "calibration": calibration, "n_test": len(y_test)}

    def learning_curves(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        fractions: list[float] | None = None,
        model_name: str = "xgboost",
    ) -> pd.DataFrame:
        """Train with increasing data fractions to see if more data helps.

        Uses a fixed test set (last test_size fraction) and varies train size.
        """
        fractions = fractions or [0.25, 0.50, 0.75, 1.0]

        X = features.dropna()
        y = labels.loc[X.index]
        X = X.loc[:, X.std() > 0]

        split_idx = int(len(X) * (1 - self.test_size))
        X_train_full, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train_full, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        results = []
        for frac in fractions:
            n = max(50, int(len(X_train_full) * frac))
            # Take the last n observations (most recent, avoids regime mismatch)
            X_tr = X_train_full.iloc[-n:]
            y_tr = y_train_full.iloc[-n:]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_test)

            model = self.models[model_name]
            model.fit(X_tr_s, y_tr)
            y_pred = model.predict(X_te_s)

            results.append({
                "fraction": frac,
                "n_train": len(X_tr),
                "accuracy": accuracy_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            })

        df = pd.DataFrame(results)
        logger.info(f"Learning curves:\n{df.to_string()}")
        return df
