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
