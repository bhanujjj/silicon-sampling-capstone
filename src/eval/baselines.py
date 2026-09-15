"""Baseline models for silicon sampling comparison."""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from .metrics import compute_metrics

logger = logging.getLogger(__name__)


class UniformRandomBaseline:
    """Predict uniformly random answer from all options."""

    def __init__(self, n_options: int):
        self.n_options = n_options

    def predict_proba(self, n_samples: int) -> np.ndarray:
        """Return uniform probabilities."""
        return np.ones((n_samples, self.n_options)) / self.n_options

    def predict(self, n_samples: int) -> np.ndarray:
        """Sample random predictions."""
        return np.random.randint(0, self.n_options, size=n_samples)


class NationalMarginalBaseline:
    """Predict the national marginal distribution (overall answer distribution)."""

    def __init__(self):
        self.marginal = None

    def fit(self, y_true: np.ndarray, weights: np.ndarray = None):
        """Fit the marginal distribution from training data.

        Pass W_WEIGHT here: the "national marginal" is a population-level
        claim, so it should be survey-weighted even though respondent-level
        predictions (this baseline's own .predict(), and any LLM's) are not.
        """
        n_bins = int(y_true.max()) + 1
        if weights is None:
            counts = np.bincount(y_true, minlength=n_bins)
        else:
            counts = np.bincount(y_true, weights=weights, minlength=n_bins)
        self.marginal = counts / counts.sum()

    def predict_proba(self, n_samples: int) -> np.ndarray:
        """Return marginal distribution for all samples."""
        return np.tile(self.marginal, (n_samples, 1))

    def predict(self, n_samples: int) -> np.ndarray:
        """Sample from marginal distribution."""
        return np.random.choice(len(self.marginal), size=n_samples, p=self.marginal)


class DemographicCellLookup:
    """
    Lookup table baseline: predict empirical distribution for demographic cell.

    If a respondent's demographic cell is not in training data,
    fall back to national marginal.
    """

    def __init__(self):
        self.cell_distributions = {}
        self.marginal = None

    def fit(self, df_train: pd.DataFrame, y_col: str, demo_cols: List[str]):
        """Fit empirical distributions per demographic cell."""
        self.demo_cols = demo_cols

        # Fit marginal
        marginal_counts = np.bincount(df_train[y_col])
        self.marginal = marginal_counts / marginal_counts.sum()

        # Fit per-cell distributions
        for cell_keys, group in df_train.groupby(demo_cols):
            y_values = group[y_col].values
            counts = np.bincount(y_values, minlength=len(self.marginal))
            dist = counts / counts.sum()
            self.cell_distributions[cell_keys] = dist

        logger.info(f"Fitted cell lookup with {len(self.cell_distributions)} cells")

    def predict_proba(self, df_test: pd.DataFrame) -> np.ndarray:
        """Get predicted probability distribution for each test sample."""
        probs = []

        for _, row in df_test.iterrows():
            cell_key = tuple(row[col] for col in self.demo_cols)

            if cell_key in self.cell_distributions:
                dist = self.cell_distributions[cell_key]
            else:
                dist = self.marginal

            probs.append(dist)

        return np.array(probs)

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        """Sample predictions from cell distributions."""
        probs = self.predict_proba(df_test)
        return np.array([np.random.choice(len(p), p=p) for p in probs])


class SupervisedClassifierBaseline:
    """Supervised classifier (logistic regression + gradient boosting) on demographics."""

    def __init__(self, model_type: str = "logistic"):
        """
        Args:
            model_type: "logistic" or "gbm"
        """
        if model_type == "logistic":
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == "gbm":
            self.model = GradientBoostingClassifier(random_state=42, n_estimators=100)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.model_type = model_type
        self.demo_cols = None
        self.label_encoders = {}

    def fit(self, df_train: pd.DataFrame, y_col: str, demo_cols: List[str]):
        """Fit supervised classifier on demographic features."""
        self.demo_cols = demo_cols

        # Prepare features (one-hot encode categorical)
        X = df_train[demo_cols].copy()

        # Encode categorical columns
        for col in X.select_dtypes(include=['object', 'string']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].fillna("Unknown"))
            self.label_encoders[col] = le

        # Fill NaNs in numeric columns
        X = X.fillna(X.mean(numeric_only=True))

        y = df_train[y_col].values

        self.model.fit(X, y)
        logger.info(f"Fitted {self.model_type} classifier on {len(df_train)} samples")

    def predict_proba(self, df_test: pd.DataFrame) -> np.ndarray:
        """Get predicted probabilities."""
        X = self._prepare_features(df_test)
        return self.model.predict_proba(X)

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        """Get predicted class labels."""
        X = self._prepare_features(df_test)
        return self.model.predict(X)

    def _prepare_features(self, df_test: pd.DataFrame) -> pd.DataFrame:
        """Prepare test features using fitted encoders."""
        X = df_test[self.demo_cols].copy()

        for col in X.select_dtypes(include=['object', 'string']).columns:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                X[col] = X[col].fillna("Unknown")
                X[col] = le.transform(X[col])
            else:
                X[col] = 0

        X = X.fillna(X.mean(numeric_only=True))

        return X


def evaluate_all_baselines(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    y_col: str = "answer",
    demo_cols: List[str] = None,
    weight_col: str = None,
    matched_demo_cols: List[str] = None,
) -> Dict[str, Dict]:
    """
    Evaluate all baseline models on test set.

    `demo_cols` drives cell_lookup (kept low-dimensional -- with the full
    14-attribute LLM feature set almost every cell would be a singleton on
    ~1.5k respondents, turning "lookup" into memorization). `matched_demo_cols`,
    when given, is used for the supervised baselines (logistic/gbm) instead,
    added as extra "logistic_matched"/"gbm_matched" entries -- this is the
    baseline that actually gets the same information the LLM's P2 prompt did.

    Returns: {
        "uniform": {"accuracy": ..., "mae": ..., ...},
        "marginal": {...},
        "cell_lookup": {...},
        "logistic": {...},
        "gbm": {...},
        "logistic_matched": {...},  # only if matched_demo_cols given
        "gbm_matched": {...},       # only if matched_demo_cols given
    }
    """
    if demo_cols is None:
        demo_cols = [col for col in df_train.columns if col not in [y_col, "respondent_id"]]

    y_test = df_test[y_col].values

    results = {}

    # 1. Uniform random
    n_options = len(np.unique(y_test))
    baseline_uniform = UniformRandomBaseline(n_options)
    y_pred = baseline_uniform.predict(len(y_test))
    probs = baseline_uniform.predict_proba(len(y_test))
    results["uniform"] = compute_metrics(y_test, y_pred, probs)

    # 2. National marginal -- survey-weighted, since this is the one baseline
    # standing in for a population-level claim rather than a per-respondent one
    baseline_marginal = NationalMarginalBaseline()
    weights = df_train[weight_col].values if weight_col else None
    baseline_marginal.fit(df_train[y_col].values, weights=weights)
    y_pred = baseline_marginal.predict(len(y_test))
    probs = baseline_marginal.predict_proba(len(y_test))
    results["marginal"] = compute_metrics(y_test, y_pred, probs)

    # 3. Demographic cell lookup
    baseline_cell = DemographicCellLookup()
    baseline_cell.fit(df_train, y_col, demo_cols)
    y_pred = baseline_cell.predict(df_test)
    probs = baseline_cell.predict_proba(df_test)
    results["cell_lookup"] = compute_metrics(y_test, y_pred, probs)

    # 4. Logistic regression
    baseline_logistic = SupervisedClassifierBaseline("logistic")
    baseline_logistic.fit(df_train, y_col, demo_cols)
    y_pred = baseline_logistic.predict(df_test)
    probs = baseline_logistic.predict_proba(df_test)
    results["logistic"] = compute_metrics(y_test, y_pred, probs)

    # 5. Gradient boosting
    baseline_gbm = SupervisedClassifierBaseline("gbm")
    baseline_gbm.fit(df_train, y_col, demo_cols)
    y_pred = baseline_gbm.predict(df_test)
    probs = baseline_gbm.predict_proba(df_test)
    results["gbm"] = compute_metrics(y_test, y_pred, probs)

    # 6/7. Matched-feature supervised baselines -- same demographic attributes
    # the LLM's P2 prompt actually saw, for a fair "did the LLM beat a model
    # with equal information" comparison.
    if matched_demo_cols:
        baseline_logistic_m = SupervisedClassifierBaseline("logistic")
        baseline_logistic_m.fit(df_train, y_col, matched_demo_cols)
        y_pred = baseline_logistic_m.predict(df_test)
        probs = baseline_logistic_m.predict_proba(df_test)
        results["logistic_matched"] = compute_metrics(y_test, y_pred, probs)

        baseline_gbm_m = SupervisedClassifierBaseline("gbm")
        baseline_gbm_m.fit(df_train, y_col, matched_demo_cols)
        y_pred = baseline_gbm_m.predict(df_test)
        probs = baseline_gbm_m.predict_proba(df_test)
        results["gbm_matched"] = compute_metrics(y_test, y_pred, probs)

    logger.info("\n=== BASELINE RESULTS ===")
    for baseline_name, metrics in results.items():
        logger.info(f"{baseline_name:20} Acc={metrics.get('accuracy', 0):.3f}, MAE={metrics.get('mae', 0):.3f}")

    return results
