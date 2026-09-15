"""Bootstrap confidence intervals for fidelity metrics."""

import logging
from typing import Callable, Dict

import numpy as np

from src.config import BOOTSTRAP_CI_LEVEL, N_BOOTSTRAP_RESAMPLES, RANDOM_SEED

logger = logging.getLogger(__name__)


def bootstrap_ci(
    values: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = N_BOOTSTRAP_RESAMPLES,
    ci_level: float = BOOTSTRAP_CI_LEVEL,
    seed: int = RANDOM_SEED,
) -> Dict[str, float]:
    """Resample-with-replacement CI for a per-row statistic (e.g. per-respondent
    correctness or absolute error).

    Returns {point_estimate, ci_low, ci_high, se}.
    """
    values = np.asarray(values)
    rng = np.random.RandomState(seed)
    n = len(values)
    point = float(statistic_fn(values))

    if n == 0:
        return {"point_estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "se": float("nan")}

    resample_stats = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = values[rng.randint(0, n, size=n)]
        resample_stats[i] = statistic_fn(sample)

    alpha = 1 - ci_level
    lo, hi = np.percentile(resample_stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point_estimate": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "se": float(resample_stats.std(ddof=1)),
        "n": n,
        "n_resamples": n_resamples,
    }


def bootstrap_diff_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = N_BOOTSTRAP_RESAMPLES,
    ci_level: float = BOOTSTRAP_CI_LEVEL,
    seed: int = RANDOM_SEED,
) -> Dict[str, float]:
    """CI on statistic(a) - statistic(b), paired by index (same respondents)."""
    values_a, values_b = np.asarray(values_a), np.asarray(values_b)
    assert len(values_a) == len(values_b), "paired bootstrap requires equal-length, aligned arrays"
    rng = np.random.RandomState(seed)
    n = len(values_a)
    point = float(statistic_fn(values_a) - statistic_fn(values_b))

    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        diffs[i] = statistic_fn(values_a[idx]) - statistic_fn(values_b[idx])

    alpha = 1 - ci_level
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point_estimate": point, "ci_low": float(lo), "ci_high": float(hi), "n": n, "n_resamples": n_resamples}
