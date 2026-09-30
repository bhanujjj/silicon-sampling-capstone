"""Shared scoring for every Pew method (baselines and the fine-tuned LLM),
so all methods in the paper are measured by the SAME code.

Input per (respondent, item) row: true option index, and a probability
vector over that item's options. Hard-prediction methods pass a one-hot row.

Metrics:
  accuracy   argmax(proba) == truth
  mae        |argmax index - true index| (ordinal steps)
  nll        -log proba[truth] (clipped) -- only meaningful for soft methods
  tv         total variation distance between the item's mean predicted
             distribution and the true answer distribution (0 = perfect
             match of the population spread, 1 = disjoint). Averaged over items.
Bootstrap CIs resample RESPONDENTS (not rows), because one respondent's
answers to different items are not independent.
"""

import numpy as np
import pandas as pd


def _item_metrics(true_idx, proba):
    pred = proba.argmax(axis=1)
    n_opt = proba.shape[1]
    true_hist = np.bincount(true_idx, minlength=n_opt) / len(true_idx)
    return {
        "accuracy": float((pred == true_idx).mean()),
        "mae": float(np.abs(pred - true_idx).mean()),
        "nll": float(-np.log(np.clip(proba[np.arange(len(true_idx)), true_idx], 1e-6, 1)).mean()),
        "tv": float(0.5 * np.abs(proba.mean(axis=0) - true_hist).sum()),
    }


def score_rows(df: pd.DataFrame) -> dict:
    """df needs columns: question_id, true_idx (int), proba (list/array per row).
    Returns micro accuracy/mae/nll over all rows, and tv averaged over items."""
    accs, maes, nlls, tvs, ns = [], [], [], [], []
    for _, g in df.groupby("question_id"):
        m = _item_metrics(g["true_idx"].to_numpy(), np.stack(g["proba"].to_numpy()))
        accs.append(m["accuracy"]); maes.append(m["mae"]); nlls.append(m["nll"]); tvs.append(m["tv"]); ns.append(len(g))
    w = np.array(ns) / sum(ns)
    return {
        "accuracy": float(np.dot(w, accs)), "mae": float(np.dot(w, maes)),
        "nll": float(np.dot(w, nlls)), "tv": float(np.mean(tvs)), "n_rows": int(sum(ns)),
    }


def bootstrap(df: pd.DataFrame, n_boot: int = 200, seed: int = 0) -> dict:
    """95% CI per metric, resampling respondents."""
    rng = np.random.default_rng(seed)
    by_resp = {r: g for r, g in df.groupby("respondent_id")}
    ids = np.array(list(by_resp))
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        stats.append(score_rows(pd.concat([by_resp[r] for r in pick], ignore_index=True)))
    out = {}
    for k in ("accuracy", "mae", "nll", "tv"):
        v = np.array([s[k] for s in stats])
        out[k] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return out
