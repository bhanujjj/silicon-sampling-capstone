"""Subgroup slicing and fidelity-gap computation.

Slices are built from the raw demographic columns rather than the model's
inputs, so the same slice definition applies identically to baselines and to
any LLM's predictions.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from src.eval.bootstrap import bootstrap_ci
from src.eval.metrics import compute_metrics, exact_match_accuracy, mean_absolute_error

logger = logging.getLogger(__name__)


def assign_subgroups(df: pd.DataFrame) -> pd.DataFrame:
    """Add subgroup label columns to a respondent-level frame.

    Expects the raw WVS columns (H_URBRURAL, Q288, Q275R, Q260, Q262,
    N_REGION_ISO) to be present -- call on ind_wvs7.parquet or a frame merged
    from it, not on a predictions-only frame.
    """
    out = df.copy()

    out["sg_urban_rural"] = out["H_URBRURAL"].map({1.0: "Urban", 2.0: "Rural"})

    out["sg_income"] = pd.qcut(out["Q288"], q=3, labels=["Low", "Mid", "High"], duplicates="drop")

    out["sg_education"] = out["Q275R"].map({1.0: "Lower", 2.0: "Middle", 3.0: "Higher"})

    out["sg_sex"] = out["Q260"].map({1.0: "Male", 2.0: "Female"})

    out["sg_age_band"] = pd.cut(
        out["Q262"], bins=[0, 24, 34, 44, 54, 64, 200],
        labels=["16-24", "25-34", "35-44", "45-54", "55-64", "65+"],
    )

    out["sg_region"] = out["N_REGION_ISO"].astype("Int64").astype(str)

    return out


SUBGROUP_AXES = ["sg_urban_rural", "sg_income", "sg_education", "sg_sex", "sg_age_band", "sg_region"]

MIN_CELL_N = 30  # below this, report the slice but flag it as underpowered


def metrics_by_subgroup_axis(
    predictions: pd.DataFrame,
    axis: str,
    y_true_col: str = "true_code_idx",
    y_pred_col: str = "pred_code_idx",
) -> pd.DataFrame:
    """Per-category accuracy/MAE (+ bootstrap CI on accuracy) for one subgroup axis.

    Deliberately skips NLL/JSD here: a subgroup slice pools predictions across
    many different survey items, and those items have different answer-scale
    sizes (4-point, 5-point, 10-point, 11-point...). There is no single shared
    probability-vector shape to compute a pooled distributional metric over --
    those belong in per_item_metrics(), where every row shares one item's scale.
    """
    rows = []
    for category, group in predictions.groupby(axis, observed=True):
        y_true = group[y_true_col].to_numpy()
        y_pred = group[y_pred_col].to_numpy()

        m = compute_metrics(y_true, y_pred)
        correct = (y_true == y_pred).astype(float)
        ci = bootstrap_ci(correct)

        rows.append(
            {
                "axis": axis,
                "category": str(category),
                "n": len(group),
                "underpowered": len(group) < MIN_CELL_N,
                "accuracy": m["accuracy"],
                "accuracy_ci_low": ci["ci_low"],
                "accuracy_ci_high": ci["ci_high"],
                "mae": m["mae"],
                "jsd": m.get("jsd"),
            }
        )
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)


def fidelity_gap_report(predictions: pd.DataFrame, axes: List[str] = None) -> pd.DataFrame:
    """Fidelity gap (best - worst category accuracy) for every subgroup axis.

    Cells below MIN_CELL_N are excluded from the max/min so a single tiny
    category can't manufacture a dramatic-looking gap.
    """
    if axes is None:
        axes = SUBGROUP_AXES

    rows = []
    for axis in axes:
        if axis not in predictions.columns:
            continue
        table = metrics_by_subgroup_axis(predictions, axis)
        powered = table[~table["underpowered"]]
        if len(powered) < 2:
            continue
        best = powered.loc[powered["accuracy"].idxmax()]
        worst = powered.loc[powered["accuracy"].idxmin()]
        rows.append(
            {
                "axis": axis,
                "n_categories": len(table),
                "n_categories_powered": len(powered),
                "best_category": best["category"],
                "best_accuracy": best["accuracy"],
                "worst_category": worst["category"],
                "worst_accuracy": worst["accuracy"],
                "fidelity_gap": best["accuracy"] - worst["accuracy"],
            }
        )
    return pd.DataFrame(rows).sort_values("fidelity_gap", ascending=False).reset_index(drop=True)


def delta_gap(gap_zero_shot: pd.DataFrame, gap_fine_tuned: pd.DataFrame) -> pd.DataFrame:
    """Δ_gap = gap(fine-tuned) - gap(zero-shot) per axis. Negative = fairer."""
    merged = gap_zero_shot.merge(gap_fine_tuned, on="axis", suffixes=("_zs", "_ft"))
    merged["delta_gap"] = merged["fidelity_gap_ft"] - merged["fidelity_gap_zs"]
    return merged[["axis", "fidelity_gap_zs", "fidelity_gap_ft", "delta_gap"]]
