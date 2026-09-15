"""Turn a Track A predictions file into the numbers and examples the
comparison report needs: overall metrics, per-subgroup fidelity gaps, and
a versus-baselines comparison.

Usage:
    python -m src.report.evaluate_run --predictions results/predictions/gemini-3.5-flash-lite_P2.parquet
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR
from src.eval.bootstrap import bootstrap_ci
from src.eval.metrics import compute_metrics
from src.eval.subgroups import SUBGROUP_AXES, fidelity_gap_report, metrics_by_subgroup_axis

logger = logging.getLogger(__name__)


def overall_metrics(predictions: pd.DataFrame) -> dict:
    """Headline numbers over every answered (respondent, item) pair, with a
    bootstrap CI on accuracy so the number carries its own uncertainty."""
    answered = predictions[predictions["pred_code_idx"].notna()].copy()
    y_true = answered["true_code_idx"].to_numpy()
    y_pred = answered["pred_code_idx"].to_numpy()

    correct = (y_true == y_pred).astype(float)
    ci = bootstrap_ci(correct)

    mae = float(np.abs(y_true - y_pred).mean())
    refusal_rate = float(predictions["pred_code_idx"].isna().mean())

    real_logprob_rate = None
    if "real_logprobs" in predictions.columns and predictions["real_logprobs"].notna().any():
        real_logprob_rate = float(predictions["real_logprobs"].fillna(False).mean())

    return {
        "n_predictions": len(predictions),
        "n_answered": len(answered),
        "refusal_rate": refusal_rate,
        "accuracy": ci["point_estimate"],
        "accuracy_ci_low": ci["ci_low"],
        "accuracy_ci_high": ci["ci_high"],
        "mae": mae,
        "real_logprob_rate": real_logprob_rate,
    }


def per_item_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for question_id, group in predictions.groupby("question_id"):
        answered = group[group["pred_code_idx"].notna()]
        if len(answered) == 0:
            continue
        y_true = answered["true_code_idx"].to_numpy()
        y_pred = answered["pred_code_idx"].to_numpy()
        acc = float((y_true == y_pred).mean())
        mae = float(np.abs(y_true - y_pred).mean())
        rows.append(
            {
                "question_id": question_id,
                "question_text": group["question_text"].iloc[0],
                "n": len(answered),
                "accuracy": acc,
                "mae": mae,
            }
        )
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)


def versus_baselines(predictions: pd.DataFrame, baselines_summary_path: Path) -> pd.DataFrame:
    """Line up the LLM's per-item accuracy against every baseline's, item by
    item -- the comparison that answers "does silicon sampling add anything".
    """
    llm_items = per_item_metrics(predictions)[["question_id", "accuracy"]].rename(columns={"accuracy": "llm"})

    by_item_path = baselines_summary_path.parent / "baselines_by_item.csv"
    if not by_item_path.exists():
        logger.warning(f"{by_item_path} not found -- run src.eval.run_baselines first")
        return llm_items

    baseline_items = pd.read_csv(by_item_path)
    pivot = baseline_items.pivot_table(index="question_id", columns="baseline", values="accuracy_mean")
    merged = llm_items.merge(pivot, on="question_id", how="inner")

    n_beats_marginal = (merged["llm"] > merged.get("marginal", -1)).sum()
    n_beats_cell = (merged["llm"] > merged.get("cell_lookup", -1)).sum()
    logger.info(
        f"LLM beats national-marginal baseline on {n_beats_marginal}/{len(merged)} items, "
        f"beats demographic-cell-lookup on {n_beats_cell}/{len(merged)} items"
    )
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", default=None, type=Path)
    args = parser.parse_args()

    predictions = pd.read_parquet(args.predictions)
    out_dir = args.output_dir or (RESULTS_DIR / "evaluated" / args.predictions.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    overall = overall_metrics(predictions)
    logger.info(f"Overall: {json.dumps(overall, indent=2)}")
    json.dump(overall, open(out_dir / "overall_metrics.json", "w"), indent=2)

    items = per_item_metrics(predictions)
    items.to_csv(out_dir / "per_item_metrics.csv", index=False)

    subgroup_tables = {}
    for axis in SUBGROUP_AXES:
        if axis in predictions.columns:
            answered = predictions[predictions["pred_code_idx"].notna()]
            subgroup_tables[axis] = metrics_by_subgroup_axis(answered, axis)
    for axis, table in subgroup_tables.items():
        table.to_csv(out_dir / f"subgroup_{axis}.csv", index=False)

    gap_report = fidelity_gap_report(predictions[predictions["pred_code_idx"].notna()])
    gap_report.to_csv(out_dir / "fidelity_gaps.csv", index=False)
    logger.info(f"Fidelity gaps:\n{gap_report.to_string(index=False)}")

    comparison = versus_baselines(predictions, RESULTS_DIR / "baselines_summary.csv")
    comparison.to_csv(out_dir / "versus_baselines.csv", index=False)

    logger.info(f"✓ All evaluation artifacts written to {out_dir}")
    return out_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
