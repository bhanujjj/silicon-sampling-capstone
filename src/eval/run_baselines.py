"""Phase 2: run every baseline on every selected item, out-of-fold.

This is the bar an LLM has to clear. Per the project's own design rule
(README, "Key design decisions"), this must exist before any LLM inference is
worth running -- otherwise there is nothing to compare Phase 3 predictions to.

Usage:
    python -m src.eval.run_baselines
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, RESULTS_DIR
from src.eval.baselines import evaluate_all_baselines
from src.eval.bootstrap import bootstrap_ci
from src.prompts.verbalize import load_codebook, verbalize_item

logger = logging.getLogger(__name__)

DEMO_COLS = ["H_URBRURAL", "Q288", "Q275R", "Q260", "Q262", "N_REGION_ISO"]

# The full attribute set the LLM's P2 prompt is verbalized from
# (src/prompts/verbalize.py: DEMOGRAPHIC_CODE_COLS + age/children/urban-rural/
# income/region/town-size/language). Used only for the "matched" supervised
# baselines -- cell_lookup stays on DEMO_COLS above since 14 dims of lookup
# on ~1.5k respondents collapses to memorization, not a real baseline.
MATCHED_DEMO_COLS = [
    "Q260", "Q273", "Q275R", "Q279", "Q281", "Q287", "Q289",
    "Q262", "Q274", "H_URBRURAL", "Q288", "N_REGION_ISO", "G_TOWNSIZE", "LNGE_ISO",
]


def _encode_item(df: pd.DataFrame, question_id: str, codebook: dict) -> pd.Series:
    """Map an item's raw WVS codes onto 0..k-1, in the same order used to
    build its answer-options text -- so baseline predictions and (later) LLM
    predictions share one label space per item."""
    item = verbalize_item(question_id, codebook)
    return df[question_id].map(item["code_to_index"])


def run_all_baselines(
    df: pd.DataFrame,
    folds: dict,
    selected_items: list,
    codebook: dict,
) -> pd.DataFrame:
    """Out-of-fold baseline evaluation for every item.

    Returns one row per (item, baseline) with accuracy/MAE averaged over the
    folds that item had enough data to fit in.
    """
    df = df.copy()
    all_demo_cols = sorted(set(DEMO_COLS) | set(MATCHED_DEMO_COLS))
    demo_features = df[["respondent_id"] + all_demo_cols].copy()
    demo_features["N_REGION_ISO"] = demo_features["N_REGION_ISO"].astype(str)

    item_rows = []

    for q_idx, question_id in enumerate(selected_items):
        y_full = _encode_item(df, question_id, codebook)
        valid_mask = y_full.notna()

        fold_metrics = {
            name: []
            for name in ["uniform", "marginal", "cell_lookup", "logistic", "gbm", "logistic_matched", "gbm_matched"]
        }

        for fold in folds["folds"]:
            train_ids = [rid for rid in fold["train"] if valid_mask.loc[df["respondent_id"] == rid].any()]
            test_ids = [rid for rid in fold["test"] if valid_mask.loc[df["respondent_id"] == rid].any()]

            train_mask = df["respondent_id"].isin(train_ids) & valid_mask
            test_mask = df["respondent_id"].isin(test_ids) & valid_mask
            if train_mask.sum() < 30 or test_mask.sum() < 10:
                continue

            df_train = demo_features[train_mask].copy()
            df_train["answer"] = y_full[train_mask].astype(int).values
            df_train["W_WEIGHT"] = df.loc[train_mask, "W_WEIGHT"].values

            df_test = demo_features[test_mask].copy()
            df_test["answer"] = y_full[test_mask].astype(int).values

            try:
                results = evaluate_all_baselines(
                    df_train, df_test, y_col="answer", demo_cols=DEMO_COLS,
                    weight_col="W_WEIGHT", matched_demo_cols=MATCHED_DEMO_COLS,
                )
            except Exception as exc:  # a degenerate fold (e.g. single-class train) shouldn't kill the run
                logger.warning(f"{question_id} fold {fold['fold']}: baseline fit failed ({exc}), skipping")
                continue

            for name, metrics in results.items():
                fold_metrics[name].append(metrics)

        if not any(fold_metrics.values()):
            logger.warning(f"{question_id}: no usable folds, skipping")
            continue

        for name, per_fold in fold_metrics.items():
            if not per_fold:
                continue
            accs = np.array([m["accuracy"] for m in per_fold])
            maes = np.array([m["mae"] for m in per_fold])
            item_rows.append(
                {
                    "question_id": question_id,
                    "baseline": name,
                    "n_folds": len(per_fold),
                    "accuracy_mean": float(accs.mean()),
                    "accuracy_std": float(accs.std()),
                    "mae_mean": float(maes.mean()),
                    "n_options": int(y_full.nunique()),
                }
            )

        if (q_idx + 1) % 20 == 0:
            logger.info(f"  ...{q_idx + 1}/{len(selected_items)} items done")

    item_results = pd.DataFrame(item_rows)
    return item_results


def summarize(item_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate item-level results into one headline row per baseline, with
    a bootstrap CI computed over per-item accuracy (the item is the
    resampling unit here, matching "how consistent is this baseline across
    the 144-item battery" rather than pretending item accuracies are iid
    respondent-level draws)."""
    rows = []
    for name, group in item_results.groupby("baseline"):
        ci = bootstrap_ci(group["accuracy_mean"].to_numpy())
        rows.append(
            {
                "baseline": name,
                "n_items": len(group),
                "mean_accuracy": ci["point_estimate"],
                "accuracy_ci_low": ci["ci_low"],
                "accuracy_ci_high": ci["ci_high"],
                "mean_mae": group["mae_mean"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_accuracy", ascending=False).reset_index(drop=True)


def main():
    df = pd.read_parquet(DATA_PROCESSED / "ind_wvs7.parquet")
    if "respondent_id" not in df.columns:
        df["respondent_id"] = range(len(df))

    with open(DATA_PROCESSED / "folds.json") as f:
        folds = json.load(f)
    with open(DATA_PROCESSED / "selected_items.json") as f:
        selected = json.load(f)["selected_items"]

    codebook = load_codebook()

    logger.info(f"Running baselines for {len(selected)} items across {folds['n_folds']} folds...")
    item_results = run_all_baselines(df, folds, selected, codebook)

    output_path = RESULTS_DIR / "baselines_by_item.csv"
    item_results.to_csv(output_path, index=False)
    logger.info(f"✓ Saved per-item baseline results to {output_path}")

    summary = summarize(item_results)
    summary_path = RESULTS_DIR / "baselines_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info(f"\n=== BASELINE SUMMARY (mean over {item_results['question_id'].nunique()} items) ===")
    logger.info("\n" + summary.to_string(index=False))
    logger.info(f"✓ Saved summary to {summary_path}")

    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
