"""Item-level difficulty table (checklist item 5).

Merges Gemini and Groq per-item accuracy/MAE (already computed by
evaluate_run.py) with the cross-model agreement rate per item, so hardest/
easiest items are visible in one place instead of three separate CSVs.

Usage:
    python -m src.eval.item_difficulty
"""

import json
import logging

import pandas as pd

from src.config import RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    g = pd.read_csv(RESULTS_DIR / "evaluated" / "gemini-3.1-flash-lite_P2" / "per_item_metrics.csv")
    q = pd.read_csv(RESULTS_DIR / "evaluated" / "groq-openai_gpt-oss-120b-low_P2" / "per_item_metrics.csv")
    agree = pd.read_csv(RESULTS_DIR / "evaluated" / "cross_model_agreement" / "per_item.csv")

    g = g.rename(columns={"accuracy": "gemini_accuracy", "mae": "gemini_mae", "n": "gemini_n"})
    q = q.rename(columns={"accuracy": "groq_accuracy", "mae": "groq_mae", "n": "groq_n"})

    merged = g.merge(q[["question_id", "groq_accuracy", "groq_mae", "groq_n"]], on="question_id")
    merged = merged.merge(agree[["question_id", "pct_agree"]], on="question_id")
    merged["mean_accuracy"] = (merged["gemini_accuracy"] + merged["groq_accuracy"]) / 2
    merged["accuracy_gap_between_models"] = (merged["gemini_accuracy"] - merged["groq_accuracy"]).abs()
    merged = merged.sort_values("mean_accuracy")

    out_path = RESULTS_DIR / "evaluated" / "item_difficulty.csv"
    merged.to_csv(out_path, index=False)

    logger.info("Hardest 5 items (lowest mean accuracy, both models):")
    logger.info(
        merged[["question_id", "question_text", "gemini_accuracy", "groq_accuracy", "pct_agree"]]
        .head(5)
        .to_string(index=False)
    )
    logger.info("\nEasiest 5 items (highest mean accuracy, both models):")
    logger.info(
        merged[["question_id", "question_text", "gemini_accuracy", "groq_accuracy", "pct_agree"]]
        .tail(5)
        .to_string(index=False)
    )
    logger.info(
        "\nItems where the two models disagree most in accuracy (model-specific difficulty, not item difficulty):"
    )
    logger.info(
        merged.sort_values("accuracy_gap_between_models", ascending=False)[
            ["question_id", "gemini_accuracy", "groq_accuracy", "accuracy_gap_between_models"]
        ]
        .head(5)
        .to_string(index=False)
    )
    logger.info(f"\n✓ Written to {out_path}")


if __name__ == "__main__":
    main()
