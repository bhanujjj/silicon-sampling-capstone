"""Cross-model agreement analysis (checklist item 4).

Compares Gemini and Groq predictions on the (respondent, item) pairs both
models actually answered, at the 15-item width. Three questions:
  1. How often do the two models agree with each other (regardless of truth)?
  2. Is agreement higher on items/respondents where both are also more
     accurate (i.e. does the panel converge on truth, or just converge)?
  3. Per-item and per-subgroup agreement rates, to see where the models
     diverge most.

Usage:
    python -m src.eval.cross_model_agreement
"""

import json
import logging

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

GEMINI_PATH = RESULTS_DIR / "predictions" / "gemini-3.1-flash-lite_P2.parquet"
GROQ_PATH = RESULTS_DIR / "predictions" / "groq-openai_gpt-oss-120b-low_P2.parquet"
OUT_DIR = RESULTS_DIR / "evaluated" / "cross_model_agreement"


def main():
    g = pd.read_parquet(GEMINI_PATH)[
        ["respondent_id", "question_id", "true_code_idx", "pred_code_idx",
         "sg_urban_rural", "sg_income", "sg_education", "sg_sex", "sg_age_band", "sg_region"]
    ].rename(columns={"pred_code_idx": "pred_gemini", "true_code_idx": "true_code_idx_g"})

    q = pd.read_parquet(GROQ_PATH)[
        ["respondent_id", "question_id", "true_code_idx", "pred_code_idx"]
    ].rename(columns={"pred_code_idx": "pred_groq", "true_code_idx": "true_code_idx_q"})

    merged = g.merge(q, on=["respondent_id", "question_id"], how="inner")
    assert (merged["true_code_idx_g"] == merged["true_code_idx_q"]).all(), "truth mismatch after merge"
    merged["true_code_idx"] = merged["true_code_idx_g"]
    merged = merged.drop(columns=["true_code_idx_g", "true_code_idx_q"])

    merged["agree"] = merged["pred_gemini"] == merged["pred_groq"]
    merged["gemini_correct"] = merged["pred_gemini"] == merged["true_code_idx"]
    merged["groq_correct"] = merged["pred_groq"] == merged["true_code_idx"]
    merged["both_correct"] = merged["gemini_correct"] & merged["groq_correct"]
    merged["both_wrong"] = (~merged["gemini_correct"]) & (~merged["groq_correct"])

    n = len(merged)
    overall = {
        "n_paired_predictions": int(n),
        "pct_agree": float(merged["agree"].mean()),
        "pct_agree_and_both_correct": float(merged["both_correct"].mean()),
        "pct_agree_and_both_wrong": float((merged["agree"] & merged["both_wrong"]).mean()),
        "pct_disagree": float((~merged["agree"]).mean()),
        "pct_disagree_gemini_right": float((~merged["agree"] & merged["gemini_correct"]).mean()),
        "pct_disagree_groq_right": float((~merged["agree"] & merged["groq_correct"]).mean()),
        "pct_disagree_both_wrong": float((~merged["agree"] & merged["both_wrong"]).mean()),
        "gemini_accuracy_on_overlap": float(merged["gemini_correct"].mean()),
        "groq_accuracy_on_overlap": float(merged["groq_correct"].mean()),
        # Cohen's kappa vs. chance agreement, since two models could "agree"
        # a lot just by both defaulting to the same modal answer option.
    }

    # Cohen's kappa against chance agreement (accounts for both models
    # leaning on the same popular answer option, which inflates raw agreement)
    def cohens_kappa(a, b):
        labels = sorted(set(a) | set(b))
        idx = {l: i for i, l in enumerate(labels)}
        k = len(labels)
        cm = np.zeros((k, k))
        for x, y in zip(a, b):
            cm[idx[x], idx[y]] += 1
        cm = cm / cm.sum()
        po = np.trace(cm)
        pe = (cm.sum(axis=1) * cm.sum(axis=0)).sum()
        return (po - pe) / (1 - pe) if pe < 1 else float("nan")

    overall["cohens_kappa"] = float(cohens_kappa(merged["pred_gemini"], merged["pred_groq"]))

    per_item = (
        merged.groupby("question_id")
        .agg(
            n=("agree", "size"),
            pct_agree=("agree", "mean"),
            gemini_acc=("gemini_correct", "mean"),
            groq_acc=("groq_correct", "mean"),
        )
        .reset_index()
        .sort_values("pct_agree")
    )

    subgroup_rows = []
    for axis in ["sg_urban_rural", "sg_income", "sg_education", "sg_sex", "sg_age_band", "sg_region"]:
        for cat, grp in merged.groupby(axis):
            if len(grp) < 15:
                continue
            subgroup_rows.append(
                {
                    "axis": axis,
                    "category": cat,
                    "n": len(grp),
                    "pct_agree": float(grp["agree"].mean()),
                    "gemini_acc": float(grp["gemini_correct"].mean()),
                    "groq_acc": float(grp["groq_correct"].mean()),
                }
            )
    per_subgroup = pd.DataFrame(subgroup_rows).sort_values(["axis", "pct_agree"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "overall.json", "w") as f:
        json.dump(overall, f, indent=2)
    per_item.to_csv(OUT_DIR / "per_item.csv", index=False)
    per_subgroup.to_csv(OUT_DIR / "per_subgroup.csv", index=False)

    logger.info(json.dumps(overall, indent=2))
    logger.info("\nPer-item agreement (lowest first):\n%s", per_item.to_string(index=False))
    logger.info("\nPer-subgroup agreement:\n%s", per_subgroup.to_string(index=False))
    logger.info(f"\n✓ Written to {OUT_DIR}")


if __name__ == "__main__":
    main()
