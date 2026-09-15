"""Investigate the sex-axis fidelity gap non-replication (checklist item 6).

Gemini shows an 11.1-pt male-vs-female accuracy gap at 15 items; Groq shows
5.6 pts. Same direction (male respondents predicted more accurately in both),
different magnitude. This breaks the gap down per-item and checks whether
the three gender-attitude items (Q29/Q31/Q32 -- "men make better leaders/
executives", "housewife as fulfilling") are driving it, since those are the
items where a model's own prior about gender roles could bias predictions
by respondent sex rather than genuinely reading the demographic signal.

Usage:
    python -m src.eval.sex_gap_investigation
"""

import json
import logging

import pandas as pd

from src.config import RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

GENDER_ATTITUDE_ITEMS = {"Q29", "Q31", "Q32"}


def per_item_sex_accuracy(pred_path, model_name):
    df = pd.read_parquet(pred_path)
    df["correct"] = df["pred_code_idx"] == df["true_code_idx"]
    rows = []
    for (qid, sex), grp in df.groupby(["question_id", "sg_sex"]):
        if len(grp) < 5:
            continue
        rows.append({"model": model_name, "question_id": qid, "sex": sex, "n": len(grp), "accuracy": grp["correct"].mean()})
    return pd.DataFrame(rows)


def main():
    gemini = per_item_sex_accuracy(RESULTS_DIR / "predictions" / "gemini-3.1-flash-lite_P2.parquet", "gemini")
    groq = per_item_sex_accuracy(RESULTS_DIR / "predictions" / "groq-openai_gpt-oss-120b-low_P2.parquet", "groq")
    combined = pd.concat([gemini, groq], ignore_index=True)

    pivot = combined.pivot_table(index=["model", "question_id"], columns="sex", values="accuracy").reset_index()
    if "Male" in pivot.columns and "Female" in pivot.columns:
        pivot["male_minus_female"] = pivot["Male"] - pivot["Female"]
    pivot["is_gender_attitude_item"] = pivot["question_id"].isin(GENDER_ATTITUDE_ITEMS)
    pivot = pivot.sort_values(["model", "male_minus_female"], ascending=[True, False])

    out_path = RESULTS_DIR / "evaluated" / "sex_gap_per_item.csv"
    pivot.to_csv(out_path, index=False)

    logger.info("Per-item male-minus-female accuracy gap, by model:\n%s", pivot.to_string(index=False))

    summary = {}
    for model in ["gemini", "groq"]:
        sub = pivot[pivot["model"] == model]
        gender_items = sub[sub["is_gender_attitude_item"]]
        other_items = sub[~sub["is_gender_attitude_item"]]
        summary[model] = {
            "mean_gap_gender_attitude_items": float(gender_items["male_minus_female"].mean()) if len(gender_items) else None,
            "mean_gap_other_items": float(other_items["male_minus_female"].mean()) if len(other_items) else None,
            "n_gender_attitude_items": int(len(gender_items)),
            "n_other_items": int(len(other_items)),
        }

    logger.info("\nSummary -- is the gap concentrated in the 3 gender-attitude items?\n%s", json.dumps(summary, indent=2))
    with open(RESULTS_DIR / "evaluated" / "sex_gap_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\n✓ Written to {out_path}")


if __name__ == "__main__":
    main()
