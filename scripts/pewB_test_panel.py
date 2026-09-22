"""Paper-ready test panel: run the trained model on a small, fixed, diverse
set of real held-out respondents and questions, and print/save a markdown
table you can paste straight into the research paper.

This is illustrative, not the headline result -- the real accuracy/MAE/
fidelity-gap numbers for the paper come from running
`python -m src.report.evaluate_run --predictions results/predictions/pewB_<model>_fold<N>_P2.parquet`
against the FULL out-of-fold predictions file pewB_finetune.py already
writes (same schema Track A uses, same bootstrap-CI machinery). This
script exists for the "show a few concrete examples" section of a paper,
where a handful of real (demographics, question, true answer, model
answer) rows read far better than a table of aggregate metrics alone.

Usage:
    python scripts/pewB_test_panel.py \\
        --model-path pewB_run/model_fold0 --base-model openai/gpt-oss-20b \\
        --fold 0 --n-respondents 8 --n-items 5
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.config import DATA_PROCESSED
from src.prompts.verbalize_pew import verbalize_demographics
from scripts.pewB_inference import PewSimulator

logger = logging.getLogger(__name__)


def pick_diverse_respondents(df: pd.DataFrame, test_ids: list, n: int) -> pd.DataFrame:
    """Spread the panel across sex x urban/rural combinations where possible,
    rather than n arbitrary/consecutive rows, so a small paper table isn't
    accidentally all-male or all-urban.

    Deliberately avoids groupby(...).apply(...): whether that silently
    drops the grouping column from the result differs across pandas
    versions (confirmed different behavior between what's installed here
    and what a fresh `pip install pandas` can resolve to at the lab), so
    group membership is read off `.groups` directly instead -- no version-
    dependent apply() semantics to depend on.
    """
    sub = df[df["respondent_id"].isin(test_ids)].copy()
    if "QGEN" in sub.columns and "Urban" in sub.columns:
        strata = sub["QGEN"].astype(str) + "_" + sub["Urban"].astype(str)
        n_groups = max(strata.nunique(), 1)
        per_group = max(n // n_groups, 1)
        picked_idx = []
        for group_index in strata.groupby(strata).groups.values():
            picked_idx.extend(group_index[:per_group])
        picked = sub.loc[picked_idx].head(n)
        if len(picked) >= n:
            return picked
    return sub.head(n)


def build_panel(sim: PewSimulator, df: pd.DataFrame, test_ids: list, selected_items: list, n_respondents: int, n_items: int) -> list:
    respondents = pick_diverse_respondents(df, test_ids, n_respondents)
    items = selected_items[:n_items]
    rows = []
    for _, row in respondents.iterrows():
        demo = verbalize_demographics(row, sim.codebook)
        demo_str = ", ".join(f"{k}={v}" for k, v in demo.items() if v is not None)
        for question_id in items:
            true_code = row.get(question_id)
            if pd.isna(true_code):
                continue
            result = sim.answer(question_id, **demo)
            rows.append({
                "respondent_id": int(row["respondent_id"]),
                "demographics": demo_str,
                "question_id": question_id,
                "question_text": result["question_text"],
                "true_code": int(true_code),
                "pred_code": result["pred_code"],
                "correct": result["pred_code"] == int(true_code),
                "refusal": result["refusal"],
                "raw_output": result["raw_output"],
            })
    return rows


def to_markdown(rows: list) -> str:
    if not rows:
        return "_No panel rows produced -- check selected_items / test_ids._"
    header = "| Respondent | Demographics | Question | True | Predicted | Correct |\n"
    header += "|---|---|---|---|---|---|\n"
    lines = [header]
    for r in rows:
        lines.append(
            f"| {r['respondent_id']} | {r['demographics']} | {r['question_text']} ({r['question_id']}) | "
            f"{r['true_code']} | {r['pred_code']} | {'Yes' if r['correct'] else 'No'} |\n"
        )
    n = len(rows)
    n_correct = sum(1 for r in rows if r["correct"])
    lines.append(f"\n**Panel accuracy: {n_correct}/{n} ({100*n_correct/n:.1f}%)** -- illustrative subsample, not the headline result; see src.report.evaluate_run for the full out-of-fold metrics.\n")
    return "".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--base-model", default="openai/gpt-oss-20b")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-respondents", type=int, default=8)
    ap.add_argument("--n-items", type=int, default=5)
    ap.add_argument("--output-dir", default="./pewB_run/predictions")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)

    df = pd.read_parquet(DATA_PROCESSED / "pew_india_2021.parquet")
    if "respondent_id" not in df.columns:
        df["respondent_id"] = range(len(df))
    with open(DATA_PROCESSED / "pew_selected_items.json") as f:
        selected_items = json.load(f)["selected_items"]
    with open(DATA_PROCESSED / "pew_folds.json") as f:
        folds = json.load(f)["folds"]
    test_ids = folds[args.fold]["test"]

    if not selected_items:
        logger.error("0 selected items -- see src/data/build_pew_codebook.py. Nothing to build a panel from.")
        sys.exit(1)

    sim = PewSimulator.load(args.model_path, args.base_model)
    rows = build_panel(sim, df, test_ids, selected_items, args.n_respondents, args.n_items)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paper_test_panel.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    md = to_markdown(rows)
    (output_dir / "paper_test_panel.md").write_text(md)

    print(md)
    logger.info(f"Saved to {output_dir}/paper_test_panel.{{md,json}}")


if __name__ == "__main__":
    main()
