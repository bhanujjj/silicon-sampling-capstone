"""Select Pew survey items for training/eval -- the Pew-side counterpart to
select_items.py, reusing its generic scoring helpers.

One extra, Pew-specific rejection reason sits in front of all of WVS's
existing criteria: "unverified_wording". An item whose codebook entry has
"verified": false (see build_pew_codebook.py's docstring) has no confirmed
question text or response-option labels -- selecting it anyway would train
on / report a fabricated-looking Pew question, so it is rejected before
any statistical screening runs, regardless of how good its numbers look.
This means selected_items will legitimately be EMPTY until Pew's own
codebook document is parsed and merged in (see parse_pew_recode_syntax.py),
which is the correct state, not a bug.
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.config import DATA_PROCESSED, DATA_REFERENCE, MAX_MISSINGNESS_PCT, MIN_MODAL_ENTROPY, MIN_RESPONSE_SCALE_SIZE
from src.data.select_items import compute_missingness, compute_modal_entropy, compute_scale_entropy

logger = logging.getLogger(__name__)

REJECTION_REASONS = [
    "no_codebook_entry",
    "unverified_wording",
    "demographic_block",
    "scale_too_small",
    "non_ordinal_codes",
    "observed_off_scale",
    "high_missingness",
    "low_entropy",
]


def load_codebook(path: Path = None) -> dict:
    if path is None:
        path = DATA_REFERENCE / "pew_codebook.json"
    if not path.exists():
        raise FileNotFoundError(f"Codebook not found at {path}\nGenerate it with: python -m src.data.build_pew_codebook")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["questions"]


def screen_item(question_id: str, series: pd.Series, codebook: dict) -> dict:
    result = {"question_id": question_id, "passed": False, "rejected_for": None}

    meta = codebook.get(question_id)
    if meta is None:
        result["rejected_for"] = "no_codebook_entry"
        return result

    result.update({"title": meta["title"], "block": meta["block"], "n_scale": meta["n_scale"]})

    if not meta.get("verified", False):
        result["rejected_for"] = "unverified_wording"
        return result

    if meta["block"] == "DEMOGRAPHIC":
        result["rejected_for"] = "demographic_block"
        return result

    if meta["n_scale"] < MIN_RESPONSE_SCALE_SIZE:
        result["rejected_for"] = "scale_too_small"
        return result

    if not meta["contiguous"]:
        result["rejected_for"] = "non_ordinal_codes"
        return result

    valid_codes = {float(code) for code in meta["valid_codes"]}
    observed = set(series.dropna().unique())
    off_scale = sorted(observed - valid_codes)
    if off_scale:
        result["rejected_for"] = "observed_off_scale"
        result["off_scale_values"] = [float(v) for v in off_scale]
        return result

    missingness = compute_missingness(series)
    result["missingness_pct"] = round(missingness, 2)
    if missingness > MAX_MISSINGNESS_PCT:
        result["rejected_for"] = "high_missingness"
        return result

    modal_entropy = compute_modal_entropy(series)
    result["modal_entropy"] = round(modal_entropy, 4)
    result["scale_entropy"] = round(compute_scale_entropy(series, meta["n_scale"]), 4)
    result["n_observed_categories"] = int(series.nunique())
    if modal_entropy < MIN_MODAL_ENTROPY:
        result["rejected_for"] = "low_entropy"
        return result

    result["scale_labels"] = meta["valid_codes"]
    result["passed"] = True
    return result


def screen_all(df: pd.DataFrame, codebook: dict = None) -> list:
    if codebook is None:
        codebook = load_codebook()
    question_cols = [c for c in df.columns if c in codebook]
    return [screen_item(c, df[c], codebook) for c in question_cols]


def select_items(df: pd.DataFrame, n_target: int = None, codebook: dict = None) -> list:
    logger.info("Starting Pew item selection...")
    results = screen_all(df, codebook)
    passed = [r for r in results if r["passed"]]
    logger.info(f"Evaluated {len(results)} candidate columns; {len(passed)} passed all criteria")

    rejected = [r for r in results if not r["passed"]]
    counts = pd.Series([r["rejected_for"] for r in rejected]).value_counts() if rejected else pd.Series(dtype=int)
    for reason in REJECTION_REASONS:
        if reason in counts:
            logger.info(f"  rejected [{reason}]: {counts[reason]}")

    passed.sort(key=lambda r: r["modal_entropy"], reverse=True)
    if n_target is not None:
        passed = passed[:n_target]

    return [r["question_id"] for r in passed]


def main(input_path: Path = None, output_path: Path = None):
    input_path = input_path or DATA_PROCESSED / "pew_india_2021.parquet"
    output_path = output_path or DATA_PROCESSED / "pew_selected_items.json"

    if not input_path.exists():
        logger.error(f"Input data not found: {input_path}\nRun: python -m src.data.load_pew")
        return False

    df = pd.read_parquet(input_path)
    codebook = load_codebook()
    results = screen_all(df, codebook)
    passed = sorted([r for r in results if r["passed"]], key=lambda r: r["modal_entropy"], reverse=True)
    selected = [r["question_id"] for r in passed]

    rejected = [r for r in results if not r["passed"]]
    ledger = pd.Series([r["rejected_for"] for r in rejected]).value_counts().to_dict() if rejected else {}

    output = {
        "selected_items": selected,
        "n_items": len(selected),
        "n_candidates_evaluated": len(results),
        "rejection_ledger": {r: int(ledger.get(r, 0)) for r in REJECTION_REASONS},
        "codebook_source": "Pew Research Center 'Religion in India' (2021)",
    }
    if not selected:
        logger.warning(
            f"0 items selected -- {ledger.get('unverified_wording', 0)} rejected for unverified_wording. "
            "This is expected until Pew's own codebook document is parsed and merged via "
            "src.data.build_pew_codebook --labels. See that module's docstring."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(selected)} selected items to {output_path}")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-target", type=int, default=None)
    args = parser.parse_args()
    exit(0 if main() else 1)
