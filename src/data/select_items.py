"""Select survey items for the study, grounded in the official WVS-7 codebook.

Every question is screened against the WVS-7 Variables Report V6.0 (parsed by
src/data/parse_codebook.py) rather than hand-written scale metadata, and every
question present in the data is evaluated -- there is no whitelist.

Screening order (first failure wins, so the rejection counts partition the pool):

    1. no_codebook_entry  -- technical/country-specific column, not a survey item
    2. demographic_block  -- Q260-Q290 condition the prompt; using one as a target
                             would leak the answer into the prompt
    3. manually_excluded  -- see EXCLUDED_ITEMS
    4. scale_too_small    -- fewer than MIN_RESPONSE_SCALE_SIZE response options
    5. non_ordinal_codes  -- response codes are not a contiguous integer run
    6. observed_off_scale -- data contains values absent from the codebook scale
    7. high_missingness   -- more than MAX_MISSINGNESS_PCT missing
    8. low_entropy        -- below MIN_MODAL_ENTROPY, i.e. near-unanimous
"""

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy

from src.config import (
    DATA_PROCESSED,
    DATA_REFERENCE,
    MAX_MISSINGNESS_PCT,
    MIN_MODAL_ENTROPY,
    MIN_RESPONSE_SCALE_SIZE,
)

logger = logging.getLogger(__name__)

# Excluded on methodological grounds regardless of how they score statistically.
# Each was checked against the WVS-7 Variables Report V6.0 by reading its
# response labels: a contiguous 1..k code range is necessary but not sufficient
# for an ordinal scale, so genuinely unordered items are listed here by hand.
EXCLUDED_ITEMS = {
    "Q144": "Factual/behavioural recall (crime victimisation), not a values or attitude item",
    # Postmaterialism battery: respondents pick one goal from an unordered menu
    # ("economic growth" / "strong defence" / "more say" / "beautiful cities").
    # The codes name alternatives, not degrees, so MAE and Wasserstein-1 -- both
    # of which assume a meaningful distance between adjacent codes -- are
    # undefined on them.
    "Q152": "Nominal: unordered first-choice among national aims",
    "Q153": "Nominal: unordered second-choice among national aims",
    "Q154": "Nominal: unordered first-choice among respondent aims",
    "Q155": "Nominal: unordered second-choice among respondent aims",
    "Q156": "Nominal: unordered first-choice among societal goals",
    "Q157": "Nominal: unordered second-choice among societal goals",
    # 1=Always, 2=Usually, 3=Never are a frequency scale, but 4='Not allowed to
    # vote' is an eligibility status. Treating it as one step beyond 'Never'
    # would make ineligibility the most extreme form of not voting.
    "Q221": "Mixed scale: code 4 ('Not allowed to vote') is an eligibility category, not a frequency level",
    "Q222": "Mixed scale: code 4 ('Not allowed to vote') is an eligibility category, not a frequency level",
}

DEMOGRAPHIC_BLOCK = "DEMOGRAPHIC"

# WVS ships collapsed recodes of several questions (Q172R, Q275R, Q94R...).
# They duplicate a question already in the pool at coarser resolution, so
# keeping both would double-count the construct.
DERIVED_ID = re.compile(r"^Q\d+R$")
DERIVED_TITLE = re.compile(r"\(constructed\)|recoded", re.I)

REJECTION_REASONS = [
    "no_codebook_entry",
    "demographic_block",
    "derived_variant",
    "manually_excluded",
    "scale_too_small",
    "non_ordinal_codes",
    "observed_off_scale",
    "high_missingness",
    "low_entropy",
]


def load_codebook(path: Path = None) -> dict:
    """Load codebook-derived question metadata."""
    if path is None:
        path = DATA_REFERENCE / "wvs7_codebook.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Codebook metadata not found at {path}\n"
            "Generate it with: python -m src.data.parse_codebook"
        )
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["questions"]


def compute_missingness(series: pd.Series) -> float:
    """Missingness as a percentage."""
    return (series.isna().sum() / len(series)) * 100


def compute_modal_entropy(series: pd.Series) -> float:
    """Shannon entropy normalised by the number of *observed* categories.

    This is the project's original degeneracy criterion and the one the
    MIN_MODAL_ENTROPY threshold is calibrated against.
    """
    counts = series.value_counts()
    if len(counts) == 0:
        return 0.0
    probs = counts / counts.sum()
    max_entropy = np.log(len(probs))
    if max_entropy == 0:
        return 0.0
    return float(entropy(probs) / max_entropy)


def compute_scale_entropy(series: pd.Series, n_scale: int) -> float:
    """Shannon entropy normalised by the *nominal* scale size.

    Reported alongside the modal entropy as a diagnostic: a large gap between
    the two means respondents only ever used a few of the available options.
    """
    counts = series.value_counts()
    if len(counts) == 0 or n_scale <= 1:
        return 0.0
    probs = counts / counts.sum()
    return float(entropy(probs) / np.log(n_scale))


def screen_item(question_id: str, series: pd.Series, codebook: dict) -> dict:
    """Screen one column. Returns its stats plus pass/fail and the first failure."""
    result = {"question_id": question_id, "passed": False, "rejected_for": None}

    meta = codebook.get(question_id)
    if meta is None:
        result["rejected_for"] = "no_codebook_entry"
        return result

    result.update(
        {
            "title": meta["title"],
            "block": meta["block"],
            "n_scale": meta["n_scale"],
            "scale_labels": meta["valid_codes"],
        }
    )

    if meta["block"] == DEMOGRAPHIC_BLOCK:
        result["rejected_for"] = "demographic_block"
        return result

    if DERIVED_ID.match(question_id) or DERIVED_TITLE.search(meta["title"]):
        result["rejected_for"] = "derived_variant"
        return result

    if question_id in EXCLUDED_ITEMS:
        result["rejected_for"] = "manually_excluded"
        result["exclusion_note"] = EXCLUDED_ITEMS[question_id]
        return result

    if meta["n_scale"] < MIN_RESPONSE_SCALE_SIZE:
        result["rejected_for"] = "scale_too_small"
        return result

    if not meta["contiguous"]:
        result["rejected_for"] = "non_ordinal_codes"
        return result

    # The codebook is authoritative, so anything outside its scale is suspect.
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

    result["passed"] = True
    return result


def screen_all(df: pd.DataFrame, codebook: dict = None) -> list[dict]:
    """Screen every Q-column in the frame and return one record each."""
    if codebook is None:
        codebook = load_codebook()
    question_cols = [col for col in df.columns if col.startswith("Q")]
    return [screen_item(col, df[col], codebook) for col in question_cols]


def select_items(
    df: pd.DataFrame,
    n_target: int | None = None,
    codebook: dict = None,
) -> list[str]:
    """Select items passing every criterion.

    Args:
        n_target: optional cap, highest modal entropy first. Left as None for the
            real study -- the criteria determine the count, not a quota.
    """
    logger.info("Starting item selection...")
    results = screen_all(df, codebook)

    passed = [r for r in results if r["passed"]]
    logger.info(f"Evaluated {len(results)} candidate questions; {len(passed)} passed all criteria")

    rejected = [r for r in results if not r["passed"]]
    counts = pd.Series([r["rejected_for"] for r in rejected]).value_counts()
    for reason in REJECTION_REASONS:
        if reason in counts:
            logger.info(f"  rejected [{reason}]: {counts[reason]}")

    passed.sort(key=lambda r: r["modal_entropy"], reverse=True)
    if n_target is not None:
        passed = passed[:n_target]
        logger.info(f"Capped to n_target={n_target}")

    selected = [r["question_id"] for r in passed]
    if selected:
        blocks = pd.Series([r["block"] for r in passed]).value_counts()
        logger.info(f"\nSelected items by block:\n{blocks.to_string()}")
    return selected


def main(input_path: Path = None, output_path: Path = None):
    if input_path is None:
        input_path = DATA_PROCESSED / "ind_wvs7.parquet"
    if output_path is None:
        output_path = DATA_PROCESSED / "selected_items.json"

    if not input_path.exists():
        logger.error(f"Input data not found: {input_path}")
        logger.info("Run: python -m src.data.load_wvs --country IND")
        return False

    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} respondents")

    codebook = load_codebook()
    results = screen_all(df, codebook)
    passed = sorted(
        [r for r in results if r["passed"]], key=lambda r: r["modal_entropy"], reverse=True
    )
    selected = [r["question_id"] for r in passed]

    if not selected:
        logger.error("Item selection produced zero items")
        return False

    rejected = [r for r in results if not r["passed"]]
    ledger = pd.Series([r["rejected_for"] for r in rejected]).value_counts().to_dict()

    output = {
        "selected_items": selected,
        "n_items": len(selected),
        "n_candidates_evaluated": len(results),
        "criteria": {
            "max_missingness_pct": MAX_MISSINGNESS_PCT,
            "min_entropy": MIN_MODAL_ENTROPY,
            "min_scale_size": MIN_RESPONSE_SCALE_SIZE,
            "excluded_items": EXCLUDED_ITEMS,
            "demographic_block_excluded": "Q260-Q290 (prompt conditioning variables)",
        },
        "rejection_ledger": {r: int(ledger.get(r, 0)) for r in REJECTION_REASONS},
        "item_details": [
            {
                "question_id": r["question_id"],
                "title": r["title"],
                "block": r["block"],
                "n_scale": r["n_scale"],
                "n_observed_categories": r["n_observed_categories"],
                "missingness_pct": r["missingness_pct"],
                "modal_entropy": r["modal_entropy"],
                "scale_entropy": r["scale_entropy"],
                "scale_labels": r["scale_labels"],
            }
            for r in passed
        ],
        "codebook_source": "WVS-7 Variables Report V6.0",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)

    logger.info(f"✓ Saved {len(selected)} selected items to {output_path}")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    raise SystemExit(0 if main(args.input, args.output) else 1)
