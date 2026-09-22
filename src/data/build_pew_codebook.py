"""Build data/reference/pew_codebook.json from the raw Pew "Religion in
India" (2021) CSV.

Structural fields (which numeric codes a column actually uses, how many
missing/DK/refused responses it has) are derived directly from the real
released data -- always safe to (re)run, no external document needed.

Question TEXT and response-option LABELS are a different matter: Pew's CSV
ships bare codes (e.g. column "Q37a", values 1/2/98/99) with no wording
attached. The only place real wording exists is Pew's own codebook
document -- the DDI metadata XML, the "recode syntax for public release"
text file, or CODEBOOK_India.pdf, all of which ship in the same release
bundle as the CSV. This script deliberately does NOT invent wording: an
item stays "title": null / "wording": null / "verified": false until one
of those documents is parsed and merged in via --labels. Putting a
guessed question in front of the LLM -- and later in the research paper --
under Pew's name would misattribute fabricated survey content to a real
organization, which is a correctness bug, not a style choice.

The two demographic mappings below (QGEN, QRELSING) are the only labels
filled in without that document, and only because they are independently
confirmable: QGEN's 1/2 split matches the near-universal Pew convention
and this survey's near-even sex ratio, and QRELSING's code counts match
Pew's own publicly published religion breakdown to the exact respondent
(22,975 Hindu / 3,336 Muslim / 1,782 Sikh / 1,011 Christian / 719
Buddhist / 109 Jain / 67 other -- see the June 2021 Pew release). Every
other demographic (age bands, education levels, caste categories, region
names, marital status, income brackets) is numerically structured here
but left unlabeled for the same reason as the items -- a wrong guess on
an age-band edge or a caste-category name would misrepresent a real
respondent's real profile.

Usage:
    python -m src.data.build_pew_codebook
        # Structural pass. Safe to run anytime.

    python -m src.data.build_pew_codebook --labels path/to/parsed_labels.json
        # Merges in real title/wording/valid_code text once available.
        # Expected shape: {"Q9": {"title": "...", "wording": "...",
        # "valid_codes": {"1": "...", "2": "..."}}, ...}
        # scripts/parse_pew_recode_syntax.py can produce this from Pew's
        # own "recode syntax for public release.txt", if you have it.
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.config import DATA_RAW, DATA_REFERENCE, resolve_pew_csv
from src.data.load_pew import MISSING_CODES, NON_ITEM_COLS

logger = logging.getLogger(__name__)

DEMOGRAPHIC_COLS = {
    "QGEN", "QAGErec", "Urban", "REGION", "QCASTE", "QCASTEb", "QHINDU",
    "QSECTrec", "QDENOMrec", "QSIKHrec", "QBUDDHISTrec", "QJAIN", "QSUFIrec",
    "QRELSING", "QMARRIEDrec", "QEDU", "ISCED", "QINCINDrec", "QHH1", "QHH2",
    "QAGErec", "QCHRELrec", "QAGErec",
}

# The only two label sets filled in without the official codebook document --
# see module docstring for why these two are independently confirmable.
CONFIRMED_LABELS = {
    "QGEN": {
        "title": "Sex",
        "wording": "Interviewer-recorded respondent sex (standard Pew convention; verify against the official codebook before publication).",
        "valid_codes": {"1": "Male", "2": "Female"},
    },
    "QRELSING": {
        "title": "Religious identity (single response)",
        "wording": "What is your present religion, if any?",
        "valid_codes": {
            "1": "Hindu", "2": "Muslim", "3": "Christian", "4": "Sikh",
            "5": "Buddhist", "6": "Jain", "7": "Other religion", "8": "Unaffiliated",
        },
    },
}


def infer_column_structure(series_raw: pd.Series) -> dict:
    """Numeric structure only, computed from the RAW (pre-recode) column so
    the missing-code cluster (96-99 etc.) is visible and reported, not
    already stripped out."""
    numeric = pd.to_numeric(series_raw, errors="coerce")
    observed = sorted(int(v) for v in numeric.dropna().unique())
    substantive = [v for v in observed if v not in MISSING_CODES]
    missing_present = [v for v in observed if v in MISSING_CODES]

    if not substantive:
        return None

    n_scale = len(substantive)
    contiguous = (max(substantive) - min(substantive) + 1) == n_scale
    return {
        "n_scale": n_scale,
        "min_code": min(substantive),
        "max_code": max(substantive),
        "contiguous": contiguous,
        "observed_substantive_codes": substantive,
        "missing_codes_present": missing_present,
    }


def build_codebook(df_raw: pd.DataFrame, labels: dict = None) -> dict:
    labels = labels or {}
    questions = {}
    n_unlabeled = 0

    for col in df_raw.columns:
        if col in NON_ITEM_COLS:
            continue
        structure = infer_column_structure(df_raw[col])
        if structure is None:
            continue

        block = "DEMOGRAPHIC" if col in DEMOGRAPHIC_COLS else "UNVERIFIED_ITEM"
        label_info = labels.get(col) or CONFIRMED_LABELS.get(col)

        entry = {
            "title": label_info["title"] if label_info else None,
            "wording": label_info["wording"] if label_info else None,
            "valid_codes": (
                label_info["valid_codes"]
                if label_info
                else {str(c): str(c) for c in structure["observed_substantive_codes"]}
            ),
            "n_scale": structure["n_scale"],
            "min_code": structure["min_code"],
            "max_code": structure["max_code"],
            "contiguous": structure["contiguous"],
            "missing_codes": sorted(set(structure["missing_codes_present"]) | MISSING_CODES),
            "numeric_variable": False,
            "block": block,
            "verified": label_info is not None,
        }
        if label_info is None:
            n_unlabeled += 1
        questions[col] = entry

    logger.info(f"Built {len(questions)} question entries, {n_unlabeled} still awaiting real wording/labels.")
    return {
        "_source": "Pew Research Center, 'Religion in India: Tolerance and Segregation' (2021), India microdata (All Vars).",
        "_note": (
            "Derived question metadata, not survey microdata. Fields with "
            "'verified': false have real numeric structure but NO confirmed "
            "wording/labels yet -- do not use as an LLM prompt or in the "
            "paper until re-built with --labels from Pew's own codebook "
            "document. See this module's docstring."
        ),
        "n_questions": len(questions),
        "n_unverified": n_unlabeled,
        "questions": questions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=None, help="JSON file of real {question_id: {title, wording, valid_codes}} parsed from Pew's own codebook document")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    csv_path = args.csv or resolve_pew_csv(DATA_RAW)
    logger.info(f"Reading raw CSV for structure inference: {csv_path}")
    df_raw = pd.read_csv(csv_path, low_memory=False)

    labels = {}
    if args.labels:
        if not args.labels.exists():
            logger.error(f"--labels file not found: {args.labels}")
            return False
        with open(args.labels, encoding="utf-8") as f:
            labels = json.load(f)
        logger.info(f"Loaded {len(labels)} real label entries from {args.labels}")

    codebook = build_codebook(df_raw, labels)

    DATA_REFERENCE.mkdir(parents=True, exist_ok=True)
    output_path = DATA_REFERENCE / "pew_codebook.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(codebook, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved codebook to {output_path} ({codebook['n_questions']} questions, {codebook['n_unverified']} unverified)")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    exit(0 if main() else 1)
