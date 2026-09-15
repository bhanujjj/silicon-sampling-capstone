"""Turn a WVS-7 respondent row (and a survey item) into prompt-ready text.

Every label comes from the codebook-derived metadata in
data/reference/wvs7_codebook.json -- nothing here is hand-typed, so relabeling
is just a re-parse of the official document, not a code change.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import DATA_REFERENCE

logger = logging.getLogger(__name__)

# Demographic columns surfaced to the P1/P2/P3 templates, mapped to the
# template's keyword argument name. Region and language are handled specially
# below since they need annex codes / a fixed ISO table respectively.
DEMOGRAPHIC_CODE_COLS = {
    "Q260": "sex",
    "Q273": "marital_status",
    "Q275": "education",
    "Q279": "employment",
    "Q281": "occupation",
    "Q287": "social_class",
    "Q289": "religion",
}

# India's 8 WVS sampling regions (N_REGION_ISO / ISO 3166-2:IN), taken
# verbatim from the codebook's own Annex (data/raw/WVS7_Codebook_Variables_
# report_V6.0.pdf, p.227 "INDIA" block) -- these are states/territory, not
# macro-zones. An earlier version of this table guessed macro-zone names
# (North/South/etc.) without having located the Annex; that guess was WRONG
# on every code (e.g. 356028 is Uttar Pradesh, not "South zone") and has
# been replaced with the verified Annex values below. See PROJECT_REPORT.md
# for the disclosure that all P1-P3 predictions collected before this fix
# were prompted with the incorrect zone guess for the region attribute.
INDIA_REGION_LABELS = {
    356004: "Bihar",
    356008: "Haryana",
    356015: "Maharashtra",
    356021: "Punjab",
    356025: "Telangana",
    356028: "Uttar Pradesh",
    356029: "West Bengal",
    356034: "Delhi",
}

LANGUAGE_LABELS = {"hi": "Hindi"}


@lru_cache(maxsize=1)
def load_codebook(path: Path = None) -> dict:
    if path is None:
        path = DATA_REFERENCE / "wvs7_codebook.json"
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["questions"]


def _label(codebook: dict, question_id: str, code) -> Optional[str]:
    """Look up a response label; None for missing/off-scale codes."""
    if pd.isna(code):
        return None
    meta = codebook.get(question_id)
    if meta is None:
        return None
    label = meta["valid_codes"].get(str(int(code)))
    if label is None:
        return None
    # The PDF wraps long labels across lines; the parser only keeps the first
    # line, which can cut an example list mid-parenthetical. Trim it cleanly
    # rather than showing a dangling "(for example: ...,".
    if label.count("(") > label.count(")"):
        label = label[: label.index("(")].strip()
    return label or None


def verbalize_demographics(row: pd.Series, codebook: dict = None) -> dict:
    """Build the kwargs expected by src.prompts.templates' P1/P2/P3 functions."""
    if codebook is None:
        codebook = load_codebook()

    demo = {}
    for col, key in DEMOGRAPHIC_CODE_COLS.items():
        demo[key] = _label(codebook, col, row.get(col)) if col in row.index else None

    demo["age"] = int(row["Q262"]) if "Q262" in row.index and pd.notna(row["Q262"]) else None
    demo["n_children"] = (
        int(row["Q274"]) if "Q274" in row.index and pd.notna(row["Q274"]) else None
    )
    demo["urban_rural"] = (
        {1.0: "Urban", 2.0: "Rural"}.get(row["H_URBRURAL"])
        if "H_URBRURAL" in row.index and pd.notna(row["H_URBRURAL"])
        else None
    )
    demo["income_decile"] = (
        f"{int(row['Q288'])} of 10"
        if "Q288" in row.index and pd.notna(row["Q288"])
        else None
    )
    demo["region"] = (
        INDIA_REGION_LABELS.get(int(row["N_REGION_ISO"]))
        if "N_REGION_ISO" in row.index and pd.notna(row["N_REGION_ISO"])
        else None
    )
    demo["town_size"] = (
        _label(codebook, "G_TOWNSIZE", row.get("G_TOWNSIZE"))
        if "G_TOWNSIZE" in row.index
        else None
    )
    demo["interview_language"] = (
        LANGUAGE_LABELS.get(row["LNGE_ISO"])
        if "LNGE_ISO" in row.index and pd.notna(row["LNGE_ISO"])
        else None
    )
    return demo


def verbalize_item(question_id: str, codebook: dict = None) -> dict:
    """Return {question_text, options_text, ordinal_values, code_to_index, n_options}."""
    if codebook is None:
        codebook = load_codebook()
    meta = codebook[question_id]

    codes = sorted(int(c) for c in meta["valid_codes"])
    options_text = "\n".join(f"{c}. {meta['valid_codes'][str(c)]}" for c in codes)
    code_to_index = {code: idx for idx, code in enumerate(codes)}

    return {
        "question_id": question_id,
        "question_text": meta["title"],
        "wording": meta["wording"],
        "options_text": options_text,
        "ordinal_values": codes,
        "code_to_index": code_to_index,
        "n_options": len(codes),
    }


def parse_predicted_code(predicted_text: Optional[str], item: dict) -> Optional[int]:
    """Map a free-text model answer back onto one of the item's valid codes.

    Tries an exact numeric match first, then a case-insensitive label match.
    Returns None (a refusal/unparseable answer) rather than guessing.
    """
    if predicted_text is None:
        return None
    text = predicted_text.strip()

    try:
        code = int(text)
        if code in item["code_to_index"]:
            return code
    except ValueError:
        pass

    text_lower = text.lower()
    codebook_labels = dict(
        zip(item["ordinal_values"], (line.split(". ", 1)[1] for line in item["options_text"].splitlines()))
    )
    for code, label in codebook_labels.items():
        if text_lower == label.lower():
            return code
    return None
