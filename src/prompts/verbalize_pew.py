"""Turn a Pew "Religion in India" (2021) respondent row (and a survey item)
into prompt-ready text -- the Pew-side counterpart to verbalize.py.

verbalize_item() and parse_predicted_code() are NOT duplicated here: they
only read {title, wording, valid_codes} off whatever codebook dict they're
given, so src.prompts.verbalize's versions work unchanged against
data/reference/pew_codebook.json. Only demographic verbalization is
Pew-specific (different column names, different label source), so that's
the one function this module defines.

A field whose codebook entry has "verified": false (i.e. build_pew_codebook.py
never got real labels for it -- see that module's docstring) returns None
here rather than the placeholder numeric code, matching build_prompt()'s
existing behaviour of silently omitting any None demographic instead of
printing "Region: 3" style noise into the prompt.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import DATA_REFERENCE

logger = logging.getLogger(__name__)

# Pew columns surfaced to the P1/P2/P3 templates, mapped to the template's
# keyword argument name. Every label comes from data/reference/pew_codebook.json
# (verified entries only -- see module docstring).
DEMOGRAPHIC_CODE_COLS = {
    "QGEN": "sex",
    "QMARRIEDrec": "marital_status",
    "QEDU": "education",
    "QRELSING": "religion",
    "QCASTE": "caste",
}


@lru_cache(maxsize=1)
def load_pew_codebook(path: Path = None) -> dict:
    if path is None:
        path = DATA_REFERENCE / "pew_codebook.json"
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["questions"]


def _label(codebook: dict, question_id: str, code) -> Optional[str]:
    """Look up a verified response label; None for missing/off-scale/
    not-yet-labeled codes."""
    if pd.isna(code):
        return None
    meta = codebook.get(question_id)
    if meta is None or not meta.get("verified", False):
        return None
    return meta["valid_codes"].get(str(int(code)))


def verbalize_demographics(row: pd.Series, codebook: dict = None) -> dict:
    """Build the kwargs expected by src.prompts.templates' P1/P2/P3 functions.

    caste is not one of P2's 14 named kwargs -- build_prompt()/format_p2_structured
    only render the keys it knows, so an extra "caste" key here is harmless
    (silently unused by P2) until templates.py is extended to include it,
    which only matters once caste labels are actually verified.
    """
    if codebook is None:
        codebook = load_pew_codebook()

    demo = {}
    for col, key in DEMOGRAPHIC_CODE_COLS.items():
        demo[key] = _label(codebook, col, row.get(col)) if col in row.index else None

    demo["age"] = _label(codebook, "QAGErec", row.get("QAGErec")) if "QAGErec" in row.index else None
    demo["urban_rural"] = (
        {1.0: "Urban", 2.0: "Rural"}.get(float(row["Urban"]))
        if "Urban" in row.index and pd.notna(row["Urban"])
        else None
    )
    demo["region"] = _label(codebook, "REGION", row.get("REGION")) if "REGION" in row.index else None
    demo["income_decile"] = _label(codebook, "QINCINDrec", row.get("QINCINDrec")) if "QINCINDrec" in row.index else None
    demo["n_children"] = None  # not asked in this survey
    demo["employment"] = None  # not confirmed present -- see PEW_INTEGRATION.md
    demo["occupation"] = None
    demo["social_class"] = None
    demo["town_size"] = None
    demo["interview_language"] = None
    return demo
