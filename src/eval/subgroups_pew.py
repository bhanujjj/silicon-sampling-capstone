"""Pew-side subgroup slicing -- assign_subgroups_pew() is the only new
logic; metrics_by_subgroup_axis(), fidelity_gap_report(), and delta_gap()
from src.eval.subgroups are fully generic (they only read whatever sg_*
columns + axis list they're given) and are reused unchanged.

Two axes here (sg_urban_rural, sg_sex, sg_religion) use real, confirmed
labels -- see verbalize_pew.py / build_pew_codebook.py for why those three
are trustworthy without the official codebook document. The rest
(sg_caste, sg_region, sg_income, sg_age_band) use their raw numeric code
under a generic "Group N" name rather than a guessed real-world label
(e.g. "Zone 3", not a guessed region name) -- honestly uninformative about
WHICH group is which, but still valid for measuring whether accuracy
DIFFERS across groups, which is what a fidelity-gap analysis needs. Once
Pew's codebook document is parsed (see build_pew_codebook.py), re-run this
with real category names substituted in -- the fold/prediction data itself
does not need to change.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def assign_subgroups_pew(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["sg_sex"] = out["QGEN"].map({1: "Male", 2: "Female"})
    out["sg_urban_rural"] = out["Urban"].map({1: "Urban", 2: "Rural"})
    out["sg_religion"] = out["QRELSING"].map({
        1: "Hindu", 2: "Muslim", 3: "Christian", 4: "Sikh",
        5: "Buddhist", 6: "Jain", 7: "Other", 8: "Unaffiliated",
    })

    for raw_col, sg_col, prefix in [
        ("QCASTE", "sg_caste", "Caste group"),
        ("REGION", "sg_region", "Zone"),
        ("QINCINDrec", "sg_income", "Income group"),
        ("QAGErec", "sg_age_band", "Age band"),
    ]:
        if raw_col in out.columns:
            out[sg_col] = out[raw_col].apply(lambda v: f"{prefix} {int(v)}" if pd.notna(v) else None)

    return out


SUBGROUP_AXES_PEW = ["sg_sex", "sg_urban_rural", "sg_religion", "sg_caste", "sg_region", "sg_income", "sg_age_band"]
