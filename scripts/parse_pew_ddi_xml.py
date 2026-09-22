"""Parse Pew's DDI-Codebook 2.5 metadata XML into the
{question_id: {title, wording, valid_codes}} shape build_pew_codebook.py
expects via --labels. This is THE primary, machine-readable source for
Pew's real question wording -- confirmed against CODEBOOK_India.pdf and
already spot-checked against the hand-derived recode-syntax labels
(Q43b's wording matches exactly), so this is the preferred path over
parse_pew_recode_syntax.py, which only ever covered ~20 items as a side
effect of unrelated derived-variable definitions.

Uses Python's stdlib xml.etree.ElementTree (no extra dependency). Handles
the file's default DDI namespace (ddi:codebook:2_5) explicitly -- ET
requires namespaced tag lookups even for the default namespace.

DDI shape per variable:
    <var name="Q37a">
      <labl>Q37. Do you ...? a. give money to a [temple/...]</labl>
      <catgry><catValu>1</catValu><labl>Yes</labl></catgry>
      <catgry><catValu>2</catValu><labl>No</labl></catgry>
      <catgry><catValu>98</catValu><labl>Don't know (DO NOT READ)</labl></catgry>
      ...
    </var>

Missing/DK/refused categories (matched against src.data.load_pew.MISSING_CODES,
so the two stay in sync automatically) are dropped from valid_codes --
they're not real answer options, and build_pew_codebook.py's own
structural pass already knows the true missing-code set for each column
from the raw data directly.

Usage:
    python scripts/parse_pew_ddi_xml.py "path/to/Pew India DDI metadata.xml" -o data/reference/pew_labels_ddi.json
"""

import argparse
import json
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.load_pew import MISSING_CODES

DDI_NS = {"ddi": "ddi:codebook:2_5"}

# Strips a leading "Q37." / "Q43b." / "QRELSING." style question-ID prefix
# off the raw <labl> text to make a shorter "title", keeping the full
# original text as "wording" regardless.
ID_PREFIX_RE = re.compile(r"^\s*[A-Za-z][A-Za-z0-9_]*\.\s*")


def parse_ddi(xml_path: Path) -> dict:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    out = {}
    for var in root.iter("{ddi:codebook:2_5}var"):
        name = var.get("name")
        if not name:
            continue

        labl_el = var.find("ddi:labl", DDI_NS)
        full_text = (labl_el.text or "").strip() if labl_el is not None else ""
        if not full_text:
            continue

        valid_codes = {}
        for catgry in var.findall("ddi:catgry", DDI_NS):
            catvalu_el = catgry.find("ddi:catValu", DDI_NS)
            catlabl_el = catgry.find("ddi:labl", DDI_NS)
            if catvalu_el is None or catvalu_el.text is None:
                continue
            try:
                code = int(float(catvalu_el.text.strip()))
            except ValueError:
                continue
            if code in MISSING_CODES:
                continue  # DK/refused/not-applicable -- not a real answer option
            label = (catlabl_el.text or "").strip() if catlabl_el is not None else str(code)
            valid_codes[str(code)] = label

        if not valid_codes:
            continue  # e.g. weight, QRID, free-text/continuous fields -- not an item

        title = ID_PREFIX_RE.sub("", full_text).strip() or full_text
        out[name] = {"title": title, "wording": full_text, "valid_codes": valid_codes}

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_xml", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=Path("data/reference/pew_labels_ddi.json"))
    args = ap.parse_args()

    labels = parse_ddi(args.input_xml)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    print(f"Parsed {len(labels)} variables with real title/wording/valid_codes -> {args.output}")
    print("Next: python -m src.data.build_pew_codebook --labels", args.output)


if __name__ == "__main__":
    main()
