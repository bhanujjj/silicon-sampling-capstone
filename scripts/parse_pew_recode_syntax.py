"""Parse Pew's SPSS-style "recode syntax for public release" text file into
the {question_id: {title, wording, valid_codes}} shape build_pew_codebook.py
expects via --labels.

NOT YET VERIFIED against the real file -- written from the standard SPSS
VARIABLE LABELS / VALUE LABELS syntax shape, but the exact dialect of
Pew's actual "India recode syntax for public release.txt" hasn't been
seen yet. Run this, then spot-check a handful of entries in the output
JSON against the .txt by eye before trusting it -- if the regexes don't
match this file's exact formatting, this will silently produce zero or
partial entries rather than wrong ones (it never guesses), so a low
entry count is the signal to fix the parser, not a sign of a subtle bug.

Standard SPSS shape this expects:

    VARIABLE LABELS
      Q9 'Some question text'
      Q13 'Other question text'
      .

    VALUE LABELS
     Q9
       1 'Label one'
       2 'Label two'
       98 'Dont know'
       99 'Refused'
     /Q13
       1 'Label a'
       2 'Label b'
       .

Usage:
    python scripts/parse_pew_recode_syntax.py "path/to/India recode syntax for public release.txt" -o data/reference/pew_labels_raw.json
"""

import argparse
import json
import re
from pathlib import Path

VAR_LABEL_RE = re.compile(r"^\s*(Q[\w]+|QRELSING|QCASTE\w*)\s+'([^']+)'", re.IGNORECASE)
VALUE_BLOCK_START_RE = re.compile(r"^\s*/?(\w+)\s*$")
VALUE_LINE_RE = re.compile(r"^\s*(-?\d+)\s+'([^']+)'")


def parse_variable_labels(text: str) -> dict:
    """VARIABLE LABELS block: varname 'question text'."""
    titles = {}
    in_block = False
    for line in text.splitlines():
        if re.search(r"VARIABLE\s+LABELS", line, re.IGNORECASE):
            in_block = True
            continue
        if in_block and line.strip() == ".":
            in_block = False
            continue
        if in_block:
            m = VAR_LABEL_RE.match(line)
            if m:
                titles[m.group(1)] = m.group(2).strip()
    return titles


def parse_value_labels(text: str) -> dict:
    """VALUE LABELS block(s): one or more `varname \\n code 'label' \\n ...` groups,
    separated by '/' or blank lines, terminated by '.'."""
    value_labels = {}
    in_block = False
    current_var = None
    for line in text.splitlines():
        if re.search(r"VALUE\s+LABELS", line, re.IGNORECASE):
            in_block = True
            current_var = None
            continue
        if not in_block:
            continue
        stripped = line.strip()
        if stripped == ".":
            in_block = False
            current_var = None
            continue
        if not stripped:
            continue

        value_match = VALUE_LINE_RE.match(line)
        if value_match and current_var:
            code, label = value_match.group(1), value_match.group(2).strip()
            value_labels.setdefault(current_var, {})[code] = label
            continue

        # Not a "code 'label'" line and not blank/terminator -> treat as a new
        # variable name (handles both "/VAR" and a bare "VAR" on its own line).
        var_match = VALUE_BLOCK_START_RE.match(stripped.lstrip("/"))
        if var_match:
            current_var = var_match.group(1)

    return value_labels


def build_labels_json(text: str) -> dict:
    titles = parse_variable_labels(text)
    value_labels = parse_value_labels(text)

    all_vars = set(titles) | set(value_labels)
    out = {}
    for var in all_vars:
        if var not in value_labels:
            continue  # a question needs its answer-option labels to be usable
        out[var] = {
            "title": titles.get(var, var),
            "wording": titles.get(var, var),
            "valid_codes": value_labels[var],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_txt", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=Path("data/reference/pew_labels_raw.json"))
    args = ap.parse_args()

    text = args.input_txt.read_text(encoding="utf-8", errors="replace")
    labels = build_labels_json(text)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    print(f"Parsed {len(labels)} variables with both title and value labels -> {args.output}")
    if len(labels) < 50:
        print(
            "WARNING: fewer than 50 variables parsed out of ~300 expected. "
            "The file's exact syntax dialect likely differs from what this "
            "parser expects -- open the .txt and compare its real format "
            "against the docstring above before trusting this output."
        )
    print("Next: python -m src.data.build_pew_codebook --labels", args.output)


if __name__ == "__main__":
    main()
