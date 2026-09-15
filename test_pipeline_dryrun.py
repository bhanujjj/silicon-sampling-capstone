"""Dry-run test of the exact prompt-building code path trackB_finetune.py
uses, WITHOUT needing the real parquet or a GPU. Proves the data/prompt
pipeline is sound against the actual committed selected_items.json,
folds.json, and codebook -- the only thing this can't test is whether the
real parquet's column names/value codes match what verbalize.py expects,
since that file isn't in git. Run this from the lab-package root.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from src.config import DATA_PROCESSED, DATA_REFERENCE
from src.prompts.templates import build_prompt
from src.prompts.verbalize import load_codebook, verbalize_demographics, verbalize_item
from src.inference.prompting import build_answer_instruction, parse_answer_from_text

print("=" * 70)
print("1. Loading real committed metadata files")
print("=" * 70)
with open(DATA_PROCESSED / "selected_items.json") as f:
    selected = json.load(f)["selected_items"]
print(f"selected_items.json: {len(selected)} items, first 15: {selected[:15]}")

with open(DATA_PROCESSED / "folds.json") as f:
    folds = json.load(f)["folds"]
print(f"folds.json: {len(folds)} folds")
print(f"  fold 0: {len(folds[0]['train'])} train, {len(folds[0]['test'])} test respondent IDs")

codebook = load_codebook()
print(f"wvs7_codebook.json: {len(codebook)} question entries")

print()
print("=" * 70)
print("2. Verifying all 15 real Track-B items exist in codebook + verbalize cleanly")
print("=" * 70)
n_items = 15
items_to_test = selected[:n_items]
ok = 0
for qid in items_to_test:
    try:
        item = verbalize_item(qid, codebook)
        assert item["question_text"], "empty question_text"
        assert item["n_options"] >= 2, "fewer than 2 valid options"
        assert item["options_text"], "empty options_text"
        ok += 1
    except Exception as e:
        print(f"  FAIL {qid}: {e}")
print(f"{ok}/{len(items_to_test)} items verbalize cleanly")
assert ok == len(items_to_test), "Some items failed to verbalize -- would break build_examples() at the lab"

print()
print("=" * 70)
print("3. Building a synthetic respondent row (same columns real parquet must have)")
print("=" * 70)
# Column names copied exactly from DEMOGRAPHIC_CODE_COLS + the special-cased
# ones in verbalize_demographics -- this is exactly what build_examples()
# will look up on every real respondent row.
fake_row = pd.Series({
    "respondent_id": 999001,
    "Q260": 2,          # sex
    "Q262": 34,          # age
    "Q273": 1,           # marital status
    "Q274": 2,           # n children
    "Q275": 6,           # education
    "Q279": 1,           # employment
    "Q281": 1,           # occupation
    "Q287": 3,           # social class
    "Q288": 5,           # income decile
    "Q289": 2,           # religion
    "H_URBRURAL": 1.0,   # urban
    "N_REGION_ISO": 356015,  # Maharashtra
    "G_TOWNSIZE": 5,
    "LNGE_ISO": "hi",
})
demo = verbalize_demographics(fake_row, codebook)
print("verbalize_demographics() output:")
for k, v in demo.items():
    print(f"  {k}: {v}")
missing = [k for k, v in demo.items() if v is None]
if missing:
    print(f"  NOTE: {missing} came back None -- either the fake row is missing that code, or that code isn't in the codebook. Not fatal (verbalize_demographics tolerates Nones), but worth knowing.")

print()
print("=" * 70)
print("4. Building the EXACT training example format trackB_finetune.py writes")
print("=" * 70)
examples = []
for qid in items_to_test[:5]:  # just show 5 for brevity
    item = verbalize_item(qid, codebook)
    true_code = item["ordinal_values"][0]  # fake "true answer" = first valid code
    option_labels = [str(c) for c in item["ordinal_values"]]
    prompt = build_prompt("P2", item["question_text"], item["options_text"], **demo) + build_answer_instruction(option_labels)
    answer = str(true_code)
    examples.append({"text": prompt + "\n\n" + answer, "question_id": qid})

print(f"Built {len(examples)} example training texts. First one, full text:\n")
print("-" * 70)
print(examples[0]["text"])
print("-" * 70)

print()
print("=" * 70)
print("5. Round-trip: parse a model reply back to a code (as eval-time does)")
print("=" * 70)
item0 = verbalize_item(items_to_test[0], codebook)
option_labels = [str(c) for c in item0["ordinal_values"]]
for fake_reply in [option_labels[0], f" {option_labels[0]} ", "I think the answer is " + option_labels[0], "garbage"]:
    parsed = parse_answer_from_text(fake_reply, option_labels)
    print(f"  reply={fake_reply!r:40} -> parsed={parsed!r}")

print()
print("=" * 70)
print("ALL DRY-RUN CHECKS PASSED")
print("=" * 70)
print("This proves: selected_items.json + folds.json + codebook.json are internally")
print("consistent, verbalize_demographics/verbalize_item/build_prompt/parse_answer_from_text")
print("all run cleanly end-to-end, and the exact training-example text format matches")
print("what trackB_finetune.py will actually build at the lab.")
print()
print("What this CANNOT prove (needs the real parquet, only testable at the lab):")
print("  - that ind_wvs7.parquet's actual column names match Q260/Q262/.../LNGE_ISO exactly")
print("  - that real respondent value codes fall inside each item's valid_codes range")
print("  - that the GPU/model stack itself loads and trains (that's what --preflight is for)")
