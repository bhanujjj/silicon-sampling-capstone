"""Dry-run test of the exact prompt-building code path pewB_finetune.py
uses, without needing a GPU. Uses the real parquet if data/processed/
pew_india_2021.parquet is present locally (gitignored, same as WVS's),
else falls back to a synthetic row built from real observed column values.

EXPECTED RESULT TODAY: 0 selected items, because data/reference/
pew_codebook.json has no verified question wording yet (see
src/data/build_pew_codebook.py's docstring) -- this is the correct, safe
state, not a test failure. What this DOES verify today: the demographic
verbalization pipeline (real column names, real value codes) works
end-to-end against a real respondent row, and the generic verbalize_item/
build_prompt/parse_answer_from_text machinery works correctly once ANY
item has verified wording (proven here with one clearly-marked synthetic
test item, never treated as real Pew content).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from src.config import DATA_PROCESSED, DATA_REFERENCE
from src.prompts.templates import build_prompt
from src.prompts.verbalize import verbalize_item
from src.prompts.verbalize_pew import load_pew_codebook, verbalize_demographics
from src.inference.prompting import build_answer_instruction, parse_answer_from_text

print("=" * 70)
print("1. Loading real committed metadata files")
print("=" * 70)
with open(DATA_PROCESSED / "pew_selected_items.json") as f:
    sel = json.load(f)
print(f"pew_selected_items.json: {sel['n_items']} items selected (rejection ledger: {sel['rejection_ledger']})")
assert sel["n_items"] == 0, (
    "Expected 0 selected items at this stage of the project (codebook wording "
    "not yet verified) -- if this is no longer 0, the codebook has been "
    "updated with real labels; update this test's expectations accordingly."
)
print("0 items is the CORRECT current state (see build_pew_codebook.py docstring), not a failure.")

with open(DATA_PROCESSED / "pew_folds.json") as f:
    folds = json.load(f)["folds"]
print(f"pew_folds.json: {len(folds)} folds, fold 0: {len(folds[0]['train'])} train / {len(folds[0]['test'])} test respondent IDs")

codebook = load_pew_codebook()
print(f"pew_codebook.json: {len(codebook)} question entries, {sum(1 for q in codebook.values() if q['verified'])} verified")

print()
print("=" * 70)
print("2. Demographic verbalization against a real respondent row")
print("=" * 70)
parquet_path = DATA_PROCESSED / "pew_india_2021.parquet"
if parquet_path.exists():
    df = pd.read_parquet(parquet_path)
    row = df.iloc[0]
    print(f"Using REAL respondent row from {parquet_path} (respondent_id={row.get('respondent_id')})")
else:
    print(f"{parquet_path} not present locally -- using a synthetic row built from real observed codes")
    row = pd.Series({"QGEN": 1, "Urban": 2, "QRELSING": 1, "QMARRIEDrec": 1, "QEDU": 3, "QCASTE": 4, "QAGErec": 3, "REGION": 2, "QINCINDrec": 1})

demo = verbalize_demographics(row, codebook)
print(f"verbalize_demographics() -> {demo}")
assert demo["sex"] in ("Male", "Female"), "sex should resolve to a verified label"
assert demo["religion"] is not None, "religion should resolve to a verified label"
assert demo["age"] is None, "age should be None (QAGErec not yet verified) -- if not, a fabricated label leaked through"
print("OK: verified fields resolve to real labels, unverified fields correctly resolve to None (no fabricated noise).")

print()
print("=" * 70)
print("3. Generic verbalize_item()/build_prompt()/parse_answer_from_text() mechanics")
print("   (using ONE synthetic test item -- proves the pipeline works, is NOT real Pew content)")
print("=" * 70)
test_codebook = dict(codebook)
test_codebook["TEST_ITEM"] = {
    "title": "[SYNTHETIC TEST ITEM -- not a real Pew question]",
    "wording": "[SYNTHETIC TEST ITEM -- not a real Pew question]",
    "valid_codes": {"1": "Strongly disagree", "2": "Disagree", "3": "Agree", "4": "Strongly agree"},
}
item = verbalize_item("TEST_ITEM", test_codebook)
assert item["n_options"] == 4
option_labels = [str(c) for c in item["ordinal_values"]]
prompt = build_prompt("P2", item["question_text"], item["options_text"], **demo) + build_answer_instruction(option_labels)
print("Constructed prompt (truncated):")
print(prompt[:400] + ("..." if len(prompt) > 400 else ""))

parsed = parse_answer_from_text("3", option_labels)
assert parsed == "3"
parsed_garbage = parse_answer_from_text("I refuse to answer", option_labels)
assert parsed_garbage is None
print("OK: build_prompt + parse_answer_from_text round-trip correctly (valid digit parses, refusal returns None).")

print()
print("=" * 70)
print("ALL DRY-RUN CHECKS PASSED")
print("Mechanically, the pipeline is sound end-to-end. The only missing piece")
print("is real Pew question wording -- see src/data/build_pew_codebook.py.")
print("=" * 70)
