# Pew Track — Lab Package

This is the Pew-dataset counterpart to `README-LAB.md` (WVS-7 Track B). Same
branch philosophy: this touches only the Pew pipeline, not your paper or
Track A results.

## Before you leave for the lab — TWO things you must add by hand

**1. The raw Pew CSV**, at this exact path:

```
data/raw/India Religion Public Data - Pew Research Center (All Vars).csv
```

Not in git (`.gitignore` excludes `data/raw/*` — same treatment as the WVS
parquet, pending confirmation of Pew's exact redistribution terms).

**2. Pew's own codebook document** (whichever you have — any one of these three):

```
CODEBOOK_India.pdf
India recode syntax for public release.txt
Pew India DDI metadata.xml
```

This second one is the real blocker (see next section) — the CSV alone is
not enough to run this pipeline.

**Current progress: 1 of ~280 candidate items verified** (`Q43b`, "how
certain are you in your belief in God" — 4-point scale), sourced from
`data/reference/pew_labels_recode_syntax.json` (real wording quoted from
Pew's own "recode syntax for public release.txt", not guessed — see that
file for provenance per item). Confirmed by an actual end-to-end run of
the real training-data-construction code against the real 29,999-row
dataset: **22,859 real (prompt, answer) training examples** were built
correctly from fold 0 alone. That's real proof the mechanics work — 1
item is just not enough to train anything useful yet. `CODEBOOK_India.pdf`
or `Pew India DDI metadata.xml` (the other two files in Pew's release,
either one, per `READ ME.txt`) has the wording for the rest.

## Why a codebook document is required, not optional

Pew's raw CSV ships bare numeric codes (`Q37a` = 1/2/98/99) with **no
question text or answer-option labels attached**. There is no public,
already-labeled version of this file. Training on / reporting a *guessed*
Pew survey question in your paper would misattribute fabricated content to
a real research organization — so this pipeline is built to refuse to do
that: `data/reference/pew_codebook.json` currently has real, verified
labels for 23 of 304 columns (sex, religion, and 21 more pulled from Pew's
"recode syntax for public release.txt" — see above), and only 1 of those
23 survives full item screening (`Q43b`) once the demographic columns are
excluded and WVS's existing scale/missingness/entropy filters are applied
(most of the 21 are plain yes/no items, filtered out by the project's
existing `MIN_RESPONSE_SCALE_SIZE = 4` — a real, considered choice, not a
bug: see the "silicon sampling" project's earlier discussion on why mixing
coarse binary items with fine-grained ones makes accuracy numbers hard to
compare fairly. That threshold lives in `src/config.py` and is shared with
WVS, so change it there, not per-track, if you want binary Pew items included).

**The recode-syntax file was NOT the primary source** — per Pew's own
`READ ME.txt`, that file only holds *derived/combined* variables (age
buckets, caste buckets, combined topline variables), not the full
question-by-question wording. It happened to have enough embedded context
in a few variable labels to confirm 21 real base items as a side effect.
The actual full codebook is `CODEBOOK_India.pdf` (human-readable) or `Pew
India DDI metadata.xml` (machine-readable, likely easier to parse
completely) — either one unlocks the rest of the ~280 remaining items.

**To unblock it**, once you have one of the three codebook documents above:

```bash
# If you have the recode-syntax .txt (try this first, it's plain text):
python scripts/parse_pew_recode_syntax.py "data/raw/India recode syntax for public release.txt" -o data/reference/pew_labels_raw.json
# NOT YET VERIFIED against the real file -- if it parses fewer than ~50
# variables, the file's exact syntax differs from what the parser expects;
# open the .txt and compare it against parse_pew_recode_syntax.py's
# docstring, or come back with the real format and this will be fixed.

python -m src.data.build_pew_codebook --labels data/reference/pew_labels_raw.json
python -m src.data.select_items_pew
```

If `select_items_pew` reports more than 0 items, you're unblocked — nothing
downstream (training script, notebook, inference, test panel) needs any
further changes.

## What's in this branch

```
scripts/
  pewB_finetune.py             <- fine-tuning script (.py version)
  pewB_finetune.ipynb          <- fine-tuning notebook (.ipynb version, for
                                   labs that require Jupyter -- same logic,
                                   same safety checks, mirrors trackB's notebook)
  pewB_requirements.txt
  pewB_inference.py            <- load a TRAINED model back up and answer new
                                   questions WITHOUT retraining
  pewB_test_panel.py           <- paper-ready markdown table: real (demographics,
                                   question, true answer, model answer) rows
                                   from held-out respondents
  parse_pew_recode_syntax.py   <- turns Pew's SPSS recode-syntax .txt into
                                   real question/answer labels (see above)

src/
  data/load_pew.py             <- raw CSV -> cleaned parquet (missing-code
                                   recoding, respondent_id)
  data/build_pew_codebook.py   <- raw CSV -> structured codebook JSON
  data/select_items_pew.py     <- screens columns for use as training targets
  data/build_folds_pew.py      <- 5-fold stratified train/test split
  prompts/verbalize_pew.py     <- Pew demographics -> prompt-ready dict
  eval/subgroups_pew.py        <- Pew-side fidelity-gap subgroup slicing
                                   (adds caste as a bonus axis WVS doesn't have)

data/
  processed/pew_selected_items.json  <- screened items (0 until codebook is done)
  processed/pew_folds.json           <- 5-fold split (respondent IDs only,
                                        safe to commit -- no survey answers)
  reference/pew_codebook.json        <- question metadata (2/304 verified as shipped)
  raw/<csv>, processed/pew_india_2021.parquet  <- YOU ADD THESE (see above)

test_pipeline_dryrun_pew.py    <- proves the prompt-building pipeline is
                                   mechanically sound against real committed
                                   metadata + a real respondent row, without
                                   needing a GPU. Already run and passed.
```

## Exact commands, in order

```bash
pip install -r scripts/pewB_requirements.txt
pip install huggingface_hub

# 1. Build the processed data (run once)
python -m src.data.load_pew
python -m src.data.build_pew_codebook --labels <your parsed labels file>   # see above
python -m src.data.select_items_pew
python -m src.data.build_folds_pew

# 2. Sanity check before touching the GPU (no CUDA needed for this one)
python test_pipeline_dryrun_pew.py

tmux new -s pewb
python -m scripts.pewB_finetune --preflight
python -m scripts.pewB_finetune --smoke-test
python -m scripts.pewB_finetune --fold 0 --n-items 15
# Ctrl+B then D to detach; `tmux attach -t pewb` to come back

# 3. After training: answer new questions without retraining
python scripts/pewB_inference.py --model-path pewB_run/model_fold0 \
  --base-model openai/gpt-oss-20b --question-id Q37a --sex Female --religion Hindu

# 4. Paper-ready results table
python scripts/pewB_test_panel.py --model-path pewB_run/model_fold0 \
  --base-model openai/gpt-oss-20b --fold 0 --n-respondents 8 --n-items 5
```

For the headline accuracy/MAE/fidelity-gap numbers (not the illustrative
test panel), run the same reporting pipeline Track A/B already use — the
predictions parquet this script writes matches that exact schema:

```bash
python -m src.report.evaluate_run --predictions results/predictions/pewB_openai_gpt-oss-20b_fold0_P2.parquet
```

## Why gpt-oss-20b instead of gpt-oss-120b

Same MoE family and chat template as the WVS Track B model, ~5x smaller
download (~13GB vs ~60GB), fits easily in 4-bit on a single lab GPU
(`MIN_GPU_MEM_GB = 16` in the preflight check, vs 60 for the 120B model).
Change `CONFIG["model"]` / `--model` back to `openai/gpt-oss-120b` if you
want the larger model instead — nothing else needs to change either way.

## A real bug this package's verification found and fixed

`src/prompts/templates.py`'s `format_p2_structured()` crashed
(`TypeError: unexpected keyword argument 'caste'`) the first time it was
actually run against Pew demographics, because it — unlike the other three
prompt templates — had no catch-all for extra kwargs. Fixed by adding
`caste` as a real, rendered field plus `**_ignored` for future keys. This
also affects the WVS Track B code path (same shared file), though it never
surfaced there since WVS's demographics dict happens to match P2's exact
14 keys.
