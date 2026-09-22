# Pew Track — Lab Package

This is the Pew-dataset counterpart to `README-LAB.md` (WVS-7 Track B). Same
branch philosophy: this touches only the Pew pipeline, not your paper or
Track A results.

## Before you leave for the lab — ONE thing you must add by hand

**The raw Pew CSV**, at this exact path:

```
data/raw/India Religion Public Data - Pew Research Center (All Vars).csv
```

Not in git (`.gitignore` excludes `data/raw/*` — same treatment as the WVS
parquet, pending confirmation of Pew's exact redistribution terms: Pew's
own `READ ME.txt` describes the study but states no redistribution terms
either way, so this stays gitignored as the cautious default).

**Everything else is already resolved and committed.** Real question
wording for **302 of 304 columns**, and **73 fully-screened items** ready
to train on, all parsed from Pew's own `Pew India DDI metadata.xml`
(DDI-Codebook 2.5, the machine-readable metadata file in Pew's release --
see `scripts/parse_pew_ddi_xml.py`) and cross-checked against
`CODEBOOK_India.pdf`. Confirmed by actually running the real
training-data-construction code against the real 29,999-row dataset:
fold 0 alone builds **349,415 real (prompt, answer) examples** across the
first 15 selected items -- a genuinely large, real, verified dataset, not
a placeholder.

An earlier hand-parse of "India recode syntax for public release.txt"
(before the DDI XML was available) recovered 21 items as a side effect of
that file's derived-variable definitions -- see
`data/reference/pew_labels_recode_syntax.json`, kept for provenance. The
DDI XML supersedes it (covers everything that file did, plus the rest).

231 of the 304 columns are correctly excluded from item selection
regardless of wording -- 195 for being binary (below the project's
existing `MIN_RESPONSE_SCALE_SIZE = 4`, shared with WVS: a real,
considered choice about comparability, not a bug -- raise it in
`src/config.py` if you want binary items included too), 19 as
demographic-conditioning columns, 11 for missingness above 10%, the rest
for non-ordinal codes or near-unanimous answers. Only `QHH1`/`QHH2`
(household composition fields) remain genuinely unlabeled.

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
  parse_pew_ddi_xml.py          <- turns Pew's DDI metadata XML into real
                                   question/answer labels (the primary
                                   source used already -- see above)
  parse_pew_recode_syntax.py   <- turns Pew's SPSS recode-syntax .txt into
                                   real question/answer labels (superseded
                                   by the DDI parser, kept for provenance)

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
  processed/pew_selected_items.json  <- 73 screened items, ready to train on
  processed/pew_folds.json           <- 5-fold split (respondent IDs only,
                                        safe to commit -- no survey answers)
  reference/pew_codebook.json        <- question metadata (302/304 verified)
  reference/pew_labels_ddi.json      <- the parsed DDI labels merged into
                                        the codebook above (regenerate with
                                        parse_pew_ddi_xml.py if Pew ever
                                        revises the release)
  raw/<csv>, processed/pew_india_2021.parquet  <- YOU ADD/GENERATE THESE (see above)

test_pipeline_dryrun_pew.py    <- proves the prompt-building pipeline is
                                   mechanically sound against real committed
                                   metadata + a real respondent row, without
                                   needing a GPU. Already run and passed.
```

## Exact commands, in order

```bash
pip install -r scripts/pewB_requirements.txt
pip install huggingface_hub

# 1. Build the processed data (run once -- codebook/selected-items/folds
#    are already committed with real labels; re-run only if you regenerate
#    data/raw/<csv> or want to re-derive them from scratch)
python -m src.data.load_pew
python -m src.data.build_pew_codebook --labels data/reference/pew_labels_ddi.json
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
