# Track B Lab Package

This branch contains ONLY what's needed to run Track B fine-tuning on the
GPU machine. Your paper, project report, and Track A results live on `main`
-- this branch does not touch them.

## Before you leave for the lab

**You must add one file yourself, by hand, before this will run:**

```
data/processed/ind_wvs7.parquet
```

This is the real WVS-7 India respondent data. It is deliberately NOT in this
repo (`.gitignore` excludes it -- WVS microdata can't be redistributed
publicly). Copy it from wherever you generated it locally onto the same USB
stick / lab machine, into that exact path.

Everything else in this branch is already here and already verified (see
`test_pipeline_dryrun.py` below).

## What's in this branch

```
scripts/
  trackB_finetune.py          <- the fine-tuning script (hardened: OOM backoff,
                                  auto-resume, live progress logging -- see its
                                  own docstring for the full output-folder layout)
  trackB_requirements.txt
  TRACKB_LAB_INSTRUCTIONS.md  <- READ THIS FIRST, step by step
  TRACKB_PACKAGE_OVERVIEW.md

src/                          <- shared pipeline code (prompt building,
                                  demographic verbalization, config)

data/
  processed/selected_items.json   <- the 144 screened items (script uses first 15)
  processed/folds.json            <- the 5-fold train/test split
  reference/wvs7_codebook.json    <- real WVS-7 question wording + answer scales
  processed/ind_wvs7.parquet      <- YOU ADD THIS (see above)

test_pipeline_dryrun.py       <- optional: proves the prompt-building pipeline
                                  works against the real metadata files, without
                                  needing the parquet or a GPU. Already run and
                                  passed once during packaging; safe to skip, or
                                  re-run any time with `python test_pipeline_dryrun.py`

pyproject.toml
CONTINUATION_BRIEF.md         <- project context, in case you need to hand this
                                  off to a fresh chat
```

## Exact commands, in order

```bash
pip install -r scripts/trackB_requirements.txt
pip install huggingface_hub

tmux new -s trackb

python -m scripts.trackB_finetune --preflight       # ~5-10 min, mandatory
python -m scripts.trackB_finetune --smoke-test       # ~10-20 min, mandatory
python -m scripts.trackB_finetune --fold 0 --n-items 15   # the real run
```

If it dies or the machine reboots, run the exact same real-run command again
-- it auto-resumes from `trackB_run/checkpoints/`, it does not restart.

Full detail on every step: `scripts/TRACKB_LAB_INSTRUCTIONS.md`.

## Bringing results back

```
trackB_run/                                  <- the whole run folder, or at minimum:
trackB_run/logs/train_progress.jsonl         <- live training log
trackB_status.json                           <- final status (repo root)
results/predictions/trackB_openai_gpt-oss-120b_fold0_P2.parquet   <- the predictions
```
