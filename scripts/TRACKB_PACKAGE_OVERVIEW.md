# What's in the Track B USB package — short version

## Top level

- **`START_HERE.md`** — the entry point. Says what's in the folder and points to the instructions.
- **`pyproject.toml`** — project metadata, lets `src/` be imported as a package. You won't need to touch it.

## `scripts/` — the actual run

- **`trackB_finetune.py`** — the fine-tuning script itself. Three things it does:
  1. Builds training examples from real respondents (demographics + survey question → their real answer), using the same prompt format Track A used zero-shot.
  2. QLoRA fine-tunes the model on those examples (freezes the model, trains small "adapter" layers on top — cheap, fast, doesn't need retraining all 120B parameters).
  3. Tests the fine-tuned model on held-out respondents it never trained on, and saves the predictions.
  Has three modes: `--preflight` (safety check, no real training), `--smoke-test` (tiny fake run to catch bugs), and the real run (`--fold 0 --n-items 15`).
- **`trackB_requirements.txt`** — the Python packages to `pip install` before anything else (torch, transformers, peft, trl, bitsandbytes, accelerate).
- **`TRACKB_LAB_INSTRUCTIONS.md`** — the actual step-by-step to follow, in order. This is the one file you actually read start to finish.

## `src/` — the shared pipeline code (same code Track A used)

- **`config.py`** — file paths and constants (where the data lives, fold count, etc.).
- **`prompts/verbalize.py`** — turns a respondent's raw numeric codes (e.g. `Q275=6`) into readable text ("Bachelor's degree") using the official codebook. Also builds each survey question's text and answer options.
- **`prompts/templates.py`** — assembles the final prompt sent to the model (the P2 condition: full demographic profile + question + answer options).
- **`inference/prompting.py`** — the closing instruction appended to every prompt ("respond with exactly one digit...") and the logic that reads the model's reply back into an answer code. Shared with Track A so the comparison is fair.
- **`eval/`** — analysis scripts (subgroup fidelity gaps, cross-model agreement, etc.) for after you have results. Not needed to run the fine-tuning itself, included so you can analyze the output on the same machine if you want.
- **`data/`, `report/`** — supporting code for the original data-cleaning and report-generation steps. Not used by the fine-tuning run itself, included for completeness.

## `data/` — the actual dataset (tiny, already cleaned)

- **`processed/ind_wvs7.parquet`** — the 1,692 India respondents, cleaned, one row each. This is what the model trains and is tested on.
- **`processed/selected_items.json`** — the 144 survey questions that passed quality screening (the script uses the first 15 by default, matching Track A).
- **`processed/folds.json`** — the 5-way train/test split. Fold 0 = 1,353 respondents to train on, 339 held out to test on.
- **`reference/wvs7_codebook.json`** — the official WVS-7 question wording and answer-scale definitions, parsed from the source PDF. This is what makes the prompts real survey text instead of "Question Q4."

## What actually happens when you run it

1. Take ~1,353 respondents × 15 questions → ~19,000 (demographics, question, real answer) examples.
2. Fine-tune the model on those.
3. Ask the fine-tuned model to predict answers for the 339 respondents it never saw.
4. Compare its guesses to their real answers → accuracy number, saved to a file you bring back.
