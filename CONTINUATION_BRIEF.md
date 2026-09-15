# Silicon Sampling — Continuation Brief (as of 2026-09-15)

Paste this whole file into a new chat, plus the files/folders listed at the
bottom, to continue this project without re-explaining anything.

## Project

Testing whether LLMs conditioned on demographic profiles can accurately
simulate individual WVS-7 India survey respondents ("silicon sampling"),
with fidelity checked across demographic subgroups (sex, age, region,
education, urban/rural).

## Status: Track A (zero-shot) is done, ~92% of the paper is written.
Track B (fine-tuning) is packaged and ready but **not yet run** — blocked on
the user physically going to their university's AI Lab.

## Track A results (final, corrected numbers)
- Best zero-shot LLM accuracy: Gemini 25.7%, Groq 23.0% (15-item scope,
  5-fold out-of-fold CV)
- Matched-feature ML baselines (same 14 demographic attributes as the LLM
  prompt): GBM 50.1% [47.6, 52.5] MAE 1.07, logistic 46.9% [44.3, 49.5]
  MAE 1.21 — baselines clearly beat the LLMs on raw accuracy
- Cross-model agreement (Gemini vs Groq): 36.8% raw agreement, Cohen's
  κ = 0.18 (only 9.3pts both-correct vs 27.6pts both-wrong)
- Central finding: no single demographic axis has a stable fidelity-gap
  ranking across models. Exception: Groq's near-zero aggregate sex gap
  is a real cancellation — a genuine 10.8pt gender-attitude-item gap is
  offset by a -3.2pt reversed gap elsewhere; Gemini shows no such
  cancellation (broad-based 11.3pt / 9.6pt gap instead).
- A real bug was found and fixed: the original India region-code labels in
  `src/prompts/verbalize.py` were guessed/wrong. Corrected against the real
  annex on p.227 of the WVS-7 codebook PDF. All predictions were re-run
  after the fix; numbers above are post-fix.

## Track B (fine-tuning) — plan and status
- Model: `openai/gpt-oss-120b` via QLoRA (4-bit), first model to run.
  Second model TBD (discussed Nemotron 3 family — Nano 31.6B/3B-active,
  Super 120B/12B-active, Ultra 550B/55B-active, hybrid Mamba-Transformer
  MoE — but no final decision made; leaning toward a cheaper second model
  like Nemotron 3 Nano or gpt-oss-20b rather than doubling down on another
  ~120B-class model, for GPU-hour budget reasons).
- Known technical risk for MoE models (gpt-oss-120b, Nemotron Super/Ultra):
  LoRA by default only targets attention q/k/v/o projections, not MoE
  router/expert weights — worth checking during preflight/smoke-test.
- Hardware: university AI Lab, 2x NVIDIA H100 (80GB HBM3) shared resource —
  NOT assumed exclusive/unlimited. Full server specs (Dell PowerEdge R760
  master + R760XA GPU node, 2x Xeon Gold 6438Y+, 512GB DDR5) were used to
  produce GPU-hour estimates, but availability/quota beyond that is unknown.
- A booking email was drafted (see `lab_access_email.txt` if still present
  in chat history / re-request if needed) asking for both H100s, remote
  SSH access, a 2-3 day window.
- **Script is written, hardened, and packaged for a one-shot run** — see
  `scripts/trackB_finetune.py`. Has `--preflight`, `--smoke-test`, and the
  real run mode. Handles: OOM auto-backoff (halves batch size, doubles
  grad accumulation, down to batch size 1), auto-resume from latest
  checkpoint on crash/reboot (same command re-run), SIGTERM/SIGINT
  emergency checkpoint save, frequent checkpointing (every 50 steps),
  partial-predictions saved every 20 respondents during eval, a sanity
  check that raises if too few training examples were built (catches
  pipeline bugs before spending GPU time).
- **Nothing has been run yet. Zero GPU-hours spent.** The next real action
  is the user going to the lab and running `--preflight`.
- Exact commands to run, in order, are in
  `scripts/TRACKB_LAB_INSTRUCTIONS.md` — follow it top to bottom, do not
  skip preflight or smoke-test.

## Data facts worth knowing
- 1,692 India WVS-7 respondents total, 613 columns after cleaning.
- 5-fold stratified split; fold 0 = 1,353 train / 339 test respondents.
- 15-item scope (used so far): 19,337 train / 4,807 test valid examples.
- 144-item scope (not yet used, flagged as a possible future widen):
  188,115 train / 46,975 test valid examples.

## Open / undecided items
1. Second model for Track B replication — not finalized.
2. Whether to widen from 15 items to the full 144-item battery — not
   actioned, just noted as a possible next step.
3. Paper citation/bibliography polish — noted as remaining ~8%, not done.
4. Track B has not been run — everything past "run preflight" is unknown
   until the user reports back real preflight/smoke-test/training output.

## Working style notes for whoever picks this up
- The user is fast-moving, informal, multitasking a presentation deadline;
  prefers short direct answers over long explanations.
- Any factual/model-identity question (e.g. "how is the Nemotron family")
  should be verified with a live search rather than answered from training
  data alone — model lineups move fast and stale answers have already
  caused one correction in this project.
- Do not recommend a specific model unless explicitly asked to (this was a
  hard constraint earlier in the project); when asked, ground the
  recommendation in the project's own measured numbers (prompt lengths,
  example counts, GPU-hour math) rather than generic claims.
- Any change to `src/prompts/verbalize.py` or other shared prompt code
  affects both Track A and Track B — re-run downstream evaluation after
  fixing anything there, don't assume unaffected.
- Data redistribution: WVS-7 microdata is public/free-use; the user has
  explicitly authorized using it in the university's own AI Lab
  infrastructure (this was previously a concern for third-party cloud
  services like Kaggle, but is resolved for the AI Lab).

## Published artifacts (already exist, can be updated in place if a new
chat has `Artifact` access — do not create duplicates)
- Project Audit / status page:
  https://claude.ai/code/artifact/8ee194e7-b11a-495c-ae2f-815aa38c5676
- Interim Findings:
  https://claude.ai/code/artifact/912449fa-4e97-4260-bfc5-f6e03eeb5301
- "One Respondent, Full Survey" (respondent #398, all 15 items,
  Gemini vs Groq):
  https://claude.ai/code/artifact/a0ac523b-232e-431e-ad8a-d4c8c49b43d4
- "All Methods, One Respondent" (6 LLM techniques + 4 statistical
  baselines, same respondent):
  https://claude.ai/code/artifact/4a75fc73-9039-4764-8c0f-beb41022f9e5
- "Silicon Sampling Pipeline" workflow diagram:
  https://claude.ai/code/artifact/aa284756-4997-4173-a18a-dd2c44d96d0a

## Files/folders to attach to a new chat to continue

**Minimum, to continue Track B lab work:**
- This file (`CONTINUATION_BRIEF.md`)
- `scripts/trackB_finetune.py`
- `scripts/TRACKB_LAB_INSTRUCTIONS.md`
- `scripts/TRACKB_PACKAGE_OVERVIEW.md`
- `scripts/trackB_requirements.txt`
- `src/` (whole folder — config.py, prompts/, inference/, eval/, data/, report/)
- `data/processed/ind_wvs7.parquet`
- `data/processed/selected_items.json`
- `data/processed/folds.json`
- `data/reference/wvs7_codebook.json`

**Add these if continuing paper/report writing or Track A analysis instead:**
- `PROJECT_REPORT.md`
- `paper/DRAFT.md`
- `src/eval/` scripts (baselines.py, run_baselines.py, cross_model_agreement.py,
  item_difficulty.py, sex_gap_investigation.py)
- `results/` folder (predictions, metrics) if it exists in the repo

**If reporting back lab results, also bring:**
- `trackB_status.json`
- `trackB_run.log`
- `results/predictions/trackB_openai_gpt-oss-120b_fold0_P2.parquet`

If in doubt, the safest single thing to hand over is the whole
`silicon-sampling-capstone` repo — it's small (data files are all under a
few MB total).
