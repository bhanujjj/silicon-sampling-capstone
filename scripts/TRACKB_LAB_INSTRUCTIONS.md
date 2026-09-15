# Track B fine-tuning at the AI Lab — READ THIS FULLY BEFORE STARTING

This is a one-shot, expensive run. The sequence below exists specifically so a
mistake costs you 10 minutes of preflight, not hours of wasted GPU time.
**Do not skip a step because "it'll probably be fine."**

## 0. Before you leave for the lab

- Copy the whole repo, or at minimum: `src/`, `scripts/`, `data/processed/`,
  `data/reference/wvs7_codebook.json`. A USB stick is fine — the data is
  under 1MB.
- Have this file and `trackB_finetune.py` on hand.

## 1. One-time environment setup

```bash
pip install -r scripts/trackB_requirements.txt
pip install huggingface_hub
```

If `openai/gpt-oss-120b` needs authentication to download:
```bash
huggingface-cli login
```

**Start a `tmux` session now, before anything else.** Everything from here on
runs inside it, so an SSH disconnect doesn't kill your job:

```bash
tmux new -s trackb
```

(To reattach later if you get disconnected: `tmux attach -t trackb`)

## 2. Preflight check — MANDATORY, do not skip

```bash
python -m scripts.trackB_finetune --preflight
```

Takes ~5–10 minutes. This checks CUDA, GPU count and memory, disk space, that
the data files are actually present, that Hugging Face Hub is reachable, and
— the important part — **actually loads the real model in 4-bit and runs one
real training step**, to prove the whole stack fits before you commit further.

- **If this fails: stop. Read the error. Do not proceed to the smoke test or
  real run.** Send me the output — this is exactly the kind of failure that's
  cheap to fix now and expensive to discover 6 hours into a real run.
- If it passes, it prints the peak GPU memory used during that one step —
  useful context if we need to debug memory headroom later.

## 3. Smoke test — MANDATORY, do not skip

```bash
python -m scripts.trackB_finetune --smoke-test
```

~10–20 minutes. Runs the full pipeline (data build → train → save → predict →
write output) on a tiny slice (5 train respondents, 5 test respondents, 3
items), including a checkpoint save/reload, so any data or schema bug shows
up here instead of after the real run has been going for hours.

If this fails, same rule: stop, send me the output, don't guess at a fix.

## 4. The real run

```bash
python -m scripts.trackB_finetune --fold 0 --n-items 15
```

(You're already inside `tmux`, so this survives disconnects on its own. You
can also add `nohup ... &` as extra insurance, but tmux is the primary
safety net here, not optional.)

**If the run dies or the machine reboots partway through: run the exact same
command again.** It automatically finds the latest checkpoint in
`./trackB_output` and resumes from there — it does NOT start over. This is
true whether it died from an OOM, a crash, a preemption, or you had to
close the session.

**If it hits a GPU-memory error mid-run:** the script catches this itself,
halves the batch size, and retries automatically — you shouldn't need to do
anything. If it still fails after retrying down to batch size 1, it will
tell you clearly and stop; at that point send me the log rather than trying
to hand-tune batch size yourself.

## 5. Checking progress without babysitting it

From another terminal (or another `tmux` pane), at any point:

```bash
cat trackB_status.json
```

This is a small file the script keeps overwriting with its current phase,
so you don't have to read the full log to know if it's still alive and
roughly where it is.

## 6. When it's done

The final summary prints to the terminal and to `trackB_run.log`, and
`trackB_status.json` will show `"phase": "complete"`. Send me:

- The full `trackB_status.json` (has the final accuracy/n/refusal numbers)
- `trackB_run.log` (or at least the last ~100 lines)
- The output file: `results/predictions/trackB_openai_gpt-oss-120b_fold0_P2.parquet`
  — get this off the lab machine (same USB stick). I need the actual
  predictions file to run the subgroup fidelity analysis, not just the
  headline accuracy number.

## If anything looks wrong that isn't covered above

**Stop and send me what you're seeing before trying to fix it yourself.**
The whole point of the preflight/smoke-test sequence is that by the time
you're in the real run, surprises should be rare — if one happens anyway,
diagnosing it correctly matters more than diagnosing it fast, especially
with real lab-time cost on the line.
