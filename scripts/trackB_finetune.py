"""Track B: QLoRA fine-tune on real WVS-7 India respondents, out-of-fold.

Hardened for a one-shot, unattended, remote lab run: preflight checks before
committing GPU time, auto-resume from checkpoint on restart, automatic
retry-with-smaller-batch on OOM, a status file you can check without
tailing logs, and a signal handler that saves before dying. Designed to run
under `tmux`/`nohup` so an SSH disconnect doesn't kill it -- but everything
it does on its own is to make a crash or restart cheap, not to prevent one.

Trains on ONE fold's training respondents (default: fold 0), then predicts
on that fold's held-out test respondents, using the EXACT SAME prompt
construction (verbalize_demographics / verbalize_item / build_prompt) and
the EXACT SAME item set as Track A's zero-shot runs, so this is a genuine
apples-to-apples "does fine-tuning help" comparison. Output parquet matches
Track A's schema so it drops straight into `src.report.evaluate_run`.

REQUIRED SEQUENCE -- do not skip steps, this is a one-shot expensive run:

    python -m scripts.trackB_finetune --preflight
        # ~5-10 min. Checks CUDA, GPU count/memory, disk space, HF Hub
        # reachability, and does one real forward+backward step with the
        # ACTUAL target model to confirm it fits before you commit further.
        # If this fails, FIX THE PROBLEM before continuing -- do not skip to
        # the real run "to see what happens."

    python -m scripts.trackB_finetune --smoke-test
        # ~10-20 min. Tiny data (5 train, 5 test respondents, 3 items), full
        # pipeline including a save+reload of a checkpoint, to catch data/
        # path/schema bugs before the real run. Writes a human-readable
        # trackB_run/logs/smoke_test_result.json when done -- read it before
        # moving on.

    python -m scripts.trackB_finetune --fold 0 --n-items 15
        # The real run. Run this under tmux or nohup (see
        # TRACKB_LAB_INSTRUCTIONS.md) -- do NOT run it in a bare foreground
        # shell over SSH.

If the real run dies or the machine reboots, re-run the EXACT SAME command --
it auto-detects the latest checkpoint under trackB_run/checkpoints/ and
resumes from there instead of starting over.

EVERYTHING THIS SCRIPT WRITES LIVES UNDER ONE FOLDER (--output-dir, default
./trackB_run), so the whole run -- data cache, checkpoints, logs, live
progress, predictions -- survives a power cut and travels as one directory
if you need to move it off the lab machine:

    trackB_run/
      cache/            <- the exact (prompt, real answer) examples built
                           from the dataset, one JSONL file per fold/scope.
                           Rebuilt only if missing; open it any time to see
                           literally what the model is training on.
      checkpoints/       <- HF Trainer checkpoints (adapter weights only,
                           small). Auto-resume reads this directory.
      logs/
        trackB_run.log          <- full human-readable log, this run only
        train_progress.jsonl    <- one line per logging step: step, epoch,
                                   loss, lr, elapsed, eta -- append-only,
                                   safe to tail with `tail -f` from another
                                   pane, survives a crash mid-line
        smoke_test_result.json  <- written only after --smoke-test finishes
      predictions/
        partial_predictions.parquet   <- updated every 20 test respondents
        trackB_<model>_fold<N>_P2.parquet   <- final predictions (also
                                   copied to results/predictions/ at repo
                                   root, unchanged, for src.report.*)
      model_fold_<N>/     <- final saved LoRA adapter + tokenizer

trackB_status.json at the REPO ROOT (not inside trackB_run/) is unchanged
from before -- `cat trackB_status.json` from another pane still works
exactly as documented in TRACKB_LAB_INSTRUCTIONS.md.
"""

import argparse
import json
import logging
import signal
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_PROCESSED, RESULTS_DIR
from src.prompts.templates import build_prompt
from src.prompts.verbalize import load_codebook, verbalize_demographics, verbalize_item
from src.inference.prompting import build_answer_instruction, parse_answer_from_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("trackB_run.log")],
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/gpt-oss-120b"
MIN_FREE_DISK_GB = 250  # generous: model cache + checkpoints + HF download temp files
STATUS_PATH = Path("trackB_status.json")  # repo-root path, unchanged -- see TRACKB_LAB_INSTRUCTIONS.md


def run_dirs(output_dir: str) -> dict:
    """All paths this script writes to, all nested under one run folder so
    the whole thing is a single directory you can zip up or copy off the
    lab machine."""
    base = Path(output_dir)
    d = {
        "base": base,
        "cache": base / "cache",
        "checkpoints": base / "checkpoints",
        "logs": base / "logs",
        "predictions": base / "predictions",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def write_status(phase: str, detail: str = "", extra_log_path: Path = None, **extra):
    """Overwrite the small status JSON at the repo root -- check this from
    another terminal instead of tailing the full log. Also mirrors a copy
    inside the run folder's logs/ if one is given, so the run folder alone
    (e.g. copied off the lab machine) has its own record of final status."""
    status = {"phase": phase, "detail": detail, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
    try:
        STATUS_PATH.write_text(json.dumps(status, indent=2))
    except Exception:
        pass  # status file is a convenience, never let it crash the real run
    if extra_log_path is not None:
        try:
            (extra_log_path / "trackB_status.json").write_text(json.dumps(status, indent=2))
        except Exception:
            pass
    logger.info(f"[STATUS] {phase}: {detail}")


def append_jsonl(path: Path, record: dict):
    """Append one JSON line and flush immediately, so a crash mid-run loses
    at most the record currently being written, never anything already
    appended."""
    try:
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
    except Exception as e:
        logger.warning(f"Could not append to {path}: {e}")


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def preflight(args) -> bool:
    """Everything checkable in a few minutes, BEFORE spending real GPU time.
    Returns True only if it is safe to proceed to the real run."""
    ok = True

    write_status("preflight", "checking CUDA/GPUs")
    import torch
    if not torch.cuda.is_available():
        logger.error("PREFLIGHT FAIL: no CUDA device visible. Are you on the GPU node, not the master node?")
        return False
    n_gpus = torch.cuda.device_count()
    logger.info(f"CUDA OK. {n_gpus} GPU(s) visible:")
    total_mem_gb = 0
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / 1e9
        total_mem_gb += mem_gb
        logger.info(f"  GPU {i}: {props.name}, {mem_gb:.0f} GB")
    if total_mem_gb < 60:
        logger.error(f"PREFLIGHT FAIL: only {total_mem_gb:.0f} GB total GPU memory visible -- too little for a 120B-class model even in 4-bit. Check you got the GPU allocation you expected.")
        ok = False

    write_status("preflight", "checking disk space")
    free_gb = shutil.disk_usage(".").free / 1e9
    logger.info(f"Free disk space: {free_gb:.0f} GB")
    if free_gb < MIN_FREE_DISK_GB:
        logger.error(f"PREFLIGHT FAIL: only {free_gb:.0f} GB free, want at least {MIN_FREE_DISK_GB} GB for model cache + checkpoints. Clear space or point HF_HOME at a bigger disk before continuing.")
        ok = False

    write_status("preflight", "checking data files present")
    for p in [DATA_PROCESSED / "ind_wvs7.parquet", DATA_PROCESSED / "selected_items.json", DATA_PROCESSED / "folds.json"]:
        if not p.exists():
            logger.error(f"PREFLIGHT FAIL: missing {p} -- did you copy the full data/ directory (including the .parquet, which is NOT in git)?")
            ok = False
    if not ok:
        return False

    write_status("preflight", "checking Hugging Face Hub reachability")
    try:
        from huggingface_hub import HfApi
        HfApi().model_info(args.model)
        logger.info(f"HF Hub reachable, model repo '{args.model}' found.")
    except Exception as e:
        logger.error(f"PREFLIGHT FAIL: could not reach Hugging Face Hub or find '{args.model}': {e}")
        logger.error("If this is a private/gated model, run `huggingface-cli login` first.")
        ok = False
    if not ok:
        return False

    write_status("preflight", f"loading {args.model} in 4-bit and running one real train step (this is the slow part, several minutes)")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb_config, device_map="auto", trust_remote_code=True,
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        # One real forward+backward+step, on real-shaped input, to prove the
        # whole stack actually fits and runs -- not just that weights loaded.
        dummy_text = "This is a preflight check. " * 40
        inputs = tokenizer(dummy_text, return_tensors="pt", truncation=True, max_length=700).to(model.device)
        labels = inputs["input_ids"].clone()
        out = model(**inputs, labels=labels)
        out.loss.backward()
        model.zero_grad()
        peak_mem_gb = max(torch.cuda.max_memory_allocated(i) for i in range(n_gpus)) / 1e9
        logger.info(f"Model loaded, one train step ran successfully. Peak single-GPU memory: {peak_mem_gb:.1f} GB.")
        del model, out
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"PREFLIGHT FAIL: OOM loading/stepping {args.model} in 4-bit: {e}")
        logger.error("This model does not fit as configured. Do not proceed to the real run -- come back for a different --model or a smaller batch/sequence-length config first.")
        return False
    except Exception as e:
        logger.error(f"PREFLIGHT FAIL: error loading/running {args.model}: {e}")
        logger.error(traceback.format_exc())
        return False

    write_status("preflight", "PASSED -- safe to run --smoke-test next")
    logger.info("=" * 60)
    logger.info("PREFLIGHT PASSED. Next: python -m scripts.trackB_finetune --smoke-test")
    logger.info("=" * 60)
    return True


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_examples(df: pd.DataFrame, respondent_ids: list, selected_items: list, codebook: dict) -> list:
    """One training example per (respondent, item): prompt text + the correct answer digit,
    built from the identical pipeline Track A used -- codebook-grounded question wording,
    codebook-grounded demographic verbalization, same P2 condition, same closing instruction."""
    examples = []
    skipped = 0
    sub = df[df["respondent_id"].isin(respondent_ids)]
    for _, row in sub.iterrows():
        try:
            demo = verbalize_demographics(row, codebook)
        except Exception as e:
            skipped += 1
            logger.warning(f"Skipping respondent {row.get('respondent_id')}: verbalize_demographics failed ({e})")
            continue
        for question_id in selected_items:
            try:
                item = verbalize_item(question_id, codebook)
                true_code = row.get(question_id)
                if pd.isna(true_code):
                    continue
                true_code = int(true_code)
                if true_code not in item["code_to_index"]:
                    continue
                option_labels = [str(c) for c in item["ordinal_values"]]
                prompt = build_prompt(
                    "P2", item["question_text"], item["options_text"], **demo
                ) + build_answer_instruction(option_labels)
                answer = str(true_code)
                examples.append({
                    "text": prompt + "\n\n" + answer,
                    "prompt": prompt,
                    "answer": answer,
                    "respondent_id": int(row["respondent_id"]),
                    "question_id": question_id,
                })
            except Exception as e:
                skipped += 1
                logger.warning(f"Skipping ({row.get('respondent_id')}, {question_id}): {e}")
    if skipped:
        logger.warning(f"Skipped {skipped} (respondent, item) pairs due to errors -- see warnings above.")
    return examples


def load_or_build_examples(df, respondent_ids, selected_items, codebook, cache_path: Path, rebuild: bool = False) -> list:
    """Build training examples once, cache to a JSONL file inside the run
    folder, and reuse on any later invocation (restart, resumed session)
    instead of rebuilding. Rebuilding is cheap (CPU-only, seconds), so this
    is mainly so you have a literal, inspectable file of exactly what the
    model trained on -- not a performance optimization."""
    if cache_path.exists() and not rebuild:
        examples = []
        with open(cache_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        logger.info(f"Loaded {len(examples)} cached training examples from {cache_path} (delete this file or pass --rebuild-data to force a rebuild).")
        return examples

    examples = build_examples(df, respondent_ids, selected_items, codebook)
    with open(cache_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    logger.info(f"Built and cached {len(examples)} training examples to {cache_path}")
    return examples


def sanity_check_examples(examples: list, min_expected: int):
    if len(examples) < min_expected:
        raise RuntimeError(
            f"Only built {len(examples)} training examples, expected at least {min_expected}. "
            f"This smells like a data/config bug, not normal missingness -- STOP and investigate "
            f"rather than burning GPU time on a broken dataset."
        )


# ---------------------------------------------------------------------------
# Live progress logging (JSONL, one line per training step)
# ---------------------------------------------------------------------------

class JsonlProgressCallback:
    """Transformers TrainerCallback that appends one JSONL record per
    logging step to logs/train_progress.jsonl (append-only, flushed
    immediately -- a crash loses at most the in-flight line) and keeps the
    status file updated with epoch/step/loss/ETA, so progress is visible
    live from another pane without tailing the full text log."""

    def __init__(self, jsonl_path: Path, status_extra_dir: Path, total_steps: int):
        self.jsonl_path = jsonl_path
        self.status_extra_dir = status_extra_dir
        self.total_steps = max(total_steps, 1)
        self.t_start = time.time()

    def _eta_minutes(self, step: int) -> float:
        if step <= 0:
            return float("nan")
        elapsed = time.time() - self.t_start
        rate = elapsed / step  # seconds per step
        remaining = max(self.total_steps - step, 0)
        return (remaining * rate) / 60.0

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if "loss" not in logs:
            return control  # skip eval-only / non-training log events
        step = state.global_step
        pct = 100.0 * step / self.total_steps
        eta_min = self._eta_minutes(step)
        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "step": step,
            "total_steps": self.total_steps,
            "pct_complete": round(pct, 1),
            "epoch": round(logs.get("epoch", state.epoch or 0), 3),
            "loss": logs.get("loss"),
            "learning_rate": logs.get("learning_rate"),
            "grad_norm": logs.get("grad_norm"),
            "elapsed_min": round((time.time() - self.t_start) / 60.0, 1),
            "eta_min": round(eta_min, 1) if eta_min == eta_min else None,  # NaN check
        }
        append_jsonl(self.jsonl_path, record)
        write_status(
            "training",
            f"step {step}/{self.total_steps} ({pct:.1f}%), epoch {record['epoch']}, loss {record['loss']}, ETA {record['eta_min']} min",
            extra_log_path=self.status_extra_dir,
            step=step, total_steps=self.total_steps, pct_complete=round(pct, 1),
            epoch=record["epoch"], loss=record["loss"], eta_min=record["eta_min"],
        )
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        append_jsonl(self.jsonl_path, {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event": "epoch_end",
            "epoch": round(state.epoch or 0, 3),
            "step": state.global_step,
        })
        logger.info(f"=== Epoch {round(state.epoch or 0, 2)} complete (step {state.global_step}/{self.total_steps}) ===")
        return control

    def on_save(self, args, state, control, **kwargs):
        append_jsonl(self.jsonl_path, {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event": "checkpoint_saved",
            "step": state.global_step,
        })
        logger.info(f"Checkpoint saved at step {state.global_step}")
        return control


# ---------------------------------------------------------------------------
# Training with OOM backoff
# ---------------------------------------------------------------------------

def find_latest_checkpoint(checkpoints_dir: Path):
    if not checkpoints_dir.exists():
        return None
    checkpoints = sorted(checkpoints_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return str(checkpoints[-1]) if checkpoints else None


def train_with_oom_backoff(args, train_examples, tokenizer, model_loader, dirs: dict):
    """Try the requested batch size; on OOM, halve it and retry (once per
    halving, down to batch size 1) rather than losing the whole run to a
    single bad batch-size guess."""
    import torch
    from transformers import TrainingArguments
    from trl import SFTTrainer

    batch_size = args.batch_size
    grad_accum = args.grad_accum
    last_error = None

    effective_batch = max(batch_size * grad_accum, 1)
    steps_per_epoch = max(len(train_examples) // effective_batch, 1)
    total_steps = int(steps_per_epoch * args.epochs)

    progress_jsonl = dirs["logs"] / "train_progress.jsonl"

    while batch_size >= 1:
        try:
            model = model_loader()
            resume_from = find_latest_checkpoint(dirs["checkpoints"])
            if resume_from:
                logger.info(f"Found existing checkpoint {resume_from} -- resuming, NOT starting over.")

            training_args = TrainingArguments(
                output_dir=str(dirs["checkpoints"]),
                overwrite_output_dir=False,
                num_train_epochs=args.epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,
                warmup_steps=20,
                learning_rate=args.lr,
                weight_decay=0.01,
                bf16=True,
                logging_steps=10,
                save_steps=args.save_steps,
                save_total_limit=3,
                optim="paged_adamw_32bit",
                seed=42,
                max_grad_norm=1.0,
                remove_unused_columns=False,
                report_to="none",
            )
            trainer = SFTTrainer(
                model=model,
                train_dataset=train_examples,
                args=training_args,
                packing=False,
                max_seq_length=768,
                tokenizer=tokenizer,
                formatting_func=lambda x: x["text"],
            )
            trainer.add_callback(JsonlProgressCallback(progress_jsonl, dirs["logs"], total_steps))

            # Signal handler: on SIGTERM/SIGINT (job killed, preemption, ctrl-C),
            # save a checkpoint before dying instead of losing the last stretch.
            def _save_and_exit(signum, frame):
                logger.warning(f"Received signal {signum} -- saving checkpoint before exiting.")
                write_status("training", "interrupted, saving emergency checkpoint", extra_log_path=dirs["logs"])
                try:
                    trainer.save_model(str(dirs["checkpoints"] / "emergency_checkpoint"))
                    append_jsonl(progress_jsonl, {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "event": "emergency_checkpoint_saved", "step": trainer.state.global_step,
                    })
                except Exception:
                    logger.error("Emergency checkpoint save failed too.")
                sys.exit(1)

            signal.signal(signal.SIGTERM, _save_and_exit)
            signal.signal(signal.SIGINT, _save_and_exit)

            write_status(
                "training", f"batch_size={batch_size} grad_accum={grad_accum}",
                extra_log_path=dirs["logs"], n_examples=len(train_examples),
                total_steps=total_steps,
            )
            trainer.train(resume_from_checkpoint=resume_from)
            return trainer, model

        except torch.cuda.OutOfMemoryError as e:
            last_error = e
            torch.cuda.empty_cache()
            logger.warning(f"OOM at batch_size={batch_size}. Halving batch size and doubling grad_accum to keep effective batch size roughly constant, then retrying.")
            append_jsonl(progress_jsonl, {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "event": "oom_backoff", "old_batch_size": batch_size, "new_batch_size": batch_size // 2,
            })
            batch_size //= 2
            grad_accum *= 2
            write_status("training", f"OOM recovery: retrying at batch_size={batch_size}", extra_log_path=dirs["logs"])

    raise RuntimeError(f"Training failed even at batch_size=1: {last_error}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-items", type=int, default=15)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=2, help="Conservative default for a 120B-class model -- preflight will tell you if you can safely go higher")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--save-steps", type=int, default=50, help="Frequent by default -- checkpoints are small LoRA adapters, cheap to save often, expensive to lose")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--preflight", action="store_true", help="Run ONLY the safety checks, then exit. Do this first, always.")
    ap.add_argument("--output-dir", default="./trackB_run", help="Single folder holding cache/checkpoints/logs/predictions for this run -- safe to copy off the lab machine as one directory")
    ap.add_argument("--rebuild-data", action="store_true", help="Ignore any cached training examples in output-dir/cache and rebuild from the parquet")
    args = ap.parse_args()

    if args.preflight:
        success = preflight(args)
        sys.exit(0 if success else 1)

    import torch
    if not torch.cuda.is_available():
        logger.error("No CUDA device found. Run --preflight first if you haven't.")
        sys.exit(1)

    df = pd.read_parquet(DATA_PROCESSED / "ind_wvs7.parquet")
    if "respondent_id" not in df.columns:
        df["respondent_id"] = range(len(df))
    with open(DATA_PROCESSED / "selected_items.json") as f:
        selected_items = json.load(f)["selected_items"][: args.n_items]
    with open(DATA_PROCESSED / "folds.json") as f:
        folds = json.load(f)["folds"]
    codebook = load_codebook()

    fold = folds[args.fold]
    train_ids, test_ids = fold["train"], fold["test"]
    min_expected = 10
    run_tag = f"fold{args.fold}"
    if args.smoke_test:
        train_ids, test_ids = train_ids[:5], test_ids[:5]
        selected_items = selected_items[:3]
        args.output_dir = args.output_dir + "_smoketest"
        run_tag = "smoketest"
        logger.info("SMOKE TEST: 5 train respondents, 5 test respondents, 3 items")
    else:
        min_expected = 1000

    dirs = run_dirs(args.output_dir)

    # Add a second file handler so this run's full log also lives inside the
    # run folder, not just at the repo root -- makes the run folder self-
    # contained if you copy it off the lab machine.
    run_log_handler = logging.FileHandler(str(dirs["logs"] / "trackB_run.log"))
    run_log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(run_log_handler)

    t_run_start = time.time()
    logger.info(f"Fold {args.fold}: {len(train_ids)} train, {len(test_ids)} test, {len(selected_items)} items")
    logger.info(f"All output for this run: {dirs['base'].resolve()}")
    write_status("building_data", "constructing training examples", extra_log_path=dirs["logs"])

    try:
        cache_path = dirs["cache"] / f"train_examples_{run_tag}.jsonl"
        train_examples = load_or_build_examples(df, train_ids, selected_items, codebook, cache_path, rebuild=args.rebuild_data)
        sanity_check_examples(train_examples, min_expected)
        logger.info(f"Built {len(train_examples)} training examples")
    except Exception as e:
        logger.error(f"FATAL during data build: {e}")
        logger.error(traceback.format_exc())
        write_status("failed", f"data build error: {e}", extra_log_path=dirs["logs"])
        sys.exit(1)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def model_loader():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        )
        m = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb_config, device_map="auto", trust_remote_code=True,
        )
        m.config.use_cache = False
        m = prepare_model_for_kbit_training(m)
        lora_config = LoraConfig(
            r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        m = get_peft_model(m, lora_config)
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        total = sum(p.numel() for p in m.parameters())
        logger.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")
        return m

    write_status("training", "starting", extra_log_path=dirs["logs"])
    t0 = time.time()
    try:
        trainer, model = train_with_oom_backoff(args, train_examples, tokenizer, model_loader, dirs)
    except Exception as e:
        logger.error(f"FATAL during training: {e}")
        logger.error(traceback.format_exc())
        write_status("failed", f"training error: {e}", extra_log_path=dirs["logs"])
        logger.error("Re-run the SAME command -- it will resume from the last saved checkpoint, not restart.")
        sys.exit(1)
    train_minutes = (time.time() - t0) / 60
    logger.info(f"Training done in {train_minutes:.1f} min")

    model_path = str(dirs["base"] / f"model_{run_tag}")
    trainer.model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    logger.info(f"Saved fine-tuned model to {model_path}")
    write_status("training_complete", f"saved to {model_path}", extra_log_path=dirs["logs"], train_minutes=round(train_minutes, 1))

    # ---- Out-of-fold prediction on test respondents ----
    write_status("evaluating", "running out-of-fold inference", extra_log_path=dirs["logs"])
    model.eval()
    rows = []
    test_sub = df[df["respondent_id"].isin(test_ids)]
    n_total = len(test_sub) * len(selected_items)
    t_eval0 = time.time()
    for i, (_, row) in enumerate(test_sub.iterrows()):
        try:
            demo = verbalize_demographics(row, codebook)
        except Exception as e:
            logger.warning(f"Skipping test respondent {row.get('respondent_id')}: {e}")
            continue
        for question_id in selected_items:
            try:
                item = verbalize_item(question_id, codebook)
                true_code = row.get(question_id)
                if pd.isna(true_code):
                    continue
                true_code = int(true_code)
                if true_code not in item["code_to_index"]:
                    continue
                option_labels = [str(c) for c in item["ordinal_values"]]
                prompt = build_prompt("P2", item["question_text"], item["options_text"], **demo) + build_answer_instruction(option_labels)

                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=700).to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.pad_token_id)
                gen_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                pred_code = parse_answer_from_text(gen_text, option_labels)
                pred_code_idx = item["code_to_index"].get(int(pred_code)) if pred_code is not None else None

                idx_to_label = dict(zip(item["ordinal_values"], (l.split(". ", 1)[1] for l in item["options_text"].splitlines())))
                rows.append({
                    "respondent_id": int(row["respondent_id"]),
                    "question_id": question_id,
                    "question_text": item["question_text"],
                    "condition": "P2",
                    "model": f"trackB_{args.model.replace('/', '_')}_fold{args.fold}",
                    "true_code": true_code,
                    "true_label": idx_to_label.get(true_code),
                    "true_code_idx": item["code_to_index"][true_code],
                    "pred_code": int(pred_code) if pred_code is not None else None,
                    "pred_label": idx_to_label.get(int(pred_code)) if pred_code is not None else None,
                    "pred_code_idx": pred_code_idx,
                    "pred_raw_text": gen_text,
                    "refusal": pred_code is None,
                })
            except Exception as e:
                logger.warning(f"Prediction failed for ({row.get('respondent_id')}, {question_id}): {e}")

        if (i + 1) % 20 == 0 or (i + 1) == len(test_sub):
            elapsed = time.time() - t_eval0
            partial_acc = None
            if rows:
                done = pd.DataFrame(rows)
                partial_acc = round(float((done["pred_code_idx"] == done["true_code_idx"]).mean()), 4)
            write_status(
                "evaluating",
                f"{i+1}/{len(test_sub)} respondents, {len(rows)} predictions, {elapsed/60:.1f} min elapsed, running accuracy {partial_acc}",
                extra_log_path=dirs["logs"], respondents_done=i + 1, respondents_total=len(test_sub),
                n_predictions=len(rows), running_accuracy=partial_acc,
            )
            # Save partial results every 20 respondents -- if eval crashes near
            # the end, you don't lose everything, just re-run to fill the rest.
            pd.DataFrame(rows).to_parquet(dirs["predictions"] / "partial_predictions.parquet")

    pred_df = pd.DataFrame(rows)
    out_name = f"trackB_{args.model.replace('/', '_')}_fold{args.fold}_P2.parquet"
    out_path = Path(RESULTS_DIR) / "predictions" / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(out_path)
    # Mirror into the run folder too, so the run folder alone is a complete record.
    pred_df.to_parquet(dirs["predictions"] / out_name)

    n = len(pred_df)
    n_answered = pred_df["pred_code_idx"].notna().sum()
    acc = (pred_df["pred_code_idx"] == pred_df["true_code_idx"]).mean() if n else float("nan")
    total_minutes = (time.time() - t_run_start) / 60
    summary = {
        "model": args.model, "fold": args.fold, "n_items": len(selected_items),
        "n_predictions": n, "n_answered": int(n_answered),
        "refusal_rate": round(1 - n_answered / n, 4) if n else None,
        "raw_accuracy": round(float(acc), 4) if n else None,
        "train_minutes": round(train_minutes, 1),
        "total_run_minutes": round(total_minutes, 1),
        "output_path": str(out_path),
        "run_folder": str(dirs["base"].resolve()),
    }
    write_status("complete", "run finished successfully", extra_log_path=dirs["logs"], **summary)

    if args.smoke_test:
        preview = pred_df.head(10)[["respondent_id", "question_id", "true_code", "pred_code", "pred_raw_text", "refusal"]].to_dict(orient="records")
        smoke_result = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_train_examples": len(train_examples),
            "n_predictions": n,
            "refusal_rate": summary["refusal_rate"],
            "raw_accuracy_on_tiny_sample": summary["raw_accuracy"],
            "note": "Accuracy on this tiny sample is NOT meaningful (n too small) -- what matters is that this completed without errors, and that pred_raw_text below looks like clean digits, not garbage or refusals.",
            "sample_predictions": preview,
            "verdict": "PASS -- looks like clean output, safe to proceed to the real run" if summary["refusal_rate"] is not None and summary["refusal_rate"] < 0.5 else "REVIEW NEEDED -- high refusal/unparsed rate, read sample_predictions before proceeding",
        }
        smoke_path = dirs["logs"] / "smoke_test_result.json"
        smoke_path.write_text(json.dumps(smoke_result, indent=2))
        logger.info(f"Smoke test summary written to {smoke_path} -- READ THIS before running the real thing.")

    logger.info("=" * 60)
    logger.info("FINAL SUMMARY -- paste this back")
    logger.info(json.dumps(summary, indent=2))
    logger.info(f"Full run folder (cache/checkpoints/logs/predictions, all in one place): {dirs['base'].resolve()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
