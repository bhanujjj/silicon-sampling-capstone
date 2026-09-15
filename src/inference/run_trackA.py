"""Phase 3, Track A: zero-shot silicon-sampling inference.

This is the one script you run once an API key exists. Everything else --
data, items, folds, prompts, verbalization, caching, metrics, the comparison
report -- is already wired. Swapping models/conditions is a flag, not a code
change.

Examples:
    # No API key needed -- validates the full pipeline end to end
    python -m src.inference.run_trackA --model mock --condition P2 --n-respondents 50

    # Real run, once you have a key
    export GOOGLE_API_KEY="..."
    python -m src.inference.run_trackA --model gemini-3.5-flash-lite --condition P2

    python -m src.inference.run_trackA --model llama-3.1-8b --condition P2
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, RESULTS_DIR
from src.eval.subgroups import SUBGROUP_AXES, assign_subgroups
from src.prompts.templates import build_prompt
from src.prompts.verbalize import load_codebook, parse_predicted_code, verbalize_demographics, verbalize_item

logger = logging.getLogger(__name__)


def get_runner(model: str, api_key: str = None, reasoning_effort: str = "low"):
    """Model name -> CachedInferenceRunner instance. Adding a model is one
    branch here; nothing else in this file changes."""
    if model == "mock":
        from src.inference.mock import MockInferenceRunner

        return MockInferenceRunner()
    if model.startswith("gemini"):
        from src.inference.gemini import GeminiInferenceRunner

        return GeminiInferenceRunner(api_key=api_key, model_name=model)
    if model == "groq" or model.startswith("groq-") or model.startswith("llama-3.3"):
        from src.inference.groq import DEFAULT_MODEL as GROQ_DEFAULT
        from src.inference.groq import GroqInferenceRunner

        groq_model = GROQ_DEFAULT if model == "groq" else model.removeprefix("groq-")
        return GroqInferenceRunner(api_key=api_key, model_name=groq_model, reasoning_effort=reasoning_effort)
    if model in ("llama-3.1-8b", "qwen2.5-7b", "gemma-3-4b"):
        from src.config import MODELS
        from src.inference.hf_local import HFLocalInferenceRunner

        return HFLocalInferenceRunner(MODELS[model]["hf_id"])
    raise ValueError(f"Unknown model: {model}")


def run_track_a(
    runner,
    df: pd.DataFrame,
    selected_items: list,
    codebook: dict,
    condition: str,
    n_respondents: int = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Run one (model, condition) pair over respondents x items.

    Returns one row per (respondent, item): demographics, the actual answer,
    the model's predicted answer + full probability vector, and correctness --
    everything the comparison report and metrics need.
    """
    if n_respondents is not None:
        df = df.sample(n=min(n_respondents, len(df)), random_state=seed)

    rows = []
    total = len(df) * len(selected_items)
    done = 0

    for _, resp_row in df.iterrows():
        demo = verbalize_demographics(resp_row, codebook)

        for question_id in selected_items:
            item = verbalize_item(question_id, codebook)
            true_code = resp_row.get(question_id)
            if pd.isna(true_code):
                continue
            true_code = int(true_code)
            if true_code not in item["code_to_index"]:
                continue

            prompt = build_prompt(condition, item["question_text"], item["options_text"], **demo)
            option_labels = [str(c) for c in item["ordinal_values"]]

            cached = runner.load_cached_result(int(resp_row["respondent_id"]), question_id, condition)
            # A cached *transient* failure (timeout/quota/network) must not be
            # treated as a final answer -- otherwise a resumed run silently
            # re-serves "no answer" forever instead of retrying. Only skip a
            # cache hit that either succeeded or failed for a durable reason
            # (e.g. content genuinely unparsable).
            transient_errors = ("504", "deadline exceeded", "timeout", "429", "quota", "503", "connection")
            was_transient_failure = (
                cached is not None
                and cached.get("predicted_answer") is None
                and cached.get("error")
                and any(marker in cached["error"].lower() for marker in transient_errors)
            )
            if cached is not None and not was_transient_failure:
                result = cached
            else:
                pred_label, probs, meta = runner.infer_single(prompt, option_labels)
                result = {
                    "respondent_id": int(resp_row["respondent_id"]),
                    "item_id": question_id,
                    "condition": condition,
                    "predicted_answer": pred_label,
                    "logprobs": probs.tolist() if probs is not None else None,
                    "refusal": meta.get("refusal", False),
                    "error": meta.get("error"),
                    "real_logprobs": meta.get("real_logprobs"),
                }
                runner.save_result(int(resp_row["respondent_id"]), question_id, condition, result)

            pred_code = parse_predicted_code(result["predicted_answer"], item)
            probs = np.array(result["logprobs"]) if result["logprobs"] else None

            rows.append(
                {
                    "respondent_id": int(resp_row["respondent_id"]),
                    "question_id": question_id,
                    "question_text": item["question_text"],
                    "condition": condition,
                    "model": runner.model_name,
                    "true_code": true_code,
                    "true_label": item["options_text"].splitlines()[item["code_to_index"][true_code]].split(". ", 1)[1],
                    "true_code_idx": item["code_to_index"][true_code],
                    "pred_code": pred_code,
                    "pred_label": (
                        item["options_text"].splitlines()[item["code_to_index"][pred_code]].split(". ", 1)[1]
                        if pred_code is not None and pred_code in item["code_to_index"]
                        else result["predicted_answer"]
                    ),
                    "pred_raw_text": result["predicted_answer"],
                    "pred_code_idx": item["code_to_index"].get(pred_code) if pred_code is not None else None,
                    "pred_probs": probs,
                    "refusal": result["refusal"],
                    "error": result["error"],
                    "real_logprobs": result.get("real_logprobs"),
                    **{f"demo_{k}": v for k, v in demo.items()},
                }
            )

            done += 1
            if done % 500 == 0:
                logger.info(f"  ...{done}/{total} (respondent, item) predictions")

    return pd.DataFrame(rows)


def attach_true_subgroups(predictions: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Merge in the subgroup axis columns (computed from raw data, not the
    verbalized demo_* text) so slicing matches the audit's definitions."""
    sg = assign_subgroups(df)[["respondent_id"] + SUBGROUP_AXES]
    return predictions.merge(sg, on="respondent_id", how="left")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="mock | gemini-3.5-flash-lite | gemini-<any model your key can access> | llama-3.1-8b | qwen2.5-7b | gemma-3-4b")
    parser.add_argument("--condition", default="P2", choices=["P0", "P1", "P2", "P3"])
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--reasoning-effort", default="low", choices=["low", "medium", "high"], help="Groq gpt-oss models only")
    parser.add_argument("--n-respondents", type=int, default=None, help="Subsample for a quick/cheap run")
    parser.add_argument("--n-items", type=int, default=None, help="Use only the first N selected items")
    args = parser.parse_args()

    df = pd.read_parquet(DATA_PROCESSED / "ind_wvs7.parquet")
    if "respondent_id" not in df.columns:
        df["respondent_id"] = range(len(df))

    selected_items = json.load(open(DATA_PROCESSED / "selected_items.json"))["selected_items"]
    if args.n_items:
        selected_items = selected_items[: args.n_items]

    codebook = load_codebook()
    runner = get_runner(args.model, api_key=args.api_key, reasoning_effort=args.reasoning_effort)

    logger.info(f"Running Track A: model={runner.model_name} condition={args.condition} "
                f"n_respondents={args.n_respondents or len(df)} n_items={len(selected_items)}")

    predictions = run_track_a(runner, df, selected_items, codebook, args.condition, n_respondents=args.n_respondents)
    predictions = attach_true_subgroups(predictions, df)

    out_dir = RESULTS_DIR / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{runner.model_name}_{args.condition}.parquet"
    predictions.to_parquet(out_path)
    logger.info(f"✓ Saved {len(predictions)} predictions to {out_path}")

    n_answered = predictions["pred_code"].notna().sum()
    accuracy = (predictions["true_code"] == predictions["pred_code"]).mean()
    logger.info(f"Quick check: {n_answered}/{len(predictions)} parsed, raw accuracy={accuracy:.3f}")

    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
