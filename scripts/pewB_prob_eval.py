"""Probability-based evaluation of a trained pewB adapter.

Instead of generating one answer per respondent (which collapses toward the
most common option), read the model's probability for EACH option at the
answer position, exactly where training put the answer (prompt + "\\n\\n" +
digit). Those probabilities are scored with the same code as the baselines
(src/eval/pew_metrics.py): accuracy of the top option, MAE, NLL, and TV
distance between the mean predicted distribution and the true one.

    python -m scripts.pewB_prob_eval --model-path pewB_run_v2/model_fold0 --fold 0 --n-items 8
    # quick check first: add --limit-respondents 200

Writes results/pewB_prob_eval_fold<N>.json/.parquet
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, RESULTS_DIR
from src.eval.pew_metrics import bootstrap, score_rows
from src.inference.prompting import build_answer_instruction
from src.prompts.templates import build_prompt
from src.prompts.verbalize import verbalize_item
from src.prompts.verbalize_pew import load_pew_codebook, verbalize_demographics

logger = logging.getLogger(__name__)


def option_probs(model, tokenizer, prompts, digit_token_ids, batch_size=16, max_length=700):
    """Softmax over ONLY the option-digit tokens at the position after the
    last prompt token. Left padding so the last position is the real end."""
    import torch

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    out = []
    for i in range(0, len(prompts), batch_size):
        enc = tokenizer(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length).to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :]
        out.append(torch.softmax(logits[:, digit_token_ids].float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def digit_ids(tokenizer, n_options_values):
    ids = []
    for v in n_options_values:
        toks = tokenizer.encode(str(v), add_special_tokens=False)
        if len(toks) != 1:
            raise ValueError(f"Option '{v}' is not a single token ({toks}); this scorer needs single-token answers.")
        ids.append(toks[0])
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-items", type=int, default=8)
    ap.add_argument("--limit-respondents", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=100)
    ap.add_argument("--zero-shot", action="store_true", help="Score the BASE model with no adapter (the 'untuned LLM' row of the paper table)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    df = pd.read_parquet(DATA_PROCESSED / "pew_india_2021.parquet")
    if "respondent_id" not in df.columns:
        df["respondent_id"] = range(len(df))
    items = json.load(open(DATA_PROCESSED / "pew_selected_items.json"))["selected_items"][: args.n_items]
    test_ids = json.load(open(DATA_PROCESSED / "pew_folds.json"))["folds"][args.fold]["test"]
    if args.limit_respondents:
        test_ids = test_ids[: args.limit_respondents]
    test = df[df["respondent_id"].isin(test_ids)]
    codebook = load_pew_codebook()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                             bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=bnb,
                                                 device_map="auto", trust_remote_code=True)
    if not args.zero_shot:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.model_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    rows, t0 = [], time.time()
    for q in items:
        item = verbalize_item(q, codebook)
        ids = digit_ids(tokenizer, item["ordinal_values"])
        labels = [str(c) for c in item["ordinal_values"]]
        prompts, meta = [], []
        for _, r in test.iterrows():
            v = r.get(q)
            if pd.isna(v) or int(v) not in item["code_to_index"]:
                continue
            demo = verbalize_demographics(r, codebook)
            p = build_prompt("P2", item["question_text"], item["options_text"], **demo) + build_answer_instruction(labels)
            prompts.append(p + "\n\n")   # same layout as training: prompt + "\n\n" + answer
            meta.append((int(r["respondent_id"]), item["code_to_index"][int(v)]))
        P = option_probs(model, tokenizer, prompts, ids, args.batch_size)
        for (rid, ti), pr in zip(meta, P):
            rows.append((rid, q, ti, pr))
        logger.info(f"{q}: {len(meta)} rows, {(time.time()-t0)/60:.1f} min elapsed")

    res = pd.DataFrame(rows, columns=["respondent_id", "question_id", "true_idx", "proba"])
    s = score_rows(res)
    s["ci95"] = bootstrap(res, n_boot=args.n_boot)
    s["per_item_accuracy"] = {q: score_rows(g)["accuracy"] for q, g in res.groupby("question_id")}
    tag = "zeroshot" if args.zero_shot else "finetuned"
    out = Path(RESULTS_DIR); out.mkdir(exist_ok=True)
    (out / f"pewB_prob_eval_{tag}_fold{args.fold}.json").write_text(json.dumps({"items": items, "fold": args.fold, "results": s}, indent=2))
    res.assign(proba=res["proba"].map(list)).to_parquet(out / f"pewB_prob_eval_{tag}_fold{args.fold}.parquet")
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
