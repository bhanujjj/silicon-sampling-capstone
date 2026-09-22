"""Load a trained pewB LoRA adapter back up and answer questions WITHOUT
retraining -- this is the "give it a Q, get an answer" piece.

Why not a single .pkl file: pewB_finetune.py already saves the right thing
via `trainer.model.save_pretrained(model_path)` -- a small (tens of MB)
LoRA adapter directory (safetensors weights + config + tokenizer files).
Pickling a multi-GB HF model directly is the wrong tool here: pickle isn't
guaranteed stable across torch/transformers versions, produces a huge and
fragile file, and `pickle.load` on an untrusted file is a code-execution
risk -- none of which apply to `save_pretrained`/`from_pretrained`, which
is the standard, version-stable way every HF model is shipped and reloaded.
This module IS the convenience layer on top of that: load once, then call
.answer() as many times as you like.

Usage (after training has produced e.g. pewB_run/model_fold0/):

    python scripts/pewB_inference.py \\
        --model-path pewB_run/model_fold0 \\
        --base-model openai/gpt-oss-20b \\
        --question-id Q37a --sex Female --religion Hindu --urban-rural Rural

    # or from Python / another notebook cell:
    from scripts.pewB_inference import PewSimulator
    sim = PewSimulator.load("pewB_run/model_fold0", base_model="openai/gpt-oss-20b")
    sim.answer("Q37a", sex="Female", religion="Hindu", urban_rural="Rural")

Optional --merge writes a second, standalone directory with the LoRA
weights merged into the base model (`merge_and_unload()`) -- larger
(full model size, not just the adapter), but loadable with plain
AutoModelForCausalLM.from_pretrained() and no PEFT dependency at all,
if that's ever useful for deployment outside this repo.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompts.templates import build_prompt
from src.prompts.verbalize import verbalize_item
from src.prompts.verbalize_pew import load_pew_codebook
from src.inference.prompting import build_answer_instruction, parse_answer_from_text

logger = logging.getLogger(__name__)


class PewSimulator:
    """Thin wrapper: load once, call .answer() repeatedly. Holds live model
    objects in memory -- not itself picklable/serializable, and shouldn't be;
    re-run .load() with the same model_path to get a fresh instance."""

    def __init__(self, model, tokenizer, codebook: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.codebook = codebook

    @classmethod
    def load(cls, model_path: str, base_model: str) -> "PewSimulator":
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info(f"Loading base model {base_model} in 4-bit...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=bnb_config, device_map="auto", trust_remote_code=True,
        )
        logger.info(f"Applying LoRA adapter from {model_path}...")
        model = PeftModel.from_pretrained(base, model_path)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        codebook = load_pew_codebook()
        return cls(model, tokenizer, codebook)

    def answer(self, question_id: str, **demographics) -> dict:
        """demographics: any of the keys verbalize_pew.verbalize_demographics
        returns (sex, religion, urban_rural, marital_status, education,
        caste, region, income_decile, age -- whichever are verified;
        pass only what you know, unknown ones are simply omitted from the
        prompt)."""
        import torch

        item = verbalize_item(question_id, self.codebook)
        option_labels = [str(c) for c in item["ordinal_values"]]
        prompt = build_prompt("P2", item["question_text"], item["options_text"], **demographics) + build_answer_instruction(option_labels)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=700).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
        gen_text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred_code = parse_answer_from_text(gen_text, option_labels)

        idx_to_label = dict(zip(item["ordinal_values"], (l.split(". ", 1)[1] for l in item["options_text"].splitlines())))
        return {
            "question_id": question_id,
            "question_text": item["question_text"],
            "prompt": prompt,
            "raw_output": gen_text,
            "pred_code": int(pred_code) if pred_code is not None else None,
            "pred_label": idx_to_label.get(int(pred_code)) if pred_code is not None else None,
            "refusal": pred_code is None,
        }

    def merge_and_save(self, output_dir: str):
        """Optional: write a standalone merged model (base + adapter combined),
        loadable with plain AutoModelForCausalLM.from_pretrained(), no PEFT
        needed. Larger on disk than the adapter alone (full model size)."""
        merged = self.model.merge_and_unload()
        merged.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Merged standalone model saved to {output_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="Path to the saved LoRA adapter dir, e.g. pewB_run/model_fold0")
    ap.add_argument("--base-model", default="openai/gpt-oss-20b")
    ap.add_argument("--question-id", required=True)
    ap.add_argument("--sex")
    ap.add_argument("--religion")
    ap.add_argument("--urban-rural", dest="urban_rural")
    ap.add_argument("--marital-status", dest="marital_status")
    ap.add_argument("--education")
    ap.add_argument("--caste")
    ap.add_argument("--merge-to", help="If given, also write a standalone merged model to this directory")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    sim = PewSimulator.load(args.model_path, args.base_model)
    demo = {
        k: v for k, v in {
            "sex": args.sex, "religion": args.religion, "urban_rural": args.urban_rural,
            "marital_status": args.marital_status, "education": args.education, "caste": args.caste,
        }.items() if v is not None
    }
    result = sim.answer(args.question_id, **demo)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.merge_to:
        sim.merge_and_save(args.merge_to)


if __name__ == "__main__":
    main()
