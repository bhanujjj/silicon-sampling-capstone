"""Group-level (distribution) comparison -- no GPU needed.

For every (subgroup, question) cell -- e.g. "Muslim respondents, ABORT" --
compare the distribution of predicted answers with the distribution of true
answers (total variation distance, 0 = same spread, 1 = completely
different), and the error in the group's MEAN answer (ordinal steps).
Methods, all on the SAME held-out respondents and questions:
  llm_hard      the fine-tuned LLM's predictions parquet (one answer each)
  gboost_hard   gradient boosting, argmax answer (apples-to-apples with the LLM)
  gboost_soft   gradient boosting, summing predicted probabilities per group
  majority      always the most common answer
Group axes: sex, religion, urban/rural, region, caste. Cells with fewer than
--min-cell respondents are skipped. 95% CIs resample respondents.

    python -m scripts.pewB_group_eval --predictions <pewB_..._fold0_P2.parquet> --fold 0
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from scripts.pewB_baselines import run_fold
from src.config import DATA_PROCESSED, RESULTS_DIR
from src.prompts.verbalize import verbalize_item
from src.prompts.verbalize_pew import load_pew_codebook

AXES = {"sex": "QGEN", "religion": "QRELSING", "urban_rural": "Urban", "region": "REGION", "caste": "QCASTE"}


def cell_metrics(d, n_opt_by_item, min_cell):
    """d columns: respondent_id, question_id, true_idx, group, plus 'hard' (int) and optional 'proba'.
    Returns per-cell TV (hard), TV (soft, if proba), mean-answer error (hard)."""
    out = []
    for (grp, q), g in d.groupby(["group", "question_id"]):
        if g["respondent_id"].nunique() < min_cell:
            continue
        k = n_opt_by_item[q]
        t = np.bincount(g["true_idx"], minlength=k) / len(g)
        h = np.bincount(g["hard"], minlength=k) / len(g)
        row = {"group": grp, "question_id": q, "n": len(g),
               "tv_hard": 0.5 * np.abs(t - h).sum(),
               "mean_err": abs((np.arange(k) * h).sum() - (np.arange(k) * t).sum())}
        if "proba" in g:
            s = np.stack(g["proba"].to_numpy()).mean(axis=0)
            row["tv_soft"] = 0.5 * np.abs(t - s).sum()
        out.append(row)
    return pd.DataFrame(out)


def summarize(cells):
    w = cells["n"]
    r = {"tv_hard": float(np.average(cells["tv_hard"], weights=w)),
         "mean_err": float(np.average(cells["mean_err"], weights=w)), "n_cells": int(len(cells))}
    if "tv_soft" in cells:
        r["tv_soft"] = float(np.average(cells["tv_soft"], weights=w))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--min-cell", type=int, default=50)
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args()

    df = pd.read_parquet(DATA_PROCESSED / "pew_india_2021.parquet")
    if "respondent_id" not in df.columns:
        df["respondent_id"] = range(len(df))
    fold_def = json.load(open(DATA_PROCESSED / "pew_folds.json"))["folds"][args.fold]
    codebook = load_pew_codebook()

    llm = pd.read_parquet(args.predictions)
    llm = llm[~llm["refusal"] & llm["pred_code_idx"].notna()]
    items = list(dict.fromkeys(llm["question_id"]))
    n_opt = {q: verbalize_item(q, codebook)["n_options"] for q in items}

    base = run_fold(df, fold_def, items, codebook)
    key = ["respondent_id", "question_id"]
    m = llm[key + ["true_code_idx", "pred_code_idx"]].rename(columns={"true_code_idx": "true_idx", "pred_code_idx": "llm"})
    m["true_idx"] = m["true_idx"].astype(int); m["llm"] = m["llm"].astype(int)
    for name, b in base.items():
        b = b.copy(); b["respondent_id"] = b["respondent_id"].astype(int)
        b[name + "_hard"] = b["proba"].map(lambda p: int(np.argmax(p)))
        b = b.rename(columns={"proba": name + "_proba"})[key + [name + "_hard", name + "_proba"]]
        m = m.merge(b, on=key, how="inner")
    demo = df.set_index("respondent_id")
    methods = {
        "llm_hard": ("llm", None), "gboost_hard": ("gboost_hard", None),
        "gboost_soft": ("gboost_hard", "gboost_proba"), "majority": ("majority_hard", None),
    }
    rng = np.random.default_rng(0)
    results = {}
    for axis, col in AXES.items():
        if col not in demo.columns:
            continue
        m["group"] = m["respondent_id"].map(demo[col])
        mm = m[m["group"].notna()]
        for mname, (hard_col, proba_col) in methods.items():
            d = mm[["respondent_id", "question_id", "true_idx", "group"]].copy()
            d["hard"] = mm[hard_col]
            if proba_col:
                d["proba"] = mm[proba_col]
            point = summarize(cell_metrics(d, n_opt, args.min_cell))
            ids = d["respondent_id"].unique()
            by_r = {r: g for r, g in d.groupby("respondent_id")}
            boots = []
            for _ in range(args.n_boot):
                pick = rng.choice(ids, size=len(ids), replace=True)
                boots.append(summarize(cell_metrics(pd.concat([by_r[r] for r in pick], ignore_index=True), n_opt, args.min_cell)))
            key_metric = "tv_soft" if proba_col else "tv_hard"
            point["headline_tv"] = point[key_metric]
            v = np.array([b[key_metric] for b in boots])
            point["headline_tv_ci95"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            results.setdefault(axis, {})[mname] = point
        print(axis, {k: round(v["headline_tv"], 3) for k, v in results[axis].items()}, flush=True)

    out = Path(RESULTS_DIR); out.mkdir(exist_ok=True)
    (out / f"pewB_group_eval_fold{args.fold}.json").write_text(json.dumps(results, indent=2))
    md = "| Group axis | Method | TV distance (95% CI) | Mean-answer error |\n|---|---|---|---|\n"
    for axis, r in results.items():
        for mname, v in r.items():
            ci = v["headline_tv_ci95"]
            md += f"| {axis} | {mname} | {v['headline_tv']:.3f} ({ci[0]:.3f}-{ci[1]:.3f}) | {v['mean_err']:.3f} |\n"
    (out / f"pewB_group_eval_fold{args.fold}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
