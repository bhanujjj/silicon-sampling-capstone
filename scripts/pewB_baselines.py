"""Statistical baselines on the SAME data, folds, items and demographics the
LLM sees -- the comparison a reviewer will ask for first ("does the LLM beat
a plain classifier on the same information?").

Methods: majority answer (from training respondents), multinomial logistic
regression, gradient boosting. Features = exactly the demographics the LLM
prompt contains (sex, age, education, religion, caste, marital status,
urban/rural, region, income). Runs on CPU in minutes.

    python -m scripts.pewB_baselines --n-items 8 --folds 0 1 2 3 4
Writes results/pewB_baselines.json and .md
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.config import DATA_PROCESSED, RESULTS_DIR
from src.eval.pew_metrics import bootstrap, score_rows
from src.prompts.verbalize import verbalize_item
from src.prompts.verbalize_pew import load_pew_codebook

FEATURES = ["QGEN", "QAGErec", "QEDU", "QRELSING", "QCASTE", "QMARRIEDrec", "Urban", "REGION", "QINCINDrec"]


def design(df, cols):
    """One-hot demographics (missing codes become their own category)."""
    return pd.get_dummies(df[cols].fillna(-1).astype(int).astype(str), dtype=float)


def full_proba(model, X, n_opt):
    """predict_proba padded to all n_opt columns (a class absent from the training split gets 0)."""
    p = model.predict_proba(X)
    out = np.zeros((len(X), n_opt))
    out[:, model.classes_.astype(int)] = p
    return out


def run_fold(df, fold_def, items, codebook):
    train = df[df["respondent_id"].isin(fold_def["train"])]
    test = df[df["respondent_id"].isin(fold_def["test"])]
    cols = [c for c in FEATURES if c in df.columns]
    X_all = design(pd.concat([train, test]), cols)
    Xtr, Xte = X_all.iloc[: len(train)], X_all.iloc[len(train):]
    rows = {"majority": [], "logistic": [], "gboost": []}
    for q in items:
        item = verbalize_item(q, codebook)
        c2i, n_opt = item["code_to_index"], item["n_options"]
        ytr = train[q].map(lambda v: c2i.get(int(v)) if pd.notna(v) and int(v) in c2i else np.nan)
        yte = test[q].map(lambda v: c2i.get(int(v)) if pd.notna(v) and int(v) in c2i else np.nan)
        mtr, mte = ytr.notna().to_numpy(), yte.notna().to_numpy()
        y, yt = ytr[mtr].astype(int).to_numpy(), yte[mte].astype(int).to_numpy()
        rid = test["respondent_id"].to_numpy()[mte]
        maj = np.bincount(y, minlength=n_opt) / len(y)
        models = {
            "logistic": LogisticRegression(max_iter=300, C=1.0).fit(Xtr[mtr], y),
            "gboost": HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=0).fit(Xtr[mtr], y),
        }
        probas = {"majority": np.tile(maj, (len(yt), 1))}
        for name, m in models.items():
            probas[name] = full_proba(m, Xte[mte], n_opt)
        for name, P in probas.items():
            for i in range(len(yt)):
                rows[name].append((rid[i], q, int(yt[i]), P[i]))
    return {k: pd.DataFrame(v, columns=["respondent_id", "question_id", "true_idx", "proba"]) for k, v in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-items", type=int, default=8)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-boot", type=int, default=100)
    args = ap.parse_args()

    df = pd.read_parquet(DATA_PROCESSED / "pew_india_2021.parquet")
    if "respondent_id" not in df.columns:
        df["respondent_id"] = range(len(df))
    items = json.load(open(DATA_PROCESSED / "pew_selected_items.json"))["selected_items"][: args.n_items]
    folds = json.load(open(DATA_PROCESSED / "pew_folds.json"))["folds"]
    codebook = load_pew_codebook()

    per_fold = {}
    for f in args.folds:
        print(f"fold {f} ...", flush=True)
        res = run_fold(df, folds[f], items, codebook)
        per_fold[f] = res
        for name, d in res.items():
            print(f"  {name:9s} acc={score_rows(d)['accuracy']:.3f}", flush=True)

    summary = {}
    for name in ("majority", "logistic", "gboost"):
        allf = pd.concat([per_fold[f][name] for f in args.folds], ignore_index=True)
        allf["respondent_id"] = allf["respondent_id"].astype(int)
        s = score_rows(allf)
        s["ci95"] = bootstrap(allf, n_boot=args.n_boot)
        s["per_fold_accuracy"] = {int(f): score_rows(per_fold[f][name])["accuracy"] for f in args.folds}
        s["per_item_accuracy"] = {q: score_rows(g)["accuracy"] for q, g in allf.groupby("question_id")}
        summary[name] = s

    out = Path(RESULTS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pewB_baselines.json").write_text(json.dumps({"items": items, "folds": args.folds, "results": summary}, indent=2))
    md = "| Method | Accuracy (95% CI) | MAE | TV distance | NLL |\n|---|---|---|---|---|\n"
    for name, s in summary.items():
        a = s["ci95"]["accuracy"]
        md += f"| {name} | {s['accuracy']:.3f} ({a[0]:.3f}-{a[1]:.3f}) | {s['mae']:.2f} | {s['tv']:.3f} | {s['nll']:.3f} |\n"
    (out / "pewB_baselines.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
