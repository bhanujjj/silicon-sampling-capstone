"""Render a self-contained HTML "silicon sampling" comparison report:
headline fidelity metrics against every baseline, subgroup fidelity gaps,
and a gallery of real respondent-vs-model transcripts.

No CDN dependencies, no JS framework -- one static file, works offline,
renders identically whether opened locally or published as an Artifact.

Usage:
    python -m src.report.comparison_report \\
        --predictions results/predictions/gemini-3.5-flash-lite_P2.parquet \\
        --output results/reports/gemini-3.5-flash-lite_P2.html
"""

import argparse
import html
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR
from src.report.evaluate_run import overall_metrics, per_item_metrics, versus_baselines
from src.eval.subgroups import SUBGROUP_AXES, fidelity_gap_report, metrics_by_subgroup_axis

logger = logging.getLogger(__name__)

AXIS_TITLES = {
    "sg_urban_rural": "Urban / Rural",
    "sg_income": "Income tercile",
    "sg_education": "Education band",
    "sg_sex": "Sex",
    "sg_age_band": "Age band",
    "sg_region": "WVS region zone",
}

BASELINE_TITLES = {
    "uniform": "Uniform random",
    "marginal": "National marginal",
    "cell_lookup": "Demographic-cell lookup",
    "logistic": "Logistic regression",
    "gbm": "Gradient boosting",
}


def esc(x) -> str:
    return html.escape(str(x))


def bar(value: float, max_value: float, color_var: str = "var(--accent)", height: int = 10) -> str:
    pct = max(0.0, min(100.0, (value / max_value) * 100)) if max_value else 0
    return (
        f'<div class="bar-track" style="height:{height}px">'
        f'<div class="bar-fill" style="width:{pct:.1f}%;background:{color_var}"></div>'
        f"</div>"
    )


def render_demo_chips(row: pd.Series) -> str:
    fields = [
        ("age", "demo_age"), ("sex", "demo_sex"), ("marital_status", "demo_marital_status"),
        ("education", "demo_education"), ("employment", "demo_employment"),
        ("income_decile", "demo_income_decile"), ("religion", "demo_religion"),
        ("urban_rural", "demo_urban_rural"), ("region", "demo_region"),
    ]
    chips = []
    for _, col in fields:
        val = row.get(col)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        chips.append(f'<span class="chip">{esc(val)}</span>')
    return "".join(chips)


def render_probability_bars(item_row: pd.Series, item_meta: dict) -> str:
    """Mini distribution over every answer option, true option marked."""
    probs = item_row.get("pred_probs")
    if probs is None or (isinstance(probs, float) and pd.isna(probs)):
        return '<p class="muted small">No probability distribution recorded.</p>'
    probs = np.asarray(probs, dtype=float)
    labels = item_meta["options_text"].splitlines()
    true_idx = item_row["true_code_idx"]
    pred_idx = item_row["pred_code_idx"]

    rows = []
    for idx, label in enumerate(labels):
        text = label.split(". ", 1)[1] if ". " in label else label
        p = probs[idx] if idx < len(probs) else 0.0
        marker = ""
        if idx == true_idx:
            marker += '<span class="tag tag-true">actual</span>'
        if idx == pred_idx:
            marker += '<span class="tag tag-pred">model</span>'
        rows.append(
            f'<div class="prob-row">'
            f'<span class="prob-label">{esc(text)}</span>'
            f'{bar(p, max(probs.max(), 1e-9))}'
            f'<span class="prob-pct">{p*100:4.1f}%</span>'
            f'<span class="prob-tags">{marker}</span>'
            f"</div>"
        )
    return f'<div class="prob-dist">{"".join(rows)}</div>'


def render_example_card(row: pd.Series, item_meta_lookup: dict) -> str:
    correct = row["true_code_idx"] == row["pred_code_idx"]
    stripe_class = "stripe-correct" if correct else "stripe-wrong"
    verdict = "MATCH" if correct else "MISS"
    item_meta = item_meta_lookup.get(row["question_id"])

    return f"""
    <article class="card example-card {stripe_class}">
      <div class="example-head">
        <span class="verdict {'verdict-correct' if correct else 'verdict-wrong'}">{verdict}</span>
        <span class="qid">{esc(row['question_id'])}</span>
        <h4>{esc(row['question_text'])}</h4>
      </div>
      <div class="chip-row">{render_demo_chips(row)}</div>
      <div class="answer-compare">
        <div class="answer-col">
          <div class="answer-label">Actual respondent answered</div>
          <div class="answer-value true-value">{esc(row['true_label'])}</div>
        </div>
        <div class="answer-arrow">vs</div>
        <div class="answer-col">
          <div class="answer-label">Model predicted</div>
          <div class="answer-value pred-value">{esc(row['pred_label'] if row['pred_label'] is not None else '(no answer parsed)')}</div>
        </div>
      </div>
      {render_probability_bars(row, item_meta) if item_meta else ''}
    </article>
    """


def render_stat_tile(label: str, value: str, sub: str = "", tone: str = "") -> str:
    return f"""
    <div class="stat-tile {tone}">
      <div class="stat-label">{esc(label)}</div>
      <div class="stat-value">{value}</div>
      {f'<div class="stat-sub">{esc(sub)}</div>' if sub else ''}
    </div>
    """


def render_baseline_comparison(comparison: pd.DataFrame) -> str:
    if comparison.empty or "llm" not in comparison.columns:
        return '<p class="muted">No baseline comparison available -- run src.eval.run_baselines first.</p>'

    cols = [c for c in ["uniform", "marginal", "cell_lookup", "logistic", "gbm"] if c in comparison.columns]
    means = {"llm": comparison["llm"].mean(), **{c: comparison[c].mean() for c in cols}}
    max_val = max(means.values()) * 1.15 if means else 1

    order = ["llm"] + cols
    rows = []
    for key in order:
        title = "This LLM (Track A)" if key == "llm" else BASELINE_TITLES.get(key, key)
        color = "var(--accent)" if key == "llm" else "var(--ink-dim)"
        beats_frac = ""
        if key != "llm" and key in comparison.columns:
            n_beat = (comparison["llm"] > comparison[key]).sum()
            beats_frac = f'<span class="muted small">beats it on {n_beat}/{len(comparison)} items</span>'
        rows.append(
            f'<div class="baseline-row {"baseline-row-llm" if key == "llm" else ""}">'
            f'<span class="baseline-name">{esc(title)}</span>'
            f'{bar(means[key], max_val, color_var=color, height=14)}'
            f'<span class="baseline-pct">{means[key]*100:4.1f}%</span>'
            f"{beats_frac}"
            f"</div>"
        )
    return f'<div class="baseline-chart">{"".join(rows)}</div>'


def render_subgroup_table(axis: str, table: pd.DataFrame) -> str:
    if table.empty:
        return ""
    rows = []
    for _, r in table.iterrows():
        flag = ' <span class="tag tag-warn">n&lt;30</span>' if r["underpowered"] else ""
        rows.append(
            f"<tr><td>{esc(r['category'])}{flag}</td>"
            f"<td class='num'>{r['n']}</td>"
            f"<td class='num'>{r['accuracy']*100:.1f}%</td>"
            f"<td class='num muted small'>[{r['accuracy_ci_low']*100:.1f}, {r['accuracy_ci_high']*100:.1f}]</td>"
            f"<td class='num'>{r['mae']:.2f}</td></tr>"
        )
    return f"""
    <div class="subgroup-block">
      <h4>{esc(AXIS_TITLES.get(axis, axis))}</h4>
      <table>
        <thead><tr><th>Category</th><th class="num">n</th><th class="num">Accuracy</th><th class="num">95% CI</th><th class="num">MAE</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
    </div>
    """


def build_report(predictions: pd.DataFrame, model_name: str, condition: str) -> str:
    from src.prompts.verbalize import load_codebook, verbalize_item

    codebook = load_codebook()
    item_meta_lookup = {q: verbalize_item(q, codebook) for q in predictions["question_id"].unique()}

    overall = overall_metrics(predictions)
    answered = predictions[predictions["pred_code_idx"].notna()].copy()

    comparison = versus_baselines(predictions, RESULTS_DIR / "baselines_summary.csv")
    gap_report = fidelity_gap_report(answered)

    subgroup_html = ""
    for axis in SUBGROUP_AXES:
        if axis in answered.columns:
            table = metrics_by_subgroup_axis(answered, axis)
            subgroup_html += render_subgroup_table(axis, table)

    # Example gallery: a mix of correct and incorrect, spread across items
    correct_examples = answered[answered["true_code_idx"] == answered["pred_code_idx"]]
    wrong_examples = answered[answered["true_code_idx"] != answered["pred_code_idx"]]
    n_each = 6
    examples = pd.concat(
        [
            correct_examples.sample(n=min(n_each, len(correct_examples)), random_state=7) if len(correct_examples) else correct_examples,
            wrong_examples.sample(n=min(n_each, len(wrong_examples)), random_state=7) if len(wrong_examples) else wrong_examples,
        ]
    ).sample(frac=1, random_state=7).reset_index(drop=True)

    example_cards = "".join(render_example_card(row, item_meta_lookup) for _, row in examples.iterrows())

    gap_rows = "".join(
        f"<tr><td>{esc(AXIS_TITLES.get(r['axis'], r['axis']))}</td>"
        f"<td>{esc(r['best_category'])} ({r['best_accuracy']*100:.1f}%)</td>"
        f"<td>{esc(r['worst_category'])} ({r['worst_accuracy']*100:.1f}%)</td>"
        f"<td class='num strong'>{r['fidelity_gap']*100:.1f} pts</td></tr>"
        for _, r in gap_report.iterrows()
    )

    n_beats_marginal = int((comparison["llm"] > comparison["marginal"]).sum()) if "marginal" in comparison.columns else None
    n_items_compared = len(comparison)
    top_line = (
        f"Beat the national-marginal baseline on {n_beats_marginal}/{n_items_compared} items"
        if n_beats_marginal is not None
        else f"Evaluated on {n_items_compared} items"
    )
    if overall.get("real_logprob_rate") is not None and overall["real_logprob_rate"] < 0.99:
        top_line += (
            f" · probabilities are API-returned logprobs on {overall['real_logprob_rate']*100:.0f}% of predictions; "
            f"the rest use a fallback confidence on the parsed text answer (see methodology note)"
        )

    return PAGE_TEMPLATE.format(
        title=f"Silicon Sampling Fidelity — {esc(model_name)} / {esc(condition)}",
        model_name=esc(model_name),
        condition=esc(condition),
        generated_note=esc(top_line),
        stat_tiles="".join(
            [
                render_stat_tile(
                    "Accuracy (exact match)",
                    f"{overall['accuracy']*100:.1f}%",
                    f"95% CI [{overall['accuracy_ci_low']*100:.1f}, {overall['accuracy_ci_high']*100:.1f}]",
                    tone="tone-accent",
                ),
                render_stat_tile("Mean absolute error", f"{overall['mae']:.2f}", "ordinal-scale distance", ),
                render_stat_tile("Refusal / unparsed rate", f"{overall['refusal_rate']*100:.1f}%", ""),
                render_stat_tile("Predictions evaluated", f"{overall['n_answered']:,}", f"of {overall['n_predictions']:,} attempted"),
            ]
        ),
        baseline_chart=render_baseline_comparison(comparison),
        gap_rows=gap_rows or "<tr><td colspan='4' class='muted'>No subgroup had two or more well-powered categories.</td></tr>",
        subgroup_tables=subgroup_html,
        example_cards=example_cards or '<p class="muted">No examples available.</p>',
    )


PAGE_TEMPLATE = """<title>{title}</title>
<style>
:root {{
  --bg: #EEF1EF;
  --surface: #FFFFFF;
  --surface-2: #E3E8E5;
  --ink: #1A2321;
  --ink-dim: #5C6A66;
  --border: #D2D9D5;
  --accent: #17615B;
  --accent-soft: #E3EEEC;
  --gold: #A6762E;
  --good: #3E7D57;
  --good-soft: #E7F1EA;
  --bad: #B24A3F;
  --bad-soft: #F7E9E6;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #121715;
    --surface: #1A211E;
    --surface-2: #202824;
    --ink: #E7ECE9;
    --ink-dim: #94A19B;
    --border: #2C3733;
    --accent: #4FB8AC;
    --accent-soft: #1D302C;
    --gold: #D6A85C;
    --good: #6FBE8C;
    --good-soft: #1C2F23;
    --bad: #E08476;
    --bad-soft: #33211D;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #121715;
  --surface: #1A211E;
  --surface-2: #202824;
  --ink: #E7ECE9;
  --ink-dim: #94A19B;
  --border: #2C3733;
  --accent: #4FB8AC;
  --accent-soft: #1D302C;
  --gold: #D6A85C;
  --good: #6FBE8C;
  --good-soft: #1C2F23;
  --bad: #E08476;
  --bad-soft: #33211D;
}}

* {{ box-sizing: border-box; }}
body {{
  background: var(--bg);
  color: var(--ink);
  font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  font-size: 15px;
  line-height: 1.5;
  margin: 0;
  padding: 0 0 4rem;
}}
h1, h2, h3, h4 {{
  font-family: Georgia, "Iowan Old Style", "Times New Roman", serif;
  text-wrap: balance;
  margin: 0 0 0.4em;
  font-weight: 600;
}}
.wrap {{ max-width: 980px; margin: 0 auto; padding: 0 1.5rem; }}
.masthead {{
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 2.2rem 0 1.6rem;
  margin-bottom: 2rem;
}}
.eyebrow {{
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 0.72rem;
  color: var(--accent);
  font-weight: 700;
  margin-bottom: 0.5rem;
}}
.masthead h1 {{ font-size: 1.9rem; }}
.masthead .sub {{ color: var(--ink-dim); font-size: 0.95rem; margin-top: 0.3rem; }}

.stat-grid {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 1px;
  background: var(--border);
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
  margin-bottom: 2.2rem;
}}
@media (max-width: 700px) {{
  .stat-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
}}
.stat-tile {{ background: var(--surface); padding: 1.1rem 1.2rem; }}
.stat-tile.tone-accent {{ background: var(--accent-soft); }}
.stat-label {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--ink-dim); margin-bottom: 0.35rem; }}
.stat-value {{ font-family: ui-monospace, "SF Mono", Menlo, monospace; font-variant-numeric: tabular-nums; font-size: 1.7rem; font-weight: 600; }}
.stat-sub {{ font-size: 0.78rem; color: var(--ink-dim); margin-top: 0.2rem; }}

section {{ margin-bottom: 2.6rem; }}
section > h2 {{ font-size: 1.25rem; margin-bottom: 0.9rem; }}
section > .lede {{ color: var(--ink-dim); font-size: 0.9rem; margin: -0.5rem 0 1rem; max-width: 62ch; }}

.card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1.1rem 1.3rem;
}}

.baseline-chart {{ display: flex; flex-direction: column; gap: 0.6rem; }}
.baseline-row {{ display: grid; grid-template-columns: 180px 1fr 56px auto; align-items: center; gap: 0.7rem; }}
.baseline-row-llm .baseline-name {{ font-weight: 700; color: var(--accent); }}
.baseline-name {{ font-size: 0.85rem; }}
.baseline-pct {{ font-family: ui-monospace, monospace; font-variant-numeric: tabular-nums; font-size: 0.85rem; text-align: right; }}
.bar-track {{ background: var(--surface-2); border-radius: 5px; overflow: hidden; width: 100%; }}
.bar-fill {{ height: 100%; border-radius: 5px; }}

table {{ width: 100%; border-collapse: collapse; font-size: 0.87rem; }}
thead th {{ text-align: left; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--ink-dim); border-bottom: 1px solid var(--border); padding: 0.4rem 0.5rem; }}
td {{ padding: 0.45rem 0.5rem; border-bottom: 1px solid var(--border); }}
td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
td.strong {{ font-weight: 700; color: var(--accent); }}

.subgroup-block {{ margin-bottom: 1.4rem; }}
.subgroup-block h4 {{ font-size: 0.95rem; margin-bottom: 0.4rem; }}

.gap-table-wrap {{ overflow-x: auto; }}

.example-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 1rem; }}
.example-card {{ border-left-width: 4px; border-left-style: solid; }}
.stripe-correct {{ border-left-color: var(--good); }}
.stripe-wrong {{ border-left-color: var(--bad); }}
.example-head {{ display: flex; align-items: baseline; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.5rem; }}
.example-head h4 {{ font-size: 0.98rem; margin: 0; flex: 1 1 auto; }}
.qid {{ font-family: ui-monospace, monospace; font-size: 0.75rem; color: var(--ink-dim); }}
.verdict {{ font-size: 0.68rem; font-weight: 800; letter-spacing: 0.05em; padding: 0.15rem 0.5rem; border-radius: 999px; }}
.verdict-correct {{ background: var(--good-soft); color: var(--good); }}
.verdict-wrong {{ background: var(--bad-soft); color: var(--bad); }}

.chip-row {{ display: flex; flex-wrap: wrap; gap: 0.35rem; margin-bottom: 0.8rem; }}
.chip {{ background: var(--surface-2); border-radius: 999px; padding: 0.15rem 0.6rem; font-size: 0.74rem; color: var(--ink-dim); }}

.answer-compare {{ display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 0.6rem; margin-bottom: 0.9rem; }}
.answer-label {{ font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--ink-dim); margin-bottom: 0.15rem; }}
.answer-value {{ font-weight: 700; font-size: 0.95rem; }}
.true-value {{ color: var(--good); }}
.pred-value {{ color: var(--gold); }}
.answer-arrow {{ color: var(--ink-dim); font-size: 0.75rem; }}

.prob-dist {{ display: flex; flex-direction: column; gap: 0.3rem; }}
.prob-row {{ display: grid; grid-template-columns: 1fr 90px 42px auto; align-items: center; gap: 0.5rem; font-size: 0.76rem; }}
.prob-label {{ color: var(--ink-dim); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.prob-pct {{ font-family: ui-monospace, monospace; font-variant-numeric: tabular-nums; text-align: right; }}
.tag {{ font-size: 0.62rem; font-weight: 700; padding: 0.05rem 0.35rem; border-radius: 4px; margin-left: 0.2rem; }}
.tag-true {{ background: var(--good-soft); color: var(--good); }}
.tag-pred {{ background: var(--accent-soft); color: var(--accent); }}
.tag-warn {{ background: var(--bad-soft); color: var(--bad); }}

.muted {{ color: var(--ink-dim); }}
.small {{ font-size: 0.78rem; }}
footer {{ color: var(--ink-dim); font-size: 0.78rem; text-align: center; margin-top: 3rem; }}
</style>

<div class="masthead">
  <div class="wrap">
    <div class="eyebrow">Silicon Sampling Fidelity Report · India, WVS-7</div>
    <h1>{model_name} &nbsp;&middot;&nbsp; condition {condition}</h1>
    <div class="sub">{generated_note}</div>
  </div>
</div>

<div class="wrap">
  <div class="stat-grid">{stat_tiles}</div>

  <section>
    <h2>Does the model beat simpler baselines?</h2>
    <p class="lede">Per-item accuracy, averaged across the battery, against every Phase 2 baseline evaluated out-of-fold on the same items. If the model doesn't clear the demographic-cell lookup, conditioning an LLM on demographics adds nothing over a lookup table.</p>
    <div class="card">{baseline_chart}</div>
  </section>

  <section>
    <h2>Fidelity gap by subgroup</h2>
    <p class="lede">Gap = best-performing category's accuracy minus worst-performing, restricted to categories with n&ge;30. This is the number the whole study is testing: does simulation fidelity hold evenly across urban/rural, income, education, and region -- or does it quietly degrade for some groups?</p>
    <div class="card gap-table-wrap">
      <table>
        <thead><tr><th>Axis</th><th>Best category</th><th>Worst category</th><th class="num">Gap</th></tr></thead>
        <tbody>{gap_rows}</tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Accuracy by subgroup category</h2>
    <p class="lede">Full breakdown behind the gap numbers above, each with its own bootstrap confidence interval and sample size.</p>
    {subgroup_tables}
  </section>

  <section>
    <h2>Respondent-by-respondent: what the model actually said</h2>
    <p class="lede">A mix of matches and misses, sampled at random from real predictions -- the model's exact answer against what that real person actually told WVS interviewers, with the model's full probability distribution over every option.</p>
    <div class="example-grid">{example_cards}</div>
  </section>

  <footer>Silicon Sampling for Indian Public Opinion &middot; World Values Survey Wave 7, India (N=1,692) &middot; generated report</footer>
</div>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", default=None, type=Path)
    args = parser.parse_args()

    predictions = pd.read_parquet(args.predictions)
    model_name = predictions["model"].iloc[0]
    condition = predictions["condition"].iloc[0]

    html_content = build_report(predictions, model_name, condition)

    output_path = args.output or (RESULTS_DIR / "reports" / f"{args.predictions.stem}.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")
    logger.info(f"✓ Report written to {output_path}")
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
