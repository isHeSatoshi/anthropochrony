"""Write notebooks/reproduce_results.ipynb. The notebook recomputes the headline table from released rows on CPU."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

CELLS = [
    (
        "md",
        "# Reproduce the Anthropochrony results on CPU\n\n"
        "This notebook rebuilds the main table (T2) and the hero figure from the released "
        "`analysis_rows.jsonl`, using the same metric code as the pipeline. It needs no GPU, "
        "no model download and no credentials. Set `ANTHROPOCHRONY_DATA` to a folder holding "
        "`analysis_rows.jsonl`, `metrics.json`, `hypotheses.json` and `simulations.json`, or "
        "let it download them from the public dataset repo in `ANTHROPOCHRONY_DATASET`.",
    ),
    (
        "code",
        "import json, os\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from anthropochrony.eval.metrics import summary, task_records\n"
        "from anthropochrony.eval.stats import stratified_bootstrap\n\n"
        "def locate():\n"
        "    for candidate in (os.environ.get('ANTHROPOCHRONY_DATA'), 'results', '../results'):\n"
        "        if candidate and (Path(candidate) / 'analysis_rows.jsonl').exists():\n"
        "            return Path(candidate)\n"
        "    from huggingface_hub import snapshot_download\n"
        "    return Path(snapshot_download(os.environ['ANTHROPOCHRONY_DATASET'], repo_type='dataset',\n"
        "                                  allow_patterns=['results/*'])) / 'results'\n\n"
        "DATA = locate()\n"
        "rows = [json.loads(line) for line in (DATA / 'analysis_rows.jsonl').read_text().splitlines()]\n"
        "print(len(rows), 'estimate rows from', DATA)",
    ),
    (
        "md",
        "## Main table: `test`, reference speed card, self frame\n\n"
        "Errors are `|log10(estimate / measured)|`, averaged over question templates within a "
        "task. Fine-tuned models are scored against their own re-measured runs, the base model "
        "and baselines against the base runs. A parse failure counts as infinite error.",
    ),
    (
        "code",
        "def select(split='test', card='REF', frame='self'):\n"
        "    groups = {}\n"
        "    for row in rows:\n"
        "        if row['split'] == split and row['card_id'] == card and row.get('frame', 'self') == frame:\n"
        "            groups.setdefault((row['estimator'], row.get('target_source')), []).append(row)\n"
        "    return groups\n\n"
        "def table(split='test', resamples=2000):\n"
        "    out = []\n"
        "    for (estimator, source), group in sorted(select(split).items(), key=str):\n"
        "        records = task_records(group)\n"
        "        s = summary(records)\n"
        "        lo, hi = stratified_bootstrap(\n"
        "            records, lambda rs: float(np.median([r['abs_error'] for r in rs])), resamples=resamples)\n"
        "        out.append({'estimator': estimator, 'targets': source or 'base', 'tasks': s['n_tasks'],\n"
        "                    'MdAE log10': s['mdae_log10'], 'CI95': f'[{lo:.3f}, {hi:.3f}]',\n"
        "                    'within 2x': s['within2x'], 'Spearman': s['spearman'],\n"
        "                    'GM ratio': s['gm_ratio'], 'AUR': s['aur'], 'parse': s['parse_rate']})\n"
        "    return pd.DataFrame(out)\n\n"
        "t2 = table('test')\n"
        "t2.round(3)",
    ),
    (
        "md",
        "## Check against the released metrics\n\n"
        "The released `metrics.json` was written with 10,000 resamples. The point estimates must "
        "match exactly; the confidence intervals match up to bootstrap noise.",
    ),
    (
        "code",
        "released = {(m['estimator'], m['split'], m['card_id'], m['frame'], m['target_source']): m\n"
        "            for m in json.loads((DATA / 'metrics.json').read_text())}\n"
        "bad = 0\n"
        "for (estimator, source), group in select('test').items():\n"
        "    mine = summary(task_records(group))['mdae_log10']\n"
        "    ref = released[(estimator, 'test', 'REF', 'self', source or 'base')]['mdae_log10']\n"
        "    bad += abs(mine - ref) > 1e-9\n"
        "print('mismatching estimators:', bad)\n"
        "assert bad == 0",
    ),
    ("md", "## Estimated versus measured completion time (hero figure)"),
    (
        "code",
        "chosen = [e for e in ('E1', 'M2-S0', 'M1-S0') if any(k[0] == e for k in select('test'))]\n"
        "fig, axes = plt.subplots(1, len(chosen), figsize=(4.2 * len(chosen), 4), squeeze=False)\n"
        "for axis, name in zip(axes.flat, chosen):\n"
        "    group = [r for (e, _), g in select('test').items() if e == name for r in g]\n"
        "    records = [r for r in task_records(group) if r['predicted']]\n"
        "    x = np.array([r['actual'] for r in records]); y = np.array([r['predicted'] for r in records])\n"
        "    axis.scatter(x, y, s=12, alpha=.55)\n"
        "    line = np.geomspace(min(x.min(), y.min()) * .7, max(x.max(), y.max()) * 1.4, 50)\n"
        "    axis.plot(line, line, 'k', lw=1); axis.fill_between(line, line / 2, line * 2, alpha=.15)\n"
        "    axis.set(xscale='log', yscale='log', title=name, xlabel='measured (s)', ylabel='estimated (s)')\n"
        "plt.tight_layout()",
    ),
    ("md", "## Pre-registered hypotheses"),
    (
        "code",
        "hypotheses = json.loads((DATA / 'hypotheses.json').read_text())\n"
        "pd.DataFrame([{'hypothesis': k, 'status': v.get('status'), 'passed': v.get('passed')}\n"
        "              for k, v in hypotheses.items()])",
    ),
    ("md", "## Scheduling and timeout simulations (exploratory)"),
    (
        "code",
        "path = DATA / 'simulations.json'\n"
        "if path.exists():\n"
        "    sim = json.loads(path.read_text())\n"
        "    display(pd.DataFrame([{'estimator': k, 'SJF vs oracle': v['scheduling']['sjf_relative_to_oracle'],\n"
        "                           'FIFO vs oracle': v['scheduling']['fifo_relative_to_oracle']}\n"
        "                          for k, v in sim.items()]).round(3))",
    ),
]


def main():
    notebook = nbf.v4.new_notebook()
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for kind, source in CELLS:
        notebook.cells.append(
            nbf.v4.new_markdown_cell(source) if kind == "md" else nbf.v4.new_code_cell(source)
        )
    target = ROOT / "notebooks/reproduce_results.ipynb"
    nbf.write(notebook, target)
    print(target)


if __name__ == "__main__":
    main()
