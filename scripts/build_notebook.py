"""Write notebooks/reproduce_results.ipynb. The notebook recomputes the headline table and post-hoc analyses from released rows on CPU."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

CELLS = [
    (
        "md",
        "# Reproduce the Anthropochrony results on CPU\n\n"
        "This notebook rebuilds the confirmatory table (`T2`) with explicit `Target source` "
        "columns (`own` vs `base`), verifies exact `1e-9` numerical agreement against "
        "`results/metrics.json`, redraws the hero scatter, and inspects our headline OOD "
        "generalization (`test_ood`: `CLASS`), token-bottleneck / verbal-arithmetic audit, "
        "workload decomposition with an exact timing model (`M1-Struct`), `val`-selection "
        "failure analysis, split-conformal calibration (`M1-Struct+CP`), `M3` forensic audit, "
        "and downstream systems simulations (`SJF` scheduling, timeout tripwires, and `15 s` "
        "multi-tier SLA routing). It requires no GPU, no model download, and no credentials.",
    ),
    (
        "code",
        "import json, os\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from IPython.display import display\n"
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
        "## 1. Confirmatory table: `test`, reference speed card (`REF`), self frame\n\n"
        "Errors are `|log10(estimate / measured)|`, averaged over question templates within each "
        "task first. Fine-tuned models are scored against their own re-measured runs (`own`), and "
        "base/baseline estimators against the base runs (`base`). Unparseable answers count as "
        "infinite error.",
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
        "        out.append({'estimator': estimator, 'Target source': source or 'base', 'tasks': s['n_tasks'],\n"
        "                    'MdAE log10': s['mdae_log10'], 'CI95': f'[{lo:.3f}, {hi:.3f}]',\n"
        "                    'within 2x': s['within2x'], 'Spearman': s['spearman'],\n"
        "                    'GM ratio': s['gm_ratio'], 'AUR': s['aur'], 'parse': s['parse_rate']})\n"
        "    return pd.DataFrame(out)\n\n"
        "print('Unseen task family OOD (test_ood: CLASS, n=93):')\n"
        "t2_ood = table('test_ood')\n"
        "display(t2_ood.round(3))\n"
        "print('In-distribution held-out tasks (test, n=255):')\n"
        "t2 = table('test')\n"
        "display(t2.round(3))",
    ),
    (
        "md",
        "## 2. Exact check against released `metrics.json`\n\n"
        "The released `metrics.json` was computed with 10,000 stratified resamples. Point "
        "estimates must match to `1e-9` across both `test` and `test_ood`.",
    ),
    (
        "code",
        "released = {(m['estimator'], m['split'], m['card_id'], m['frame'], m['target_source']): m\n"
        "            for m in json.loads((DATA / 'metrics.json').read_text())}\n"
        "bad = 0\n"
        "for split_name in ('test', 'test_ood'):\n"
        "    for (estimator, source), group in select(split_name).items():\n"
        "        mine = summary(task_records(group))['mdae_log10']\n"
        "        ref = released[(estimator, split_name, 'REF', 'self', source or 'base')]['mdae_log10']\n"
        "        bad += abs(mine - ref) > 1e-9\n"
        "print('mismatching estimators across test + test_ood:', bad)\n"
        "assert bad == 0",
    ),
    ("md", "## 3. Estimated versus measured completion time (`test`, `REF`)"),
    (
        "code",
        "chosen = [e for e in ('E1', 'E2', 'M2-S0', 'M1-S0', 'B2') if any(k[0] == e for k in select('test'))]\n"
        "fig, axes = plt.subplots(1, len(chosen), figsize=(3.6 * len(chosen), 3.6), squeeze=False)\n"
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
    ("md", "## 4. Pre-registered confirmatory hypotheses (`H1`–`H6`)"),
    (
        "code",
        "hypotheses = json.loads((DATA / 'hypotheses.json').read_text())\n"
        "display(pd.DataFrame([{'hypothesis': k, 'status': v.get('status'), 'passed': v.get('passed')}\n"
        "                      for k, v in hypotheses.items()]))",
    ),
    (
        "md",
        "## 5. Headline Token-Bottleneck Audit, Workload Decomposition with an Exact Timing Model (`M1-Struct`), `val`-Selection Failure, and Conformal Calibration (`X1`)",
    ),
    (
        "code",
        "extra_path = DATA / 'extra-analyses.json'\n"
        "if extra_path.exists():\n"
        "    extra = json.loads(extra_path.read_text())\n"
        "    ref_post = extra.get('posthoc_ref_summary', {})\n"
        "    post_rows = []\n"
        "    for split_name in ('test_ood', 'test', 'val'):\n"
        "        for est, s in ref_post.get(split_name, {}).items():\n"
        "            default_src = 'base' if (split_name == 'val' or not est.startswith(('M1', 'M2'))) else 'own'\n"
        "            post_rows.append({'split': split_name, 'estimator': est,\n"
        "                              'Target source': s.get('target_source', default_src),\n"
        "                              'MdAE log10': s['mdae_log10'],\n"
        "                              'within 2x': s['within2x'], 'Spearman': s['spearman']})\n"
        "    if post_rows:\n"
        "        print('Post-hoc REF accuracy (workload decomposition + exact timing M1-Struct, with Target source):')\n"
        "        display(pd.DataFrame(post_rows).round(3))\n"
        "    tok = extra.get('token_and_arithmetic_audit', {})\n"
        "    if tok:\n"
        "        tok_rows = []\n"
        "        for split_name in ('test', 'test_ood'):\n"
        "            for est, s in tok.get(split_name, {}).items():\n"
        "                tok_rows.append({'split': split_name, 'estimator': est,\n"
        "                                 'Target source': 'own' if est.startswith(('M1', 'M2')) else 'base',\n"
        "                                 'token MdAE': s['token_mdae_log10'],\n"
        "                                 'token within 2x': s['token_within2x'],\n"
        "                                 'bare verbal div err': s['arithmetic_inconsistency_bare_mdae'],\n"
        "                                 'latency-adj div err': s['arithmetic_inconsistency_with_latency_mdae']})\n"
        "        print('Headline diagnostic — Token-workload self-knowledge vs. verbal arithmetic bottleneck:')\n"
        "        display(pd.DataFrame(tok_rows).round(3))\n"
        "    extrap = extra.get('speed_card_disentanglement', {}).get('by_split_and_card', {})\n"
        "    if extrap:\n"
        "        ex_rows = []\n"
        "        for split_name in ('test', 'test_ood'):\n"
        "            for est, v in extrap.get(split_name, {}).get('extrap-low', {}).items():\n"
        "                ex_rows.append({'split': split_name, 'estimator': est,\n"
        "                                'Target source': 'own' if est.startswith(('M1', 'M2')) else 'base', **v})\n"
        "        print('Slow-hardware extrapolation (extrap-low, 1.5 tok/s):')\n"
        "        display(pd.DataFrame(ex_rows).round(3))\n"
        "    hyb = extra.get('hybrid_ensemble', {}).get('splits', {})\n"
        "    if hyb:\n"
        "        hyb_rows = []\n"
        "        for sp, d in hyb.items():\n"
        "            tsrc = d.get('target_source', 'base' if sp == 'val' else 'own')\n"
        "            for label, key in (('val-selected blend (w=0.80)', 'val_selected_blend'),\n"
        "                               ('equal geometric blend (w=0.50)', 'equal_geometric_blend'),\n"
        "                               ('standalone B2 (w=0.00)', 'standalone_b2_same_target'),\n"
        "                               ('standalone M1-Struct (w=1.00)', 'standalone_m1_struct_same_target')):\n"
        "                if key in d:\n"
        "                    hyb_rows.append({'split': sp, 'policy': label, 'Target source': tsrc,\n"
        "                                     'MdAE log10': d[key]['mdae_log10'],\n"
        "                                     'within 2x': d[key]['within2x'],\n"
        "                                     'Spearman': d[key]['spearman']})\n"
        "        print('Explicit val-selection failure on M1-Struct + B2 blend (base val targets -> own test targets):')\n"
        "        display(pd.DataFrame(hyb_rows).round(3))\n"
        "    x1 = extra.get('x1_interval_calibration', {}).get('fresh_trajectories', {})\n"
        "    if x1:\n"
        "        x1_rows = []\n"
        "        for sp in ('test', 'test_ood'):\n"
        "            for est, v in x1.get(sp, {}).items():\n"
        "                x1_rows.append({'split': sp, 'estimator': est,\n"
        "                                'Target source': 'own' if est.startswith('M') else 'base', **v})\n"
        "        print('X1 fresh-trajectory 80% interval calibration (exchangeable test vs CLASS OOD shift):')\n"
        "        display(pd.DataFrame(x1_rows).round(3))",
    ),
    (
        "md",
        "## 6. Downstream systems benchmark: `SJF` scheduling, `1.0x` timeout tripwires, and `15 s` SLA routing",
    ),
    (
        "code",
        "path = DATA / 'simulations.json'\n"
        "if path.exists():\n"
        "    sim = json.loads(path.read_text())\n"
        "    sched_rows = []\n"
        "    for k, v in sim.items():\n"
        "        if k.startswith('_') or not isinstance(v, dict) or 'scheduling' not in v:\n"
        "            continue\n"
        "        tw1 = next((t for t in v.get('tripwire', []) if t['multiplier'] == 1), {})\n"
        "        sched_rows.append({\n"
        "            'estimator': k,\n"
        "            'SJF vs oracle': v['scheduling']['sjf_relative_to_oracle'],\n"
        "            'FIFO vs oracle': v['scheduling']['fifo_relative_to_oracle'],\n"
        "            'k=1.0x false stop': tw1.get('false_stop_rate'),\n"
        "            'k=1.0x failed GPU saved': tw1.get('failed_time_saved_share'),\n"
        "        })\n"
        "    display(pd.DataFrame(sched_rows).round(3))\n"
        "    sla = sim.get('_systems_summary', {}).get('sla_tier_routing_15s', {})\n"
        "    if sla:\n"
        "        print('Multi-tier Edge ($0.25/h) vs Datacenter ($1.80/h) SLA routing (15 s deadline):')\n"
        "        display(pd.DataFrame([{'policy': k, **v} for k, v in sla.items()]).round(3))",
    ),
]


def main():
    notebook = nbf.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    for kind, source in CELLS:
        notebook.cells.append(
            nbf.v4.new_markdown_cell(source) if kind == "md" else nbf.v4.new_code_cell(source)
        )
    target = ROOT / "notebooks/reproduce_results.ipynb"
    nbf.write(notebook, target)
    print(target)


if __name__ == "__main__":
    main()
