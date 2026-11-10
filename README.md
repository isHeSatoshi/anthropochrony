# Anthropochrony

**Measuring and Correcting Human-Time Bias in LLM Self-Estimates**

*Hackathon: Track 1 (Fine-tuning and Data Curation), problem #8 (data-scarce domains), with the #2 ROI comparison. Pipeline: CuratorKIT 0.1.23 → AlignTune 0.1.32 → AuditKIT 1.1.6. Subject model: `CohereLabs/tiny-aya-global` @ `af89d219b53ed9b13b8a4645f9c8028973510324`.*

**Anthropochrony** is our project name for using human-scale durations when a model estimates its own runtime. The prior concept is *anthropocentric time bias* (Monperrus and Stenqvist). We do not claim the phenomenon is new. This repository measures it for one small coding model and tests a repair: fine-tune the model on its own measured run times, with the serving speed written in the prompt.

The full write-up is **[`paper/article.md`](paper/article.md)** (the nine sections the hackathon asks for). This README is the short version plus reproduction notes.

## Result in brief

All numbers are from **one training seed**, 255 held-out tasks (`test`) and 93 tasks from a task family never seen in training (`test_ood`), reference speed card, 95% bootstrap CIs. Error is the median |log10(estimate / measured own time)| (MdAE). Lower is better.

| Estimator | `test` MdAE | within 2× | `test_ood` MdAE |
|---|---|---|---|
| Base model, no speed card | 1.76 | 0.4% | 1.62 |
| Base model, speed card | 1.38 [1.32, 1.41] | 0.4% | 1.20 |
| Base model, 5 measured examples (few-shot) | 0.24 [0.21, 0.30] | 55% | 0.50 |
| Text-feature regressor (no LM) | 0.14 [0.12, 0.16] | 83% | 0.13 |
| Tuned on teacher-guessed labels (M2) | 0.29 [0.26, 0.33] | 51% | 0.29 |
| Tuned on raw measured labels (M3) | 0.21 [0.18, 0.23] | 73% | 0.24 |
| **Tuned on curated measured labels (M1)** | **0.24 [0.21, 0.27]** | **66%** | **0.30 [0.23, 0.40]** |

On a desktop CPU with a 4-bit GGUF, 100 held-out tasks: 52% of tuned-model estimates are within 2× of the real wall-clock (base: 3%). pass@1 and human-frame answers did not change measurably; IFEval (first 200 prompts) fell 5 points, within noise for one seed; ARC-Challenge did not move.

Pre-registered hypotheses: H2 (in-distribution), H3 (unseen family), H4 (held-out speeds), H5 (measured beats guessed) and H6a/H6b (coding and human-frame answers preserved) were met on the single seed. **H1 (the base model's large human-scale overestimate with a speed card) was not met.** See [Limitations](#limitations).

![Hero figure](paper/figures/F1-estimated-vs-measured.png)

## Limitations

Read these before using any number above.

- **One training seed** for M1, M2 and M3. The $30 compute cap could not fund seeds 1 and 2. No seed-to-seed variance exists, so ranking differences of a few hundredths (M1 vs M3, M1 vs few-shot) are not established.
- **M1 does not beat few-shot prompting in-distribution** (0.24 vs 0.24). It is better within 2× (66% vs 55%), on the unseen family (0.30 vs 0.50) and at speeds beyond the few-shot examples.
- **A text-feature regressor (B2) beats every language-model estimator** in-distribution (0.14). The tuned model is not the best predictor; it is the one that can state the estimate in the model's own voice and follow the speed card.
- **Curation did not help**: raw measured labels (M3, 0.21) were as good as curated ones (M1, 0.24).
- **H1 not met as pre-registered.** With a speed card the base model is erratic, not biased toward hours.
- **Poor interval calibration** (stated 80% range covers 19% of fresh runs), **weak OOD ranking** (Spearman 0.08 on the unseen family), and **bad extrapolation to a much slower card** (0.49).
- The **G6 merged-serving parity criterion was changed** after seeing validation results (before test data): bf16 merging flips some near-tied digits. See `docs/PREREGISTRATION.md` §7.
- **General capability** was checked only on IFEval (first 200 prompts) and ARC-Challenge. M1 scored 5 points lower on IFEval prompt-level strict accuracy (0.750 to 0.700; within noise for one seed, but not a clean pass).
- The estimate parser was **hand-audited on 60 rows by the research agent**, not an independent human (58/60 correct). No LLM judge was used.
- The ROI comparison covers cost only; the closed model's accuracy on this task was not measured.
- Several exploratory questions were not run (probe, cross-model, success prediction, GRPO). The list is in the article, §7.11.

## Quick reproduction (CPU, no credentials)

Rebuild the main table from the released per-task estimate rows and check it against the released `metrics.json`.

```sh
git clone https://github.com/isHeSatoshi/anthropochrony && cd anthropochrony
. ./env.sh                      # sets cache dirs; never sets HF_HOME
uv sync --locked --extra demo
uv run pytest -q -m "not security"
hf download isHeSatoshi/anthropochrony-eval --repo-type dataset --local-dir artifacts/released-eval
ANTHROPOCHRONY_DATA=$PWD/artifacts/released-eval/results \
  uv run python scripts/run_notebook.py
```

The notebook (`notebooks/reproduce_results.ipynb`) recomputes the point estimates and bootstrap intervals with the pipeline's own metric code, asserts that they match `metrics.json`, and redraws the hero figure. If `ANTHROPOCHRONY_DATA` is unset it downloads `results/*` from the dataset named in `ANTHROPOCHRONY_DATASET`. The final figures are also in the dataset's `figures/` folder; `make figures` regenerates them from a full pipeline run (it reads `data/eval/estimates.jsonl`, `results/*.json` and `data/x10/`).

## Full pipeline

Needs an HF token with Jobs access and about **$27 of HF Jobs credit plus $1.7 of Inference Providers calls** (our total was inside the $30 cap) at `l40sx1` ($1.80/h). Candidate code (model-written Python) runs only in a credential-free local bubblewrap sandbox, never in a job and never next to a token.

| Step | Command | Where | Rough cost / time |
|---|---|---|---|
| Environments | `make setup exec-env` | local | minutes |
| Smoke and gates G0 | `make smoke preflight` | local | minutes |
| Task pool (CuratorKIT) | `make tasks LOCAL_DATA=…` | local | ~10 min |
| Speed calibration and throughput validation | `make calibrate` | HF Jobs | ~$1.5 |
| Base trajectories (K=8, up to 3 attempts, 11,720 runs) | `make measure RUN=base PHASE=full` | HF Jobs + local sandbox | ~$4.5 (all generation jobs) |
| Labels, baselines, noise ceiling | `make labels baselines` | local | minutes |
| Base and teacher estimates | `anthropochrony estimate-run …`, `anthropochrony teacher` | HF Jobs / Providers | ~$1.4 + $1.7 |
| SFT data (CuratorKIT) | `make sft-data VARIANT=m1` (also `m2`, `m3`) | local | minutes |
| LR sweep, then training | `anthropochrony train --stage lr-sweep …`; `make train VARIANT=m1 SEED=0` | HF Jobs | ~$1.2 per sweep run; $2.2–2.7 per final run (about 70 min) |
| Re-measure tuned models, tuned estimates | `make remeasure RUN=m1-s0 PHASE=full`, `anthropochrony estimate-run …` | HF Jobs + local sandbox | ~$0.8 + $0.4 per model |
| Evaluate and figures | `make eval && make figures` | local | ~10 min |
| Capability | `anthropochrony lmeval …` | HF Jobs | ~$0.4–1.0 per model |

Every stage command fails closed if the acceptance evidence for the previous gate is missing (`results/gates/`). `docs/PLAN.md` is the full design, `docs/PREREGISTRATION.md` the frozen protocol (git tag `prereg-v1`), `docs/PROMPTS.md` every prompt, `RUNLOG.md` the timestamped run log and `artifacts/budget.json` the spend ledger.

## Demo (local)

A Gradio app runs the base and tuned GGUFs with llama.cpp on your machine, measures your own prefill and decode speed, writes that speed into the prompt, and compares each model's estimate with a real sandboxed run of a verified task. Pasted code is never executed.

```sh
. ./env.sh
llama-server -m base-q4_k_m.gguf    --port 8081 &     # base
llama-server -m m1-s0-q4_k_m.gguf   --port 8082 &     # Anthropochrony
ANTHROPOCHRONY_BASE_URL=http://127.0.0.1:8081 ANTHROPOCHRONY_TUNED_URL=http://127.0.0.1:8082 \
ANTHROPOCHRONY_GGUF=m1-s0-q4_k_m.gguf ANTHROPOCHRONY_LLAMA_BENCH=$(which llama-bench) \
  make demo
```

We did not host it as a Space: free accounts cannot create Gradio CPU Spaces, and the demo needs a CPU build of llama.cpp and the sandbox.

## Pins and environments

- Two separate environments: `stack` (Python 3.12, `uv.lock`, pinned versions, including `transformers==5.21.1`) and `exec` (Python 3.10, `src/anthropochrony/execenv/requirements.lock`) used only inside the sandbox.
- Subject model and every dataset are pinned by revision in `configs/pins.yaml`.
- vLLM 0.33.0 (Docker `vllm/vllm-openai:v0.33.0`) on HF Jobs, merged weights only (LoRA serving crashes on Cohere2).
- llama.cpp b12110 for GGUF export and the CPU deployment test.

## Sandbox and security

Model-written code runs under bubblewrap (no network, read-only root, empty home, resource limits, no environment, no credentials). HF Jobs never execute candidate code. The security tests (`pytest -m security`) exercise the real isolation and need user namespaces. Tokens are passed to jobs as secrets and are never logged or committed. See `docs/EXECUTION_BLOCKER.md`.

## Licenses

- Code: [Apache-2.0](LICENSE).
- Tiny Aya Global and everything derived from it (adapters, GGUF): **CC-BY-NC-4.0** plus the [Cohere Labs Acceptable Use Policy](https://docs.cohere.com/docs/c4ai-acceptable-use-policy). Non-commercial.
- CuratorKIT LSAL v1.1 and AuditKIT LSAL v1.2 are non-commercial; AlignTune MIT. We used them for individual research under those terms.
- ClassEval data is CC BY-NC 4.0. LeetCodeDataset, CodeContests and TACO statements are third-party; we release IDs and a loader, not their text. Per-dataset terms are in `paper/article.md` §4.

## Citation

```bibtex
@misc{anthropochrony2026,
  title  = {Anthropochrony: Measuring and Correcting Human-Time Bias in LLM Self-Estimates},
  author = {isHeSatoshi},
  year   = {2026},
  note   = {Hackathon submission, Track 1}
}
```

Prior and concurrent work: Monperrus and Stenqvist (anthropocentric time bias); MetaTime; AgentTime. See `research/related_work.md`.

## Repository layout

```text
src/anthropochrony/   package: tasks, sandbox, gen, measure, timemodel, labels, baselines,
                      curate, train, eval, release, demo
jobs/                 scripts that run inside HF Jobs
scripts/              analysis helpers (ROI, CPU deployment, notebook builder, G6 parity)
configs/              speeds, pins, measurement and baseline settings
docs/                 PLAN, PREREGISTRATION, PROMPTS, HACKATHON_REQUIREMENTS
results/              metrics, hypotheses, simulations, ROI, gate evidence
paper/                article and figures
notebooks/            CPU reproduction notebook
space/                Gradio demo app
tests/                176 tests
```
