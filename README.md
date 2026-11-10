# Anthropochrony

**Measuring and Correcting Human-Time Bias in LLM Self-Estimates**

**Anthropochrony** means using human-scale durations when estimating a model's own runtime. We use the term as a project framing, crediting Monperrus and Stenqvist for the prior concept of anthropocentric time bias; we do not claim the phenomenon is new.

Anthropochrony studies whether a coding agent can learn to estimate its own completion
time from measured runs and the serving speed stated in its prompt, while
preserving its coding ability and estimates for human developers.

The planned pipeline uses **CuratorKIT → AlignTune → AuditKIT**, with
`CohereLabs/tiny-aya-global` as the subject model.

## Status

Implementation and acceptance testing are in progress. Nothing is public.
No fine-tuned research results or measured improvements are claimed.

- **Gates passed:** G0 security/HF-security/template, G1 (1,465 tasks: 869 train,
  114 validation, 255 test, 93 unseen-family test, 134 extra), G2 (per-call
  latency fallback, a dated deviation in the preregistration), G3 pilot and
  invariance, and G3-base-full (11,720 K=8 trajectories on the base model).
- **Not yet passed:** G4 (estimate parse and audit), G5 (SFT data and dry run), G6
  (merged-serving parity) and the overall release gates.
- Base labels, the B0 to B2 baselines and the noise ceiling are built. Teacher
  (B4) quantities and base estimates are being produced.
- Candidate code runs only in the credential-free local sandbox; HF Jobs does
  generation and training. See [execution blocker](docs/EXECUTION_BLOCKER.md).
- Merged-serving tooling was exercised on a real L40S with a random adapter
  (`results/smoke-merged.json`). That is a tooling check, not a research result.
- Budget is tracked in `artifacts/budget.json` and `RUNLOG.md`. Reserved amounts
  are conservative until reconciled against job start and finish times.

- [Implementation plan](docs/PLAN.md)
- [Pre-registration](docs/PREREGISTRATION.md)
- [Prompt specification](docs/PROMPTS.md)
- [Hackathon checklist](docs/HACKATHON_REQUIREMENTS.md)

## Naming

- Project and paper name: **Anthropochrony**
- Repository directory and Python package: `anthropochrony`
- Command name: `anthropochrony`
- Planned Hub artifact prefix: `anthropochrony-`

The research reports outside this repository retain the original working name
as historical background. The research protocol is unchanged by the rename.

## Licenses

Code uses [Apache-2.0](LICENSE). Model derivatives and benchmark
data retain their own licenses, including Tiny Aya's CC-BY-NC-4.0 and acceptable
use policy. Artifacts stay private until the release stage is approved.

## Local checks

The project uses separate stack and credential-free execution environments.
The existing `env.sh` selects them; it does not change `HF_HOME`.

```sh
. ./env.sh
uv sync --extra demo
uv run --extra demo ruff check src tests jobs space
uv run --extra demo pytest -q
uv run --extra demo anthropochrony status
```

Research generation and training run on HF Jobs, not the local GPU. Stage
commands fail closed when prerequisite acceptance evidence is missing.
Measurement defaults to the 100-task, K=8 pilot. `measure --phase full`
requires G3, including independent 100-task validation at both extreme speed
cards. Each extreme must pass the 0.05 median absolute log-token-ratio limit;
pooling the two conditions cannot hide a failure.
`measure --auto` adds resumable batched GPU generation and local sandbox tests
between rounds. Each GPU job must be terminal before candidate execution.
The pilot runs first; its quality check cannot pass G3 without the independent
validation-only invariance runs. All commands still require G2, so no new
measurement job can launch while its current accuracy failure is unresolved.
The real one-step training check is reachable with `train --stage dry-run`;
`train --stage lr-sweep --learning-rate 0.0001` prepares the first preregistered
sweep run. These are distinct from `--dry-run`, which previews a paid launch.
Neither command bypasses its prerequisite gates.
`select-lr` validates both complete self-REF validation sets against their
private checkpoint identities and one-epoch training records, then freezes
one rate for M1, M2, and M3. The final two-epoch runs cannot bypass that record.
Adapter uploads include training logs and state, refuse public repositories,
and pin the portable base-model identity. The dry-run stage forwards no HF token.
Reports now write `results/hypotheses.json`, keep training seeds separate for
H4/H6b, and accept H6a pairs only for M1 seed 0. The `capability` command derives
those pairs from complete K=8 own trajectories. Fine-tuned test reporting
requires the frozen preregistration.
The Gradio demo currently has no trained backend configured; it does not
substitute canned predictions.
