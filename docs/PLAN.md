# Anthropochrony: execution plan for the delegate agent

**Paper title:** Anthropochrony: Measuring and Correcting Human-Time Bias in LLM Self-Estimates.

Version 1.1, 2026-11-09. Written for an autonomous agent that will run the project end to end. Read sections 0 to 2 fully before touching anything.

Changes in 1.1:
- HF login is done and verified (U1).
- Facts were checked against the gated Tiny Aya files (S0.2, S0.5).
- The workspace and disk layout are fixed (§4.2).
- Merged models are no longer pushed to the Hub, because a free account has 100 GB of private storage.
- M3 (raw vs curated labels) moves to P0, because problem #8 asks for exactly that comparison.
- A per-attempt context budget is defined (§2).
- HF PRO is not used (user decision): the write-up goes in the GitHub README (U7).
- All GPU generation runs on HF Jobs `l40sx1` (user decision; §4.3), with binding rules for full GPU use (§4.3.1). The reference card REF is measured on `l40sx1`. Tests still run locally, where no token exists.
- Demo hosting is corrected: without PRO, a personal account can create only ZeroGPU (Gradio) or static Spaces (U4, S9).
- The edge-GPU speed profile is widened so that REF and the measured cards fall inside the training profiles (S2.4).
- Estimate answers get a 512-token cap, so long base-model answers are not cut off (S4.2).
- Pooled confidence intervals use a bootstrap stratified by family, because 4 or 5 families are too few clusters for a cluster bootstrap (§7.3).
- The hackathon checklist is written (`docs/HACKATHON_REQUIREMENTS.md`).

Companion documents (binding):

| File | Role |
|---|---|
| `docs/PREREGISTRATION.md` | Hypotheses, metrics and analysis plan. Freeze it (tag `prereg-v1`) before any fine-tuned model is evaluated on `test` or `test_ood`. |
| `docs/PROMPTS.md` | The only source of prompt text, answer format and parser rules. |
| `docs/HACKATHON_REQUIREMENTS.md` | Submission checklist. |
| `research/lexsi_stack_audit.md` | CuratorKIT / AlignTune / AuditKIT API facts, pitfalls and snippets. |
| `research/hf_compute_and_model.md` | HF Jobs flavors and prices, Tiny Aya facts, vLLM/llama.cpp support, command templates. |
| `research/coding_datasets.md` | Dataset mix, filters, splits, sandbox design, loader sketch (`scratch/loader_sketch.py`). |
| `research/related_work.md`, `research/references.bib` | Novelty statement, claims to avoid, 51 references. |
| `pilot/` | Pilot evidence (Cohere demo Space, one sample per prompt). Rerun before quoting. |

If a research report disagrees with this plan, `docs/PREREGISTRATION.md` or `docs/PROMPTS.md`, those three files win. The reports are background written before the HF login, and some of their commands are superseded (for example, merged-model pushes and the `[cli]` install extra).

---

## 0. Read first

### 0.1 Summary

Language models answer "how long will this take you?" in human time. In our pilot, Tiny Aya Global wrote a to-do CLI in 2.6 s (the code compiled; it was not run) and estimated "6-9 hours". Telling it its own speed barely helped: its estimates were still about 1,250 times its generation time (geometric mean over 12 answers on 6 tasks). Anthropochrony teaches Tiny Aya Global to predict its **own** completion time for a coding run (up to 3 attempts with test feedback), given the serving speed stated in its system prompt. Its answers about human developers stay in human time, and its coding ability stays unchanged.

The data are measurements of the model's own runs, curated with **CuratorKIT**. The model is trained with LoRA SFT in **AlignTune**, and everything is evaluated with **AuditKIT** under a pre-registered protocol.

### 0.2 Hard constraints

| Constraint | Value |
|---|---|
| Deadline | Submission form by **2026-11-14 23:59 AoE**; the latest submission counts. Aim to submit by 2026-11-14 12:00 AoE. |
| Track | Track 1 (Fine-tuning and Data Curation). Problem statement **#8** (primary), **#2** (ROI section), **#6** only if GRPO (M8) is done. |
| Lexsi repos | CuratorKIT `0.1.23` → AlignTune `0.1.32` → AuditKIT `1.1.6`, all three chained. |
| Models | Subject: `CohereLabs/tiny-aya-global` @ `af89d219b53ed9b13b8a4645f9c8028973510324`. Aya Expanse 8B only for the cross-prediction ablation (an allowed model). Other models only as teacher, judge or input-only baseline: `deepseek-ai/DeepSeek-V4-Pro`, `openai/gpt-oss-120b`, `BAAI/bge-small-en-v1.5`. |
| Deliverables | Public, un-gated HF model collection and dataset collection (models, adapters, **logs**, datasets); HF article with the 9 required sections (or the GitHub README if there is no article access, U7); public GitHub repo with a reproducibility README; public demo; abstract under 500 words; at least 2 before/after metrics; ROI comparison. Full checklist: `docs/HACKATHON_REQUIREMENTS.md`. |
| Licenses | Tiny Aya is CC-BY-NC-4.0 plus the Cohere Labs AUP, so every model artifact is CC-BY-NC-4.0 and non-commercial. ClassEval data is CC BY-NC 4.0. Do not redistribute LeetCode or CodeContests statement text; release IDs and a loader instead. |
| Budget | **Hard cap $30 total** (HF Jobs + Inference Providers; user decision). P0 target ≤ $24 with $6 kept back; P1 only from what is left. Details and cut order: §9.1. |

### 0.3 Rules for the executor

1. **Pre-registration is the contract.** Never change the time formula, metrics, thresholds or splits after seeing `test` results. Record any change in `docs/PREREGISTRATION.md` §7 with the date and the reason.
2. **Don't commit the bias we study.** Never plan or report your own work in human days or weeks. Plan by dependencies, compute hours and dollars.
3. **Reproducibility.** Pin everything (model revision, dataset revisions, image digests, lockfiles, llama.cpp tag, teacher model plus provider plus date). Every output file gets a sidecar `<name>.manifest.json` containing: git SHA, config hash, input file hashes, environment lock hash, hardware, job ID, start and end time.
4. **Single sources of truth:**
   - prompts come from `docs/PROMPTS.md`, compiled to `src/anthropochrony/prompts/*.yaml` with a test that the two match;
   - the time formula lives only in `src/anthropochrony/timemodel.py`;
   - metrics live only in `src/anthropochrony/eval/metrics.py`.
5. **Sandbox.** Model-generated code runs only in the sandbox (S0.6). Never run it anywhere a token exists.
6. **Secrets.** Never print, log, commit, or pass `HF_TOKEN` into a sandbox. Jobs get it via `-s HF_TOKEN`. `hf auth login` stored the token in `~/.cache/huggingface/token` (mode 600). Do **not** set `HF_HOME`, which would hide that file; set the cache variables in §4.2 instead. The token is temporary: on any 401/403 from the Hub, stop and ask the user.
7. **Money.** Log every paid job in `RUNLOG.md` (job ID, flavor, minutes, $, outcome). Ask the user before any single job estimated above $5, and stop before cumulative spend (including the next job's estimate) would pass **$30**.
8. **Outward actions.** All Hub and GitHub repos stay private until S10. Ask the user before making anything public and before submitting the form.
9. **Lexsi stack pitfalls** (details in `research/lexsi_stack_audit.md`):
   - CuratorKIT: `clean=False`; raise `max_tokens`; split by task *before* CuratorKIT; write your own JSONL from `result.passed`.
   - AlignTune:
     - training: `backend="trl"`, `use_peft=True`, `assistant_only_loss=True`, `max_seq_length=2048` (3,072 if the S5.4 rule triggers), `packing=False`, `load_best_model_at_end=False`;
     - publishing: `push_to_hub(..., private=True)`;
     - GRPO rewards are **per-sample** functions.
   - AuditKIT: unique output dirs, a content hash in `RunConfig(extra=...)`, and versioned scorer names (the cache is keyed by path and name).
   - Pin `transformers==5.21.1` in the shared environment.
10. **vLLM + LoRA crashes on Cohere2** (vLLM PR #65532). Serve **merged** models only.
11. **Gates.** Every stage ends with an acceptance gate. If a gate fails, fix it or apply the listed fallback, and record what you did. Never skip a gate silently.

### 0.4 User actions (blocking; ask the user)

| ID | Action | Status (2026-11-09) | Needed by |
|---|---|---|---|
| U1 | HF CLI and login on this machine | **Done.** `hf` installed; a fine-grained token with the permissions this project needs (own repos, Jobs, Inference Providers) is used. Gated download of Tiny Aya Global at the pinned revision works. | S0 |
| U2 | Compute credits: the project may use **$30** of HF credit. Jobs need a positive credit balance, and Inference Providers bill the same account. | **Done** | S0 |
| U3 | Accept the Aya Expanse 8B license. | **Done** (user, 2026-11-09). Re-check with one gated download in S0. | S6 (M6) |
| U4 | Choose demo hosting. The HF docs (read 2026-11-09) say that creating a Gradio or Docker Space on a personal account needs a paid plan (PRO). The options: (a) a **ZeroGPU Gradio Space**, free for accounts in good standing (verified email, older than 30 days), at most 2 per account; this account dates from 2023-11-30; (b) with PRO (U7), a CPU Space running the GGUF with llama.cpp (cpu-basic hardware is free, cpu-upgrade is billed); (c) repurpose one of the account's existing Spaces by renaming it to `anthropochrony-demo`, which replaces its content (the user picks which one); (d) as a last resort, a static Space with recorded runs. | Open | S9 |
| U5 | Confirm that the team is registered (registration closed Nov 5), that the registration lists HF username `isHeSatoshi` (needed for credits), and that one member is on Discord. Provide the registered email, Discord username and teammates for the form. | Open | S10 |
| U6 | (Recommended) Open a Discord ticket in #report-issues. Ask: may Tiny Aya derivatives (LoRA adapters, GGUF, merged weights) be public and un-gated under CC-BY-NC-4.0? Is Aya Expanse 8B fine as a cross-prediction control? Is the topic in scope for Track 1, problem #8? | Open | before S10 |
| U7 | HF PRO: **not buying** (user decision). The write-up goes in the GitHub README (rules accept it); the demo uses U4 (a), (c) or (d). | Closed | S9 |

---

## 1. Research design (summary; the details are binding in PREREGISTRATION.md)

**Question.** Do small LLMs know how long a coding task takes *them*, and can measured self-labels teach it?

**Hypotheses.**
- H1, bias: the base model overestimates by at least 10×, even when its speed is stated.
- H2, in-distribution fix: median error within 2×.
- H3, held-out family fix: median error within 3.2× on ClassEval.
- H4, speed conditioning, including speeds held out of training.
- H5, measured labels beat teacher-guessed labels.
- H6, no collateral damage: coding pass@1 and human-frame answers are unchanged.

**Contributions.**
1. C1: a measurement protocol and public data. About 1,400 tasks × 8 own runs (about 11k trajectories with tokens, attempts and test outcomes), with a time model validated against real wall-clock time.
2. C2: evidence of the anthropocentric time bias in Tiny Aya Global, robust to wording, with human-frame and generic-AI controls. Credit the term to Monperrus and Stenqvist.
3. C3: Anthropochrony, LoRA SFT on curated measured self-labels with varied stated speeds. We evaluate accuracy, ranking, interval calibration, a held-out family, unseen stated speeds and real deployments.
4. C4: controls. Teacher-guessed labels, raw labels, input-only regressors, a hidden-state probe, self vs cross prediction, the speed-rescaling rule, and re-measurement after fine-tuning.
5. C5: an open release of data, traces, adapters, GGUF, demo and code.

**Novelty statement** (use this wording; see `research/related_work.md` §3):

> To our knowledge, this is the first work to train a model to predict its own wall-clock completion time for multi-attempt, test-feedback coding runs as an explicit function of the serving speed given in context. It evaluates generalization to held-out task families and unseen speeds, self- vs cross-prediction, and the preservation of human-developer estimates.

Name MetaTime (ICLR 2027 submission, OpenReview `CAQeizWcB2`) as concurrent work. Respect the 12 "claims to avoid" in `research/related_work.md` §4. In particular:
- never claim to be the first to train time prediction;
- never call the result "introspection";
- say "unseen *stated* speeds" except for the real-deployment results (X10).

---

## 2. Definitions (the measurement contract)

| Term | Definition |
|---|---|
| Agent | Tiny Aya Global plus our harness: the system prompt in PROMPTS.md §1, R = 3 attempts, feedback per PROMPTS.md §3, temperature 0.1 and top_p 0.95 (model card; `generation_config.json` sets no sampling defaults, so always pass them), stop token IDs [6, 261001], context 8,192. The new-token cap for attempt j is min(2,048 (4,096 for PROJ), 8,192 − prompt_tokens(j)). If fewer than 256 tokens remain before an attempt, the run ends as `context_exhausted`: it counts as failed, and its time so far counts. |
| Task | Family, task turn (PROMPTS.md §2), hidden tests, public examples, time limits. |
| Run (trajectory) τ | Attempts j = 1..n with n ≤ 3; stops at the first pass or after attempt 3. **Run boundary:** from the start of reading the task (the first prefill) to the end of the last test run, whether or not it passes. |
| Own completion time | `t(τ; S, P) = Σ_j ( u_j / P + o_j / S ) + Σ_j e_j` |
| `o_j` | Generated tokens of attempt j, **including the stop token exactly once**. Verify the vLLM convention in S0.5. |
| `u_j` | New (uncached) prompt tokens at call j. For j = 1: rendered prompt tokens minus rendered system-turn tokens (the preamble and system prompt are assumed cached). For j > 1: `prompt_tokens(j) − prompt_tokens(j−1) − o_{j−1}`. |
| `e_j` | Sandbox wall time of the test run for attempt j, capped at the task's time limit. A timeout counts the cap. |
| Speed card c = (S, P) | Decode and prefill speed in tokens/s, stated in the system prompt (PROMPTS.md §1.1). Both are rounded to 2 significant figures, and labels use the rounded values that the model sees. |
| Reference card REF | (S_ref, P_ref), measured single-stream on HF `l40sx1` (the generation GPU) in S2.1. Measurement runs always state REF. |
| Labels for (task, c) | `T` = median over the K = 8 runs of t; `q10`, `q90`; `TOK` = median tokens; `ATT` = median attempts; `PFP` = shrunk first-attempt pass rate. Formats are in PROMPTS.md §5. |
| Frames | self, human, generic-AI, plan (PROMPTS.md §4). |
| Families | FUNC, LIB, ALGO_FN, ALGO_IO (train/val/test); CLASS (`test_ood` only); PROJ-SYN (P1, train/val/test); PROJ-8 (showcase only); `raw_extra` (tasks rejected by the S1 quality gates; M3 training only). |

---

## 3. Architecture and Lexsi stack usage

```text
HF datasets @ pinned SHAs
  → S1 loaders → ground-truth / flakiness / vacuous-test checks (sandbox)
  → CuratorKIT (schema gate, MinHash dedup, rejection reasons, manifest)          → anthropochrony-tasks
S2 speed calibration (HF l40sx1 = REF; local CPU llama.cpp)                     → configs/speeds.yaml
S3 measurement: vLLM attempt rounds (GPU, no code execution) ⇄ sandbox test rounds (no secrets)
                                                                                  → anthropochrony-traces
S4 labels (timemodel) + base / teacher / regressor / probe estimates (AuditKIT)  → anthropochrony-eval (base)
S5 SFT sets (CuratorKIT exact-dedup, schema gate, trainer-ready export)          → anthropochrony-sft
S6 AlignTune LoRA SFT on HF Jobs (sweep, seeds, ablations; push adapters)       → adapters (merged locally)
S7 re-measure fine-tuned models (own runs) + capability (AuditKIT lm-eval)       → anthropochrony-traces (ft)
S8 AuditKIT evaluation (custom scorers, callable vLLM model, compare) + stats,
   probes, simulations, real deployments                                         → results/, figures/
S9 demo Space (llama.cpp GGUF or ZeroGPU)        S10 release, collections, article, submission
```

| Lexsi repo | What we use it for | Features exercised | Stages |
|---|---|---|---|
| CuratorKIT 0.1.23 | Curate the task pool and the SFT sets; the raw vs curated comparison (problem #8) | Readers (JSONL), SchemaGate, `dedup="minhash"` for tasks and `"exact"` for SFT examples, `preprocessing_fn` rejection with reasons, manifests and provenance, `messages` export, `push_to_hub` | S1, S5 |
| AlignTune 0.1.32 | LoRA SFT for all variants and seeds; optional GRPO | `create_sft_trainer` (TRL backend, assistant-only loss, LoRA, seeds), `lexsi_provenance.json`, `push_to_hub` for adapters (`push_merged_to_hub` only for M6); `create_rl_trainer` with per-sample custom rewards (P2) | S6 |
| AuditKIT 1.1.6 | Every estimate evaluation; capability and safety checks; paired comparisons | `ak.evaluate` with a callable vLLM model and custom `@ak.scorer`s, `RunConfig(extra=...)` fingerprints, `ak.compare(...).significance()`, `ak.run_lmeval` (IFEval, ARC-Challenge), one registered safety scenario (P1) | S4, S7, S8 |

---

## 4. Environments and repository

### 4.1 Repository (GitHub `isHeSatoshi/anthropochrony`, private until S10)

- **Work root:** `/run/media/adi/Adi/Project/lexsi` (the user's project folder). Clone the repo to `/run/media/adi/Adi/Project/lexsi/anthropochrony`.
- **At S0.3:** copy `PLAN.md` and `docs/*.md` into the repo's `docs/`, and `research/references.bib` into `paper/`. From then on, edit only the repo copies. The research reports and `scratch/` stay outside the repo.
- **NTFS:** the work root is on an NTFS drive. If git reports spurious mode changes, run `git config core.fileMode false` in the repo.

```text
anthropochrony/
├── README.md              reproducibility README (hackathon requirement)
├── LICENSE                Apache-2.0 for code (data/models licensed separately)
├── pyproject.toml uv.lock  stack env (Python 3.12)
├── Makefile               smoke | tasks | calibrate | measure | labels | baselines | sft-data | train | remeasure | eval | figures | demo | release
├── configs/               pins.yaml speeds.yaml tasks.yaml measure.yaml labels.yaml eval.yaml sft/*.yaml
├── docs/                  PREREGISTRATION.md PROMPTS.md HACKATHON_REQUIREMENTS.md DATASHEET.md MODEL_CARD.md
├── src/anthropochrony/
│   ├── tasks/             loaders (port scratch/loader_sketch.py), unified schema, splits
│   ├── sandbox/           bwrap runner, in-job fallback runner, harness adapters (evalplus_check, unittest, lcd_check, stdio, pytest)
│   ├── execenv/           Python 3.10 execution-environment lockfile
│   ├── gen/               vLLM engine wrapper, chat rendering, stop tokens, token accounting
│   ├── measure/           round orchestration, trajectories, resume
│   ├── timemodel.py       the time formula (only here)
│   ├── prompts/           YAML compiled from docs/PROMPTS.md + loader
│   ├── labels/            label builder, answer formatter, conformal widths
│   ├── parse.py           estimate parser (regex, then LLM judge)
│   ├── baselines/         constant, family median, input-only regressor, probe, teacher
│   ├── curate/            CuratorKIT pipelines
│   ├── train/             AlignTune job scripts, merge, push
│   ├── eval/              AuditKIT wrappers, scorers, metrics, stats, figures
│   └── sim/               shortest-job-first scheduling, timeouts/tripwires
├── jobs/                  PEP 723 scripts for HF Jobs (calibrate_l4.py, train_sft.py, train_grpo.py, measure_round.py)
├── space/                 demo app
├── notebooks/reproduce_results.ipynb   CPU-only; downloads released data; rebuilds every table and figure
├── paper/                 article.md (HF article), figures/, tables/, latex/ (P1)
├── tests/                 timemodel, parser, labels, splits, sandbox security, metrics, prompt sync
└── RUNLOG.md
```

### 4.2 Environments

| Env | Python | Contents | Used for |
|---|---|---|---|
| `stack` | 3.12 (uv) | `torch==2.13.0 transformers==5.21.1 trl==1.7.1 peft==0.21.2 curatorkit==0.1.23 aligntune==0.1.32 auditkit[transformers,lmeval,interop]==1.1.6` + scipy, scikit-learn, pandas, matplotlib, datasketch, sentence-transformers. Resolved together in the audit (`research/lexsi_stack_audit.md` §2.1). | curation, training (HF Jobs), lm-eval, stats, figures |
| `gen` | 3.12 (uv, separate) | `vllm==0.33.0`, `auditkit==1.1.6` (no extras), our package. On HF Jobs use `vllm/vllm-openai:v0.33.0` pinned by digest. | all generation (measurement, estimates) |
| `exec` | **3.10** (uv, hash-locked) | BigCodeBench `requirements-eval.txt` minus the excluded libraries, ClassEval extras (python-docx, openpyxl, PyPDF2, bs4, Pillow, nltk with corpora pre-downloaded), numpy, pytest, func_timeout | tests only, inside bwrap, mounted read-only |
| `llamacpp` | n/a | llama.cpp at a pinned tag (record the `bNNNN`), CPU build (+ CUDA build optional) | GGUF conversion, local CPU deployment, demo |

`configs/pins.yaml` holds the model revision, all dataset revisions (from `research/coding_datasets.md` §3), the image digest, the llama.cpp tag, and teacher/judge model IDs with provider and date.

**Disk.** Measured on 2026-11-09:
- the work root's drive (`/run/media/adi/Adi`, NTFS via ntfs3) has about 27 GB free (94% used);
- `/home` (btrfs, same NVMe) has about 31 GB free;
- `/tmp` is a 16 GB RAM-backed tmpfs, so never put large files there.

The peak need is about 40 GB, so use both drives:

| What | Where | Size |
|---|---|---|
| Repo, data, traces, results, logs | work root | 3-5 GB |
| HF caches (`HF_HUB_CACHE`, `HF_DATASETS_CACHE`, `HF_XET_CACHE`) | `<work root>/.hf/{hub,datasets,xet}` | about 10 GB (base model 6.7 GB plus datasets) |
| Merged models built from adapters (S6.3) | `<work root>/models/` | 6.7 GB each; one at a time, or two if at least 12 GB stays free |
| uv environments and the uv cache | `/home/adi/.local/share/anthropochrony-envs/<env>` (via `UV_PROJECT_ENVIRONMENT`) and the default uv cache | 15-18 GB |
| `TMPDIR` (merges, GGUF intermediates) | `/home/adi/.cache/anthropochrony-tmp` | up to 9 GB, transient |

- Put these exports in `env.sh` at the repo root, and have the Makefile source it.
- Never set `HF_HOME` (rule 6).
- The environments live on `/home` because Python imports thousands of small files, which is much faster on btrfs than on NTFS, and uv can hardlink from its cache on the same filesystem.
- Check `df` before any step that writes more than 5 GB.
- `scratch/` (1.8 GB) holds the research workers' files. Port what S1 needs, then ask the user before deleting anything there.
- If the user frees space on the work drive, move the merged models and `TMPDIR` there.

### 4.3 Where compute runs (default)

| Work | Where | Why |
|---|---|---|
| Generation (measurement, re-measurement, estimates, time-model validation) | **HF Jobs `l40sx1`** (L40S 48 GB, 8 vCPU, $1.80/h), vLLM batched | User choice: HF GPU. Lowest cost per generated token of the single-GPU flavors (about 5-9k tok/s batched for a 3.35B model, vs 1.5-3k on `l4x1` at $0.80/h and 7-12k on `a100-large` at $2.50/h) and 3× faster than `l4x1`. H200 ($5/h) is not faster per dollar for a 3B model. The local 4060 Ti is only a fallback. |
| Test execution | **local**, bwrap plus systemd scope | Verified on this machine; no secrets in the sandbox. |
| Training | **HF Jobs `l40sx1`**, PEP 723 uv scripts | User preference; parallel seeds; fast. |
| REF speed calibration | HF Jobs `l40sx1` | Same GPU as all generation, so REF is the real measurement speed. |
| Teacher and judge | HF Inference Providers | DeepSeek-V4-Pro (deepinfra) and gpt-oss-120b. |
| Demo | HF Space (U4) | Public deployment. |

Aya Expanse 8B (M6) never runs locally: its bf16 weights (16 GB) do not fit the 16 GB GPU. Its training and evaluation run in HF Jobs (S6.4).

### 4.3.1 Get the most out of every `l40sx1` hour (binding)

The goal is the fastest wall time and the lowest $ per job. Paid GPU time must be spent generating, not loading, waiting or idling. Never change the model's outputs to save time: keep bf16 weights, the exact sampling settings and the stop tokens. No quantization, no speculative decoding and no shorter caps.

1. **One engine, one big call.** Load vLLM once per job and submit all prompts of a round (every task × K run × model) in a single `llm.generate` call. Never loop over prompts one at a time; batch-1 runs are only for S2.1 and S2.3, which measure single-stream speed on purpose.
2. **Engine settings** (start here, then tune on the S0.5 smoke and record the winners in `configs/measure.yaml`): `dtype="bfloat16"`, `max_model_len=8192`, `gpu_memory_utilization=0.92`, `enable_prefix_caching=True`, `enable_chunked_prefill=True`, `max_num_seqs=512`, `max_num_batched_tokens=16384`, CUDA graphs on (never `enforce_eager`). If vLLM reports KV-cache preemptions, lower `max_num_seqs` until they stop.
3. **Prefix reuse.** Put every prompt that shares the system turn and task turn in the same call, so the K = 8 runs of a task and all estimate templates of a task prefill once.
4. **Few, full jobs.** Each job starts with about 5-10 minutes of image pull and model download, so pack the work:
   - one job per measurement round, covering all tasks (never one job per task or per family);
   - one job for all S4 estimates (every estimator and frame), and one per fine-tuned model for S7 + S8;
   - for S7, load the base model once and run the models one after another in the same job (merge adapter in memory, generate, free GPU memory, next).
5. **Fast start.** Use the pinned `vllm/vllm-openai:v0.33.0` image (no pip installs at job start). Pre-stage the base model, adapters and inputs in the `<user>/anthropochrony-runs` bucket and mount it, so downloads come from HF storage at full speed.
6. **No idle GPU between rounds.** Prefer to run the tests inside the job on its 8 vCPUs while the GPU generates the next batch (pipeline: GPU works on model or round A while the CPU tests model or round B). This is allowed only if the job, after staging, deletes the token file, unsets `HF_TOKEN` and writes outputs only to the mounted bucket, and the S0.6 security tests pass inside a job (S0.7). Otherwise end the job after each round and run the tests locally.
7. **Watch utilization.** Log `nvidia-smi dmon -s um` every 5 s to the job log. During generation, GPU utilization should stay above 85% and memory near the cap. If it does not, raise `max_num_seqs` / `max_num_batched_tokens` or pack more prompts per call, and record the change.
8. **Stop paying at once.** Every job exits as soon as its outputs are written; `--timeout` is 2 × the expected time. Log the tokens/s achieved and the $ per 1M output tokens for every job in `RUNLOG.md`.

### 4.4 Hub naming (private until S10)

`<user>` is `isHeSatoshi`.

| Kind | Names |
|---|---|
| Datasets | `<user>/anthropochrony-tasks`, `<user>/anthropochrony-traces`, `<user>/anthropochrony-sft`, `<user>/anthropochrony-eval`, `<user>/anthropochrony-logs` (training logs and `trainer_state.json`, HF Jobs logs, `RUNLOG.md`, manifests, AuditKIT run records) |
| Models | `<user>/anthropochrony-tiny-aya-global-lora-<run>` (adapters, public at release); `<user>/anthropochrony-tiny-aya-global-GGUF`; `<user>/anthropochrony-tiny-aya-global-m1-s0` (merged M1-s0, created at S10 only if U6 allows public merged weights). Push no other merged models, except the private M6 model (S6.4): a free account has 100 GB of private storage, and a dozen merged models would fill most of it. |
| Bucket | `<user>/anthropochrony-runs` (HF Jobs outputs) |
| Space | `<user>/anthropochrony-demo` |
| Collections | "Anthropochrony models", "Anthropochrony data" |

Run names: `m1-s0`, `m1-s1`, `m1-s2`, `m2-s0` … `m5-s0`, `m1b-s0`, `m6-expanse8b-s0`, `lr-5e-5` and so on.

---

## 5. Stages

Each stage lists what to do, its outputs, a gate, and fallbacks. "Local" means this machine; "$" means HF credits.

### S0. Preflight (local, plus one HF job of about $0.2)

1. Confirm U1 (`hf auth whoami` prints `isHeSatoshi`) and U2 (credits > 0; the S0.7 `cpu-basic` job fails without them).
2. Download the gated files at the pinned revision into `pins/tiny-aya-global/`: `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json` and `special_tokens_map.json`. The repo has no separate `chat_template.jinja`; the template is the single `default` entry in `tokenizer_config.json`. The planner checked the following on 2026-11-09 with transformers 5.21.1. Re-check each one and record the results in `configs/pins.yaml`:
   - `tokenizer.json` sha256 is `2227ea9c52e8afb3f98bfed2679008b275f2664de69dfde174b374389eb0225d` (matched).
   - The chat template equals `scratch/tiny_aya_meta/chat_template.jinja` (matched). Still compare it with TRL 1.7.1 `cohere2.jinja`.
   - Token IDs: `<BOS_TOKEN>` 2, `<EOS_TOKEN>` 3, `<|START_OF_TURN_TOKEN|>` 5, `<|END_OF_TURN_TOKEN|>` 6, `<|START_RESPONSE|>` 261000, `<|END_RESPONSE|>` 261001, `<PAD>` 0.
   - The tokenizer's eos is ID 6, but `config.json` and `generation_config.json` declare only eos 3, and no sampling defaults. Always pass `stop_token_ids=[6, 261001]`, temperature and top_p explicitly.
   - `max_position_embeddings` is 8,192. The MLX mirror's 500,000 is wrong.
3. Create the GitHub repo (private), the scaffold from §4.1, CI (ruff + pytest on CPU), and `RUNLOG.md`.
4. Build the `stack`, `gen`, `exec` and `llamacpp` environments. Commit the lockfiles and record their hashes.
5. **Generation smoke** (HF `l40sx1` job, `--timeout 30m`, about $0.5): 10 FUNC tasks × 1 attempt; also download Aya Expanse 8B `config.json` once to confirm U3.
   - The token IDs sent to vLLM equal those of `tokenizer.apply_chat_template(..., tokenize=True, return_dict=False)` in both the `gen` and the `stack` environment (training renders in `stack`), on 20 conversations that include feedback turns. Record each environment's transformers version. In transformers 5.21.1 the call returns a `BatchEncoding` unless `return_dict=False`. Pass `TokensPrompt` to avoid a double BOS: re-tokenizing the rendered text adds a second BOS (`[2, 2, …]`, checked).
   - `stop_token_ids=[6, 261001]`; the stop token is counted exactly once in `o_j`.
   - The system-turn token count is recorded. Checked values: 359 tokens with no system message, and 510 with the sc01 agent card at S = 35, P = 5,000. A system message adds a 33-token "# Developer Preamble" header plus its own tokens.
   - Batched throughput is measured.
6. **Sandbox security tests** (pytest; all must pass), using the bwrap + `systemd-run --user --scope` recipe in `research/coding_datasets.md` §7.2:
   1. `HF_TOKEN` is not visible (neither in the environment nor via `/proc`);
   2. there is no network;
   3. no writes outside `$WORK`;
   4. timeouts kill the whole process tree;
   5. the memory cap is enforced;
   6. a fork bomb is contained;
   7. a known-good and a known-bad solution are classified correctly for each harness (`evalplus_check`, `unittest`, `lcd_check`, `stdio`, `pytest`).
7. **HF Jobs smoke.** First run a 1-minute `cpu-basic` job (`hf jobs run --flavor cpu-basic --timeout 2m python:3.12 python -c "print(1)"`) to confirm that the credit balance allows Jobs. Then run `l40sx1` with `--timeout 20m`. Verify:
   - the command overrides the vLLM image entrypoint;
   - the bucket volume mounts read-write;
   - `-s HF_TOKEN` downloads the gated weights;
   - which GPU is attached;
   - whether `id -u` = 0 and whether bwrap works inside the job (needed only for X10 on HF).

**Gate G0:** steps 2 and 5 to 7 pass. After steps 4 and 5, `df` shows at least 15 GB free on the work-root drive and 10 GB on `/home` (§4.2); if not, ask the user to free space.
**Fallback:** if local vLLM fails, generate on HF Jobs (Appendix A.3) and keep execution local.

### S1. Task pool and splits (local CPU)

1. Port `scratch/loader_sketch.py` into `src/anthropochrony/tasks/`. Load the 5 families at the pinned revisions (`research/coding_datasets.md` §8.1):

   | Family | Source | Tasks | Harness |
   |---|---|---:|---|
   | FUNC | HumanEval+ 164 + MBPP+ 235 | 399 | `evalplus_check` |
   | CLASS | ClassEval | 100 | `unittest` (own sandboxed runner) |
   | LIB | BigCodeBench v0.1.4 (136 Hard + 214 others) | 350 | `unittest` |
   | ALGO_FN | LeetCodeDataset (228 test-split + 122 train) | 350 | `lcd_check` |
   | ALGO_IO | CodeContests valid+test with a Python 3 solution | 217 | `stdio` |

2. Apply the filters in `research/coding_datasets.md` §8.2, in order. Write every drop to `tasks_dropped.jsonl` with its reason.
3. **Ground-truth checks** in the `exec` sandbox (§7.4 of that report):
   - the reference passes 3 of 3 runs;
   - flakiness: the 3 runs agree, and drop if the coefficient of variation is above 0.5 and the max time is above 1 s;
   - a vacuous solution must fail;
   - record GT time.

   Time limits:
   - per test: max(min limit, 4 × GT);
   - per-task caps: FUNC 60 s, CLASS 30 s, LIB 120 s, ALGO_FN 10 s, ALGO_IO 3 × the problem limit per test and 60 s in total;
   - ALGO_IO runs at most 50 tests per attempt (all public + private + a fixed seeded subset of generated tests), all in one guarded process.
4. **PROJ-8 showcase tasks (P0).** The 8 tasks in `research/coding_datasets.md` §10 (to-do CLI, Flask notes API, snake logic, LRU cache, CSV report, Markdown→HTML, URL shortener, bank ledger).
   - DeepSeek-V4-Pro drafts the spec, reference and hidden pytest suite (PROMPTS.md §7.2 adapted to the fixed interface); then review them.
   - Verify: reference 3/3, a stub fails ≥ 90% of the tests, deterministic, headless (Flask `test_client()`), ≤ 10 s, and every import is available in the `exec` environment (Python 3.10).
   - Never use them in train, val or test.
5. **PROJ-SYN (P1).** Generate 300 tasks with PROMPTS.md §7.2 (seeded categories × 3 size bins). Apply the same verification, plus: no forbidden imports, and the spec does not leak the reference (longest common substring < 200 chars). Keep the roughly 200 that pass. **Keep the raw 300 too** for M3 (raw vs curated).
6. **CuratorKIT pass** over the pool:
   - `clean=False`, `dedup="minhash"`, `max_tokens=100_000`;
   - a `preprocessing_fn` rejects tasks flagged in steps 2 and 3, so CuratorKIT's report lists every rejection reason;
   - keep the manifest and provenance.

   Then run our own cross-split near-duplicate check: MinHash on prompts *and* reference-code shingles, Jaccard ≥ 0.7, with each group kept in one split.
7. **Splits** (seed 20261109, deterministic, near-duplicate groups kept whole):
   - CLASS → `test_ood` (all of it). Drop any other task that shares a near-duplicate group with a CLASS task, so the family stays unseen.
   - Other families: 20% `test` (stratified by reference-length quintile within the family), then 10% of the remainder `val`, the rest `train`. Expected sizes: train ≈ 950, val ≈ 105, test ≈ 265, test_ood ≈ 100.
   - PROJ-SYN: 70/10/20.
   - `raw_extra` (M3 only, never evaluated): tasks dropped only by the ground-truth, flakiness or vacuous-test checks, or as near-duplicates within the training side. Leave out anything in a near-duplicate group with a `val`, `test` or `test_ood` task, so M3 cannot leak evaluation tasks into training.
8. Tokenize the rendered prompts (system + task turn) with the official tokenizer, and drop any above 2,500 tokens.
9. Push `<user>/anthropochrony-tasks` (private) with a dataset card listing sources, revisions, per-row licenses, counts and drop reasons. The public release later ships LeetCode and CodeContests as IDs plus a loader.

**Gate G1:**
- at least 1,250 tasks kept, and every family at ≥ 80% of its planned count;
- test ≥ 240, val ≥ 90, test_ood ≥ 90;
- 100% of kept tasks pass GT 3 of 3;
- no near-duplicate group spans two splits.

### S2. Speed calibration and time-model validation

1. **REF card on HF `l40sx1`** (about $0.6).
   - vLLM v0.33.0 image, bf16, single stream.
   - `S_ref`: median decode tok/s at output lengths {256, 512, 1,024}.
   - `P_ref`: median prefill tok/s at prompt lengths {256, 512, 1,024, 2,048}, with prefix caching off.
   - 5 repetitions each; record GPU clocks. The command template is in `research/hf_compute_and_model.md` §7.1.
2. Calibrate the local setups the same way:
   - HF `l40sx1` (vLLM; this is REF);
   - the local CPU with llama.cpp Q4_K_M and Q8_0 (`llama-bench`, fixed thread count);
   - (P1) HF `l40sx1`, for X10.
3. **Time-model validation** (one HF `l40sx1` job, about $1: real agent loop, batch 1, prefix caching on, tests run in line by the in-job runner in a subprocess with an empty environment). Before running any model code, the job deletes the downloaded token file and unsets `HF_TOKEN`; it pushes results with nothing else. Use 100 `val` tasks stratified by family.
   - Record the wall time per call and the run total (from the first request to the end of the last test).
   - Compare with `t(τ; S_local, P_local)` computed from the run's own token counts and `e_j`.
   - Report R² on log time, the median |relative error|, and the bias.
   - Re-run the tests of these runs in the S3 worker pool (8 parallel workers) and report the median ratio of pooled to in-line `e_j`. If it is above 1.10, shrink the pool until it is not.
4. Write `configs/speeds.yaml`:
   - **Training profiles.** Sample one profile per example (seeded), log-uniform within it, and round to 2 significant figures:
     - cpu: S ∈ [2, 25], P/S ∈ [4, 15]
     - edge_gpu: S ∈ [20, 120], P/S ∈ [20, 300]
     - dc_gpu: S ∈ [60, 400], P/S ∈ [50, 300]
     - asic: S ∈ [400, 2,500], P/S ∈ [10, 40]

     **Reject S ∈ [150, 260]**: this band is held out.

     **Coverage check:** REF, the seen-range cards and every measured real card must each lie inside at least one profile (both S and P/S). Single-stream GPUs typically have P/S of about 100-300, which is why edge_gpu reaches 300. If a card falls outside, widen the nearest profile before S5 and record the change.
   - **Evaluation cards:**
     - REF;
     - seen range: (8, 60), (60, 3,000), (400, 40,000);
     - held-out band: (190, 19,000), (230, 23,000);
     - extrapolation: (1.5, 10), (5,000, 100,000);
     - the measured real cards from step 2.

**Gate G2:** median |relative error| ≤ 10% and R²(log) ≥ 0.95.
**Fallback:** fit a per-call latency L on these runs, add `L·n` to the formula, and record it as a deviation **before S4**.

### S3. Base measurement (HF Jobs `l40sx1` for generation, local sandbox for tests)

1. **Pilot first:** 100 tasks (stratified) × K = 8. Check every gate below before the full run.
2. **Full run:** every task in train, val, test, test_ood and `raw_extra` (+ PROJ-SYN, + PROJ-8), K = 8 runs, with the REF card in the system prompt.
3. **Round-synchronous loop.** Round r:
   - Generate attempt r for every active run with the vLLM offline engine: prefix caching on, per-run seed = hash(task_id, k, r), `n = 1` per run.
   - Write the outputs, then run tests in a sandbox pool of 8 workers, each pinned to 1 CPU.
   - Build the feedback (PROMPTS.md §3), then start the next round.
   - Checkpoint after every round; the run must be resumable.
4. Save each trajectory using the `traces` schema (§6). Keep full model texts in `outputs.jsonl`, keyed by text SHA.
5. **Speed-card invariance check:** 100 `val` tasks × K = 2 with the cards (2, 20) and (2,500, 50,000) in the system prompt. Compare their token counts with the REF runs.
6. Push the results to `<user>/anthropochrony-traces` (config `base`).

**Gate G3:**
- harness errors < 1% of executions; ≥ 99% of runs complete;
- `finish_reason == "length"` < 5% of attempts in every family (otherwise raise that family's cap and rerun those tasks);
- invariance check: median |log10 token ratio| ≤ 0.05 (otherwise stop and ask the user, because labels for other cards would then need separate runs);
- pass@1 is reported per family.

**Fallback:** if a family has > 80% of its tasks at 0/8 passes, replace part of ALGO_IO with deduplicated TACO EASY/MEDIUM tasks and cap the hardest stratum (`research/coding_datasets.md` §8.1).

**Compute:** HF `l40sx1`, one job per round (3 rounds), about 0.5-0.8 h of GPU in total plus about 10 min startup per job: about $2-3. Tests run locally between rounds (no token on the execution host). Size `--timeout` at 2× the expected round time.

### S4. Labels and baselines

1. **Labels** (`labels.parquet`) for every (task, card) that is needed:
   - the evaluation cards for val, test and test_ood;
   - 6 sampled training cards per train task (seeded).

   Exclude tasks with harness errors in more than 2 of their K runs (PREREGISTRATION.md §6), and log them.
2. **Base-model estimates** through the AuditKIT callable wrapper (Appendix A.6): local vLLM, greedy, `max_tokens` 512 (1,024 for plan prompts and for the train human-frame answers that become training targets). Record `finish_reason`. A human-frame answer that hits the cap is never used as a training target. Estimators:
   - E0, E1, E2 (5-shot, same family, nearest prompt length, same card), E3;
   - the wording set W (test);
   - the human frame on **all** splits: the train answers become preservation targets, and the test answers are the H6b reference;
   - the plan frame (test + PROJ-8).
3. **Parse** with regex, then the judge (PROMPTS.md §6). Audit 200 answers by hand.
4. **Non-LLM and external baselines:**
   - B0: train median at each card.
   - B1: family median.
   - B2: input-only regressor. HistGradientBoosting on prompt features (prompt tokens, chars, lines, number of functions or classes in the stub, number of public examples, family) plus ridge on `bge-small-en-v1.5` prompt embeddings. It predicts log tokens, attempts and log overhead; the time formula then gives the time. Tune it on `val`.
   - B2+: B2 plus privileged features (reference length) as an upper-bound reference.
   - B4: DeepSeek-V4-Pro once per task (PROMPTS.md §7.1), cached, then the time formula.
   - (P1) B3 probe: base-model hidden states at the last prompt token, layers {6, 12, 18, 24, 30, 36}, ridge predicting the same targets. Select the layer on `val`.
   - N: the noise ceiling, from the leave-one-out median of the K runs.
5. Base results on `val` (development) and `test`. The base model is not fine-tuned, so H1 can be confirmed on `test` now.

**Gate G4:** E0/E1 parse rate ≥ 90%; parser vs hand audit agreement ≥ 90%; H1 table produced.

### S5. SFT datasets (CuratorKIT)

1. **M1 data, per train task:**
   - 6 self-frame examples, each with a distinct sampled training card, a template from q01-q20 and a phrasing from sc01-sc06 (all seeded);
   - 1 human-frame example (h01-h08), whose target is the base model's greedy answer;
   - answers in the canonical format (PROMPTS.md §5).

   Expect about 950 × 7 ≈ 6,650 examples, about 5 M tokens per epoch. Build the `val` examples the same way, but from `val` tasks.
2. **Variants.** Same prompts as M1, so that each differs only in its stated factor:

   | Variant | Difference from M1 |
   |---|---|
   | M2 | Teacher-guessed labels (B4 quantities through the same formula) |
   | M3 (**P0**) | Raw measured labels: no ground-truth, flakiness, vacuous-test or dedup gates, so `train` plus `raw_extra` (plus raw PROJ-SYN once P1 PROJ-SYN exists). This is the curated-vs-raw comparison that problem #8 asks for. |
   | M4 (P1) | No human-frame examples |
   | M5 (P1) | No speed card line; REF labels only (MetaTime-style "SFT without speed") |
   | M7 (P1, only if X3 fails) | Plus plan-frame examples (PROMPTS.md §5) |
   | M6 (P1) | Identical to M1, rendered with the Aya Expanse 8B chat template |

3. **CuratorKIT run per split and variant:**
   - SchemaGate with a high `max_tokens`;
   - **`dedup="exact"` only.** MinHash would merge the 6 speed-card examples of a task, which differ only in their numbers. Task-level dedup already happened in S1.
   - Then write our own JSONL (`sft` schema, §6) from `result.passed`, because the `messages` exporter drops IDs and labels.
4. Drop and log any rendered example above 2,048 tokens, and report the dropped share by family. If any family loses more than 2%, raise `max_seq_length` to 3,072 for every variant and record the change. Push to `<user>/anthropochrony-sft`, one config per variant, with manifests.
5. Automated diff test: for the same `example_id`, M1 and M2 have identical `system`/`user` messages and different assistant content.

**Gate G5:** counts as expected; diff test passes; a 1-step dry run shows that only the assistant answer plus `<|END_RESPONSE|><|END_OF_TURN_TOKEN|>` is unmasked.

### S6. Training (AlignTune on HF Jobs `l40sx1`)

1. **Dry run** (1 step). Run the mask check, and confirm the LoRA target-module names with `print(model)`. Download the base model at the pinned revision with `snapshot_download` and pass that local path as `model_name`, which pins the revision.
2. **Learning-rate sweep** (P0): M1, seed 0, 1 epoch, lr ∈ {1e-4, 2e-4} ($30 cap). Evaluate each run in an `l40sx1` job (in-memory merge), the self-REF `val` set, and pick the lowest MdAE. Freeze the result in `configs/sft/m1.yaml`; every variant reuses it.
3. **Main runs** (P0), launched as parallel jobs:
   - M1 × seeds {0, 1, 2}, M2 × seeds {0, 1, 2} and M3 × seed 0, 2 epochs each;
   - each job trains, then pushes the adapter (private) together with `lexsi_provenance.json`, `trainer_state.json` and the training log. It does **not** push merged weights (storage cap, §4.4). Locally, `make merge RUN=<run>` builds the bf16 merged model from the pinned base and the adapter into `models/<run>/`; record its sha256 in the manifest.
4. **P1 runs:**
   - M5 and M4 (seed 0 each).
   - M1b: compute per-family q10/q90 of M1-s0's log residuals on `val`, rebuild the interval labels as `[T·10^(−q90), T·10^(−q10)]` (widened by the per-task K spread), and retrain seed 0.
   - M6: Aya Expanse 8B, same data, with a 2-point lr check on `val`. Aya Expanse 8B never runs locally (§4.3). Its training job merges inside the job and pushes that merged model privately (the only merged push, about 16 GB). One `l40sx1` vLLM job then runs the `val` check for both learning rates and the S8 estimates for the chosen one (about $1).
5. **Freeze the pre-registration** once M1-s0 passes G6: commit, then `git tag prereg-v1`.

| Hyperparameter | Value |
|---|---|
| LoRA | r 16, α 32, dropout 0.05, targets q, k, v, o, gate, up, down (verify names; never `lm_head` or `embed_tokens`, which are tied) |
| Optimizer | AdamW, lr from the sweep (default 1e-4), cosine schedule, warmup ratio 0.05 |
| Batch | 8 × gradient accumulation 4 = 32 effective |
| Epochs | 2; final checkpoint (no early stopping; pre-registered) |
| Sequence | `max_seq_length` 2,048, `packing=False`, `group_by_length=False` |
| Loss | `assistant_only_loss=True` (TRL swaps in `cohere2_training.jinja`) |
| Precision | bf16, gradient checkpointing |
| Seeds | `seed = data_seed = split_seed = s` |

**Gate G6:**
- the loss falls smoothly;
- M1-s0 `val` MdAE is below that of the best base-prompt estimator (E0-E3), and its parse rate is ≥ 98%;
- the merged model under vLLM matches HF + PEFT greedy outputs on 50 `val` prompts (≤ 2 numeric mismatches allowed).

**Compute:** about 0.4-0.7 h per 2-epoch run and about 0.3-0.4 h per 1-epoch sweep run, at $1.80/h.

### S7. Re-measurement and capability retention (HF Jobs `l40sx1`)

1. Run the S3 protocol (K = 8) on test, test_ood and PROJ-8 for M1-s0/s1/s2 and M2-s0/s1/s2, each in its own `l40sx1` job set (3 round jobs; tests between rounds run locally). The job loads the base model plus the adapter, merges in memory, and serves the merged weights with vLLM (never vLLM LoRA, rule 10); no merged weights are pushed. Run that model's S8.1 estimates in the same jobs. Locally merged weights (`make merge` rebuilds them when S7.3, S8 or S9 needs them). For M3 and the P1 models, use base labels and run a shift check on 100 tasks.
2. **Coding capability.** Compare with base, paired over tasks:
   - pass@1 (first attempt) and pass@3 per family → H6a;
   - behavior shift: per-task |log10 token ratio| and the change in the attempt distribution.
3. **AuditKIT lm-eval** for base, M1-s0 and M2-s0, using the HF backend on the merged model:
   - IFEval (prompt-level strict accuracy);
   - ARC-Challenge (acc_norm, with `dtype="float32"`, because AuditKIT issue #26 computes log_softmax in bf16).
4. (P1) One registered AuditKIT safety scenario (list them with the `ak` registry; prefer harmful-request refusal plus over-refusal), base vs M1-s0.

**Gate G7:** re-measurements are complete, and the capability table is produced.

### S8. Evaluation, statistics, exploratory analyses (local)

1. **Estimates for every estimator** on the evaluation sets in §7.1. Use vLLM in `l40sx1` jobs through the AuditKIT callable, greedy decoding, and `RunConfig(extra={"model_sha", "evalset_sha", "code_sha"})`. M6 estimates come from its `l4x1` job (S6.4), wrapped the same way. Scorers: `dc_abs_log10_err_v1`, `dc_within2x_v1`, `dc_parse_ok_v1`, `dc_human_units_v1`.
2. **Metrics and statistics** in `anthropochrony.eval`, exactly as PREREGISTRATION.md §5-6 specifies. Primary targets:
   - fine-tuned models: their own re-measured runs;
   - base model and non-model baselines: the base runs.

   As a robustness check, also score the fine-tuned models against the base labels. A comparison that includes a model without its own re-measurement (M3 and the P1 models) scores every model in it against the base labels.
3. Test H1-H6 with Holm correction and write `results/hypotheses.json`.
4. **Exploratory analyses**, in P1 value order:

   | ID | Analysis | Details |
   |---|---|---|
   | X10 | Real deployments | HF `l40sx1` vLLM (in-job runner, token removed before model code runs); local CPU llama.cpp Q4_K_M; (P1) local 4060 Ti vLLM. On each setup, measure S and P at the start, ask M1-s0 (and E1) for an estimate in a separate call, run 100 test tasks for real one at a time, and compare with the measured wall-clock. HF runs use stdlib-only families. |
   | X9 | Conditioning vs rescaling | M1 vs M5 + rescale vs E4 across the evaluation cards |
   | X2 | Probe | B3 vs verbal estimates (base and M1), by layer |
   | X4 | Cross-prediction | M6 vs M1, both scored against **base** Tiny Aya test labels, plus input-only B2 |
   | X1 | Calibration | Coverage and Winkler score, M1 vs M1b |
   | X5 | Curation | M3 vs M1 |
   | X6 | Simulations | (a) Shortest-job-first: 1,000 random queues of 50 test tasks; mean completion time relative to the oracle, for each estimator vs FIFO. (b) Estimate-based timeouts and tripwires: stop a run once its elapsed time exceeds k × its own upper estimate, k ∈ {1, 1.5, 2, 3}; report the false-stop rate on runs that would pass and the time saved on runs that would fail. |
   | X7 | In-context estimate | 300 test tasks × K = 2 with M1-s0; ask for an estimate, then "Now do it." in the same conversation, and compare tokens with the normal runs |
   | X3 | Plan frame | Anthropocentric-unit rate (AUR) and MdAE of plan totals, base vs M1 |
   | X8 | First-pass prediction | Brier, AUROC and ECE of PFP |
   | X11 | Wording | Base model under the 8 wordings |
   | X12 | Generic-AI control | Generic-AI frame vs self frame |

5. Build the figures and tables (§7.4) with `make figures`, using **only released data**, on CPU.

**Gate G8:** every P0 table has CIs, and `notebooks/reproduce_results.ipynb` runs end to end on CPU from Hub data.

### S9. Demo Space

1. Convert merged M1-s0 to GGUF (bf16 → Q4_K_M) with the pinned llama.cpp, writing the bf16 intermediate under `TMPDIR` (§4.2), then push `<user>/anthropochrony-tiny-aya-global-GGUF`. Run `llama-bench` and record the results. The GGUF is a release artifact and the X10 CPU model, whatever U4 decides.
2. Build a Gradio app:
   - pick a task (PROJ-8 + 20 test tasks whose base runs take under 60 s at REF) or paste one;
   - show the **base estimate** (precomputed for listed tasks, live for pasted ones) next to the **Anthropochrony estimate** (live) and Anthropochrony's human-developer estimate;
   - **Run it** (listed tasks only): live agent loop with a stopwatch and sandboxed tests, then estimate vs actual;
   - the speed card is measured and shown before each estimate;
   - pasted tasks get an estimate and the code, but no execution.
3. **Backend, by U4:**
   - ZeroGPU (the default without PRO): transformers in bf16 with the M1-s0 adapter, or the merged model if U6 allows. ZeroGPU is Gradio-only and its GPU is shared, so the speed changes between calls. Run the speed measurement, the estimate and the whole run (all attempts and their tests) inside one `@spaces.GPU` call, and time from GPU allocation so queue time stays out of the comparison. Logged-out visitors get 2 GPU minutes a day, which is why listed tasks take under 60 s.
   - CPU Space (PRO, or a repurposed Space): llama.cpp with the GGUF, with speeds from `llama-bench` at startup. Label its numbers "demo only": llama.cpp splits digits differently, and the model is quantized.
   - Static fallback: recorded runs plus instructions for running the GGUF locally.
4. **Security:**
   - no write-capable token in the Space. The GGUF and a public merged model need no token. If the gated base model must be downloaded, ask the user for a separate fine-grained token that can only read `CohereLabs/tiny-aya-global`, and store it as a Space secret;
   - tests run in a subprocess with an empty environment, as an unprivileged user, with rlimits and timeouts;
   - only curated tasks are ever executed.
5. Set a 1 h sleep time where the hardware allows it.

**Gate G9:** from a logged-out browser, the public URL completes one full estimate → run → compare cycle.

### S10. Release, write-up, submission

1. After the user approves, make public:
   - the datasets, each with a dataset card: tasks, traces, sft, eval and logs. For LeetCode and CodeContests rows, release IDs plus a loader instead of statement text, in every dataset that would contain it (tasks, and the user turns in sft and eval), and ship `rebuild_messages.py`;
   - the adapters and GGUF, each with a model card: CC-BY-NC-4.0, AUP link, base model, data, metrics, limitations, intended use.

   Upload merged M1-s0 as a public repo only if U6 confirms it. Otherwise, the adapters and the GGUF are the model artifacts.
2. Create the two collections, then make the GitHub repo public with the reproducibility README (§12.3).
3. Publish the HF article (§12.1) and the abstract; the LaTeX paper is P1.
4. The user submits the form. After submission, change artifacts only to fix errors, and log each change.

**Gate G10:** every item in `docs/HACKATHON_REQUIREMENTS.md` is ticked, and a logged-out check opens every link.

---

## 6. Data schemas

`tasks.jsonl`, one row per task:
```text
task_id, family, source, source_revision, license, split, difficulty, near_dup_group,
task_turn, entry_point, harness, tests_ref, public_examples, reference_solution,
ref_tokens, prompt_tokens, time_limits{per_test_s, per_task_s}, gt{passes, times_s[3]}, flags{}
```

`traces.jsonl`, one row per run (full texts go in `outputs.jsonl`, keyed by `text_sha`):
```text
run_id, task_id, model_id, model_sha, engine{name, version, lock_or_digest}, hardware{gpu, cpu},
card{id, S, P}, sampling{temperature, top_p, max_tokens, seed}, k,
attempts[{j, prompt_tokens_total, u_tokens, o_tokens, finish_reason, text_sha, code_found,
          exec{status: pass|fail|error|timeout|no_code, wall_s, cpu_s, first_failure, feedback_tokens}}],
n_attempts, passed, first_pass, total_o, total_u, total_exec_s, harness_error, git_sha, created_at
```

`labels.parquet`, one row per (task, card):
```text
task_id, split, family, card_id, S, P, T_med_s, T_q10_s, T_q90_s, TOK_med, ATT_med, PFP,
overhead_med_s, n_runs, label_source(base|<model run>), label_version
```

`estimates.jsonl`, one row per (estimator, prompt):
```text
est_id, estimator, model_id, frame, template_id, card_id, task_id, prompt_sha, raw_text, finish_reason,
parsed{point_s, lo_s, hi_s, tokens, attempts, pfp, units_used, has_human_units},
parse_method(regex|judge|none), auditkit_fingerprint
```

`sft.jsonl`, one row per training example:
```text
example_id, task_id, variant(m1..m7), frame, template_id, card_id,
messages[system, user, assistant], label{T, LO, HI, TOK, ATT, PFP}, label_source(measured|teacher|raw), label_version
```

---

## 7. Evaluation protocol (PREREGISTRATION.md is binding)

### 7.1 Evaluation sets (per model)

| Set | Tasks | Cards | Templates | Used for |
|---|---|---|---|---|
| self-REF | test, test_ood | REF | 2 held-out (q21-q28), hash-assigned; held-out card phrasing sc07/sc08 | H2, H3, H5 |
| self-speeds | test | 3 seen + 2 held-out band + 2 extrapolation | 1 held-out | H4, X9 |
| human | test | REF | 1 of h09-h12 | H6b |
| generic | test | REF | 1 of g01-g04 | X12 |
| plan | test + PROJ-8 | REF | 1 of p01-p04 | X3 |
| wording | test (base and M1-s0 only) | REF | w01-w08 | X11 |
| real | 100 test tasks per setup | measured | 1 held-out | X10 |
| val (development only) | val | REF + 2 seen | train templates | lr sweep, G6, M1b widths |

### 7.2 Metrics

All metrics follow PREREGISTRATION.md §5:
- error measures: MdAE in log10 (primary), median |ln ratio|, mean |err|, signed bias, geometric-mean ratio;
- shares: within 2× and within 10×;
- ranking: Spearman ρ, Pearson r on logs, pairwise-ordering accuracy, and accuracy on **counter-intuitive pairs**, where the base model's human-frame answers order the two tasks opposite to their measured times (the "two clocks" test);
- intervals: coverage of the 80% range and the Winkler score;
- token-level MdAE and arithmetic consistency;
- anthropocentric-unit rate (AUR) and parse rate;
- PFP Brier, AUROC and ECE;
- speed elasticity;
- capability: pass@1, pass@3, IFEval, ARC-C.

### 7.3 Statistics

- 10,000-resample task bootstrap. Pooled results resample tasks within each family (a stratified bootstrap); H3 and LOFO, not the pooled CIs, test generalization across families.
- Paired Wilcoxon tests with Holm correction across {H2, H3, H5}.
- 3 seeds for M1 and M2, reported as mean ± SD.
- The noise ceiling is shown in every accuracy table.

### 7.4 Tables and figures (the paper and article draw on these)

| ID | Content |
|---|---|
| T1 | Task pool by family, source, license; split sizes; drops by reason |
| T2 | Main results, `test`, REF card, all estimators with CIs (MdAE, within-2×, Spearman, Pearson log, GM ratio, AUR, parse rate) |
| T3 | `test_ood` (CLASS) results |
| T4 | Speed generalization by card (seen, held-out band, extrapolation, real) |
| T5 | Capability retention (pass@1, pass@3, IFEval, ARC-C, safety if run) and behavior shift |
| T6 | Frames: self, human, generic and plan, with AUR and the human-frame preservation ratio |
| T7 | ROI in the form the rules ask for: a closed-source API model's token billing vs self-hosted Anthropochrony, at a realistic volume (for example 100k estimates per month). Inputs: measured tokens per estimate, measured Anthropochrony tokens/s on `l4x1`, GPU hours, $/h, and the closed model's list prices (cite the page and date). Show accuracy side by side: Anthropochrony (M1) vs the large-model API baseline we measured (B4, DeepSeek-V4-Pro through a provider). Mark the closed model's accuracy "not measured" unless the user provides a key for it. |
| T8 | Noise ceiling, token-level vs time-level accuracy, arithmetic consistency |
| F1 | Hero figure: estimated vs actual (log-log), base E0/E1 vs Anthropochrony, with y = x and 2× bands |
| F2 | Bias by wording (X11) |
| F3 | Speed conditioning across cards: M1 vs M5 + rescale vs E4 |
| F4 | Ablations with CIs: M1, M2, M3, M4, M5, M6, B0-B4, N |
| F5 | Calibration: interval coverage (M1 vs M1b); PFP reliability |
| F6 | Probe by layer vs verbal estimates (X2) |
| F7 | Downstream: shortest-job-first completion time; timeout trade-off curves (X6) |
| F8 | Real deployments: estimate vs wall-clock on each setup (X10) |
| F9 | Human-frame preservation scatter (base vs M1 human answers) |
| F10 | Qualitative: PROJ-8 plan answers before and after, with the measured run time |

---

## 8. Estimators and ablations

| ID | What | Priority | Paid compute |
|---|---|---|---|
| E0-E3, W | Base prompting: none, speed card, 5-shot, generic-AI; wording set | P0 | none (local) |
| E4 | Speed-rescaling rule (MetaTime) | P0 | none |
| B0, B1 | Constant, family median | P0 | none |
| B2, B2+ | Input-only regressor; plus privileged features | P0 | none |
| B4 | Teacher guess (DeepSeek-V4-Pro) | P0 | about $3 |
| N | Noise ceiling | P0 | none |
| M1 ×3 | Anthropochrony SFT | P0 | about $3 |
| M2 ×3 | Teacher-label SFT | P0 | about $3 |
| M3 | Raw labels: curated vs raw (+ raw PROJ-SYN in P1) | P0 | about $1 (+$4.5 for PROJ-SYN, P1) |
| B3 | Hidden-state probe | P1 | none (local) |
| M5 | No speed card + rescaling | P1 | about $1 |
| M6 | Aya Expanse 8B cross-prediction (training plus an `l40sx1` evaluation job) | P1 | about $5 |
| M1b | Conformal intervals | P1 | about $1 |
| M4 | No human-frame data | P1 | about $1 |
| M7 | Plus plan-frame data (if X3 fails) | P1 | about $1 |
| M8 | GRPO from M1 with a log-error + pinball per-sample reward (`research/lexsi_stack_audit.md` §5) | P2 | about $5 |
| M9 | Full 2×2 self/cross design (measure Expanse; train each model on each model's labels) | P2 | about $6 |
| LOFO | Leave-one-family-out over the 4 training families | P2 | about $4 |
| RAFT, temperature sensitivity, harness variant (R = 1) | | P2 | small |

---

## 9. Compute and budget

### 9.1 Paid (HF credits)

| Item | Flavor | Hours | Expected $ | Priority |
|---|---|---|---:|---|
| S0 jobs smoke (cpu-basic check) | cpu-basic | 0.05 | 0.01 | P0 |
| S2 REF calibration + time-model validation + generation smoke | l40sx1 | 1.2 | 2.2 | P0 |
| S3 base measurement (3 round jobs) | l40sx1 | 1.2-1.6 | 2.2-2.9 | P0 |
| S4 + S8 estimates (all estimators, all sets) | l40sx1 | 1-1.5 | 1.8-2.7 | P0 |
| S7 re-measurement + S8 estimates, 6 models in 2 packed jobs | l40sx1 | 1.5-2 | 2.7-3.6 | P0 |
| S6 dry run + lr sweep (2 × 1 epoch) | l40sx1 | 1.0 | 1.8 | P0 |
| S6 M1 ×3 + M2 ×3 (2 epochs) | l40sx1 | 2.4-4.2 | 4.3-7.6 | P0 |
| S6 M3 (raw labels, seed 0) | l40sx1 | 0.5 | 1 | P0 |
| Teacher labels (B4/M2), PROJ-8 drafting, judge | Inference Providers | | 4 | P0 |
| Demo Space | ZeroGPU (free) | | 0 | P0 |
| **P0 subtotal** | | | **about 24-35 before the $30 cap rules below; about 19-24 after them** | |

**Hard budget: $30 total (user decision, 2026-11-09).** This covers everything billed to the account: HF Jobs and Inference Providers. The account holds about $40; the last $10 is not ours to spend. Rules:
- **P0 target ≤ $24, with $6 kept back** for failed or repeated jobs. P1 items run only from money left after P0 results exist, in the §10 value order, and never past $30.
- **Savings that keep the science intact:**
  - lr sweep: 2 points (1e-4, 2e-4), not 3;
  - S7: all 6 fine-tuned models re-measured in **2** packed jobs (3 models each, §4.3.1 item 4), about $3;
  - all S4 estimates in one job, and all S8 estimates in the S7 jobs;
  - demo on ZeroGPU only ($0);
  - teacher (B4/M2) and judge: cache everything, one call per task, cheapest provider for the pinned model; stop at $3.
- **Cut order if the forecast passes $30:** P1 first; then M2 seeds 1-2 (M2 keeps seed 0, and H5 is reported with 1 M2 seed as a logged deviation); then M1 seed 2. Never cut M1 seed 0, M3, re-measurement of M1-s0, or the H1-H6 tests.
- Before each job, add its estimate to the running total in `RUNLOG.md`. If spent + this job's estimate would pass $30, stop and ask the user.
| M4, M5, M1b | l40sx1 | 2 | 3.5 | P1 |
| M6 Expanse 8B + lr check, plus its evaluation job | l40sx1 | 3.2 | 5 | P1 |
| PROJ-SYN generation (300) | Inference Providers | | 4.5 | P1 |
| X10 local 4060 Ti (free; HF l40sx1 is covered by S2.3) | local | | 0 | P1 |
| **P0 + P1** | | | **only what fits under $30** | |
| M8 GRPO, M9, LOFO | a100-large, l40sx1 | 5 | 15 | P2 |
| Contingency (failed or repeated runs) | | | 6 (inside the $30) | |

No HF PRO (user decision).

### 9.2 Free (local machine)

| Work | Approximate wall time |
|---|---|
| S1 checks | 0.5-1 h CPU |
| S2 local calibration and validation | 0.5-1 h |
| S3 tests between rounds | 0.5-1 h CPU |
| S4 parsing, baselines B0-B2 | 0.5 h CPU |
| S7 tests between rounds | 1-1.5 h CPU |
| S8 statistics, figures | 0.5 h CPU |
| X10 local CPU | 3-4 h |

The critical path is about 10-14 h of machine time plus the executor's own work.

---

## 10. Order of work, parallelism and cut lines

**Dependencies:** S0 → S1 → S2 → S3 → S4 → S5 → S6 → S7 → S8 → S10. S9 starts once M1-s0 exists.

**Run in parallel:**
- PROJ-8 drafting and B4 teacher calls run during S1-S3;
- the S6 jobs run in parallel with each other;
- S7 re-measurement overlaps the P1 training jobs;
- the article skeleton and README start at S4.

**Deadline checkpoints** (AoE; they exist only because the deadline is fixed):

| Checkpoint | Must be true | If not |
|---|---|---|
| CP1, 2026-11-11 23:59 | G3 passed (base traces complete) | Drop PROJ-SYN; cap ALGO_IO's hardest stratum; use K = 4 for train tasks |
| CP2, 2026-11-12 23:59 | M1-s0 passes G6; pre-registration frozen | Drop all P1/P2; run M1 and M2 with 1 seed first, then M3, then the other seeds |
| CP3, 2026-11-13 23:59 | All P0 results; demo live; article draft complete | Cut P1 items in reverse value order |
| CP4, 2026-11-14 12:00 | Form submitted (by the user) | Submit what exists; the latest submission counts, so resubmit if improved before the deadline |

**P1 order by value per dollar:** X10 → M5/X9 → B3/X2 → M6/X4 → M1b/X1 → PROJ-SYN (then M3 with raw PROJ-SYN)/X5 → X6 → X7 → M4 → M7.

### 10.1 If shortlisted (Nov 16-22: $200 extra credit, Demo Day on Nov 22)

Demo Day teams are announced by Nov 16. Then:
1. Run the "with more credits" items in this order: M9 (the 2×2 self/cross design), M8 (GRPO, which adds problem #6), then LOFO.
2. Update article section 8 with their before/after results.
3. Rehearse a 5-minute live demo (`docs/HACKATHON_REQUIREMENTS.md` §9).

The same rules still apply. Pre-registered analyses stay fixed, and every new analysis is labeled exploratory.

---

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| No credits, or the token is revoked or expires | U2 before S0; the S0.7 cpu-basic job checks the balance; on any 401/403, stop and ask the user |
| vLLM LoRA crash on Cohere2 | Serve merged models; GRPO uses merged-weight sync |
| Stop-token or eos mismatch distorts token counts | S0.5 checks; explicit `stop_token_ids=[6, 261001]`; count the stop token once |
| The fixed system turn (359 tokens; 510 with the agent card) truncates training examples | `max_seq_length` 2,048, a pre-filter, and the S5.4 drop-share rule |
| CuratorKIT cleaning breaks indentation; MinHash merges speed-card variants | `clean=False`; `exact` dedup for SFT examples |
| AuditKIT replays stale cached results | Unique dirs, content hashes in `extra`, versioned scorer names |
| Tiny Aya solves almost no hard tasks | Pilot in S3; replacement rule (TACO EASY/MEDIUM); labels still valid, since failing runs have well-defined times |
| Heavy-tailed timeouts dominate times | Median labels; record timeouts; report the share of timeouts |
| Fine-tuning changes coding behavior (Blandfort and Pawar) | Re-measure (S7); score against own runs; report the shift |
| Self-fulfilling estimates | Separate-call protocol; never reward agreement; X7 measures the effect |
| "It only learned arithmetic" | Token-level metrics, the overhead term, B2/E4/M5 baselines, arithmetic-consistency metric |
| Speeds are synthetic | Claim "stated speeds"; X10 real deployments |
| REF lies outside the training speed profiles, so "in distribution" (H2) would be false | S2.4 coverage check |
| Model code escapes the sandbox | bwrap with no network, a clean environment and its own temp dir; security tests in S0.6; no tokens on the execution host |
| Disk space (27 GB free on the work drive, 31 GB on `/home`) | §4.2 layout; one merged model at a time; never set `HF_HOME`; G0 disk check |
| Hub private storage (100 GB on a free account) | Push adapters only; rebuild merged weights locally; the only merged push is M6 |
| A 3-attempt run overflows the 8,192-token context | Per-attempt cap min(2,048 (4,096 for PROJ), 8,192 − prompt tokens); `context_exhausted` status; report its share |
| Licenses (CC-BY-NC, ClassEval NC, LeetCode statements, AuditKIT LSAL §1A for organizations) | NC release; IDs plus loader for LeetCode/CodeContests; U6 ticket; individual research use |
| Demo hosting limits (CPU Spaces need PRO; ZeroGPU has daily quotas) | U4: ZeroGPU by default; a CPU Space with PRO or a repurposed Space; a static Space with recorded runs as the last resort |
| No HF Article (no PRO) | The rules accept the GitHub README |
| Deadline | Checkpoints CP1-CP4; P0-first ordering |

---

## 12. Write-up, release and submission

### 12.1 HF article (required sections, in this order)

Write it as `paper/article.md`. It becomes the "Write-up" section of the GitHub README (no PRO, U7), which the rules accept. Rule 5 also requires three things, each under its own subheading:
- the metrics chosen, and why;
- the approach, and the rationale for the datasets, tasks and models;
- how the resulting model is meaningfully better in use.

1. **Problem and approach.** The anthropocentric time bias, the pilot anecdote, Anthropochrony in one figure (F1), and why it matters for safety: honest self-reports, agent planning and budgeting, and oversight through self-estimate tripwires (X6).
2. **Hackathon repos used.** The §3 table: CuratorKIT, AlignTune and AuditKIT, with the features used and where. Mention the three upstream pitfalls we hit and how we handled them.
3. **Models and licenses.**
   - Tiny Aya Global (CC-BY-NC-4.0 + AUP) and Aya Expanse 8B (CC-BY-NC-4.0);
   - teacher DeepSeek-V4-Pro and judge gpt-oss-120b (check each license on its model card), and bge-small (MIT);
   - our adapters and GGUF (CC-BY-NC-4.0).
4. **Datasets and benchmarks and their licenses.** HumanEval+ and MBPP+ (Apache-2.0; MIT and CC-BY-4.0 upstream), ClassEval (CC BY-NC 4.0 data), BigCodeBench (Apache-2.0), LeetCodeDataset (Apache-2.0 tag; statements under LeetCode copyright), CodeContests (CC-BY-4.0), IFEval, ARC-C. Also our released datasets.
5. **Model artifacts collection.** Link, plus a table of the adapters, GGUF and what each is for.
6. **Data artifacts collection.** Link, plus a table: tasks, traces, sft, eval.
7. **Experiments, findings, what else we tested.** T2-T8, F1-F10, all ablations including negative results, and the deviations log.
8. **With more credits.** The M9 2×2 self/cross design, Aya Expanse 32B, GRPO and RAFT, LOFO, a larger hardware matrix for real deployments, multi-file repository tasks. Include before/after metrics from what was run.
9. **Teammates.** Tag them.

The judging rubric (kickoff slides) is write-up clarity /10, practical usefulness /10, code/notebook quality /10, and an overall review /100. Several judges score each submission double-blind, after an organizer validity check (public artifacts, write-up, checklist, no duplicates). Aim for:
- **Clarity:** one hero figure, a results table with CIs, plain language.
- **Usefulness:** the demo, the shortest-job-first and timeout simulations, the plan-frame examples, and real deployments.
- **Code quality:** tests, CI, the CPU-only reproduction notebook, pinned environments.

### 12.2 Abstract (under 500 words; fill in the numbers last)

The structure:
1. The problem (1-2 sentences, crediting the term).
2. The pilot gap.
3. The method (measured multi-attempt self-runs, speed cards, CuratorKIT → AlignTune → AuditKIT).
4. The main results with CIs: MdAE before and after, within-2×, the held-out family, held-out speeds, and real deployments.
5. The controls: teacher labels, cross-model, input-only.
6. What is preserved: pass@1 and human-frame answers.
7. The release.

### 12.3 GitHub README (reproducibility)

The sections:
- **Quick reproduction:** rebuild every table and figure from the released data on CPU: `uv sync && make figures`, or the notebook.
- **Full pipeline:** `make smoke tasks calibrate measure labels baselines sft-data train remeasure eval`, with the hardware, expected wall times and $ cost of each step.
- **Pins and environments.**
- **The sandbox, and its security notes.**
- **Licenses.**
- **Citation:** BibTeX for our work and the key references.
- **Known limitations.**

### 12.4 Paper (P1)

A LaTeX paper (8 pages plus appendix) with the same content, the pre-registration and its deviations as an appendix, and a NeurIPS-style checklist.

---

## Appendix A. Command templates (verify each with `--dry-run` or a smoke run)

**A.1 HF CLI** (done on this machine on 2026-11-09; shown for reproducibility):
```bash
uv tool install huggingface_hub==2.4.0      # provides `hf`; the [cli] extra no longer exists
hf auth login                               # stores the token in ~/.cache/huggingface/token
hf auth whoami                              # isHeSatoshi
hf download CohereLabs/tiny-aya-global config.json generation_config.json tokenizer.json tokenizer_config.json \
  special_tokens_map.json --revision af89d219b53ed9b13b8a4645f9c8028973510324 --local-dir pins/tiny-aya-global
```

**A.2 Local vLLM generation** (round r, sketch):
```python
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

llm = LLM(
    model=MODEL_DIR,
    dtype="bfloat16",
    max_model_len=8192,
    enable_prefix_caching=True,
    gpu_memory_utilization=0.90,
    seed=0,
)
sp = lambda seed, n_prompt: SamplingParams(
    temperature=0.1,
    top_p=0.95,
    seed=seed,
    max_tokens=min(CAP, 8192 - n_prompt),  # §2; end the run if < 256
    stop_token_ids=[6, 261001],
)
outs = llm.generate(
    [TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids],
    [sp(seed, len(ids)) for seed, ids in zip(seeds, prompt_ids)],
)
```

**A.3 HF Jobs** (templates and caveats are in `research/hf_compute_and_model.md` §7):
- REF calibration: §7.1.
- Generation fallback: §7.2. In that template, set `--k 1` per run seed and drop in-job test execution; execution stays local.
- Training: §7.3, and the training script below.

```bash
for s in 0 1 2; do
  hf jobs uv run -d --flavor l40sx1 --timeout 3h -s HF_TOKEN --name anthropochrony-m1-s$s \
    -l project=anthropochrony -l stage=sft -l run=m1-s$s jobs/train_sft.py -- --config configs/sft/m1.yaml --seed $s
done
hf jobs ps -a; hf jobs logs -f <job_id>; hf jobs wait <job_id>
```

**A.4 Sandbox:** the bwrap + systemd scope command in `research/coding_datasets.md` §7.2. The executor writes `solution.py`, `run_tests.py` and `task.json` into a fresh `$WORK` for each execution.

**A.5 AlignTune SFT** (inside `jobs/train_sft.py`; verify the kwargs on 0.1.32 during the dry run):
```python
from huggingface_hub import snapshot_download
from aligntune.core.backend_factory import create_sft_trainer

base = snapshot_download("CohereLabs/tiny-aya-global", revision=REV, local_dir="/tmp/base")
trainer = create_sft_trainer(
    model_name=base,
    dataset_name=DATA_DIR,
    backend="trl",
    output_dir=OUT,
    num_epochs=2,
    batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=LR,
    warmup_ratio=0.05,
    max_seq_length=2048,
    use_peft=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    lora_target_modules=TARGETS,
    assistant_only_loss=True,
    packing=False,
    gradient_checkpointing=True,
    precision="bf16",
    seed=SEED,
    data_seed=SEED,
    split_seed=SEED,
    group_by_length=False,
    load_best_model_at_end=False,
    report_to="none",
    curator_clean=False,
    curator_dedup="none",
)
trainer.train()
trainer.push_to_hub(f"{USER}/anthropochrony-tiny-aya-global-lora-{RUN}", private=True)
# No merged push (100 GB private-storage cap); `make merge RUN=...` rebuilds it locally. Exception: M6 (S6.4).
```

**A.6 AuditKIT callable model** (vLLM inside, exact chat rendering; each sample's `input` is a JSON string with `messages`; `sp` is greedy and uses the same `max_tokens` as the `RunConfig`):
```python
import json, math, auditkit as ak
from auditkit.runspec import RunConfig
from vllm.inputs import TokensPrompt


def vllm_model(llm, tok, sp):
    def generate(inputs: list[str]) -> list[str]:
        ids = [
            tok.apply_chat_template(
                json.loads(x)["messages"],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
            )
            for x in inputs
        ]
        return [
            o.outputs[0].text
            for o in llm.generate([TokensPrompt(prompt_token_ids=i) for i in ids], sp)
        ]

    return generate


@ak.scorer(direction=ak.Direction.MINIMIZE, name="dc_abs_log10_err_v1")
def abs_log10_err(sample, output):
    p = parse(output).point_s
    return 3.0 if p is None else min(3.0, abs(math.log10(p / sample.metadata["T_med_s"])))


res = ak.evaluate(
    samples,
    model=vllm_model(llm, tok, sp),
    scorers=[abs_log10_err, ...],
    config=RunConfig(
        temperature=0.0,
        max_tokens=512,
        seed=0,
        extra={"model_sha": MODEL_SHA, "evalset_sha": SET_SHA, "code_sha": GIT_SHA},
    ),
)
# Corpus metrics (MdAE, Spearman, coverage, ...) come from res.predictions via anthropochrony.eval.metrics.
```

The per-sample cap of 3.0 applies only inside AuditKIT's per-sample mean. Corpus metrics follow the PREREGISTRATION.md rule: a parse failure counts as +∞.

**A.7 GGUF conversion:** `research/hf_compute_and_model.md` §6.

**A.8 Teacher call:** `research/hf_compute_and_model.md` §7.6. Use `deepseek-ai/DeepSeek-V4-Pro:deepinfra` and cache every response.

---

## Appendix B. Definition of done

1. G0-G10 passed. Any deviations are logged in the pre-registration, and the `prereg-v1` tag predates every fine-tuned `test` evaluation.
2. P0 complete:
   - E0-E4, B0-B2, B4, N;
   - M1 ×3 and M2 ×3, re-measured; M3 (curated vs raw);
   - H1-H6 tested;
   - T1-T8 (T7 ROI) and F1-F5, F9, F10.
3. Public: the model collection (adapters + GGUF), the data collection (tasks, traces, sft, eval, logs), the GitHub repo, the write-up with all 9 sections (an HF article, or the README per U7), and a working demo. A logged-out check passes on every link.
4. `make figures` reproduces every reported number from the released data on CPU, and the notebook runs clean.
5. `RUNLOG.md` lists every paid job; the total stays within budget.
6. The user has the abstract (under 500 words) and the form fields ready.
