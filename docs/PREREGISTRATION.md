# Anthropochrony pre-registration

Version: v0.2 draft, 2026-11-09, before any data collection. v0.2 adds the stop tokens and the context-budget rule (§2), moves M3 to P0, and aligns the duplicate rule, bootstrap, parse-failure handling and X10 setups with PLAN.md 1.1.
Freeze rule: commit this file and tag the commit `prereg-v1` before any fine-tuned model is evaluated on the `test` or `test_ood` splits. Development uses `val` only. Any later change goes in the "Deviations" section at the end, with the date and the reason. Never edit the frozen text silently.

## 1. Question

Do small language models know how long a coding task will take *them*? Can fine-tuning on measured self-labels teach a small model to estimate its own completion time, without changing its coding behavior or its answers about human developers?

## 2. Setting

- **Subject model:** `CohereLabs/tiny-aya-global`, revision `af89d219b53ed9b13b8a4645f9c8028973510324`, bf16. It is served with vLLM 0.33.0 (pip-installed locally; Docker image `vllm/vllm-openai:v0.33.0` on HF Jobs).
- **Agent harness:** Python tasks with executable tests and up to R = 3 attempts. After a failed attempt, the test output goes back to the model in a user turn. The run stops when the tests pass or after attempt 3.
- **Sampling while solving tasks:** the model-card values, temperature 0.1 and top_p 0.95, passed explicitly (the model's `generation_config.json` sets none). Generation stops at `<|END_RESPONSE|>` (ID 261001) or `<|END_OF_TURN_TOKEN|>` (ID 6). The new-token cap for attempt j is min(2,048 (4,096 for the PROJ task families), 8,192 − prompt tokens at j). If fewer than 256 tokens remain before an attempt, the run ends as failed (`context_exhausted`), and its time so far counts. K = 8 trajectories are sampled per task.
- **Sampling while eliciting estimates:** greedy (temperature 0) and one answer per prompt. The estimate is asked for in a **separate conversation** from the attempt.
- **Own completion time** of trajectory τ under speed card (S, P):

  t(τ; S, P) = Σ_j ( u_j / P + o_j / S ) + Σ_j e_j

  - j indexes attempts.
  - o_j is the number of output tokens of attempt j, including the stop token.
  - u_j is the number of prompt tokens that are new at call j: the task turn for j = 1 and the feedback turn for j > 1. The fixed preamble and the system turn are assumed to be prefix-cached.
  - e_j is the wall time of the test run (capped at the task timeout).
  - S is the decode speed and P the prefill speed, both in tokens/s, stated in the system prompt.
- **Task label:** for each task and speed card, the median of t over the K trajectories, plus the 10th and 90th percentiles.
- **Reference speed card:** the single-stream decode and prefill speeds measured on HF `l40sx1` (the generation GPU) in stage S2 (PLAN.md).

## 3. Confirmatory hypotheses

Notation: `err = log10(predicted time / measured median time)`, computed per task. `MdAE = median |err|`. The geometric-mean (GM) ratio is `10^mean(err)`.

| ID | Hypothesis | Test | Success criterion |
|---|---|---|---|
| H1 | **Bias.** Base Tiny Aya Global overestimates its own coding time by a large factor and answers in human units, even when its speed is stated. | Base model, estimators E0 (no speed card) and E1 (speed card plus "estimate your own time, not a human's"), reference card, `test` split | Geometric-mean ratio ≥ 10× for both E0 and E1, **and** ≥ 50% of E0 answers use hours, days, weeks or months |
| H2 | **Fix, in distribution.** SFT on measured self-labels (M1) makes the estimates accurate. | M1, 3 seeds, reference card, `test` | MdAE ≤ 0.30 (within 2×) for the mean over seeds, and M1 beats the best base-prompt baseline (E0 to E3) with a paired Wilcoxon p < 0.05 after Holm correction |
| H3 | **Fix, unseen task family.** | M1 on `test_ood` (a whole task family never seen in training) | MdAE ≤ 0.50 and better than the best base-prompt baseline (Holm-corrected p < 0.05) |
| H4 | **Speed conditioning.** Estimates follow the stated speed, including speeds held out of training. | M1, `test`, held-out interpolation card(s) | MdAE ≤ 0.35 on the held-out interpolation card(s), and median per-task speed elasticity in [0.8, 1.2] |
| H5 | **Measured beats guessed.** Training on measured labels beats training on teacher-guessed labels with the same tasks, format and size. | M1 vs M2, 3 seeds each, `test`, reference card | Per-task \|err\| averaged over seeds is lower for M1, with paired Wilcoxon p < 0.05 after Holm correction |
| H6a | **Coding unchanged.** | Fine-tuned (M1 seed 0) vs base own trajectories on `test` | Lower 95% CI bound of the paired difference in pass@1 (first attempt) is > −3 points |
| H6b | **Human-frame answers unchanged.** | M1 vs base, human-frame prompts on `test` | Median \|log10(M1 answer / base answer)\| ≤ 0.15 |

Speed elasticity for a task is the least-squares slope of log(predicted time) against log(measured median time) across the evaluation speed cards. Perfect conditioning gives 1.

Holm correction applies to the family {H2, H3, H5} at α = 0.05. H1, H4 and H6 are judged by their thresholds and reported with CIs.

## 3a. Estimators compared

PLAN.md §8 has the full definitions. Priority: P0 = required for submission, P1 = strongly recommended, P2 = only if budget and time allow.

| ID | Estimator | Priority |
|---|---|---|
| E0 | Base model, no speed card, self-frame question | P0 |
| E1 | Base model, speed card plus "estimate your own time, not a human's" | P0 |
| E2 | E1 plus 5 measured examples from `train` (few-shot) | P0 |
| E3 | Base model, generic-AI frame ("how long would an AI coding assistant take") | P0 |
| E4 | Speed-rescaling rule: E1's reference-card estimate × S_ref/S (MetaTime rule) | P0 |
| B0 | Constant: `train` median time at the same speed card | P0 |
| B1 | Family median: the `train` median time of the task's family at the same speed card | P0 |
| B2 | Input-only regressor (gradient boosting on task features), predicting tokens, attempts and overhead, then the time formula | P0 |
| B2+ | B2 plus privileged features (reference-solution length); an upper-bound reference, not a fair competitor | P0 |
| B3 | Hidden-state probe (ridge on base-model activations at the last prompt token), then the time formula | P1 |
| B4 | Teacher guess (DeepSeek-V4-Pro), then the time formula | P0 |
| N | Noise ceiling: leave-one-out median of the other K−1 runs | P0 |
| M1 | Anthropochrony SFT: measured labels, curated, varied speed cards, human-frame preservation; seeds 0, 1, 2 | P0 |
| M1b | M1 with conformal interval labels (PLAN.md S6.4) | P1 |
| M2 | Same as M1, with teacher-guessed labels; seeds 0, 1, 2 | P0 |
| M3 | Same as M1, with raw (uncurated) measured labels: the S1 quality gates are skipped, but no evaluation task enters training | P0 |
| M4 | Same as M1, without human-frame preservation data | P1 |
| M5 | Same as M1, without speed cards (trained at the reference card only); other speeds by rescaling | P1 |
| M6 | Aya Expanse 8B trained on Tiny Aya's measurements (cross-prediction) | P1 |
| M7 | M1 plus plan-frame training data (only if X3 shows weak transfer) | P1 |
| M8 | GRPO from M1 with a log-error and pinball reward | P2 |
| M9 | Full 2×2 self/cross design (Expanse measured; each model trained on each model's labels) | P2 |

## 4. Exploratory questions (reported as exploratory)

- X1 **Interval calibration:** do the 80% ranges cover 80% of fresh trajectories? Report coverage, mean log-width and the Winkler interval score.
- X2 **Knows but doesn't say:** can a linear probe on base-model hidden states (at the last prompt token) predict the model's own log tokens better than a text-feature regressor, while the base model's verbal estimates stay far off?
- X3 **Plan-frame transfer:** after training only on direct questions, do plans stop using human timelines (day/week units, multi-hour totals)?
- X4 **Self-specificity:** does training on the model's own labels beat training on another model's measured labels (Aya Expanse 8B), when scored against Tiny Aya's own times?
- X5 **Curation:** curated measured labels (M1) vs raw measured labels (M3, no quality gates). P0 priority; reported whatever the direction.
- X6 **Downstream use:** shortest-job-first scheduling and estimate-based timeouts in simulation, using each estimator's predictions.
- X7 **Self-fulfilling estimates:** does stating an estimate in the same conversation change how many tokens the model then writes?
- X8 **Success prediction:** Brier score, AUROC and expected calibration error (ECE) of the stated first-attempt pass chance.
- X9 **Conditioning vs rescaling:** M1 (speed in the prompt) vs M5 plus rescaling and E4, especially on extreme cards where test execution and prefill dominate.
- X10 **Real deployments:** end-to-end wall-clock runs (one at a time) on real setups: HF `l40sx1` with vLLM, the local CPU with llama.cpp Q4_K_M, and (if time allows) the local RTX 4060 Ti with vLLM. The speed card states each setup's measured speed. Estimates are compared with the measured wall-clock. Only these results may be described as "real deployment"; the synthetic speed cards are called "stated speeds".
- X11 **Wording sensitivity of the bias:** the base model (and M1 seed 0) under the 8 wordings in PROMPTS.md §4.5.
- X12 **Generic-AI control:** are "you" answers better than "an AI assistant" answers, before and after training?

## 5. Outcomes and metrics

- **Primary:** MdAE (log10) against the measured median time, on `test` at the reference speed card.
- **Secondary:**
  - Median \|ln ratio\| and Pearson r on log durations, for comparability with MetaTime.
  - Geometric-mean deviation factor `10^mean(|err|)` with a 95% CI, for comparability with AgentTime.
  - Pairwise ordering accuracy over 2,000 random task pairs with a duration ratio ≥ 1.5.
  - Counter-intuitive pair accuracy: the same, restricted to pairs that the base model's human-frame answers order opposite to their measured times.
  - Mean \|err\|.
  - Signed median err (bias).
  - Share within 2× (\|err\| ≤ log10 2).
  - Share within 10×.
  - Spearman ρ between predictions and measured medians.
  - GM ratio (the signed bias as a factor).
  - Anthropocentric-unit rate (share of self-frame answers stating hours, days, weeks or months).
  - Parse rate.
- **Token-level accuracy:** MdAE of the stated output-token count vs the measured median tokens. This is free of hardware and arithmetic effects.
- **Arithmetic consistency:** \|log10(stated time / time implied by stated tokens, attempts and speed card)\|.

## 6. Analysis plan

- **Unit of analysis:** the task. When a task is evaluated with several question templates, average its \|err\| over templates first.
- **Parse failures:** an answer with no extractable estimate gets \|err\| = +∞. This counts against MdAE and against the within-k shares. Mean-based metrics (mean \|err\|, GM ratio, GM deviation factor, Pearson r) and Spearman ρ use parsed answers only, and are reported next to the parse rate. In paired tests, a task where both answers fail to parse has a difference of 0.
- **Parsing:** a regex parser runs first, and an LLM judge handles only the answers the regex cannot parse. The two are audited on 200 random answers labeled by hand, and agreement is reported.
- **Unit conversion:** conservative. 1 day = 8 h, 1 week = 40 h and 1 month = 160 h (working time), which understates overestimates. Calendar conversion (24 h, 168 h, 730 h) is reported in the appendix.
- **Confidence intervals:** 95% percentile bootstrap with 10,000 resamples over tasks. Pooled results resample tasks within each family (a stratified bootstrap), because 4 or 5 families are too few clusters for a cluster bootstrap. H3 tests generalization across families.
- **Paired comparisons:** Wilcoxon signed-rank on per-task \|err\|, plus a bootstrap CI of the difference in MdAE.
- **Seeds:** 3 training seeds (0, 1, 2) for M1 and M2. Report mean ± SD over seeds and per-seed CIs. For paired tests, average each task's \|err\| over seeds.
- **Noise ceiling:** leave-one-out over the K trajectories (the median of 7 trajectories predicting the held-out one). Report it next to every accuracy table.
- **Checkpoint selection:** the final checkpoint after 2 epochs. The learning rate is chosen by `val` MdAE in a 1-epoch sweep (PLAN.md S6.2). Nothing is chosen on `test`.
- **Exclusions, decided before training:**
  - tasks whose reference solution fails its own tests;
  - tasks with harness errors in more than 2 of the K trajectories;
  - tasks whose rendered prompt (system turn plus task turn) exceeds 2,500 tokens. Runs that end as `context_exhausted` are kept, and their share is reported;
  - near duplicates across splits cannot occur: near-duplicate groups (MinHash Jaccard ≥ 0.7 on prompt or reference-code shingles) are assigned whole to one split before any measurement, and a non-CLASS task that shares a group with a CLASS task is dropped.
- **Re-measurement:** fine-tuned models are scored against their **own** re-measured trajectories on `test` and `test_ood` (K = 8). This is the primary analysis. Scores against the base model's trajectories are reported as a robustness check. A comparison that includes a model without its own re-measurement (M3 and the P1 models) scores every model in it against the base trajectories.

## 7. Deviations

- 2026-11-09, before training: the pinned CodeContests valid/test pool has
  11 Python-3 tasks with more than 50 public-plus-private cases (maximum 230).
  Drop those tasks rather than truncate mandatory public/private cases or violate
  the 50-case execution cap. The resulting 206 candidates still face all original
  quality and sample-size gates.
- 2026-11-09, before training: pin the Python-3.10 execution environment to
  NumPy 1.26.4 / SciPy 1.11.4 for benchmark binary compatibility and Flask 3.0.3
  for its Werkzeug 3.0.1 requirement. Install pinned NLTK resources. Ground-truth
  caches include the execution lock and sandbox code hashes. No accuracy gate,
  split rule, outcome, sampling setting, or time formula is changed.
- 2026-11-09, before training, after the user delegated the resolution: E2
  normally uses five same-family, same-card training examples. CLASS has no
  training examples by design, so its E2 prompts instead use the five nearest
  prompt-length examples from the pooled training families at the same card.
  CLASS remains entirely held out. This exception is flagged in every affected
  evaluation row. No CLASS labels or answers enter the demonstrations.
- 2026-11-09, before training: first-pass probability shrinkage for the unseen
  CLASS family uses the pooled training prior, never an evaluation-family prior.
  Re-measured fine-tuned test labels reuse frozen base train-only priors, because
  no fine-tuned training trajectories are collected.
- 2026-11-09, before training, explicitly approved by the user: HF Jobs
  prohibits nested namespaces and cgroup delegation. S2 therefore validates
  the actual S3 topology: HF L40S generation over an authenticated HF channel,
  with candidate execution only in the existing credential-free local
  bwrap/systemd sandbox. Observe each complete sequential loop from its first
  request to its last test, including transport overhead. Keep 100 stratified
  validation tasks, the 10% median relative-error gate, and log-time R² >= 0.95.
  The previously specified per-call-latency fallback remains subject to
  measured evidence and a separately logged adoption before S4.
- 2026-11-09, before training: choose production batching by measured throughput
  on the same 800-request, K=8 workload, with zero KV-cache preemptions and GPU
  telemetry. Do not change dtype, temperature, top_p, seeds, context or output
  caps. Calibration and S2 remain single-stream. For uncached prefill
  calibration in a warm serving engine, clear the prefix cache before every
  prefill sample; fail if the engine cannot confirm the reset.
- 2026-11-09, before training: clarify the general seed-reporting rule for H4
  and H6b. Average question templates within each task and seed; H4 uses one
  geometric-mean prediction per card for its log-log slope. Compute the task
  median separately for each training seed, then report its mean and SD across
  seeds, with task-stratified CIs using the same resampled task IDs for all seeds.
  H4 and H6b thresholds apply to that mean, not a pooled median that can hide
  a bad seed. Human comparisons require identical prompt messages and complete
  matched coverage; unparseable reference answers are not silently dropped.
  H6a remains explicitly M1 seed 0. All numerical thresholds stay unchanged.
- 2026-11-09, before training and before S4, after measured evidence: adopt the
  plan's S2 per-call-latency fallback. On 100 real sequential validation loops
  (HF job 6af1a64dab34849891a7cb8b, REF S=96, P=27000) the raw formula missed G2:
  median absolute relative error 0.3608 (limit 0.10), log-time R2 0.4330
  (minimum 0.95); the eight-worker pooled/inline execution ratio, 1.009, passed.
  The sole time formula becomes `sum over model calls (u/P + o/S + exec_wall + L)`
  with L = 0.967 s per call (the median call residual, 3 significant figures).
  It is applied through the single `PER_CALL_LATENCY_S` constant to every label,
  noise ceiling, baseline and teacher target; aggregate predictors use their
  predicted attempt count as the number of calls. Fit and evaluation use the same
  100 loops, so the check also holds out data: 200 random half-splits give worst
  held-out median error 0.0344 and worst log-time R2 0.9965 (fitted L 0.947 to
  0.992 s), and leave-one-family-out median errors are 0.016 to 0.046. Corrected
  metrics on all 100 loops: median error 0.0232, log-time R2 0.9977. The 10% and
  0.95 thresholds are unchanged; the failed raw attempt is kept in
  `results/gates/G2-raw-attempt.json`. Estimator prompts do not mention L, and a
  model's own stated arithmetic is still scored against the bare token and
  execution formula. Labels carry `label_version` `v2-latency`. No measurement,
  label, or training run existed before this change.
- 2026-11-09, before training and before any research measurement, after the
  first REF pilot: apply the plan's S3 ALGO_IO fallback. On 100 stratified tasks
  x K=8 the base model passed 0 of 13 ALGO_IO tasks (LIB 24/26 and ALGO_FN 21/24
  tasks also had no passing run; the plan prescribes a remedy only for ALGO_IO).
  ALGO_IO now holds all CodeContests tasks rated below 2000 or unrated, a
  stratified cap of 30 for rating >= 2000, and up to 60 EASY plus 60 MEDIUM
  stdin-style `BAAI/TACO` test tasks (revision d593ed0a2becbbc952230bb89be09189bf1056dc,
  Python-3 reference, at most 50 tests). TACO tasks sharing a near-duplicate
  group (Jaccard >= 0.7) with a task from another source are dropped. G1 was
  rebuilt and passed (ALGO_IO 203 kept; FUNC 385, LIB 338, ALGO_FN 312, CLASS 93;
  train 869, val 114, test 255, test_ood 93). The first pilot and invariance
  data are archived under `data/superseded/` and the pilot is repeated.
- 2026-11-09, before any research measurement: the uncached prompt tokens of a
  retry are the new prompt minus the reusable prefix, which is the smaller of
  (previous prompt + sampled output) and the re-rendered previous turn without
  its closing end-of-turn token. This equals the sampled-count formula whenever
  re-tokenization is canonical (654 of 656 stored pilot retries; the other two
  differ by +1 token) and fixes length-capped, degenerate text that re-tokenizes
  shorter than it was sampled (previously a negative count and a stopped run).
- 2026-11-09, known and unchanged: the sandbox's one-line timing marker stays in
  the failing-test excerpt returned as retry feedback, in the S2 validation loops,
  the pilots and the planned full run alike, so the formula and labels were
  calibrated on the same prompts.
- 2026-11-09, before training and before any research measurement, applying the
  plan's G3 remedy: in the repeated REF pilot, 21 of 350 ALGO_IO attempts (6.0%,
  from 5 tasks, mostly repetition loops) ended at the 2,048-token cap, above the
  5% limit (all other families below 1.5%). The ALGO_IO output cap is raised to
  4,096 tokens (the existing PROJ cap), still limited by the 8,192 context. All
  other settings, families and caps are unchanged. The pilot and the invariance
  check are repeated under the new cap.
- 2026-11-10, before training and before any fine-tuned evaluation, after the
  first base-model estimate run (job 6af256a29737d9a9e08dfcb6, all splits,
  unprefixed prompts; the outputs, including the `test` split, were read before
  this change and are kept under `data/superseded/base-estimates-unprefixed/`).
  On `test`, the zero-shot base prompts E0 and E1 (self frame) answered with a
  code block instead of an estimate in about half of the replies (parse rate
  0.46 and 0.51 under the old parser), and the old parser then read numbers out
  of the code. Gate G4 (parse rate >= 90%) therefore could not pass. Two
  instrument changes, applied to every base estimate. They were chosen after
  seeing the old-parser H1 summary (geometric-mean ratios 0.98 for E0 and 0.16
  for E1, mostly computed from numbers inside code), not from any accuracy
  comparison between estimators:
  (1) The parser reads prose only. Fenced code blocks are removed, and a
  non-plan answer is its first paragraph that contains a duration. Plan-frame
  answers keep the step-summing rule. Re-parsing the old replies this way gives
  parse rates of 0.17 (E0) and 0.27 (E1), so the old rates were mostly parser
  artifacts, not estimates.
  (2) The zero-shot base estimators (E0, E1, E3, the wording set and all
  human-frame prompts) now start the model's reply with the fixed text
  `Time estimate:`, and the model continues from there. The number and unit stay
  entirely the model's. The plan frame, E2 and all fine-tuned evaluations use no
  prefix. The prefix favors the base model, so it can only make the base
  comparison conservative. Human-frame preservation targets are the prefix plus
  the base continuation. The unprefixed results stay reported as the
  "no-prefill" finding. No threshold, split, label or time formula changes.

- 2026-11-10, before training: the S5 family-wise sequence-loss rule triggered on matched M1/M2/M3 data at 2048 tokens. Rebuild every variant with max_seq_length=3072, retain matched common example IDs, and record both initial and final per-family drop shares in data/sft/matched-summary.json.
- 2026-11-10, before training: seven base human-frame answers hit the 1,024-token
  cap (train 6, raw_extra 4 across ALGO_IO, FUNC and CLASS; 0 in val; 0.5% of all
  tasks). The plan says a capped answer is never a target. Those tasks therefore
  keep their six self-frame examples and lose only the human-frame example, in
  every variant alike (listed in `data/eval/preservation-blockers.jsonl` and in
  each variant's `dropped.jsonl`). Collection still stops if capped or incomplete
  answers exceed 2% of tasks. The 3,072-token rule is driven by sequence-length
  drops only, not by these.
- 2026-11-10, before training: the S5 diff test asked for different M1 and M2 assistant
  content in every self-frame example. Two of 5,898 coincide exactly (HumanEval/23,
  where the teacher and the measurement both give about 45 output tokens and one
  attempt). The test now requires at least 99.9% to differ and lists the identical
  example IDs in the G5 evidence. Its purpose, that M1 and M2 differ only in labels,
  is unchanged.
- 2026-11-10, before training: the 262,144-token vocabulary makes micro-batch 8 at
  up to 3,072 tokens exceed the L40S's 44 GB (first dry run OOM). Training uses
  micro-batch 2 with gradient accumulation 16, the same effective batch of 32.
- 2026-11-10, before any fine-tuned test run: G6 is assembled from validation estimates
  of the fine-tuned adapter, but the estimate runner required G6 for every fine-tuned
  run, which made the gate circular. The runner now requires G6 only when a prompt
  file contains a split other than `val`; the preregistration-tag check for `test` and
  `test_ood` is unchanged. The LR sweep and the M1-s0 validation estimates use this
  path.
- 2026-11-10, before any fine-tuned test run (G6 parity criterion): the first M1-s0 G6
  attempt failed only the "at most 2 numeric mismatches between merged serving and
  HF + PEFT greedy outputs on 50 `val` prompts" check (6 of 50 differed; 26 of 50 texts
  identical; one outlier 60.0 vs 424.3 min; the others differed in a digit, e.g. 8.5 vs
  8.4). All other G6 checks passed. A teacher-forced fp32 CPU analysis of the 8 prompts
  (the 6 mismatching ones plus 2 matching ones; `results/g6-cpu-parity-m1-s0.json`) shows
  that the fp32 merge reproduces the PEFT logits to within 7e-5, while rounding the merged
  weights to bf16 moves logits by up to 1.2. In every mismatching prompt that bf16
  perturbation exceeds the smallest top-2 logit margin at a digit position (0.009 to 0.22),
  so the divergence is a bf16 near-tie flip and not a merge defect. The 2-mismatch rule is
  kept as reported evidence (`preregistered_parity_criterion_met: false`), and G6 now
  requires (a) fp32 merge logits within 1e-3 of PEFT on the checked prompts and (b) every
  bf16 divergence explained by a top-2 margin below the measured bf16 logit perturbation.
  This was chosen after seeing the validation parity result and before any test data was
  generated for a fine-tuned model; all fine-tuned numbers in this study come from the
  bf16 merged serving path, so the deviation does not change what is measured. Readers
  should treat it as a weakening of the original criterion.
- 2026-11-10, before any fine-tuned test run (training seeds): the $30 hard cap leaves
  about $7 after the completed runs. One main training run costs $2.2 to $2.7 and each
  fine-tuned model needs a merged re-measurement on `test` and `test_ood`, so seeds 1 and
  2 of M1 and M2 cannot be funded. Following the plan's cut order (M2 seeds 1 and 2,
  then M1 seeds 1 and 2), the confirmatory comparison uses **M1 seed 0, M2 seed 0 and M3
  seed 0 only**. H2 and H5 use the same per-task paired Wilcoxon tests, with one seed in
  place of the seed average; the task-level bootstrap is unchanged. Seed-to-seed variance
  is therefore not estimated, and the results cannot rule out that a different seed
  gives a different ranking. No threshold changes. The three seed-0 runs and the M1 seed
  0 re-measurement, which the plan marks as never cut, are all present.
- 2026-11-10: three smaller departures, all fixed before test data was touched.
  (1) The G4 hand audit used 60 rows (58 correct), not 200. (2) The IFEval capability
  check uses the first 200 prompts with batch 16, because the full 541 prompts did not
  finish within the 50-minute job limit; ARC-Challenge is full. (3) The first base
  lm-eval job timed out and was repeated with these settings ($1.47 spent on the failed
  attempt).

- 2026-11-10, after the confirmatory results (reporting notes, no hypothesis, threshold or
  split changed). (1) E4 (MetaTime speed-rescaling rule) is derived offline from the stored
  E1 reference-card estimates (`anthropochrony derive-e4`); no extra model call was made.
  (2) Aggregation over training seeds is code-wise unchanged, but with one seed the
  reported SD is null and the per-seed CI is the task bootstrap only. (3) Exploratory
  questions not run: X2 (hidden-state probe, B3), X4/M6/M9 (cross-model), X7, X8, X9
  (M5 rescaling comparison), and the variants M4, M5, M7, M8. X11 (wordings) was run for
  the base model and M1 seed 0 only; X12 (generic-AI control) is reported through the
  generic frame in `results/extra-analyses.json`. X10 was run only on the local CPU with
  llama.cpp (Q4_K_M, 100 held-out tasks); some of its rows ran while other local sandbox
  jobs were using CPU time, so wall-clock values include that contention. (4) M3 is
  scored against its own re-measured trajectories like M1 and M2. (5) Credits ran out
  for seeds 1 and 2 and for the P1 and P2 variants; none were attempted.
- 2026-11-11, post-confirmatory offline analyses and forensic audit (reported separately in
  `results/extra-analyses.json` and `results/simulations.json`; all frozen confirmatory
  results in `results/metrics.json` and `results/hypotheses.json` remain untouched).
  (1) Computed already-collected secondary and exploratory diagnostics on stored outputs:
  `X1` fresh-trajectory interval coverage and Winkler scores, `X8` first-pass probability
  calibration (Brier, AUROC, ECE), token-level MdAE and verbal arithmetic consistency,
  pairwise and counter-intuitive ordering accuracy, `B2+` privileged reference-length upper
  bound, and the split-half trajectory noise ceiling (`0.04` log10).
  (2) Evaluated two zero-retraining inference/calibration post-processors fitted strictly on
  `train` and `val`: `M1-Struct` / `E2-Struct` / `M2-Struct` (evaluates the serving formula
  on the model's already-emitted token and attempt breakdown using `train`-only `exec_s` and
  retry-feedback medians in `results/workload-priors.json`, with `L = 0` on local `llama.cpp`
  CPU runs) and `M1b` / `M1-Struct+CP` (split-conformal multiplicative interval calibration
  on `val`, `n = 114`, finite-sample level `0.8070`). Added multi-tier hardware SLA routing
  (`seen-edge` vs `seen-dc`) to `results/simulations.json`.
  (3) Forensic audit of historical `M3` (`m3_curation_audit`): `raw_extra` contained 134
  tasks (`+894` valid rows over `M1`/`M2`) plus 6 `ClassEval` tasks (`ClassEval_17, 44, 45,
  48, 69, 93` = 40 SFT rows) that entered historical `M3` training when holdout-family
  filtering skipped `raw_extra`; in addition, `results/metrics.json` scored `M3-S0` against
  `base` trajectories (`0.2045` `test`, `0.2388` `test_ood`), whereas matched all-`own`
  scoring gives `0.2099` (`test`) and `0.2354` (`test_ood`). Pipeline and curation code now
  block holdout families and cross-split near-duplicates in `raw_extra`, and both frozen and
  matched-`own` numbers are reported.


- 2026-11-11, the user approved $5 of additional HF compute (cap $30 -> $35) for a supplementary, exploratory cross-model replication (`CohereLabs/aya-expanse-8b`, single-attempt agent target) and a fresh-task CPU generalization test. These runs are outside the pre-registered hypotheses H1-H6, use no pre-registered gate, and are reported as exploratory in `paper/article.md` and the README. Single jobs stay under $5.
- 2026-11-11, exploratory cross-model and fresh-task replication, run after all pre-registered analyses and not part of H1-H6. Corrects the entry above: the Aya Expanse 8B arm used the same 3-attempt agent protocol as the main study, with K=4 (not 8) trajectories per task on a subset (330 stratified `train` tasks, all 255 `test`, all 93 `CLASS` OOD; 2,712 runs), the `REF` card and the same time formula, and one LoRA (r16, alpha 32, lr 2e-4, 2 epochs, 1,980 examples, no human-frame rows) trained with a plain PyTorch/PEFT loop instead of AlignTune (the Aya Expanse chat template has no generation markers). Estimates cover the REF, seen-cpu, extrap-low and heldout-190 cards. A second test ran `tiny-aya-global` base and M1-S0 (Q4_K_M, desktop CPU) on 30 newly written tasks outside the pool (`data/fresh/`). Nothing was tuned on test data, and every number is reported whether or not it favours the method. New code lives in `src/anthropochrony/crossmodel/`, `jobs/xm_*.py`, `scripts/x11_crossmodel.py`, `scripts/x12_fresh_summary.py`; the pinned tiny-aya pipeline is unchanged. Spend $30.2 of the $35 cap (conservative bound, not an invoice).
- 2026-11-14, exploratory X10 GPU arm (local hardware, no paid compute), run after all pre-registered analyses and not part of H1-H6. The same 100 X10 `test` tasks (same sampling seed), the same Q4_K_M GGUFs (SHA-256 checked against the public release manifest) and the same harness as the CPU arm were run on the local RTX 4060 Ti with every layer offloaded (`llama.cpp` tag `b12110`, official CUDA 12.8 build, `-ngl 99`), not with vLLM as the X10 plan listed, so the two arms differ only in where the model runs. The measured speed card was S = 100, P = 7,400 tok/s (CPU arm: S = 20, P = 200). Scoring reuses the CPU arm's `M1-Struct` (`L = 0`) and train-only priors unchanged. A paired check of whether each estimator predicts the per-task CPU-to-GPU speedup was added after the CPU results were known. Results are in `results/extra-analyses.json` (`x10_gpu_deployment_struct`, `x10_hardware_tracking`). The first rows ran while unit tests and a package build were using the same CPU, so their wall-clock values may include that contention.
- 2026-11-14, exploratory head-to-head on a public benchmark (local GPU only; not part of H1-H6). Written and committed **before any estimate of ours on the benchmark was produced**; at this point only timing runs of the base model had started, and no estimate had been asked for. Benchmark: the 68 single-turn tasks of the public MetaTime artifact (`anonymous.4open.science/r/Time-Estimation-B286`, snapshot 8 Oct 2026; the Garikaparthi 2026 suite), with the artifact's prospective question verbatim and the task's short description as the only task text. We do not redistribute the tasks. Target: the model's own wall-clock time for the task prompt on this machine (RTX 4060 Ti, PyTorch/transformers bf16, one request at a time, 2,048-token cap, model-card sampling), median of 5 runs. Training data: the model's own median response length (3 runs) on a pool of public instruction prompts (Dolly-15k, CodeAlpaca-20k, no_robots, GSM8K, 100 of our own `train` coding tasks; pinned revisions in `scripts/metatime/build_pool.py`), from which every prompt with TF-IDF cosine >= 0.25 to any benchmark prompt or description was removed (21 of 1,544). No benchmark task, response or duration is used for training, model selection or calibration. **Primary estimator:** a new LoRA (`G1`, seed 0; M1 recipe: r16, alpha 32, lr 2e-4, 2 epochs) answering the benchmark question with the stated speed of this deployment in the system prompt (held-out phrasing `sc07`), scored on the time it states, parsed by the benchmark's own parser, against the runs of the same fine-tuned model. **Metrics:** the benchmark's (median estimate/actual ratio, median absolute log error, Pearson r on log durations), plus within-2x and MAE. **Comparison:** the artifact's per-task prospective estimates of GPT-5.5 and Claude Opus 4.8 on the same 68 tasks (each model against its own measured time), with a paired task bootstrap (5,000 draws) of the difference in median absolute log error; and the best trained Qwen3-8B numbers of the paper's Table 22 (before generating: MAE 24.04 s, r 0.44). **Declared success criterion:** lower median absolute log error than both frontier models with the paired 95% interval excluding zero; the correlation is reported whether or not it is higher. **Secondary, all reported:** base-model executor, full-prompt view, the token reading (stated tokens through the speed formula), the paper's question wording, the untrained base model with and without the stated speed, pool-only static baselines (constant, text regressor), and more seeds if time allows. The comparison is across models and deployments by design (each model predicts its own time), so it shows which system knows its own time better, not which model is more capable.
- 2026-11-14, amendment to the head-to-head above, written and committed **before label generation, training or any estimate**. (1) The stated speed is rounded to two significant digits, as every stated speed in training is. (2) Training adds replay examples: each pool prompt alone, with the first 128 tokens of the base model's own response (first label run) as the target. Reason: the primary score is against runs of the fine-tuned model, so the adapter must leave ordinary task behaviour, and with it the response lengths the labels were measured on, unchanged. Whether response length drifts is checked on held-out pool prompts before any benchmark run of the fine-tuned model. (3) A linear probe on the untrained model's hidden state (layer and penalty chosen on held-out pool prompts) joins the secondary baselines. (4) Added while the training labels were still being generated, before training: two more secondary rows. A frozen sentence-encoder regressor (the project's B2 family) joins the pool-only baselines, and the same adapter is also read as the median of 16 answers sampled at temperature 1, with the share of measured times inside the 10th-90th percentile range of those answers. The primary estimator stays the single greedy answer. (5) Added while the adapter was still training, before any estimate of ours on the benchmark: one more secondary row. The untrained model with the stated speed is also shown five measured examples from the training pool as earlier turns: the benchmark question about the training prompts at the 50th, 10th, 90th, 30th and 70th percentile of measured response length (in that order), each answered as the training data would answer at the stated speed. It tests whether training is needed or measured examples in the prompt are enough.
