# Head-to-head on a public benchmark (MetaTime)

The first study of this project ([`legacy/`](../legacy/README.md)) measures self-time
estimates on its own coding tasks. This page answers a different question: on a benchmark we
did not build, against the best published numbers, does the method still work?

**Short answer.** Yes, on the measure we declared in advance. On all 68 single-turn tasks,
Tiny Aya (3.35B) with one new adapter estimates its own running time with a median absolute
log error of 0.38, which is a typical factor of 1.46x, and 79% of its estimates are within
2x. The published estimates of GPT-5.5 and Claude Opus 4.8 for their own time on the same
tasks have errors of 1.24 (3.47x, 12% within 2x) and 1.06 (2.87x, 25% within 2x). Ours is
closer on 62 and 61 of the 68 tasks. It is not better at ranking tasks than Claude Opus 4.8
(correlation 0.78 against 0.81), and it needs the machine's speed stated in the prompt. A
second adapter, trained for this one machine and asked with no speed at all, exactly as the
frontier models were, scores 0.42 (76% within 2x), or 0.40 (82% within 2x) against its own
timed runs. Two more training seeds of the first adapter score 0.43 and 0.50, so the primary
seed is the best of three; each of the three is closer than both frontier models, with
paired intervals that exclude zero. The section "What this does not show" lists every
caveat we know of.

![Head-to-head on MetaTime](../paper/figures/F9-metatime-head-to-head.png)

## The benchmark

MetaTime (anonymous ICLR 2027 submission, OpenReview `CAQeizWcB2`) is the largest published
study of whether models know how long their own work takes. Its public artifact
(`https://anonymous.4open.science/r/Time-Estimation-B286`) contains 68 single-turn tasks
(the suite of Garikaparthi et al., arXiv 2604.00010: code, debugging, summaries, reasoning,
writing, creative, Q&A) and, for GPT-5.5 and Claude Opus 4.8, each task's measured time and
the model's estimate before starting.

We do not redistribute the tasks. The scripts download the artifact at run time and the
committed records contain task ids, numbers and our model's answers, but no task text.

## Protocol

Everything below was written into `legacy/docs/PREREGISTRATION.md` (section 7, entries of
2026-11-14) and pushed before any estimate of ours existed on the benchmark. The whole
head-to-head came after the first study, so within this project it is exploratory work with
a protocol fixed in advance, not part of the original registered plan.

| Item | Choice |
|---|---|
| Tasks | All 68 single-turn tasks of the artifact |
| Question | The artifact's own prospective question, word for word, with the task's short description as the only task text |
| What is predicted | The model's own wall-clock time for the full task prompt: one request at a time on an RTX 4060 Ti, PyTorch bf16, 2,048-token cap, model-card sampling; median of 5 runs |
| Training data | The model's own response lengths (median of 3 runs) on 1,523 public instruction prompts (Dolly-15k, CodeAlpaca-20k, no_robots, GSM8K, 100 of our `train` coding tasks); 1,353 for training and 170 held out. Every prompt with TF-IDF cosine of 0.25 or more to any benchmark prompt or description was removed (21 prompts). Training also contains each pool prompt with the start of the model's own ordinary response, so that the adapter leaves ordinary responses, and with them the lengths being predicted, unchanged |
| Never used | No benchmark task, response or duration for training, model selection or calibration |
| Primary estimator | `G1`: one LoRA on Tiny Aya (rank 16, alpha 32, lr 2e-4, 2 epochs, seed 0). It answers the benchmark question with the measured speed of the deployment in the system prompt. The score uses the time it states, read by the benchmark's own parser, against runs of the same fine-tuned model |
| Metrics | The benchmark's: median estimate/actual ratio, median absolute log error (natural log), Pearson r on log durations; plus the share within 2x and mean absolute error |
| Comparison | The artifact's per-task estimates of GPT-5.5 and Claude Opus 4.8 (each against its own measured time), paired bootstrap over tasks (5,000 draws); and the paper's best trained open model (Qwen3-8B, Table 22) |
| Success criterion | Lower median absolute log error than both frontier models, with the paired 95% interval excluding zero |
| Declared secondary rows | The untrained model with and without the stated speed, and with five measured training examples in the prompt; three predictors that never ask the model (a constant, two text regressors); a linear probe on the untrained model's hidden state; `G1` shown the full prompt, asked with the paper's wording, asked without the speed, read through its stated token count, read as the median of 16 sampled answers, and scored against the untrained model's runs; the same question with four other machines' speeds stated; more seeds and smaller training sets |
| Added after the first results (post-hoc; each written down and pushed before it was run) | A range calibrated on held-out training-pool prompts; `G1-fixed`, an adapter that needs no stated speed, asked exactly as the frontier models were |

## How to read the comparison

Each system predicts its own time, so the comparison is across models and deployments by
design. It shows which system knows its own running time better. It does not show which
model is more capable.

Three things differ between our row and the published frontier rows, and the tables report a
control for each:

1. **We state the deployment's speed in the prompt.** The frontier models were asked
   without it. The row "untrained, speed stated" shows what the stated speed alone does, the
   row "G1, no speed stated" shows what the adapter does without it, and `G1-fixed` is an
   adapter trained to need no stated speed, asked exactly as the frontier models were.
2. **We trained on this model's own measurements.** The rows "same constant for every
   task", the text regressors and the hidden-state probe use the same training labels
   without asking the model, and show how much of the result is calibration of the average
   and how much is telling tasks apart.
3. **Durations are on different scales.** Mean absolute error in seconds depends on how
   fast a deployment is, so the ratio, the log error and the correlation are the fair
   comparisons. Seconds are reported for completeness.

"Typical factor off" is `e` raised to the median absolute log error: 1.0x is perfect and
2.0x means the typical estimate is half or double the measured time.

## Results

### The declared comparison

| System | Median estimate/actual | Median abs. log error [95% CI] | Typical factor off | r (log) | Within 2x | MAE (s) | Parsed |
|---|---|---|---|---|---|---|---|
| GPT-5.5 (published) | 3.47x | 1.24 [1.10, 1.42] | 3.47x | 0.55 | 12% | 92.2 | 68/68 |
| Claude Opus 4.8 (published) | 2.87x | 1.06 [0.90, 1.14] | 2.87x | 0.81 | 25% | 30.5 | 68/68 |
| Tiny Aya, untrained, speed stated | 5.39x | 1.71 [1.59, 1.90] | 5.53x | 0.41 | 4% | 70.9 | 68/68 |
| **Tiny Aya + G1 (primary)** | 0.74x | 0.38 [0.32, 0.45] | 1.46x | 0.78 | 79% | 6.6 | 68/68 |

| Comparison (ours minus theirs, median abs. log error) | Difference | Paired 95% CI | Tasks where ours is closer |
|---|---|---|---|
| **Tiny Aya + G1 (primary)** vs GPT-5.5 | -0.87 | [-1.06, -0.70] | 62/68 |
| **Tiny Aya + G1 (primary)** vs Claude Opus 4.8 | -0.68 | [-0.79, -0.51] | 61/68 |

A negative difference means our error is lower. Both paired intervals exclude zero, so the
success criterion declared in advance is met. `G1` is also closer in each of the seven task
categories:

| Median abs. log error by task category | code_generation (12) | creative (8) | debugging (8) | qa (8) | reasoning (12) | summarization (10) | writing (10) |
|---|---|---|---|---|---|---|---|
| GPT-5.5 (published) | 2.27 | 1.20 | 1.42 | 0.80 | 1.06 | 1.04 | 1.54 |
| Claude Opus 4.8 (published) | 1.17 | 0.91 | 1.21 | 0.53 | 0.99 | 1.17 | 1.09 |
| Tiny Aya + G1 | 0.62 | 0.28 | 0.47 | 0.45 | 0.33 | 0.36 | 0.28 |
| G1 predicting the untrained model's runs | 0.36 | 0.33 | 0.42 | 0.40 | 0.47 | 0.39 | 0.31 |

Category sizes are 8 to 12 tasks, so the per-category numbers are descriptive only.

### Asked as the frontier models were: no speed in the prompt

The frontier models got the question and nothing else. `G1` is also told its machine's
speed, and without it scores 1.77. So we trained one more adapter to find out whether the
stated speed can be dropped. `G1-fixed` has the same recipe, training prompts, labels and
seed as `G1`, but its training examples have no system prompt, and every training answer is
computed at one speed: the untrained model's measured speed on this GPU (34 tokens a second
writing, 5,700 reading). It is then asked the benchmark's question exactly as the frontier
models were. We wrote this adapter into the preregistration log after the main results and
before training it, so it is post-hoc.

It is scored twice. The rows marked "the untrained model's runs" use the untrained model's
five timed runs, as the other extra adapters do. That was the scoring declared before
training, and `G1-fixed`'s ordinary responses are as long as the untrained model's (median
difference +0.6% on held-out prompts). The rows marked "its own runs" use three timed runs
of `G1-fixed` itself with the adapter merged into the weights, as a deployment would serve
it. That timing pass was declared after the first rows existed and before it was run.

| System | Median estimate/actual | Median abs. log error [95% CI] | Typical factor off | r (log) | Within 2x | MAE (s) | Parsed |
|---|---|---|---|---|---|---|---|
| GPT-5.5 (published) | 3.47x | 1.24 [1.10, 1.42] | 3.47x | 0.55 | 12% | 92.2 | 68/68 |
| Claude Opus 4.8 (published) | 2.87x | 1.06 [0.90, 1.14] | 2.87x | 0.81 | 25% | 30.5 | 68/68 |
| Tiny Aya, untrained, benchmark question as is | 5.61x | 1.74 [1.54, 1.99] | 5.70x | 0.32 | 7% | 92.0 | 68/68 |
| G1, no speed stated | 0.17x | 1.77 [1.51, 1.94] | 5.86x | 0.34 | 7% | 14.8 | 68/68 |
| G1-fixed, no speed stated (the untrained model's runs) | 0.74x | 0.42 [0.37, 0.47] | 1.52x | 0.77 | 76% | 6.0 | 68/68 |
| G1-fixed, no speed stated (its own runs, adapter merged) | 0.82x | 0.40 [0.27, 0.46] | 1.49x | 0.73 | 82% | 5.9 | 68/68 |
| G1-fixed, no speed stated, full prompt (the untrained model's runs) | 0.85x | 0.31 [0.19, 0.38] | 1.37x | 0.80 | 85% | 5.1 | 68/68 |
| G1-fixed, no speed stated, full prompt (its own runs, adapter merged) | 0.85x | 0.28 [0.22, 0.39] | 1.33x | 0.77 | 82% | 5.2 | 68/68 |

| Comparison (ours minus theirs, median abs. log error) | Difference | Paired 95% CI | Tasks where ours is closer |
|---|---|---|---|
| G1-fixed, no speed stated (the untrained model's runs) vs GPT-5.5 | -0.82 | [-1.02, -0.67] | 62/68 |
| G1-fixed, no speed stated (the untrained model's runs) vs Claude Opus 4.8 | -0.64 | [-0.74, -0.47] | 59/68 |
| G1-fixed, no speed stated (its own runs, adapter merged) vs GPT-5.5 | -0.84 | [-1.06, -0.68] | 62/68 |
| G1-fixed, no speed stated (its own runs, adapter merged) vs Claude Opus 4.8 | -0.66 | [-0.80, -0.48] | 56/68 |
| G1-fixed, no speed stated, full prompt (the untrained model's runs) vs GPT-5.5 | -0.93 | [-1.13, -0.76] | 63/68 |
| G1-fixed, no speed stated, full prompt (the untrained model's runs) vs Claude Opus 4.8 | -0.74 | [-0.88, -0.58] | 63/68 |
| G1-fixed, no speed stated, full prompt (its own runs, adapter merged) vs GPT-5.5 | -0.96 | [-1.15, -0.79] | 63/68 |
| G1-fixed, no speed stated, full prompt (its own runs, adapter merged) vs Claude Opus 4.8 | -0.77 | [-0.88, -0.61] | 62/68 |

Asked cold, `G1-fixed` has an error of 0.42 (a typical factor of 1.52x, 76% within 2x) and
a correlation of 0.77. It is closer than GPT-5.5 on 62 of the 68 tasks and closer than
Claude Opus 4.8 on 59. So the stated speed is not what separates our rows from the frontier
rows. A model trained on its own measurements on one machine answers the bare question
about as well as `G1` does with the speed stated (0.37 against the same runs).

Against its own clock the picture is the same. Merged into the weights, `G1-fixed` writes
about 35 tokens a second, where it was trained to assume 34. Its 68 answers score 0.40 (a
typical factor of 1.49x, 82% within 2x, correlation 0.73) against the median of its own
three runs. It is closer than GPT-5.5 on 62 of the 68 tasks and closer than Claude Opus 4.8
on 56. Shown the full prompt it scores 0.28.

What it costs: `G1-fixed` knows one machine. `G1` follows a stated speed across a 270-fold
range (see "Stating another machine's speed"), while `G1-fixed` has to be retrained when the
hardware changes. And it is still not a like-for-like contest, because the frontier models
were not trained for this question at all.

### Telling tasks apart

Lower error does not mean better ranking. Claude Opus 4.8 orders the tasks by duration as
well as `G1` does:

| Telling tasks apart | Pearson r, log durations [95% CI] | Spearman | Pairs ordered correctly | Pearson r, raw seconds |
|---|---|---|---|---|
| GPT-5.5 (published) | 0.55 [0.33, 0.73] | 0.55 | 69% | 0.40 |
| Claude Opus 4.8 (published) | 0.81 [0.71, 0.87] | 0.77 | 76% | 0.68 |
| Tiny Aya, untrained, speed stated | 0.41 [0.09, 0.67] | 0.46 | 65% | 0.47 |
| **Tiny Aya + G1 (primary)** | 0.78 [0.64, 0.87] | 0.70 | 76% | 0.66 |
| G1, shown the full prompt | 0.83 [0.71, 0.91] | 0.85 | 82% | 0.77 |
| G1-fixed, no speed stated (the untrained model's runs) | 0.77 [0.59, 0.88] | 0.76 | 78% | 0.71 |
| G1-fixed, no speed stated (its own runs, adapter merged) | 0.73 [0.55, 0.85] | 0.69 | 75% | 0.55 |

`G1` ranks tasks better than GPT-5.5 and about the same as Claude Opus 4.8 (the intervals
overlap). The frontier models' error is mostly an error of scale: both say roughly three
times too long on almost every task. The next table uses the benchmark's own outcomes, so no
row in it is an estimator anyone could have run in advance. "Own-median constant" predicts,
for each task, the median measured time of the other 67 tasks. "Rescaled" divides each
estimate by the median estimate/actual ratio of the other 67 tasks.

| Context (uses the benchmark's own outcomes) | Log error as asked | Own-median constant | Rescaled by one constant | r (log) |
|---|---|---|---|---|
| GPT-5.5 (published) | 1.24 | 0.26 | 0.41 | 0.55 |
| Claude Opus 4.8 (published) | 1.06 | 0.43 | 0.27 | 0.81 |
| Tiny Aya, untrained, benchmark question as is | 1.74 | 0.40 | 0.53 | 0.32 |
| Tiny Aya, untrained, speed stated | 1.71 | 0.40 | 0.42 | 0.41 |
| Tiny Aya, untrained, speed stated, 5 measured examples | 1.29 | 0.40 | 0.52 | 0.52 |
| Same constant for every task (no model call) | 0.95 | 0.40 | 0.40 | 0.00 |
| TF-IDF regressor on the description | 0.95 | 0.40 | 0.29 | 0.51 |
| Sentence-encoder regressor, description | 0.90 | 0.40 | 0.31 | 0.33 |
| Linear probe on the untrained hidden state | 0.66 | 0.40 | 0.29 | 0.61 |
| **Tiny Aya + G1 (primary)** | 0.38 | 0.37 | 0.31 | 0.78 |
| G1, stated tokens through the speed formula | 0.39 | 0.37 | 0.32 | 0.78 |
| G1, median of 16 sampled answers | 0.46 | 0.37 | 0.23 | 0.83 |
| G1, the paper's question wording | 0.36 | 0.37 | 0.30 | 0.78 |
| G1, no speed stated | 1.77 | 0.37 | 0.53 | 0.34 |
| G1, shown the full prompt | 0.32 | 0.37 | 0.23 | 0.83 |
| G1 predicting the untrained model's runs | 0.37 | 0.40 | 0.32 | 0.80 |
| G1, full prompt, the untrained model's runs | 0.33 | 0.40 | 0.26 | 0.84 |
| G1-fixed, no speed stated (the untrained model's runs) | 0.42 | 0.40 | 0.24 | 0.77 |
| G1-fixed, no speed stated (its own runs, adapter merged) | 0.40 | 0.45 | 0.37 | 0.73 |
| G1-fixed, no speed stated, full prompt (the untrained model's runs) | 0.31 | 0.40 | 0.27 | 0.80 |
| G1-fixed, no speed stated, full prompt (its own runs, adapter merged) | 0.28 | 0.45 | 0.30 | 0.77 |
| TF-IDF regressor on the full prompt | 0.75 | 0.40 | 0.37 | 0.36 |
| Sentence-encoder regressor, full prompt | 0.82 | 0.40 | 0.33 | 0.10 |
| Linear probe, state before answering the full prompt | 0.36 | 0.40 | 0.25 | 0.86 |

Two plain readings of this table:

- With one well-chosen correction factor, Claude Opus 4.8 would be at least as accurate as `G1`
  (0.27 after rescaling, against 0.38 for `G1` as asked and 0.31 for `G1` rescaled). The
  factor has to come from measured outcomes, which is what our training supplies for our
  model.
- On our machine these 68 tasks take similar times, so a single number chosen with
  knowledge of the outcomes would already score 0.37. `G1`'s raw error (0.38) is no better
  than that. What `G1` adds is that it gets there without seeing any benchmark outcome (the
  honest constant from the training prompts scores 0.95, because benchmark responses are
  about 2.4 times longer than the training prompts' responses), and that it ranks the tasks
  (r = 0.78, where any constant has none).

### Every row

| System | Median estimate/actual | Median abs. log error [95% CI] | Typical factor off | r (log) | Within 2x | MAE (s) | Parsed |
|---|---|---|---|---|---|---|---|
| GPT-5.5 (published) | 3.47x | 1.24 [1.10, 1.42] | 3.47x | 0.55 | 12% | 92.2 | 68/68 |
| Claude Opus 4.8 (published) | 2.87x | 1.06 [0.90, 1.14] | 2.87x | 0.81 | 25% | 30.5 | 68/68 |
| Tiny Aya, untrained, benchmark question as is | 5.61x | 1.74 [1.54, 1.99] | 5.70x | 0.32 | 7% | 92.0 | 68/68 |
| Tiny Aya, untrained, speed stated | 5.39x | 1.71 [1.59, 1.90] | 5.53x | 0.41 | 4% | 70.9 | 68/68 |
| Tiny Aya, untrained, speed stated, 5 measured examples | 0.30x | 1.29 [1.15, 1.48] | 3.63x | 0.52 | 22% | 31.2 | 68/68 |
| Same constant for every task (no model call) | 0.42x | 0.95 [0.77, 1.05] | 2.59x | 0.00 | 34% | 10.4 | 68/68 |
| TF-IDF regressor on the description | 0.39x | 0.95 [0.87, 1.08] | 2.59x | 0.51 | 26% | 10.3 | 68/68 |
| Sentence-encoder regressor, description | 0.44x | 0.90 [0.72, 1.00] | 2.47x | 0.33 | 35% | 9.8 | 68/68 |
| Linear probe on the untrained hidden state | 0.53x | 0.66 [0.56, 0.83] | 1.94x | 0.61 | 54% | 8.6 | 68/68 |
| **Tiny Aya + G1 (primary)** | 0.74x | 0.38 [0.32, 0.45] | 1.46x | 0.78 | 79% | 6.6 | 68/68 |
| G1, stated tokens through the speed formula | 0.73x | 0.39 [0.33, 0.46] | 1.48x | 0.78 | 79% | 6.6 | 68/68 |
| G1, median of 16 sampled answers | 0.64x | 0.46 [0.40, 0.56] | 1.59x | 0.83 | 78% | 7.1 | 68/68 |
| G1, the paper's question wording | 0.82x | 0.36 [0.28, 0.45] | 1.44x | 0.78 | 82% | 6.6 | 68/68 |
| G1, no speed stated | 0.17x | 1.77 [1.51, 1.94] | 5.86x | 0.34 | 7% | 14.8 | 68/68 |
| G1, shown the full prompt | 0.82x | 0.32 [0.24, 0.38] | 1.38x | 0.83 | 88% | 5.8 | 68/68 |
| G1 predicting the untrained model's runs | 0.78x | 0.37 [0.26, 0.50] | 1.45x | 0.80 | 79% | 5.7 | 68/68 |
| G1, full prompt, the untrained model's runs | 0.83x | 0.33 [0.24, 0.38] | 1.40x | 0.84 | 90% | 4.9 | 68/68 |
| G1-fixed, no speed stated (the untrained model's runs) | 0.74x | 0.42 [0.37, 0.47] | 1.52x | 0.77 | 76% | 6.0 | 68/68 |
| G1-fixed, no speed stated (its own runs, adapter merged) | 0.82x | 0.40 [0.27, 0.46] | 1.49x | 0.73 | 82% | 5.9 | 68/68 |
| G1-fixed, no speed stated, full prompt (the untrained model's runs) | 0.85x | 0.31 [0.19, 0.38] | 1.37x | 0.80 | 85% | 5.1 | 68/68 |
| G1-fixed, no speed stated, full prompt (its own runs, adapter merged) | 0.85x | 0.28 [0.22, 0.39] | 1.33x | 0.77 | 82% | 5.2 | 68/68 |
| TF-IDF regressor on the full prompt | 0.50x | 0.75 [0.60, 0.84] | 2.11x | 0.36 | 44% | 9.1 | 68/68 |
| Sentence-encoder regressor, full prompt | 0.46x | 0.82 [0.70, 0.92] | 2.27x | 0.10 | 35% | 10.0 | 68/68 |
| Linear probe, state before answering the full prompt | 0.75x | 0.36 [0.25, 0.45] | 1.43x | 0.86 | 85% | 5.7 | 68/68 |

| Comparison (ours minus theirs, median abs. log error) | Difference | Paired 95% CI | Tasks where ours is closer |
|---|---|---|---|
| **Tiny Aya + G1 (primary)** vs GPT-5.5 | -0.87 | [-1.06, -0.70] | 62/68 |
| **Tiny Aya + G1 (primary)** vs Claude Opus 4.8 | -0.68 | [-0.79, -0.51] | 61/68 |
| G1, stated tokens through the speed formula vs GPT-5.5 | -0.85 | [-1.05, -0.69] | 62/68 |
| G1, stated tokens through the speed formula vs Claude Opus 4.8 | -0.66 | [-0.78, -0.50] | 61/68 |
| G1, median of 16 sampled answers vs GPT-5.5 | -0.78 | [-0.97, -0.61] | 62/68 |
| G1, median of 16 sampled answers vs Claude Opus 4.8 | -0.59 | [-0.70, -0.40] | 54/68 |
| G1, the paper's question wording vs GPT-5.5 | -0.88 | [-1.08, -0.69] | 62/68 |
| G1, the paper's question wording vs Claude Opus 4.8 | -0.69 | [-0.80, -0.50] | 61/68 |
| G1, no speed stated vs GPT-5.5 | +0.52 | [+0.25, +0.74] | 23/68 |
| G1, no speed stated vs Claude Opus 4.8 | +0.71 | [+0.42, +0.94] | 14/68 |
| G1, shown the full prompt vs GPT-5.5 | -0.92 | [-1.11, -0.77] | 63/68 |
| G1, shown the full prompt vs Claude Opus 4.8 | -0.74 | [-0.86, -0.58] | 60/68 |
| G1 predicting the untrained model's runs vs GPT-5.5 | -0.87 | [-1.07, -0.66] | 62/68 |
| G1 predicting the untrained model's runs vs Claude Opus 4.8 | -0.68 | [-0.83, -0.47] | 62/68 |
| G1, full prompt, the untrained model's runs vs GPT-5.5 | -0.91 | [-1.10, -0.76] | 62/68 |
| G1, full prompt, the untrained model's runs vs Claude Opus 4.8 | -0.72 | [-0.87, -0.56] | 59/68 |
| G1-fixed, no speed stated (the untrained model's runs) vs GPT-5.5 | -0.82 | [-1.02, -0.67] | 62/68 |
| G1-fixed, no speed stated (the untrained model's runs) vs Claude Opus 4.8 | -0.64 | [-0.74, -0.47] | 59/68 |
| G1-fixed, no speed stated (its own runs, adapter merged) vs GPT-5.5 | -0.84 | [-1.06, -0.68] | 62/68 |
| G1-fixed, no speed stated (its own runs, adapter merged) vs Claude Opus 4.8 | -0.66 | [-0.80, -0.48] | 56/68 |
| G1-fixed, no speed stated, full prompt (the untrained model's runs) vs GPT-5.5 | -0.93 | [-1.13, -0.76] | 63/68 |
| G1-fixed, no speed stated, full prompt (the untrained model's runs) vs Claude Opus 4.8 | -0.74 | [-0.88, -0.58] | 63/68 |
| G1-fixed, no speed stated, full prompt (its own runs, adapter merged) vs GPT-5.5 | -0.96 | [-1.15, -0.79] | 63/68 |
| G1-fixed, no speed stated, full prompt (its own runs, adapter merged) vs Claude Opus 4.8 | -0.77 | [-0.88, -0.61] | 62/68 |

What the controls show:

- **`G1` needs the stated speed.** Without it `G1` is worse than both frontier models
  (1.77). The adapter learned how much it will write; it has no way to know how fast the
  machine is unless told, or unless it is trained for one machine as `G1-fixed` is.
- **The stated speed is not sufficient.** The untrained model with the same speed in the
  prompt is no better than without it; 59 of its 68 answers are "1 minute" or "2 minutes".
  Adding five measured examples to the prompt moves its scale but not its accuracy.
- **Predictors that never ask the model are worse from the description.** A constant, a
  TF-IDF regressor and a sentence-encoder regressor trained on the same labels all miss by
  about 2.5x, mostly because benchmark responses are longer than training responses and
  those predictors cannot tell.
- **The untrained model already carries the information.** A linear probe on the untrained
  model's hidden state, trained on the same labels, reaches 0.66 from the benchmark
  question. Read just before the model starts answering the full prompt, the probe does
  about as well as the adapter shown the full prompt. So the adapter is not creating
  knowledge that was absent. It makes the model say, in the benchmark's format, something
  its internal state already predicts.
- **The wording does not matter much.** The paper's question wording gives the same result
  as the artifact's wording.
- **The adapter did not change how long the model's responses are**, so its estimates are as
  accurate for the untrained model's runs as for its own.

For reference, the two frontier models were also asked after finishing each task. Our
estimate, made before starting, is closer than Claude Opus 4.8's estimate made afterwards
and not as close as GPT-5.5's:

| After completing the task (published, context only) | Median estimate/actual | Median abs. log error | Typical factor off | r (log) | Within 2x |
|---|---|---|---|---|---|
| GPT-5.5 | 0.93x | 0.30 | 1.35x | 0.67 | 94% |
| Claude Opus 4.8 | 1.80x | 0.59 | 1.80x | 0.81 | 56% |

### Stating another machine's speed

The same question was asked again with four other speeds in the system prompt. We do not
have those machines, so the targets are the measured runs replayed through the speed
formula (prompt tokens / reading speed + response tokens / writing speed). This checks
whether the estimate follows the stated speed, not whether it is right on real hardware of
that speed; sections 7.3 and 7.7 of the write-up do the latter for the first study.

| Estimator | Stated speed (tokens/s writing, reading) | Median target (s) | Median estimate (s) | Median estimate/actual | Median abs. log error | r (log) | Within 2x | Tokens through the formula | Stated writing speed vs this machine | Estimate vs its estimate at this machine's speed (median per task) |
|---|---|---|---|---|---|---|---|---|---|---|
| G1 | 1.5, 10 | 394.2 | 270.0 | 0.67x | 0.46 | 0.31 | 68% | 0.31 | 20x slower | 22.1x longer |
| G1 | 8, 60 | 73.0 | 57.1 | 0.76x | 0.34 | 0.78 | 93% | 0.33 | 3.75x slower | 4.29x longer |
| G1 | 30, 3300 (this machine, measured) | 17.7 | 13.6 | 0.74x | 0.38 | 0.78 | 79% | 0.39 | same | same |
| G1 | 190, 19000 | 2.8 | 2.1 | 0.75x | 0.39 | 0.79 | 81% | 0.39 | 6.33x faster | 6.33x shorter |
| G1 | 400, 40000 | 1.3 | 1.0 | 0.75x | 0.36 | 0.79 | 81% | 0.39 | 13.3x faster | 13.4x shorter |
| Untrained | 1.5, 10 | 404.1 | 120.0 | 0.25x | 1.39 | 0.24 | 9% | n/a | 22.7x slower | unchanged |
| Untrained | 8, 60 | 74.9 | 120.0 | 1.32x | 0.38 | 0.32 | 75% | n/a | 4.25x slower | unchanged |
| Untrained | 34, 5700 (this machine, measured) | 15.4 | 60.0 | 5.39x | 1.71 | 0.41 | 4% | n/a | same | same |
| Untrained | 190, 19000 | 2.8 | 60.0 | 30.58x | 3.42 | 0.39 | 3% | n/a | 5.59x faster | unchanged |
| Untrained | 400, 40000 | 1.3 | 60.0 | 63.78x | 4.16 | 0.27 | 3% | n/a | 11.8x faster | unchanged |

`G1`'s estimates follow the stated speed across a 270-fold range. Told the machine writes
20 times slower, it gives estimates a median of 22 times longer for the same task (the
targets grow 22-fold, because reading is slowed even more than writing). Told the machine
is 6.3 or 13 times faster, it gives estimates 6.3 or 13 times shorter. The untrained model
gives the same answer to a task at every stated speed, so the one speed at which it looks
accurate (8 tokens a second) is a coincidence of scale.

At the slowest speed `G1`'s stated time loses its ranking (r = 0.31) while the token count
it states, put through the formula, keeps its accuracy (0.31). This is the same division
weakness the first study found, and the same remedy.

### A range around each estimate

Asking 16 times and taking the 10th to 90th percentile of the answers gives a range that is
too narrow:

Sampled answers:
- G1, benchmark question (primary): 16 draws, 10-90% range covers 50%
- G1, shown the full prompt: 16 draws, 10-90% range covers 57%
- G1, the paper's question wording: 16 draws, 10-90% range covers 56%
- G1, no speed stated: 16 draws, 10-90% range covers 9%

For the primary question the range covers 50% of measured times where a calibrated one
would cover 80%. The median of the 16 answers ranks tasks slightly better (r = 0.83) but
runs lower (0.64x), so its error is higher than the single answer's.

After seeing that, we added one more range. Its rule was written into the preregistration
log and pushed before it was computed (entry of 2026-11-15), but after the benchmark
estimates existed, so it is post-hoc. The rule is split-conformal calibration on the 170
held-out pool prompts: take `G1`'s absolute log error on each of them, find the 137th
smallest (the 80% level for 170 prompts), and call it `q`. Every benchmark estimate then
gets the range from the stated time divided by `e^q` to the stated time multiplied by
`e^q`. No benchmark outcome enters the rule.

| Range calibrated on held-out pool prompts (80% target) | Calibration prompts | q (log units) | Range: stated time divided and multiplied by | Benchmark measured times inside | 10-90% range of 16 samples covers |
|---|---|---|---|---|---|
| G1, benchmark question (primary), its own runs | 170 | 0.92 | 2.52x | 93% | 50% |
| G1, shown the full prompt, its own runs | 170 | 0.63 | 1.88x | 85% | 57% |
| G1, benchmark question, the untrained model's runs | 170 | 0.92 | 2.52x | 93% | n/a |
| G1, shown the full prompt, the untrained model's runs | 170 | 0.63 | 1.88x | 82% | n/a |
| G1-fixed, no speed stated, the untrained model's runs | 170 | 1.06 | 2.88x | 99% | n/a |
| G1-fixed, no speed stated, full prompt, the untrained model's runs | 170 | 0.61 | 1.84x | 79% | n/a |
| G1-fixed, no speed stated, its own runs (adapter merged) | 170 | 1.06 | 2.88x | 96% | n/a |
| G1-fixed, no speed stated, full prompt, its own runs (adapter merged) | 170 | 0.61 | 1.84x | 75% | n/a |

From the description, the range is the stated time divided and multiplied by 2.52, and 93%
of the benchmark's measured times are inside it. That is more than the 80% aimed for, so on
this suite the range is wider than it needs to be. From the full prompt the factor is 1.88
and 85% are inside (82% against the untrained model's runs). For `G1-fixed` the factor is
2.88 from the description with 99% inside (96% against its own runs), and 1.84 from the
full prompt with 79% inside (75% against its own runs, a little under the target).
For scale, the edge of `G1`'s range (2.52x from the stated time) is closer than the typical
miss of the frontier models' point estimates (3.47x and 2.87x). The guarantee behind
split-conformal ranges assumes the benchmark tasks resemble the calibration prompts, which
they do not exactly, so these shares are observations on 68 tasks, not guarantees.

### More seeds and smaller training sets

The amendment of 2026-11-14 also declared, to run if time allowed, two more seeds of the
same recipe and adapters trained on 450 and 150 of the 1,353 training prompts. Each is
scored against the untrained model's runs, because timing every adapter's own runs would
not fit in the time left. The first row is the primary adapter scored the same way.

| Adapter (against the untrained model's runs) | Stated time: log error | r (log) | Within 2x | Stated tokens through the formula: log error | r (log) | Stated time agrees with its own token counts (within 10%) | Full prompt, stated time: log error | Full prompt, tokens through the formula | Response-length drift |
|---|---|---|---|---|---|---|---|---|---|
| G1 seed 0 (1,353 prompts) | 0.37 | 0.80 | 79% | 0.38 | 0.80 | 68/68 | 0.33 | 0.33 | +0.5% |
| Seed 1 (1,353 prompts) | 0.43 | 0.76 | 71% | 0.43 | 0.76 | 68/68 | 0.39 | 0.39 | +0.0% |
| Seed 2 (1,353 prompts) | 0.50 | 0.78 | 75% | 0.50 | 0.77 | 67/68 | 0.38 | 0.38 | +1.7% |
| 450 prompts, seed 0 | 0.41 | 0.64 | 78% | 0.42 | 0.64 | 68/68 | 0.34 | 0.32 | +0.0% |
| 150 prompts, seed 0 | 2.95 | -0.06 | 3% | 0.64 | 0.45 | 0/68 | 4.25 | 0.67 | -1.6% |

**Three seeds: 0.37, 0.43 and 0.50.** Seed 0 is the primary adapter. It was fixed before
any result, and it turned out to be the best of the three. Against the same runs, seeds 1
and 2 score 0.43 and 0.50 (71% and 75% within 2x), so the average over seeds is 0.43, and
that is the fairer number for what the recipe gives. Ranking is steady across seeds (r =
0.80, 0.76, 0.78) and so is the error from the full prompt (0.33, 0.39, 0.38). The seeds
differ mostly in scale: their median estimate/actual ratios are 0.78x, 0.69x and 0.63x, so
the weaker seeds run shorter. On the held-out pool prompts they score 0.38, 0.49 and 0.42
from the description, which is a different order, so that check would not have told us
which seed to prefer.

Every seed is still far closer than the frontier models. The weakest, seed 2, has an error
lower than GPT-5.5's by 0.74 (paired 95% interval 0.59 to 0.95) and lower than Claude Opus
4.8's by 0.55 (0.38 to 0.68), and is closer on 61 and 58 of the 68 tasks. For seed 1 the
differences are 0.82 (0.60 to 1.02) and 0.63 (0.44 to 0.74). Differences of about 0.1
between our own rows should not be read as real.

**450 prompts are nearly enough.** With a third of the training pool the error is 0.41
against 0.37, and 78% of estimates are within 2x. Ranking is weaker (r = 0.64 against
0.80). The error is still lower than GPT-5.5's by 0.84 and Claude Opus 4.8's by 0.65, with
paired intervals that exclude zero.

**150 prompts teach the length but not the division.** The adapter answers in the right
format, but none of its 68 stated times agrees with its own token counts. For example it
states about 220 response tokens and 0.256 seconds at 35 tokens a second, where the formula
gives 6.4 seconds. Its stated time is therefore useless (2.95). The token count it states,
put through the formula, scores 0.64 with a correlation of 0.45, which is still lower than
both frontier models' error (by 0.60 and 0.42, paired intervals excluding zero). The model
learns roughly how much it will write from little data and learns the division later. This
is the division weakness of section 7.3 of the write-up again, and the reason the main
study's structured variant computes the time outside the model.

### The timed runs

Executors:
- untrained model: runs per task [5], median 15.4 s (range 0.6 to 45.2), median response 522 tokens, capped runs 2, run-to-run abs. log error 0.070, measured length through the speed formula vs wall time 0.015, speed cards [(34.0, 5586), (34.2, 5657), (35.5, 5827)]
- G1-fixed, adapter merged into the weights: runs per task [3], median 14.2 s (range 0.9 to 34.4), median response 502 tokens, capped runs 1, run-to-run abs. log error 0.065, measured length through the speed formula vs wall time 0.007, speed cards [(34.7, 5730), (35.8, 5855)]
- fine-tuned model (G1): runs per task [5], median 17.7 s (range 1.0 to 42.2), median response 526 tokens, capped runs 1, run-to-run abs. log error 0.071, measured length through the speed formula vs wall time 0.029, speed cards [(29.1, 3213), (30.2, 3313)]

Each task was run 5 times with the untrained model and with `G1`, and 3 times with
`G1-fixed`. A repeat run differs from the median of the others by about 7%, which is the
floor for any estimator here. The speed in the system prompt is the measured one rounded to
two digits (30 tokens a second writing and 3,300 reading for `G1`; 34 and 5,700 for the
untrained model). `G1` is slower because its adapter layers were left unmerged; `G1-fixed`,
merged, runs at the untrained model's speed, which is the speed it was trained to assume.
The measured
response length put through the speed formula reproduces the measured wall time to within
3%, so on this machine the whole difficulty is predicting how much the model will write.

### Check on held-out training-pool prompts

Before the adapter saw any benchmark task we scored it on the 170 held-out prompts of the
training pool. This check, not the benchmark, was the place to catch a broken adapter.

Held-out pool prompts (170; run-to-run abs. log error 0.107):

| Estimator | Median estimate/actual | Median abs. log error | r (log) | Within 2x | Parsed | 10-90% range of the samples covers |
|---|---|---|---|---|---|---|
| g1-s0/card_description | 0.98x | 0.38 | 0.42 | 69% | 170/170 |  |
| g1-s0/card_description+sampled_median | 0.88x | 0.43 | 0.49 | 71% | 170/170 | 62% |
| g1-s0/card_full_prompt | 0.88x | 0.32 | 0.68 | 82% | 170/170 |  |
| g1-s0/card_full_prompt+sampled_median | 0.88x | 0.32 | 0.77 | 84% | 170/170 | 62% |
| static/pool_median_constant | 0.94x | 0.56 | n/a | 56% | 170/170 |  |
| static/text_regressor_description | 0.81x | 0.56 | 0.32 | 61% | 170/170 |  |
| static/embedding_regressor_description | 0.83x | 0.48 | 0.45 | 66% | 170/170 |  |
| static/text_regressor_full | 0.90x | 0.35 | 0.53 | 73% | 170/170 |  |
| static/embedding_regressor_full | 0.87x | 0.48 | 0.53 | 68% | 170/170 |  |
| g1-fixed/plain_full_prompt | 0.91x | 0.28 | 0.72 | 85% | 170/170 |  |
| g1-fixed/protocol | 1.07x | 0.41 | 0.44 | 70% | 170/170 |  |
| g1-n150/card_description | 0.08x | 2.53 | 0.11 | 7% | 170/170 |  |
| g1-n150/card_full_prompt | 0.03x | 3.51 | 0.20 | 5% | 170/170 |  |
| g1-n450/card_description | 1.19x | 0.39 | 0.46 | 68% | 170/170 |  |
| g1-n450/card_full_prompt | 1.00x | 0.35 | 0.63 | 76% | 170/170 |  |
| g1-s1/card_description | 0.99x | 0.49 | 0.41 | 66% | 170/170 |  |
| g1-s1/card_full_prompt | 0.93x | 0.31 | 0.69 | 84% | 170/170 |  |
| g1-s2/card_description | 1.04x | 0.42 | 0.44 | 71% | 170/170 |  |
| g1-s2/card_full_prompt | 0.84x | 0.36 | 0.70 | 73% | 170/170 |  |
| untrained/card_description | 11.33x | 2.44 | 0.11 | 1% | 170/170 |  |
| untrained/card_fewshot | 0.43x | 1.16 | 0.22 | 36% | 170/170 |  |

Response length on the held-out pool prompts, against the untrained model:
- g1-fixed: response length vs untrained +0.6% (median), median abs. log ratio 0.161
- g1-n150: response length vs untrained -1.6% (median), median abs. log ratio 0.143
- g1-n450: response length vs untrained +0.0% (median), median abs. log ratio 0.134
- g1-s0: response length vs untrained +0.5% (median), median abs. log ratio 0.145
- g1-s1: response length vs untrained +0.0% (median), median abs. log ratio 0.157
- g1-s2: response length vs untrained +1.7% (median), median abs. log ratio 0.150

Row names are estimator/question: `card_description` is the benchmark question with the
description and the stated speed, `card_full_prompt` shows the whole prompt, `+sampled_median`
is the median of 16 sampled answers, `static/` rows never ask the model, and `card_fewshot`
adds five measured examples to the untrained model's prompt. For `g1-fixed`, `protocol` is
the benchmark question with the description and no speed, and `plain_full_prompt` shows the
whole prompt, also with no speed. The targets here are response lengths put through the
speed formula, not timed runs.

From the description alone `G1` scores 0.38 on these prompts, the same as on the benchmark,
against 0.56 for a constant and 0.48 for the better text regressor. Its correlation here is
modest (0.42), because many pool prompts have short answers of similar length. With the
full prompt it scores 0.32, close to the best text regressor (0.35). The untrained model
with the speed stated is off by a factor of eleven. The fine-tuned model's ordinary
responses are the same length as the untrained model's (median difference +0.5%), which is
why its estimates can also be scored against the untrained model's runs.

### The paper's own trained model

The paper trains Qwen3-8B to estimate its time on the same 68 tasks (its Appendix C, Table
22, estimates before generating). The best row there, RAFT with an asymmetric loss, has a
mean absolute error of 24.04 s and a Pearson correlation of 0.44 on raw durations; plain
fine-tuning with the speed stated has 30.45 s and 0.39. `G1` has 6.6 s and 0.66. The
models and machines differ, so the seconds are not comparable (our runs have a median of
17.7 s). The correlation is the comparable number.

## What this does not show

- **It is not a like-for-like contest.** The frontier models were asked cold and were not
  trained for this. `G1` was trained on its own measurements and told its speed. `G1-fixed`
  drops the stated speed and still scores 0.42 (0.40 on its own runs), but it too was trained on this model's
  measurements on this machine. The paper reports that stating the
  measured pace helps the frontier models on its long-horizon tasks (Table 10: GPT-5.5
  1.18x and 0.84 log error, Claude Opus 4.8 1.12x and 0.52). That ablation does not exist
  for these 68 tasks, so we cannot say how close the frontier models would get with the same
  help here.
- **Our clock is easier to know.** A hosted model's time includes provider load and hidden
  reasoning. Ours runs alone on one local GPU, where a repeat run differs by a median of 7%.
- **Ranking is not better than Claude Opus 4.8's**, and one correction factor would bring
  Opus level with us or ahead (see "Telling tasks apart").
- **Our raw error is no better than a single well-chosen number** for this suite on this
  machine. The gain over that number is in ranking and in not needing the outcomes.
- **The adapter adds little over a probe** when both see the full prompt.
- **Estimates run short**, at a median of 0.74x of the measured time.
- **The sampled range is too narrow** (50% coverage for a nominal 80%). The range
  calibrated on held-out training prompts, added afterwards, covers 93% but is wide: a
  factor of 2.52 each way.
- **In-prompt examples did not work** for the untrained model.
- **The primary seed is the best of three.** Against the same runs, seeds 0, 1 and 2 of
  `G1` score 0.37, 0.43 and 0.50, an average of 0.43. The primary row uses seed 0, which
  was fixed before any result, and it came out on the favourable side. `G1-fixed` and the
  smaller training sets have one seed each.
- **68 tasks, one model, one GPU, single-turn only.** The benchmark's agentic and
  long-horizon suites were not run.
- **The frontier rows are the artifact's published numbers.** We did not rerun those models.
- **The other-speed rows use computed targets**, not runs on other machines.

## What is committed

- `results/metatime.json`: every row of the tables above, with intervals and paired comparisons.
- `results/metatime/runs.jsonl`: every timed run (task id, prompt and response tokens, seconds).
- `results/metatime/estimates.jsonl`: every answer of ours with the time read from it. An answer that quotes a benchmark task is scored but not written out.
- `results/metatime/reference-published.csv`: the artifact's per-task times and estimates for the two frontier models, as downloaded.
- `results/metatime/pool-labels.jsonl`: the training pool's ids, sources and measured response lengths.
- `tests/test_metatime_records.py` and `notebooks/benchmark_head_to_head.ipynb` recompute the table rows from these records.

The adapters are not uploaded yet; each is 121 MB and can be retrained with the commands below.

## Reproduce

Scoring runs on CPU in the pinned environment of the repository root. It downloads the
benchmark artifact once:

```bash
uv sync --locked
. ./env.sh
uv run python scripts/metatime/score.py --suite /path/to/metatime-artifact   # needs data/metatime/ for a full rescore
uv run python scripts/metatime/report.py                                     # prints the tables on this page
uv run python scripts/metatime/figure.py                                     # redraws the figure
```

The GPU steps need PyTorch, `transformers` and `peft`, which the pinned project environment
does not have. With such an environment active, from the repository root, on one 16 GB GPU:

```bash
M=CohereLabs/tiny-aya-global; D=data/metatime; SUITE=/path/to/metatime-artifact; X=scripts/metatime
python $X/build_pool.py --suite $SUITE --out $D/pool.jsonl                      # training pool; drops look-alikes of benchmark tasks; reads the first study's legacy/data/tasks/tasks.jsonl
python $X/gen_labels.py --model $M --pool $D/pool.jsonl --out $D/pool-labels.jsonl   # the model's own response lengths, 3 passes
python $X/train.py --model $M --pool $D/pool.jsonl --labels $D/pool-labels.jsonl --output adapters/g1-s0
python $X/estimate.py --model $M --adapter adapters/g1-s0 --executor adapter --suite $SUITE --out $D/est-g1-s0.jsonl --samples 16
python $X/time_runs.py --model $M --adapter adapters/g1-s0 --suite $SUITE --out $D/time-g1-s0.jsonl --runs 5
python $X/time_runs.py --model $M --suite $SUITE --out $D/time-base.jsonl --runs 5
python $X/estimate.py --model $M --executor base --suite $SUITE --pool $D/pool.jsonl --labels $D/pool-labels.jsonl --out $D/est-untrained.jsonl --variants protocol card_description card_fewshot
python $X/probe.py --model $M --suite $SUITE --pool $D/pool.jsonl --labels $D/pool-labels.jsonl --out $D/probe.json
# G1-fixed: no speed in the prompt; 34 and 5700 are the untrained model's measured speeds on this GPU
python $X/train.py --model $M --pool $D/pool.jsonl --labels $D/pool-labels.jsonl --output adapters/g1-fixed --fixed-speed 34 5700
python $X/estimate.py --model $M --adapter adapters/g1-fixed --executor base --suite $SUITE --out "$D/est-g1-fixed@base.jsonl" --variants protocol plain_full_prompt
python $X/time_runs.py --model $M --adapter adapters/g1-fixed --merge --suite $SUITE --out $D/time-g1-fixed.jsonl --runs 3
# more seeds and smaller training sets: add --seed 1 (or 2) or --train-prompts 450 (or 150) to train.py, then
python $X/estimate.py --model $M --adapter adapters/g1-s1 --executor base --suite $SUITE --out "$D/est-g1-s1@base.jsonl" --variants card_description card_full_prompt
```

Timed runs must have the GPU to themselves. Times depend on the machine, so a rerun on other
hardware gives other durations and other stated speeds; the comparison is always each model
against its own runs.
