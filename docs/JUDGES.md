# Judge guide

## 30-second summary

Ask a model how long a task will take it and it answers on a human clock. On the 68 single-turn tasks of the public MetaTime benchmark, the published self-estimates of GPT-5.5 and Claude Opus 4.8 are typically 3.47x and 2.87x off their own measured times (12% and 25% within 2x). We trained one LoRA adapter (`G1`) on Cohere Labs' Tiny Aya Global (3.35B), using only measurements of how long the model's own answers were on 1,353 public prompts, none from the benchmark. Told its machine's speed, the untrained model is 5.5x off (4% within 2x); with `G1` it is 1.46x off (79% within 2x, median absolute log error 0.38 [0.32, 0.45]). Ours is closer than GPT-5.5 on 62 of 68 tasks and than Claude Opus 4.8 on 61, with paired 95% intervals that exclude zero, which is the success criterion we pushed before the first estimate.

The track's three libraries are chained in one more adapter, `G1-kit`: CuratorKIT gates the measured pool (1,353 → 1,342), AlignTune trains the LoRA on what passed, AuditKIT scores it on the benchmark and on ARC with the adapter off and on. Against the untrained model's runs `G1-kit` scores 0.45 [0.37, 0.59] (76% within 2x), inside the spread of `G1`'s three seeds (0.37 to 0.50), and also beats both frontier models (paired −0.79 and −0.60, intervals exclude zero). AuditKIT's paired bootstrap: mean log error 1.78 → 0.50 and within 2x 4% → 76% against the untrained model, p = 0.001. Its lm-eval run found a small, significant ARC loss with the adapter on (Easy 73.0% → 71.1%, Challenge 42.2% → 40.2%). 1,000 estimates cost 3 to 5 cents of GPU time, against at least 57 to 60 cents for the same question at frontier list prices.

The caveats are up front in every document: each model is judged on its own clock; `G1` is told its speed (`G1-fixed`, told nothing, scores 0.42); Claude Opus 4.8 ranks tasks as well as `G1` (r 0.81 against 0.78); the primary seed is the best of three (0.37, 0.43, 0.50 on the same runs); `G1-kit` loses about 2 ARC points; one GPU, English only.

## What to open, in order

1. [`README.md`](../README.md) (2 minutes): the result in plain words, the pipeline, the caveats.
2. [`paper/article.md`](../paper/article.md) (10 minutes): the write-up, in the order of the hackathon's write-up checklist.
3. [`docs/METATIME.md`](METATIME.md) (5 minutes): every table and control row, what the comparison does not show, the commands.
4. [`docs/PREREGISTRATION.md`](PREREGISTRATION.md) (2 minutes): each declaration word for word, with the commit and time that pushed it.
5. [`notebooks/benchmark_head_to_head.ipynb`](../notebooks/benchmark_head_to_head.ipynb) (2 minutes, CPU): recomputes the rows, the paired comparison and the figure from the committed records (`uv run python scripts/run_notebook.py`).

## Rubric and rules map

| Criterion or rule | Where it is answered |
|---|---|
| Write-up clarity | [`README.md`](../README.md) for a general reader; [`paper/article.md`](../paper/article.md) follows the write-up checklist section by section |
| Practical usefulness | A 3B model that runs on one consumer GPU or a CPU and gives its own time within 2x for 79% of tasks, following the stated hardware across a 270-fold speed range (article §7.4); cost per 1,000 estimates against API billing (§8); a local demo that times your own task (`space/`) |
| Code and notebook quality | `scripts/metatime/` (one script per stage), 22 tests and a CPU notebook that recompute every row from committed records, ruff-clean, CI on every push ([`.github/workflows/`](../.github/workflows/)) |
| Built during the hackathon | All work 9-15 November 2026; commit history; article §13 |
| Chain of track repos (3 preferred) | CuratorKIT 0.1.23 → AlignTune 0.1.32 → AuditKIT 1.1.6: `scripts/metatime/curate.py`, `train_aligntune.py`, `audit.py`; article §2 and §7.3 |
| Cohere Labs open-weights model | `CohereLabs/tiny-aya-global`, revision `af89d219`; article §3 |
| Public HF collections | [Models](https://huggingface.co/collections/isHeSatoshi/anthropochrony-self-time-models-6af986a95a140f4184fae18f) (LoRAs, GGUFs), [data and logs](https://huggingface.co/collections/isHeSatoshi/anthropochrony-self-time-data-6af986a99994d1dfbbad03aa) (pool, labels, CuratorKIT report, training logs, AuditKIT report, benchmark records) |
| Before-and-after metrics, more than one | Median absolute log error, typical factor, estimate/actual ratio, correlation, within 2x, MAE, parse rate, with 95% CIs and paired tests; ARC-Easy and ARC-Challenge for retention; article §5 and §7 |
| Metrics, data, tasks, models and why | Article §1, §4, §5 |
| How the model is meaningfully better | Article §7.1 (before → after, against frontier models), §7.4 (follows the hardware), §8 (cost) |
| ROI, closed model against a small model | Article §8 |

## Strongest points

1. A benchmark we did not build, its own question and parser, and a success criterion pushed before the first estimate, met with paired intervals that exclude zero.
2. Every control we could think of is reported, including the ones that cut against us: the outcome-fitted constant, the hidden-state probe, the rescaled frontier models, the weaker seeds.
3. The estimate follows the hardware: told a 13x faster machine, `G1` gives 13x shorter estimates; the untrained model does not move.
4. Fully reproducible on CPU from committed records; all artifacts public.

## Weakest points

1. Not a like-for-like contest: the frontier models were asked cold and never trained for this question.
2. Claude Opus 4.8 ranks tasks as well as `G1`; its error is mostly a scale factor.
3. One model, one GPU, 68 single-turn tasks, English only; the primary seed is the best of three.
4. `G1-kit` costs about 2 points of ARC accuracy with the adapter on (article §7.7).
5. No hosted demo: Docker and Gradio Spaces need a paid Hugging Face plan, so the demo runs locally.
