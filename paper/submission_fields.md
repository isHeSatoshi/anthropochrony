# Submission form values

Fill the `TODO` lines before submitting at https://hackathon.lexsi.ai/submission/. Every other value is checked against the repository and the live Hugging Face pages.

| Field | Value |
|---|---|
| Registered email | `TODO: your registered email` |
| Discord username (optional) | `TODO: your Discord handle` |
| Teammates | Solo ([@isHeSatoshi](https://huggingface.co/isHeSatoshi)) |
| HF model collection | https://huggingface.co/collections/isHeSatoshi/anthropochrony-self-time-models-6af986a95a140f4184fae18f |
| HF dataset collection | https://huggingface.co/collections/isHeSatoshi/anthropochrony-self-time-data-6af986a99994d1dfbbad03aa |
| Write-up | https://github.com/isHeSatoshi/anthropochrony/blob/main/paper/article.md |
| GitHub repo | https://github.com/isHeSatoshi/anthropochrony |
| Demo | Local Gradio app in `space/` (CPU, 4-bit GGUFs from `isHeSatoshi/anthropochrony-tiny-aya-global-g1-kit-gguf`); a hosted Space needs a paid plan |
| Repos used | CuratorKIT 0.1.23, AlignTune 0.1.32, AuditKIT 1.1.6 |
| Models used | `CohereLabs/tiny-aya-global` (revision `af89d219b53ed9b13b8a4645f9c8028973510324`); published GPT-5.5 and Claude Opus 4.8 numbers from the MetaTime artifact as baselines (not called) |
| Track | Track 1: Fine-tuning and Data Curation |
| Problem statements | #2 Replacing a closed-model API; Other: model self-knowledge of its own running time |

## Abstract (under 500 words)

Ask a language model how long a task will take it and it answers on a human clock: "a few minutes" for work it finishes in seconds, because the text it learned from describes how long people take. Those estimates feed real decisions, such as job queues, timeouts, routing between machines and what users are told. On the 68 single-turn tasks of the public MetaTime benchmark, the published self-estimates of GPT-5.5 and Claude Opus 4.8 are typically 3.47x and 2.87x off their own measured times (12% and 25% within 2x).

We taught Cohere Labs' Tiny Aya Global (3.35B) its own clock with the track's full pipeline, with no teacher model and no paid labels. We ran the untrained model three times on each of 1,523 public instruction prompts (Dolly-15k, CodeAlpaca-20k, no_robots, GSM8K and 100 coding tasks of ours), after removing 21 prompts that resembled a benchmark task, and recorded how long its answers were. CuratorKIT gated the 1,353 training prompts (schema, exact and near-duplicate removal, secrets, and our rule against labels cut off by the generation cap; 1,342 passed). AlignTune trained a LoRA adapter, `G1-kit`, to answer "how long will this take you?" with the work first and the time second: the tokens it expects to read and write, then the seconds at the machine speed stated in its prompt. AuditKIT evaluated it.

The task list, the benchmark's own question and parser, the metrics and the success criterion were pushed to GitHub before the first estimate, and each `G1-kit` row before it was run. Before → after on the same model, each against its own timed runs: median absolute log error 1.71 → 0.44 (typical miss 5.5x → 1.55x; within 2x 4% → 72%). Ours is closer than GPT-5.5 on 60 of 68 tasks and than Claude Opus 4.8 on 55, and both paired 95% intervals exclude zero, so the declared criterion is met. Told another machine's speed, the estimates scale with it across a 270-fold range. AuditKIT's paired bootstrap confirms the gain over the untrained model (mean log error 1.78 → 0.50, p = 0.001) and finds a cost: ARC accuracy falls by about 2 points (Easy 73.0% → 71.1%, Challenge 42.2% → 40.2%), so the adapter is best switched on only for the time question. 1,000 estimates cost 3 to 5 cents of GPU time; the same question at frontier list prices costs at least 57 to 60 cents.

Limits: each model is judged on its own clock, and ours is told its speed (asked cold it scores 1.53, worse than both frontier models); Claude Opus 4.8 ranks tasks better than ours (r 0.81 against 0.71); one training run. One model, one consumer GPU, English, single-turn tasks. All compute was local, at no cost.
