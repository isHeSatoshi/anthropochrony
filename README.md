# Anthropochrony

### Ask an AI how long a task will take it and it answers on a human clock. We taught Cohere's Tiny Aya Global to know its own.

_Built for the Lexsi AI Safety & Alignment Hackathon, Track 1 (CuratorKIT → AlignTune → AuditKIT)._

[Write-up](paper/article.md) · [Full results](docs/METATIME.md) · [For judges](docs/JUDGES.md) · [What was declared, and when](docs/PREREGISTRATION.md) · [Check the numbers](notebooks/benchmark_head_to_head.ipynb) · [Models](https://huggingface.co/collections/isHeSatoshi/anthropochrony-self-time-models-6af986a95a140f4184fae18f) · [Data and logs](https://huggingface.co/collections/isHeSatoshi/anthropochrony-self-time-data-6af986a99994d1dfbbad03aa)

![Head-to-head on the MetaTime benchmark](paper/figures/F9-metatime-head-to-head.png)

## The problem

Ask a frontier model how long a piece of work will take. Give Claude a refactor, a new feature, a migration, and you'll hear "this is a few days of work" or "figure on a week". Then you let it run, and it's done in minutes.

The model isn't lying. It learned from text written by people, and when people say how long a job takes, they mean how long a person takes. So you get the time a human engineer would need, from something that isn't one. It's like asking a calculator how long long division takes and hearing "about ten minutes, with a pencil".

This matters because estimates drive decisions: how to queue jobs, when to time out a run, whether a task goes to a phone or a server, what to tell a customer about when their work will be done.

There is now a public test for it. [MetaTime](https://openreview.net/forum?id=CAQeizWcB2), a paper under review at ICLR 2027, asks a model how long a task will take it, runs the task, and compares. Its 68 single-turn tasks are everyday requests: write a function, fix a bug, summarise a text, solve a puzzle, write an essay or a story, answer a question. The paper's public files include what GPT-5.5 and Claude Opus 4.8 answered and how long each one really took. Both say about three times too long for nearly everything.

## The result

We took Cohere Labs' Tiny Aya Global (`CohereLabs/tiny-aya-global`), an open model with 3.35 billion parameters, and measured how long its own answers were on 1,353 ordinary public prompts. The track's three libraries then built one small adapter (a LoRA, 121 MB) from those measurements: CuratorKIT cleaned the data, AlignTune trained the adapter and AuditKIT tested it. We wrote the rules down and pushed them before running anything: all 68 tasks, the benchmark's own question and scoring, and no benchmark task anywhere in training. Then we told the model the measured speed of our graphics card, asked, and timed its real runs.

| Estimating its own time | Typically off by (lower is better, 1x is perfect) | Within 2x of the real time (higher is better) |
|---|--:|--:|
| GPT-5.5 (published) | 3.5x | 12% |
| Claude Opus 4.8 (published) | 2.9x | 25% |
| Tiny Aya, untrained, told its speed | 5.5x | 4% |
| **Tiny Aya with our adapter** | **1.6x** | **72%** |

Ours was closer than GPT-5.5 on 60 of the 68 tasks and closer than Claude Opus 4.8 on 55, and closer than both on each of the seven kinds of task. That is the result we said in advance would count as success. The paper also trained an 8B open model for this job; its best version reaches a correlation of 0.44 with the real times, and ours reaches 0.56.

## The pipeline: CuratorKIT → AlignTune → AuditKIT

Each library's output is the next one's input.

1. **CuratorKIT** gates the 1,353 measured training prompts: schema gate, exact and MinHash deduplication, secrets gate, and one rule of ours as its hook (drop a prompt when a measured run hit the generation cap, because its label is then only a lower bound). 1,342 pass; 10 are dropped by the cap rule and 1 by the secrets gate.
2. **AlignTune** trains the LoRA on what passed, with its SFT trainer (TRL backend, PEFT LoRA, loss on the answer only). The result is the adapter `G1-kit`.
3. **AuditKIT** tests the untrained model and `G1-kit` on the 68 tasks with three scorers of ours and its paired bootstrap, and runs ARC-Easy and ARC-Challenge with the adapter off and on.

AuditKIT's paired bootstrap confirms the gain: against the untrained model, the mean log error falls from 1.78 to 0.50 and the share within 2x rises from 4% to 76% (p = 0.001). It also found the cost. With the adapter on, ARC-Easy accuracy drops from 73.0% to 71.1% and ARC-Challenge from 42.2% to 40.2%: small, but statistically significant. The adapter is a LoRA, so a deployment can switch it on only for the time question and keep the untrained model for everything else.

**What it costs to run.** 1,000 estimates take about 15 minutes of this GPU, 3 to 5 cents at a public hourly price. The same tokens at GPT-5.5 or Claude Opus 4.8 list prices cost $3.16 to $3.27, and even the bare question alone costs 57 to 60 cents, before any hidden reasoning tokens, which we cannot count ([article §8](paper/article.md#8-cost-and-roi)).

## Read it with care

- **Each model is judged on its own clock.** The result says a 3-billion-parameter model can be taught to know its own running time better than the largest models know theirs. It says nothing about which model is smarter.
- **Ours is told its machine's speed; the big models were asked cold.** Without the speed in its prompt, ours is worse than both (off by 4.6x), because it has no other way to know how fast the machine is. Measuring the speed takes seconds on any deployment, but the big models were not given that help. The big models were never trained for this question either, so it is not an even contest.
- **Their mistake is mostly one of scale.** Both say about three times too long for nearly everything. Claude Opus 4.8 sorts tasks from quick to slow better than ours does (correlation 0.81 against 0.71), and one correction factor would bring it ahead of us. Finding that factor takes measurements, which is what our method supplies.
- **The untrained model already knows more than it says.** A simple readout of its internal state predicts its time about as well as the adapter does when both see the full task. The adapter teaches it to say so.
- **Smaller caveats.** Ours guesses short (about three-quarters of the real time). The range you get by asking it 16 times is too narrow; a range set from held-out training prompts holds 91% of the real times, but it is wide (2.8x either way). The adapter was trained once, with one seed, so we cannot put an error bar on the recipe itself. This is one machine, and English only.
- **One earlier adapter.** The success criterion was first applied, on 14 November, to an adapter we trained with our own code rather than the track's libraries. It is not part of this submission; [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md) keeps its log.

Every control, the full tables and the commands are in [`docs/METATIME.md`](docs/METATIME.md).

## How it works

1. **Measure.** We collected 1,523 public prompts (Dolly-15k, CodeAlpaca-20k, no_robots, GSM8K, and 100 coding tasks from an earlier study of ours), after removing 21 that looked like a benchmark task. We ran the untrained model on each one three times and recorded how long its answers were. 1,353 prompts are for training and 170 are held out for checking.
2. **Curate.** CuratorKIT gates the training prompts (1,353 → 1,342).
3. **Train.** AlignTune trains the adapter to answer "how long will this take you?" with the work first and the time second, for example `Prompt: about 380 tokens. Response: about 520 tokens.` and then `Time: 15.4 seconds`. Each training prompt states a machine speed, drawn from a wide range (writing speeds from 2 to 2,500 tokens a second), and the time in the answer is worked out from the token counts at that speed. So the model learns how much work a task is for it, and that the time depends on the machine.
4. **Ask and check.** On the benchmark, the prompt states the speed we measured on our own machine (an RTX 4060 Ti, with the adapter on: about 30 tokens a second writing and 3,300 reading), followed by the benchmark's own question. The benchmark's own parser reads the time out of the answer. We compare it with the median of five timed runs of the same model on the full task. AuditKIT runs its own check of the gain and of general ability.

Compute: a local RTX 4060 Ti and about $40 of Hugging Face compute.

## Check it yourself

The records are in `results/`, so this needs only a CPU.

```bash
git clone https://github.com/isHeSatoshi/anthropochrony.git
cd anthropochrony
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is not installed
uv sync --locked
uv run pytest                              # expect: 22 passed
uv run python scripts/run_notebook.py      # expect: notebook executed without errors
uv run python scripts/metatime/report.py   # prints the tables of docs/METATIME.md
```

`pytest` recomputes every reported row from the records and checks that no benchmark task text is committed. The [notebook](notebooks/benchmark_head_to_head.ipynb) does the same step by step and draws the figure. Retraining and timing need a 16 GB GPU; see [`docs/METATIME.md`](docs/METATIME.md#reproduce).
## Try it on your own machine

[`space/`](space/) is a small Gradio app. It measures your CPU's speed, asks the untrained model and our adapter how long your task will take, then runs the task and times it. It uses the 4-bit files in [`isHeSatoshi/anthropochrony-tiny-aya-global-g1-kit-gguf`](https://huggingface.co/isHeSatoshi/anthropochrony-tiny-aya-global-g1-kit-gguf) and a llama.cpp build; [`space/README.md`](space/README.md) has the steps. A hosted Hugging Face Space needs a paid plan for this kind of app, so it runs locally.

## What is in this repository

| Path | What it is |
|---|---|
| [`paper/article.md`](paper/article.md) | The hackathon write-up |
| [`docs/METATIME.md`](docs/METATIME.md) | The full head-to-head: the rules, every control row, what it does not show, and the commands |
| [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md) | Every declaration, word for word, with the commit that pushed it |
| [`docs/JUDGES.md`](docs/JUDGES.md) | Where to look for each judging criterion |
| [`scripts/metatime/`](scripts/metatime/) | Building the training pool, measuring answer lengths, CuratorKIT, AlignTune training, asking, timing, scoring, AuditKIT, and the figure |
| [`results/metatime.json`](results/metatime.json), [`results/metatime/`](results/metatime/) | Every number in the tables, and the per-task records behind them (task ids and numbers, no benchmark text) |
| [`notebooks/`](notebooks/benchmark_head_to_head.ipynb), [`tests/`](tests/) | Recompute the rows from the records |
| [`space/`](space/) | The local demo |

## Where this came from

Earlier in the hackathon the project ran a first study on 1,465 Python coding tasks, with the same small model as a coding agent. It is no longer in this repository's tree and can be read at commit [`aa450e4`](https://github.com/isHeSatoshi/anthropochrony/tree/aa450e4c6a2fd6094b800d75fe26ad4216888796). The benchmark work reuses its training recipe, its ways of stating a machine's speed and 100 of its coding tasks.

## Limits

- The benchmark comparison is not an even contest, for the reasons under "Read it with care".
- It is one model, one machine, one training run, single-turn tasks and English. The rows for other machines' speeds use computed times, not runs on those machines.
- The adapter costs about 2 points of ARC accuracy when it is switched on.
- The rows for GPT-5.5 and Claude Opus 4.8 are the paper's published numbers. We did not rerun those models.
- No hosted demo; the demo runs locally.

Subject model: Cohere Labs' `CohereLabs/tiny-aya-global` (3.35B, CC-BY-NC-4.0). Adapter and data: CC-BY-NC-4.0. Code: Apache-2.0.
