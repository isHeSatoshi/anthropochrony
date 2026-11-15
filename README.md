# Anthropochrony

### Ask an AI how long a task will take it and it answers on a human clock. We taught a small model to know its own.

_Built for the Lexsi AI Safety & Alignment Hackathon, Track 1._

[Benchmark head-to-head](docs/METATIME.md) · [For judges](docs/JUDGES.md) · [Write-up](paper/article.md) · [Check the numbers](notebooks/benchmark_head_to_head.ipynb) · [The first study](https://github.com/isHeSatoshi/anthropochrony/tree/aa450e4c6a2fd6094b800d75fe26ad4216888796) · [Models](https://huggingface.co/collections/isHeSatoshi/anthropochrony-models-6af31e769e666b4001e8c5f2) · [Data](https://huggingface.co/collections/isHeSatoshi/anthropochrony-data-6af31e776d6f33265b0c33a8)

![Head-to-head on the MetaTime benchmark](paper/figures/F9-metatime-head-to-head.png)

## The problem

Ask a frontier model how long a piece of work will take. Give Claude a refactor, a new feature, a migration, and you'll hear "this is a few days of work" or "figure on a week". Then you let it run, and it's done in minutes.

The model isn't lying. It learned from text written by people, and when people say how long a job takes, they mean how long a person takes. So you get the time a human engineer would need, from something that isn't one. It's like asking a calculator how long long division takes and hearing "about ten minutes, with a pencil".

This matters because estimates drive decisions: how to queue jobs, when to time out a run, whether a task goes to a phone or a server, what to tell a customer about when their work will be done.

There is now a public test for it. MetaTime, a paper under review at ICLR 2027, asks a model how long a task will take it, runs the task, and compares. Its 68 single-turn tasks are everyday requests: write a function, fix a bug, summarise a text, solve a puzzle, write an essay or a story, answer a question. The paper's public files include what GPT-5.5 and Claude Opus 4.8 answered and how long each one really took. Both say about three times too long for nearly everything.

## The result

We took `CohereLabs/tiny-aya-global`, an open model with 3.35 billion parameters, and trained one small adapter (a LoRA, 121 MB) on how long its own answers were to 1,353 ordinary public prompts. We wrote the rules down and pushed them before running anything: all 68 tasks, the benchmark's own question and scoring, and no benchmark task anywhere in training. Then we told the model the measured speed of our graphics card, and asked.

| Estimating its own time | Typically off by | Within 2x of the real time |
|---|--:|--:|
| GPT-5.5 (published) | 3.5x | 12% |
| Claude Opus 4.8 (published) | 2.9x | 25% |
| Tiny Aya, untrained, told its speed | 5.5x | 4% |
| **Tiny Aya with our adapter** | **1.5x** | **79%** |
| Tiny Aya with a second adapter, told nothing about its speed | 1.5x | 76% |

Ours was closer than GPT-5.5 on 62 of the 68 tasks, closer than Claude Opus 4.8 on 61, and closer on each of the seven kinds of task. That is the result we said in advance would count as success. The paper also trained an 8B open model for this job; its best version reaches a correlation of 0.44 with the real times, and ours reaches 0.66.

## Read it with care

- **Each model is judged on its own clock.** The result says a 3-billion-parameter model can be taught to know its own running time better than the largest models know theirs. It says nothing about which model is smarter.
- **The big models were asked cold, so we also asked ours cold.** The adapter in the main row is told its machine's speed, and without it is worse than both (off by 5.9x). So we trained a second adapter that learned one machine's speed during training and is told nothing (last row of the table). Asked exactly as the big models were, it is closer than GPT-5.5 on 62 of the 68 tasks and closer than Claude Opus 4.8 on 59. Timed on its own runs afterwards, it scores about the same (off by 1.5x, 82% within 2x). It only knows that one machine, and we added it after seeing the first results. The big models were never trained for this question, so it is still not an even contest.
- **Their mistake is mostly one of scale.** Both say about three times too long for nearly everything. Claude Opus 4.8 sorts tasks from quick to slow as well as ours does, and one correction factor would bring it level with us or a little ahead. Finding that factor takes measurements, which is what our method supplies.
- **The untrained model already knows more than it says.** A simple readout of its internal state predicts its time about as well as the adapter does when both see the full task. The adapter teaches it to say so.
- **Smaller caveats.** Ours guesses short (about three-quarters of the real time), the range you get by asking it 16 times is too narrow (a range set from held-out training prompts holds 93% of the real times, but it is wide: 2.5x either way), two repeat training runs of the main adapter came out worse than the first (off by 1.5x and 1.7x, with 71% and 75% within 2x, still well ahead of both big models), and this is one machine. The adapters are not uploaded yet.

Every control, the full tables and the commands are in [`docs/METATIME.md`](docs/METATIME.md).

## How it works

1. **Measure.** We collected 1,523 public prompts (Dolly-15k, CodeAlpaca-20k, no_robots, GSM8K, and 100 coding tasks from our first study), after removing 21 that looked like a benchmark task. We ran the untrained model on each one three times and recorded how long its answers were. 1,353 prompts are for training and 170 are held out for checking.
2. **Train.** The adapter learns to answer "how long will this take you?" with the work first and the time second, for example `Prompt: about 380 tokens. Response: about 520 tokens.` and then `Time: 15.4 seconds`. Each training prompt states a machine speed, drawn from a wide range (writing speeds from 2 to 2,500 tokens a second), and the time in the answer is worked out from the token counts at that speed. So the model learns how much work a task is for it, and that the time depends on the machine.
3. **Ask.** On the benchmark, the prompt states the speed we measured on our own machine (an RTX 4060 Ti: about 30 tokens a second writing and 3,300 reading), followed by the benchmark's own question. The benchmark's own parser reads the time out of the answer. We compare it with the median of five timed runs of the same model on the full task.

The second adapter has the same recipe with no stated speed: every training answer was computed at this one machine's speed. Everything was trained and timed on one desktop GPU, with no paid compute.

## Check it yourself

```bash
uv sync --locked
uv run pytest                              # recomputes the published rows and ours from the committed records
uv run python scripts/run_notebook.py      # the same check as a notebook, plus the paired comparison and the figure
uv run python scripts/metatime/report.py   # prints every table of docs/METATIME.md from results/metatime.json
```

None of this needs a GPU or the benchmark download. The commands that rebuild the training pool, retrain the adapters and rerun the timing are in [`docs/METATIME.md`](docs/METATIME.md#reproduce).

## What is in this repository

| Path | What it is |
|---|---|
| [`docs/METATIME.md`](docs/METATIME.md) | The full head-to-head: the rules, every control row, what it does not show, and the commands |
| [`scripts/metatime/`](scripts/metatime/) | Building the training pool, measuring answer lengths, training, asking, timing, scoring, and the figure |
| [`results/metatime.json`](results/metatime.json), [`results/metatime/`](results/metatime/) | Every number in the tables, and the per-task records behind them (task ids and numbers, no benchmark text) |
| [`notebooks/`](notebooks/benchmark_head_to_head.ipynb), [`tests/`](tests/) | Recompute the rows from the records |
| [`paper/article.md`](paper/article.md) | The hackathon write-up, covering both parts of the project |
| [`docs/JUDGES.md`](docs/JUDGES.md) | Where to look for each judging criterion |
| [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md) | Every declaration, word for word, with the commit that pushed it |

## Where this came from: the first study

The method was built in the project's first study, which is no longer in this repository's tree; it can be read at commit [`aa450e4`](https://github.com/isHeSatoshi/anthropochrony/tree/aa450e4c6a2fd6094b800d75fe26ad4216888796). There we ran the same small model as a coding agent on 1,465 Python tasks, eight runs each, and timed every run. Untrained, it guessed about 57 times too long. An adapter trained on the measured times, with a plain formula that turns its predicted tokens and tries into seconds (we call it M1-Struct), is off by about 1.6 times. It holds up on a kind of task it never saw and on a real CPU and GPU, and the method repeats on a second, larger model.

That study is where the three hackathon libraries are chained: CuratorKIT for the data, AlignTune for the training, AuditKIT for the evaluation. It has the released models and datasets, a command-line tool for your own agent, and a local demo. The benchmark adapter reuses its recipe, its ways of stating a machine's speed and 100 of its coding tasks, and was trained on one local GPU with plain PyTorch and PEFT.

## Limits

- The benchmark comparison is not an even contest, for the reasons under "Read it with care".
- It is one model, one machine and single-turn tasks. The rows for other machines' speeds use computed times, not runs on those machines.
- The main adapter's first training run is the best of three.
- The rows for GPT-5.5 and Claude Opus 4.8 are the paper's published numbers. We did not rerun those models.
- The benchmark comparison was planned after the first study. Its rules were fixed before its first run, but it is not part of the plan we registered at the start.
- The benchmark adapters are separate from the released coding model and are not uploaded yet. Each can be retrained with the commands in `docs/METATIME.md`.
- No live hosted demo yet.

Subject model: `CohereLabs/tiny-aya-global` (3.35B). Code: Apache-2.0.
