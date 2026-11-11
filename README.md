# Anthropochrony

### Ask an AI how long a job will take and it says days. It finishes in minutes.

_Built for the Lexsi AI Safety & Alignment Hackathon, Track 1._

[Write-up](paper/article.md) · [For judges](docs/JUDGES.md) · [Technical tables](docs/README-technical.md) · [Models](https://huggingface.co/collections/isHeSatoshi/anthropochrony-models-6af31e769e666b4001e8c5f2) · [Data](https://huggingface.co/collections/isHeSatoshi/anthropochrony-data-6af31e776d6f33265b0c33a8)

![Estimated vs measured time](paper/figures/F1-estimated-vs-measured.png)

## The problem is everywhere

Ask a frontier model how long a piece of work will take. Give Claude a refactor, a new feature, a migration, and you'll hear "this is a few days of work" or "figure on a week". Then you let it run, and it's done in minutes.

The model isn't lying. It learned from text written by people, and when people say how long a job takes, they mean how long a person takes. So you get the time a human engineer would need, from something that isn't one. It's like asking a calculator how long long division takes and hearing "about ten minutes, with a pencil".

We wanted to know how bad this is, so we tested it on a small model where we could measure everything. `tiny-aya-global` writes a Python function in about 8 seconds. Before each task, we asked how long it would take.

"About two hours."

We ran this on 1,465 tasks. On average the guess was 55 times too long. Telling it "you generate about 96 tokens a second" barely helped. It nodded and ignored the number.

This matters because estimates drive decisions: how to queue jobs, when to time out a run, whether a task goes to a phone or a server, what to tell a customer about when their work will be done.

## So we taught it to know itself

First we measured the truth: we ran the model on every task, eight times each, and wrote down how long each run really took. Then we trained small adapters (LoRA) on those real times, so the model learns what its own work costs on a machine with a stated speed.

That helped, but it still got something strange wrong. We asked for its estimate on a slow machine (1.5 tokens a second) and it wrote:

> 550 tokens at 1.5 tokens per second is about 2.1 minutes.

550 divided by 1.5 is 6.1 minutes. The model had committed to "2.1 minutes" first and then made the maths agree with it.

The useful part is what the fine-tuning did learn: how much work a task is, meaning how many tokens it will write and how many tries it will need. It predicts those better than a model shown five worked examples: its token count is off by 1.9x against 2.1x on familiar tasks, and by 2.1x against 3.0x on a kind of task it never saw. The division it cannot do. So we let it predict tokens and tries, and a plain formula turns them into seconds. We call that version M1-Struct. It needs no extra training.

## Did it work?

A few tasks the model had never seen, in seconds:

| Task | What really happened | Untrained model said | Ours said |
|---|--:|--:|--:|
| HumanEval/138 | 2.3 | 424 | 4.8 |
| HumanEval/116 | 3.3 | 424 | 3.3 |
| BigCodeBench/903 | 11.3 | 300 | 7.9 |
| BigCodeBench/845 | 11.2 | 424 | 15.2 |

Over the full test set, the untrained model is off by about 57 times and ours by about 1.6 times. On a machine 64 times slower than the one it trained on, the adapter alone is off by 3 times and with the formula by 1.4 times.

We also put it on an ordinary CPU and gave it eight bigger jobs: a todo app, a URL shortener, a bank ledger, a snake game. They took 99 to 271 seconds. The untrained model predicted 7 minutes to 2.5 hours. The tuned one said 2 to 4.5 minutes. Neither version passed the hidden tests on those jobs, so this is about timing, not about the code being right.

One honest comparison: a plain table of average times per kind of task (on the small model) is about as accurate as our adapter on familiar tasks. The adapter pulls ahead on a kind of task it never saw, and on slow hardware.

## Does it work on other models and other tasks?

We tested two ways it could fail to carry over.

**A different, bigger model.** We repeated the whole method on Aya Expanse 8B, a different Cohere model with 2.4 times the parameters: measure it, train an adapter on what it measured, and compare. Time estimates at the same reference speed, off by how many times (test tasks / tasks of a new kind):

| Estimator | Familiar tasks | New kind of task |
|---|--:|--:|
| Untrained, no hint | 65x | 83x |
| Untrained, told its speed | 15x | 19x |
| Five worked examples | 1.5x | 2.5x |
| One average time per speed | 1.6x | 2.1x |
| **Adapter** | **1.4x** | **1.4x** |
| **Adapter + formula** | **1.3x** | **1.5x** |

The pattern from the small model repeats, including the token counts (off by 1.4x against 1.6x with examples). The untrained model is wildly off, the adapter fixes it, the formula rescues slow hardware (2.8x off with the adapter alone at 1.5 tokens a second, 1.3x with the formula), and the adapter's lead is clear on the kind of task it never saw. On familiar tasks its margin over a plain average is small. One seed, 4 runs per task instead of 8, trained with the same recipe but not through AlignTune.

**Brand-new tasks.** We wrote 30 coding tasks that are in none of our data (small functions up to 100-line modules) and ran `tiny-aya-global` on an ordinary CPU with real tests. These were hard: it solved 3 of 30. The untrained model was off by 7.3 times. The adapter was off by 1.9 times, with 17 of 30 estimates within 2x. The formula made it worse here (3.4 times) because the model needed all three tries on 27 tasks and the adapter had assumed fewer. And inside a size class, neither version ranked tasks well: its overall ranking comes mostly from task size, not from telling apart tasks of similar size.

What we have not tested: anything that is not Python coding, and a second seed.

## Limits

- One training run per model, so small gaps between our own variants mean little.
- One variant (M3) was trained on data that included some hidden test tasks. We do not use it for any conclusion.
- The formula idea came after we saw the first results, so it is not part of the plan we registered in advance. The same goes for the Aya Expanse 8B and brand-new-task tests.
- Coding tasks only, and no live hosted demo yet. It runs on your machine.

## Try it

```bash
uv sync && make figures    # rebuild every table and figure on CPU
make demo                  # the local demo
```

Built with the hackathon stack: CuratorKIT for the data, AlignTune for the training, AuditKIT for the evaluation. Subject model: `CohereLabs/tiny-aya-global` (3.35B).
