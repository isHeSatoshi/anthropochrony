# Anthropochrony

### Ask any AI how long a job will take. It answers for a human. It finishes in minutes.

_Built for the Lexsi AI Safety & Alignment Hackathon, Track 1._

[Write-up](paper/article.md) · [For judges](docs/JUDGES.md) · [Technical tables](docs/README-technical.md) · [Models](https://huggingface.co/collections/isHeSatoshi/anthropochrony-models-6af31e769e666b4001e8c5f2) · [Data](https://huggingface.co/collections/isHeSatoshi/anthropochrony-data-6af31e776d6f33265b0c33a8)

![Estimated vs measured time](paper/figures/F1-estimated-vs-measured.png)

## The problem is everywhere

Ask a frontier model how long a piece of work will take. Hand Claude a refactor, a new feature, a migration, and it will often tell you it's "a few hours of work", or "two to three days". Then you let it run, and it's done before your coffee cools.

It isn't lying. It learned from text written by people, and when people say how long something takes, they mean how long a person takes. The model gives you a human's answer to a question about itself.

We wanted to know how bad this is, so we tested it on a small model where we could measure everything. `tiny-aya-global` writes a Python function in about 8 seconds. Before each task, we asked how long it would take.

"About two hours."

We ran this on 1,465 tasks. On average the guess was 55 times too long. Telling it "you generate about 96 tokens a second" barely helped. It nodded and ignored the number.

This matters because estimates drive decisions: how to queue jobs, when to time out a run, whether a task goes to a phone or a server, what to tell a customer about when their work will be done.

## So we taught it to know itself

First we measured the truth: we ran the model on every task, eight times each, and wrote down how long each run really took. Then we trained small adapters on those real times, so the model could learn what its own work costs.

That helped. But it still got something strange wrong. We asked for its estimate on a slow machine (1.5 tokens a second) and it wrote:

> 550 tokens at 1.5 tokens per second is about 2.1 minutes.

550 divided by 1.5 is 6.1 minutes. The model had committed to "2.1 minutes" first and then made the maths agree with it.

That told us something. The model is actually decent at guessing how much work a task is: how many tokens it will write, how many tries it will need. What it can't do is the division. So we stopped asking it to divide. We take its guess for tokens and tries, and a plain formula turns that into seconds. We call that version M1-Struct. It needs no extra training.

## Did it work?

A few tasks the model had never seen, in seconds:

| Task | What really happened | Untrained model said | Ours said |
|---|--:|--:|--:|
| HumanEval/138 | 2.3 | 424 | 4.8 |
| HumanEval/116 | 3.3 | 424 | 3.3 |
| BigCodeBench/903 | 11.3 | 300 | 7.9 |
| BigCodeBench/845 | 11.2 | 424 | 15.2 |

Then we put it on an ordinary CPU and gave it eight bigger jobs: a todo app, a URL shortener, a bank ledger, a snake game. They took 99 to 271 seconds. The untrained model predicted 7 minutes to 2.5 hours. The tuned one said 2 to 4.5 minutes. Neither version passed the hidden tests on those jobs, so this is about timing, not about the code being right.

Over the full test set, the untrained model is off by about 57 times and ours by about 1.6 times. On a machine 64 times slower than the one it trained on, the adapter alone is off by 3 times and with the formula by 1.4 times.

## Where we lost

Be warned: the boring tools are strong. For familiar kinds of tasks, a table of average times per task type, or a small statistical model with no AI in it, is as accurate as our adapter or better. A statistical model also did well on the one task family we kept hidden. We think our approach is worth having because it reads the task and responds to the hardware, which a lookup table can't. But we won't tell you it beat every baseline, because it didn't.

We also had $30 and one training run per model, so small gaps between our own variants don't mean much. One variant, trained without cleaning the data, accidentally saw some of the hidden test tasks. We don't draw any conclusion from it. The formula idea came after we saw the first results, so it isn't part of the plan we registered in advance. And we haven't hosted a live demo yet. It runs on your machine.

## Try it

```bash
uv sync && make figures    # rebuild every table and figure on CPU
make demo                  # the local demo
```

Built with the hackathon stack: CuratorKIT for the data, AlignTune for the training, AuditKIT for the evaluation. Subject model: `CohereLabs/tiny-aya-global` (3.35B).
