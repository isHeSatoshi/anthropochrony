# Submission form values (the user submits the form)

Fill the bracketed items. Everything else is final unless a number changes after the last evaluation.

| Field | Value |
|---|---|
| Registered email | [the exact registered email] |
| Discord username (optional) | [username] |
| HF model collection | https://huggingface.co/collections/isHeSatoshi/anthropochrony-models-6af31e769e666b4001e8c5f2 |
| HF dataset collection | https://huggingface.co/collections/isHeSatoshi/anthropochrony-data-6af31e776d6f33265b0c33a8 |
| Write-up | https://github.com/isHeSatoshi/anthropochrony (README + `paper/article.md`; no HF Blog access) |
| GitHub repo | https://github.com/isHeSatoshi/anthropochrony |
| Demo | Local Gradio app in the repo (`make demo`), no hosted Space; the 100-task CPU deployment results are in the article |
| Repos and models used | CuratorKIT 0.1.23, AlignTune 0.1.32, AuditKIT 1.1.6; CohereLabs/tiny-aya-global @ af89d219 |
| Track | Track 1: Fine-tuning and Data Curation |
| Problem statements | #8 Data-scarce domains (primary), #2 Replacing a closed-model API (ROI comparison) |

## Abstract (under 500 words)

Language models often describe their own work in human time. A coding agent that finishes a task in eight seconds may say it will take two hours. Monperrus and Stenqvist call this anthropocentric time bias; we use "anthropochrony" as a project name for it and do not claim the phenomenon is new. We ask a narrower question: can a small open model learn to estimate its own completion time from its own measured runs and the serving speed written in its prompt, without losing coding ability or changing what it says about human developers?

We study `CohereLabs/tiny-aya-global` on 1,465 verifiable Python tasks. Each task is run eight times in an agent loop with up to three attempts. Tokens, test time and stated prefill and decode speeds give a completion time; the median of the eight runs is the label. We built the pool and training sets with CuratorKIT, trained LoRA adapters with AlignTune, and scored every estimator with AuditKIT under a pre-registered protocol (tagged before any fine-tuned test run, deviations dated).

Without a speed card, the base model's typical answer is about seven minutes for work it finishes in about eight seconds (55-fold overestimate). With a card it stops using human units but stays erratic: median absolute error 1.38 log10, under 1% of answers within a factor of two. Training on the model's own measured labels (M1) cuts the error to 0.24 log10 on 255 held-out tasks (66% within 2×) and 0.30 on a task family never seen in training, and it follows speeds held out of training. Training on a teacher's guessed times is worse (0.29; Holm-adjusted p < 0.001). Pass@1 (difference CI -2.2 to +3.0 points) and human-frame answers are unchanged; IFEval fell 5 points on 200 prompts (within noise for one seed). On a desktop CPU with a 4-bit GGUF, 52% of the tuned model's estimates are within 2× of real wall-clock on 100 held-out tasks, against 3% for the base model. In simulation, its estimates lower mean completion time under shortest-job-first scheduling from 1.69× to 1.36× the oracle, and stopping a run at twice its estimate saves 19% of failed-run time with no false stops. At a million estimates a month, a saturated L40S costs about $16 against $452 for a small closed model (cost only; its accuracy was not measured).

We report what did not work. The pre-registered base-bias hypothesis (H1) was not met for the speed-card prompt. Five measured examples in the prompt match M1 in distribution (0.24), a feature-based regressor beats every language-model estimator (0.14), curation did not help (uncurated measured labels reach 0.21), stated 80% ranges cover 19% of runs, and extrapolation to a much slower speed fails. All results come from one training seed because of the compute cap, so differences of a few hundredths are not established.

We release adapters, GGUF files, tasks, traces, training sets, every estimate, logs and a CPU notebook that rebuilds the headline table.
