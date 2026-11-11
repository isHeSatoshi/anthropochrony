# Submission form values (the user submits the form)

Fill the bracketed items on Page 1 of https://hackathon.lexsi.ai/submission/. Everything else is final and verified against `results/` and live public URLs.

| Field | Value |
|---|---|
| Registered email | [the exact registered email] |
| Discord username (optional) | [username] |
| HF model collection | https://huggingface.co/collections/isHeSatoshi/anthropochrony-models-6af31e769e666b4001e8c5f2 |
| HF dataset collection | https://huggingface.co/collections/isHeSatoshi/anthropochrony-data-6af31e776d6f33265b0c33a8 |
| Write-up | https://github.com/isHeSatoshi/anthropochrony (`README.md` + `paper/article.md`) |
| GitHub repo | https://github.com/isHeSatoshi/anthropochrony |
| Demo | Self-contained 4-tab Gradio CPU workbench (`make demo` / `space/app.py`: hardware physics explorer, safe AST workload + exact-timing simulator, live SLA/ROI calculator, and optional live `llama.cpp` GGUF sandbox); 100-task real CPU deployment results in `paper/article.md` §7.7 |
| Repos and models used | CuratorKIT 0.1.23, AlignTune 0.1.32, AuditKIT 1.1.6; CohereLabs/tiny-aya-global @ af89d219; DeepSeek-V4-Pro (teacher control); BAAI/bge-small-en-v1.5 (tabular baseline) |
| Track | Track 1: Fine-tuning and Data Curation |
| Problem statements | #8 Data-scarce domains (primary), #2 Replacing a closed-model API (ROI and accuracy comparison) |

## Abstract (under 500 words)

Language models routinely estimate their own runtime on a human developer's clock, predicting minutes or hours for coding tasks their decoding loop finishes in seconds (anthropocentric time bias, Monperrus & Stenqvist; "Anthropochrony" is our project name, not a novelty claim). In-distribution—when demonstrations match the target task families at one fixed speed—5-shot prompting (`E2`, 0.240 log10 error) and family medians (`B1`, 0.123) look competitive with fine-tuning. Out-of-distribution, on an unseen multi-method class family (`CLASS`, 93 tasks) or 64x slower hardware (1.5 tok/s), both shortcuts collapse (`E2`: 0.503, 30.1% within 2x; `B1`: 0.336).

Across 1,465 executable Python tasks curated with CuratorKIT, we ran eight 3-attempt sandboxed trajectories per task on `CohereLabs/tiny-aya-global` (3.35B), trained rank-16 LoRA adapters with AlignTune, re-measured each fine-tuned checkpoint on held-out tasks (`own` vs. `base` target sources tracked throughout), and evaluated with AuditKIT under a pre-registered protocol. Without a speed card, the base model overestimates 55-fold (~7 min vs. ~8 s); adding a speed card stops hour-scale units (`H1` not met as pre-registered) yet leaves 1.38 log10 error (0.4% within 2x). Fine-tuning on curated measured trajectories (`M1-S0`, one seed) cuts `CLASS` OOD error to 0.300 (50.5% within 2x; Holm p = 0.00025 vs. `E2`) and `test` error to 0.242 (65.9% vs. 55.3% within 2x; Holm p = 0.0032), beats SFT on frontier-teacher guesses (`M2-S0`: 0.287; Holm p = 0.00056), and preserves pass@1 (-2.2 to +3.0 pp CI) and human-developer estimates.

Our headline diagnostic explains what caps raw verbal SFT: `M1-S0` already predicts its own output tokens significantly better than 5-shot `E2` (0.268 vs. 0.313 on `test`; 0.314 vs. 0.476 on unseen `CLASS`), but emitting headline seconds before its `(tokens, attempts)` breakdown introduces 0.272 log10 single-pass division error and 1.5 tok/s collapse (0.492). Pairing `M1-S0`'s workload decomposition with an exact timing model (`M1-Struct`, zero retraining, train-only execution medians) cuts error to 0.269 on `CLASS` (lifting Spearman rho 4.6x to 0.369 and winning `CLASS` OOD SJF queue scheduling at 1.183x oracle) and 0.213 on `test` (71.8% within 2x)—beating DeepSeek-V4-Pro (0.294 / 0.244) at $16 vs. ~$1,700 per million batched estimates while slashing 1.5 tok/s error 3.17-fold to 0.155 (73.7% within 2x). Split-conformal calibration restores 75.8% fresh-trajectory `test` coverage (64.9% OOD), cuts timeout false aborts 10.5-fold (1.33%), and saves 26.4% in Edge/Datacenter SLA routing. On a desktop CPU (4-bit GGUF, 100 tasks), within-2x wall-clock accuracy rises from 3% (base) and 52% (verbal `M1-S0`) to 63% (`M1-Struct`).

We report every failure: one seed under the $30 cap; static GBDT `B2` reaches 0.137 / 0.130, and `val` selection of an `M1-Struct + B2` blend failed (picking w = 0.80 on `base` `val` targets, scoring 0.178 / 0.202 on `own` targets vs. `B2` alone); historical `M3` (0.210) had +62% algorithmic rows plus 40 leaked `CLASS` rows (remediated); and IFEval dipped 5 points.
