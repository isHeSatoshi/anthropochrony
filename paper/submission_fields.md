# Submission form values

Fill the `TODO` placeholders below before submitting at https://hackathon.lexsi.ai/submission/. All other fields and numbers are verified against `results/*.json` and live public URLs.

| Field | Value |
|---|---|
| Registered email | `TODO: fill in your exact registered email` |
| Discord username (optional) | `TODO: fill in your Discord handle` |
| Teammates | `TODO: fill in teammate names/handles, or "Solo"` |
| HF model collection | https://huggingface.co/collections/isHeSatoshi/anthropochrony-models-6af31e769e666b4001e8c5f2 |
| HF dataset collection | https://huggingface.co/collections/isHeSatoshi/anthropochrony-data-6af31e776d6f33265b0c33a8 |
| Write-up | https://github.com/isHeSatoshi/anthropochrony (`README.md` + `paper/article.md`) |
| GitHub repo | https://github.com/isHeSatoshi/anthropochrony |
| Demo | Local 4-tab Gradio CPU workbench (`make demo` / `space/app.py`: benchmark and 8-speed-card explorer, static AST workload and hardware simulator, empirical SLA and closed-API ROI calculator, and optional live `llama.cpp` GGUF sandbox); no hosted HF Space; `anthropochrony estimate` CLI and Python API for local `llama.cpp` (`docs/USAGE.md`); 100-task (`X10`) real CPU and GPU deployment results and the 8-project (`PROJ-8`) CPU showcase in `README.md` and `paper/article.md` §7.7 |
| Repos and models used | CuratorKIT 0.1.23, AlignTune 0.1.32, AuditKIT 1.1.6; CohereLabs/tiny-aya-global @ af89d219b53ed9b13b8a4645f9c8028973510324; DeepSeek-V4-Pro (teacher control); BAAI/bge-small-en-v1.5 (tabular baseline) |
| Track | Track 1: Fine-tuning and Data Curation |
| Problem statements | #8 Data-scarce domains (primary), #2 Replacing a closed-model API (ROI and accuracy comparison) |

## Abstract (under 500 words)

Language models estimate their own runtime on a human developer's clock, predicting minutes or hours for coding tasks their decoding loop finishes in seconds. Monperrus and Stenqvist described this anthropocentric time bias; we claim no novelty for discovering it. On familiar task families at one fixed speed, 5-shot prompting (`E2`, 0.240 log10 error, 1.74x) and per-family lookup tables (`B1`, 0.123, 1.33x) compete with fine-tuning. On an unseen multi-method class family (`CLASS`, 93 tasks) or 64x slower hardware (1.5 tok/s), both shortcuts degrade (`E2`: 0.503, 3.18x; `B1`: 0.336, 2.17x).

Across 1,465 executable Python tasks curated with CuratorKIT 0.1.23, we ran eight 3-attempt sandboxed trajectories per task on `CohereLabs/tiny-aya-global` (3.35B), trained rank-16 QLoRA adapters with AlignTune 0.1.32, re-measured each checkpoint on held-out tasks, and evaluated with AuditKIT 1.1.6 under a pre-registered plan (`prereg-v1`). Without a speed card, the base model (`E0`) overestimates 55-fold on `test` (1.760 log10 error). Adding a speed card (`E1`) leaves 1.375 (23.7x). Fine-tuning on curated measured runs (`M1-S0`) cuts `CLASS` error to 0.300 (2.00x; Holm p = 0.00025 vs `E2`) and `test` error to 0.242 (1.75x), beats SFT on teacher guesses (`M2-S0`: 0.287 on `test`, Holm p = 0.00056), and preserves `pass@1`.

`M1-S0` predicts its output tokens better than 5-shot `E2` (0.268 vs 0.313 on `test`; 0.314 vs 0.476 on `CLASS`), yet writing headline seconds before `(tokens, attempts)` introduces 0.272 log10 division error and slow-hardware failure (0.492 at 1.5 tok/s). Pairing its predicted tokens and attempts with the exact timing formula (`M1-Struct`, zero retraining) cuts `CLASS` error to 0.269 (1.86x, 58.1% within 2x), `test` error to 0.213 (1.63x, 71.8% within 2x), and 1.5 tok/s `test` error 3.17x to 0.155. `M1-Struct` beats `DeepSeek-V4-Pro` point accuracy (0.294 on `CLASS`, 0.244 on `test`) at $16.01 vs ~$1,700 per million batched estimates. Split-conformal calibration (`M1-Struct+CP`) raises fresh-trajectory coverage to 75.8% on `test` and cuts timeout false aborts 10.5x. On a desktop CPU (100 tasks), within-2x accuracy rises from 3.0% (base) to 63.0% (`M1-Struct`); on a consumer GPU running the same tasks 5.95x faster, `M1-Struct` predicts each task's speedup within 2x for 82%.

On a public benchmark we did not build (MetaTime, 68 single-turn tasks; protocol committed before the first run), a second adapter trained the same way on 1,353 public prompts has a median absolute natural-log error of 0.38 against its own measured time (79% within 2x), versus 1.24 and 1.06 for the published self-estimates of GPT-5.5 and Claude Opus 4.8 (paired 95% intervals exclude zero; closer on 62 and 61 of 68 tasks). Each model is scored on its own clock, and Claude Opus 4.8 ranks tasks equally well (r 0.81 vs 0.78). Ours is told its speed; a variant trained for one machine and asked cold, as theirs were, scores 0.42.

Limitations: one seed (`S0`); GBDT regressor `B2` leads fixed-speed point error (0.137 `test`, 0.130 `CLASS`); `M1-Struct` and the benchmark arm are post-hoc; uncurated `M3` leaked 40 `CLASS` rows, so we draw no conclusions from it; local CPU demo only; IFEval dipped 5.0 points.
