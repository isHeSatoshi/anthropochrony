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
| Demo | Local 4-tab Gradio CPU workbench (`make demo` / `space/app.py`: benchmark and 8-speed-card explorer, static AST workload and hardware simulator, empirical SLA and closed-API ROI calculator, and optional live `llama.cpp` GGUF sandbox); no hosted HF Space; 100-task (`X10`) and 8-project (`PROJ-8`) real CPU deployment results in `README.md` and `paper/article.md` §7.7 |
| Repos and models used | CuratorKIT 0.1.23, AlignTune 0.1.32, AuditKIT 1.1.6; CohereLabs/tiny-aya-global @ af89d219b53ed9b13b8a4645f9c8028973510324; DeepSeek-V4-Pro (teacher control); BAAI/bge-small-en-v1.5 (tabular baseline) |
| Track | Track 1: Fine-tuning and Data Curation |
| Problem statements | #8 Data-scarce domains (primary), #2 Replacing a closed-model API (ROI and accuracy comparison) |

## Abstract (under 500 words)

Language models routinely estimate their own runtime on a human developer's clock, predicting minutes or hours for coding tasks their decoding loop finishes in seconds. Monperrus and Stenqvist described this anthropocentric time bias; "Anthropochrony" is our project name, and we claim no novelty for discovering the bias. On familiar task families at one fixed speed, 5-shot prompting (`E2`, 0.240 log10 error, 1.74x) and per-family lookup tables (`B1`, 0.123, 1.33x) compete with fine-tuning. On an unseen multi-method class family (`CLASS`, 93 tasks) or 64x slower hardware (1.5 tok/s), both shortcuts degrade (`E2`: 0.503, 3.18x, 30.1% within 2x; `B1`: 0.336, 2.17x, 46.2% within 2x).

Across 1,465 executable Python tasks curated with CuratorKIT 0.1.23, we ran eight 3-attempt sandboxed trajectories per task on `CohereLabs/tiny-aya-global` (3.35B), trained rank-16 QLoRA adapters with AlignTune 0.1.32, re-measured each checkpoint on held-out tasks (`own` vs `base` targets tracked throughout), and evaluated with AuditKIT 1.1.6 under a pre-registered plan (`prereg-v1`). Without a speed card, the base model (`E0`) overestimates 55-fold on `test` (424 s vs ~8 s real, 1.760 log10 error). Adding a speed card (`E1`) stops hour-scale units (`H1` not met as pre-registered) while leaving 1.375 log10 error (23.7x, 0.4% within 2x). Fine-tuning on curated measured runs (`M1-S0`, one seed) cuts `CLASS` error to 0.300 (2.00x, 50.5% within 2x; Holm p = 0.00025 vs `E2`) and `test` error to 0.242 (1.75x, 65.9% within 2x; Holm p = 0.0032), beats SFT on teacher guesses (`M2-S0`: 0.287 on `test`, Holm p = 0.00056), and preserves `pass@1` (-2.2 to +3.0 pp CI) and human-developer estimates.

Inspecting `M1-S0` outputs explains what caps verbal SFT. `M1-S0` predicts output tokens better than 5-shot `E2` (0.268 vs 0.313 on `test`; 0.314 vs 0.476 on `CLASS`), yet writing headline seconds before `(tokens, attempts)` introduces 0.272 log10 division error and slow-hardware failure (0.492 at 1.5 tok/s). Pairing `M1-S0`'s predicted tokens and attempts with the exact timing formula (`M1-Struct`, zero retraining, train-only execution medians) cuts `CLASS` error to 0.269 (1.86x, 58.1% within 2x, lifting Spearman rho 4.6x to 0.369 and tying `DeepSeek-V4-Pro` on `CLASS` SJF scheduling at 1.18x oracle) and `test` error to 0.213 (1.63x, 71.8% within 2x). `M1-Struct` beats `DeepSeek-V4-Pro` point accuracy (0.294 on `CLASS`, 0.244 on `test`) at $16.01 vs ~$1,700 per million batched estimates and cuts 1.5 tok/s `test` error 3.17x to 0.155 (73.7% within 2x). Split-conformal calibration (`M1-Struct+CP`) raises fresh-trajectory coverage to 75.8% on `test` (64.9% on `CLASS`), cuts timeout false aborts 10.5x (1.33%), and cuts 15s edge-vs-datacenter SLA violations from 31.2% to 6.3% while saving 26.4% vs always-datacenter ($2.04 vs $2.77 per 1k runs). On desktop CPU (`llama.cpp` Q4_K_M, 20 tok/s, 100 tasks), within-2x accuracy rises from 3.0% (base) and 52.0% (`M1-S0`) to 63.0% (`M1-Struct`).

Limitations: one seed (`S0`); GBDT regressor `B2` leads fixed-speed point error (0.137 `test`, 0.130 `CLASS`); `M1-Struct` is post-hoc; uncurated `M3` leaked 40 `CLASS` rows (6 `ClassEval` tasks) and was not retrained, so we draw no conclusions from `M3`; local CPU demo only; one 3.35B model; IFEval dipped 5.0 points.
