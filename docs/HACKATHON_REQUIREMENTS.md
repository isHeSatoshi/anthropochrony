# Hackathon requirements and submission checklist

Version 1.1, 2026-11-09. This file is the checklist behind gate G10 (PLAN.md S10). Everything below was read from the official pages on 2026-11-09:

- Rules, Submit, FAQ, Schedule and Structure: https://hackathon.lexsi.ai/ (the `/tracks` and `/rules` URLs render the same page)
- Submission form, page 1 of 5: https://hackathon.lexsi.ai/submission/
- Kickoff slides (submission checklist and judging): https://hackathon.lexsi.ai/kickoff-slides/
- HF Blog Article eligibility: https://huggingface.co/docs/hub/blog-articles
- HF storage limits: https://huggingface.co/docs/hub/storage-limits
- HF Spaces plans and ZeroGPU: https://huggingface.co/docs/hub/spaces-overview and https://huggingface.co/docs/hub/spaces-zerogpu

The organizers "reserve the right to modify the schedule or event rules", and the kickoff checklist says "Complete details will be shared soon". Re-read the Rules and Submit sections, the form and the Discord announcements at S10, and again right before the user submits.

## 1. Dates (all AoE)

| When | What |
|---|---|
| Nov 5 (passed) | Registration closed. Only registered teams can submit. |
| **Nov 15, 23:59 AoE** *(updated on live site from initial Nov 14, 23:59 AoE)* | Submission deadline. You can submit more than once; the latest submission counts. "Late or incomplete submissions may not be considered." |
| Nov 16 | Judging. Up to 20 teams learn by Nov 16 that they are through to Demo Day. |
| Nov 16-21 | Shortlisted teams "prove their thesis", with a further $200 in HF credits. |
| Nov 22 | Demo Day: a live demo (in person in Bengaluru or online), then final judging. |

Internal target: everything ready ahead of the initial Nov 14, 12:00 AoE checkpoint (PLAN.md CP4).

## 2. Team, accounts and credits

- Teams have 1 to 3 members, and each person is in one team only. At least one member must join the Discord server (https://discord.gg/dtEDQ2Z3eg).
- HF credits: registered teams that joined Discord can redeem a hackathon coupon; the project budget is capped at $30.
- Public Hub namespace: `isHeSatoshi`. Model derivatives push LoRA adapters and GGUF artifacts rather than multi-copy full bf16 merges so every artifact remains compact and reproducible.
- Demo & write-up formats: the write-up is shipped as `paper/article.md` plus `README.md` (an explicitly accepted submission format), and `space/app.py` provides a self-contained 4-tab Gradio workbench that runs on CPU out of the box (with optional live local `llama.cpp` GGUF execution).

## 3. Rules and how Anthropochrony meets them

| # | Official rule | Anthropochrony | Evidence for G10 |
|---|---|---|---|
| R1 | "Projects must be developed during the hackathon." | Built Nov 9-14 | Git history, `RUNLOG.md` dates |
| R2 | "All submissions must chain together repos from their track into a real pipeline." Minimum: 2 repos from the track, used together. Preferred: 3 or more. | Track 1: CuratorKIT → AlignTune → AuditKIT, all three | Article section 2; `src/anthropochrony/curate`, `train`, `eval` |
| R3 | "Use a dense, open-weights LLM from Cohere Labs": Tiny Aya Base, Global, Fire, Water or Earth, or Aya Expanse 8B or 32B. FAQ: other models only "in a supplementary way, such as a teacher model, for validation, or an LLM as a judge." | Core model: Tiny Aya Global. Aya Expanse 8B appears only in M6. DeepSeek-V4-Pro (teacher), gpt-oss-120b (judge) and bge-small (B2 features) are supplementary. | Article section 3 |
| R4 | "Host all artifacts created on Hugging Face: Models, Adapters, Logs, Datasets." Kickoff slide: "Resulting models, adapters, logs, datasets, benchmarks: everything publicly accessible." | Model collection: adapters, GGUF, and merged M1-s0 if U6 allows. Data collection: tasks, traces, sft, eval, logs. | Collections; logged-out check |
| R5 | "Projects must be deployed publicly (e.g. a public demo or repository link) by the submission deadline." | Demo Space and public GitHub repo | URLs |
| R6 | "Showcase before-and-after metrics (more than one metric)." | Base vs Anthropochrony: MdAE, within-2×, Spearman ρ, anthropocentric-unit rate, parse rate; plus pass@1 retention | T2-T6 |
| R7 | The write-up must cover "the metrics chosen and why, the approach and rationale behind the choice of datasets, tasks, and models, and how the resulting model is meaningfully better for usage." | One subheading for each of the three items (PLAN.md §12.1) | Article |
| R8 | Write-up formats: "A HF Article" and "A HF Space (if relevant)". | Article (see §5 for access) and demo Space | URLs |
| R9 | Optional but preferred: "An ROI comparison of a closed-source model vs. the fine-tuned/aligned smaller model — self-hosting cost vs. closed-source token billing, using a realistic sample volume, GPU runtime hours, per-hour GPU pricing, and an estimated tokens/sec throughput." | T7 (PLAN.md §7.4) | Article section 7 |
| R10 | Originality: "unique and your own work". Open-source libraries and public datasets "may be used where permitted". | Our own code; licenses in article sections 3 and 4 | Article |
| R11 | Focus on one preferred track. Ideas outside the listed problems should be validated through a Discord ticket in #report-issues. | Track 1, problem statements #8 (primary), #2 (ROI) and #6 (only if M8 runs); U6 ticket | Form |
| R12 | Teams keep ownership; organizers may showcase projects. The code of conduct applies. | None needed | None |

## 4. Submission form

Page 1 of 5 (read on 2026-11-09):

- **Registered email** (required): "The exact email you or your teammate registered with. Any member of the team can submit."
- **Abstract** (required): "A summary of the submission in under 500 words."
- **Discord username** (optional): "We'll reach out here for any follow-up questions."

Pages 2 to 5 are not visible without filling page 1. The Submit section ("What you'll need") lists what they ask for:

| Field | Official requirement | Our value (fill in at S10) |
|---|---|---|
| HF model collection | "Add all models, adapters, etc. into a HF collection, with all individual repos and the collection public (un-gated)." | "Anthropochrony models" URL |
| HF dataset collection | "Add all datasets, logs, traces, benchmarks, curated or synthetic data, etc. into a HF collection, with all individual repos and the collection public." | "Anthropochrony data" URL |
| Write-up or article | "If HF articles are accessible, use them; otherwise, a GitHub repo README is also acceptable." | Article URL, or the README URL |
| GitHub repo | "If a repo is used, ensure the README covers reproducibility notes." Notebooks are submitted through a GitHub repo. | https://github.com/isHeSatoshi/anthropochrony |
| Repos and models used | | CuratorKIT 0.1.23, AlignTune 0.1.32, AuditKIT 1.1.6; Tiny Aya Global @ `af89d219` (plus Aya Expanse 8B if M6 ran) |
| Track(s) | | Track 1: Fine-tuning and Data Curation |
| Problem statements | | #8 Data-scarce domains (primary); #2 Replacing a closed-model API (ROI); #6 Reasoning with verifiable answers, only if M8 (GRPO) ran |

The form link is also emailed "a couple of days before the deadline". The user submits it (PLAN.md rule 8). Collect every value above in `paper/submission_fields.md` beforehand.

## 5. Write-up: required sections

From the kickoff slides' "Submission checklist", in this order:

1. Problem statement and approach, in brief
2. Hackathon repos used: which ones, for what, and which features
3. Models used and their licenses
4. Datasets and benchmarks used and their licenses
5. Model artifacts (weights, adapters, …) public, as an HF collection
6. Data artifacts (datasets, logs, benchmarks, …) public, as an HF collection
7. Experiments run, findings and what else was tested
8. With more credits: your approach, before and after results
9. Teammates tagged

PLAN.md §12.1 maps our content onto these sections. The three R7 items must appear explicitly.

**Write-up format.** Both HF Blog Articles and GitHub repository write-ups (`README.md` + `paper/article.md`) are accepted under the official rules. Keeping `paper/article.md` in version control guarantees full reproducibility of every table and figure link.

## 6. Judging

- **Validity check.** The organizers' team checks "public artifacts, submitted write-up, checklist, no duplicates", then tags each submission to the closest track or sub-track. A missing or gated artifact can make the submission invalid.
- **Scoring.** A pool of judges (startup founders, academics, industry research labs and the organizers' research teams) scores submissions double-blind. Each team gets more than one score, and normalized averages pick the 10 to 20 Demo Day teams.
- **Rubric:** write-up clarity /10, practical usefulness /10, code/notebook quality /10, overall review /100.
- **Demo Day:** a live demo, after which the judges decide the winners.

What this means for us (PLAN.md §12.1):
- **Clarity:** one hero figure, a results table with CIs, plain language.
- **Usefulness:** the demo, the scheduling and timeout simulations, and the real-deployment results.
- **Code quality:** tests, CI, pinned environments and the CPU-only reproduction notebook.

## 7. Licenses and publication

| Item | License or terms | Release rule |
|---|---|---|
| Tiny Aya Global | CC-BY-NC-4.0 plus the Cohere Labs Acceptable Use Policy; gated on HF (automatic approval) | Adapters and GGUF are public and un-gated under CC-BY-NC-4.0, with attribution and the AUP link. Merged weights are public only if U6 says yes. |
| Aya Expanse 8B | CC-BY-NC-4.0; gated | Same rule (M6 only) |
| CuratorKIT | LSAL v1.1, non-commercial | Research use |
| AuditKIT | LSAL v1.2. Under §1A, organizations need written permission for internal evaluation or benchmarking. | Individual hackathon use. Raise it in the U6 ticket if unsure. |
| AlignTune | MIT (the wheel bundles AGPL and LGPL parts) | Nothing to do |
| ClassEval | CC BY-NC 4.0 | Keep non-commercial. Never merge it into a permissively licensed file. |
| LeetCodeDataset, CodeContests | Problem statements are third-party copyright | Release IDs plus a loader. In `sft` and `eval`, replace the statement text with the ID, and ship `rebuild_messages.py`. |
| Our code | Apache-2.0 | Nothing to do |

## 8. Final checklist (gate G10): tick everything before the user submits

**Artifacts**
- [ ] The model collection is public, and every repo in it is public and un-gated: adapters (M1 ×3, M2 ×3, M3, and any P1 variants that ran), the GGUF, and merged M1-s0 if allowed.
- [ ] The data collection is public: tasks, traces, sft, eval, and logs (training logs, HF Jobs logs, `RUNLOG.md`, manifests).
- [ ] Every repo has a card: license, base model and revision, data, metrics, limitations, intended use and citation.
- [ ] A logged-out browser (a private window) opens every link: both collections, each repo, the article, GitHub and the Space.

**Write-up**
- [ ] The article (or README) has the 9 sections in order, plus the three R7 items.
- [ ] Before/after results: at least 2 metrics with 95% CIs (T2), plus capability retention (T5).
- [ ] An ROI section (T7) with volume, GPU hours, $/h, tokens/s, and the closed model's prices with their source and date.
- [ ] The novelty wording and the "claims to avoid" are respected (PLAN.md §1), and MetaTime is cited as concurrent work.
- [ ] Teammates are tagged.

**Code and demo**
- [ ] The GitHub repo is public, its README has reproducibility notes (PLAN.md §12.3), CI is green, and `make figures` and the notebook run on CPU from Hub data.
- [ ] The demo Space is public and works logged out (G9).
- [ ] No secrets anywhere: `git log -p | grep -E 'hf_[A-Za-z0-9]{30,}'` finds nothing, and the Space holds no write-capable token (PLAN.md S9.4).
- [ ] The public copies of the docs carry no account details (payment status, token names or scopes, private repos). Trim §2 of this file and PLAN.md U1/U2 before the repo goes public.

**Form**
- [ ] The abstract is under 500 words (check with `wc -w`).
- [ ] Every field value from §4 is in `paper/submission_fields.md`.
- [ ] The user has submitted, the confirmation is saved, and the time is logged in `RUNLOG.md`.
- [ ] After submission, change artifacts only to fix errors, and log each fix.

## 9. If shortlisted (Nov 16-22)

- Spend the $200 credit on the "with more credits" items in PLAN.md §10.1, and update article section 8 with their before/after results.
- Prepare a 5-minute live demo: one PROJ-8 task, base estimate vs Anthropochrony estimate, the live run with a stopwatch, then the hero figure.
- Keep the HF token valid until Nov 22. The user revokes it afterwards.
