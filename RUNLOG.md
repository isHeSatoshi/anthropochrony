# Anthropochrony run log

Paid launches are listed below. USD values remain conservative timeout reservations
until reconciled against billing, not claims of actual charges.
Budget: $30 hard cap; $24 P0 target; $6 reserve. Single jobs estimated over
$5 require approval. Reservations use the job timeout, not optimistic runtime.
The machine-readable ledger is `artifacts/budget.json`.

| Job ID | Stage | Flavor | Minutes | USD | Outcome |
|---|---|---|---:|---:|---|
| 6af18ce30e0338d1bb8565ff | s0-cpu-smoke | cpu-basic | ≤2.0 | ≤0.0020 | Succeeded |
| 6af18d35e6320c66fad82768 | s0-stage-model | cpu-basic | ≤30.0 | ≤0.0300 | Succeeded |
| 6af18e090e0338d1bb856774 | s0-gpu-smoke-v1 | l40sx1 | ≤30.0 | ≤0.9000 | Failed after 10 outputs: git absent during manifest write. Outputs recovered privately; G0 not passed. |
| 6af190110e0338d1bb856a00 | s0-gpu-smoke-calibration-v2 | l40sx1 | ≤20.0 | ≤0.6000 | Succeeded: parity 20/20; REF S=96, P=26000; L40S; 297.8 output tok/s on 10-task smoke batch. |
| 6af193f2e6320c66fad830a7 | s2-stage-exec-v1 | cpu-basic | ≤30.0 | ≤0.0300 | Failed: bucket rejected NLTK archive's `.DS_Store` filename. No candidate execution. |
| 6af194ee0e0338d1bb85701f | s2-stage-exec-v2 | cpu-basic | ≤20.0 | ≤0.0200 | Failed: bucket does not retain POSIX executable bits/symlinks for a mounted venv. |
| 6af196a50e0338d1bb857220 | s2-stage-exec-v3 | cpu-basic | ≤20.0 | ≤0.0200 | Completed probe; blocked: kernel denies unprivileged namespaces. No candidate execution or full environment build. |
| 6af19c7be6320c66fad83c40 | s2-throughput-validation-v1 | l40sx1 | ≤120.0 | ≤3.6200 | Canceled after 600 s hardware-queue timeout. Terminal state verified; no start timestamp, logs, or measurements. Reservation retained pending billing. |
| 6af19f86ab34849891a7c566 | s2-throughput-validation-v2 | l40sx1 | ≤120.0 | ≤3.6200 | Reached L40S; all three probes completed generation but failed collecting preemptions. Cancellation verified. No G2 or qualified throughput result. |
| 6af1a64dab34849891a7cb8b | s2-throughput-validation-v3 | l40sx1 | ≤60.0 | ≤1.8100 | Two throughput probes and 100 actual validation loops completed. G2 accuracy failed; cancellation verified. Reservation retained pending billing. |

Total timeout reservations: **$10.652**. Actual billing has not been reconciled.

## Local validation and blockers

- The pinned tokenizer's `chat_template` is a single-entry dictionary, not a
  string. Selecting its existing default matches TRL 1.7.1 byte-for-byte.
  Twenty actual conversations passed assistant-mask and unchanged-rendering checks.
- CodeContests' upstream harness uses sized stdin. A pipe incorrectly presents
  `st_size == 0`. The regular-file fix restores both reproduced 1598_C / 1598_D
  references to 3/3 passes. G1 v4 passed with 1,308 kept tasks, including
  177 ALGO_IO and 93 CLASS tasks; 73 raw-only extra tasks remain separate.
- Latest CPU correctness/security suite: **162 passed** (25.06 s), including
  sandbox isolation and interrupted-round resume checks. Real pinned BGE ONNX
  feature smoke passed (384 dimensions), not a research performance result.
- S2 inline remote execution is blocked. HF's official sandbox documentation
  confirms Jobs disallow nested namespaces, mounts, and cgroup delegation.
  Managed pooled sandboxes are not an equivalent fallback: limits are not
  aggregate quotas and detached descendants can escape process-group timeouts.
  Keep local execution isolated; do not silently replace the security boundary.
- The user approved the HF-generation/local-sandbox topology, retaining all
  S2 sample-size and accuracy gates. No G2 pass is claimed.
- Production measurement and estimate jobs now record generation-window GPU
  telemetry and KV preemptions. The pure-Python AuditKIT 1.1.6 payload, including
  its license, passed an actual callable evaluation with site-packages disabled.
- Added reachable one-step and one-epoch training stages. These have local
  orchestration tests only; real GPU training has not run.
- HF pricing documentation confirms a $0.01/hour exposed-port fee in addition
  to L40S hardware, billed by the minute. Authenticated session reservations
  now include that fee and round partial minutes upward. The first session's
  upper bound was corrected by $0.02, without claiming an actual charge.
  Source: https://huggingface.co/docs/hub/jobs-pricing (2026-11-09).
- Hardware allocation now has a separate bounded 3,600 s queue window; server
  readiness still has 600 s after allocation. Authentication failures stop
  immediately. Failed startup cancels the remote job.
- Measurement now defaults to a separate 100-task K=8 pilot. Full measurement
  requires G3. Its report includes pass@1 by family, and each speed-card
  invariance condition must independently satisfy the original 0.05 limit
  on 100 validation tasks, against REF K=8. Fine-tuned measurement additionally
  requires G6 and the frozen preregistration.

Session 6af19f86ab34849891a7c566: cancellation accepted=True; hard timeout remains reserved. Billing not yet reconciled.

## 2026-11-09: S2 instrumentation correction

The failed probe matched every metric containing `preempt`, including vLLM's
`request_num_preemptions` histogram, which has no scalar `value`. The pinned
v0.33.0 upstream reader and logger confirm the required counter's exact name
is `vllm:num_preemptions`. An offline regression reproduces the mixed
counter/histogram schema. Probes now preserve generation and GPU samples
before reading final statistics, and stop on non-memory failures rather than
paying for further identical failures. No scientific threshold or sampling
setting changed. Retry v3 uses a one-hour hard timeout ($1.81 including its
authenticated port), rather than the prior two-hour reservation.

## 2026-11-09: tested data, training, and analysis safeguards

- CuratorKIT exports keep M1/M2 example IDs and prompts matched after joint
  length filtering. A family losing over 2% triggers a shared 3,072-token
  rebuild, with retained IDs and drop shares recorded.
- Human-preservation collection merges immutable cache snapshots and rejects
  truncated, empty, context-exhausted, or uncalled outputs. AuditKIT uses
  estimate IDs, not prompt text, to distinguish identical-looking inputs.
- Two-point learning-rate selection checks private checkpoint provenance,
  complete matched validation prompts, one-epoch step counts, and L40S evidence.
  All variants reuse the frozen selection. Actual training remains unrun.
- Private adapter publication preserves the pinned base identity, trainer state,
  log, and provenance. Authorization denial remains sticky even if an upstream
  callback catches its first error. One-step training needs no forwarded token.
- H4 and H6b keep seeds separate and pair templates correctly. H6a derives
  first-attempt outcomes from complete K=8 M1-seed-0 and base trajectories.
  Reports include hypotheses without inventing missing measurements.
- Unchanged launch payloads can be reused after a preview; changed or unexpected
  files are never overwritten. Failed CLI submissions retain their reservations
  rather than claiming zero actual billing.

Session 6af1a64dab34849891a7cb8b: cancellation accepted=True; hard timeout remains reserved. Billing not yet reconciled.

## 2026-11-09: S2 v3 terminal evidence and next-round preparation

- Live authenticated inspection verified the GPU job is `CANCELED`.
  Both throughput probes and all 100 real sequential validation loops completed.
- The selected 512-sequence configuration measured 185,346 output tokens in
  56.542 s: 3,278.0 tokens/s over 800 requests, zero preemptions, and
  92.16% mean GPU utilization over 57 generation-window samples. Model loading
  and startup are excluded from that throughput and generation-only cost rate.
- G2 did not pass: median absolute relative error 0.360750 exceeds 0.10;
  log-time R² 0.432992 is below 0.95. The eight-worker pooled/inline execution
  ratio, 1.009001, passed its 1.10 limit.
- The candidate 0.967021 s per-call latency correction yields median absolute
  relative error 0.023163 and log-time R² 0.997730 on these observations.
  It remains unadopted. No gate was waived and no training or subsequent
  paid measurement was launched.
- Prepared `measure --auto`: resumable full-batch generation, namespaced
  condition IDs, immutable request/config/source hashes, verified downloads,
  and secure local tests only after the GPU job is terminal. Partial checkpoint
  recovery, authorization failures, and no-double-payment behavior have
  offline tests; actual remote orchestration remains gated and unrun.
- The pilot quality report cannot substitute for G3. Independent 100-task
  validation-only REF K=8 and extreme-card K=2 experiments remain required.
- Ruff passed and the complete local suite passed: 162 tests in 25.06 s.
  Initial development failures (an unapplied patch and an incorrect fingerprint
  path) were corrected before this final run.
- Conservative reservations remain $10.652 of $30. Actual billing has not
  been reconciled. Work is paused after preparation, with G2 unresolved.

## 2026-11-09: G2 passed via the plan's latency fallback

- The raw formula failed G2 (median error 0.3608, log-R2 0.4330). Per PLAN S2 the
  fallback was adopted as a dated deviation (PREREGISTRATION section 7), not a waiver.
- L = 0.967 s per model call, in the single `PER_CALL_LATENCY_S` constant, used by
  labels, noise ceilings, baselines and teacher targets. Labels are `v2-latency`.
- Same 10% / 0.95 thresholds on all 100 loops: median error 0.0232, log-R2 0.9977.
  Held out: 200 half-splits worst error 0.0344, worst R2 0.9965; leave-one-family-out
  0.016 to 0.046.
- `results/gates/G2.json` now passed with `fallback_adopted`; the failed raw record is
  kept in `results/gates/G2-raw-attempt.json`. No new paid job. Local suite: 164 passed.

Reservation generate-fea580b281a7976399f98bd7: ≤$0.6000, before submitting the paid job.
| 6af1b4639737d9a9e08c508b | generate-fea580b281a7976399f98bd7 | l40sx1 | ≤20.0 | ≤0.6000 | Launched |

2026-11-09T11:57:52.131746+00:00: generation 6af1b4639737d9a9e08c508b terminal=COMPLETED; reservation retained pending billing. No candidate ran in the GPU job.

Reservation generate-a6208962562a6fc89b733565: ≤$0.6000, before submitting the paid job.
| 6af1b57d9737d9a9e08c51c3 | generate-a6208962562a6fc89b733565 | l40sx1 | ≤20.0 | ≤0.6000 | Launched |

2026-11-09T12:01:48.002798+00:00: generation 6af1b57d9737d9a9e08c51c3 terminal=COMPLETED; reservation retained pending billing. No candidate ran in the GPU job.

Reservation generate-cd073e41454f341ffe7a4435: ≤$0.6000, before submitting the paid job.
| 6af1b6699737d9a9e08c537f | generate-cd073e41454f341ffe7a4435 | l40sx1 | ≤20.0 | ≤0.6000 | Launched |

2026-11-09T12:10:37.971718+00:00: generation 6af1b6699737d9a9e08c537f terminal=COMPLETED; reservation retained pending billing. No candidate ran in the GPU job.

Reservation generate-5e280355be41979a27b1a026: ≤$0.6000, before submitting the paid job.
| 6af1b87e9737d9a9e08c56e7 | generate-5e280355be41979a27b1a026 | l40sx1 | ≤20.0 | ≤0.6000 | Launched |

2026-11-09T12:21:34.470962+00:00: generation 6af1b87e9737d9a9e08c56e7 terminal=COMPLETED; reservation retained pending billing. No candidate ran in the GPU job.

Reservation generate-6c5263f828f11043ac297d70: ≤$0.6000, before submitting the paid job.
| 6af1bb269737d9a9e08c5bdd | generate-6c5263f828f11043ac297d70 | l40sx1 | ≤20.0 | ≤0.6000 | Launched |

2026-11-09T12:56:08.564694+00:00: generation 6af1bb269737d9a9e08c5bdd terminal=COMPLETED; reservation retained pending billing. No candidate ran in the GPU job.

Reservation generate-73c0a0735af25861db6987ac: ≤$0.6000, before submitting the paid job.

## 2026-11-09: pilot attempt 1 findings, fixes, and repeat

- Automatic pilot worked end to end (GPU jobs terminal before local tests). Pilot
  quality gate passed, but ALGO_IO passed 0/13 tasks, triggering the plan's TACO fallback.
- The invariance run stopped on a negative uncached-token count after a length-capped
  attempt; fixed (see PREREGISTRATION section 7) with a regression test. Suite: 165 passed.
- G1 rebuilt with TACO fallback and passed (ALGO_IO 203). First pilot/invariance data
  archived in `data/superseded/`; pilot repeated on the final code.
- Budget reconciled for terminal jobs with a conservative bound (whole billed minutes
  from started_at to finished_at, plus one minute, at the flavor rate; canceled S2 jobs
  without finish times keep full reservations): ledger $13.65 -> $9.96. Not an invoice.
| 6af1ce90ab34849891a808c7 | generate-73c0a0735af25861db6987ac | l40sx1 | ≤20.0 | ≤0.6000 | Launched |

2026-11-09T13:48:46.253718+00:00: generation 6af1ce90ab34849891a808c7 terminal=COMPLETED; reservation retained pending billing. No candidate ran in the GPU job.

Reservation generate-c1d0b07a6b71b44f32f2360a: ≤$0.6000, before submitting the paid job.
| 6af1cf7aab34849891a80b22 | generate-c1d0b07a6b71b44f32f2360a | l40sx1 | ≤20.0 | ≤0.6000 | Launched |

2026-11-09T14:30:38.289187+00:00: generation 6af1cf7aab34849891a80b22 terminal=COMPLETED; reservation retained pending billing. No candidate ran in the GPU job.

Reservation generate-d9e0cdc8ca0c0f0a05a9da66: ≤$0.6000, before submitting the paid job.
| 6af1d946ab34849891a823e2 | generate-d9e0cdc8ca0c0f0a05a9da66 | l40sx1 | ≤20.0 | ≤0.6000 | Launched |

2026-11-09T15:00:17.348169+00:00: generation 6af1d946ab34849891a823e2 terminal=COMPLETED; reservation retained pending billing. No candidate ran in the GPU job.

Reservation generate-188dbe627262f347e532e10e: ≤$0.6000, before submitting the paid job.
| 6af1e3a29737d9a9e08cb9ff | generate-188dbe627262f347e532e10e | l40sx1 | ≤20.0 | ≤0.6000 | Launched |

2026-11-09T16:15:13.270353+00:00: generation 6af1e3a29737d9a9e08cb9ff terminal=COMPLETED; reservation retained pending billing. No candidate ran in the GPU job.

Reservation generate-a69686df0305bebf25744485: ≤$0.6000, before submitting the paid job.
| 6af1f1ceab34849891a852ce | generate-a69686df0305bebf25744485 | l40sx1 | ≤20.0 | ≤0.6000 | Launched |

2026-11-09T16:22:14.518734+00:00: generation 6af1f1ceab34849891a852ce terminal=COMPLETED; reservation retained pending billing. No candidate ran in the GPU job.

Reservation generate-46b19966a0f10ea644f0b7cc: ≤$0.6000, before submitting the paid job.
| 6af1f36b9737d9a9e08cd31b | generate-46b19966a0f10ea644f0b7cc | l40sx1 | ≤20.0 | ≤0.6000 | Launched |

2026-11-09T16:28:26.145291+00:00: generation 6af1f36b9737d9a9e08cd31b terminal=COMPLETED; reservation retained pending billing. No candidate ran in the GPU job.

Reservation generate-91be1cb9d0dd081d33b82c0f: ≤$0.6000, before submitting the paid job.
| 6af1f4e39737d9a9e08cd52d | generate-91be1cb9d0dd081d33b82c0f | l40sx1 | ≤20.0 | ≤0.6000 | Launched |

2026-11-09T16:33:19.694605+00:00: generation 6af1f4e39737d9a9e08cd52d terminal=COMPLETED; reservation retained pending billing. No candidate ran in the GPU job.

Reservation generate-9d6d2b56fd93a5e1ff274f98: ≤$0.6000, before submitting the paid job.
| 6af1f61fab34849891a85868 | generate-9d6d2b56fd93a5e1ff274f98 | l40sx1 | ≤20.0 | ≤0.6000 | Launched |

2026-11-09T16:44:17.153385+00:00: generation 6af1f61fab34849891a85868 terminal=COMPLETED; reservation retained pending billing. No candidate ran in the GPU job.

Reservation generate-0b65646fe0576de27513d4d0: ≤$0.6000, before submitting the paid job.
| 6af1f8a7ab34849891a85cbb | generate-0b65646fe0576de27513d4d0 | l40sx1 | ≤20.0 | ≤0.6000 | Launched |

2026-11-09T16:50:10.358409+00:00: generation 6af1f8a7ab34849891a85cbb terminal=COMPLETED; reservation retained pending billing. No candidate ran in the GPU job.

Reservation generate-5bf6e605267fecd53c391a8c: ≤$2.7000, before submitting the paid job.
| 6af200059737d9a9e08ce7b1 | generate-5bf6e605267fecd53c391a8c | l40sx1 | ≤90.0 | ≤2.7000 | Launched |

2026-11-09T17:35:02.744858+00:00: generation 6af200059737d9a9e08ce7b1 terminal=COMPLETED; reservation retained pending billing. No candidate ran in the GPU job.

## 2026-11-10 — Full measurement controller stopped by sandbox false positive
- The restarted full-measurement controller aborted on the first FTP-mock candidate: `sandbox/runner.py` treated candidate output containing "Failed to connect" as sandbox-launcher failure (`SandboxUnavailable`). No attempt had been recorded (checkpoint: 0 attempts for all 11,720 states; verified).
- Fix: launcher-failure detection now only matches line-anchored launcher messages and only when the worker's timing marker is absent; regression test `test_candidate_failure_text_is_not_sandbox_unavailable`.
- Because `sandbox/runner.py` is part of the measurement fingerprint, the zero-attempt checkpoint was archived to `data/superseded/measure-base-full-r1-initial-checkpoint/` and re-initialised. Round-1 GPU outputs (job 6af200059737d9a9e08ce7b1) are hash-verified and reused; no new GPU spend.

Reservation generate-6bc988309a5988033840d864: ≤$2.7000, before submitting the paid job.
| 6af21795ab34849891a89978 | generate-6bc988309a5988033840d864 | l40sx1 | ≤90.0 | ≤2.7000 | Launched |
| 921388bded9e6a01 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003528 | Reserved before request |

2026-11-09T20:05:38.218426+00:00: generation 6af21795ab34849891a89978 terminal=COMPLETED; reservation retained pending billing. No candidate ran in the GPU job.

Reservation generate-5d61e3c70f9ad0063c9ea193: ≤$2.7000, before submitting the paid job.
| 6af22a15ab34849891a8d161 | generate-5d61e3c70f9ad0063c9ea193 | l40sx1 | ≤90.0 | ≤2.7000 | Launched |

2026-11-10: Ledger reconciled. S2 v1 was CANCELED with no start (cost 0); S2 v2/v3 were bounded by verified-terminal times ($0.45, $0.84). Full-measurement rounds 1/2 reconciled to start..finish bounds ($0.51, $0.78). Conservative total now includes only the round-3 reservation as unreconciled.

## 2026-11-10 — Base full measurement progress and new automation
- Round 1 and 2 GPU jobs (6af200059737d9a9e08ce7b1, 6af21795ab34849891a89978) COMPLETED; local sandbox tests of round 1 took ~10 minutes with 8 workers. HF hardware queues were 45-50 minutes per round (the job itself ran 10-20 minutes), so every extra GPU stage costs about an hour of wall-clock time.
- Queue wait now has a six-hour limit. A queue timeout cancels the job, HF confirms `CANCELED` with no start time, the reservation is settled at $0 (`canceled_unstarted`), and the retry uses a `-retryN` name.
- New generic `GPUJob` (`src/anthropochrony/gpu_jobs.py`): resumable, budget-reserved, hash-checked downloads; used for merged-model rounds (`jobs/ft_round.py`), G6 merged-vs-PEFT parity (`jobs/g6_check.py`) and estimate generation (`jobs/eval_estimates.py`, AuditKIT inside the job). Adapters are merged in the job (peft installed there), never served as vLLM LoRA.
- New CLI: `estimate-run`, `fetch-adapter`, `g6`, `parser-audit`, `merge-estimates`, `teacher`; `measure --auto --phase full --run m1-s0,m2-s0,...` now runs merged fine-tuned rounds (needs G6, adapters, frozen preregistration).
- Teacher probe: one DeepSeek-V4-Pro (deepinfra) call cost $0.0008; the 1,465-task teacher pass is estimated at about $1.2 against the $3 provider cap.

2026-11-09T21:16:31.158236+00:00: generation 6af22a15ab34849891a8d161 terminal=None; reservation retained pending billing. No candidate ran in the GPU job.

Reservation generate-5d61e3c70f9ad0063c9ea193-retry1: ≤$2.7000, before submitting the paid job.
| 6af2390dab34849891a90ba1 | generate-5d61e3c70f9ad0063c9ea193-retry1 | l40sx1 | ≤90.0 | ≤2.7000 | Launched |

Reservation smoke-g6-850261bc805f96b15e17d4d2: ≤$0.9000, before submitting the paid job.

Reservation smoke-ftround-d474f2901e17ca894cb2709d: ≤$0.9000, before submitting the paid job.
| 6af23b4cab34849891a912e9 | smoke-g6-850261bc805f96b15e17d4d2 | l40sx1 | ≤30.0 | ≤0.9000 | Launched |
| 6af23b4dab34849891a912ee | smoke-ftround-d474f2901e17ca894cb2709d | l40sx1 | ≤30.0 | ≤0.9000 | Launched |

2026-11-09T22:10:39.073039+00:00: smoke-ftround 6af23b4dab34849891a912ee terminal=COMPLETED; reservation retained pending billing. No candidate ran in the GPU job.

2026-11-09T22:17:52.348142+00:00: smoke-g6 6af23b4cab34849891a912e9 terminal=COMPLETED; reservation retained pending billing. No candidate ran in the GPU job.
- Merged-serving tooling smoke (random small LoRA with the real Cohere2 shapes; tooling check only, no research claim): in-job `pip install peft==0.21.2`, bf16 merge into the bucket base (84 s), vLLM serving of the merged weights, and HF+PEFT-vs-merged-vLLM greedy parity on 50 validation prompts all worked. 0 numeric mismatches, 32/50 identical texts; `results/smoke-merged.json`. Whole `ft_round` process 157 s for one adapter. The two smoke jobs got hardware faster than the 5400 s full round (jobs submitted later with 1800 s timeouts were scheduled first), so later GPU stages use tight timeouts.
- Ledger note: a cancel attempt on queued round-3 job `…90ba1` found it had just started (RUNNING); the controller was restarted and re-attached to the same job by its stored provenance, with no second payment.

2026-11-09T22:48:48.101943+00:00: generation 6af2390dab34849891a90ba1 terminal=COMPLETED; reservation retained pending billing. No candidate ran in the GPU job.

Reservation est-base-6aa80c5cd6c470e02d11a091: ≤$1.8000, before submitting the paid job.
| 6af256a29737d9a9e08dfcb6 | est-base-6aa80c5cd6c470e02d11a091 | l40sx1 | ≤60.0 | ≤1.8000 | Launched |
| bb4678e75431dfc7 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002766 | Reserved before request |
| 97ec4fbad437244e | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002952 | Reserved before request |
| 0db723c346e27518 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003111 | Reserved before request |
| 6f3a289220723a89 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003219 | Reserved before request |
| 6474bd57847d2e3c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003750 | Reserved before request |
| bb13adec56eb4cc0 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004342 | Reserved before request |
| e23dc07789f8de04 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002631 | Reserved before request |
| a258f9b9bd3fd678 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003723 | Reserved before request |
| 1fa689ce43b064be | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003316 | Reserved before request |
| 6f8d6efba46b6ab4 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003094 | Reserved before request |
| 21cbdbcecf006b00 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003058 | Reserved before request |
| b5a39f0b21533c86 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002991 | Reserved before request |
| 486fd9ddda45cab8 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004196 | Reserved before request |
| a068ac1adbbffa82 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002726 | Reserved before request |
| 549fb06ee3c49a13 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004358 | Reserved before request |
| 44fcc388e30146e5 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003338 | Reserved before request |
| 0a86513f753f2f54 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003275 | Reserved before request |
| 199f1c0183c6a102 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003241 | Reserved before request |
| 16a2bfed3b02b562 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002552 | Reserved before request |
| b0edf08db6747893 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003061 | Reserved before request |
| e837da0b2aa74e49 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002990 | Reserved before request |
| 6e320643fd14eb9c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002627 | Reserved before request |
| c4ac54e58e9d30be | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002670 | Reserved before request |
| f8c834295945fd8a | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002642 | Reserved before request |
| dbc226310f9e0ce3 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002467 | Reserved before request |
| 9c415d4a8a0ed08c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002701 | Reserved before request |
| afbd731cf40f439d | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002450 | Reserved before request |
| 881265ed3861d71e | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002510 | Reserved before request |
| 84d41a19d8e29202 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002396 | Reserved before request |
| aedeee9c40f41550 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002627 | Reserved before request |
| 664ba340a400f9b9 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003038 | Reserved before request |
| ca7b968b5694f0bb | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002894 | Reserved before request |
| 31cc184cee00a4bc | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002642 | Reserved before request |
| 35ce02c8f7f5e88f | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002514 | Reserved before request |
| 134f88a4122c096e | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002915 | Reserved before request |
| 52f7ba2e3c564703 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002829 | Reserved before request |
| b7e583336effc54d | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003780 | Reserved before request |
| db44d586a91561b4 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002850 | Reserved before request |
| 0dd532dce9003546 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003650 | Reserved before request |
| b6beabc288699de4 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003678 | Reserved before request |
| 6cc845a60c03d51c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002517 | Reserved before request |
| 858329c98204e367 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003106 | Reserved before request |
| b290dd20b50da2d8 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002665 | Reserved before request |
| 1a193fe7a4bb549c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002896 | Reserved before request |
| c7575a0ca183ad2f | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003106 | Reserved before request |
| 5bdeb0501e162bdc | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002800 | Reserved before request |
| 93efd19f993d0e98 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002972 | Reserved before request |
| 39d1a2e9561448f9 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002726 | Reserved before request |
| aabd92564bfca2ce | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002505 | Reserved before request |
| 49a5ecb84fa5e741 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002525 | Reserved before request |
| b32d23865feff834 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002911 | Reserved before request |
| 5dc57f24e3c0f609 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003866 | Reserved before request |
| 877d9d11084c2044 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002718 | Reserved before request |
| b67a7b9c294e2b39 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002674 | Reserved before request |
| 520174405c1883a3 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002436 | Reserved before request |
| 45e1e49a879479f3 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003102 | Reserved before request |
| a9bf5b6fda1c49f4 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002625 | Reserved before request |
| ef38124c83329599 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003158 | Reserved before request |
| 57b07d1a4bc2b9cd | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002674 | Reserved before request |
| 90ac00e31ca8473d | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003012 | Reserved before request |
| 1f2b8a0d842b68c8 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003328 | Reserved before request |
| 6612a09182e9f9c0 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003253 | Reserved before request |
| b1d1ef3970aaea5e | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003004 | Reserved before request |
| 64e4a22bf8c09bf9 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002922 | Reserved before request |
| 84ec363f571f4426 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002616 | Reserved before request |
| 6f8868b145bf0fdc | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003337 | Reserved before request |
| e67f299013a07acc | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003414 | Reserved before request |
| 9bf850138fe50cda | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002588 | Reserved before request |
| 6f105f413534571e | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002645 | Reserved before request |
| d01249ff562ababa | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002539 | Reserved before request |
| 4b3bdab97e1257ab | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003435 | Reserved before request |
| 66a1a58507e394d2 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002734 | Reserved before request |
| f63a5f30fdbdc1f1 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003303 | Reserved before request |
| 390604d49aa84abb | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003188 | Reserved before request |
| a2d67cfaebff529c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002747 | Reserved before request |
| 6062e6502dbeaf87 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003371 | Reserved before request |
| e6cdd98df3123b11 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002662 | Reserved before request |
| da8b677904ccbff9 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002655 | Reserved before request |
| 74ae91a7e2c8a97c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002735 | Reserved before request |
| b0c92474e2abad59 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002510 | Reserved before request |
| 9d6593fd20c7ccc5 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002367 | Reserved before request |
| 1085dbc560e499cb | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002790 | Reserved before request |
| 4ee9980b185ed67c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003061 | Reserved before request |
| f1f943af7b9657ef | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003012 | Reserved before request |
| d5590622aa7729d9 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002617 | Reserved before request |
| 9609a3092a17e32c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002710 | Reserved before request |
| a8a85d2e255e02b0 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002714 | Reserved before request |
| a8ee947301c1dcaf | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002844 | Reserved before request |
| b020257fdae6e787 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002627 | Reserved before request |
| fe5f4228df2742c1 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002738 | Reserved before request |
| dc1c182d6827ed2d | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002829 | Reserved before request |
| 1f2b766c3bcd6ae9 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003061 | Reserved before request |
| 18c95b034139bab3 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002570 | Reserved before request |
| 46fa1959958141cc | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003165 | Reserved before request |
| e714c115e06505d3 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002501 | Reserved before request |
| 8e4b8f110715c08e | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002752 | Reserved before request |
| 234e5c7ed687b1f8 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002865 | Reserved before request |
| bed5246ec243a5ed | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002651 | Reserved before request |
| b92389d63c68c997 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002738 | Reserved before request |
| 51dd4081e16fcd39 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002655 | Reserved before request |
| 0137f8f36158cba7 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003026 | Reserved before request |
| 5b4011bc94cfcac3 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003177 | Reserved before request |
| 8d2d27f3f12d9b9b | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002733 | Reserved before request |
| f090e7aa74909da4 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002716 | Reserved before request |
| 8c5402d8369133ef | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002497 | Reserved before request |
| 52d5edfa33c90b35 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002885 | Reserved before request |
| f4be0f052b0745b7 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002813 | Reserved before request |
| 7dcf10afce76fe29 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003054 | Reserved before request |
| 72e561089817df41 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002998 | Reserved before request |
| 7a21fa751a7182de | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002898 | Reserved before request |
| 17e457849fdfa4ea | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003572 | Reserved before request |
| f476407113794f65 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002626 | Reserved before request |
| 3687f6ed295193bd | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003054 | Reserved before request |
| 821bf563de6093c0 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002974 | Reserved before request |
| cb2f908bc19b28a1 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002617 | Reserved before request |
| be4761ca2e1158f0 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002974 | Reserved before request |
| efb47489115a3839 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003004 | Reserved before request |
| 97d5b5d7d7ee9041 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002750 | Reserved before request |
| f5d847e3baf2378f | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002982 | Reserved before request |
| 9111c659250fe97b | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002492 | Reserved before request |
| 36882b7f8c99eabb | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002986 | Reserved before request |
| 230bc7cf4bd32945 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003452 | Reserved before request |
| 8f10edd439872e25 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003207 | Reserved before request |
| 46ffbe2715087c14 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002753 | Reserved before request |
| 75648a27c7cecdb4 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002954 | Reserved before request |
| 0759b30fc16a2b2e | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002964 | Reserved before request |
| 2626a551079bd732 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003090 | Reserved before request |
| acece806cc7ad57f | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002658 | Reserved before request |
| 7d951d6bfaf2d167 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002733 | Reserved before request |
| d88e2e6e50b65835 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002900 | Reserved before request |
| 6003ea812ea6c46e | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003505 | Reserved before request |
| ab859537df5972ee | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002811 | Reserved before request |
| 5c045833d1d3da03 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003240 | Reserved before request |
| 68320804ea012479 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003063 | Reserved before request |
| a492f47d1a8f93aa | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002954 | Reserved before request |
| 7779c19708948d14 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002721 | Reserved before request |
| 91c36c5b08395d1c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002828 | Reserved before request |
| 69502dabac5fb36f | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003036 | Reserved before request |
| 5ba64dfe5dfec100 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002671 | Reserved before request |
| 5db5aa4b2b0c009c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002423 | Reserved before request |
| c14538287515bcf7 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002665 | Reserved before request |
| a1b46055f4f4eaa2 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002722 | Reserved before request |
| ab95af481d0faf20 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002761 | Reserved before request |
| 202a6a1af329e9b6 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002586 | Reserved before request |
| f1e57585a09f8867 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003214 | Reserved before request |
| 827b7cad5b9816d2 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002564 | Reserved before request |
| 6cb440618cf59c16 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002930 | Reserved before request |
| c308c51bcac2cad3 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003074 | Reserved before request |
| 4b414b3109fde0dd | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002499 | Reserved before request |
| 2aba2be8829539b6 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002526 | Reserved before request |
| 423ae0042ceed8b8 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003172 | Reserved before request |
| 1bd0386148bd768e | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003012 | Reserved before request |
| 0c84b2bbb0134e96 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002660 | Reserved before request |
| 56ad9e5d2466e0f5 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002710 | Reserved before request |
| a305f669f276b1d5 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002571 | Reserved before request |
| 97f884fee7fd21a4 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002744 | Reserved before request |
| 1082062e0b110fab | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002712 | Reserved before request |
| 3752bf5ac398e9df | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002852 | Reserved before request |
| 34f3a259b96a4b19 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002919 | Reserved before request |
| a166b43b89f05908 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002574 | Reserved before request |
| 67b80184003d6cf1 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002592 | Reserved before request |
| 5ec8fdce719cd881 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002669 | Reserved before request |
| 756324d2be4dbfaa | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002538 | Reserved before request |
| e6e74453abf9b256 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002465 | Reserved before request |
| 385e2376b8fba390 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002833 | Reserved before request |
| 868b4713ca90b0b7 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002678 | Reserved before request |
| f59f7a35c965a656 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002817 | Reserved before request |
| f98a2a5cfca3190a | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002742 | Reserved before request |
| 8dfb060b2d3e1cbc | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002647 | Reserved before request |
| 329112a6e5e27843 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002410 | Reserved before request |
| af0e36ee396892a7 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002440 | Reserved before request |
| fefab664ec9ba53a | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002582 | Reserved before request |
| 698428dab5cf1b6c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002682 | Reserved before request |
| 904172df4bfc71d0 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002460 | Reserved before request |
| b8cbce9951105178 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002610 | Reserved before request |
| 22a56d2c4aa63e72 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002552 | Reserved before request |
| 49c7c35ad0185289 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002402 | Reserved before request |
| 0ee9708e9c1334b2 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002604 | Reserved before request |
| f018af04e104c16c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002696 | Reserved before request |
| 8dd96776725d9863 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002597 | Reserved before request |
| 5bf8ea45d98a96ea | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002512 | Reserved before request |
| d4a56b76ad6becb5 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002708 | Reserved before request |
| 0cdb9447a624d15a | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003506 | Reserved before request |
| 61da6f561ba3f313 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002425 | Reserved before request |
| 9a70233edd7f713c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003295 | Reserved before request |
| c8698040f6c00252 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003012 | Reserved before request |
| 8bb9b207085e3cae | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004391 | Reserved before request |
| 1e22018cafb0b7f2 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002677 | Reserved before request |
| 60d2c2e7a67519e0 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002655 | Reserved before request |
| 8232ae11f8cf6210 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002514 | Reserved before request |
| 0a80e39dce6556eb | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002708 | Reserved before request |
| 7b3bdb9138373583 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002586 | Reserved before request |
| 02b4eea79619cf4a | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003354 | Reserved before request |
| 62bc1921459b3e5b | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002518 | Reserved before request |
| 2def7ac3ef2f43de | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003146 | Reserved before request |
| 9341a030a465ce36 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003042 | Reserved before request |
| dfdf859b5357e754 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002434 | Reserved before request |
| 5886724dda9d9958 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003505 | Reserved before request |
| 72ddea04fc1ec3b2 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003554 | Reserved before request |
| 453676a5f68603d4 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002512 | Reserved before request |
| e3f8fbfd696bc8c8 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002727 | Reserved before request |
| 1d394ede9929e118 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002898 | Reserved before request |
| c4098ea9d391455d | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003103 | Reserved before request |
| 1f1997cdaa3952cb | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003041 | Reserved before request |
| 4594f69601d54a4d | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002818 | Reserved before request |
| 137c0de6ef9eda61 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002816 | Reserved before request |
| 5571ddb30fa8900e | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003050 | Reserved before request |
| a2eb79777aa3573f | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002678 | Reserved before request |
| a5abc617a1d44296 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003010 | Reserved before request |
| 029edfa5124cc6a2 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002899 | Reserved before request |
| e57976ed7827d9a1 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003297 | Reserved before request |
| 6aef182f159994c4 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003359 | Reserved before request |
| 1a86a4e04706c822 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003604 | Reserved before request |
| e3f8fb51f4392d07 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002716 | Reserved before request |
| b1b9f071c28df29f | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002545 | Reserved before request |
| 3d1e72b20b535546 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002766 | Reserved before request |
| f3bb99e2ef6ce0d0 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003267 | Reserved before request |
| fc5ff0257eab5b94 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002501 | Reserved before request |
| 01a7ce37f876295f | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002779 | Reserved before request |
| 93727290a3b40e07 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002661 | Reserved before request |
| 0dd4f119d63b6d3e | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002960 | Reserved before request |
| 9561ef2449cee866 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002747 | Reserved before request |
| 0035d2b49fa85585 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003141 | Reserved before request |
| 298f02a6c23fdd62 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002757 | Reserved before request |
| ea39895ba7eed5ad | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002916 | Reserved before request |
| e367c46448e9dd17 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002748 | Reserved before request |
| 4209d3a9e1a1c2d6 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002794 | Reserved before request |
| 03a7135793c21d62 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002616 | Reserved before request |
| 416703d223b7f20d | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003015 | Reserved before request |
| 43c3ef811c90d337 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002591 | Reserved before request |
| d865f618d0d3c210 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002928 | Reserved before request |
| a6c15110e6758812 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003020 | Reserved before request |
| 9a5dcf1eb1673d82 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002688 | Reserved before request |
| 2da11b51cb2a620c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003319 | Reserved before request |
| e4b4d6289ad02075 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002759 | Reserved before request |
| 1f8e8391cc3cea65 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002895 | Reserved before request |
| dbbde2ab1e9e4a00 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003333 | Reserved before request |
| 1eab65973c2e526c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003142 | Reserved before request |
| 35f97aa49ce0bb53 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002973 | Reserved before request |
| 2c1f9853076d5da3 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002513 | Reserved before request |
| 6467c89ddcc87e72 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002718 | Reserved before request |
| 333bb97f4ac692d0 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002815 | Reserved before request |
| 8243ccf0c5eefa2f | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002678 | Reserved before request |
| 7acacdd4202ff5b3 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002573 | Reserved before request |
| 516e49e86a7d2f9d | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002471 | Reserved before request |
| 71b9d32b9de9c21d | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002632 | Reserved before request |
| d2a203d16986d54c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002500 | Reserved before request |
| 378ca0e0e4ff94e8 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003112 | Reserved before request |
| 2bd9d2ff5806b6cd | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002580 | Reserved before request |
| ed58a22251892cef | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002939 | Reserved before request |
| 49d227edbf4b4913 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002575 | Reserved before request |
| a750d34c67a9cc76 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002397 | Reserved before request |
| 33d015be6d245848 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002935 | Reserved before request |
| 7e983ac254137f78 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002709 | Reserved before request |
| d27755ec815bb3cb | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002828 | Reserved before request |
| 68b62d53a0fd94d9 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002367 | Reserved before request |
| 43f8478a3eefd9f1 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003420 | Reserved before request |
| b47cc6f1369643dd | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002973 | Reserved before request |
| 5884a64ff5ce0fc0 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002803 | Reserved before request |
| 51376322178477e9 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002894 | Reserved before request |
| 32c602abcdaa663b | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002549 | Reserved before request |
| ce23897e16068dea | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002857 | Reserved before request |
| 05c3c37d358c96e8 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002557 | Reserved before request |
| b1e19e5752411e99 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003156 | Reserved before request |
| dc81fe422277c5c1 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003167 | Reserved before request |
| 2d9c1d72f3a43c17 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002703 | Reserved before request |
| b9af4b817bc43fdf | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002830 | Reserved before request |
| 1e12d7962c645549 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002564 | Reserved before request |
| 80e78de1707479e7 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003828 | Reserved before request |
| 8e42796022577a23 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002343 | Reserved before request |
| 004326c7f9d742ee | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002690 | Reserved before request |
| 04b04d3db2306fb4 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002747 | Reserved before request |
| 519facea967cae40 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002530 | Reserved before request |
| 218069492653598c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002665 | Reserved before request |
| 12bf4fac5d192372 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002661 | Reserved before request |
| 02e88c43af832efa | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002483 | Reserved before request |
| ec00c0b4e98fca1f | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002334 | Reserved before request |
| ed81b72ff52fbbc9 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002570 | Reserved before request |
| 6920420b3f73e7ac | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003107 | Reserved before request |
| 70e7e72e399980ce | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002852 | Reserved before request |
| 0d9ec765ce2ac3cd | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002887 | Reserved before request |
| ef5c6a7e2168f6fd | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002822 | Reserved before request |
| 94d68844cde4dbb0 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003068 | Reserved before request |
| 94f4345a222087c2 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002626 | Reserved before request |
| 711311e7e533a23c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003093 | Reserved before request |
| dd81f842b4c1257f | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002634 | Reserved before request |
| 40f4ec4fc7788431 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002352 | Reserved before request |
| dbeb4d867e836a7e | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002558 | Reserved before request |
| 57e3a3a2e42e6088 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002665 | Reserved before request |
| 31336cf5ff3a2b93 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003116 | Reserved before request |
| 51e7f078f99f0688 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002538 | Reserved before request |
| fbb7079eb0d46d35 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002361 | Reserved before request |
| e27aae06df3ed456 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002753 | Reserved before request |
| fad010a78109ab3a | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002444 | Reserved before request |
| 2d824a0f822edd3a | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002558 | Reserved before request |
| 019629b49846313f | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002491 | Reserved before request |
| 50dd460e1c24d374 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002515 | Reserved before request |
| de676266c0886da7 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002410 | Reserved before request |
| ce9eae57a144573d | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002233 | Reserved before request |
| be76cf3df501e933 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002284 | Reserved before request |
| d871a120a3c6941b | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002431 | Reserved before request |
| cecc6096e8786354 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002293 | Reserved before request |
| 4e1781c9769865cc | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002337 | Reserved before request |
| eafab515f94982f4 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002643 | Reserved before request |
| 0303ed863d2a0b92 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002462 | Reserved before request |
| a32a3955470cc4e4 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002614 | Reserved before request |
| 9948e6942ef5e176 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002254 | Reserved before request |
| 50c0ab2188348dfa | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002357 | Reserved before request |
| 7472150ed9967a83 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002798 | Reserved before request |
| 0b3de6187a05be26 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002399 | Reserved before request |
| dfd9766199adadf2 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002690 | Reserved before request |
| 0e6e6b8270989325 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002450 | Reserved before request |
| fb3b56c69adab30d | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002240 | Reserved before request |
| cccd9a64cc9c64a6 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002649 | Reserved before request |
| 22a19afbc7f36911 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002279 | Reserved before request |
| 95e308195a80ebae | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002378 | Reserved before request |
| b19942bc055ab14a | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002352 | Reserved before request |
| 4f95bce5d72db26c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002434 | Reserved before request |
| a76e560de980c4a7 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002493 | Reserved before request |
| 450e5526ac01af56 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002557 | Reserved before request |
| c4779e35ed843e0e | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002353 | Reserved before request |
| 6a15d4f3291f224f | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002210 | Reserved before request |
| c07a5f18059219b8 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002621 | Reserved before request |
| 13b080db48b9830d | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002230 | Reserved before request |
| 4ce8600daef9defc | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002513 | Reserved before request |
| 6426d9ea175abd2b | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002365 | Reserved before request |
| d4ae6acc50daf2c3 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002365 | Reserved before request |
| e90971f15a68abd3 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002432 | Reserved before request |
| 58855d02576584b7 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002613 | Reserved before request |
| c64a8c53dfe7528a | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002445 | Reserved before request |
| d585331d39e8e227 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003578 | Reserved before request |
| 9af4f5d908a68226 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002699 | Reserved before request |
| 0573f43124da947d | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002489 | Reserved before request |
| 87a6bf6c733f04bf | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002561 | Reserved before request |
| cb33166b85bb96ec | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003011 | Reserved before request |
| 7e1c5aa5cdfb0fec | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002674 | Reserved before request |
| cf7fb7701b9c5c18 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002870 | Reserved before request |
| 674ef19a30c517bf | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002614 | Reserved before request |
| 3f047214d7928a4f | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003266 | Reserved before request |
| 5deca50cc6c2ab8b | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002789 | Reserved before request |
| e04a13a8457bbfdc | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002544 | Reserved before request |
| 4336ea2096d125b1 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002564 | Reserved before request |
| 45005d0ae7153608 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003406 | Reserved before request |
| 11436a642b830840 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002467 | Reserved before request |
| 97f7db79520b9893 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002274 | Reserved before request |
| 2a67793168368623 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003041 | Reserved before request |
| c2d8ac838ab32c8b | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002626 | Reserved before request |
| 861647063d04d60a | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002435 | Reserved before request |
| a89efd332be96f1a | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002530 | Reserved before request |
| 0435ee57a22439f6 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002544 | Reserved before request |
| deabefa6434a109a | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002573 | Reserved before request |
| 9d75154d37a3f418 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002851 | Reserved before request |
| ee520cf4e5846e62 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002575 | Reserved before request |
| e434e3e768d245df | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002510 | Reserved before request |
| 87f25dd2d0a3ba4c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002188 | Reserved before request |
| 18f32d2e90fc42a4 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002369 | Reserved before request |
| aa20c76b1dec3889 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002233 | Reserved before request |
| 39e91239fe140717 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002192 | Reserved before request |
| 295c9e9b6aa8b6c0 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002210 | Reserved before request |
| 9d7136815d810c87 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002259 | Reserved before request |
| 65841d5569cde1b7 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002224 | Reserved before request |
| 91ad211b918a95f3 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002189 | Reserved before request |
| b96220fba35b220f | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002235 | Reserved before request |
| 8fbc7cf81da18690 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002162 | Reserved before request |
| 6ac33948f5325c50 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002296 | Reserved before request |
| 92d7eadaf47951cc | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002265 | Reserved before request |
| 743f247ba3208524 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002283 | Reserved before request |
| 8a10367b2468a1bd | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002245 | Reserved before request |
| 9635a570e47871dd | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002311 | Reserved before request |
| ddc1c3877545c297 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002204 | Reserved before request |
| 0d7eeb8e9dfcd946 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002210 | Reserved before request |
| 5489fa64dd8461f0 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002218 | Reserved before request |
| 96e3d763626506e1 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002180 | Reserved before request |
| 71a2c133991e9f06 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002430 | Reserved before request |
| 17f8f48e22a6086a | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002162 | Reserved before request |
| f07d2411e53f3f0f | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002306 | Reserved before request |
| 4c121a9c14212439 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002178 | Reserved before request |
| bc1e19b9a7962b7c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002185 | Reserved before request |
| 70c493fdc7558aa3 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002219 | Reserved before request |
| 258991fe3197eb9f | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002205 | Reserved before request |
| 711de738fa5d8c24 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002370 | Reserved before request |
| 7c02d49ef6f73558 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002291 | Reserved before request |
| d5554c3d93e121e4 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002159 | Reserved before request |
| 0e8b271db2ef0562 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002453 | Reserved before request |
| a98aa8e6cd87dc04 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002165 | Reserved before request |
| c9647eadc2de6c4c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002280 | Reserved before request |
| b9dba485554324fb | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002314 | Reserved before request |
| 5d6b2d4dfe5b8d58 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002253 | Reserved before request |
| 27b5da86fdb92287 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002473 | Reserved before request |
| 8a4389d3e6f4d794 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002209 | Reserved before request |
| 7c2d718ef120f56c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002184 | Reserved before request |
| f4832ed3fa043343 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002310 | Reserved before request |
| eeb61b6425ddb9ee | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002258 | Reserved before request |
| 91cae2de4bfbe06f | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002210 | Reserved before request |
| 3c80deaa2d9b131d | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002219 | Reserved before request |
| eec5d1510d1ea35b | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002180 | Reserved before request |
| 27442b046b92e89e | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002270 | Reserved before request |
| db6c00c6834c0b76 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002317 | Reserved before request |
| 6a0e493a0e22022d | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002253 | Reserved before request |
| 3cc774299adfffb2 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002211 | Reserved before request |
| 3177e10ac7a3c460 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002214 | Reserved before request |
| 8c16e3a3f0d60669 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002263 | Reserved before request |
| 46030afea369f491 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002174 | Reserved before request |
| a349542e563c94bc | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002202 | Reserved before request |
| 6a2b8bbff1d34008 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002300 | Reserved before request |
| f66b8aeae3485a62 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002246 | Reserved before request |
| 35cb506804fc0473 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002183 | Reserved before request |
| c8bcbeacf8bca676 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002188 | Reserved before request |
| 95e468995e514ed4 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002209 | Reserved before request |
| 1eb307c9d29d725c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002243 | Reserved before request |
| 51d27794b47fab98 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002175 | Reserved before request |
| 8630de60e5849399 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002180 | Reserved before request |
| 4a89b6e432ca536e | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002176 | Reserved before request |
| 5de30d9e63578460 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002191 | Reserved before request |
| fbad47b6533d1223 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002174 | Reserved before request |
| 3b219ca9d1d54aa8 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002250 | Reserved before request |
| 78796f5e3cea81d5 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002311 | Reserved before request |
| 77052889d9453aaf | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002228 | Reserved before request |
| 62bfef1c02c3b0cc | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002183 | Reserved before request |
| 283138c75ddf6e24 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002198 | Reserved before request |
| 28d6cf82d85ba693 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002237 | Reserved before request |
| 42e7da7a2bf9ba55 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002222 | Reserved before request |
| 06727496cd83b20f | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002643 | Reserved before request |
| 3e304499c8df3252 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002231 | Reserved before request |
| b903ff8f80ec7e53 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002250 | Reserved before request |
| f9f1f281e8a8da93 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002235 | Reserved before request |
| 5f8501ed4964f0ce | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002201 | Reserved before request |
| cf68f1ca1a0bcd69 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002217 | Reserved before request |
| d1a1fa34d699bf62 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002219 | Reserved before request |
| c2d1d888fb6e4160 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002285 | Reserved before request |
| 67909f1a0adc32f2 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002249 | Reserved before request |
| a64d7efbb4ed2967 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002198 | Reserved before request |
| bb8ff09692107547 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002184 | Reserved before request |
| 3f667858f2b0915e | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002230 | Reserved before request |
| 16bd9961152a0fc4 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002215 | Reserved before request |
| f3ce9cb5ec697159 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002204 | Reserved before request |
| 702b99b077318395 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002396 | Reserved before request |
| 8ce8ccb3f352bd54 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002219 | Reserved before request |
| 522d81e05837fb98 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002230 | Reserved before request |
| cd1e051b42248117 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002163 | Reserved before request |
| ed73b7e2f8cfc9e5 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002253 | Reserved before request |
| b68cd8c8b634219a | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002224 | Reserved before request |
| 55e1591466738442 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002217 | Reserved before request |
| 3ec670dd8a39f657 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002219 | Reserved before request |
| ae6f819804d84a88 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002185 | Reserved before request |
| 3fa737cd9aaa6a9d | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002215 | Reserved before request |
| 8e9cb0390b54daa3 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002240 | Reserved before request |
| a0850d16b32a5c15 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002265 | Reserved before request |
| ad6a5adf48985211 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002263 | Reserved before request |
| 60b53db632e500d9 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002262 | Reserved before request |
| cefd3476d12bbffc | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002258 | Reserved before request |
| 96d862197c0620d3 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002240 | Reserved before request |
| 8b83786f8cc889db | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002222 | Reserved before request |
| 3abdc8c5af1f75ba | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002276 | Reserved before request |
| 40fa0addc2bf9416 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002334 | Reserved before request |
| 21726953be773133 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002383 | Reserved before request |
| ede7ed99237331ad | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002248 | Reserved before request |
| 03344e3ab591ec72 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002197 | Reserved before request |
| 60dfb2313402694d | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002191 | Reserved before request |
| 3b55b4e08c6fd139 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002297 | Reserved before request |
| e7adf6a3d251c073 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002245 | Reserved before request |
| 4c082e2522ac168d | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002314 | Reserved before request |
| 4485c5688d27a2ee | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002674 | Reserved before request |
| 897935d7a327656a | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002593 | Reserved before request |
| 6067cb018ddc90f7 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002304 | Reserved before request |
| 5b66b15621de40d1 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002354 | Reserved before request |
| c71ade09dfb2a91c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002258 | Reserved before request |
| 09610f0f93b75972 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002244 | Reserved before request |
| f61fc9bd6a8ef28b | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002311 | Reserved before request |
| 685a840f86bce804 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002269 | Reserved before request |
| e1583b8751b151a3 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002322 | Reserved before request |
| 8137cc70e9802d74 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002309 | Reserved before request |
| d7f13835bc80fd6a | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002369 | Reserved before request |
| b0a19da57d87a22c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002306 | Reserved before request |
| 7bc96d9604071059 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002189 | Reserved before request |
| 892407fced6e7e90 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002209 | Reserved before request |
| 9d6257d2788bad3c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002440 | Reserved before request |
| 04bf2574d5b7a0d4 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002393 | Reserved before request |
| 4fe0ab2a7d4f17d9 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002205 | Reserved before request |
| c8599445c02f5796 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002335 | Reserved before request |
| 749222f07edd8ac6 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002436 | Reserved before request |
| d4d81d801e83a66f | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002502 | Reserved before request |
| 2b8963cd8d7f40c3 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002267 | Reserved before request |
| 3649e5c53504d22f | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002319 | Reserved before request |
| 63b1ec89659af8e8 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002194 | Reserved before request |
| 7c6c42bb1d38ee08 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002287 | Reserved before request |
| b9c195900d65458b | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002293 | Reserved before request |
| 82d212cdbeac8817 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002185 | Reserved before request |
| f50be8b6e8414bd5 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002259 | Reserved before request |
| 91d7083d362bf1f8 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002239 | Reserved before request |
| 2d0968c7282009ac | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002214 | Reserved before request |
| 8e943bbbfa1a975a | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002287 | Reserved before request |
| 9859143c7f463d38 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002224 | Reserved before request |
| cbeb18a5a63d9115 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002214 | Reserved before request |
| 2b82bdb494f68743 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002159 | Reserved before request |
| b53c33d872f7d76a | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002231 | Reserved before request |
| d9a6624723a53b52 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002168 | Reserved before request |
| e85b2c543d31d1fa | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002211 | Reserved before request |
| 6b870d62840eb225 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002301 | Reserved before request |
| d94df6001255b7e0 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002230 | Reserved before request |
| ae993180ca38a6aa | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002200 | Reserved before request |
| 54fef3be5269aab7 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002228 | Reserved before request |
| 1f472aa18cf1aa52 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002184 | Reserved before request |
| 91d8ffe654e231d9 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002306 | Reserved before request |
| 7dea9fe3b5a67100 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002170 | Reserved before request |
| b23207d9e662273b | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002285 | Reserved before request |
| 2a593a093200608d | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002198 | Reserved before request |
| 626fe1f90fe9ec64 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002170 | Reserved before request |
| 1d105ea2a595ca42 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002353 | Reserved before request |
| 1f26ad77f4813e8d | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002389 | Reserved before request |
| 0d393ec7ca1883d0 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004306 | Reserved before request |
| f641b6d99a450454 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.005126 | Reserved before request |
| 0f2541c47beb9de3 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.005993 | Reserved before request |
| 188d8af57e2ab7c9 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.005832 | Reserved before request |
| f53ab728e1086500 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.005629 | Reserved before request |
| e00e7eb32213643f | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004706 | Reserved before request |
| e370f3c9f7729b46 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004931 | Reserved before request |
| 9a9b7f49605dbb03 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.005402 | Reserved before request |
| 02ded9ed6eed395c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.006209 | Reserved before request |
| 6c8569a276dc7b52 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.006115 | Reserved before request |
| a25d239c1bf9845d | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002942 | Reserved before request |
| 785aef9b4ffb7d59 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003796 | Reserved before request |
| 547469ac2e237048 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004874 | Reserved before request |
| 327a45c8b4b5943b | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.005389 | Reserved before request |
| d64c4ae25cc3a29d | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004475 | Reserved before request |
| 27a46313ed6cef35 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004575 | Reserved before request |
| 495c25b1c062a69a | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004832 | Reserved before request |
| 5f09ea5059441f01 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004970 | Reserved before request |
| 4e5e85f00665086d | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004858 | Reserved before request |
| b38ad44b58112e22 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004515 | Reserved before request |
| 6928975ca84fcc6d | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004975 | Reserved before request |
| 1adc4e152aa3c7ce | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004768 | Reserved before request |
| c33b47314259d401 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.005035 | Reserved before request |
| 3b1005fbd5b3ddf2 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003364 | Reserved before request |
| 0aa370f720653e78 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003489 | Reserved before request |
| 49d5713446237a00 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.005487 | Reserved before request |
| da3bdeb074ccacc7 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004462 | Reserved before request |
| 11dab0ccb11215ed | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003454 | Reserved before request |
| 5ddcf86f50b5679d | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.006850 | Reserved before request |
| 383094516af05002 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.006670 | Reserved before request |
| e7d1b3502bf6ef96 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004625 | Reserved before request |
| b630f27731b3bdb3 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004421 | Reserved before request |
| 3a57635e655b2fd8 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.005343 | Reserved before request |
| d98143be58497001 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.005803 | Reserved before request |
| 243cd167b1582727 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004269 | Reserved before request |
| cc87f81ab27d4f39 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.005633 | Reserved before request |
| bc1a881f9773fd04 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004530 | Reserved before request |
| 12878a6d5b2d6bba | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.007853 | Reserved before request |
| a46e1f224b35610f | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.006058 | Reserved before request |
| 819cbd6088f47ba7 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.006226 | Reserved before request |
| 43956e7fe5596d6d | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004550 | Reserved before request |
| aefdd60724a04923 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004610 | Reserved before request |
| 7b45e02332586215 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003865 | Reserved before request |
| bc865fc6972fa853 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004076 | Reserved before request |
| 68c4bff10c5f3b93 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004160 | Reserved before request |
| 9722d4034deb0185 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.005273 | Reserved before request |
| 305e68c68f40f217 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.005477 | Reserved before request |
| cb0c180aae1c9470 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003823 | Reserved before request |
| d26498ae7f0092e8 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.005116 | Reserved before request |
| 1cd1f5822afd9177 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004619 | Reserved before request |
| 9a9a750ad03efc72 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003656 | Reserved before request |
| 66cff6e84c1f6315 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003426 | Reserved before request |
| 070d68a94f1e8ac0 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004564 | Reserved before request |
| 5278b3986850282f | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004776 | Reserved before request |
| 5e5aa5176ae9936d | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004572 | Reserved before request |
| 85ed880cfc81ca2d | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004484 | Reserved before request |
| e0989e6591c0e758 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004655 | Reserved before request |
| bd2a3509e28372e5 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004345 | Reserved before request |
| 6d5f02473543f882 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004100 | Reserved before request |
| 00c36b527ea8cf0e | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003416 | Reserved before request |
| 1dedc2995b098fe3 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004027 | Reserved before request |
| 9c0928dff9fb2b63 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.007418 | Reserved before request |
| f59e83c0f5af8857 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004732 | Reserved before request |
| 1c4b7dcb144d8ffc | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003530 | Reserved before request |
| f795326bae1adb80 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004712 | Reserved before request |
| bfdfb623d2dace3f | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.005712 | Reserved before request |
| 78cf9250c2f6de83 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003969 | Reserved before request |
| e241ec6bbcbf200a | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.005073 | Reserved before request |
| 424f19f0d3798cd0 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.005253 | Reserved before request |
| ba7b964d55c203ca | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.005772 | Reserved before request |
| 45b78511a7d9b64b | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003771 | Reserved before request |
| 2c799fe33e9bbbc6 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004733 | Reserved before request |
| 906e88e3b6c63a25 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003565 | Reserved before request |
| c1e7d9bab0ea52c5 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004442 | Reserved before request |
| 0f8fef1735644f23 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.005769 | Reserved before request |
| 5d074d04d0cbd994 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004443 | Reserved before request |
| 029740237f0a0253 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.007822 | Reserved before request |
| b951cc0405b738cf | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.005885 | Reserved before request |
| c52ef69d9ec36253 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004796 | Reserved before request |
| af61ee0ff97ddd2f | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004575 | Reserved before request |
| ef5f772293cd4144 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004460 | Reserved before request |
| fed130391e6f8e0a | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.005580 | Reserved before request |
| d4884154338ef307 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003610 | Reserved before request |
| 5e44dcd5d9156a46 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.005221 | Reserved before request |
| 0d8f2361948a9d66 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.005346 | Reserved before request |
| 94b8bc45eb5cdfad | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.005032 | Reserved before request |
| 4bad5ea7b1166f88 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003442 | Reserved before request |
| 0905e52c35bf9a7d | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.005463 | Reserved before request |
| b3cdcce6e13b15cb | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004696 | Reserved before request |
| 2bc317ea8be61d30 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004176 | Reserved before request |
| 024d1a16e001fd73 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.005216 | Reserved before request |
| 86834968217751ce | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004904 | Reserved before request |
| 24fc464302036212 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004033 | Reserved before request |
| 7af705036810851f | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003847 | Reserved before request |
| 274036c854f3fe69 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004647 | Reserved before request |
| b1e8c3614e4cb922 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.005533 | Reserved before request |
| f975b9e0856a3bd3 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004575 | Reserved before request |
| 46016cfd1c5919eb | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004343 | Reserved before request |
| 59e31c6108db00c8 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004827 | Reserved before request |
| 958bf1b43da2fe31 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.006747 | Reserved before request |
| 19ed81cbf4f0133a | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003926 | Reserved before request |
| a133e4029499913a | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004445 | Reserved before request |
| 6dcf49c44051ea95 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003920 | Reserved before request |
| 5c360a84d106d3b2 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003084 | Reserved before request |
| 4f041bf81372d21b | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004633 | Reserved before request |
| 28d3ee1c7ec11056 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003656 | Reserved before request |
| 33f6af507f5c43f4 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003236 | Reserved before request |
| 70f449d15d5718eb | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004017 | Reserved before request |
| e6476655380de254 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003826 | Reserved before request |
| 435499d5e8231037 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003345 | Reserved before request |
| 9fd2aea420708938 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003428 | Reserved before request |
| 0262bb50d333d6ec | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003465 | Reserved before request |
| 7c67b0561f37b362 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003968 | Reserved before request |
| b1ad5cb4fe3cdfbd | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003476 | Reserved before request |
| 5527a6953a86ba94 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003204 | Reserved before request |
| e4626760faa71ceb | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003498 | Reserved before request |
| 396847b30f238fdb | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004113 | Reserved before request |
| b712fa839f3b25e7 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004057 | Reserved before request |
| f4f4bbe64bed5ed6 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002929 | Reserved before request |
| ae6d6086fe95ca35 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003385 | Reserved before request |
| f38226fccb00c7dd | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003561 | Reserved before request |
| 3fb103b635901720 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002969 | Reserved before request |
| a7fe1e466a3ff42c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003591 | Reserved before request |
| bb2aeae0e2213ee0 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003825 | Reserved before request |
| 57f1b021eb83f7eb | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003493 | Reserved before request |
| 5cb2c9e0aadf795a | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003828 | Reserved before request |
| ab149f7603aeaf12 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003342 | Reserved before request |
| d0e7657b44c412d5 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004476 | Reserved before request |
| f545994f37b446f7 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003749 | Reserved before request |
| 54037013574ba672 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003405 | Reserved before request |
| 82ce6e115119fccf | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003543 | Reserved before request |
| 05a09603c35ababd | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003198 | Reserved before request |
| c0d2c4d8cd8b3599 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003392 | Reserved before request |
| 18b9cd1e2e4f48c7 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004246 | Reserved before request |
| 1646ee6419b7cdbf | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004085 | Reserved before request |
| 2aad217bef28bc6e | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002917 | Reserved before request |
| cd9ba526d49b168a | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004524 | Reserved before request |
| 86307852546ebc06 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003229 | Reserved before request |
| 9c514e1711f3d762 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004018 | Reserved before request |
| 66dbf4d90f66b013 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004419 | Reserved before request |
| cd9f46942344e97e | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003637 | Reserved before request |
| ae10416637a780fb | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003474 | Reserved before request |
| 6357f2692341ed47 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003245 | Reserved before request |
| 6c308b15c679cb60 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003393 | Reserved before request |
| 28af6327c943306a | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003787 | Reserved before request |
| a255825bffa7342a | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003331 | Reserved before request |
| d9b0edd0891ae3db | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003938 | Reserved before request |
| 1652a508389aa9f1 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003936 | Reserved before request |
| 6cb5170a715d96e9 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003748 | Reserved before request |
| 0a36ddbde1c6ebc5 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003641 | Reserved before request |
| 3733b116b929a12c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003813 | Reserved before request |
| 43e4e86c6518086c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004096 | Reserved before request |
| 470ca8571f3e9a4d | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004447 | Reserved before request |
| 3a05e509ab02d75f | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003479 | Reserved before request |
| c390f74879aad2b7 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003891 | Reserved before request |
| a772c917edf45d01 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004102 | Reserved before request |
| 2ce24f4b1dcfadce | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003841 | Reserved before request |
| a27b93c9006641cb | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004289 | Reserved before request |
| e31c3a6ad3ca725d | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.006856 | Reserved before request |
| a08101ebc6a3effe | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.005724 | Reserved before request |
| e2d5614607318b90 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003422 | Reserved before request |
| 7a34c69c42941e05 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004087 | Reserved before request |
| dab2083dc1659084 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003891 | Reserved before request |
| d69f4386475cdc5c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003907 | Reserved before request |
| 969e91e3f58180a8 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004684 | Reserved before request |
| 59331963b0ac1107 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003353 | Reserved before request |
| 26283d07bb86090a | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004282 | Reserved before request |
| d6b76c10acf2da3d | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003484 | Reserved before request |
| 5eeb6c26668bc202 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003849 | Reserved before request |
| 3011933fcc3ec088 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003962 | Reserved before request |
| b34cc3ec5d9adc8b | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004944 | Reserved before request |
| e28f6a3ca592050d | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003273 | Reserved before request |
| 1c328550821a753e | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002896 | Reserved before request |
| 8d77cfe1857ef44c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003484 | Reserved before request |
| d09c65fc5e2230cc | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003796 | Reserved before request |
| ff76c5e6471fff8b | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003314 | Reserved before request |
| 7085ed40046cbe26 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003052 | Reserved before request |
| 71d6bded44de115e | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003281 | Reserved before request |
| 8dec359815275526 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003076 | Reserved before request |
| 71f34a1a571e6c57 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003426 | Reserved before request |
| e404f8479d677b0b | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003792 | Reserved before request |
| 89a10e1d98c77b23 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003550 | Reserved before request |
| 3d7804ff5958d0c3 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003862 | Reserved before request |
| 07ac7f5175614692 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003362 | Reserved before request |
| 94d39708266c3480 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004272 | Reserved before request |
| 27bd2c1e53132618 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004039 | Reserved before request |
| b121a03a1a166c4e | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004714 | Reserved before request |
| 28eec2d810bddcfc | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004724 | Reserved before request |
| 9d6e71344b74e395 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003815 | Reserved before request |
| f3a0e4de7b88cc11 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003959 | Reserved before request |
| f305e60c8d1b3531 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003275 | Reserved before request |
| c3fabf640d68f506 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002969 | Reserved before request |
| a58e5e7e00b9d1ab | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003423 | Reserved before request |
| 3488c3ba954f56f7 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003783 | Reserved before request |
| 683fd1dc515b9396 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004893 | Reserved before request |
| 8dbdb9e76aeeddfa | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004333 | Reserved before request |
| d77771d378ce822d | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003338 | Reserved before request |
| 81ad8639b542987e | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003532 | Reserved before request |
| 509934c2f995670e | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003943 | Reserved before request |
| 53d7cc97ddf6b49e | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003700 | Reserved before request |
| 6f150cace7333d76 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004363 | Reserved before request |
| ac6f66c89aba59d5 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003401 | Reserved before request |
| e5af9faac3380f51 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003796 | Reserved before request |
| 3fd51dffa6cddc9c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002772 | Reserved before request |
| 764fedea1e2740a5 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003626 | Reserved before request |
| 4731a8b9601694b0 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002912 | Reserved before request |
| e63dfdb0f58145ee | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003028 | Reserved before request |
| ffe5e040f8f909c1 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004169 | Reserved before request |
| 0609986d4f0df3fa | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004152 | Reserved before request |
| ac4688e895b7c9e3 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003336 | Reserved before request |
| 455e436679abaffe | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003225 | Reserved before request |
| c2a1f59ff1496a2f | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003132 | Reserved before request |
| 7a5e2e1349af497c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003571 | Reserved before request |
| d5025ed8e7954886 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003691 | Reserved before request |
| 2b160544ca188729 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003650 | Reserved before request |
| 55d2966b1b0cc123 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004857 | Reserved before request |
| 099908ae2ba16c91 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004044 | Reserved before request |
| 21a69290e96e431c | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003433 | Reserved before request |
| 41e858db8bf930df | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.002744 | Reserved before request |
| 79b3d34178bb8237 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003600 | Reserved before request |
| 6e300ac93893bb6d | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003647 | Reserved before request |
| efb6b593c3910278 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003054 | Reserved before request |
| 510b2f4f167966cf | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003737 | Reserved before request |
| 5b3062df1a32c4c4 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003688 | Reserved before request |
| 6acbf1a71cd04dca | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003169 | Reserved before request |
| ef0b68969f078a6e | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003676 | Reserved before request |
| 4cc5212579ec1c63 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003328 | Reserved before request |
| 7495e62fda692b20 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003411 | Reserved before request |
| 6b9af300d460f86b | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003961 | Reserved before request |
| 2cc8df7f81d220d2 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004917 | Reserved before request |
| 9db980563bd0e015 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003279 | Reserved before request |
| 0b065d54e8fccc90 | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.004395 | Reserved before request |
| 0cf0ad6503dbc92a | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003419 | Reserved before request |
| 39b9565e614e93ab | provider deepseek-ai/DeepSeek-V4-Pro | deepinfra | — | ≤0.003761 | Reserved before request |
