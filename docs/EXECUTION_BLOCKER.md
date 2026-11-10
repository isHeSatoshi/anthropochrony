# S2 secure execution blocker

Date: 2026-11-09. No research acceptance threshold has been waived.

## Evidence

The token-free CPU probe, HF Job `6af196a50e0338d1bb857220`, installed
Debian's bubblewrap, dropped to uid/gid 65534, and attempted `--unshare-all`
with an empty environment. It returned:

```text
bwrap: No permissions to create new namespace, likely because the kernel
does not allow non-privileged user namespaces.
```

The private machine-readable evidence is `data/smoke/exec-stage-v3.json`.
It records `security_tests_passed: false`. No benchmark or generated candidate
was executed in that job. The script stops before building the full environment.

This is a platform restriction, not an import or executable-permission error.
HF documents that Jobs lack nested namespaces, mount privileges, and cgroup
delegation:

- https://huggingface.co/docs/huggingface_hub/en/concepts/sandbox

HF's managed pooled sandboxes do not satisfy the original requirements as-is.
Their per-process resource limits are not aggregate quotas. A descendant can
call `setsid()` and outlive a process-group timeout. Dedicated sandboxes protect
the VM boundary but do not by themselves add the required per-attempt limits.

## Storage correction

The earlier staging attempts also established that a mounted bucket is not a
POSIX execution-environment store. Remote files default to mode 0644 and lack
POSIX metadata. The environment must be built locally inside a job and archived
with its executable bits and symlinks, then unpacked locally in the consumer.
The corrected script probes namespace support before paying for that build.

- https://github.com/huggingface/hf-mount

## Approved protocol decision

Validate the actual S3 topology: HF L40S generation connected to the existing,
credential-free local bwrap/systemd execution sandbox through an authenticated
transport. Time the real sequential loop, including transport overhead.

The user approved this change on 2026-11-09. The dated pre-training deviation
is recorded in PREREGISTRATION.md. Keep the original 100 stratified validation tasks,
median relative-error threshold of 10%, and log-time R² threshold of 0.95.
Use the already specified per-call latency fallback only if the observations
justify it. Never treat sums of separately replayed calls as observed end-to-end
elapsed times.

The implementation is being validated. G2 remains unpassed until the observed
loops meet its numerical gates. Do not begin full measurement or training yet.

## Current run

Session `6af19c7be6320c66fad83c40` exhausted a 600-second readiness window
while waiting for L40S hardware. Cancellation was verified. HF recorded no
start timestamp or logs; no throughput or timing result was produced.

Retry `6af19f86ab34849891a7c566` reached L40S and completed generation for all
three throughput probes. Our reader then confused the preemption histogram
with the scalar counter and failed before preserving results. Cancellation
was verified. No timing validation or qualified throughput result followed.

Corrected retry `6af1a64dab34849891a7cb8b` was submitted with a one-hour
hardware timeout and a $1.81 reservation, including its authenticated port.
It retains the separate hardware-allocation and server-readiness deadlines.
The exact pinned vLLM counter schema has a regression test; outputs now persist
before statistics are collected. Previous reservations remain in the ledger
pending actual billing reconciliation. No accuracy threshold changed.

## Completed validation and current accuracy blocker

The corrected retry completed two throughput probes and all 100 actual
sequential validation loops. Authenticated inspection verified its terminal
state is `CANCELED`; no GPU is left running for this session.

G2 failed the unchanged accuracy gates:

- Median absolute relative error: 0.360750, above 0.10.
- Log-time R²: 0.432992, below 0.95.
- Eight-worker pooled/inline execution ratio: 1.009001, within 1.10.

A fitted candidate per-call latency of 0.967021 s gives median error 0.023163
and log-time R² 0.997730 on these same observations. It has not been adopted
or propagated to labels and baselines. It does not constitute a G2 pass.
Any adoption must follow the specified dated-deviation and shared-formula
requirements before research measurement proceeds.

Private evidence: `data/calibration/s2-throughput-validation-v3/validation-summary.json`
and `results/gates/G2.json`. Next-round automation is locally tested but
remains blocked by G2. No fine-tuning has run.

## Resolution

G2 is resolved by the plan's own fallback (per-call latency L = 0.967 s), adopted as
a dated deviation with held-out checks. Thresholds were not changed. The accuracy
blocker above is cleared; the next blocker is G3 (pilot and speed-card invariance).
