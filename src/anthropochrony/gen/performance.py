"""GPU telemetry alongside the real generation call, not a synthetic load."""

import csv
import statistics
import subprocess
import threading
import time


def summarize_gpu(samples, start, end):
    active = [row for row in samples if start <= row["monotonic_s"] <= end]
    if not active:
        return {"samples": 0, "mean_gpu_utilization_pct": None}
    return {
        "samples": len(active),
        "mean_gpu_utilization_pct": statistics.mean(r["gpu_utilization_pct"] for r in active),
        "min_gpu_utilization_pct": min(r["gpu_utilization_pct"] for r in active),
        "mean_memory_bandwidth_utilization_pct": statistics.mean(
            r["memory_utilization_pct"] for r in active
        ),
        "peak_memory_used_mb": max(r["memory_used_mb"] for r in active),
        "memory_total_mb": active[0]["memory_total_mb"],
        "mean_power_w": statistics.mean(r["power_w"] for r in active),
    }


class GPUMonitor:
    def __enter__(self):
        self.samples = []
        self.errors = []
        fields = "utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.sm"
        self.process = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits", "-l", "1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )

        def collect():
            for line in self.process.stdout:
                try:
                    values = [float(v.strip()) for v in next(csv.reader([line]))]
                    if len(values) != 6:
                        raise ValueError("Unexpected telemetry columns")
                    self.samples.append(
                        dict(
                            zip(
                                (
                                    "gpu_utilization_pct",
                                    "memory_utilization_pct",
                                    "memory_used_mb",
                                    "memory_total_mb",
                                    "power_w",
                                    "sm_clock_mhz",
                                ),
                                values,
                                strict=True,
                            )
                        )
                        | {"monotonic_s": time.monotonic()}
                    )
                except ValueError as error:
                    self.errors.append(str(error))

        self.thread = threading.Thread(target=collect, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.thread.join(timeout=5)
        self.process.stdout.close()

    def summary(self, start, end):
        return summarize_gpu(self.samples, start, end)


def preemptions(llm):
    # vLLM also exposes request_num_preemptions, a Histogram without .value.
    # The pinned reader's Counter name omits Prometheus's "_total" suffix.
    metrics = [m for m in llm.get_metrics() if m.name == "vllm:num_preemptions"]
    return sum(float(m.value) for m in metrics) if metrics else None


def measured_generate(engine, requests, *, greedy=False, output_sink=None):
    """Measure production batching without counting model loading as generation."""
    before = preemptions(engine.llm)
    with GPUMonitor() as monitor:
        started = time.monotonic()
        rows = engine.generate(requests, greedy=greedy)
        ended = time.monotonic()
        if output_sink is not None:
            output_sink(rows)
    after = preemptions(engine.llm)
    report = engine.last_performance | {
        "cost_scope": "generation_only_excludes_startup",
        "requests": len(requests),
        "prompt_tokens": sum(len(r["prompt_token_ids"]) for r in requests),
        "preemptions": after - before if before is not None and after is not None else None,
        "gpu": monitor.summary(started, ended),
        "telemetry_errors": monitor.errors,
    }
    return rows, report, monitor.samples


def select_batch_config(candidates):
    legal = [c for c in candidates if c["preemptions"] == 0 and c["gpu"]["samples"] > 0]
    if not legal:
        raise ValueError("No measured non-preempting configuration with GPU telemetry")
    return max(legal, key=lambda c: c["tokens_per_second"])
