"""Run an agent command under a timeout taken from Anthropochrony's own-time estimate.

Before the work starts, the tuned model estimates how long Tiny Aya will take on this machine.
The run is stopped if it passes `k` times the 80% upper bound (the tripwire).

The estimate describes Tiny Aya solving the task through your local llama-server, so --cmd
should be an agent that uses that model. The command is yours; this script never runs code
written by the model. The command receives the task text on stdin.

    python examples/agent_with_tripwire.py --task task.md --speed 20,200 \
        --cmd "python my_agent.py"
"""

import argparse
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from anthropochrony.cli import speed_pair
from anthropochrony.families import FAMILIES
from anthropochrony.selftime import SelfTimeEstimator


@dataclass(frozen=True)
class RunReport:
    elapsed_s: float
    returncode: int | None
    tripped: bool


def run_with_tripwire(command: str, task_text: str, timeout_s: float) -> RunReport:
    """Run `command` with the task on stdin and kill it once it passes `timeout_s`."""
    started = time.monotonic()
    try:
        done = subprocess.run(
            shlex.split(command), input=task_text, text=True, timeout=timeout_s, check=False
        )
    except subprocess.TimeoutExpired:
        return RunReport(time.monotonic() - started, None, True)
    return RunReport(time.monotonic() - started, done.returncode, False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", type=Path, required=True, help="File with the task description")
    parser.add_argument("--cmd", required=True, help="Your agent command, e.g. 'python agent.py'")
    parser.add_argument("--server", default="http://127.0.0.1:8082")
    parser.add_argument("--speed", type=speed_pair, metavar="S,P", help="Decode,prefill tok/s")
    parser.add_argument("--llama-bench", help="Measure the speed with llama-bench instead")
    parser.add_argument("--gguf", help="GGUF for llama-bench")
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--k", type=float, default=1.5, help="Tripwire = k x 80%% upper bound")
    args = parser.parse_args(argv)
    if args.speed is None and not (args.llama_bench and args.gguf):
        parser.error("give --speed S,P, or --llama-bench and --gguf")

    estimator = SelfTimeEstimator.from_llama_server(
        args.server, speed=args.speed, llama_bench=args.llama_bench, gguf=args.gguf
    )
    task_text = args.task.read_text(encoding="utf-8")
    estimate = estimator.estimate(task_text, family=args.family)
    timeout_s = estimator.tripwire_seconds(estimate, k=args.k)
    print(f"Estimate: {estimate.summary()}")
    print(f"Tripwire: stop after {timeout_s:.1f} s ({args.k:g} x the 80% upper bound)")

    report = run_with_tripwire(args.cmd, task_text, timeout_s)
    ratio = report.elapsed_s / estimate.seconds
    print(f"Actual: {report.elapsed_s:.1f} s ({ratio:.2f} x the estimate)")
    if report.tripped:
        print("Tripwire fired: the run passed the upper bound and was stopped.")
        return 124
    inside = estimate.low <= report.elapsed_s <= estimate.high
    print(f"Finished with exit code {report.returncode}; inside the 80% range: {inside}")
    return report.returncode


if __name__ == "__main__":
    sys.exit(main())
