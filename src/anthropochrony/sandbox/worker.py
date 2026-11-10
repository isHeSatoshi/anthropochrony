"""Trusted harness entry point, mounted read-only inside bubblewrap. Python 3.10."""

import contextlib
import io
import json
import math
import re
import resource
import runpy
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path

sys.path.insert(0, "/inputs")


def outputs_match(output, expected):
    # Port of google-deepmind/code_contests OutputsMatch/ValuesMatch, Apache-2.0,
    # execution/tester_sandboxer.cc @ fa7a4f8139aab08362503f3344778eb86901709a.
    lower = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")
    actual = [x.translate(lower) for x in re.split(r"[ \n\t\r\v]+", output) if x]
    wanted = [x.translate(lower) for x in re.split(r"[ \n\t\r\v]+", expected) if x]
    if actual == wanted:
        return True
    if len(actual) != len(wanted):
        return False
    numeric = r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?|inf(?:inity)?|nan)"
    for first, second in zip(actual, wanted, strict=True):
        if first == second:
            continue
        if not re.fullmatch(numeric, first, re.ASCII) or not re.fullmatch(
            numeric, second, re.ASCII
        ):
            return False
        a, b = float(first), float(second)
        if not math.isfinite(a) or not math.isfinite(b) or not abs(a - b) < 1e-5:
            return False
    return True


def execute(task):
    code = Path("/inputs/solution.py").read_text()
    tests = task.get("test_code", "")
    harness = task["harness"]
    if harness == "evalplus_check":
        namespace = {"__name__": "__main__", "__file__": "/inputs/solution.py"}
        exec(compile(code, "<candidate>", "exec"), namespace)
        entry = task.get("entry_point")
        if entry and callable(namespace.get(entry)):
            candidate = namespace[entry]
            depth = 0
            limit = task.get("time_limits", {}).get("per_test_s", 4)

            def expired(signum, frame):
                raise subprocess.TimeoutExpired("candidate", limit)

            signal.signal(signal.SIGALRM, expired)

            def guarded(*args, **kwargs):
                nonlocal depth
                if depth == 0:
                    signal.setitimer(signal.ITIMER_REAL, limit)
                depth += 1
                try:
                    return candidate(*args, **kwargs)
                finally:
                    depth -= 1
                    if depth == 0:
                        signal.setitimer(signal.ITIMER_REAL, 0)

            namespace[entry] = guarded
        exec(compile(tests, "<tests>", "exec"), namespace)
    elif harness == "lcd_check":
        if "# <SOLUTION>" in tests:
            source = tests.replace("# <SOLUTION>", code)
        else:
            source = code + "\n" + tests
        namespace = {"__name__": "__main__", "__file__": "/inputs/solution.py"}
        exec(compile(source, "<candidate-and-tests>", "exec"), namespace)
    elif harness == "unittest":
        namespace = {"__name__": "candidate_tests", "__file__": "/inputs/solution.py"}
        # Imported benchmark test code and solution share the official harness namespace.
        exec(compile(code + "\n" + tests, "<candidate-and-tests>", "exec"), namespace)
        suite = unittest.TestSuite()
        for value in list(namespace.values()):
            if (
                isinstance(value, type)
                and issubclass(value, unittest.TestCase)
                and value is not unittest.TestCase
            ):
                suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(value))
        if suite.countTestCases() == 0:
            raise RuntimeError("Harness contains no unittest tests")
        result = unittest.TextTestRunner(stream=sys.stderr, verbosity=1).run(suite)
        if not result.wasSuccessful():
            raise AssertionError("unittest failed")
    elif harness == "pytest":
        import pytest

        if (
            pytest.main(
                ["-q", "-x", "--tb=short", "-p", "no:cacheprovider", "/inputs/test_solution.py"]
            )
            != 0
        ):
            raise AssertionError("pytest failed")
    elif harness == "stdio":
        examples = task["io_examples"]
        if not examples["input"] or len(examples["input"]) != len(examples["output"]):
            raise RuntimeError("Invalid stdin/stdout test set")
        limit = task.get("time_limits", {}).get("per_test_s", 3)
        for index, (given, expected) in enumerate(
            zip(examples["input"], examples["output"], strict=True)
        ):
            # Each case gets a fresh interpreter; all children share the sandbox cgroup.
            # Upstream maps a sized memory-file descriptor to stdin. A pipe has
            # st_size == 0 and breaks valid os.read(0, os.fstat(0).st_size) solutions.
            with tempfile.TemporaryFile(mode="w+b") as stdin:
                stdin.write(given.encode("utf-8"))
                stdin.seek(0)
                result = subprocess.run(
                    [sys.executable, "-I", "/inputs/solution.py"],
                    stdin=stdin,
                    text=True,
                    capture_output=True,
                    timeout=limit,
                    check=False,
                )
            if result.returncode or not outputs_match(result.stdout, expected):
                raise AssertionError(f"stdio case {index} failed")
    elif harness == "security":
        runpy.run_path("/inputs/solution.py", run_name="__main__")
    else:
        raise RuntimeError(f"Unknown harness: {harness}")


def main():
    task = json.loads(Path("/inputs/task.json").read_text())
    start = time.monotonic()
    status = 0
    try:
        # Capture ordinary candidate prints; assertions remain available as failure feedback.
        with contextlib.redirect_stdout(io.StringIO()):
            execute(task)
    except subprocess.TimeoutExpired:
        status = 124
    except BaseException:
        traceback.print_exc(limit=12)
        status = 1
    usage = resource.getrusage(resource.RUSAGE_SELF)
    print(
        "ANTHROPOCHRONY_TIMING "
        + json.dumps(
            {
                "wall_s": time.monotonic() - start,
                "cpu_s": usage.ru_utime + usage.ru_stime,
            }
        )
    )
    return status


if __name__ == "__main__":
    sys.exit(main())
