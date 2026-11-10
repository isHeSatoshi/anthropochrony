import time

import pytest

from anthropochrony.sandbox import Sandbox

pytestmark = pytest.mark.security


@pytest.fixture
def sandbox():
    return Sandbox(memory_mb=128, tasks_max=16)


def run(sandbox, code, timeout=3):
    return sandbox.run(code, {"harness": "security"}, wall_s=timeout)


def test_no_environment_or_proc_secrets(sandbox, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "SECURITY_TEST_SENTINEL_NOT_A_TOKEN")
    result = run(
        sandbox,
        """
import os
from pathlib import Path
assert 'HF_TOKEN' not in os.environ
for p in Path('/proc').glob('[0-9]*/environ'):
    try:
        assert b'SECURITY_TEST_SENTINEL_NOT_A_TOKEN' not in p.read_bytes()
    except (PermissionError, FileNotFoundError, ProcessLookupError):
        pass
assert not Path('/home/adi/.cache/huggingface/token').exists()
""",
    )
    assert result["status"] == "pass", result


def test_candidate_failure_text_is_not_sandbox_unavailable(sandbox):
    result = run(sandbox, "raise AssertionError('Failed to connect to FTP server')")
    assert result["status"] == "fail", result


def test_no_network(sandbox):
    result = run(
        sandbox,
        """
import socket
s=socket.socket()
s.settimeout(.2)
try:
    s.connect(('1.1.1.1', 443))
except OSError:
    pass
else:
    raise AssertionError('Network reachable')
""",
    )
    assert result["status"] == "pass", result


def test_no_host_or_input_writes(sandbox):
    result = run(
        sandbox,
        """
from pathlib import Path
Path('/work/allowed').write_text('ok')
for p in ['/usr/forbidden', '/inputs/solution.py', '/home/forbidden']:
    try:
        Path(p).write_text('bad')
    except OSError:
        pass
    else:
        raise AssertionError(p)
""",
    )
    assert result["status"] == "pass", result


def test_memory_cap(sandbox):
    result = run(sandbox, "x = bytearray(512 * 1024 * 1024)")
    assert result["status"] != "pass"


def test_fork_bomb_contained(sandbox):
    started = time.monotonic()
    result = run(
        sandbox,
        """
import os, time
while True:
    try:
        os.fork()
    except OSError:
        time.sleep(.1)
""",
        timeout=1,
    )
    assert result["status"] == "timeout", result
    assert time.monotonic() - started < 8


def test_timeout_kills_descendants(sandbox, tmp_path):
    marker = tmp_path / "must-not-exist"
    code = f"""
import os, time
from pathlib import Path
if os.fork() == 0:
    time.sleep(2)
    Path({str(marker)!r}).write_text('escaped')
else:
    time.sleep(30)
"""
    assert run(sandbox, code, timeout=1)["status"] == "timeout"
    time.sleep(2)
    assert not marker.exists()


def test_per_case_timeout_records_actual_elapsed_not_whole_task_cap(sandbox):
    task = {
        "harness": "stdio",
        "io_examples": {"input": ["1"], "output": ["1"]},
        "time_limits": {"per_test_s": 0.15, "per_task_s": 5},
    }
    result = sandbox.run("import time\ntime.sleep(20)", task)
    assert result["status"] == "timeout"
    assert 0.15 <= result["wall_s"] < 2, result


def test_function_per_case_timeout(sandbox):
    task = {
        "harness": "evalplus_check",
        "entry_point": "f",
        "test_code": "assert f(1) == 1",
        "time_limits": {"per_test_s": 0.15, "per_task_s": 5},
    }
    result = sandbox.run("def f(x):\n while True: pass", task)
    assert result["status"] == "timeout"
    assert 0.15 <= result["wall_s"] < 2, result


def test_stdio_has_sized_seekable_input_and_fresh_cases(sandbox):
    task = {
        "harness": "stdio",
        "io_examples": {"input": ["héllo\n", "world\n"], "output": ["héllo", "world"]},
    }
    result = sandbox.run("import os\nprint(os.read(0, os.fstat(0).st_size).decode().strip())", task)
    assert result["status"] == "pass", result


@pytest.mark.parametrize(
    ("harness", "good", "bad", "tests", "extra"),
    [
        ("evalplus_check", "def f(x): return x+1", "def f(x): return 0", "assert f(1)==2", {}),
        (
            "lcd_check",
            "class Solution:\n def f(self): return 2",
            "class Solution:\n def f(self): return 0",
            "# <SOLUTION>\nassert Solution().f()==2",
            {},
        ),
        (
            "unittest",
            "def f(): return 2",
            "def f(): return 0",
            "import unittest\nclass T(unittest.TestCase):\n def test_f(self): self.assertEqual(f(),2)",
            {},
        ),
        (
            "pytest",
            "def f(): return 2",
            "def f(): return 0",
            "from solution import f\ndef test_f(): assert f()==2",
            {},
        ),
        (
            "stdio",
            "print(int(input())+1)",
            "print(0)",
            "",
            {"io_examples": {"input": ["1"], "output": ["2"]}},
        ),
    ],
)
def test_all_harnesses(sandbox, harness, good, bad, tests, extra):
    task = {"harness": harness, "test_code": tests, **extra}
    assert sandbox.run(good, task, wall_s=5)["status"] == "pass"
    assert sandbox.run(bad, task, wall_s=5)["status"] == "fail"
