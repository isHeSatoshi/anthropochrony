import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from anthropochrony.artifacts import file_hash


class SandboxUnavailable(RuntimeError):
    pass


def extract_code(text):
    matches = re.findall(r"```python[ \t]*\n(.*?)```", text, re.S | re.I)
    return "\n\n".join(matches) if matches else None


class Sandbox:
    def __init__(self, python=None, memory_mb=2048, tasks_max=128, cpu=None):
        self.python = Path(
            python
            or os.environ.get(
                "ANTHROPOCHRONY_EXEC_PYTHON",
                str(Path.home() / ".local/share/anthropochrony-envs/exec/bin/python"),
            )
        )
        self.memory_mb, self.tasks_max, self.cpu = memory_mb, tasks_max, cpu
        for binary in ("bwrap", "systemd-run", "systemctl", "timeout", "prlimit"):
            if shutil.which(binary) is None:
                raise SandboxUnavailable(f"{binary} is required; unsafe execution is not supported")
        if not self.python.is_file():
            raise SandboxUnavailable(
                f"Execution environment missing: {self.python}; run make exec-env"
            )

    def command(self, inputs, work, wall_s, scope):
        executable = self.python.readlink() if self.python.is_symlink() else self.python
        if not executable.is_absolute():
            executable = self.python.parent / executable
        runtime = executable.parent.parent
        args = [
            "systemd-run",
            "--user",
            "--scope",
            "--quiet",
            f"--unit={scope}",
            "-p",
            f"MemoryMax={self.memory_mb}M",
            "-p",
            "MemorySwapMax=0",
            "-p",
            f"TasksMax={self.tasks_max}",
            "-p",
            "CPUQuota=100%",
        ]
        if self.cpu is not None:
            args += ["-p", f"AllowedCPUs={self.cpu}"]
        args += [
            "--",
            "bwrap",
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--ro-bind",
            "/usr",
            "/usr",
            "--symlink",
            "usr/lib",
            "/lib",
            "--symlink",
            "usr/lib",
            "/lib64",
            "--symlink",
            "usr/bin",
            "/bin",
        ]
        if not str(runtime).startswith("/usr/"):
            args += ["--ro-bind", str(runtime.resolve()), str(runtime)]
        args += [
            "--ro-bind",
            str(self.python.parent.parent),
            "/env",
            "--ro-bind",
            str(inputs),
            "/inputs",
            "--bind",
            str(work),
            "/work",
            "--ro-bind",
            str(Path(__file__).with_name("worker.py")),
            "/runner.py",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--remount-ro",
            "/",
            "--chdir",
            "/work",
            "--clearenv",
        ]
        environment = {
            "PATH": "/env/bin:/usr/bin",
            "HOME": "/work",
            "PYTHONHASHSEED": "0",
            "TZ": "UTC",
            "LC_ALL": "C.UTF-8",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MPLBACKEND": "Agg",
            "CUDA_VISIBLE_DEVICES": "",
            "NLTK_DATA": "/env/nltk_data",
            "TF_CPP_MIN_LOG_LEVEL": "3",
        }
        for key, value in environment.items():
            args += ["--setenv", key, value]
        args += [
            "--",
            "/usr/bin/timeout",
            "-k",
            "1",
            str(wall_s),
            "/usr/bin/prlimit",
            f"--cpu={max(1, math.ceil(wall_s))}",
            "--fsize=67108864",
            "--nofile=256",
            "--core=0",
            "--",
            "/env/bin/python",
            "-I",
            "/runner.py",
        ]
        return args

    def run(self, code, task, *, wall_s=None):
        if code is None:
            return {
                "status": "no_code",
                "wall_s": 0.0,
                "cpu_s": 0.0,
                "first_failure": "",
                "feedback_tokens": 0,
            }
        wall_s = wall_s or task.get("time_limits", {}).get("per_task_s", 10)
        if not 0 < wall_s <= 120:
            raise ValueError("Sandbox timeout must be in (0, 120] seconds")
        scope = f"anthropochrony-{uuid.uuid4().hex}.scope"
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="anthropochrony-exec-") as directory:
            base = Path(directory)
            inputs, work = base / "inputs", base / "work"
            inputs.mkdir()
            work.mkdir()
            (inputs / "solution.py").write_text(code)
            (inputs / "task.json").write_text(json.dumps(task))
            (inputs / "test_solution.py").write_text(task.get("test_code") or "")
            command = self.command(inputs, work, wall_s, scope)
            # A bounded file prevents unbounded PIPE memory use from hostile stdout.
            with tempfile.TemporaryFile() as output:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=output,
                    env={
                        "PATH": os.environ.get("PATH", "/usr/bin"),
                        "XDG_RUNTIME_DIR": os.environ.get(
                            "XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"
                        ),
                        "DBUS_SESSION_BUS_ADDRESS": os.environ.get("DBUS_SESSION_BUS_ADDRESS", ""),
                    },
                    start_new_session=True,
                )
                timed_out = False
                try:
                    process.wait(timeout=wall_s + 8)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    subprocess.run(
                        ["systemctl", "--user", "kill", "--signal=KILL", scope],
                        capture_output=True,
                        check=False,
                    )
                    process.kill()
                    process.wait()
                finally:
                    # Also kill daemons that survived a successful harness.
                    subprocess.run(
                        ["systemctl", "--user", "kill", "--signal=KILL", scope],
                        capture_output=True,
                        check=False,
                    )
                elapsed = time.monotonic() - started
                output.seek(0)
                log = output.read(65536).decode(errors="replace")
            timing = re.search(r"ANTHROPOCHRONY_TIMING (\{[^\n]+\})", log)
            # Candidate output can say "Failed to connect"; only launcher failures count, and the
            # worker's timing marker proves the launcher succeeded.
            if not timing and re.search(
                r"^(bwrap: |Failed to connect to bus|Failed to start transient|Can't find)",
                log,
                re.MULTILINE,
            ):
                raise SandboxUnavailable(log[:1200])
            timeout = timed_out or process.returncode in {124, 137}
            status = "timeout" if timeout else "pass" if process.returncode == 0 else "fail"
            measured = json.loads(timing[1]) if timing else {}
            if process.returncode == 86:
                status = "error"
            return {
                "status": status,
                "wall_s": min(elapsed, wall_s),
                "cpu_s": measured.get("cpu_s", 0.0),
                "first_failure": "\n".join(log.splitlines()[-25:])[-1200:]
                if status != "pass"
                else "",
                "feedback_tokens": 0,
                "execution_lock_sha": file_hash(
                    Path(__file__).parents[1] / "execenv/requirements.lock"
                )
                if (Path(__file__).parents[1] / "execenv/requirements.lock").exists()
                else None,
            }
