"""Fail-closed acceptance gates and pre-registration enforcement."""

import json
import subprocess
from pathlib import Path

from anthropochrony.artifacts import digest, write_json


class GateError(RuntimeError):
    pass


def record_gate(root, name, checks, *, evidence=None):
    passed = bool(checks) and all(checks.values())
    write_json(
        Path(root) / "results" / "gates" / f"{name}.json",
        {"gate": name, "passed": passed, "checks": checks, "evidence": evidence or {}},
        config=checks,
    )
    if not passed:
        raise GateError(f"{name} failed: {[k for k, v in checks.items() if not v]}")


def require_gate(root, name):
    path = Path(root) / "results" / "gates" / f"{name}.json"
    if not path.exists() or not json.loads(path.read_text()).get("passed"):
        raise GateError(f"Required acceptance gate {name} has not passed")


def require_preregistration(root, splits, fine_tuned):
    if not fine_tuned or not set(splits) & {"test", "test_ood"}:
        return
    root = Path(root)
    frozen = subprocess.run(
        ["git", "show", "prereg-v1:docs/PREREGISTRATION.md"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if frozen.returncode:
        raise GateError("Freeze docs/PREREGISTRATION.md with tag prereg-v1 before test evaluation")
    current = (root / "docs/PREREGISTRATION.md").read_bytes()
    if current != frozen.stdout:
        import re

        prefix = frozen.stdout.rstrip()
        appended = current[len(prefix) :] if current.startswith(prefix) else b""
        if b"## 7. Deviations" not in prefix or not re.match(rb"\s+- \d{4}-\d{2}-\d{2},", appended):
            raise GateError(
                "Frozen pre-registration changed; only appended, dated deviations are permitted"
            )


def input_fingerprint(tasks, config, model_sha):
    return digest({"tasks": tasks, "config": config, "model_sha": model_sha})
