"""Atomic artifacts with content-addressed provenance. Never record environment values."""

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path


def now():
    return datetime.now(UTC).isoformat()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha(root=None):
    identity = Path(root or ".") / "SOURCE_IDENTITY.json"
    if identity.exists():
        return json.loads(identity.read_text())["git_sha"]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return "unavailable"
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"


def manifest(path, *, inputs=(), config=None, started=None, hardware=None):
    path = Path(path)
    root = Path(__file__).resolve().parents[2]
    lock = root / "uv.lock"
    record = {
        "schema_version": 1,
        "artifact_sha256": file_hash(path),
        "git_sha": git_sha(root),
        "source_identity": json.loads((root / "SOURCE_IDENTITY.json").read_text())
        if (root / "SOURCE_IDENTITY.json").exists()
        else None,
        "config_hash": digest(config or {}),
        "input_hashes": {str(Path(p)): file_hash(p) for p in inputs},
        "environment_lock_hash": file_hash(lock) if lock.exists() else None,
        "hardware": hardware or {"cpu": platform.processor(), "machine": platform.machine()},
        "job_id": os.environ.get("HF_JOB_ID"),
        "started_at": started or now(),
        "ended_at": now(),
        "python": platform.python_version(),
    }
    _atomic_text(Path(f"{path}.manifest.json"), json.dumps(record, indent=2) + "\n")
    return record


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path, value, **provenance):
    _atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return manifest(path, **provenance)


def write_jsonl(path, rows, **provenance):
    _atomic_text(
        path,
        "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows),
    )
    return manifest(path, **provenance)


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_parquet(path, rows, **provenance):
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pending.parquet")
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    os.replace(temporary, path)
    return manifest(path, **provenance)
