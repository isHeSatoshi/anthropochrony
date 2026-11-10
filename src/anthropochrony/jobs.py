import json
import os
import re
import shutil
import subprocess
from importlib.metadata import distribution
from pathlib import Path

from huggingface_hub import get_token

from anthropochrony.artifacts import digest, file_hash, git_sha, now, write_json
from anthropochrony.budget import Budget, job_cost


def launch(
    root,
    name,
    *,
    flavor,
    timeout_s,
    image,
    command,
    volumes=(),
    secret=False,
    approved_over_five=False,
    dry_run=False,
    uv_script=False,
):
    root = Path(root)
    args = [
        "hf",
        "jobs",
        *(["uv", "run"] if uv_script else ["run"]),
        "--detach",
        "--quiet",
        "--name",
        name,
        "--flavor",
        flavor,
        "--timeout",
        f"{timeout_s}s",
        "--attempts",
        "1",
        "--label",
        "project=anthropochrony",
    ]
    for volume in volumes:
        args += ["--volume", volume]
    if secret:
        args += ["--secrets", "HF_TOKEN"]
    if dry_run:
        return {
            "command": args + ["--dry-run", image] + list(command),
            "upper_bound_usd": job_cost(flavor, timeout_s),
        }
    env = dict(os.environ)
    if secret:
        token = get_token()
        if not token:
            raise PermissionError("HF login is missing; stop remote work and renew access")
        env["HF_TOKEN"] = token
    cost = job_cost(flavor, timeout_s)
    budget = Budget(root)
    budget.reserve(name, cost, approved_over_five=approved_over_five)
    with (root / "RUNLOG.md").open("a") as log:
        log.write(f"\nReservation {name}: ≤${cost:.4f}, before submitting the paid job.\n")
    result = subprocess.run(
        args + [image] + list(command), env=env, capture_output=True, text=True, check=False
    )
    if result.returncode:
        # A CLI error is not billing evidence: submission may have reached HF.
        budget.update(name, state="submission_failed_reservation_retained")
        write_json(
            root / "artifacts/jobs" / f"{name}.json",
            {
                "name": name,
                "flavor": flavor,
                "timeout_s": timeout_s,
                "upper_bound_usd": cost,
                "submission_outcome": "unconfirmed",
            },
        )
        with (root / "RUNLOG.md").open("a") as log:
            log.write(
                f"\nSubmission {name} failed; outcome unconfirmed and the full reservation is retained.\n"
            )
        if re.search(r"\b(?:401|403)\b", result.stderr):
            raise PermissionError(
                "HF authorization failed; stop remote work and request renewed access"
            )
        raise RuntimeError(result.stderr[:1600])
    job_id = result.stdout.strip().splitlines()[-1]
    budget.update(name, state="launched", job_id=job_id, flavor=flavor, timeout_s=timeout_s)
    write_json(
        root / "artifacts/jobs" / f"{name}.json",
        {
            "job_id": job_id,
            "name": name,
            "flavor": flavor,
            "timeout_s": timeout_s,
            "upper_bound_usd": cost,
            "launched_at": now(),
        },
    )
    with (root / "RUNLOG.md").open("a") as log:
        log.write(
            f"| {job_id} | {name} | {flavor} | ≤{timeout_s / 60:.1f} | ≤{cost:.4f} | Launched |\n"
        )
    return {"job_id": job_id, "reserved_usd": cost}


def vendor_auditkit(destination):
    """Copy the installed, pinned pure-Python wheel, including its license."""
    dependency = distribution("auditkit")
    if dependency.version != "1.1.6":
        raise ValueError("GPU evaluation requires the pinned AuditKIT 1.1.6")
    destination = Path(destination)
    hashes = {}
    for relative in dependency.files or ():
        parts = relative.parts
        if not parts or parts[0] not in {"auditkit", "auditkit-1.1.6.dist-info"}:
            continue
        if "__pycache__" in parts or relative.name in {"RECORD", "INSTALLER", "direct_url.json"}:
            continue
        if relative.suffix == ".pyc":
            continue
        source = dependency.locate_file(relative)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[str(relative)] = file_hash(target)
    if "auditkit-1.1.6.dist-info/licenses/LICENSE.md" not in hashes:
        raise ValueError("AuditKIT redistribution license is missing")
    write_json(
        destination / "VENDOR_PROVENANCE.json",
        {"distribution": "auditkit", "version": dependency.version, "files": hashes},
    )


def payload(root, destination, files=(), *, include_auditkit=False):
    """Stage source/config only, never token caches, .git, env files, or unrelated user files."""
    root, destination = Path(root), Path(destination)
    files = list(files)
    if any(Path(relative).is_absolute() or ".." in Path(relative).parts for _, relative in files):
        raise ValueError("Payload destinations must stay inside their immutable directory")
    if destination.exists():
        if destination.is_symlink() or include_auditkit:
            raise FileExistsError("Use a fresh immutable payload name")
        # A launch preview must not prevent its subsequent real submission.
        expected = {}
        for name in ("src", "configs", "jobs", "docs/PROMPTS.md", "uv.lock"):
            source = root / name
            candidates = source.rglob("*") if source.is_dir() else [source]
            for item in candidates:
                if item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc":
                    expected[str(item.relative_to(root))] = file_hash(item)
        for source, relative in files:
            expected[str(relative)] = file_hash(source)
        identity = destination / "SOURCE_IDENTITY.json"
        allowed = set(expected) | {"SOURCE_IDENTITY.json", "SOURCE_IDENTITY.json.manifest.json"}
        actual = {
            str(path.relative_to(destination)): path
            for path in destination.rglob("*")
            if path.is_file()
        }
        if (
            set(actual) == allowed
            and not any(path.is_symlink() for path in actual.values())
            and json.loads(identity.read_text())["files"] == expected
            and json.loads(identity.read_text())["source_sha"] == digest(expected)
            and all(file_hash(destination / name) == value for name, value in expected.items())
        ):
            return destination
        raise FileExistsError("Existing payload changed; use a new immutable launch name")
    destination.mkdir(parents=True)
    for name in ("src", "configs", "jobs", "docs/PROMPTS.md", "uv.lock"):
        source = root / name
        target = destination / name
        if source.is_dir():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        elif source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    for source, relative in files:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    if include_auditkit:
        vendor_auditkit(destination / "vendor")
    hashes = {
        str(p.relative_to(destination)): file_hash(p)
        for p in sorted(destination.rglob("*"))
        if p.is_file()
    }
    write_json(
        destination / "SOURCE_IDENTITY.json",
        {"git_sha": git_sha(root), "source_sha": digest(hashes), "files": hashes},
    )
    return destination
