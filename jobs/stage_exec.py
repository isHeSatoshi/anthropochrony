# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["uv==0.13.7"]
# ///
"""Probe namespaces first, then archive a token-free execution environment."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirements", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if os.environ.get("HF_TOKEN"):
        raise RuntimeError("Execution-environment staging must not receive a token")
    # Object storage does not preserve executable bits or symlinks. Build locally
    # at a fixed prefix and transport one archive, never a loose mounted venv.
    target = Path("/opt/anthropochrony-exec")
    target.mkdir(parents=True, exist_ok=False)
    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run(
        ["apt-get", "install", "-y", "--no-install-recommends", "bubblewrap"], check=True
    )
    binary = target / "tools/usr/bin/bwrap"
    binary.parent.mkdir(parents=True)
    shutil.copy2("/usr/bin/bwrap", binary)

    def drop():
        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)

    probe = subprocess.run(
        [
            str(binary),
            "--unshare-all",
            "--ro-bind",
            "/usr",
            "/usr",
            "--symlink",
            "usr/lib",
            "/lib",
            "--symlink",
            "usr/lib",
            "/lib64",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--clearenv",
            "--",
            "/usr/bin/true",
        ],
        env={"PATH": "/usr/bin"},
        capture_output=True,
        text=True,
        preexec_fn=drop,
        timeout=10,
        check=False,
    )
    evidence = {
        "job_id": os.environ.get("JOB_ID"),
        "bubblewrap": str(binary),
        "bubblewrap_sha256": sha256(binary),
        "requirements_sha256": sha256(args.requirements),
        "namespace_probe": {"returncode": probe.returncode, "stderr": probe.stderr[:1200]},
        "credentials_in_environment": False,
        "security_tests_passed": False,
        "note": "Staging is not permission to execute candidates; full security probes remain required.",
    }
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    if probe.returncode:
        evidence["stage_status"] = "blocked_namespace_probe"
        report.write_text(json.dumps(evidence, indent=2))
        print("Namespace probe failed; no candidates executed and no environment build attempted.")
        return

    env = dict(os.environ, UV_PYTHON_INSTALL_DIR=str(target / "python"))
    python = target / "venv/bin/python"
    subprocess.run(["uv", "venv", "--python", "3.10", str(target / "venv")], env=env, check=True)
    subprocess.run(
        ["uv", "pip", "sync", "--python", str(python), args.requirements], env=env, check=True
    )
    with tempfile.TemporaryDirectory(prefix="nltk-staging-") as data:
        subprocess.run(
            [
                str(python),
                "-c",
                "import nltk,sys; [nltk.download(x,download_dir=sys.argv[1],quiet=True,raise_on_error=True) "
                "for x in ['punkt','punkt_tab','stopwords','wordnet','averaged_perceptron_tagger',"
                "'averaged_perceptron_tagger_eng']]",
                data,
            ],
            env=env,
            check=True,
        )
        shutil.copytree(
            data,
            target / "venv/nltk_data",
            ignore=shutil.ignore_patterns(".DS_Store", "__MACOSX", "._*"),
        )
    shutil.copyfile(args.requirements, target / "requirements.lock")
    archive = Path(args.archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz", compresslevel=3) as output:
        output.add(target, arcname="anthropochrony-exec")
    evidence.update(
        stage_status="archived_security_tests_pending",
        python=str(python),
        archive=archive.name,
        archive_sha256=sha256(archive),
        archive_bytes=archive.stat().st_size,
        extraction_prefix="/opt",
    )
    report.write_text(json.dumps(evidence, indent=2))
    print("Execution environment archived; full security validation still required.")


if __name__ == "__main__":
    main()
