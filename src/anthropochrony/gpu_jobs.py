"""Generic, resumable, budget-reserved GPU-only job with hash-checked downloads.

Used for merged-model serving, G6 equivalence and estimate generation. No generated
code runs in these jobs; candidate execution stays in the local credential-free sandbox.
"""

import json
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from anthropochrony.artifacts import digest, file_hash, now, write_json
from anthropochrony.budget import Budget
from anthropochrony.cli import config
from anthropochrony.gates import require_gate
from anthropochrony.jobs import launch, payload
from anthropochrony.measure.remote import (
    BUCKET_ID,
    TERMINAL,
    QueueTimeout,
    launch_name,
    settle_unstarted,
    wait_for_job,
)
from anthropochrony.remote import authorization_guard

COMMON_SOURCES = (
    "src/anthropochrony/gen/engine.py",
    "src/anthropochrony/gen/performance.py",
    "src/anthropochrony/artifacts.py",
    "src/anthropochrony/timemodel.py",
    "src/anthropochrony/parse.py",
    "src/anthropochrony/train/sft.py",
    "jobs/model_prep.py",
    "configs/pins.yaml",
    "uv.lock",
)


class GPUJob:
    """Launch once, wait to a verified terminal state, then download hashed artifacts."""

    def __init__(
        self,
        root,
        directory,
        *,
        kind,
        script,
        files,
        arguments,
        outputs,
        timeout_s,
        context,
        sources=(),
        include_auditkit=False,
        required_gates=("G0-security", "G1", "G2"),
    ):
        self.root, self.directory = Path(root), Path(directory)
        self.kind, self.script, self.files = kind, script, [(Path(p), n) for p, n in files]
        self.arguments, self.outputs, self.timeout_s = list(arguments), list(outputs), timeout_s
        self.include_auditkit, self.required_gates = include_auditkit, required_gates
        self.sources = tuple(dict.fromkeys(COMMON_SOURCES + (script,) + tuple(sources)))
        self.context = {
            "kind": kind,
            "batch_id": str(self.directory.relative_to(self.root)),
            "inputs": {name: file_hash(path) for path, name in self.files},
            "arguments": self.arguments,
            "source_hashes": {name: file_hash(self.root / name) for name in self.sources},
            "timeout_s": timeout_s,
            "extra": context,
        }
        self.base_name = f"{kind}-{digest(self.context)[:24]}"
        self.name = self.base_name

    def _complete(self):
        if not all((self.directory / name).exists() for name in self._names()):
            return False
        for name in self._names():
            if name.endswith(".manifest.json"):
                continue
            record = json.loads((self.directory / f"{name}.manifest.json").read_text())
            if record["artifact_sha256"] != file_hash(self.directory / name):
                raise ValueError(f"Cached GPU artifact failed its content hash: {name}")
        marker = self.directory / "job-context.json"
        if not marker.exists() or json.loads(marker.read_text()) != self.context:
            raise ValueError("Cached GPU artifacts belong to a different job context")
        return True

    def _names(self):
        names = []
        for name in self.outputs:
            names += [name, f"{name}.manifest.json"]
        return names

    def run(self):
        for gate in self.required_gates:
            require_gate(self.root, gate)
        self.directory.mkdir(parents=True, exist_ok=True)
        if self._complete():
            return self.directory
        self.name = launch_name(self.root, self.base_name)
        metadata = self.root / f"artifacts/jobs/{self.name}.json"
        if metadata.exists():
            job = json.loads(metadata.read_text())
            if job.get("round_context") != self.context:
                raise ValueError("Existing GPU job lacks matching provenance; inspect first")
        else:
            with Budget(self.root).ledger() as ledger:
                if any(row["name"] == self.name for row in ledger["entries"]):
                    raise ValueError("GPU submission already reserved; inspect its outcome")
            code = payload(
                self.root,
                self.root / f"artifacts/payloads/{self.name}",
                files=self.files,
                include_auditkit=self.include_auditkit,
            )
            launch(
                self.root,
                self.name,
                flavor="l40sx1",
                timeout_s=self.timeout_s,
                image=config(self.root, "pins.yaml")["generation_image"],
                volumes=[f"{code}:/code", f"hf://buckets/{BUCKET_ID}:/out"],
                command=[
                    "python3",
                    f"/code/{self.script}",
                    *self.arguments,
                    "--output-dir",
                    f"/out/automatic/{self.name}",
                ],
                secret=False,
            )
            job = json.loads(metadata.read_text()) | {"round_context": self.context}
            write_json(metadata, job)
        api, terminal = HfApi(), None
        try:
            with authorization_guard():
                terminal = wait_for_job(api, job["job_id"], self.timeout_s)
                Budget(self.root).update(self.name, state=terminal.lower(), terminal_verified=True)
                if terminal != "COMPLETED":
                    raise RuntimeError(f"GPU job ended {terminal}; inspect it before retrying")
                self._download(api)
        except BaseException as error:
            if isinstance(error, QueueTimeout) and settle_unstarted(
                self.root, api, self.name, job["job_id"]
            ):
                terminal = "CANCELED_UNSTARTED"
                raise
            if terminal not in TERMINAL:
                accepted = False
                try:
                    api.cancel_job(job_id=job["job_id"])
                    accepted = True
                finally:
                    Budget(self.root).update(
                        self.name,
                        state="cancel_requested" if accepted else "cancel_failed_timeout_reserved",
                    )
            raise
        finally:
            with (self.root / "RUNLOG.md").open("a") as log:
                log.write(
                    f"\n{now()}: {self.kind} {job['job_id']} terminal={terminal}; "
                    "reservation retained pending billing. No candidate ran in the GPU job.\n"
                )
        return self.directory

    def _download(self, api):
        with tempfile.TemporaryDirectory(prefix=".download-", dir=self.directory) as temporary:
            temporary = Path(temporary)
            for name in self._names():
                (temporary / name).parent.mkdir(parents=True, exist_ok=True)
            api.download_bucket_files(
                BUCKET_ID,
                [(f"automatic/{self.name}/{name}", temporary / name) for name in self._names()],
                raise_on_missing_files=True,
            )
            for name in self.outputs:
                record = json.loads((temporary / f"{name}.manifest.json").read_text())
                if record["artifact_sha256"] != file_hash(temporary / name):
                    raise ValueError(f"Downloaded GPU artifact failed its hash: {name}")
            for name in self._names()[1::2] + self._names()[0::2]:
                target = self.directory / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and target.read_bytes() != (temporary / name).read_bytes():
                    raise FileExistsError("Refusing to overwrite a changed GPU artifact")
                if not target.exists():
                    (temporary / name).replace(target)
            write_json(self.directory / "job-context.json", self.context)
