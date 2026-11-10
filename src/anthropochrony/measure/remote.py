"""GPU-only round jobs; the VM terminates before any local candidate tests."""

import json
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi

from anthropochrony.artifacts import digest, file_hash, now, read_jsonl, write_json
from anthropochrony.budget import Budget
from anthropochrony.cli import config
from anthropochrony.gates import require_gate
from anthropochrony.jobs import launch, payload
from anthropochrony.remote import authorization_guard

BUCKET_ID = "isHeSatoshi/anthropochrony-runs"
TERMINAL = {"COMPLETED", "ERROR", "FAILED", "CANCELED", "DELETED", "TIMEOUT"}
GENERATION_SOURCES = (
    "jobs/measure_round.py",
    "src/anthropochrony/gen/engine.py",
    "src/anthropochrony/gen/performance.py",
    "src/anthropochrony/artifacts.py",
    "src/anthropochrony/timemodel.py",
    "configs/pins.yaml",
    "uv.lock",
)


class QueueTimeout(TimeoutError):
    """Hardware was never allocated, so no paid time has started."""


def launch_name(root, base):
    """Reuse `base`, or `base-retryN` after earlier attempts that were canceled unstarted."""
    from anthropochrony.budget import Budget

    with Budget(root).ledger() as ledger:
        states = {row["name"]: row.get("state") for row in ledger["entries"]}
    for attempt in range(8):
        name = base if attempt == 0 else f"{base}-retry{attempt}"
        if states.get(name) != "canceled_unstarted":
            return name
    raise RuntimeError("Eight queued GPU attempts were canceled; inspect HF capacity first")


def settle_unstarted(root, api, name, job_id):
    """Zero-cost the reservation only after HF confirms cancellation before any start."""
    from anthropochrony.budget import Budget

    api.cancel_job(job_id=job_id)
    for _ in range(40):
        job = api.inspect_job(job_id=job_id)
        if job.status.stage in TERMINAL:
            break
        time.sleep(3)
    else:
        return False
    if job.status.stage != "CANCELED" or getattr(job, "started_at", None):
        return False
    Budget(root).update(name, state="canceled_unstarted", terminal_verified=True, actual_usd=0.0)
    metadata = Path(root) / f"artifacts/jobs/{name}.json"
    metadata.replace(metadata.with_suffix(".unstarted.json"))
    return True


def wait_for_job(api, job_id, timeout_s, queue_timeout_s=21600):
    queued_at, allocated_at, previous = time.monotonic(), None, None
    while True:
        job = api.inspect_job(job_id=job_id)
        stage = job.status.stage
        if stage != previous:
            print(f"GPU round {job_id}: {stage}", flush=True)
            previous = stage
        if stage in TERMINAL:
            return stage
        current = time.monotonic()
        if stage in {"STARTING", "RUNNING"} and allocated_at is None:
            allocated_at = current
        if allocated_at is None and current - queued_at >= queue_timeout_s:
            raise QueueTimeout("GPU round exceeded its hardware-allocation deadline")
        if allocated_at is not None and current - allocated_at >= timeout_s + 120:
            raise TimeoutError("GPU round did not reach a terminal state after its hard timeout")
        time.sleep(15)


class HFRemoteRound:
    def __init__(self, root, directory, measure_config, *, timeout_s=1200):
        self.root, self.directory = Path(root), Path(directory)
        self.config, self.timeout_s = measure_config, timeout_s
        self.directory.mkdir(parents=True, exist_ok=True)
        # Fine-tuned serving must use the forthcoming checked in-memory merge,
        # never a fictitious bucket path or the base model under an FT identity.
        subject = config(self.root, "pins.yaml")["subject"]
        if (measure_config["model_sha"], measure_config["model_id"]) != (
            subject["revision"],
            subject["repo"],
        ):
            raise ValueError("This runner currently accepts only the pinned base model")

    def _source_hashes(self):
        return {name: file_hash(self.root / name) for name in GENERATION_SOURCES}

    def generate_file(self, requests):
        require_gate(self.root, "G0-security")
        require_gate(self.root, "G1")
        require_gate(self.root, "G2")
        requests = Path(requests)
        output = self.directory / "generation.jsonl"
        sha = file_hash(requests)
        if output.exists():
            self._verify_output(output, sha)
            self._verify_telemetry(self.directory)
            return read_jsonl(output)
        context = {
            "batch_id": str(self.directory.relative_to(self.root)),
            "requests_sha256": sha,
            "config": self.config,
            "generation_source_hashes": self._source_hashes(),
            "timeout_s": self.timeout_s,
        }
        name = launch_name(self.root, f"generate-{digest(context)[:24]}")
        metadata = self.root / f"artifacts/jobs/{name}.json"
        round_config = self.directory / "round-config.json"
        write_json(round_config, self.config)
        if metadata.exists():
            job = json.loads(metadata.read_text())
            if job.get("round_context") != context:
                raise ValueError(
                    "Existing GPU job lacks matching round provenance; inspect before retrying"
                )
        else:
            with Budget(self.root).ledger() as ledger:
                if any(row["name"] == name for row in ledger["entries"]):
                    raise ValueError(
                        "GPU submission already reserved; inspect its outcome before retrying"
                    )
            code = payload(
                self.root,
                self.root / f"artifacts/payloads/{name}",
                files=[(requests, "requests.jsonl"), (round_config, "round-config.json")],
            )
            launch(
                self.root,
                name,
                flavor="l40sx1",
                timeout_s=self.timeout_s,
                image=config(self.root, "pins.yaml")["generation_image"],
                volumes=[f"{code}:/code", f"hf://buckets/{BUCKET_ID}:/out"],
                command=[
                    "python3",
                    "/code/jobs/measure_round.py",
                    "--requests",
                    "/code/requests.jsonl",
                    "--model",
                    "/out/base",
                    "--config",
                    "/code/round-config.json",
                    "--output",
                    f"/out/automatic/{name}/generation.jsonl",
                ],
                secret=False,
            )
            job = json.loads(metadata.read_text()) | {"round_context": context}
            write_json(metadata, job)
        api, terminal = HfApi(), None
        try:
            with authorization_guard():
                terminal = wait_for_job(api, job["job_id"], self.timeout_s)
                Budget(self.root).update(name, state=terminal.lower(), terminal_verified=True)
                if terminal != "COMPLETED":
                    raise RuntimeError(
                        f"GPU generation ended {terminal}; inspect the job before retrying"
                    )
                self._download(api, name, sha)
        except BaseException as error:
            if isinstance(error, QueueTimeout) and settle_unstarted(
                self.root, api, name, job["job_id"]
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
                        name,
                        state="cancel_requested" if accepted else "cancel_failed_timeout_reserved",
                    )
            raise
        finally:
            with (self.root / "RUNLOG.md").open("a") as log:
                log.write(
                    f"\n{now()}: generation {job['job_id']} terminal={terminal}; "
                    "reservation retained pending billing. No candidate ran in the GPU job.\n"
                )
        return read_jsonl(output)

    def _verify_output(self, output, request_sha):
        record = json.loads(Path(f"{output}.manifest.json").read_text())
        if (
            record["artifact_sha256"] != file_hash(output)
            or record["input_hashes"].get("/code/requests.jsonl") != request_sha
            or record["config_hash"] != digest(self.config)
            or any(
                (record.get("source_identity") or {}).get("files", {}).get(name) != sha
                for name, sha in self._source_hashes().items()
            )
        ):
            raise ValueError("Saved GPU outputs do not match their requests and configuration")

    @staticmethod
    def _verify_telemetry(directory):
        for filename in ("generation.gpu.json", "generation.performance.json"):
            record = json.loads((directory / f"{filename}.manifest.json").read_text())
            if record["artifact_sha256"] != file_hash(directory / filename):
                raise ValueError("GPU telemetry download failed its content hash")

    def _download(self, api, name, request_sha):
        names = [
            "generation.jsonl",
            "generation.jsonl.manifest.json",
            "generation.gpu.json",
            "generation.gpu.json.manifest.json",
            "generation.performance.json",
            "generation.performance.json.manifest.json",
        ]
        # Never let an interrupted download become a seemingly complete cache.
        with tempfile.TemporaryDirectory(prefix=".download-", dir=self.directory) as temporary:
            temporary = Path(temporary)
            api.download_bucket_files(
                BUCKET_ID,
                [(f"automatic/{name}/{filename}", temporary / filename) for filename in names],
                raise_on_missing_files=True,
            )
            self._verify_output(temporary / "generation.jsonl", request_sha)
            self._verify_telemetry(temporary)
            # Move the verified output last, after its required evidence.
            for filename in names[1:] + names[:1]:
                target = self.directory / filename
                if target.exists() and target.read_bytes() != (temporary / filename).read_bytes():
                    raise FileExistsError(
                        "Refusing to overwrite a changed local generation artifact"
                    )
                if not target.exists():
                    (temporary / filename).replace(target)
