"""Bounded, ledgered HF Job with the pinned SDK's authenticated command channel."""

import secrets
import time

from huggingface_hub import HfApi, Sandbox, Volume, get_token, sync_bucket
from huggingface_hub import _sandbox as protocol

from anthropochrony.artifacts import now, write_json
from anthropochrony.budget import Budget, job_cost


def check_authorization(response):
    if response.status_code in {401, 403}:
        raise PermissionError("HF authorization failed; stop remote work and renew access")


def wait_for_allocation(api, job, *, timeout_s=3600):
    """Allow a bounded hardware queue separately from the server startup window."""
    deadline = time.monotonic() + timeout_s
    previous = None
    while time.monotonic() < deadline:
        current = api.inspect_job(job_id=job.id, namespace=job.owner.name)
        if current.status.stage == "RUNNING":
            return current
        if current.status.stage in protocol._TERMINAL_STAGES:
            raise RuntimeError(f"HF job ended before allocation: {current.status.stage}")
        if current.status.stage != previous:
            print(f"HF job {job.id}: {current.status.stage}; waiting for hardware", flush=True)
            previous = current.status.stage
        time.sleep(min(15, max(0, deadline - time.monotonic())))
    raise TimeoutError(f"HF hardware was not allocated within {timeout_s}s")


def launch_session(root, name, image, code, *, timeout_s=7200, queue_timeout_s=3600):
    token = get_token()
    if not token:
        raise PermissionError("HF login is missing; restore access before remote work")
    budget = Budget(root)
    cost = job_cost("l40sx1", timeout_s, exposed_ports=True)
    budget.reserve(name, cost)
    api = HfApi()
    prefix = f"code/{name}"
    try:
        sync_bucket(str(code), f"hf://buckets/isHeSatoshi/anthropochrony-runs/{prefix}", quiet=True)
    except Exception as error:
        budget.update(name, state="payload_upload_failed", actual_usd=0)
        if getattr(getattr(error, "response", None), "status_code", None) in {401, 403}:
            raise PermissionError(
                "HF authorization failed; stop remote work and renew access"
            ) from None
        raise
    nonce = secrets.token_hex(16)
    command, env, secret_values, volumes = protocol._bootstrap_job_spec(
        api,
        token,
        env={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        secrets=None,
        volumes=[
            Volume(type="bucket", source="isHeSatoshi/anthropochrony-runs", mount_path="/out"),
            Volume(
                type="bucket",
                source="isHeSatoshi/anthropochrony-runs",
                path=prefix,
                mount_path="/code",
                read_only=True,
            ),
        ],
        idle_timeout=1200,
        forward_hf_token=False,
        sandbox_token=protocol._derive_sandbox_token(token, nonce),
    )
    try:
        job = api.run_job(
            name=name,
            image=image,
            command=command,
            env=env,
            secrets=secret_values,
            flavor="l40sx1",
            timeout=timeout_s,
            attempts=1,
            expose=[protocol.SANDBOX_SERVER_PORT],
            volumes=volumes,
            labels={
                "project": "anthropochrony",
                protocol.SANDBOX_LABEL: "1",
                protocol.MODE_LABEL: protocol.MODE_DEDICATED,
                protocol.NONCE_LABEL: nonce,
            },
        )
    except Exception as error:
        budget.update(name, state="launch_failed_no_job")
        status = getattr(getattr(error, "response", None), "status_code", None)
        if status in {401, 403}:
            raise PermissionError(
                "HF authorization failed; stop remote work and renew access"
            ) from None
        raise
    budget.update(name, state="launched", job_id=job.id, flavor="l40sx1", timeout_s=timeout_s)
    write_json(
        root / f"artifacts/jobs/{name}.json",
        {
            "name": name,
            "job_id": job.id,
            "flavor": "l40sx1",
            "timeout_s": timeout_s,
            "upper_bound_usd": cost,
            "launched_at": now(),
            "hf_token_forwarded": False,
            "public_port": False,
            "sdk_server_sha256": protocol.SANDBOX_SERVER_SHA256,
            "queue_timeout_s": queue_timeout_s,
            "server_startup_timeout_s": 600,
            "exposed_port_hourly_usd": 0.01,
        },
    )
    with (root / "RUNLOG.md").open("a") as log:
        log.write(
            f"| {job.id} | {name} | l40sx1 | ≤{timeout_s / 60:.1f} | ≤{cost:.4f} | Launched, authenticated generation-only session |\n"
        )
    print(f"HF generation session {job.id}, timeout reservation ${cost:.2f}", flush=True)
    server = None
    try:
        job = wait_for_allocation(api, job, timeout_s=queue_timeout_s)
        server = protocol._SandboxServer.from_job(
            job=job,
            nonce=nonce,
            sandbox_token=protocol._derive_sandbox_token(token, nonce),
            api=api,
        )
        # SDK readiness polling otherwise suppresses non-200 health responses.
        server._client.event_hooks.setdefault("response", []).append(check_authorization)
        server.wait_ready(600)
    except BaseException as error:
        try:
            api.cancel_job(job_id=job.id, namespace=job.owner.name)
        finally:
            budget.update(name, state="startup_failed_cancel_attempted")
            if server is not None:
                server.close()
        if getattr(getattr(error, "response", None), "status_code", None) in {401, 403}:
            raise PermissionError(
                "HF authorization failed; stop remote work and renew access"
            ) from None
        raise
    return job, Sandbox(
        id=job.id, server=server, local_id=None, owns_sandbox=False, owns_server=True
    )
