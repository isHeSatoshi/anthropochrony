"""Locked, conservative spend reservations shared by Jobs and provider calls."""

import fcntl
import json
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path

from anthropochrony.artifacts import now, write_json

RATES = {"cpu-basic": Decimal("0.06"), "l40sx1": Decimal("1.80")}
CAP = Decimal("30")


class BudgetError(RuntimeError):
    pass


class Budget:
    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / "artifacts/budget.json"

    @contextmanager
    def ledger(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = (
                json.loads(self.path.read_text())
                if self.path.exists()
                else {"cap_usd": 30, "entries": []}
            )
            yield data
            write_json(self.path, data)

    def reserve(self, name, cost, *, approved_over_five=False):
        cost = Decimal(str(cost))
        if cost <= 0:
            raise BudgetError("Reserve a positive upper-bound cost")
        if cost > 5 and not approved_over_five:
            raise BudgetError("A single job over $5 requires explicit user approval")
        with self.ledger() as data:
            if any(e["name"] == name for e in data["entries"]):
                raise BudgetError(f"Reservation {name} already exists")
            total = sum(
                Decimal(str(e.get("actual_usd", e["reserved_usd"]))) for e in data["entries"]
            )
            if total + cost > CAP:
                raise BudgetError(f"Budget cap: ${total} + ${cost} would exceed $30")
            data["entries"].append(
                {
                    "name": name,
                    "reserved_usd": float(cost),
                    "state": "reserved",
                    "created_at": now(),
                }
            )
        return float(cost)

    def update(self, name, **fields):
        with self.ledger() as data:
            row = next(e for e in data["entries"] if e["name"] == name)
            if "actual_usd" in fields:
                if not 0 <= fields["actual_usd"] <= row["reserved_usd"]:
                    raise BudgetError(
                        "Actual charge exceeds reservation; stop and reconcile billing"
                    )
            row.update(fields)

    def total(self):
        with self.ledger() as data:
            return sum(e.get("actual_usd", e["reserved_usd"]) for e in data["entries"])


def job_cost(flavor, timeout_s, *, exposed_ports=False):
    if flavor not in RATES:
        raise BudgetError("Only approved cpu-basic and l40sx1 flavors are enabled")
    rate = RATES[flavor] + (Decimal("0.01") if exposed_ports else Decimal(0))
    # HF bills by the minute; reserve the final partial minute too.
    minutes = (Decimal(str(timeout_s)) / Decimal(60)).to_integral_value(rounding="ROUND_CEILING")
    return float(rate * minutes / Decimal(60))


def reconcile_terminal_jobs(root, api):
    """Replace reservations of finished jobs with a conservative billed-time bound.

    Whole minutes from start to finish plus one extra minute, at the flavor rate.
    It is not an invoice. Jobs without a start or finish time keep their reservation.
    """
    import math

    budget = Budget(root)
    with budget.ledger() as data:
        entries = [dict(e) for e in data["entries"]]
    terminal = {"COMPLETED", "ERROR", "FAILED", "CANCELED", "DELETED", "TIMEOUT"}
    changes = {}
    for entry in entries:
        job_id = entry.get("job_id")
        if not job_id or "actual_usd" in entry:
            continue
        job = api.inspect_job(job_id=job_id)
        flavor = str(getattr(job.flavor, "value", job.flavor))
        if (
            job.status.stage not in terminal
            or not (job.started_at and job.finished_at)
            or flavor not in RATES
        ):
            continue
        minutes = math.ceil((job.finished_at - job.started_at).total_seconds() / 60) + 1
        actual = min(float(Decimal(minutes) * RATES[flavor] / 60), entry["reserved_usd"])
        changes[entry["name"]] = round(actual, 4)
        budget.update(
            entry["name"],
            actual_usd=round(actual, 4),
            billing_basis=f"ceil(started_at..finished_at)+1 min={minutes} at ${RATES[flavor]}/h; "
            "conservative bound, not invoice",
        )
    return changes
