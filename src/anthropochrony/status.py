import json
from pathlib import Path


def status(root):
    root = Path(root)
    gates = {}
    for name in ("G0-security", "G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10"):
        path = root / "results/gates" / f"{name}.json"
        gates[name] = (
            json.loads(path.read_text()) if path.exists() else {"passed": False, "state": "not_run"}
        )
    ledger = root / "artifacts/budget.json"
    budget = json.loads(ledger.read_text()) if ledger.exists() else {"cap_usd": 30, "entries": []}
    return {
        "project": "Anthropochrony",
        "gates": gates,
        "budget": budget,
        "external_release_approved": False,
        "trained_model_available": any((root / "adapters").glob("*/adapter_config.json")),
    }
