import json
import sys
from types import SimpleNamespace

import pytest

from anthropochrony.artifacts import file_hash, read_jsonl, write_json, write_jsonl, write_parquet
from anthropochrony.cli import config, main
from anthropochrony.eval.prompts import build_evaluation_prompts
from anthropochrony.gates import GateError
from anthropochrony.train.selection import require_selected_learning_rate, select_learning_rate


def sweep_fixture(root, predictions=(20, 2)):
    subject = {"repo": "fixture/base", "revision": "base-sha"}
    write_json(root / "configs/pins.yaml", {"subject": subject})
    write_json(root / "results/gates/G5.json", {"passed": True})
    for name in ("m1", "m2", "m3"):
        write_json(
            root / f"configs/sft/{name}.yaml",
            {
                "variant": name,
                "learning_rate": 1e-4,
                "learning_rate_selected_on_val": False,
                "max_seq_length": 2048,
                "epochs": 2,
            },
        )
    train = root / "data/sft/m1/train/data.jsonl"
    write_jsonl(train, [{"fixture": i} for i in range(40)])
    tasks = [
        {"task_id": str(i), "family": "FUNC", "task_turn": "Implement f().", "split": "val"}
        for i in range(4)
    ]
    labels = [
        {
            "task_id": task["task_id"],
            "card_id": "REF",
            "S": 96,
            "P": 26000,
            "T_med_s": 2,
            "split": "val",
            "label_source": "base",
        }
        for task in tasks
    ]
    write_jsonl(root / "data/tasks/tasks.jsonl", tasks)
    write_parquet(root / "data/labels/base.parquet", labels)
    candidates = {}
    for rate, prediction in zip((1e-4, 2e-4), predictions, strict=True):
        directory = root / f"sweep/{rate}"
        training = directory / "training-summary.json"
        cfg = config(root, "sft/m1.yaml") | {
            "epochs": 1,
            "learning_rate": rate,
            "base": subject,
        }
        write_json(
            training,
            {
                "seed": 0,
                "dry_run": False,
                "config": cfg,
                "train_data_sha256": file_hash(train),
                "training": {"steps": 2},
            },
        )
        sha = f"checkpoint-{rate}"
        write_json(
            directory / "identity.json",
            {
                "private": True,
                "model_sha": sha,
                "base": subject,
                "training_summary_sha256": file_hash(training),
            },
        )
        write_json(directory / "hardware.json", {"cuda": True, "gpu": "NVIDIA L40S"})
        prompts = [
            row
            | {"seed": 0, "target_source": "base", "model_sha": sha, "raw_text": f"{prediction} s."}
            for row in build_evaluation_prompts(tasks, labels, split="val", fine_tuned="m1-s0")
            if row["frame"] == "self"
        ]
        estimates = directory / "estimates.jsonl"
        write_jsonl(estimates, prompts)
        candidates[rate] = {"estimates": estimates, "training_summary": training}
    return candidates


def test_actual_cli_freezes_val_only_learning_rate_for_every_variant(tmp_path):
    candidates = sweep_fixture(tmp_path)
    args = ["--root", str(tmp_path), "select-lr"]
    for label, rate in (("low", 1e-4), ("high", 2e-4)):
        args += [
            f"--{label}-estimates",
            str(candidates[rate]["estimates"]),
            f"--{label}-training-summary",
            str(candidates[rate]["training_summary"]),
        ]
    main(args)
    result = json.loads((tmp_path / "results/lr-selection.json").read_text())
    assert result["learning_rate"] == 2e-4
    assert result["scores_val_mdae_log10"] == {"0.0001": 1, "0.0002": 0}
    for name in ("m1", "m2", "m3"):
        cfg = config(tmp_path, f"sft/{name}.yaml")
        assert cfg["learning_rate"] == 2e-4
        assert cfg["epochs"] == 2
        require_selected_learning_rate(tmp_path, cfg)
    assert select_learning_rate(tmp_path, candidates) == result
    changed = config(tmp_path, "sft/m2.yaml") | {"learning_rate": 1e-4}
    write_json(tmp_path / "configs/sft/m2.yaml", changed)
    with pytest.raises(GateError, match="same frozen"):
        require_selected_learning_rate(tmp_path, config(tmp_path, "sft/m1.yaml"))


def test_preview_then_real_training_launch_reuses_the_unchanged_payload(tmp_path, monkeypatch):
    from anthropochrony import jobs, workflow
    from anthropochrony.budget import Budget

    sweep_fixture(tmp_path)
    write_json(tmp_path / "results/gates/G4.json", {"passed": True})
    for name in ("src/anthropochrony/train/sft.py", "jobs/train_sft.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Fixture source only, never executed.\n")
    calls = []

    def run(command, **kwargs):
        if command[0] == "git":
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="a" * 24 + "\n", stderr="")

    monkeypatch.setattr(jobs.subprocess, "run", run)
    args = SimpleNamespace(
        stage="dry-run",
        variant="m1",
        seed=0,
        learning_rate=None,
        dry_run=True,
        launch_name=None,
    )
    workflow.train_job(tmp_path, args)
    assert not calls and not (tmp_path / "artifacts/budget.json").exists()
    args.dry_run = False
    workflow.train_job(tmp_path, args)
    assert len(calls) == 1 and "--secrets" not in calls[0]
    assert Budget(tmp_path).total() == 0.6


@pytest.mark.parametrize(
    "bad", ["test", "test_ood", "missing", "duplicate", "checkpoint", "target"]
)
def test_lr_selection_rejects_contamination_or_cherry_picked_validation(tmp_path, bad):
    candidates = sweep_fixture(tmp_path)
    path = candidates[1e-4]["estimates"]
    rows = read_jsonl(path)
    if bad in {"test", "test_ood"}:
        rows.append(rows[0] | {"split": bad})
    elif bad == "missing":
        rows.pop()
    elif bad == "duplicate":
        rows.append(rows[0])
    elif bad == "checkpoint":
        rows[0]["model_sha"] = "wrong-checkpoint"
    else:
        rows[0]["T_med_s"] = 20
    write_jsonl(path, rows)
    with pytest.raises(GateError):
        select_learning_rate(tmp_path, candidates)
    assert not (tmp_path / "results/lr-selection.json").exists()
    assert config(tmp_path, "sft/m1.yaml")["learning_rate_selected_on_val"] is False


def test_lr_ties_choose_lower_rate_and_nonfinite_candidates_do_not_select(tmp_path):
    candidates = sweep_fixture(tmp_path, predictions=(2, 2))
    assert select_learning_rate(tmp_path, candidates)["learning_rate"] == 1e-4
    other = tmp_path / "unparseable"
    candidates = sweep_fixture(other, predictions=("unknown", "unknown"))
    with pytest.raises(GateError, match="Neither"):
        select_learning_rate(other, candidates)


def test_authorization_guard_blocks_requests_after_swallowed_denial(monkeypatch):
    from anthropochrony import remote

    for status in (401, 403):
        client = SimpleNamespace(event_hooks={"request": [], "response": []})
        monkeypatch.setattr(remote, "get_session", lambda client=client: client)
        with pytest.raises(PermissionError, match="renew access"):
            with remote.authorization_guard():
                # Simulate a third-party callback swallowing its first failure.
                with pytest.raises(PermissionError):
                    client.event_hooks["response"][-1](SimpleNamespace(status_code=status))
                with pytest.raises(PermissionError):
                    client.event_hooks["request"][-1](None)
        assert client.event_hooks == {"request": [], "response": []}


@pytest.mark.parametrize("private", [True, False])
def test_adapter_upload_is_private_only_and_records_final_commit(tmp_path, monkeypatch, private):
    from anthropochrony import remote
    from anthropochrony.train import publish

    calls = []
    monkeypatch.setattr(remote, "get_session", lambda: SimpleNamespace(event_hooks={}))
    subject = {"repo": "fixture/base", "revision": "pinned-base-sha"}
    write_json(tmp_path / "training-summary.json", {"seed": 0})

    class FakePeft:
        def __init__(self):
            self.peft_config = {"default": SimpleNamespace()}

        def save_pretrained(self, path, **kwargs):
            (path / "adapter_model.safetensors").write_bytes(b"FAKE TEST ADAPTER")
            write_json(path / "adapter_config.json", vars(self.peft_config["default"]))

    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=FakePeft))
    api = SimpleNamespace(
        create_repo=lambda *a, **kw: calls.append(("create", kw)),
        model_info=lambda *a: SimpleNamespace(private=private, sha="a" * 40),
        upload_folder=lambda **kw: calls.append(("upload", kw)),
    )
    monkeypatch.setattr(publish, "HfApi", lambda: api)
    trainer = SimpleNamespace(
        model=FakePeft(),
        push_to_hub=lambda *a, **kw: calls.append(("push", kw)),
    )
    if not private:
        with pytest.raises(PermissionError, match="non-private"):
            publish.publish_adapter(trainer, tmp_path, "fixture/adapter", subject)
        assert [event for event, _ in calls] == ["create"]
        return
    identity = publish.publish_adapter(trainer, tmp_path, "fixture/adapter", subject)
    assert identity["model_sha"] == "a" * 40
    assert identity["private"] is True and identity["base"] == subject
    assert identity["training_summary_sha256"] == file_hash(tmp_path / "training-summary.json")
    assert calls[0][1]["private"] is True and calls[1] == ("push", {"private": True})
    assert "model.safetensors" not in calls[2][1]["allow_patterns"]
    assert "training-log.jsonl" in calls[2][1]["allow_patterns"]
    assert json.loads((tmp_path / "adapter_config.json").read_text()) == {
        "base_model_name_or_path": "fixture/base",
        "revision": "pinned-base-sha",
    }
