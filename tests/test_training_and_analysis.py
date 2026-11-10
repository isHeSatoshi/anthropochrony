import copy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from anthropochrony.baselines.regressors import TunedInputRegressor
from anthropochrony.eval.hypotheses import confirmatory
from anthropochrony.eval.prompts import build_evaluation_prompts
from anthropochrony.gates import GateError, require_preregistration
from anthropochrony.train.sft import (
    TARGETS,
    train,
    verify_assistant_mask,
    verify_lora_targets,
    verify_training_collator,
)


def processed_trainer_fixture():
    example = {"input_ids": [2, 77, 88, 261001, 6], "assistant_masks": [0, 0, 1, 1, 1]}
    values = {
        "input_ids": [[2, 77, 88, 261001, 6, 0]],
        "labels": [[-100, -100, 88, 261001, 6, -100]],
        "attention_mask": [[1, 1, 1, 1, 1, 0]],
    }
    return SimpleNamespace(
        save_state=lambda: None,
        state=SimpleNamespace(log_history=[{"step": 1, "loss": 1.5}]),
        train_dataset=[example],
        data_collator=lambda examples: {
            key: SimpleNamespace(tolist=lambda value=value: value) for key, value in values.items()
        },
    ), values


class DictTokenizer:
    chat_template = {"default": "official-template"}

    def get_chat_template(self):
        return (
            self.chat_template.get("default")
            if isinstance(self.chat_template, dict)
            else self.chat_template
        )

    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        if not tokenize:
            return "unchanged-rendering"
        return {"input_ids": [0, 1, 2, 3], "assistant_masks": [0, 1, 1, 1]}

    def decode(self, tokens, **kwargs):
        assert tokens == [1, 2, 3]
        return "About 2 s.<|END_RESPONSE|><|END_OF_TURN_TOKEN|>"

    def save_pretrained(self, path):
        path.mkdir(parents=True)
        (path / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": self.chat_template})
        )


MESSAGES = [
    {"role": "user", "content": "Estimate your own time."},
    {"role": "assistant", "content": "About 2 s."},
]


def install_template_stub(monkeypatch):
    def get_template(tokenizer):
        assert tokenizer.chat_template == "official-template"
        return "official-training-template"

    monkeypatch.setitem(
        sys.modules,
        "trl.chat_template_utils",
        SimpleNamespace(get_training_chat_template=get_template),
    )


def test_dict_default_selected_before_trl_template_lookup(monkeypatch):
    install_template_stub(monkeypatch)
    tokenizer = DictTokenizer()
    result = verify_assistant_mask(tokenizer, MESSAGES)
    assert result == {"masked_tokens": 3, "rendering_identical": True}
    assert tokenizer.chat_template == "official-template"


def test_training_uses_normalized_tokenizer_and_one_step_dry_run(tmp_path, monkeypatch):
    install_template_stub(monkeypatch)
    tokenizer = DictTokenizer()
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: tokenizer)
    captured = {}
    parameter = SimpleNamespace(requires_grad=True, numel=lambda: 16)
    model = SimpleNamespace(
        named_modules=lambda: [
            (f"layers.0.{name}", SimpleNamespace(lora_A={})) for name in TARGETS
        ],
        named_parameters=lambda: [("layers.0.q_proj.lora_A.default.weight", parameter)],
    )

    def trained():
        captured["trained"] = True
        return {"steps": 1}

    fake_trainer = SimpleNamespace(
        train=trained,
        setup_model=lambda: captured.update(model_loaded=True),
        model=model,
        trainer=processed_trainer_fixture()[0],
    )

    def create(**kwargs):
        captured.update(kwargs)
        return fake_trainer

    monkeypatch.setitem(
        sys.modules, "aligntune.core.backend_factory", SimpleNamespace(create_sft_trainer=create)
    )
    data = tmp_path / "dataset/train"
    data.mkdir(parents=True)
    (data / "data.jsonl").write_text(json.dumps({"messages": MESSAGES}) + "\n")
    train(
        "base",
        data.parent,
        tmp_path / "output",
        {"epochs": 2, "learning_rate": 0.0001, "max_seq_length": 2048},
        0,
        dry_run=True,
    )
    assert captured["trained"]
    assert captured["model_loaded"]
    assert captured["max_steps"] == 1
    assert captured["assistant_only_loss"] is True
    assert captured["packing"] is False
    path = tmp_path / "output/normalized-tokenizer"
    assert captured["tokenizer_name_or_path"] == str(path)
    assert (
        json.loads((path / "tokenizer_config.json").read_text())["chat_template"]
        == "official-template"
    )
    summary = json.loads((tmp_path / "output/training-summary.json").read_text())
    assert summary["training"]["steps"] == 1
    assert summary["lora"]["targets"] == sorted(TARGETS)
    assert summary["actual_collator"]["assistant_only"]


@pytest.mark.parametrize("position", [0, 5])
def test_actual_training_collator_rejects_unmasked_user_or_padding(position):
    trainer, values = processed_trainer_fixture()
    values["labels"][0][position] = values["input_ids"][0][position]
    with pytest.raises(ValueError, match="assistant-only|padding"):
        verify_training_collator(trainer)


def test_lora_target_verification_rejects_tied_heads_and_unfrozen_base():
    parameter = SimpleNamespace(requires_grad=True, numel=lambda: 16)
    model = SimpleNamespace(
        named_modules=lambda: [
            (name, SimpleNamespace(lora_A={})) for name in TARGETS + ["lm_head"]
        ],
        named_parameters=lambda: [("model.embed_tokens.weight", parameter)],
    )
    with pytest.raises(ValueError, match="Unexpected LoRA targets"):
        verify_lora_targets(model)
    model.named_modules = lambda: [(name, SimpleNamespace(lora_A={})) for name in TARGETS]
    with pytest.raises(ValueError, match="Only the specified"):
        verify_lora_targets(model)


@pytest.mark.parametrize("stage", ["dry-run", "lr-sweep"])
def test_training_prerequisite_stages_are_reachable_without_selected_lr(
    tmp_path, monkeypatch, stage
):
    from anthropochrony import workflow

    captured, gates = {}, []
    monkeypatch.setattr(workflow, "require_gate", lambda root, gate: gates.append(gate))
    monkeypatch.setattr(
        workflow,
        "config",
        lambda root, name: (
            {"subject": {"repo": "fixture", "revision": "fixture"}}
            if name == "pins.yaml"
            else {
                "epochs": 2,
                "learning_rate": 0.0001,
                "learning_rate_selected_on_val": False,
                "max_seq_length": 2048,
            }
        ),
    )
    monkeypatch.setattr(workflow, "file_hash", lambda path: path.name)
    monkeypatch.setattr(workflow, "payload", lambda root, target, **kwargs: target)
    monkeypatch.setattr(
        workflow, "launch", lambda root, name, **kwargs: captured.update(name=name, **kwargs)
    )
    args = SimpleNamespace(
        stage=stage, variant="m1", seed=0, learning_rate=0.0002, dry_run=True, launch_name=None
    )
    workflow.train_job(tmp_path, args)
    config_path = tmp_path / f"artifacts/training-configs/{captured['name']}.json"
    actual = json.loads(config_path.read_text())
    assert gates == (["G4"] if stage == "dry-run" else ["G5"])
    assert captured["timeout_s"] == (1200 if stage == "dry-run" else 3600)
    if stage == "dry-run":
        assert "--dry-run" in captured["command"]
        assert "--push-repo" not in captured["command"]
    else:
        assert actual["epochs"] == 1
        assert actual["learning_rate"] == 0.0002
        assert "--dry-run" not in captured["command"]
        assert "--push-repo" in captured["command"]


def test_finetuned_prompts_use_training_system_and_model_seed_identifier():
    task = {"task_id": "fixture", "task_turn": "Implement f().", "family": "FUNC", "split": "val"}
    labels = [
        {
            "task_id": "fixture",
            "card_id": "REF",
            "S": 35,
            "P": 5000,
            "T_med_s": 2,
            "label_source": "m1-s0",
        }
    ]
    rows = build_evaluation_prompts([task], labels, split="val", fine_tuned="m1-s0")
    assert {r["estimator"] for r in rows} == {"M1-S0"}
    assert all(r["prompt_estimator"] == ("E1" if r["frame"] == "human" else "agent") for r in rows)
    assert len([r for r in rows if r["frame"] == "self"]) == 2
    assert all(r["label_source"] == "m1-s0" for r in rows)
    with pytest.raises(ValueError, match="base-prompt"):
        build_evaluation_prompts([task], labels, fine_tuned="m1-s0", include_fewshot=True)


def test_frozen_preregistration_allows_only_appended_dated_deviations(tmp_path, monkeypatch):
    frozen = b"# Frozen protocol\n\n## 7. Deviations\n\n- Original decisions.\n"
    monkeypatch.setattr(
        "anthropochrony.gates.subprocess.run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout=frozen),
    )
    path = tmp_path / "docs/PREREGISTRATION.md"
    path.parent.mkdir()
    path.write_bytes(frozen)
    require_preregistration(tmp_path, ["test"], True)
    path.write_bytes(frozen + b"\n- 2026-11-09, documented reason.\n")
    require_preregistration(tmp_path, ["test_ood"], True)
    for altered in (frozen.replace(b"Frozen", b"Changed"), frozen + b"\nSilent change.\n"):
        path.write_bytes(altered)
        with pytest.raises(GateError, match="Frozen"):
            require_preregistration(tmp_path, ["test"], True)
    require_preregistration(tmp_path, ["val"], True)


def observation(task_id, estimator, error, split="test", seed=None):
    return {
        "task_id": task_id,
        "family": "FUNC",
        "split": split,
        "card_id": "REF",
        "estimator": estimator,
        "frame": "generic" if estimator == "E3" else "self",
        "parsed": {"point_s": 10**error, "has_human_units": error >= 1},
        "T_med_s": 1,
        "seed": seed,
        "target_source": "own" if estimator.startswith("M") else "base",
    }


def test_generic_e3_selected_on_val_and_m1b_excluded_from_confirmatory_m1():
    rows = []
    for split in ("val", "test", "test_ood"):
        for i in range(24):
            for estimator, error in (("E0", 1.2), ("E1", 1.1), ("E2", 0.7), ("E3", 0.5)):
                rows.append(observation(str(i), estimator, error, split))
            if split != "val":
                for seed in range(3):
                    rows += [
                        observation(str(i), f"M1-s{seed}", 0.1, split, seed),
                        observation(str(i), f"M1b-s{seed}", 5, split, seed),
                        observation(str(i), f"M2-s{seed}", 0.4, split, seed),
                    ]
    result = confirmatory(rows, resamples=20)
    assert result["H1"]["passed"]
    for hypothesis in ("H2", "H3", "H5"):
        assert result[hypothesis]["passed"], result[hypothesis]
    assert result["H2"]["selected_base_prompt_on_val"] == "E3"
    assert result["H2"]["mean_seed_mdae"] == pytest.approx(0.1)
    assert result["H6a"]["status"] == "not_measured"


def test_no_parsed_points_does_not_crash_or_pass_h1():
    rows = [observation("a", name, 1) for name in ("E0", "E1")]
    for row in rows:
        row["parsed"]["point_s"] = None
    assert confirmatory(rows)["H1"]["passed"] is False


def test_tuned_embedding_regressor_ignores_test_labels():
    tasks, traces, labels, vectors = [], [], [], {}
    for i in range(24):
        split = "train" if i < 18 else "val" if i < 22 else "test"
        task_id = str(i)
        tasks.append(
            {
                "task_id": task_id,
                "split": split,
                "family": "FUNC",
                "task_turn": f"def f{i}(x): return x+{i}",
                "prompt_tokens": 100 + i,
                "ref_tokens": 20 + i,
            }
        )
        vectors[task_id] = np.asarray([i / 24, 1 - i / 24])
        if split == "train":
            traces.append(
                {
                    "task_id": task_id,
                    "total_o": 100 + i,
                    "n_attempts": 1,
                    "total_exec_s": 0.5,
                    "total_u": 100 + i,
                    "attempts": [],
                }
            )
        labels.append(
            {
                "task_id": task_id,
                "split": split,
                "card_id": "REF",
                "S": 10,
                "P": 1000,
                "T_med_s": (100 + i) / 10 + (100 + i) / 1000 + 0.5,
            }
        )
    first = TunedInputRegressor().fit(
        tasks, traces, embeddings=vectors, labels=labels, alphas=(1, 10), blends=(0, 1)
    )
    changed = copy.deepcopy(labels)
    for row in changed:
        if row["split"] == "test":
            row["T_med_s"] = 1e30
    second = TunedInputRegressor().fit(
        tasks, traces, embeddings=vectors, labels=changed, alphas=(1, 10), blends=(0, 1)
    )
    assert first.selection == second.selection
    assert len(first.selection["candidates"]) == 4
    assert np.isfinite(first.predict_values(tasks[-1])).all()
