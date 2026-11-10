import random
from collections import Counter
from pathlib import Path

from anthropochrony.artifacts import digest, write_json, write_jsonl
from anthropochrony.curate.pipeline import curate
from anthropochrony.gen.engine import render
from anthropochrony.labels import canonical_answer
from anthropochrony.prompts import estimate_messages
from anthropochrony.timemodel import SpeedCard, completion_time


def teacher_label(task, card, quantities, feedback_tokens):
    attempts = quantities["attempts_median"]
    u = task["uncached_task_tokens"] + (attempts - 1) * feedback_tokens
    test = attempts * quantities["test_seconds"]
    point = completion_time(u, quantities["tokens_median"], test, card, calls=attempts)
    low = completion_time(u, quantities["tokens_p10"], test, card, calls=attempts)
    high = completion_time(u, quantities["tokens_p90"], test, card, calls=attempts)
    return {
        "T_med_s": point,
        "T_q10_s": min(low, point / 1.25),
        "T_q90_s": max(high, point * 1.25),
        "TOK_med": quantities["tokens_median"],
        "ATT_med": attempts,
        "PFP": quantities["p_first_pass"],
        "S": card.S,
        "P": card.P,
    }


def build_sft(
    tasks,
    labels,
    human_targets,
    tokenizer,
    *,
    variant="m1",
    teachers=None,
    feedback_by_family=None,
    max_length=2048,
):
    if variant not in {"m1", "m2", "m3", "m4", "m5", "m6", "m7"}:
        raise ValueError("Unsupported SFT variant")
    label_map = {}
    for label in labels:
        if label["card_id"].startswith("train-") or variant == "m5" and label["card_id"] == "REF":
            label_map.setdefault(label["task_id"], []).append(label)
    rows, drops = [], []
    for task in sorted(tasks, key=lambda t: t["task_id"]):
        if task["split"] not in (
            {"train", "val", "raw_extra"} if variant == "m3" else {"train", "val"}
        ):
            continue
        targets = label_map.get(task["task_id"], [])
        if len(targets) < (1 if variant == "m5" else 6):
            raise ValueError(f"Missing six speed labels for {task['task_id']}")
        rng = random.Random(int(digest([task["task_id"], 20261109])[:16], 16))
        for index, original in enumerate(sorted(targets, key=lambda label: label["card_id"])[:6]):
            card = SpeedCard(original["card_id"], original["S"], original["P"])
            template = f"q{rng.randint(1, 20):02d}"
            phrasing = f"sc{rng.randint(1, 6):02d}"
            messages = estimate_messages(
                task,
                card,
                template,
                phrasing,
                estimator="M5" if variant == "m5" else "agent",
            )
            label = original
            if variant == "m2":
                if not teachers or task["task_id"] not in teachers:
                    raise ValueError(
                        "M2 requires a real, cached teacher prediction for every matched task"
                    )
                label = teacher_label(
                    task,
                    card,
                    teachers[task["task_id"]],
                    (feedback_by_family or {})[task["family"]],
                )
            messages.append({"role": "assistant", "content": canonical_answer(label)})
            row = {
                "example_id": digest([task["task_id"], "self", index]),
                "task_id": task["task_id"],
                "family": task["family"],
                "variant": variant,
                "frame": "self",
                "template_id": template,
                "phrasing_id": phrasing,
                "card_id": card.id,
                "messages": messages,
                "label": label,
                "label_source": "teacher"
                if variant == "m2"
                else "raw"
                if variant == "m3"
                else "measured",
                "label_version": "v2-latency",
                "split": "train" if task["split"] == "raw_extra" else task["split"],
                "source": task["source"],
            }
            if len(render(tokenizer, messages)) > max_length:
                drops.append(
                    {
                        "example_id": row["example_id"],
                        "task_id": task["task_id"],
                        "family": task["family"],
                        "reason": "sequence_length",
                    }
                )
            else:
                rows.append(row)
        if variant != "m4":
            target = human_targets.get(task["task_id"])
            if not target or target["finish_reason"] == "length":
                raise ValueError(
                    f"Missing untruncated human preservation target for {task['task_id']}"
                )
            row = {
                "example_id": digest([task["task_id"], "human"]),
                "task_id": task["task_id"],
                "family": task["family"],
                "variant": variant,
                "frame": "human",
                "template_id": target["template_id"],
                "phrasing_id": target["phrasing_id"],
                "card_id": "REF",
                "messages": target["messages"]
                + [{"role": "assistant", "content": target["raw_text"]}],
                "label": {},
                "label_source": "base-self-distillation",
                "label_version": "v1",
                "split": "train" if task["split"] == "raw_extra" else task["split"],
                "source": task["source"],
            }
            if len(render(tokenizer, row["messages"])) <= max_length:
                rows.append(row)
            else:
                drops.append(
                    {
                        "example_id": row["example_id"],
                        "task_id": task["task_id"],
                        "family": task["family"],
                        "reason": "sequence_length",
                    }
                )
    return rows, drops


def export_sft(rows, directory):
    directory = Path(directory)
    exported = []
    for split in ("train", "val"):
        subset = [row for row in rows if row["split"] == split]
        passed, _ = curate(subset, directory / "curation" / split, kind="sft")
        exported.extend(passed)
        trainer_split = "validation" if split == "val" else "train"
        write_jsonl(directory / trainer_split / "data.jsonl", passed)
    write_json(directory / "counts.json", dict(Counter(row["split"] for row in exported)))
    return exported


def check_matched_variants(first, second):
    a, b = ({r["example_id"]: r for r in rows} for rows in (first, second))
    if len(a) != len(first) or len(b) != len(second):
        raise ValueError("Duplicate SFT example IDs")
    if set(a) != set(b):
        raise ValueError("M1 and M2 must have identical example IDs")
    for key in a:
        if a[key]["messages"][:-1] != b[key]["messages"][:-1]:
            raise ValueError("A matched M1/M2 prompt changed")
        if a[key]["frame"] == "human" and a[key]["messages"] != b[key]["messages"]:
            raise ValueError("Human preservation targets must match")
    return True


def build_matched_sft(
    tasks, labels, human_targets, tokenizer, *, teachers, feedback_by_family, max_length=2048
):
    """Use one length limit and a shared retained-ID set for the P0 variants."""
    if max_length not in {2048, 3072}:
        raise ValueError("The preregistered SFT limits are 2048 and 3072")

    def build(limit):
        variants, exclusions = {}, {}
        for name in ("m1", "m2", "m3"):
            variants[name], exclusions[name] = build_sft(
                tasks,
                labels,
                human_targets,
                tokenizer,
                variant=name,
                teachers=teachers,
                feedback_by_family=feedback_by_family,
                max_length=limit,
            )
        common_task_ids = {t["task_id"] for t in tasks if t["split"] in {"train", "val"}}
        shared = {row["example_id"] for row in variants["m1"]} & {
            row["example_id"] for row in variants["m2"]
        }
        for name in variants:
            kept = []
            for row in variants[name]:
                if row["task_id"] in common_task_ids and row["example_id"] not in shared:
                    exclusions[name].append(
                        {
                            "example_id": row["example_id"],
                            "task_id": row["task_id"],
                            "family": row["family"],
                            "reason": "matched_sequence_length",
                        }
                    )
                else:
                    kept.append(row)
            variants[name] = kept
        fractions = {}
        for name, rows in variants.items():
            kept = Counter(row["family"] for row in rows)
            dropped = Counter(row["family"] for row in exclusions[name])
            fractions[name] = {
                family: dropped[family] / (kept[family] + dropped[family])
                for family in kept.keys() | dropped.keys()
            }
        check_matched_variants(variants["m1"], variants["m2"])
        return variants, exclusions, fractions

    variants, exclusions, initial_fractions = build(max_length)
    expanded = max_length == 2048 and any(
        fraction > 0.02 for families in initial_fractions.values() for fraction in families.values()
    )
    if expanded:
        max_length = 3072
        variants, exclusions, fractions = build(max_length)
    else:
        fractions = initial_fractions
    return (
        variants,
        exclusions,
        {
            "max_seq_length": max_length,
            "expanded_to_3072": expanded,
            "initial_drop_fractions": initial_fractions,
            "drop_fractions": fractions,
            "expected_example_ids": {
                name: sorted(row["example_id"] for row in rows) for name, rows in variants.items()
            },
        },
    )
