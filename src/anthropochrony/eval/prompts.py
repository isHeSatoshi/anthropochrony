from anthropochrony.artifacts import digest
from anthropochrony.labels import canonical_answer
from anthropochrony.prompts import estimate_messages
from anthropochrony.timemodel import SpeedCard

# Zero-shot base prompts answered with code about half the time and parsed at under 20% (a dated
# deviation in PREREGISTRATION.md section 7). The base model's reply starts after this text, and
# the choice of number and unit stays its own. Fine-tuned models are evaluated without it.
BASE_PREFILL = "Time estimate:"


def build_evaluation_prompts(
    tasks, labels, *, split="test", include_fewshot=False, fine_tuned=None
):
    if fine_tuned and include_fewshot:
        raise ValueError("E2 demonstrations belong to the base-prompt baseline only")
    if fine_tuned and split == "raw_extra":
        raise ValueError("Raw-extra tasks are training-only, never fine-tuned evaluation")
    output = []
    by_task = {}
    for label in labels:
        by_task.setdefault(label["task_id"], []).append(label)
    for task in sorted(tasks, key=lambda row: row["task_id"]):
        if task["split"] != split:
            continue
        assignment = int(digest(task["task_id"])[:8], 16)
        for label in by_task.get(task["task_id"], []):
            if label["card_id"].startswith("train-"):
                continue
            if split in {"train", "raw_extra"} and label["card_id"] != "REF":
                continue
            if split == "val" and label["card_id"] not in {"REF", "seen-cpu", "seen-edge"}:
                continue
            card = SpeedCard(label["card_id"], label["S"], label["P"])
            if card.id == "REF":
                specifications = [
                    ("self", f"q{21 + assignment % 8:02d}", "E0"),
                    ("self", f"q{21 + assignment % 8:02d}", "E1"),
                    ("self", f"q{21 + (assignment + 3) % 8:02d}", "E1"),
                    ("human", f"h{9 + assignment % 4:02d}", "E1"),
                    ("generic", f"g{1 + assignment % 4:02d}", "E3"),
                    ("plan", f"p{1 + assignment % 4:02d}", "E1"),
                ]
                if split == "test":
                    specifications += [("wording", f"w{i:02d}", "E1") for i in range(1, 9)]
            else:
                specifications = [("self", f"q{21 + assignment % 8:02d}", "E1")]
            if split in {"train", "raw_extra"}:
                specifications = [("human", f"h{1 + assignment % 8:02d}", "E1")]
            for frame, template, estimator in specifications:
                if fine_tuned and estimator == "E0":
                    continue
                if split in {"train", "val", "raw_extra"}:
                    if frame == "self":
                        template = f"q{1 + (assignment + int(template[1:]) - 21) % 20:02d}"
                    elif frame == "human":
                        template = f"h{1 + assignment % 8:02d}"
                    elif frame == "plan":
                        template = f"p{5 + assignment % 4:02d}"
                phrasing = (
                    f"sc{1 + assignment % 6:02d}"
                    if split in {"train", "val", "raw_extra"}
                    else f"sc{7 + assignment % 2:02d}"
                )
                prompt_estimator = "agent" if fine_tuned and frame != "human" else estimator
                prefill = BASE_PREFILL if not fine_tuned and frame != "plan" else ""
                messages = estimate_messages(
                    task,
                    card,
                    template,
                    phrasing,
                    estimator=prompt_estimator,
                )
                output.append(
                    {
                        "est_id": digest(
                            [task["task_id"], card.id, frame, template, fine_tuned or estimator]
                            + ([prefill] if prefill else [])
                        ),
                        "task_id": task["task_id"],
                        "family": task["family"],
                        "split": split,
                        "frame": frame,
                        "template_id": template,
                        "phrasing_id": phrasing,
                        "prompt_estimator": prompt_estimator,
                        "prefill": prefill,
                        "estimator": fine_tuned.upper() if fine_tuned else estimator,
                        "label_source": label.get("label_source", "base"),
                        "card_id": card.id,
                        "messages": messages,
                        "prompt_sha": digest(messages),
                        "T_med_s": label["T_med_s"],
                        "S": card.S,
                        "P": card.P,
                        "prompt_tokens": task.get("prompt_tokens", 0),
                        "max_tokens": 1024
                        if frame == "plan"
                        or frame == "human"
                        and split in {"train", "val", "raw_extra"}
                        else 512,
                    }
                )
    if include_fewshot:
        extra = []
        task_map = {task["task_id"]: task for task in tasks}
        # Index once; each target then ranks only its legal same-card training labels.
        pools = {}
        for label in labels:
            source = task_map.get(label["task_id"])
            if source and source["split"] == "train":
                pools.setdefault((label["card_id"], source["family"]), []).append((source, label))
        for target in output:
            if target["estimator"] != "E1" or target["frame"] != "self":
                continue
            card = SpeedCard(target["card_id"], target["S"], target["P"])
            cross_family = target["family"] == "CLASS"
            legal = [
                pair
                for (card_id, family), pairs in pools.items()
                if card_id == target["card_id"] and (cross_family or family == target["family"])
                for pair in pairs
            ]
            ranked = sorted(
                (pair for pair in legal if pair[0]["task_id"] != target["task_id"]),
                key=lambda pair: (
                    abs(pair[0]["prompt_tokens"] - target["prompt_tokens"]),
                    pair[0]["task_id"],
                ),
            )
            if len(ranked) < 5:
                raise ValueError(
                    f"E2 has fewer than five legal same-family examples for {target['family']}"
                )
            examples = [
                {
                    "task_id": source["task_id"],
                    "family": source["family"],
                    "split": "train",
                    "card_id": card.id,
                    "prompt_tokens": source["prompt_tokens"],
                    "messages": estimate_messages(source, card, "q01", "sc01", estimator="E1"),
                    "answer": canonical_answer(label),
                }
                for source, label in ranked[:5]
            ]
            messages = few_shot(target, examples, cross_family=cross_family)
            extra.append(
                target
                | {
                    "est_id": digest([target["est_id"], "E2"]),
                    "estimator": "E2",
                    "prompt_estimator": "E2",
                    "prefill": "",
                    "messages": messages,
                    "prompt_sha": digest(messages),
                    "cross_family_fewshot": cross_family,
                }
            )
        output += extra
    if len({row["est_id"] for row in output}) != len(output):
        raise ValueError("Duplicate evaluation prompt IDs")
    return output


def few_shot(target, training_examples, count=5, *, cross_family=False):
    candidates = [
        row
        for row in training_examples
        if row["split"] == "train"
        and (cross_family or row["family"] == target["family"])
        and row["card_id"] == target["card_id"]
        and row["task_id"] != target["task_id"]
    ]
    candidates.sort(
        key=lambda row: (abs(row["prompt_tokens"] - target["prompt_tokens"]), row["task_id"])
    )
    if len(candidates) < count:
        raise ValueError("Five legal same-card training examples required for E2")
    selected = candidates[:count]
    messages = target["messages"][:1]
    for row in selected:
        messages += [row["messages"][1], {"role": "assistant", "content": row["answer"]}]
    return messages + target["messages"][1:]
