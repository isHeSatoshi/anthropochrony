"""Keep own-trajectory primary results separate from common-base robustness results."""

from anthropochrony.artifacts import digest


def primary_rows(rows):
    selected = []
    for row in rows:
        model = row["estimator"].lower().split("-s")[0]
        if model in {"m1", "m2"} and row["split"] in {"test", "test_ood"}:
            source = row.get("target_source")
            if source not in {"own", "base"}:
                raise ValueError("Primary M1/M2 evaluation must identify its target source")
            if source == "base":
                continue
        selected.append(row)
    return selected


def common_base_targets(rows, labels):
    index = {}
    for label in labels:
        if label.get("label_source") != "base":
            raise ValueError("Common-target comparisons require genuine base-trajectory labels")
        key = (label["task_id"], label["card_id"])
        if key in index:
            raise ValueError("Duplicate base target labels")
        index[key] = label
    output = {}
    for row in rows:
        label = index[(row["task_id"], row["card_id"])]
        if any(row[key] != label[key] for key in ("S", "P") if key in row):
            raise ValueError("Common targets must use the same stated speed card")
        key = (
            row["estimator"],
            row["task_id"],
            row["card_id"],
            row.get("frame", "self"),
            row.get("template_id"),
            row.get("phrasing_id"),
            row.get("seed"),
        )
        converted = row | {
            "T_med_s": label["T_med_s"],
            "actual_s": label["T_med_s"],
            "label_source": "base",
            "target_source": "base",
            "analysis_protocol": "common_base_targets",
        }
        for field in (
            "T_q10_s",
            "T_q90_s",
            "TOK_med",
            "ATT_med",
            "PFP",
            "overhead_med_s",
            "n_runs",
        ):
            if field in row and field in label:
                converted[field] = label[field]
        previous = output.get(key)
        if previous and digest(previous) != digest(converted):
            raise ValueError("Conflicting outputs for one common-target estimate")
        output[key] = converted
    return list(output.values())


def own_trajectory_targets(rows, labels, *, expected_source):
    """Retarget stored fine-tuned estimates (e.g. M3-S0) onto their own re-measured labels."""
    index = {}
    for label in labels:
        if label.get("label_source") != expected_source:
            raise ValueError(
                f"Own-target retargeting requires genuine {expected_source!r} trajectory labels"
            )
        key = (label["task_id"], label["card_id"])
        if key in index:
            raise ValueError("Duplicate own-trajectory target labels")
        index[key] = label
    output = {}
    for row in rows:
        label = index[(row["task_id"], row["card_id"])]
        if any(row[key] != label[key] for key in ("S", "P") if key in row):
            raise ValueError("Own targets must use the same stated speed card")
        key = (
            row["estimator"],
            row["task_id"],
            row["card_id"],
            row.get("frame", "self"),
            row.get("template_id"),
            row.get("phrasing_id"),
            row.get("seed"),
        )
        converted = row | {
            "T_med_s": label["T_med_s"],
            "actual_s": label["T_med_s"],
            "label_source": expected_source,
            "target_source": "own",
            "analysis_protocol": "own_trajectory_targets",
        }
        for field in (
            "T_q10_s",
            "T_q90_s",
            "TOK_med",
            "ATT_med",
            "PFP",
            "overhead_med_s",
            "n_runs",
        ):
            if field in row and field in label:
                converted[field] = label[field]
        previous = output.get(key)
        if previous and digest(previous) != digest(converted):
            raise ValueError("Conflicting outputs for one own-target estimate")
        output[key] = converted
    return list(output.values())
