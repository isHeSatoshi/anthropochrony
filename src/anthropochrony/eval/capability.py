"""Paired first-attempt capability from genuine K=8 own trajectories."""

import numpy as np


def capability_pairs(tasks, base, fine_tuned, base_sha, fine_tuned_sha, k=8):
    by_task = {row["task_id"]: row for row in tasks if row["split"] == "test"}
    if not by_task:
        raise ValueError("No test tasks for the capability comparison")
    reference = None

    def first_attempts(traces, model_sha):
        nonlocal reference
        selected = [row for row in traces if row["task_id"] in by_task]
        if not selected:
            raise ValueError("Missing own test trajectories")
        if reference is None:
            reference = selected[0]["card"]
        results = {}
        for trace in selected:
            key = (trace["task_id"], trace["k"])
            if (
                key in results
                or trace["model_sha"] != model_sha
                or trace["card"] != reference
                or trace["card"]["id"] != "REF"
                or not trace.get("attempts")
            ):
                raise ValueError("Capability trajectories must be unique, pinned own REF runs")
            passed = trace["attempts"][0]["exec"]["status"] == "pass"
            if trace["first_pass"] != passed:
                raise ValueError("First-attempt outcome disagrees with its recorded test result")
            results[key] = passed
        if set(results) != {(task, index) for task in by_task for index in range(k)}:
            raise ValueError("Capability needs the complete K runs for every paired test task")
        return results

    before, after = first_attempts(base, base_sha), first_attempts(fine_tuned, fine_tuned_sha)
    return [
        {
            "task_id": task,
            "family": by_task[task]["family"],
            "split": "test",
            "estimator": "M1-s0",
            "seed": 0,
            "k": k,
            "base_model_sha": base_sha,
            "ft_model_sha": fine_tuned_sha,
            "base_pass1": float(np.mean([before[(task, index)] for index in range(k)])),
            "ft_pass1": float(np.mean([after[(task, index)] for index in range(k)])),
        }
        for task in sorted(by_task)
    ]
