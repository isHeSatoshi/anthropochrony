import random
import re
from collections import defaultdict

import numpy as np
from datasketch import MinHash, MinHashLSH

from anthropochrony.artifacts import digest


def shingles(text):
    words = re.findall(r"\w+", text.lower())
    if not words:
        return set()
    return {" ".join(words[i : i + 5]) for i in range(max(1, len(words) - 4))}


def duplicate_groups(tasks, threshold=0.7):
    """LSH proposes candidates; exact Jaccard verifies both prompt and code matches."""
    parent = list(range(len(tasks)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for field in ("task_turn", "reference_solution"):
        lsh = MinHashLSH(threshold=threshold, num_perm=128)
        seen = {}
        for i, task in enumerate(tasks):
            pieces = shingles(task.get(field, ""))
            if not pieces:
                continue
            mh = MinHash(num_perm=128, seed=20261109)
            for piece in sorted(pieces):
                mh.update(piece.encode())
            for other in lsh.query(mh):
                j, other_pieces = int(other), seen[int(other)]
                if len(pieces & other_pieces) / len(pieces | other_pieces) >= threshold:
                    parent[find(i)] = find(j)
            lsh.insert(str(i), mh)
            seen[i] = pieces
    groups = defaultdict(list)
    for i, task in enumerate(tasks):
        groups[find(i)].append(task["task_id"])
    return {task_id: digest(sorted(ids))[:20] for ids in groups.values() for task_id in ids}


def assign_splits(tasks, groups=None, seed=20261109):
    groups = groups or duplicate_groups(tasks)
    class_groups = {groups[t["task_id"]] for t in tasks if t["family"] == "CLASS"}
    kept, dropped, buckets = [], [], defaultdict(list)
    for task in sorted(tasks, key=lambda t: t["task_id"]):
        task = dict(task, near_dup_group=groups[task["task_id"]])
        if task["family"] != "CLASS" and task["near_dup_group"] in class_groups:
            dropped.append(dict(task_id=task["task_id"], reason="class_overlap"))
            continue
        if task["family"] == "CLASS":
            task["split"] = "test_ood"
        elif task["family"] == "PROJ-8":
            task["split"] = "showcase"
        else:
            buckets[task["family"]].append(task)
        kept.append(task)
    assignments = {}
    rng = random.Random(seed)
    for family, pool in sorted(buckets.items()):
        edges = np.quantile([t["ref_tokens"] for t in pool], [0.2, 0.4, 0.6, 0.8])
        strata = defaultdict(list)
        for task in pool:
            strata[int(np.searchsorted(edges, task["ref_tokens"]))].append(task)
        for _, stratum in sorted(strata.items()):
            group_ids = sorted({t["near_dup_group"] for t in stratum})
            rng.shuffle(group_ids)
            n_test, n_val = 0, 0
            for group_id in group_ids:
                size = sum(t["near_dup_group"] == group_id for t in stratum)
                if group_id not in assignments:
                    assignments[group_id] = (
                        "test"
                        if n_test < 0.2 * len(stratum)
                        else "val"
                        if n_val < (0.1 if family == "PROJ-SYN" else 0.08) * len(stratum)
                        else "train"
                    )
                if assignments[group_id] == "test":
                    n_test += size
                elif assignments[group_id] == "val":
                    n_val += size
    for task in kept:
        if task["family"] not in {"CLASS", "PROJ-8"}:
            task["split"] = assignments[task["near_dup_group"]]
    verify_splits(kept)
    return kept, dropped


def verify_splits(tasks):
    groups = defaultdict(set)
    ids = set()
    for task in tasks:
        if task["task_id"] in ids:
            raise ValueError("Duplicate task ID")
        ids.add(task["task_id"])
        groups[task["near_dup_group"]].add(task["split"])
        if task["family"] == "CLASS" and task["split"] != "test_ood":
            raise ValueError("CLASS must remain entirely held out")
        if task["family"] == "PROJ-8" and task["split"] != "showcase":
            raise ValueError("PROJ-8 must remain showcase-only")
    for group_splits in groups.values():
        non_raw = group_splits - {"raw_extra"}
        if len(non_raw) > 1 or (
            "raw_extra" in group_splits and group_splits & {"val", "test", "test_ood", "showcase"}
        ):
            raise ValueError("Near-duplicate group leaks across splits")
