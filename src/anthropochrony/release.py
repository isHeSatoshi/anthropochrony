"""License-aware release staging. No publication side effect is implicit."""

import copy
import json
import shutil
import subprocess
from pathlib import Path

from anthropochrony.artifacts import read_jsonl, write_json
from anthropochrony.gates import GateError

RESTRICTED_SOURCES = {"newfacade/LeetCodeDataset", "deepmind/code_contests", "BAAI/TACO"}


def redact_record(record, task_index):
    result = copy.deepcopy(record)
    task = task_index[result["task_id"]]
    if task["source"] not in RESTRICTED_SOURCES:
        return result
    # Release only IDs/provenance, not paraphrases, completions, prompt hashes of statements,
    # tests, reference code, human answers or embedded judge/provider transcripts.
    safe = {
        key: result[key]
        for key in (
            "task_id",
            "family",
            "source",
            "source_revision",
            "license",
            "split",
            "card_id",
            "variant",
            "estimator",
            "frame",
            "template_id",
            "label_version",
            "label_source",
            "label",
            "parsed",
            "T_med_s",
            "S",
            "P",
            "example_id",
        )
        if key in result
    }
    safe.update(
        task_id=task["task_id"],
        source=task["source"],
        source_revision=task["source_revision"],
        restricted_statement=True,
        rebuild={
            "task_id": task["task_id"],
            "source": task["source"],
            "revision": task["source_revision"],
        },
    )
    return safe


TASK_MARKER = "{{TASK_TURN}}"
TEXT_FIELDS = ("raw_text", "base_raw", "tuned_raw", "human_raw", "text")
TASK_SAFE_FIELDS = (
    "task_id",
    "family",
    "source",
    "source_revision",
    "license",
    "split",
    "harness",
    "near_dup_group",
    "difficulty",
    "ref_tokens",
    "prompt_tokens",
    "uncached_task_tokens",
    "time_limits",
    "gt",
    "flags",
)


def _copy(source, target):
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, target)


def _stream(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _dump(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            count += 1
    return count


def redact_row(record, task_index):
    """Restricted-source rows keep prompts only with the statement replaced by a marker.

    The marker is filled back by ``rebuild_messages.py`` from the pinned public dataset.
    Free-text model answers are dropped for restricted tasks because they may quote the
    statement; every parsed or numeric field stays.
    """
    if "messages" in record and (
        record.get("estimator") == "E2" or "cross_family_fewshot" in record
    ):
        # Few-shot prompts embed demonstration tasks, some of them from restricted sources.
        record = {**record, "messages": None, "messages_withheld": "few-shot prompt"}
    task = task_index.get(record.get("task_id"))
    if task is None or task["source"] not in RESTRICTED_SOURCES:
        return record
    result = copy.deepcopy(record)
    turn = task.get("task_turn") or ""
    for key in ("messages",):
        if result.get(key):
            messages = []
            for message in result[key]:
                content = message.get("content", "")
                if message.get("role") == "user" and turn and turn in content:
                    content = content.replace(turn, TASK_MARKER)
                elif message.get("role") == "user":
                    content = "[withheld: restricted statement]"
                elif message.get("role") == "assistant" and result.get("frame") == "human":
                    # Human-frame answers continue past the estimate and may quote the problem.
                    content = content.split("\n", 1)[0]
                messages.append({**message, "content": content})
            result[key] = messages
    for key in TEXT_FIELDS:
        if key in result:
            result[key] = None
    result["restricted_statement"] = True
    result["source"] = task["source"]
    result["source_revision"] = task["source_revision"]
    return result


def _task_rows(root):
    return read_jsonl(root / "data/tasks/tasks.jsonl")


def _stage_tasks(root, out, tasks):
    def rows():
        for task in tasks:
            if task["source"] in RESTRICTED_SOURCES:
                safe = {key: task[key] for key in TASK_SAFE_FIELDS if key in task}
                safe["restricted_statement"] = True
                safe["rebuild"] = {
                    "task_id": task["task_id"],
                    "source": task["source"],
                    "revision": task["source_revision"],
                }
                yield safe
            else:
                yield task

    _dump(out / "tasks.jsonl", rows())
    for name in ("tasks_dropped.jsonl", "proj8.jsonl"):
        _copy(root / "data/tasks" / name, out / name)
    for name in ("counts.json", "proj8-report.json"):
        _copy(root / "data/tasks" / name, out / name)


def _stage_traces(root, out, index):
    runs = [
        "base",
        "m1-s0",
        "m2-s0",
        "m3-s0",
        "showcase-base",
        "showcase-m1-s0",
    ]
    for run in runs:
        source = root / "data/measure" / run
        if not (source / "traces.jsonl").exists():
            continue
        restricted_texts = set()
        rows = []
        for row in _stream(source / "traces.jsonl"):
            task = index.get(row["task_id"])
            if task and task["source"] in RESTRICTED_SOURCES:
                restricted_texts.update(a["text_sha"] for a in row["attempts"] if "text_sha" in a)
            rows.append(row)
        _dump(out / run / "traces.jsonl", rows)
        for path in sorted(source.glob("outputs-r*.jsonl")):
            _dump(
                out / run / path.name,
                (r for r in _stream(path) if r["text_sha"] not in restricted_texts),
            )
    for name in ("base", "m1-s0", "m2-s0", "m3-s0"):
        _copy(root / "data/labels" / f"{name}.parquet", out / f"labels-{name}.parquet")
    for name in ("training-priors.json",):
        _copy(root / "data/labels" / name, out / name)
    for name in ("feedback-medians.json", "invariance.json", "showcase-summary.json"):
        if (root / "data/measure" / name).exists():
            _copy(root / "data/measure" / name, out / name)


def _stage_sft(root, out, index):
    for variant in ("m1", "m2", "m3"):
        base = root / "data/sft" / variant
        for split, name in (("train", "train"), ("validation", "validation")):
            source = base / split / "data.jsonl"
            if source.exists():
                _dump(
                    out / variant / f"{name}.jsonl",
                    (redact_row(r, index) for r in _stream(source)),
                )
        for name in ("dropped.jsonl", "counts.json"):
            if (base / name).exists():
                _copy(base / name, out / variant / name)
    _copy(root / "data/sft/matched-summary.json", out / "matched-summary.json")


def _stage_eval(root, out, index):
    _dump(
        out / "estimates.jsonl",
        (redact_row(r, index) for r in _stream(root / "data/eval/estimates.jsonl")),
    )
    for name in ("parser-audit.jsonl", "statistical.jsonl", "preservation-blockers.jsonl"):
        if (root / "data/eval" / name).exists():
            _copy(root / "data/eval" / name, out / name)
    for name in ("capability-m1-s0.jsonl", "e4-derived-estimates.jsonl"):
        _copy(root / "data/eval" / name, out / name)
    for name in ("common-base-estimates.jsonl",):
        _dump(
            out / name,
            (redact_row(r, index) for r in _stream(root / "data/eval" / name)),
        )
    for path in sorted((root / "data/x10").glob("*.json*")):
        if path.name.endswith(".manifest.json"):
            continue
        if path.name == "cpu-deployment.jsonl":
            _dump(out / "x10" / path.name, (redact_row(r, index) for r in _stream(path)))
        else:
            _copy(path, out / "x10" / path.name)
    results = root / "results"
    for path in sorted(results.glob("*.json*")):
        if path.name.endswith(".manifest.json") or path.name == "analysis_rows.jsonl":
            continue
        (out / "results").mkdir(parents=True, exist_ok=True)
        _copy(path, out / "results" / path.name)
    (out / "results").mkdir(parents=True, exist_ok=True)
    _dump(
        out / "results/analysis_rows.jsonl",
        (redact_row(r, index) for r in _stream(results / "analysis_rows.jsonl")),
    )
    figures = root / "paper/figures"
    for path in sorted(figures.glob("*.png")):
        (out / "figures").mkdir(parents=True, exist_ok=True)
        _copy(path, out / "figures" / path.name)


def _stage_logs(root, out):
    (out / "docs").mkdir(parents=True, exist_ok=True)
    _copy(root / "RUNLOG.md", out / "RUNLOG.md")
    for name in ("PREREGISTRATION.md", "PLAN.md", "PROMPTS.md", "EXECUTION_BLOCKER.md"):
        _copy(root / "docs" / name, out / "docs" / name)
    _copy(root / "artifacts/budget.json", out / "budget.json")
    for path in sorted((root / "results/gates").glob("*.json")):
        (out / "gates").mkdir(exist_ok=True)
        _copy(path, out / "gates" / path.name)
    for adapter in sorted((root / "adapters").iterdir()):
        for name in (
            "training-log.jsonl",
            "trainer_state.json",
            "training-summary.json",
            "identity.json",
            "hardware.json",
            "adapter_config.json",
        ):
            if (adapter / name).exists():
                (out / "training" / adapter.name).mkdir(parents=True, exist_ok=True)
                _copy(adapter / name, out / "training" / adapter.name / name)
    for path in sorted((root / "data/capability/jobs").glob("*/lmeval.json")):
        target = out / "capability" / path.parent.name
        target.mkdir(parents=True, exist_ok=True)
        _copy(path, target / "lmeval.json")
    jobs = root / "artifacts/jobs"
    if jobs.exists():
        shutil.copytree(jobs, out / "jobs", dirs_exist_ok=True)
    _copy(root / "data/calibration/reference.json", out / "speed-reference.json")


def _snippets(tasks):
    """Windows of 48 JSON-safe characters from lines of every restricted statement.

    Windows that also occur in any other task (template boilerplate) are skipped, because
    they cannot identify a statement.
    """
    from collections import Counter

    def windows(task):
        for line in (task.get("task_turn") or "").split("\n"):
            line = line.strip()
            if len(line) < 60:
                continue
            window = line[len(line) // 2 - 24 : len(line) // 2 + 24]
            if all(c.isascii() and c not in '"\\' and c.isprintable() for c in window):
                yield window

    owners = Counter()
    for task in tasks:
        owners.update(set(windows(task)))
    found = set()
    for task in tasks:
        if task["source"] not in RESTRICTED_SOURCES:
            continue
        chosen = [w for w in windows(task) if owners[w] == 1]
        found.update(chosen[:3])
    return sorted(found)


def leak_scan(root, directory, tasks):
    snippets = _snippets(tasks)
    pattern = Path(root) / "artifacts/release/.snippets.txt"
    pattern.write_text("\n".join(snippets) + "\n")
    try:
        result = subprocess.run(
            ["grep", "-rlF", "-f", str(pattern), str(directory)],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        pattern.unlink()
    hits = [line for line in result.stdout.splitlines() if line]
    return {"snippets": len(snippets), "files_with_restricted_text": hits}


def prepare_release(root, approved=False):
    root = Path(root)
    if approved:
        raise GateError(
            "Public release must be performed explicitly after reviewing the staged bundle and licenses"
        )
    tasks = _task_rows(root)
    index = {t["task_id"]: t for t in tasks}
    destination = root / "artifacts/release"
    datasets = destination / "datasets"
    if datasets.exists():
        shutil.rmtree(datasets)
    _stage_tasks(root, datasets / "anthropochrony-tasks", tasks)
    _stage_traces(root, datasets / "anthropochrony-traces", index)
    _stage_sft(root, datasets / "anthropochrony-sft", index)
    _stage_eval(root, datasets / "anthropochrony-eval", index)
    _stage_logs(root, datasets / "anthropochrony-logs")
    _copy(
        root / "scripts/rebuild_messages.py",
        datasets / "anthropochrony-tasks" / "rebuild_messages.py",
    )
    scan = leak_scan(root, datasets, tasks)
    write_json(
        destination / "release-status.json",
        {
            "private_staging_only": True,
            "publication_approved": False,
            "leak_scan": scan,
            "requires": [
                "G0-G9",
                "license confirmation U6",
                "public-release approval",
                "model/data/log cards",
                "account details removed from public docs",
            ],
        },
    )
    if scan["files_with_restricted_text"]:
        raise GateError(f"Restricted statement text found in {scan['files_with_restricted_text']}")
    return destination


def rebuild_task(record, pins, local_root=None):
    from anthropochrony.tasks.loaders import load_pool

    if not record.get("restricted_statement"):
        return record
    tasks, _ = load_pool(
        {"datasets": {record["source"]: pins["datasets"][record["source"]]}}, local_root
    )
    for task in tasks:
        if task["task_id"] == record["task_id"]:
            return task
    raise KeyError(record["task_id"])
