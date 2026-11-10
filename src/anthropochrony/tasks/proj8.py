"""PROJ-8 showcase tasks: eight small projects drafted by the teacher and verified locally.

They are never part of train, val or test. Each task has a spec, a reference solution and a
hidden unittest file, and passes the same ground-truth checks as the benchmark tasks plus a stub
check on every individual test.
"""

import ast
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from anthropochrony.artifacts import digest, now, write_json, write_jsonl
from anthropochrony.tasks.pipeline import vacuous_solution, verify_reference
from anthropochrony.tasks.schema import Task

CATEGORIES = {
    "PROJ8-todo-cli": "A command-line to-do list: a `main(argv, store_path)` entry point that parses add, done, remove and list sub-commands and keeps tasks in a JSON file.",
    "PROJ8-notes-api": "A notes REST API in Flask: `create_app()` returning a Flask app with create, read, update, delete and search endpoints, backed by an in-memory store. Tests use `app.test_client()`.",
    "PROJ8-snake": "Snake game logic with no display: a `Game` class with a grid, movement, food placement from an injected random source, growth, collisions and score.",
    "PROJ8-lru-cache": "An LRU cache class with a capacity, optional per-entry time-to-live driven by an injected clock, hit and miss statistics, and a decorator for functions.",
    "PROJ8-csv-report": "A CSV report builder: read sales rows from a CSV string, group and total them by region and month, and render a fixed-width text report.",
    "PROJ8-markdown": "A Markdown to HTML converter for headings, emphasis, inline code, fenced code blocks, links, and ordered and unordered lists, with HTML escaping.",
    "PROJ8-url-shortener": "A URL shortener service class: deterministic short codes from a counter, custom aliases, validation, expiry with an injected clock, and click counts.",
    "PROJ8-bank-ledger": "A double-entry bank ledger: accounts, transfers, overdraft rules, an append-only transaction log, balances at a point in time, and reconciliation.",
}

PROMPT = """Create one self-contained Python programming task of the "small project" kind.
Project: {description}
Target reference size: about {lines} lines.
Constraints: Python 3.10; the standard library only{flask}; everything in one module named solution.py; deterministic; no network, no sleeping, no reading files outside a tempfile directory it creates; all tests must finish in under 5 seconds. Tests must only use the public names the spec defines.
Return exactly three sections:
<spec>A clear task description for a developer (what to build, public functions/classes with exact signatures, behaviors, edge cases). Do not include the implementation.</spec>
<reference>The complete reference solution.py.</reference>
<tests>A unittest file (one or more unittest.TestCase classes, 10 to 15 test methods) that checks the spec'd behavior. The solution code is placed in the same namespace right before the tests, so do not import solution; use its names directly and import any standard-library modules the tests need.</tests>"""


def fence(text):
    text = text.strip()
    match = re.fullmatch(r"```(?:python)?\n(.*?)\n```", text, re.S)
    return match.group(1) if match else text


def parse_draft(answer):
    sections = {}
    for name in ("spec", "reference", "tests"):
        match = re.search(rf"<{name}>(.*?)</{name}>", answer, re.S)
        if not match:
            raise ValueError(f"missing <{name}> section")
        sections[name] = fence(match.group(1))
    return sections


def draft(provider, task_id, description, *, lines=150, variant=0):
    flask = " and Flask 3" if "Flask" in description else ""
    prompt = PROMPT.format(description=description, lines=lines, flask=flask)
    if variant:
        prompt += (
            f"\nDraft variant {variant}: choose different concrete details from other variants."
        )
    return parse_draft(provider.call(prompt, max_tokens=7000, temperature=0.2))


def _is_test(node):
    return isinstance(node, ast.FunctionDef) and node.name.startswith("test")


def _has_tests(node):
    return isinstance(node, ast.ClassDef) and any(_is_test(item) for item in node.body)


def test_ids(test_code):
    return [
        (node.name, item.name)
        for node in ast.parse(test_code).body
        if _has_tests(node)
        for item in node.body
        if _is_test(item)
    ]


def select_tests(test_code, keep):
    """The test module restricted to `keep` (class, method) pairs; imports and helpers stay."""
    tree = ast.parse(test_code)
    body = []
    for node in tree.body:
        if not _has_tests(node):
            body.append(node)
            continue
        node.body = [
            item for item in node.body if not _is_test(item) or (node.name, item.name) in keep
        ]
        if any(_is_test(item) for item in node.body):
            body.append(node)
    tree.body = body
    return ast.unparse(ast.fix_missing_locations(tree))


def single_test_sources(test_code):
    return [select_tests(test_code, {test_id}) for test_id in test_ids(test_code)]


def prune_to_reference(task, sandbox, workers=4):
    """Keep the tests the reference passes; the dropped ones disagreed with the reference."""
    ids = test_ids(task["test_code"])

    def passes(test_id):
        source = select_tests(task["test_code"], {test_id})
        return (
            sandbox.run(task["reference_solution"], dict(task, test_code=source))["status"]
            == "pass"
        )

    with ThreadPoolExecutor(workers) as pool:
        flags = list(pool.map(passes, ids))
    kept = {test_id for test_id, ok in zip(ids, flags, strict=True) if ok}
    dropped = [f"{c}.{m}" for (c, m), ok in zip(ids, flags, strict=True) if not ok]
    return select_tests(task["test_code"], kept), len(ids), dropped


def stub_failure_share(task, sandbox, workers=4):
    stub = vacuous_solution(task)
    sources = single_test_sources(task["test_code"])
    if not sources:
        return 0.0, 0

    def failed(source):
        return sandbox.run(stub, dict(task, test_code=source))["status"] != "pass"

    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(failed, sources))
    return sum(results) / len(results), len(sources)


def to_task(task_id, sections, source_info, tokenizer):
    spec = sections["spec"]
    turn = f"Task:\n{spec}\n\nWrite the complete solution in Python in one ```python code block."
    task = Task(
        task_id=task_id,
        family="PROJ-8",
        source="anthropochrony/proj8",
        source_revision=source_info["revision"],
        license="Apache-2.0",
        task_turn=turn,
        harness="unittest",
        reference_solution=sections["reference"],
        test_code=sections["tests"],
        split="showcase",
        time_limits={"per_test_s": 10, "per_task_s": 30},
    ).as_dict()
    task["ref_tokens"] = len(tokenizer.encode(sections["reference"], add_special_tokens=False))
    task["tests_ref"] = digest({"test": sections["tests"], "io": None})
    return task


def build_proj8(root, provider, sandbox, tokenizer, card):
    from anthropochrony.gen.engine import render, system_token_count
    from anthropochrony.prompts import system_prompt

    root = Path(root)
    kept, report = [], []
    revision = digest(CATEGORIES)[:12]
    for task_id, description in CATEGORIES.items():
        for variant in range(3):
            entry = {"task_id": task_id, "variant": variant}
            try:
                sections = draft(provider, task_id, description, variant=variant)
            except ValueError as error:
                entry["dropped"] = f"draft: {error}"
                report.append(entry)
                continue
            task = to_task(task_id, sections, {"revision": revision}, tokenizer)
            task["prompt_tokens"] = len(
                render(
                    tokenizer,
                    [
                        {"role": "system", "content": system_prompt(card, estimator="agent")},
                        {"role": "user", "content": task["task_turn"]},
                    ],
                )
            )
            task["uncached_task_tokens"] = task["prompt_tokens"] - system_token_count(
                tokenizer, system_prompt(card, estimator="agent")
            )
            task["test_code"], entry["drafted_tests"], entry["dropped_tests"] = prune_to_reference(
                task, sandbox
            )
            task["tests_ref"] = digest({"test": task["test_code"], "io": None})
            task, flag = verify_reference(task, sandbox)
            entry["reference_flag"] = flag
            entry["reference_times_s"] = task["gt"]["times_s"]
            if flag is None:
                share, count = stub_failure_share(task, sandbox)
                entry["stub_failure_share"], entry["tests"] = share, count
                if share < 0.9 or count < 8:
                    entry["dropped"] = "stub passes too many tests or too few tests"
                elif max(task["gt"]["times_s"]) > 10:
                    entry["dropped"] = "slower than 10 s"
            else:
                entry["dropped"] = flag
            report.append(entry)
            if "dropped" not in entry:
                kept.append(task)
                break
    write_jsonl(root / "data/tasks/proj8.jsonl", kept)
    write_json(
        root / "data/tasks/proj8-report.json",
        {"created_at": now(), "kept": [t["task_id"] for t in kept], "report": report},
    )
    return kept, report
