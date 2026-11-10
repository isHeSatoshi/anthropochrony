import copy
import itertools
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from anthropochrony.artifacts import digest, file_hash, now, read_jsonl, write_json, write_jsonl
from anthropochrony.gates import input_fingerprint
from anthropochrony.gen.engine import render, system_token_count
from anthropochrony.prompts import feedback, system_prompt
from anthropochrony.sandbox.runner import Sandbox, extract_code
from anthropochrony.timemodel import attempt_budget, uncached_tokens


def initial_states(tasks, card, config):
    states = []
    for task in tasks:
        for k in range(config["k"]):
            run_id = digest({"task_id": task["task_id"], "k": k, "model": config["model_sha"]})
            states.append(
                {
                    "run_id": run_id,
                    "task_id": task["task_id"],
                    "model_id": config["model_id"],
                    "model_sha": config["model_sha"],
                    "k": k,
                    "card": card.as_dict(),
                    "attempts": [],
                    "n_attempts": 0,
                    "passed": False,
                    "first_pass": False,
                    "harness_error": False,
                    "done": False,
                    "created_at": now(),
                    "sampling": {key: config[key] for key in ("temperature", "top_p")},
                    "messages": [
                        {"role": "system", "content": system_prompt(card, estimator="agent")},
                        {"role": "user", "content": task["task_turn"]},
                    ],
                }
            )
    return states


def prepare_round(states, tasks, tokenizer, round_number):
    by_task = {t["task_id"]: t for t in tasks}
    requests = []
    for state in states:
        if state["done"] or len(state["attempts"]) != round_number - 1:
            continue
        ids = render(tokenizer, state["messages"])
        maximum = attempt_budget(len(ids), by_task[state["task_id"]]["family"])
        if maximum == 0:
            state.update(done=True, terminal_status="context_exhausted")
            continue
        request = {
            "request_id": state["run_id"],
            "prompt_token_ids": ids,
            "max_tokens": maximum,
            "seed": int(digest([state["task_id"], state["k"], round_number])[:8], 16),
            "system_tokens": system_token_count(tokenizer, state["messages"][0]["content"]),
        }
        if state["attempts"]:
            # Degenerate, length-capped text can re-tokenize shorter than it was
            # sampled, so sampled counts alone may exceed what the retry prompt
            # can reuse. The re-rendered turn's closing token was never sampled.
            prefix = render(tokenizer, state["messages"][:-1], generation_prompt=False)
            if prefix[-1] != 6:
                raise ValueError("Re-rendered assistant turn does not end with END_OF_TURN")
            request["reusable_prefix_tokens"] = len(prefix) - 1
        requests.append(request)
    return requests


def apply_round(states, tasks, requests, outputs, sandbox, round_number, workers=8):
    tasks_by_id = {t["task_id"]: t for t in tasks}
    request_map = {r["request_id"]: r for r in requests}
    output_map = {o["request_id"]: o for o in outputs}
    if (
        len(request_map) != len(requests)
        or len(output_map) != len(outputs)
        or set(output_map) != set(request_map)
    ):
        raise ValueError("Generation results do not match the requested run IDs")
    by_run = {s["run_id"]: s for s in states}
    if not set(request_map) <= set(by_run):
        raise ValueError("Unknown trajectory")
    for run_id, request in request_map.items():
        state, output = by_run[run_id], output_map[run_id]
        if output["prompt_tokens_total"] != len(request["prompt_token_ids"]):
            raise ValueError("vLLM changed rendered prompt token IDs")
        if len(state["attempts"]) >= round_number:
            previous = state["attempts"][round_number - 1]
            if (
                previous["text_sha"] != digest(output["text"])
                or previous["o_tokens"] != output["o_tokens"]
            ):
                raise ValueError("A completed round has different generation outputs")
        elif state["done"] or len(state["attempts"]) != round_number - 1:
            raise ValueError("Out-of-order measurement round")
    active = [
        s for s in states if s["run_id"] in request_map and len(s["attempts"]) == round_number - 1
    ]
    local = threading.local()
    cpus = sorted(os.sched_getaffinity(0))[:workers]
    worker_index = itertools.count()

    def initialize():
        local.sandbox = Sandbox(
            python=sandbox.python,
            memory_mb=sandbox.memory_mb,
            tasks_max=sandbox.tasks_max,
            cpu=cpus[next(worker_index) % len(cpus)],
        )

    def execute(state):
        output = output_map[state["run_id"]]
        code = extract_code(output["text"])
        result = local.sandbox.run(code, tasks_by_id[state["task_id"]])
        return state["run_id"], result, code is not None

    with ThreadPoolExecutor(max_workers=workers, initializer=initialize) as pool:
        executions = {
            run_id: (result, code_found) for run_id, result, code_found in pool.map(execute, active)
        }
    texts = []
    for state in active:
        request, output = request_map[state["run_id"]], output_map[state["run_id"]]
        execution, code_found = executions[state["run_id"]]
        previous = state["attempts"][-1] if state["attempts"] else None
        total = output["prompt_tokens_total"]
        if total != len(request["prompt_token_ids"]):
            raise ValueError("vLLM changed rendered prompt token IDs")
        sampled = previous["prompt_tokens_total"] + previous["o_tokens"] if previous else None
        if previous and "reusable_prefix_tokens" in request:
            # Equals the sampled-count formula unless re-tokenization shortened the text.
            u = total - min(sampled, request["reusable_prefix_tokens"])
            if u <= 0:
                raise ValueError("A retry must add feedback tokens")
        else:
            u = uncached_tokens(
                total,
                request["system_tokens"],
                previous["prompt_tokens_total"] if previous else None,
                previous["o_tokens"] if previous else None,
            )
        text_sha = digest(output["text"])
        attempt = {
            "j": round_number,
            "prompt_tokens_total": total,
            "u_tokens": u,
            "o_tokens": output["o_tokens"],
            "finish_reason": output["finish_reason"],
            "text_sha": text_sha,
            "code_found": code_found,
            "exec": execution,
        }
        state["attempts"].append(attempt)
        state["n_attempts"] = len(state["attempts"])
        state["passed"] = execution["status"] == "pass"
        state["first_pass"] = state["attempts"][0]["exec"]["status"] == "pass"
        state["harness_error"] |= execution["status"] == "error"
        state["done"] = state["passed"] or state["n_attempts"] == 3 or state["harness_error"]
        state["messages"].append({"role": "assistant", "content": output["text"]})
        if not state["done"]:
            state["messages"].append({"role": "user", "content": feedback(round_number, execution)})
        state["total_o"] = sum(a["o_tokens"] for a in state["attempts"])
        state["total_u"] = sum(a["u_tokens"] for a in state["attempts"])
        state["total_exec_s"] = sum(a["exec"]["wall_s"] for a in state["attempts"])
        texts.append({"text_sha": text_sha, "text": output["text"]})
    return states, texts


class Measurement:
    def __init__(self, directory, tasks, config, card):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.tasks, self.config, self.card = tasks, config, card
        execution = Path(__file__).parents[1]
        environment = {
            "card": card.as_dict(),
            "execution": {
                name: file_hash(execution / name)
                for name in (
                    "execenv/requirements.lock",
                    "sandbox/runner.py",
                    "sandbox/worker.py",
                    "gen/engine.py",
                    "measure/rounds.py",
                    "prompts/__init__.py",
                    "prompts/templates.yaml",
                    "timemodel.py",
                )
            },
        }
        self.fingerprint = input_fingerprint(tasks, config | environment, config["model_sha"])
        self.checkpoint = self.directory / "checkpoint.json"

    def states(self):
        import json

        if not self.checkpoint.exists():
            return initial_states(self.tasks, self.card, self.config)
        record = json.loads(self.checkpoint.read_text())
        if record["fingerprint"] != self.fingerprint:
            raise ValueError("Refusing to resume with changed tasks, model, or configuration")
        return record["states"]

    def save(self, states):
        write_json(
            self.checkpoint, {"fingerprint": self.fingerprint, "states": states}, config=self.config
        )
        traces = [{k: v for k, v in s.items() if k != "messages"} for s in states]
        write_jsonl(self.directory / "traces.jsonl", traces, config=self.config)

    def run(self, engine, sandbox):
        states = self.states()
        for round_number in range(1, 4):
            requests = prepare_round(states, self.tasks, engine.tokenizer, round_number)
            if not requests:
                self.save(states)
                continue
            output_path = self.directory / f"generation-r{round_number}.jsonl"
            request_path = self.directory / f"requests-r{round_number}.jsonl"
            if output_path.exists():
                # Crash after generation must not pay for new random outputs.
                if not request_path.exists() or read_jsonl(request_path) != requests:
                    raise ValueError("Saved generation belongs to a different round request")
                outputs = read_jsonl(output_path)
            else:
                write_jsonl(request_path, requests, config=self.config)
                outputs = engine.generate(requests)
                write_jsonl(output_path, outputs, inputs=[request_path], config=self.config)
            states, texts = apply_round(
                states,
                self.tasks,
                requests,
                outputs,
                sandbox,
                round_number,
                self.config["workers"],
            )
            write_jsonl(
                self.directory / f"outputs-r{round_number}.jsonl", texts, inputs=[output_path]
            )
            self.save(states)
        return copy.deepcopy(states)
