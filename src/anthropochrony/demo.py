"""Demo controller: curated-task execution only; pasted code is never executed."""

import json
import subprocess
import time
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from anthropochrony.eval.prompts import BASE_PREFILL
from anthropochrony.parse import answer_segment
from anthropochrony.prompts import estimate_messages, feedback, system_prompt
from anthropochrony.sandbox import Sandbox
from anthropochrony.sandbox.runner import extract_code
from anthropochrony.timemodel import SpeedCard


class BackendUnavailable(RuntimeError):
    pass


class ContextExhausted(RuntimeError):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Local model endpoints must not redirect")


class LlamaBackend:
    def __init__(self, base_url, tuned_url, model_path, bench, threads=8, gpu_layers=0):
        for url in (base_url, tuned_url):
            parsed = urlparse(url)
            if (
                parsed.scheme != "http"
                or parsed.hostname not in {"127.0.0.1", "localhost"}
                or not parsed.port
                or parsed.username
                or parsed.password
                or parsed.path not in {"", "/"}
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("Demo backends must be local llama.cpp servers")
        self.base_url, self.tuned_url = base_url, tuned_url
        self.model_path, self.bench, self.threads = model_path, bench, threads
        self.gpu_layers = int(gpu_layers)

    def calibrate(self):
        command = [
            self.bench,
            "-m",
            str(self.model_path),
            "-p",
            "512",
            "-n",
            "128",
            "-t",
            str(self.threads),
            "-r",
            "5",
            "-o",
            "json",
            # GPU builds of llama-bench offload every layer unless told otherwise.
            "-ngl",
            str(self.gpu_layers),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=300)
        rows = json.loads(result.stdout)
        prefill = next(r["avg_ts"] for r in rows if r["n_prompt"] and not r["n_gen"])
        decode = next(r["avg_ts"] for r in rows if r["n_gen"] and not r["n_prompt"])
        return SpeedCard("demo-llamacpp", decode, prefill)

    def request(self, url, path, data):
        body = json.dumps(data).encode()
        with build_opener(NoRedirect).open(
            Request(url.rstrip("/") + path, body, {"Content-Type": "application/json"}),
            timeout=180,
        ) as response:
            return json.load(response)

    def generate(self, messages, *, base=False, max_tokens=512, coding=False, prefill=""):
        url = self.base_url if base else self.tuned_url
        prompt = self.request(url, "/apply-template", {"messages": messages})["prompt"] + prefill
        tokens = self.request(url, "/tokenize", {"content": prompt, "add_special": False})["tokens"]
        remaining = 8192 - len(tokens)
        if remaining < 256:
            raise ContextExhausted("Fewer than 256 context tokens remain")
        if prefill:
            # The prompt already ends with the prefill, so continue it as raw text.
            result = self.request(
                url,
                "/completion",
                {
                    "prompt": prompt,
                    "n_predict": min(max_tokens, remaining),
                    "temperature": 0,
                    "stop": ["<|END_RESPONSE|>", "<|END_OF_TURN_TOKEN|>"],
                    "special": False,
                },
            )
            return prefill + result["content"]
        result = self.request(
            url,
            "/v1/chat/completions",
            {
                "messages": messages,
                "max_tokens": min(max_tokens, remaining),
                "temperature": 0.1 if coding else 0,
                "top_p": 0.95 if coding else 1,
                "stop": ["<|END_RESPONSE|>", "<|END_OF_TURN_TOKEN|>"],
            },
        )
        return result["choices"][0]["message"]["content"]


class Demo:
    def __init__(self, tasks, backend=None, sandbox=None):
        self.tasks = {t["task_id"]: t for t in tasks}
        self.backend = backend
        self.sandbox = sandbox
        self.last_card = None

    def estimate(self, task_id, pasted=""):
        if self.backend is None:
            raise BackendUnavailable(
                "No trained backend is configured. No substitute estimates are shown."
            )
        from anthropochrony.prompts import task_turn

        task = {"task_turn": task_turn(pasted)} if pasted.strip() else self.tasks[task_id]
        card = self.backend.calibrate()
        self.last_card = card
        base = answer_segment(
            self.backend.generate(
                estimate_messages(task, card, estimator="E0"), base=True, prefill=BASE_PREFILL
            )
        )
        tuned = self.backend.generate(estimate_messages(task, card, estimator="agent"))
        human = self.backend.generate(
            estimate_messages(task, card, template_id="h09", estimator="agent")
        )
        return {
            "card": card.as_dict(),
            "base": base,
            "anthropochrony": tuned,
            "human": human,
            "execution_allowed": not pasted.strip(),
            "deployment": "Quantized llama.cpp demo only, not the bf16 reference deployment.",
        }

    def run(self, task_id, *, pasted="", card=None):
        if pasted.strip() or task_id not in self.tasks:
            raise ValueError("Only allowlisted, curated tasks can be executed")
        card = card or self.last_card
        if self.backend is None or card is None:
            raise BackendUnavailable("Measure a speed card and request estimates first")
        task = self.tasks[task_id]
        if not task.get("gt", {}).get("passes"):
            raise ValueError("Demo execution requires verified ground truth")
        sandbox = self.sandbox or Sandbox()
        messages = [
            {"role": "system", "content": system_prompt(card, estimator="agent")},
            {"role": "user", "content": task["task_turn"]},
        ]
        attempts = []
        started = time.monotonic()
        terminal_status = None
        for number in range(1, 4):
            try:
                answer = self.backend.generate(
                    messages,
                    max_tokens=4096
                    if task["family"].startswith("PROJ") or task["family"] == "ALGO_IO"
                    else 2048,
                    coding=True,
                )
            except ContextExhausted:
                terminal_status = "context_exhausted"
                break
            execution = sandbox.run(extract_code(answer), task)
            attempts.append({"attempt": number, "answer": answer, "execution": execution})
            if execution["status"] in {"pass", "error"}:
                break
            messages += [
                {"role": "assistant", "content": answer},
                {"role": "user", "content": feedback(number, execution)},
            ]
        return {
            "wall_s": time.monotonic() - started,
            "attempts": attempts,
            "passed": bool(attempts) and attempts[-1]["execution"]["status"] == "pass",
            "terminal_status": terminal_status,
        }
