import json
from pathlib import Path

from huggingface_hub import InferenceClient

from anthropochrony.artifacts import digest, now, write_json
from anthropochrony.budget import Budget, BudgetError
from anthropochrony.prompts import load


class CachedProvider:
    def __init__(self, root, *, model, provider, input_per_million, output_per_million, cap=3):
        self.root, self.model, self.provider = Path(root), model, provider
        self.input_price, self.output_price, self.cap = input_per_million, output_per_million, cap
        self.budget = Budget(root)
        self.cache = self.root / "data/provider-cache"
        self.cache.mkdir(parents=True, exist_ok=True)

    def call(self, prompt, max_tokens=300, temperature=0):
        key = digest(
            {
                "model": self.model,
                "provider": self.provider,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        path = self.cache / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())["answer"]
        # UTF-8 bytes safely upper-bound ordinary tokenizer input token counts.
        upper = (len(prompt.encode()) * self.input_price + max_tokens * self.output_price) / 1e6
        with self.budget.ledger() as data:
            provider_total = sum(
                e.get("actual_usd", e["reserved_usd"])
                for e in data["entries"]
                if e["name"].startswith("provider-")
            )
        if provider_total + upper > self.cap:
            raise BudgetError(f"Teacher/judge provider cap ${self.cap} would be exceeded")
        name = f"provider-{key}"
        self.budget.reserve(name, upper)
        with (self.root / "RUNLOG.md").open("a") as log:
            log.write(
                f"| {key[:16]} | provider {self.model} | {self.provider} | — | ≤{upper:.6f} | Reserved before request |\n"
            )
        # Tokens are read by the SDK; they never enter prompts, cache files, or sandbox envs.
        client = InferenceClient(provider=self.provider)
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as error:
            self.budget.update(name, state="request_failed_charge_unknown")
            status = getattr(getattr(error, "response", None), "status_code", None)
            if status in {401, 403}:
                raise PermissionError(
                    "HF authorization failed; stop remote work and request renewed access"
                ) from error
            raise
        usage = response.usage
        actual = (
            usage.prompt_tokens * self.input_price + usage.completion_tokens * self.output_price
        ) / 1e6
        self.budget.update(name, actual_usd=actual, state="completed")
        answer = response.choices[0].message.content
        write_json(
            path,
            {
                "model": self.model,
                "provider": self.provider,
                "date": now(),
                "answer": answer,
                "prompt_sha": digest(prompt),
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "cost_usd": actual,
            },
        )
        return answer

    def teacher(self, task):
        prompt = load()["teacher"].replace("{task_turn}", task["task_turn"])
        for attempt in range(2):
            answer = self.call(prompt + ("\nReply with valid JSON only." if attempt else ""))
            try:
                result = json.loads(answer.removeprefix("```json").removesuffix("```").strip())
                assert 0 < result["tokens_p10"] <= result["tokens_median"] <= result["tokens_p90"]
                assert result["attempts_median"] in {1, 2, 3}
                assert 0 <= result["p_first_pass"] <= 1 and result["test_seconds"] >= 0
                return result
            except (ValueError, KeyError, TypeError, AssertionError):
                continue
        return None

    def judge(self, question, answer):
        prompt = load()["judge"].replace("{question}", question).replace("{answer}", answer)
        return json.loads(self.call(prompt, max_tokens=300))
