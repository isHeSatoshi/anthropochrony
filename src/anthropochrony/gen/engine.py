import time

from anthropochrony.timemodel import output_token_count


def render(tokenizer, messages, *, generation_prompt=True):
    ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=generation_prompt,
        tokenize=True,
        return_dict=False,
    )
    if not isinstance(ids, list) or not all(isinstance(i, int) for i in ids):
        raise TypeError("Chat template must return a flat token-ID list")
    if ids[:2] == [2, 2]:
        raise ValueError("Double BOS")
    return ids


def system_token_count(tokenizer, system):
    ids = render(
        tokenizer, [{"role": "system", "content": system}, {"role": "user", "content": ""}]
    )
    return ids.index(6) + 1


def prepare_estimate_requests(tokenizer, prompts, context=8192):
    requests, exhausted = [], {}
    if len({row["est_id"] for row in prompts}) != len(prompts):
        raise ValueError("Duplicate estimate prompt IDs")
    for row in prompts:
        ids = render(tokenizer, row["messages"])
        remaining = context - len(ids)
        if remaining <= 0:
            exhausted[row["est_id"]] = {
                "request_id": row["est_id"],
                "text": "",
                "token_ids": [],
                "o_tokens": 0,
                "prompt_tokens_total": len(ids),
                "finish_reason": "context_exhausted",
                "stop_reason": None,
            }
        else:
            requests.append(
                {
                    "request_id": row["est_id"],
                    "prompt_token_ids": ids,
                    "max_tokens": min(row["max_tokens"], remaining),
                    "seed": 0,
                }
            )
    return requests, exhausted


class VLLMEngine:
    def __init__(self, model, config):
        from vllm import LLM

        self.llm = LLM(model=str(model), **config)
        self.tokenizer = self.llm.get_tokenizer()

    def generate(self, requests, *, greedy=False, max_tokens=None):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        params = [
            SamplingParams(
                temperature=0.0 if greedy else 0.1,
                top_p=1.0 if greedy else 0.95,
                seed=r["seed"],
                max_tokens=max_tokens or r["max_tokens"],
                stop_token_ids=[6, 261001],
                n=1,
            )
            for r in requests
        ]
        started = time.monotonic()
        outputs = self.llm.generate(
            [TokensPrompt(prompt_token_ids=r["prompt_token_ids"]) for r in requests],
            params,
        )
        elapsed = time.monotonic() - started
        results = []
        for request, output in zip(requests, outputs, strict=True):
            generated = output.outputs[0]
            results.append(
                {
                    "request_id": request["request_id"],
                    "text": generated.text,
                    "token_ids": list(generated.token_ids),
                    "o_tokens": output_token_count(generated.token_ids, generated.stop_reason),
                    "prompt_tokens_total": len(output.prompt_token_ids),
                    "finish_reason": generated.finish_reason,
                    "stop_reason": generated.stop_reason,
                }
            )
        total = sum(row["o_tokens"] for row in results)
        self.last_performance = {
            "elapsed_s": elapsed,
            "output_tokens": total,
            "tokens_per_second": total / elapsed,
            "usd_per_million_output_tokens": elapsed * 1.8 / 3600 / total * 1e6 if total else None,
        }
        return results
