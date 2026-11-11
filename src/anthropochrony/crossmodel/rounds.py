"""Round preparation for Aya Expanse 8B (single end-of-turn token). Execution and bookkeeping
reuse the pinned `apply_round`, which carries no tokenizer-specific constants."""

from anthropochrony.artifacts import digest
from anthropochrony.crossmodel import EOT
from anthropochrony.gen.engine import render
from anthropochrony.timemodel import attempt_budget


def system_token_count(tokenizer, system):
    ids = render(
        tokenizer, [{"role": "system", "content": system}, {"role": "user", "content": ""}]
    )
    return ids.index(EOT) + 1


def prepare_round(states, tasks, tokenizer, round_number, system_tokens):
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
            "system_tokens": system_tokens,
        }
        if state["attempts"]:
            prefix = render(tokenizer, state["messages"][:-1], generation_prompt=False)
            if prefix[-1] != EOT:
                raise ValueError("Re-rendered assistant turn does not end with END_OF_TURN")
            request["reusable_prefix_tokens"] = len(prefix) - 1
        requests.append(request)
    return requests
