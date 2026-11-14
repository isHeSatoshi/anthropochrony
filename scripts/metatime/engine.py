"""GPU engine for the MetaTime head-to-head: timed single-stream runs and batched label runs.

Runs outside the pinned stack environment because it needs PyTorch; see docs/METATIME.md.
"""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.1
TOP_P = 0.95


def _sample(logits, temperature, top_p, generator):
    if temperature <= 0:
        return logits.argmax(-1)
    probs = torch.softmax(logits.float() / temperature, -1)
    ordered, index = probs.sort(-1, descending=True)
    keep = ordered.cumsum(-1) - ordered < top_p
    ordered = ordered * keep
    choice = torch.multinomial(ordered / ordered.sum(-1, keepdim=True), 1, generator=generator)
    return index.gather(-1, choice).squeeze(-1)


class Engine:
    def __init__(self, model_dir, adapter=None, load_in_4bit=False):
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        kwargs = {"device_map": {"": 0}}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["dtype"] = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
        self.adapter = None
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
            self.adapter = adapter
        self.model.eval()
        stops = {self.tok.eos_token_id}
        for name in ("<|END_OF_TURN_TOKEN|>", "<|END_RESPONSE|>", "<|im_end|>", "<|endoftext|>"):
            token = self.tok.convert_tokens_to_ids(name)
            if isinstance(token, int) and token >= 0 and token != self.tok.unk_token_id:
                stops.add(token)
        self.stops = torch.tensor(sorted(stops), device="cuda")
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        self.template_kwargs = {}

    def use_adapter(self, enabled):
        """Switch the LoRA on or off without reloading; a no-op for a plain base model."""
        if self.adapter is None:
            return
        if enabled:
            self.model.enable_adapter_layers()
        else:
            self.model.disable_adapter_layers()

    def encode(self, user, system=None, history=()):
        messages = [{"role": "system", "content": system}] if system else []
        for question, answer in history:
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": user})
        return self.tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, **self.template_kwargs
        )

    def _ids(self, encoded):
        return encoded["input_ids"] if hasattr(encoded, "keys") else encoded

    @torch.inference_mode()
    def generate_timed(
        self,
        user,
        system=None,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        seed=0,
        history=(),
    ):
        """One request on an otherwise idle GPU; wall time runs from request to last token."""
        generator = torch.Generator(device="cuda").manual_seed(seed)
        torch.cuda.synchronize()
        start = time.perf_counter()
        ids = torch.tensor([self._ids(self.encode(user, system, history))], device="cuda")
        out = self.model(input_ids=ids, use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        token = _sample(out.logits[:, -1], temperature, top_p, generator)
        torch.cuda.synchronize()
        first = time.perf_counter()
        produced = []
        finish = "length"
        while True:
            if torch.isin(token, self.stops).item():
                finish = "stop"
                break
            produced.append(token.item())
            if len(produced) >= max_new_tokens:
                break
            out = self.model(
                input_ids=token[:, None], past_key_values=past, use_cache=True, logits_to_keep=1
            )
            past = out.past_key_values
            token = _sample(out.logits[:, -1], temperature, top_p, generator)
        torch.cuda.synchronize()
        end = time.perf_counter()
        return {
            "text": self.tok.decode(produced, skip_special_tokens=True),
            "n_in": ids.shape[1],
            "n_out": len(produced),
            "wall_s": end - start,
            "prefill_s": first - start,
            "decode_s": end - first,
            "finish": finish,
        }

    @torch.inference_mode()
    def generate_batch(
        self,
        encoded,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        seed=0,
        keep_text=True,
    ):
        """Sample every prompt once; finished rows leave the batch so cost follows total tokens."""
        generator = torch.Generator(device="cuda").manual_seed(seed)
        rows = [self._ids(item) for item in encoded]
        width = max(len(row) for row in rows)
        ids = torch.full((len(rows), width), self.pad, dtype=torch.long, device="cuda")
        mask = torch.zeros((len(rows), width), dtype=torch.long, device="cuda")
        for index, row in enumerate(rows):
            ids[index, width - len(row) :] = torch.tensor(row, device="cuda")
            mask[index, width - len(row) :] = 1
        out = self.model(input_ids=ids, attention_mask=mask, use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        active = torch.arange(len(rows), device="cuda")
        produced = [[] for _ in rows]
        finish = ["length"] * len(rows)
        token = _sample(out.logits[:, -1], temperature, top_p, generator)
        while True:
            stopped = torch.isin(token, self.stops)
            for position, row in enumerate(active.tolist()):
                if stopped[position]:
                    finish[row] = "stop"
                else:
                    produced[row].append(token[position].item())
            done = stopped | torch.tensor(
                [len(produced[row]) >= max_new_tokens for row in active.tolist()], device="cuda"
            )
            if done.all():
                break
            if done.any():
                keep = (~done).nonzero().squeeze(-1)
                past.batch_select_indices(keep)
                active, token, mask = active[keep], token[keep], mask[keep]
            mask = torch.cat([mask, torch.ones_like(mask[:, :1])], 1)
            out = self.model(
                input_ids=token[:, None],
                attention_mask=mask,
                past_key_values=past,
                use_cache=True,
                logits_to_keep=1,
            )
            past = out.past_key_values
            token = _sample(out.logits[:, -1], temperature, top_p, generator)
        return [
            {
                "n_in": len(rows[index]),
                "n_out": len(produced[index]),
                "finish": finish[index],
                **(
                    {"text": self.tok.decode(produced[index], skip_special_tokens=True)}
                    if keep_text
                    else {}
                ),
            }
            for index in range(len(rows))
        ]

    def speed_card(self, repeats=3):
        """Measured decode and prompt-processing speed of this deployment, in tokens per second."""
        filler = "The quick brown fox jumps over the lazy dog. " * 60
        decode, prefill = [], []
        for seed in range(repeats + 1):
            run = self.generate_timed(
                "Count from 1 to 200, one number per line.", max_new_tokens=256, seed=seed
            )
            long = self.generate_timed(
                filler + "\nReply with the single word OK.", max_new_tokens=4, seed=seed
            )
            if seed:  # the first pass warms the GPU
                decode.append(run["n_out"] / run["decode_s"])
                prefill.append(long["n_in"] / long["prefill_s"])
        decode.sort()
        prefill.sort()
        return {"S": decode[len(decode) // 2], "P": prefill[len(prefill) // 2]}
