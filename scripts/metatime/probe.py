"""Linear probe baseline: does the untrained model's hidden state already encode its length?

A ridge regression on the base model's last-prompt-token hidden state predicts the log of
its own response length (the linear-head family of MetaTime, EGTP and ProD; this project's
planned B3). The layer and penalty are chosen on held-out pool prompts, never on the benchmark.
"""

import argparse
import json

import numpy as np
import torch
from common import PRE_ESTIMATE, load_tasks, task_view
from engine import Engine
from sklearn.linear_model import Ridge
from train import load_pool

LAYERS = [6, 12, 18, 24, 30, 36]


@torch.inference_mode()
def features(engine, texts, batch=8):
    rows = [engine._ids(engine.encode(text)) for text in texts]
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]))
    out = [None] * len(rows)
    for start in range(0, len(order), batch):
        group = order[start : start + batch]
        width = max(len(rows[i]) for i in group)
        ids = torch.full((len(group), width), engine.pad, dtype=torch.long, device="cuda")
        mask = torch.zeros_like(ids)
        for position, i in enumerate(group):
            ids[position, width - len(rows[i]) :] = torch.tensor(rows[i], device="cuda")
            mask[position, width - len(rows[i]) :] = 1
        hidden = engine.model(
            input_ids=ids, attention_mask=mask, output_hidden_states=True, logits_to_keep=1
        ).hidden_states
        layers = [layer for layer in LAYERS if layer < len(hidden)]
        stacked = torch.stack([hidden[layer][:, -1].float() for layer in layers], 1).cpu().numpy()
        for position, i in enumerate(group):
            out[i] = stacked[position]
    return np.stack(out), layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-thinking", action="store_true")
    args = parser.parse_args()

    engine = Engine(args.model)
    if args.no_thinking:
        engine.template_kwargs = {"enable_thinking": False}
    pool = load_pool(args.pool, args.labels)
    tasks = load_tasks(args.suite)
    target = np.log([max(1.0, row["n_out"]) for row in pool])
    train = np.array([row["split"] == "train" for row in pool])
    result = {"layers": LAYERS, "contexts": {}}
    # "task": the state just before the model starts its real answer to the full prompt.
    # "question": the state after the benchmark's question, which shows only the description.
    contexts = {
        "task": lambda row: row["prompt"],
        "question": lambda row: PRE_ESTIMATE.format(description=task_view(row, "description")),
    }
    for name, render in contexts.items():
        pool_x, layers = features(engine, [render(row) for row in pool])
        task_x, _ = features(engine, [render(task) for task in tasks])
        best = None
        for index, layer in enumerate(layers):
            mean = pool_x[train, index].mean(0)
            scale = pool_x[train, index].std(0) + 1e-6
            x = (pool_x[:, index] - mean) / scale
            for alpha in (10.0, 100.0, 1000.0, 10000.0):
                model = Ridge(alpha=alpha).fit(x[train], target[train])
                error = float(np.median(np.abs(model.predict(x[~train]) - target[~train])))
                if best is None or error < best["val_median_abs_log_error"]:
                    best = {
                        "layer": layer,
                        "alpha": alpha,
                        "val_median_abs_log_error": error,
                        "val_pearson_log": float(
                            np.corrcoef(model.predict(x[~train]), target[~train])[0, 1]
                        ),
                        "tokens": np.exp(model.predict((task_x[:, index] - mean) / scale)).tolist(),
                    }
        result["contexts"][name] = {
            **{key: value for key, value in best.items() if key != "tokens"},
            "predicted_tokens": dict(
                zip([task["id"] for task in tasks], best["tokens"], strict=True)
            ),
        }
        print(name, {key: value for key, value in best.items() if key != "tokens"}, flush=True)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=1)


if __name__ == "__main__":
    main()
