"""Pinned BGE prompt embeddings on CPU, without loading Torch or executing tasks."""

from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer

from anthropochrony.artifacts import digest, manifest, write_json


def embeddings(tasks, config, directory):
    import onnxruntime as ort

    directory = Path(directory)
    identity = digest({"config": config, "inputs": [(t["task_id"], t["task_turn"]) for t in tasks]})
    path = directory / f"{identity}.npz"
    if path.exists():
        saved = np.load(path, allow_pickle=False)
        return dict(zip(saved["ids"].tolist(), saved["features"], strict=True))
    model = hf_hub_download(config["repo"], config["onnx_file"], revision=config["revision"])
    tokenizer_dir = snapshot_download(
        config["repo"],
        revision=config["revision"],
        allow_patterns=["tokenizer*", "special_tokens_map.json", "vocab.txt", "config.json"],
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(model, sess_options=options, providers=["CPUExecutionProvider"])
    names = {value.name for value in session.get_inputs()}
    vectors = []
    for offset in range(0, len(tasks), config["batch_size"]):
        batch = tasks[offset : offset + config["batch_size"]]
        inputs = tokenizer(
            [t["task_turn"] for t in batch],
            padding=True,
            truncation=True,
            max_length=config["max_length"],
            return_tensors="np",
        )
        outputs = session.run(None, {name: inputs[name].astype(np.int64) for name in names})
        cls = outputs[0][:, 0, :]
        cls /= np.maximum(np.linalg.norm(cls, axis=1, keepdims=True), 1e-12)
        vectors.append(cls)
    features = np.concatenate(vectors)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=np.asarray([t["task_id"] for t in tasks]), features=features)
    manifest(path, config=config)
    write_json(
        path.with_suffix(".json"),
        {
            "identity": identity,
            "shape": list(features.shape),
            "model": config,
            "onnxruntime": ort.__version__,
        },
    )
    return dict(zip([t["task_id"] for t in tasks], features, strict=True))
