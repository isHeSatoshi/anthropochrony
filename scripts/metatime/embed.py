"""Pinned BGE sentence embeddings on CPU (ONNX Runtime, no PyTorch), cached by content."""

import hashlib
import json
from pathlib import Path

import numpy as np


def embeddings(rows, config, directory):
    """Unit-length CLS vectors for `rows` (`{"id", "text"}`), keyed by id.

    The cache file is named by a hash of the encoder settings and the exact inputs, so a
    change to either is recomputed and nothing else is.
    """
    directory = Path(directory)
    identity = hashlib.sha256(
        json.dumps(
            {"config": config, "inputs": [(row["id"], row["text"]) for row in rows]},
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    path = directory / f"{identity}.npz"
    if path.exists():
        saved = np.load(path, allow_pickle=False)
        return dict(zip(saved["ids"].tolist(), saved["features"], strict=True))

    import onnxruntime as ort
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoTokenizer

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
    for offset in range(0, len(rows), config["batch_size"]):
        batch = rows[offset : offset + config["batch_size"]]
        inputs = tokenizer(
            [row["text"] for row in batch],
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
    ids = [row["id"] for row in rows]
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=np.asarray(ids), features=features)
    return dict(zip(ids, features, strict=True))
