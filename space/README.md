---
title: Anthropochrony
emoji: ⏱️
colorFrom: indigo
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: cc-by-nc-4.0
short_description: A small model that knows how long its own answers take
models:
  - CohereLabs/tiny-aya-global
  - isHeSatoshi/anthropochrony-tiny-aya-global-g1-gguf
---

# Anthropochrony demo

Ask an AI how long a task will take it and it answers on a human clock. This Space runs
Cohere Labs' Tiny Aya Global (3.35B) on its own CPU twice, untrained and with our G1 adapter
merged in (4-bit GGUF, llama.cpp `b12110`). It measures this machine's speed, asks both how
long your task will take, then runs the task and times it.

The second tab shows the benchmark head-to-head on the 68 MetaTime single-turn tasks. Code,
rules declared before the first run, and the write-up: https://github.com/isHeSatoshi/anthropochrony

CPU and 4-bit quantization change the numbers; the benchmark results are for bf16 on a GPU.
Non-commercial (CC-BY-NC-4.0, from the base model's license).

## Run it locally

Needs Python 3.10 or newer, about 5 GB of disk for the two GGUF files and a llama.cpp
release build (tested with `b12110`, Linux x64 CPU).

```bash
# from the repository root
mkdir -p /tmp/llama && curl -fsSL https://github.com/ggml-org/llama.cpp/releases/download/b12110/llama-b12110-bin-ubuntu-x64.tar.gz \
  | tar -xz -C /tmp/llama --strip-components=1 && export LLAMA_DIR=/tmp/llama
python3 -m venv /tmp/demo-env && /tmp/demo-env/bin/pip install gradio==6.32.0 huggingface_hub
THREADS=4 /tmp/demo-env/bin/python space/app.py        # then open http://127.0.0.1:7860
```

The app downloads `base-q4_k_m.gguf` and `g1-s0-q4_k_m.gguf` from
[`isHeSatoshi/anthropochrony-tiny-aya-global-g1-gguf`](https://huggingface.co/isHeSatoshi/anthropochrony-tiny-aya-global-g1-gguf)
on first start; set `GGUF_BASE` and `GGUF_G1` to local paths to skip that. `THREADS` sets
the CPU threads and `MAX_RUN_TOKENS` caps the timed run (768 by default). The same folder
builds as a Docker Space (`Dockerfile`), which needs a paid Hugging Face plan.

`space/build_data.py` rebuilds `space/data/` (the benchmark tab) from `results/metatime.json`.
