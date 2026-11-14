# Using Anthropochrony in your own agent

Anthropochrony is a fine-tuned Tiny Aya Global that says how long it will take *itself* to solve
a Python coding task on *your* machine. This page shows how to run it locally and ask it before a
task starts. Everything here runs on your machine with llama.cpp. No code from the model is run.

## 1. Get the model

```sh
hf download isHeSatoshi/anthropochrony-tiny-aya-global-gguf m1-s0-q4_k_m.gguf --local-dir models
```

`m1-s0-q4_k_m.gguf` is the tuned model (M1-S0). The same repo also has `base-q4_k_m.gguf`, the
untuned model, for comparison.

## 2. Start llama-server

Use llama.cpp's `llama-server` (the project pins tag `b12110`, see `configs/pins.yaml`):

```sh
llama-server -m models/m1-s0-q4_k_m.gguf --port 8082 -c 8192
```

Add `-ngl 99` to run on a GPU. Keep `-c 8192`: the client stops when fewer than 256 context tokens
remain. Only local `http://127.0.0.1:<port>` or `http://localhost:<port>` servers are accepted.

## 3. Tell it how fast your machine is

The estimate depends on two speeds: decode `S` and prefill `P`, in tokens per second. Measure
them once with llama-bench:

```sh
anthropochrony speedcard --llama-bench /path/to/llama-bench --gguf models/m1-s0-q4_k_m.gguf
# add --gpu-layers 99 if the server runs on a GPU
```

Use the same llama.cpp build and the same number of GPU layers as the server. A CUDA build
measures different speeds than a CPU-only build even with `--gpu-layers 0`, because it can still
send large prompt batches to the GPU.

It prints JSON such as `{"id": "demo-llamacpp", "S": ..., "P": ...}`. Both speeds are rounded to
two significant figures, as in training.

## 4. Ask for an estimate (CLI)

```sh
anthropochrony estimate --task task.md --speed S,P
anthropochrony estimate --text "Write a function add(a, b) ..." --speed S,P --human
```

Instead of `--speed`, pass `--llama-bench PATH --gguf PATH` to measure the speed on the spot.
Other options: `--server` (default `http://127.0.0.1:8082`), `--family` (default: inferred from
the text; one of `FUNC`, `LIB`, `ALGO_FN`, `ALGO_IO`, `CLASS`), and `--json`.

The output is one line:

```text
~<time> on this machine (80% range <low> to <high>; ~<N> output tokens over <A> attempts)
```

With `--human`, a second line gives the same model's guess for a human developer, so you can see
both clocks. With `--json` you get `{"self": {...}, "human": {...} or null}`. The `self` object has
`seconds`, `low`, `high`, `tokens`, `attempts`, `verbal_seconds`, `family`, `card`, `method`, and
`raw_text`.

## 5. Ask for an estimate (Python)

```python
from anthropochrony.selftime import SelfTimeEstimator

estimator = SelfTimeEstimator.from_llama_server("http://127.0.0.1:8082", speed=(S, P))
# or: from_llama_server(url, llama_bench="/path/to/llama-bench", gguf="models/m1-s0-q4_k_m.gguf")

estimate = estimator.estimate(open("task.md").read())
print(estimate.summary())
print(estimate.seconds, estimate.low, estimate.high)

timeout_s = estimator.tripwire_seconds(estimate)  # = estimate.high * 1.5 by default
human = estimator.human_estimate(open("task.md").read())
print(human.summary())  # seconds is None if the model answered with its own time instead
```

`SelfTimeEstimator(backend, card=None, *, latency_s=0.0, ...)` also accepts any object with a
`generate(messages)` method (and `calibrate()` if you do not pass a `SpeedCard`). That makes it
easy to test with a fake backend.

### How the number is made

The model writes a short answer that includes its predicted output tokens and attempts. The
estimator does not trust the model's arithmetic. It feeds those two numbers into the exact timing
formula (`src/anthropochrony/timemodel.py`) with your `S` and `P`, plus train-only priors for
retry feedback and test time per task family. This is the paper's **M1-Struct** method, with no
per-call network latency because the server is local.

The 80% range is the point times or divided by **2.087**. That factor was fit on the validation
split with split-conformal calibration (`results/extra-analyses.json`,
`conformal_calibration.struct_factor`). The priors and factor ship inside the package
(`src/anthropochrony/selftime_defaults.json`), so a plain install works without the repo.

If the answer has no token count, the estimator falls back to the model's verbal time and widens
its verbal range by the validation-fit factor 1.752 (`conformal_calibration.cqr_factor`, the
paper's M1b). `method` then says `verbal-fallback`.

## 6. Use it as a timeout (tripwire)

`examples/agent_with_tripwire.py` asks for an estimate, sets a timeout at `k` times the upper
bound (default `k = 1.5`), runs your own agent command with the task on stdin, and reports
estimate against actual time:

```sh
python examples/agent_with_tripwire.py --task task.md --speed S,P --cmd "python my_agent.py"
```

It exits with code 124 if the tripwire fired.

Why 1.5: in the paper's simulation on 2,040 bf16 `test` runs, a timeout at 1.0x the conformal
upper bound stopped 1.33% of the runs that would have passed and saved 19.9% of the time spent on
failing runs (`results/simulations.json`, `M1-Struct+CP|own`, `tripwire`, multiplier 1). On
the real 4-bit runs (100 tasks each on a desktop CPU and a desktop GPU, 34 passing runs on each),
1.0x would have stopped 14.7% (CPU) and 17.6% (GPU) of the passing runs, and 1.5x stopped none
(`results/extra-analyses.json`, `x10_deployment_checks.*.tripwire_false_abort_rate`).

## 7. Limits

- **Python coding tasks only.** The model was trained on function, library, algorithm, and class
  tasks.
- **Tiny Aya's own time.** It predicts how long *this model* takes through llama.cpp on your
  machine, not how long another model or a person takes. Point `--cmd` at an agent that uses
  this model.
- **Quantized model.** The released GGUF is `Q4_K_M`. Most research numbers come from the bf16
  model. With this GGUF on an 8-thread desktop CPU (speed card S=20, P=200), the local M1-Struct
  estimate was within 2x of the real time on 63 of 100 held-out tasks; on an RTX 4060 Ti with
  every layer offloaded (S=100, P=7,400), on 60 of the same 100. Its estimates shrank 5.3x from
  CPU to GPU against a real speedup of 6.0x, and matched each task's speedup within 2x for 82 of
  100 (`results/extra-analyses.json`, `x10_cpu_deployment_struct`, `x10_gpu_deployment_struct`,
  `x10_hardware_tracking`; `paper/article.md`, section 7.7). On both machines it ran low: the
  geometric mean of estimate / actual was 0.65 (CPU) and 0.73 (GPU).
- **The range is calibrated, not guaranteed.** The 80% range covered 75.8% of fresh runs on
  held-out tasks from familiar families, and 64.9% on an unseen task family (`CLASS`)
  (`results/extra-analyses.json`, `x1_interval_calibration.fresh_trajectories`;
  `paper/article.md`, section 7.4). On the real 4-bit runs it covered 64% (CPU) and 63% (GPU)
  (`x10_deployment_checks`).
- **The human estimate can be missing.** On the real 4-bit runs the model answered the
  human-developer question with its own token breakdown on 29% (CPU) and 20% (GPU) of tasks
  (`x10_deployment_checks.*.human_frame_own_time_format_rate`). `human_estimate()` then returns
  `seconds=None` with `own_time_format=True` rather than passing its own time off as a human's.
- **Pasted tasks are approximated.** The task's prompt length is estimated from its characters
  (about 3.6 per token). Pass `task_tokens=` to `estimate()` if you know the exact count.

## License

The model weights are derived from Tiny Aya Global and are released under CC-BY-NC-4.0 plus the
Cohere Labs Acceptable Use Policy (non-commercial use). The code in this repository is
Apache-2.0 (`pyproject.toml`).
