# Anthropochrony prompt specification

Version v0.2 (2026-11-09; token counts and token IDs checked against the pinned tokenizer; prompt formats mapped to task families). This file is the single source for every prompt string. Code loads prompts from `src/anthropochrony/prompts/*.yaml` generated from this file. Every record stores the `template_id` it used.

## 0. Rules for all prompts

1. Always render with `tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=False)` for `CohereLabs/tiny-aya-global` at the pinned revision, and pass the token IDs to vLLM as a `TokensPrompt` (re-tokenizing rendered text adds a second BOS). Never build prompt strings by hand.
2. The Tiny Aya template always inserts a fixed system turn (safety text, "Your name is Aya", cutoff June 2024) of 359 tokens when there is no system message. Our `system` message lands under a `# Developer Preamble` header (33 tokens) inside that turn. With the sc01 agent card, the system turn is 510 tokens. Count the whole rendered prompt with the tokenizer and store `prompt_tokens_total`.
3. Roles must alternate user, assistant, user. Test feedback is always a **user** turn.
4. Stop tokens: set `stop_token_ids=[6, 261001]` (`<|END_OF_TURN_TOKEN|>` and `<|END_RESPONSE|>`), and check them on 5 samples in stage S0. The generation prompt ends with `<|START_RESPONSE|>`, and a training target ends with `<|END_RESPONSE|><|END_OF_TURN_TOKEN|>`.
5. Estimates are always requested in a **separate conversation** from the coding run. A run never sees its own estimate, except in experiment X7, which tests exactly that.
6. Template sets are split into `train` and `heldout`. Held-out templates never appear in training data.

## 1. System prompt: the agent card

The same system prompt is used to **measure** runs and to **ask for estimates**, so the agent whose time is estimated is the same agent that was measured. For measurement runs the speed card always holds the reference values (`S_ref`, `P_ref` from stage S2).

```text
You are a Python coding agent.
How a run works: you read the task and write a complete solution, then the harness runs the task's tests. If a test fails, you see the failure and write a corrected full solution. You get at most 3 attempts. The run ends when the tests pass or after your third attempt.
A run's time counts from when you start reading the task until the last test run finishes, whether or not the tests pass.
{speed_card}
```

### 1.1 Speed card phrasings (`{speed_card}`)

`S` is the decode speed and `P` the prefill speed, both in tokens/s. Both are formatted to 2 significant figures with thousands separators.

| id | split | text |
|---|---|---|
| sc01 | train | `Your serving speed: you write about {S} tokens per second and read about {P} prompt tokens per second.` |
| sc02 | train | `Throughput on this deployment: {S} output tokens/s, {P} input tokens/s.` |
| sc03 | train | `You are running on hardware that generates about {S} tokens/s (prompt processing about {P} tokens/s).` |
| sc04 | train | `Speed: decode {S} tok/s; prefill {P} tok/s.` |
| sc05 | train | `This server streams your replies at roughly {S} tokens a second and reads prompts at about {P} tokens a second.` |
| sc06 | train | `Measured speed for you here: {S_per_min} output tokens per minute and {P} input tokens per second.` |
| sc07 | heldout | `Benchmark for this deployment: {S} tokens/sec generation, {P} tokens/sec prompt.` |
| sc08 | heldout | `You produce text at {S} tokens per second, and reading input runs at {P} tokens per second.` |

The no-card condition (estimator E0 and model M5) drops the last line of the system prompt.

E1 adds one more line to the system prompt: `When asked how long something will take you, estimate your own time as this model at this speed, not a human's time.`

## 2. Task turn (user message of a coding run)

Generic form. Stage S1 fixes the `{spec}` rendering of each format, and every format ends with the same instruction line.

```text
Task:
{spec}

Write the complete solution in Python in one ```python code block.
```

`{spec}` rendering by format:

| format | families | spec content |
|---|---|---|
| function | FUNC (HumanEval+) | The function stub with its docstring, in a ```python block, followed by `Complete this function.` |
| function-nl | FUNC (MBPP+) | The natural-language description, plus `Your code should pass tests like:` and the first public test assert |
| library | LIB | The instruction prompt (the benchmark's "instruct" variant), listing the allowed libraries |
| class | CLASS | The class skeleton with method docstrings in a ```python block, followed by `Implement every method.` |
| leetcode | ALGO_FN | The problem description, then `Starter code:` and the starter code (`class Solution`) in a ```python block |
| program | ALGO_IO | The problem statement, plus `Read from standard input and write to standard output.` and one sample input/output |
| project | PROJ-SYN, PROJ-8 | The multi-part spec, plus `Put everything in one module named solution.py. Tests will import it.` |

## 3. Feedback turn (user message after a failed attempt)

```text
The tests failed (attempt {j} of 3).
{failure_summary}

Reply with the complete corrected solution in one ```python code block.
```

`failure_summary` takes the first case that applies:
1. No code block: `Your reply had no ```python code block.`
2. Timeout: `The tests timed out after {timeout_s} s.`
3. Error or failure: the first failing test's name and assertion, plus the last 25 lines of the traceback, cut to 1,200 characters. Never include hidden expected outputs beyond what the failing assertion prints.

## 4. Estimation prompts (separate conversation)

User message = task turn (section 2, verbatim) + `\n\n---\n` + question.

### 4.1 Self frame

| id | split | question |
|---|---|---|
| q01 | train | `Before you start: how long will it take you to finish this task, including any fixes after the tests run? Just give the estimate; don't solve it yet.` |
| q02 | train | `How long will this take you? Answer with a time estimate only.` |
| q03 | train | `What's your ETA for finishing this? Don't write the code yet.` |
| q04 | train | `Estimate how long you will need to complete this task. Reply with the estimate only.` |
| q05 | train | `If I give you this task right now, how long until you're done?` |
| q06 | train | `How much time will you spend on this, from reading it to the final test run?` |
| q07 | train | `Give me a time estimate for you to finish this, then stop.` |
| q08 | train | `Roughly how long will it take you to get this passing its tests?` |
| q09 | train | `Quick check before you begin: how long do you expect this to take you?` |
| q10 | train | `How long should I expect to wait for you to complete this?` |
| q11 | train | `Tell me how long this will take you, not a human. Estimate only, no code.` |
| q12 | train | `Predict your total time for this task, including retries.` |
| q13 | train | `I need to plan around you. How long will you take to finish this?` |
| q14 | train | `Estimate your completion time for this task. Do not start the task.` |
| q15 | train | `When will you be done with this if you start now? Give a duration.` |
| q16 | train | `How long will it take you to write and test a solution for this?` |
| q17 | train | `Can you estimate how long you'll take on this one?` |
| q18 | train | `What's your time estimate for this job?` |
| q19 | train | `Before coding, state how long this task will take you.` |
| q20 | train | `How long do you need for this? Give me your best estimate.` |
| q21 | heldout | `How long is this going to take you? Just the estimate, please.` |
| q22 | heldout | `Give me an estimate of the time you need to finish this task, retries included.` |
| q23 | heldout | `Without solving it yet, how long will you need for this?` |
| q24 | heldout | `Estimate how long it will take you, the model, to complete this task.` |
| q25 | heldout | `If you start right now, how long until the tests pass or you give up?` |
| q26 | heldout | `How long would it take you to do this? Answer briefly.` |
| q27 | heldout | `Time check: how long will you be busy with this task?` |
| q28 | heldout | `What's your expected turnaround time for this?` |

### 4.2 Human frame (preservation)

| id | split | question |
|---|---|---|
| h01 | train | `How long would an experienced human developer take to complete this task?` |
| h02 | train | `Estimate how long this would take a professional programmer working by hand.` |
| h03 | train | `If a mid-level software engineer did this, how long would it take them?` |
| h04 | train | `How much time would a human need to implement and test this?` |
| h05 | train | `For a person coding this by hand, what's a realistic time estimate?` |
| h06 | train | `How long would this take a developer on my team?` |
| h07 | train | `Give a time estimate for a human programmer to finish this.` |
| h08 | train | `How long would it take a skilled human to write and test this code?` |
| h09 | heldout | `How long would a typical software developer need for this?` |
| h10 | heldout | `Estimate the time for an experienced engineer to do this manually.` |
| h11 | heldout | `If I did this myself as a competent programmer, how long would it take me?` |
| h12 | heldout | `How long would a human expert take to finish this task?` |

### 4.3 Generic-AI frame (evaluation only; control in the style of Blandfort and Pawar)

| id | question |
|---|---|
| g01 | `How long would an AI coding assistant take to complete this task?` |
| g02 | `Estimate how long a language-model coding agent would need for this.` |
| g03 | `How long would it take an AI model to do this?` |
| g04 | `If an AI agent did this task, how long would it take?` |

### 4.4 Plan frame (evaluation; the training set is used only by ablation M7)

| id | split | question |
|---|---|---|
| p01 | heldout | `Make a plan for building this, with a timeline.` |
| p02 | heldout | `Break this task into steps and say how long each step will take you.` |
| p03 | heldout | `Plan your work on this and tell me when you'll be done.` |
| p04 | heldout | `Outline how you'll approach this and estimate the time for each part.` |
| p05 | train | `Give me a step-by-step plan for this and how long it will take you.` |
| p06 | train | `How would you tackle this, and how long will each part take you?` |
| p07 | train | `Draft a work plan with time estimates for doing this yourself.` |
| p08 | train | `Plan this out and tell me your total time.` |

### 4.5 Wording-sensitivity set (base model and M1 seed 0; robustness of H1 and of the fix)

`{S}` is the reference decode speed. A W-id may also be used as an estimator ID in the analysis.

| id | question |
|---|---|
| w01 | `How long will this take you?` |
| w02 | `How long will it take you, the AI model, to complete this?` |
| w03 | `You generate about {S} tokens per second. How long will it take you to finish this, counting only your own generation and test runs?` |
| w04 | `Ignore how long a human would take. How long will it take you?` |
| w05 | `In seconds, how long will you take to finish this?` |
| w06 | `How many output tokens will you write for this, and how long will that take at {S} tokens per second?` |
| w07 | `Plan this task and give a timeline.` |
| w08 | `How long would a human developer take?` |

## 5. Target answer format (self frame)

Canonical form (one line):

```text
About {T} (likely {LO}–{HI}). Breakdown: ~{TOK} output tokens at {S} tok/s ≈ {GEN}{OVH}; {ATT}. First-attempt pass chance: {PFP}.
```

- `{T}`, `{LO}`, `{HI}`, `{GEN}` come from `fmt_duration(seconds)`. Below 10 s use 2 significant figures (`4.5 s`). From 10 s to under 100 s use whole seconds (`85 s`). From 100 s to under 6,000 s use minutes to 2 significant figures (`1.7 min`, `14 min`). From 6,000 s up use hours to 2 significant figures (`1.7 h`). If `LO` and `HI` share a unit, write it once (`15–45 s`); otherwise write both (`40 s–2.5 min`).
- `{TOK}`: 2 significant figures with thousands separators (`900`, `1,200`).
- `{S}`: as stated in the speed card. For card `sc06` (per minute), convert to per second in the breakdown.
- `{OVH}`: `, plus ~{overhead} reading the task and running tests` when overhead ≥ 5% of `T`; otherwise empty.
- `{ATT}`: `1 attempt expected`, `about 2 attempts expected`, or `all 3 attempts expected`.
- `{PFP}`: rounded to the nearest 10% and clamped. Write `under 10%`, `about 20%` … `about 80%`, or `over 90%`.

Example (S = 35, P = 5,000):

```text
About 30 s (likely 20–55 s). Breakdown: ~900 output tokens at 35 tok/s ≈ 26 s, plus ~4.3 s reading the task and running tests; 1 attempt expected. First-attempt pass chance: about 70%.
```

Label arithmetic, per task i and speed card c:
- `T = median_k t(τ_k; c)`
- `TOK = median_k tokens_k`
- `GEN = TOK / S`
- `overhead = T − GEN`. If overhead < 0.05·T, drop the clause and keep `T`.
- `LO` and `HI`: P0 uses `q10` and `q90` of `t` over the K runs, widened to at least `T/1.25` and `T·1.25`. P1 (model M1b) uses the conformal multipliers from PLAN.md S6.4.
- `PFP = (first_pass_count + 2·p_family) / (K + 2)`, where `p_family` is the mean first-attempt pass rate of the task's family on `train`.

Human-frame targets: the base model's own greedy answer to the same prompt (self-distillation), unchanged. Answers that hit the token cap are not used. Plan-frame targets (M7 only): the base model's greedy plan with every duration phrase removed by the parser (section 6), followed by `Total time for me: {canonical self-frame answer}`.

## 6. Parser specification (`src/anthropochrony/parse.py`)

Order: (1) canonical regex, (2) general regex, (3) LLM judge for the rest. Store `parse_method`.

1. **Canonical regex** (anchored at the start):
   - `^About (?P<pt>[\d.,]+) ?(?P<pu>s|min|h)\b`
   - `likely (?P<lo>[\d.,]+) ?(?P<lu>s|min|h)?\s*[–-]\s*(?P<hi>[\d.,]+) ?(?P<hu>s|min|h)`
   - `~(?P<tok>[\d,]+) output tokens`
   - `(?P<att>1|about 2|all 3) attempts? expected`
   - `pass chance: (?P<pq>under|about|over) (?P<p>\d+)%`
2. **General regex** for free-form answers:
   - numbers: digits, decimals, and number words one to twenty, plus "a couple", "a few" (= 3) and "half an hour";
   - units: sec/second(s)/s, min/minute(s), h/hr(s)/hour(s), day(s), week(s), month(s);
   - ranges: `X–Y unit`, `X to Y unit`, `between X and Y unit`, `X-Y unit`; the point estimate is the geometric mean of the ends;
   - if the answer lists several steps with durations and no total, sum the steps (plans);
   - work-time conversion (primary): 1 day = 8 h, 1 week = 40 h, 1 month = 160 h. Calendar conversion (24 h, 168 h, 730 h) is reported in the appendix;
   - record `units_used` (set) and `has_human_units = any(unit in {h, day, week, month})`.
3. **LLM judge** (only when 1 and 2 fail or disagree): the prompt is in section 7.3, the model is `openai/gpt-oss-120b` through HF Inference Providers, temperature 0, cached.
4. **Audit:** 200 random answers stratified by estimator. Two passes (the regex/judge pipeline vs a hand label by the agent with the rules above), with agreement reported (exact unit match; |log10| difference ≤ 0.05).

## 7. Teacher and judge prompts

### 7.1 Teacher-guessed labels (estimator B4 and training set M2)

The teacher predicts quantities that don't depend on hardware. Code converts them to time with the same formula used for measured labels, so M1 and M2 differ **only** in where the labels come from.

```text
You are predicting the behavior of a specific small AI coding agent on a task. Do not solve the task.
Agent: Tiny Aya Global, an open 3.35B-parameter multilingual chat model, temperature 0.1.
Harness: the agent writes a complete Python solution; tests run; on failure it sees the failure and rewrites the full solution; at most 3 attempts; it stops when tests pass.
Task given to the agent:
<<<
{task_turn}
>>>
Predict for this agent on this task:
- tokens_median: total output tokens across all attempts (median over repeated runs)
- tokens_p10, tokens_p90: 10th and 90th percentile of that total
- attempts_median: 1, 2 or 3
- p_first_pass: probability its first attempt passes all tests
- test_seconds: wall time of one test run
Reply with JSON only: {"tokens_median": int, "tokens_p10": int, "tokens_p90": int, "attempts_median": int, "p_first_pass": float, "test_seconds": float}
```

Model `deepseek-ai/DeepSeek-V4-Pro:deepinfra`, temperature 0, `max_tokens` 300, with reasoning disabled if the provider allows it. Retry once on invalid JSON, then drop the task from M2 and record it.

The teacher does not predict prompt tokens. For its labels, `u_1` is the task turn's token count, and each later attempt adds the family's median feedback-turn length on `train`.

### 7.2 Synthetic project tasks (PROJ-SYN; also the starting point for drafting PROJ-8)

```text
Create one self-contained Python programming task of the "small project" kind.
Category: {category}. Difficulty: {difficulty} (target reference size: {lines} lines).
Constraints: Python 3.10 standard library only; everything in one module named solution.py; deterministic; no network, no sleeping, no reading files outside a pytest tmp_path; all tests must finish in under 5 seconds.
Return exactly three sections:
<spec>A clear task description for a developer (what to build, public functions/classes with signatures, behaviors, edge cases). Do not include the implementation.</spec>
<reference>The complete reference solution.py.</reference>
<tests>A pytest file (8 to 15 tests) that imports from solution and checks the spec'd behavior.</tests>
```

Model `deepseek-ai/DeepSeek-V4-Pro:deepinfra`, temperature 0.7, with categories and sizes drawn from a seeded list. Verification gates live in PLAN.md S1.4 and S1.5. Tests run in the Python 3.10 `exec` environment.

### 7.3 Estimate-extraction judge

```text
Extract the time estimate from an answer. The answer replies to the question shown.
Question: {question}
Answer: <<<{answer}>>>
Rules: If a range is given, lo and hi are its ends and point is their geometric mean. If only per-step durations are given, sum them. Convert with 1 day = 8 h, 1 week = 40 h, 1 month = 160 h. If no duration is stated, has_estimate is false.
Return JSON only: {"has_estimate": bool, "point_seconds": number|null, "lo_seconds": number|null, "hi_seconds": number|null, "units_used": [string], "referent": "self"|"human"|"generic_ai"|"unclear"}
```
