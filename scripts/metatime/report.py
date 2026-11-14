"""Print the head-to-head tables in Markdown from `results/metatime.json`.

The documentation tables are pasted from this output, so every number in them traces to
the results file.
"""

import argparse
import json
import math
from pathlib import Path

PUBLISHED = {"gpt-5.5": "GPT-5.5 (published)", "claude-opus-4-8": "Claude Opus 4.8 (published)"}
# (key, label, which reading of the answers)
ROWS = [
    ("untrained/protocol", "Tiny Aya, untrained, benchmark question as is", "stated"),
    ("untrained/card_description", "Tiny Aya, untrained, speed stated", "stated"),
    ("untrained/card_fewshot", "Tiny Aya, untrained, speed stated, 5 measured examples", "stated"),
    ("static/pool_median_constant", "Same constant for every task (no model call)", "stated"),
    ("static/text_regressor_description", "TF-IDF regressor on the description", "stated"),
    ("static/embedding_regressor_description", "Sentence-encoder regressor, description", "stated"),
    ("probe/question", "Linear probe on the untrained hidden state", "stated"),
    ("g1-s0/card_description", "**Tiny Aya + G1 (primary)**", "stated"),
    ("g1-s0/card_description", "G1, stated tokens through the speed formula", "token_reading"),
    ("g1-s0/card_description+sampled_median", "G1, median of 16 sampled answers", "stated"),
    ("g1-s0/card_paper_wording", "G1, the paper's question wording", "stated"),
    ("g1-s0/protocol", "G1, no speed stated", "stated"),
    ("g1-s0/card_full_prompt", "G1, shown the full prompt", "stated"),
    ("g1-s0@base/card_description", "G1 predicting the untrained model's runs", "stated"),
    ("static/text_regressor_full", "TF-IDF regressor on the full prompt", "stated"),
    ("static/embedding_regressor_full", "Sentence-encoder regressor, full prompt", "stated"),
    ("probe/task", "Linear probe, state before answering the full prompt", "stated"),
]


def number(value, digits=2):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}" if math.isfinite(value) else "inf"


def line(label, row):
    low, high = row.get("ci95", {}).get("median_abs_log_error", (None, None))
    interval = f" [{number(low)}, {number(high)}]" if low is not None else ""
    return (
        f"| {label} | {number(row['median_ratio'])}x | "
        f"{number(row['median_abs_log_error'])}{interval} | {number(row['pearson_log'])} | "
        f"{row['within_2x']:.0%} | {number(row['mae_s'], 1)} | {row['parsed']}/{row['n']} |"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/metatime.json")
    args = parser.parse_args()
    result = json.loads(Path(args.results).read_text())
    systems = result["systems"]

    print(
        "| System | Median estimate/actual | Median abs. log error [95% CI] | r (log) | Within 2x | MAE (s) | Parsed |"
    )
    print("|---|---|---|---|---|---|---|")
    for model, label in PUBLISHED.items():
        print(line(label, result["published"][model]))
    for key, label, reading in ROWS:
        if key in systems and reading in systems[key]:
            print(line(label, systems[key][reading]))

    print(
        "\n| Comparison (ours minus theirs, median abs. log error) | Difference | Paired 95% CI | Tasks where ours is closer |"
    )
    print("|---|---|---|---|")
    for key, label, reading in ROWS:
        if not key.startswith("g1-s0") or key not in systems or reading not in systems[key]:
            continue
        for model, name in PUBLISHED.items():
            versus = systems[key][reading]["versus"][model]
            print(
                f"| {label} vs {name.split(' (')[0]} | {versus['difference']:+.2f} | "
                f"[{versus['ci95'][0]:+.2f}, {versus['ci95'][1]:+.2f}] | "
                f"{versus['tasks_a_closer']}/{versus['tasks']} |"
            )

    print(
        "\n| Context (uses the benchmark's own outcomes) | Log error as asked | Own-median constant | Rescaled by one constant | r (log) |"
    )
    print("|---|---|---|---|---|")
    context = [(label, result["published"][model]) for model, label in PUBLISHED.items()]
    context += [
        (label, systems[key][reading])
        for key, label, reading in ROWS
        if key in systems and reading in systems[key] and "context" in systems[key][reading]
    ]
    for label, row in context:
        extra = row["context"]
        print(
            f"| {label} | {number(row['median_abs_log_error'])} | "
            f"{number(extra['own_median_leave_one_out']['median_abs_log_error'])} | "
            f"{number(extra['rescaled_leave_one_out']['median_abs_log_error'])} | "
            f"{number(row['pearson_log'])} |"
        )

    # The same question with other speeds stated, per estimator, slowest machine first.
    measured = {"g1-s0/card_description": "G1", "untrained/card_description": "Untrained"}
    sweep = sorted(
        (
            (measured.get(key) or ("G1" if key.startswith("g1-s0@") else "Untrained"), key, row)
            for key, row in systems.items()
            if row.get("replayed_at_stated_speed") or key in measured
        ),
        key=lambda item: (item[0], item[2]["stated_speed"]["S"]),
    )
    if any(row.get("replayed_at_stated_speed") for _, _, row in sweep):
        print(
            "\n| Estimator | Stated speed (tokens/s writing, reading) | Median target (s) | Median estimate (s) | Median estimate/actual | Median abs. log error | r (log) | Within 2x | Tokens through the formula |"
        )
        print("|---|---|---|---|---|---|---|---|---|")
        for name, _, row in sweep:
            stated, tokens = row["stated"], row.get("token_reading")
            note = "" if row["replayed_at_stated_speed"] else " (this machine, measured)"
            print(
                f"| {name} | {row['stated_speed']['S']:g}, {row['stated_speed']['P']:g}{note} | "
                f"{number(row['median_target_seconds'], 1)} | "
                f"{number(row['median_estimate_seconds'], 1)} | "
                f"{number(stated['median_ratio'])}x | {number(stated['median_abs_log_error'])} | "
                f"{number(stated['pearson_log'])} | {stated['within_2x']:.0%} | "
                f"{number(tokens['median_abs_log_error']) if tokens else 'n/a'} |"
            )

    print("\nExecutors:")
    for name, row in result["executors"].items():
        print(
            f"- {name}: runs per task {row['runs_per_task']}, median {row['median_seconds']:.1f} s "
            f"(range {row['min_seconds']:.1f} to {row['max_seconds']:.1f}), median response "
            f"{row['median_response_tokens']:.0f} tokens, capped runs {row['capped_runs']}, "
            f"run-to-run abs. log error {row['retest_ceiling_abs_log_error']:.3f}, speed cards "
            f"{[(round(c['S'], 1), round(c['P'])) for c in row['speed_cards']]}"
        )
    for key, row in systems.items():
        if "range_80_coverage" in row:
            print(
                f"- {key}: {row['draws']} draws, 10-90% range covers {row['range_80_coverage']:.0%}"
            )
    if "pool_validation" in result:
        print("\nPool validation:", json.dumps(result["pool_validation"], indent=1)[:3000])


if __name__ == "__main__":
    main()
