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
    ("g1-kit/card_description", "**Tiny Aya + G1-kit (primary)**", "stated"),
    ("g1-kit/card_description", "G1-kit, stated tokens through the speed formula", "token_reading"),
    ("g1-kit/card_description+sampled_median", "G1-kit, median of 16 sampled answers", "stated"),
    ("g1-kit/card_paper_wording", "G1-kit, the paper's question wording", "stated"),
    ("g1-kit/protocol", "G1-kit, no speed stated", "stated"),
    ("g1-kit/card_full_prompt", "G1-kit, shown the full prompt", "stated"),
    ("g1-kit@base/card_description", "G1-kit predicting the untrained model's runs", "stated"),
    ("g1-kit@base/card_full_prompt", "G1-kit, full prompt, the untrained model's runs", "stated"),
    ("static/text_regressor_full", "TF-IDF regressor on the full prompt", "stated"),
    ("static/embedding_regressor_full", "Sentence-encoder regressor, full prompt", "stated"),
    ("probe/task", "Linear probe, state before answering the full prompt", "stated"),
]
EXECUTORS = {"base": "untrained model", "g1-kit": "fine-tuned model (G1-kit, adapter unmerged)"}
SAMPLED = {
    "card_description": "G1-kit, benchmark question (primary)",
    "card_full_prompt": "G1-kit, shown the full prompt",
    "card_paper_wording": "G1-kit, the paper's question wording",
    "protocol": "G1-kit, no speed stated",
}


def number(value, digits=2):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}" if math.isfinite(value) else "inf"


def change(ratio, up, down):
    """A ratio in words: `20x slower`, `6.3x faster`, or `unchanged` within 1%."""
    if ratio is None:
        return "n/a"
    if abs(math.log(ratio)) < 0.01:
        return "unchanged"
    return f"{ratio:.3g}x {up}" if ratio > 1 else f"{1 / ratio:.3g}x {down}"


def line(label, row):
    low, high = row.get("ci95", {}).get("median_abs_log_error", (None, None))
    interval = f" [{number(low)}, {number(high)}]" if low is not None else ""
    return (
        f"| {label} | {number(row['median_ratio'])}x | "
        f"{number(row['median_abs_log_error'])}{interval} | "
        f"{number(math.exp(row['median_abs_log_error']))}x | {number(row['pearson_log'])} | "
        f"{row['within_2x']:.0%} | {number(row['mae_s'], 1)} | {row['parsed']}/{row['n']} |"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/metatime.json")
    args = parser.parse_args()
    result = json.loads(Path(args.results).read_text())
    systems = result["systems"]

    print(
        "| System | Median estimate/actual | Median abs. log error [95% CI] | Typical factor off | r (log) | Within 2x | MAE (s) | Parsed |"
    )
    print("|---|---|---|---|---|---|---|---|")
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
        if not key.startswith("g1-") or key not in systems or reading not in systems[key]:
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

    print(
        "\n| Telling tasks apart | Pearson r, log durations [95% CI] | Spearman | Pairs ordered correctly | Pearson r, raw seconds |"
    )
    print("|---|---|---|---|---|")
    ranking = [(label, result["published"][model]) for model, label in PUBLISHED.items()]
    ranking += [
        (label, systems[key]["stated"])
        for key, label in (
            ("untrained/card_description", "Tiny Aya, untrained, speed stated"),
            ("g1-kit/card_description", "**Tiny Aya + G1-kit (primary)**"),
            ("g1-kit/card_full_prompt", "G1-kit, shown the full prompt"),
        )
        if key in systems
    ]
    for label, row in ranking:
        low, high = row["ci95"]["pearson_log"]
        print(
            f"| {label} | {number(row['pearson_log'])} [{number(low)}, {number(high)}] | "
            f"{number(row['spearman'])} | {row['ordering_accuracy']:.0%} | "
            f"{number(row['pearson_raw'])} |"
        )

    after = result.get("published_context", {}).get("single_turn_after_completing", {})
    if after:
        print(
            "\n| After completing the task (published, context only) | Median estimate/actual | Median abs. log error | Typical factor off | r (log) | Within 2x |"
        )
        print("|---|---|---|---|---|---|")
        for model, label in PUBLISHED.items():
            row = after[model]
            print(
                f"| {label.split(' (')[0]} | {number(row['median_ratio'])}x | "
                f"{number(row['median_abs_log_error'])} | "
                f"{number(math.exp(row['median_abs_log_error']))}x | "
                f"{number(row['pearson_log'])} | {row['within_2x']:.0%} |"
            )

    shown = [(label, result["published"][model]) for model, label in PUBLISHED.items()]
    for key, label in (
        ("g1-kit/card_description", "Tiny Aya + G1-kit"),
        ("g1-kit@base/card_description", "G1-kit predicting the untrained model's runs"),
    ):
        if key in systems:
            shown.append((label, systems[key]["stated"]))
    if len(shown) > 2:
        names = sorted(shown[0][1]["by_category"])
        print(
            "\n| Median abs. log error by task category | "
            + " | ".join(f"{name} ({shown[0][1]['by_category'][name]['n']})" for name in names)
            + " |"
        )
        print("|---|" + "---|" * len(names))
        for label, row in shown:
            cells = " | ".join(
                number(row["by_category"][name]["median_abs_log_error"]) for name in names
            )
            print(f"| {label} | {cells} |")

    # The same question with other speeds stated, per estimator, slowest machine first.
    measured = {"g1-kit/card_description": "G1-kit", "untrained/card_description": "Untrained"}
    sweep = sorted(
        (
            (
                measured.get(key) or ("G1-kit" if key.startswith("g1-kit@") else "Untrained"),
                key,
                row,
            )
            for key, row in systems.items()
            if row.get("replayed_at_stated_speed") or key in measured
        ),
        key=lambda item: (item[0], item[2]["stated_speed"]["S"]),
    )
    if any(row.get("replayed_at_stated_speed") for _, _, row in sweep):
        print(
            "\n| Estimator | Stated speed (tokens/s writing, reading) | Median target (s) | Median estimate (s) | Median estimate/actual | Median abs. log error | r (log) | Within 2x | Tokens through the formula | Stated writing speed vs this machine | Estimate vs its estimate at this machine's speed (median per task) |"
        )
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for name, _, row in sweep:
            stated, tokens = row["stated"], row.get("token_reading")
            note = "" if row["replayed_at_stated_speed"] else " (this machine, measured)"
            print(
                f"| {name} | {row['stated_speed']['S']:g}, {row['stated_speed']['P']:g}{note} | "
                f"{number(row['median_target_seconds'], 1)} | "
                f"{number(row['median_estimate_seconds'], 1)} | "
                f"{number(stated['median_ratio'])}x | {number(stated['median_abs_log_error'])} | "
                f"{number(stated['pearson_log'])} | {stated['within_2x']:.0%} | "
                f"{number(tokens['median_abs_log_error']) if tokens else 'n/a'} | "
                + (
                    f"{change(move['slowdown_stated'], 'slower', 'faster')} | "
                    f"{change(move['median_estimate_ratio'], 'longer', 'shorter')} |"
                    if (move := row.get("versus_measured_speed"))
                    else "same | same |"
                )
            )

    # Stated time against stated tokens, and response-length drift, per scoring target.
    extra = [
        ("g1-kit", "G1-kit, its own runs"),
        ("g1-kit@base", "G1-kit, the untrained model's runs"),
    ]
    drift = result.get("pool_validation", {}).get("length_drift", {})
    if any(f"{name}/card_description" in systems for name, _ in extra):
        print(
            "\n| Adapter | Stated time: log error | r (log) | Within 2x | Stated tokens through the formula: log error | r (log) | Stated time agrees with its own token counts (within 10%) | Full prompt, stated time: log error | Full prompt, tokens through the formula | Response-length drift |"
        )
        print("|---|---|---|---|---|---|---|---|---|---|")
        for name, label in extra:
            short = systems.get(f"{name}/card_description", {})
            full = systems.get(f"{name}/card_full_prompt", {})
            if not all("token_reading" in system for system in (short, full)):
                continue
            shift = drift.get(name.split("@")[0], {}).get("median_log_ratio_to_base")
            stated, tokens = short["stated"], short["token_reading"]
            agree = short["stated_time_matches_stated_tokens"]
            print(
                f"| {label} | {number(stated['median_abs_log_error'])} | "
                f"{number(stated['pearson_log'])} | {stated['within_2x']:.0%} | "
                f"{number(tokens['median_abs_log_error'])} | {number(tokens['pearson_log'])} | "
                f"{agree['within_10_percent']}/{agree['answers']} | "
                f"{number(full['stated']['median_abs_log_error'])} | "
                f"{number(full['token_reading']['median_abs_log_error'])} | "
                f"{'n/a' if shift is None else f'{math.expm1(shift):+.1%}'} |"
            )

    print("\nExecutors:")
    for name, row in result["executors"].items():
        print(
            f"- {EXECUTORS.get(name, name)}: runs per task {row['runs_per_task']}, median {row['median_seconds']:.1f} s "
            f"(range {row['min_seconds']:.1f} to {row['max_seconds']:.1f}), median response "
            f"{row['median_response_tokens']:.0f} tokens, capped runs {row['capped_runs']}, "
            f"run-to-run abs. log error {row['retest_ceiling_abs_log_error']:.3f}, measured "
            f"length through the speed formula vs wall time "
            f"{row['measured_length_through_formula']['median_abs_log_error']:.3f}, speed cards "
            f"{[(round(c['S'], 1), round(c['P'])) for c in row['speed_cards']]}"
        )
    ranged = [
        (key, label, systems[key])
        for key, label in (
            ("g1-kit/card_description", "G1-kit, benchmark question (primary), its own runs"),
            ("g1-kit/card_full_prompt", "G1-kit, shown the full prompt, its own runs"),
            (
                "g1-kit@base/card_description",
                "G1-kit, benchmark question, the untrained model's runs",
            ),
            (
                "g1-kit@base/card_full_prompt",
                "G1-kit, shown the full prompt, the untrained model's runs",
            ),
        )
        if "conformal_80" in systems.get(key, {})
    ]
    if ranged:
        print(
            "\n| Range calibrated on held-out pool prompts (80% target) | Calibration prompts | q (log units) | Range: stated time divided and multiplied by | Benchmark measured times inside | 10-90% range of 16 samples covers |"
        )
        print("|---|---|---|---|---|---|")
        for key, label, system in ranged:
            row = system["conformal_80"]
            sampled = systems.get(f"{key}+sampled_median", {}).get("range_80_coverage")
            print(
                f"| {label} | {row['calibration_n']} | {number(row['q'])} | "
                f"{number(row['factor'])}x | {row['coverage']:.0%} | "
                f"{'n/a' if sampled is None else f'{sampled:.0%}'} |"
            )

    print("\nSampled answers:")
    for key, row in systems.items():
        if "range_80_coverage" in row:
            variant = key.split("/")[1].removesuffix("+sampled_median")
            print(
                f"- {SAMPLED.get(variant, key)}: {row['draws']} draws, 10-90% range covers "
                f"{row['range_80_coverage']:.0%}"
            )
    if "pool_validation" in result:
        check = result["pool_validation"]
        print(
            f"\nHeld-out pool prompts ({check['prompts']}; run-to-run abs. log error "
            f"{check['retest_ceiling_abs_log_error']:.3f}):\n"
        )
        print(
            "| Estimator | Median estimate/actual | Median abs. log error | r (log) | Within 2x | Parsed | 10-90% range of the samples covers |"
        )
        print("|---|---|---|---|---|---|---|")
        seen = set()
        # The rows that never ask the model are listed once, on the primary adapter's targets.
        ordered = sorted(check["estimators"].items(), key=lambda item: item[0] != "g1-kit")
        for name, rows in ordered:
            for key, row in rows.items():
                label = key if key.startswith("static/") else f"{name}/{key}"
                if label in seen:
                    continue
                seen.add(label)
                print(
                    f"| {label} | {number(row['median_ratio'])}x | "
                    f"{number(row['median_abs_log_error'])} | {number(row['pearson_log'])} | "
                    f"{row['within_2x']:.0%} | {row['parsed']}/{row['n']} | "
                    + (f"{row['range_80_coverage']:.0%}" if "range_80_coverage" in row else "")
                    + " |"
                )
        print("\nResponse length on the held-out pool prompts, against the untrained model:")
        for name, row in check["length_drift"].items():
            print(
                f"- {name}: response length vs untrained {math.expm1(row['median_log_ratio_to_base']):+.1%} "
                f"(median), median abs. log ratio {row['median_abs_log_ratio_to_base']:.3f}"
            )


if __name__ == "__main__":
    main()
