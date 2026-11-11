"""Build, verify and write the FRESH task set (30 original tasks outside the existing pool).

Run from the repository root:
    . ./env.sh && PYTHONPATH=src /home/adi/.local/share/anthropochrony-envs/stack/bin/python scripts/build_fresh_tasks.py
"""

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIZES = ("small", "medium", "large")
TASKS = []


def task(slug, size, entry, spec, ref, tests):
    TASKS.append(
        {
            "slug": slug,
            "size": size,
            "entry_point": entry,
            "spec": spec.strip(),
            "reference": ref.strip() + "\n",
            "tests": tests.strip() + "\n",
        }
    )


# ---------------------------------------------------------------- SMALL

task(
    "semver-compare",
    "small",
    "compare_versions",
    r"""
Write a function `compare_versions(a, b)` that compares two version strings and returns -1 if `a` is older than `b`, 0 if they are equal, and 1 if `a` is newer.

A version has the form `MAJOR.MINOR.PATCH` (three non-negative integers separated by dots), optionally followed by `-` and a pre-release tag, for example `1.4.0` or `2.0.0-beta`. Rules:
- The three numbers are compared numerically, left to right (so `1.10.0` is newer than `1.9.0`).
- If the numbers are equal, a version with a pre-release tag is older than the same version without one.
- If both have pre-release tags and the numbers are equal, compare the tags as plain strings.

Examples:
compare_versions("1.2.3", "1.2.4") == -1
compare_versions("1.10.0", "1.9.9") == 1
compare_versions("2.0.0-rc", "2.0.0") == -1
compare_versions("2.0.0-alpha", "2.0.0-beta") == -1
compare_versions("3.1.4", "3.1.4") == 0
""",
    r"""
def compare_versions(a, b):
    def parse(v):
        core, _, tag = v.partition("-")
        return tuple(int(x) for x in core.split(".")), tag

    (ca, ta), (cb, tb) = parse(a), parse(b)
    if ca != cb:
        return -1 if ca < cb else 1
    if ta == tb:
        return 0
    if not ta or not tb:
        return 1 if not ta else -1
    return -1 if ta < tb else 1
""",
    r"""
import unittest


class TestCompare(unittest.TestCase):
    def test_patch(self):
        self.assertEqual(compare_versions("1.2.3", "1.2.4"), -1)
        self.assertEqual(compare_versions("1.2.4", "1.2.3"), 1)

    def test_numeric_not_lexicographic(self):
        self.assertEqual(compare_versions("1.10.0", "1.9.9"), 1)
        self.assertEqual(compare_versions("0.2.0", "0.10.0"), -1)

    def test_major_dominates(self):
        self.assertEqual(compare_versions("2.0.0", "1.99.99"), 1)

    def test_equal(self):
        self.assertEqual(compare_versions("3.1.4", "3.1.4"), 0)
        self.assertEqual(compare_versions("1.0.0-rc", "1.0.0-rc"), 0)

    def test_prerelease_older_than_release(self):
        self.assertEqual(compare_versions("2.0.0-rc", "2.0.0"), -1)
        self.assertEqual(compare_versions("2.0.0", "2.0.0-rc"), 1)

    def test_prerelease_string_order(self):
        self.assertEqual(compare_versions("2.0.0-alpha", "2.0.0-beta"), -1)
        self.assertEqual(compare_versions("2.0.0-rc", "2.0.0-beta"), 1)

    def test_numbers_beat_prerelease(self):
        self.assertEqual(compare_versions("1.0.1-alpha", "1.0.0"), 1)
        self.assertEqual(compare_versions("1.0.0-zzz", "1.0.1-aaa"), -1)
""",
)

task(
    "dict-diff",
    "small",
    "diff_dicts",
    r"""
Write a function `diff_dicts(old, new)` that compares two flat dictionaries and returns a dictionary with exactly three keys:
- `"added"`: a dict of the keys present only in `new`, mapped to their values in `new`.
- `"removed"`: a dict of the keys present only in `old`, mapped to their values in `old`.
- `"changed"`: a dict of the keys present in both whose values differ (compared with `!=`), mapped to the tuple `(old_value, new_value)`.

Keys with equal values do not appear anywhere in the result. Neither input may be modified. All three sub-dicts are always present, even when empty.

Example:
diff_dicts({"a": 1, "b": 2, "c": 3}, {"b": 2, "c": 4, "d": 5}) == {"added": {"d": 5}, "removed": {"a": 1}, "changed": {"c": (3, 4)}}
""",
    r"""
def diff_dicts(old, new):
    return {
        "added": {k: v for k, v in new.items() if k not in old},
        "removed": {k: v for k, v in old.items() if k not in new},
        "changed": {k: (old[k], new[k]) for k in old if k in new and old[k] != new[k]},
    }
""",
    r"""
import unittest


class TestDiff(unittest.TestCase):
    def test_example(self):
        self.assertEqual(
            diff_dicts({"a": 1, "b": 2, "c": 3}, {"b": 2, "c": 4, "d": 5}),
            {"added": {"d": 5}, "removed": {"a": 1}, "changed": {"c": (3, 4)}},
        )

    def test_identical(self):
        d = {"x": 1, "y": [1, 2]}
        self.assertEqual(diff_dicts(d, dict(d)), {"added": {}, "removed": {}, "changed": {}})

    def test_empty_old(self):
        self.assertEqual(diff_dicts({}, {"a": 1}), {"added": {"a": 1}, "removed": {}, "changed": {}})

    def test_empty_new(self):
        self.assertEqual(diff_dicts({"a": 1}, {}), {"added": {}, "removed": {"a": 1}, "changed": {}})

    def test_type_change_counts(self):
        self.assertEqual(diff_dicts({"a": 1}, {"a": "1"})["changed"], {"a": (1, "1")})

    def test_none_values(self):
        r = diff_dicts({"a": None}, {"a": 0, "b": None})
        self.assertEqual(r["changed"], {"a": (None, 0)})
        self.assertEqual(r["added"], {"b": None})

    def test_inputs_untouched(self):
        old, new = {"a": 1}, {"a": 2}
        diff_dicts(old, new)
        self.assertEqual((old, new), ({"a": 1}, {"a": 2}))

    def test_nested_values_compared_whole(self):
        r = diff_dicts({"a": {"k": 1}}, {"a": {"k": 2}})
        self.assertEqual(r["changed"], {"a": ({"k": 1}, {"k": 2})})
""",
)

task(
    "parse-duration",
    "small",
    "parse_duration",
    r"""
Write a function `parse_duration(text)` that converts a compact duration string into a whole number of seconds.

The string is one or more segments, each an unsigned integer immediately followed by a unit letter: `d` (86400 seconds), `h` (3600), `m` (60) or `s` (1). Segments may appear in any order and a unit may be repeated; all segments are summed. No spaces or other characters are allowed.

Raise `ValueError` if the string is empty, contains anything other than valid segments (including a number with no unit, a unit with no number, spaces, signs, decimals, uppercase units, or unknown units).

Examples:
parse_duration("1h30m") == 5400
parse_duration("45s") == 45
parse_duration("2d") == 172800
parse_duration("1m1m") == 120
parse_duration("90") raises ValueError
""",
    r"""
import re


def parse_duration(text):
    if not re.fullmatch(r"(\d+[dhms])+", text):
        raise ValueError(f"bad duration: {text!r}")
    units = {"d": 86400, "h": 3600, "m": 60, "s": 1}
    return sum(int(n) * units[u] for n, u in re.findall(r"(\d+)([dhms])", text))
""",
    r"""
import unittest


class TestDuration(unittest.TestCase):
    def test_single_units(self):
        self.assertEqual(parse_duration("45s"), 45)
        self.assertEqual(parse_duration("3m"), 180)
        self.assertEqual(parse_duration("2h"), 7200)
        self.assertEqual(parse_duration("2d"), 172800)

    def test_combined(self):
        self.assertEqual(parse_duration("1h30m"), 5400)
        self.assertEqual(parse_duration("1d1h1m1s"), 86400 + 3600 + 60 + 1)

    def test_any_order(self):
        self.assertEqual(parse_duration("10s2m"), 130)

    def test_repeated_units_sum(self):
        self.assertEqual(parse_duration("1m1m"), 120)
        self.assertEqual(parse_duration("30s45s"), 75)

    def test_large_and_zero(self):
        self.assertEqual(parse_duration("0s"), 0)
        self.assertEqual(parse_duration("100m"), 6000)

    def test_missing_unit(self):
        for bad in ("90", "1h30"):
            with self.assertRaises(ValueError):
                parse_duration(bad)

    def test_empty_and_garbage(self):
        for bad in ("", "h", "1x", "1 h", "-5s", "1.5h", "1H", "abc"):
            with self.assertRaises(ValueError, msg=bad):
                parse_duration(bad)
""",
)

task(
    "markdown-table",
    "small",
    "render_table",
    r"""
Write a function `render_table(headers, rows)` that renders a GitHub-style Markdown table as a string.

- `headers` is a list of column titles; `rows` is a list of lists, each with the same number of cells as `headers`. Every cell is converted with `str()`.
- Each column's width is the longest cell text in that column (including the header), but at least 3.
- Every cell is left-aligned and padded with spaces on the right to the column width.
- A row is rendered as `"| "` + cells joined by `" | "` + `" |"`.
- The second line is the separator: for each column, `-` repeated to the column width, joined and wrapped the same way as a row.
- Lines are joined with `"\n"`, with no trailing newline. With no rows, only the header and separator lines are returned.

Example:
render_table(["name", "qty"], [["apple", 3], ["kiwi", 12]]) ==
"| name  | qty |\n| ----- | --- |\n| apple | 3   |\n| kiwi  | 12  |"
""",
    r"""
def render_table(headers, rows):
    cells = [[str(c) for c in r] for r in [headers, *rows]]
    widths = [max(3, *(len(r[i]) for r in cells)) for i in range(len(headers))]

    def fmt(row):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |"

    separator = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([fmt(cells[0]), separator, *map(fmt, cells[1:])])
""",
    r"""
import unittest


class TestTable(unittest.TestCase):
    def test_example(self):
        self.assertEqual(
            render_table(["name", "qty"], [["apple", 3], ["kiwi", 12]]),
            "| name  | qty |\n| ----- | --- |\n| apple | 3   |\n| kiwi  | 12  |",
        )

    def test_no_rows(self):
        self.assertEqual(render_table(["a", "bb"], []), "| a   | bb  |\n| --- | --- |")

    def test_min_width_three(self):
        self.assertEqual(render_table(["x"], [["y"]]), "| x   |\n| --- |\n| y   |")

    def test_header_wider_than_cells(self):
        self.assertEqual(
            render_table(["description"], [["a"]]),
            "| description |\n| ----------- |\n| a           |",
        )

    def test_non_strings(self):
        out = render_table(["v"], [[1.5], [None], [True]]).split("\n")
        self.assertEqual(out[2:], ["| 1.5  |", "| None |", "| True |"])

    def test_no_trailing_newline(self):
        self.assertFalse(render_table(["a"], [["b"]]).endswith("\n"))

    def test_line_count(self):
        out = render_table(["a", "b", "c"], [[1, 2, 3]] * 4)
        self.assertEqual(len(out.split("\n")), 6)
        self.assertTrue(all(line.startswith("| ") and line.endswith(" |") for line in out.split("\n")))
""",
)

task(
    "length-convert",
    "small",
    "convert_length",
    r"""
Write a function `convert_length(value, from_unit, to_unit)` that converts a length between units and returns a float rounded to 6 decimal places (with `round(x, 6)`).

Supported units and their size in metres: `mm` = 0.001, `cm` = 0.01, `m` = 1, `km` = 1000, `in` = 0.0254, `ft` = 0.3048, `yd` = 0.9144, `mi` = 1609.344. Unit names are case-sensitive. Raise `ValueError` if either unit is not supported. Negative values are allowed and converted normally.

Examples:
convert_length(1, "km", "m") == 1000.0
convert_length(12, "in", "ft") == 1.0
convert_length(1, "mi", "km") == 1.609344
""",
    r"""
FACTORS = {"mm": 0.001, "cm": 0.01, "m": 1, "km": 1000, "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.344}


def convert_length(value, from_unit, to_unit):
    if from_unit not in FACTORS or to_unit not in FACTORS:
        raise ValueError("unsupported unit")
    return round(value * FACTORS[from_unit] / FACTORS[to_unit], 6)
""",
    r"""
import unittest


class TestLength(unittest.TestCase):
    def test_metric(self):
        self.assertEqual(convert_length(1, "km", "m"), 1000.0)
        self.assertEqual(convert_length(250, "cm", "m"), 2.5)
        self.assertEqual(convert_length(3, "m", "mm"), 3000.0)

    def test_imperial(self):
        self.assertEqual(convert_length(12, "in", "ft"), 1.0)
        self.assertEqual(convert_length(3, "ft", "yd"), 1.0)
        self.assertEqual(convert_length(1, "mi", "yd"), 1760.0)

    def test_mixed(self):
        self.assertEqual(convert_length(1, "mi", "km"), 1.609344)
        self.assertEqual(convert_length(1, "in", "cm"), 2.54)

    def test_same_unit(self):
        self.assertEqual(convert_length(7.25, "ft", "ft"), 7.25)

    def test_rounding(self):
        self.assertEqual(convert_length(1, "mm", "mi"), 1e-06)
        self.assertEqual(convert_length(1, "m", "mi"), 0.000621)

    def test_negative_and_zero(self):
        self.assertEqual(convert_length(-2, "m", "cm"), -200.0)
        self.assertEqual(convert_length(0, "km", "in"), 0.0)

    def test_returns_float(self):
        self.assertIsInstance(convert_length(1, "m", "m"), float)

    def test_unknown_unit(self):
        with self.assertRaises(ValueError):
            convert_length(1, "parsec", "m")
        with self.assertRaises(ValueError):
            convert_length(1, "m", "KM")
""",
)

task(
    "cron-field",
    "small",
    "expand_cron_field",
    r"""
Write a function `expand_cron_field(field, lo, hi)` that expands one field of a cron expression into the sorted list of integers it matches, where `lo` and `hi` are the inclusive minimum and maximum allowed for that field (for example 0 and 59 for minutes).

A field is one or more comma-separated parts. Each part is one of:
- `*` meaning every value from `lo` to `hi`;
- `N` a single integer;
- `A-B` an inclusive range.
Any part may be followed by `/S` with a positive integer step `S`: the values are taken every `S` starting from the first value of the part (`*/15` over 0-59 gives 0, 15, 30, 45; `10-20/5` gives 10, 15, 20). A single number with a step, such as `5/10`, is treated as just that number.

The result contains each value once, in ascending order. Raise `ValueError` for anything malformed: non-numeric text, an empty field or part, a value outside `lo..hi`, a range whose start is greater than its end, or a step below 1.

Examples:
expand_cron_field("*/15", 0, 59) == [0, 15, 30, 45]
expand_cron_field("1,5,10-12", 0, 59) == [1, 5, 10, 11, 12]
expand_cron_field("5-1", 0, 59) raises ValueError
""",
    r"""
def expand_cron_field(field, lo, hi):
    out = set()
    for part in field.split(","):
        rng, _, step = part.partition("/")
        a, _, b = (f"{lo}-{hi}" if rng == "*" else rng).partition("-")
        a, b, step = int(a), int(b or a), int(step or 1)
        if step < 1 or a < lo or b > hi or a > b:
            raise ValueError(field)
        out.update(range(a, b + 1, step))
    return sorted(out)
""",
    r"""
import unittest


class TestCronField(unittest.TestCase):
    def test_star(self):
        self.assertEqual(expand_cron_field("*", 1, 5), [1, 2, 3, 4, 5])

    def test_star_step(self):
        self.assertEqual(expand_cron_field("*/15", 0, 59), [0, 15, 30, 45])

    def test_list_and_range(self):
        self.assertEqual(expand_cron_field("1,5,10-12", 0, 59), [1, 5, 10, 11, 12])

    def test_range_step(self):
        self.assertEqual(expand_cron_field("10-20/5", 0, 59), [10, 15, 20])
        self.assertEqual(expand_cron_field("1-10/4", 0, 59), [1, 5, 9])

    def test_sorted_unique(self):
        self.assertEqual(expand_cron_field("5,1,5,1-3", 0, 10), [1, 2, 3, 5])

    def test_single_with_step(self):
        self.assertEqual(expand_cron_field("5/10", 0, 59), [5])

    def test_bounds_inclusive(self):
        self.assertEqual(expand_cron_field("0,23", 0, 23), [0, 23])
        self.assertEqual(expand_cron_field("1-12", 1, 12), list(range(1, 13)))

    def test_errors(self):
        for field, lo, hi in [("", 0, 59), ("a", 0, 59), ("60", 0, 59), ("0", 1, 31), ("5-1", 0, 59),
                              ("1,,2", 0, 59), ("*/0", 0, 59), ("1-70", 0, 59), ("*/x", 0, 59)]:
            with self.assertRaises(ValueError, msg=field):
                expand_cron_field(field, lo, hi)
""",
)

task(
    "paginate",
    "small",
    "paginate",
    r"""
Write a function `paginate(items, page, per_page)` that returns one page of a sequence together with navigation metadata.

Pages are numbered from 1. The function returns a dict with these keys:
- `"items"`: a new list with the elements of that page (empty if the page is past the end);
- `"page"` and `"per_page"`: the arguments as given;
- `"total_pages"`: how many pages are needed for all items (0 when `items` is empty);
- `"has_next"`: True if a later page exists, i.e. `page < total_pages`;
- `"has_prev"`: True if `page > 1`.

Raise `ValueError` if `page` or `per_page` is less than 1.

Example:
paginate([1, 2, 3, 4, 5], 2, 2) == {"items": [3, 4], "page": 2, "per_page": 2, "total_pages": 3, "has_next": True, "has_prev": True}
""",
    r"""
def paginate(items, page, per_page):
    if page < 1 or per_page < 1:
        raise ValueError("page and per_page must be positive")
    pages = -(-len(items) // per_page)
    start = (page - 1) * per_page
    return {"items": list(items[start:start + per_page]), "page": page, "per_page": per_page,
            "total_pages": pages, "has_next": page < pages, "has_prev": page > 1}
""",
    r"""
import unittest


class TestPaginate(unittest.TestCase):
    def test_example(self):
        self.assertEqual(
            paginate([1, 2, 3, 4, 5], 2, 2),
            {"items": [3, 4], "page": 2, "per_page": 2, "total_pages": 3, "has_next": True, "has_prev": True},
        )

    def test_first_page(self):
        r = paginate(list(range(10)), 1, 4)
        self.assertEqual(r["items"], [0, 1, 2, 3])
        self.assertFalse(r["has_prev"])
        self.assertTrue(r["has_next"])

    def test_last_partial_page(self):
        r = paginate(list(range(10)), 3, 4)
        self.assertEqual(r["items"], [8, 9])
        self.assertFalse(r["has_next"])

    def test_exact_multiple(self):
        r = paginate(list(range(8)), 2, 4)
        self.assertEqual((r["total_pages"], r["has_next"]), (2, False))

    def test_past_end(self):
        r = paginate([1, 2, 3], 5, 2)
        self.assertEqual(r["items"], [])
        self.assertEqual(r["total_pages"], 2)
        self.assertTrue(r["has_prev"])
        self.assertFalse(r["has_next"])

    def test_empty(self):
        r = paginate([], 1, 10)
        self.assertEqual((r["items"], r["total_pages"], r["has_next"], r["has_prev"]), ([], 0, False, False))

    def test_string_and_tuple_input(self):
        self.assertEqual(paginate("abcdef", 2, 4)["items"], ["e", "f"])
        self.assertEqual(paginate((1, 2, 3), 1, 2)["items"], [1, 2])

    def test_invalid(self):
        for page, per in [(0, 5), (1, 0), (-1, 3), (2, -4)]:
            with self.assertRaises(ValueError):
                paginate([1], page, per)

    def test_returns_new_list(self):
        src = [1, 2, 3]
        paginate(src, 1, 3)["items"].append(9)
        self.assertEqual(src, [1, 2, 3])
""",
)

task(
    "invoice-total",
    "small",
    "invoice_total",
    r"""
Write a function `invoice_total(lines, tax_bp, discount_pct=0)` that computes an invoice in integer cents.

- `lines` is a list of `(quantity, unit_price_cents)` tuples, both integers.
- `subtotal` is the sum of quantity times unit price over all lines.
- `discount` is `subtotal * discount_pct // 100` (integer percent, rounded down).
- `tax` is charged on `subtotal - discount` at `tax_bp` basis points (1 bp = 0.01%), rounded half up to a whole cent, using integer arithmetic only: `((subtotal - discount) * tax_bp + 5000) // 10000`.
- `total` is `subtotal - discount + tax`.

Return a dict with the keys `"subtotal"`, `"discount"`, `"tax"` and `"total"`, all ints.

Example:
invoice_total([(2, 1999), (1, 500)], 825, 10) == {"subtotal": 4498, "discount": 449, "tax": 334, "total": 4383}
""",
    r"""
def invoice_total(lines, tax_bp, discount_pct=0):
    subtotal = sum(q * p for q, p in lines)
    discount = subtotal * discount_pct // 100
    tax = ((subtotal - discount) * tax_bp + 5000) // 10000
    return {"subtotal": subtotal, "discount": discount, "tax": tax, "total": subtotal - discount + tax}
""",
    r"""
import unittest


class TestInvoice(unittest.TestCase):
    def test_example(self):
        self.assertEqual(
            invoice_total([(2, 1999), (1, 500)], 825, 10),
            {"subtotal": 4498, "discount": 449, "tax": 334, "total": 4383},
        )

    def test_no_discount_default(self):
        self.assertEqual(invoice_total([(3, 100)], 1000), {"subtotal": 300, "discount": 0, "tax": 30, "total": 330})

    def test_empty(self):
        self.assertEqual(invoice_total([], 825), {"subtotal": 0, "discount": 0, "tax": 0, "total": 0})

    def test_zero_tax(self):
        self.assertEqual(invoice_total([(1, 999)], 0, 50)["total"], 999 - 499)

    def test_discount_rounds_down(self):
        self.assertEqual(invoice_total([(1, 199)], 0, 15)["discount"], 29)

    def test_tax_rounds_half_up(self):
        self.assertEqual(invoice_total([(1, 50)], 100)["tax"], 1)
        self.assertEqual(invoice_total([(1, 49)], 100)["tax"], 0)
        self.assertEqual(invoice_total([(1, 150)], 100)["tax"], 2)

    def test_full_discount(self):
        r = invoice_total([(4, 250)], 2000, 100)
        self.assertEqual((r["discount"], r["tax"], r["total"]), (1000, 0, 0))

    def test_types(self):
        r = invoice_total([(7, 333)], 725, 5)
        self.assertTrue(all(isinstance(v, int) for v in r.values()))
        self.assertEqual(r["total"], r["subtotal"] - r["discount"] + r["tax"])
""",
)

task(
    "redact-log",
    "small",
    "redact_log_line",
    r"""
Write a function `redact_log_line(line)` that masks personal data in a log line and returns the new line.

1. IPv4 addresses: any four groups of 1 to 3 digits separated by dots, standing alone (not directly attached to other digits or letters on either side) have their last group replaced by `0`. `10.1.2.37` becomes `10.1.2.0`. The groups are not range-checked.
2. Email addresses: text of the form `local@domain` where `local` consists of one or more of the characters `A-Z a-z 0-9 . _ % + -` and `domain` consists of `A-Z a-z 0-9 . -` characters ending with a dot and at least two letters (for example `example.com`). Each email is replaced by the first character of `local`, then `***`, then `@`, then the domain unchanged: `alice@example.com` becomes `a***@example.com`.

Everything else in the line is unchanged. Several matches may occur in one line.

Example:
redact_log_line("login ok user=bob.smith@mail.example.org from 192.168.1.77") == "login ok user=b***@mail.example.org from 192.168.1.0"
""",
    r"""
import re


def redact_log_line(line):
    line = re.sub(r"\b(\d{1,3}\.\d{1,3}\.\d{1,3})\.\d{1,3}\b", r"\1.0", line)
    return re.sub(r"([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*@([A-Za-z0-9.-]+\.[A-Za-z]{2,})", r"\1***@\2", line)
""",
    r"""
import unittest


class TestRedact(unittest.TestCase):
    def test_example(self):
        self.assertEqual(
            redact_log_line("login ok user=bob.smith@mail.example.org from 192.168.1.77"),
            "login ok user=b***@mail.example.org from 192.168.1.0",
        )

    def test_ip_only(self):
        self.assertEqual(redact_log_line("GET /x 10.0.0.255 200"), "GET /x 10.0.0.0 200")

    def test_email_only(self):
        self.assertEqual(redact_log_line("contact alice@example.com now"), "contact a***@example.com now")

    def test_multiple(self):
        self.assertEqual(
            redact_log_line("1.2.3.4 -> 5.6.7.8 cc a@x.io, zed@y.co"),
            "1.2.3.0 -> 5.6.7.0 cc a***@x.io, z***@y.co",
        )

    def test_nothing_to_redact(self):
        s = "service started in 12.5 seconds, version 1.2.3"
        self.assertEqual(redact_log_line(s), s)

    def test_not_an_email(self):
        s = "user@localhost and @mention and a@b"
        self.assertEqual(redact_log_line(s), s)

    def test_trailing_period(self):
        self.assertEqual(redact_log_line("sent to carol@corp.net."), "sent to c***@corp.net.")

    def test_plus_and_dash_local(self):
        self.assertEqual(redact_log_line("x=+tag-1@d.dev"), "x=+***@d.dev")
""",
)

task(
    "next-business-day",
    "small",
    "next_business_day",
    r"""
Write a function `next_business_day(day, holidays=())` that returns the first business day strictly after `day`.

`day` is a `datetime.date`. Saturdays and Sundays are not business days, and neither is any date contained in `holidays` (a collection of `datetime.date` objects, possibly empty). The returned value is a `datetime.date`. The input is never itself returned, even if it is a business day.

Examples (2024-03-01 is a Friday):
next_business_day(date(2024, 3, 1)) == date(2024, 3, 4)
next_business_day(date(2024, 3, 4), {date(2024, 3, 5)}) == date(2024, 3, 6)
""",
    r"""
from datetime import timedelta


def next_business_day(day, holidays=()):
    day += timedelta(days=1)
    while day.weekday() >= 5 or day in holidays:
        day += timedelta(days=1)
    return day
""",
    r"""
import unittest
from datetime import date


class TestNextBusinessDay(unittest.TestCase):
    def test_midweek(self):
        self.assertEqual(next_business_day(date(2024, 3, 6)), date(2024, 3, 7))

    def test_friday_to_monday(self):
        self.assertEqual(next_business_day(date(2024, 3, 1)), date(2024, 3, 4))

    def test_saturday_and_sunday(self):
        self.assertEqual(next_business_day(date(2024, 3, 2)), date(2024, 3, 4))
        self.assertEqual(next_business_day(date(2024, 3, 3)), date(2024, 3, 4))

    def test_holiday_skipped(self):
        self.assertEqual(next_business_day(date(2024, 3, 4), {date(2024, 3, 5)}), date(2024, 3, 6))

    def test_holiday_on_monday_after_weekend(self):
        self.assertEqual(next_business_day(date(2024, 3, 1), [date(2024, 3, 4)]), date(2024, 3, 5))

    def test_consecutive_holidays(self):
        hol = {date(2024, 12, 25), date(2024, 12, 26)}
        self.assertEqual(next_business_day(date(2024, 12, 24), hol), date(2024, 12, 27))

    def test_year_end(self):
        self.assertEqual(next_business_day(date(2023, 12, 29)), date(2024, 1, 1))

    def test_leap_day(self):
        self.assertEqual(next_business_day(date(2024, 2, 28)), date(2024, 2, 29))
        self.assertEqual(next_business_day(date(2024, 2, 29)), date(2024, 3, 1))

    def test_holiday_on_weekend_irrelevant(self):
        self.assertEqual(next_business_day(date(2024, 3, 1), {date(2024, 3, 2)}), date(2024, 3, 4))
""",
)

# ---------------------------------------------------------------- MEDIUM

task(
    "token-bucket",
    "medium",
    "TokenBucket",
    r"""
Implement a rate limiter class `TokenBucket` with an injected clock, so it can be tested without real time.

`TokenBucket(rate, capacity, clock)`:
- `rate` is the number of tokens added per second, `capacity` the maximum number of tokens the bucket can hold, and `clock` a zero-argument callable returning the current time in seconds as a number. Raise `ValueError` if `rate` or `capacity` is not positive. The bucket starts full, and the starting time is read from `clock()` at construction.
- Tokens refill continuously: whenever a method is called, first add `(now - last) * rate` tokens, cap the total at `capacity`, and remember `now`. If the clock has moved backwards, add nothing.

Methods:
- `allow(cost=1)`: refill, then if at least `cost` tokens are available remove `cost` tokens and return True; otherwise leave the tokens untouched and return False.
- `available()`: refill, then return the current number of tokens (a float or int).
- `wait_time(cost=1)`: refill, then return the number of seconds until `cost` tokens will be available (0.0 if they already are). It does not consume tokens.
- `reset()`: refill the bucket to full capacity.

`allow` and `wait_time` raise `ValueError` if `cost` is not positive or is greater than `capacity`.

Example: with rate 2, capacity 5 and a clock at 0, `allow(3)` is True and `available()` is 2; one second later `available()` is 4.
""",
    r"""
class TokenBucket:
    def __init__(self, rate, capacity, clock):
        if rate <= 0 or capacity <= 0:
            raise ValueError("rate and capacity must be positive")
        self.rate, self.capacity, self.clock = rate, capacity, clock
        self.tokens = capacity
        self.last = clock()

    def _refill(self):
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + max(0, now - self.last) * self.rate)
        self.last = now

    def _check(self, cost):
        if cost <= 0 or cost > self.capacity:
            raise ValueError("invalid cost")

    def allow(self, cost=1):
        self._check(cost)
        self._refill()
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False

    def available(self):
        self._refill()
        return self.tokens

    def wait_time(self, cost=1):
        self._check(cost)
        self._refill()
        return 0.0 if self.tokens >= cost else (cost - self.tokens) / self.rate

    def reset(self):
        self._refill()
        self.tokens = self.capacity
""",
    r"""
import unittest


class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class TestTokenBucket(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.b = TokenBucket(2, 5, self.clock)

    def test_starts_full(self):
        self.assertEqual(self.b.available(), 5)

    def test_allow_consumes(self):
        self.assertTrue(self.b.allow(3))
        self.assertEqual(self.b.available(), 2)

    def test_denied_does_not_consume(self):
        self.b.allow(3)
        self.assertFalse(self.b.allow(3))
        self.assertEqual(self.b.available(), 2)

    def test_refill_over_time(self):
        self.b.allow(5)
        self.assertEqual(self.b.available(), 0)
        self.clock.advance(1)
        self.assertAlmostEqual(self.b.available(), 2)
        self.clock.advance(0.5)
        self.assertAlmostEqual(self.b.available(), 3)

    def test_capped_at_capacity(self):
        self.b.allow(2)
        self.clock.advance(1000)
        self.assertEqual(self.b.available(), 5)

    def test_wait_time(self):
        self.b.allow(3)
        self.assertAlmostEqual(self.b.wait_time(4), 1.0)
        self.assertEqual(self.b.wait_time(2), 0.0)
        self.assertAlmostEqual(self.b.available(), 2)

    def test_wait_then_allow(self):
        self.b.allow(5)
        w = self.b.wait_time(3)
        self.assertAlmostEqual(w, 1.5)
        self.clock.advance(w)
        self.assertTrue(self.b.allow(3))

    def test_clock_backwards(self):
        self.clock.t = 10
        self.b.allow(5)
        self.clock.t = 5
        self.assertEqual(self.b.available(), 0)
        self.clock.t = 6
        self.assertAlmostEqual(self.b.available(), 2)

    def test_reset(self):
        self.b.allow(5)
        self.b.reset()
        self.assertEqual(self.b.available(), 5)

    def test_start_time_read_at_construction(self):
        clock = FakeClock(100.0)
        b = TokenBucket(1, 2, clock)
        b.allow(2)
        clock.advance(1)
        self.assertAlmostEqual(b.available(), 1)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            TokenBucket(0, 5, self.clock)
        with self.assertRaises(ValueError):
            TokenBucket(1, -1, self.clock)
        for bad in (0, -1, 6):
            with self.assertRaises(ValueError):
                self.b.allow(bad)
            with self.assertRaises(ValueError):
                self.b.wait_time(bad)
""",
)

task(
    "retry-backoff",
    "medium",
    "retry_call",
    r"""
Write retry helpers with exponential backoff and an injected `sleep` function (so tests never really sleep). Provide three public names.

`backoff_delays(attempts, base, factor=2.0, max_delay=None)` returns the list of delays to wait between attempts: `attempts - 1` numbers, where the i-th (starting at 0) is `base * factor ** i`, capped at `max_delay` if that is not None. Raise `ValueError` if `attempts < 1`.

`retry_call(func, attempts=3, base=0.1, factor=2.0, max_delay=None, retry_on=(Exception,), sleep=time.sleep, on_retry=None)` calls `func()` with no arguments and returns its result. If it raises an exception that is an instance of one of the classes in `retry_on`, wait for the next delay from `backoff_delays` by calling `sleep(delay)` and try again, up to `attempts` calls in total. When the last attempt fails, re-raise that last exception. An exception that is not an instance of `retry_on` propagates immediately with no sleeping. If `on_retry` is not None, it is called as `on_retry(attempt, exc, delay)` just before each `sleep(delay)`, where `attempt` counts from 1 (the attempt that just failed) and `exc` is the exception raised; it is not called after the final attempt. `sleep` is only called between attempts, never after the final one or after a success. Raise `ValueError` if `attempts < 1`.

`retrying(**options)` is a decorator factory: `@retrying(attempts=4, sleep=fake_sleep)` wraps a function so that calling the wrapped function with any arguments behaves like `retry_call(lambda: f(*args, **kwargs), **options)`. The wrapper keeps the original function's `__name__`.

Example: delays for `backoff_delays(5, 1, 2, max_delay=5)` are `[1, 2, 4, 5]`.
""",
    r"""
import functools
import time


def backoff_delays(attempts, base, factor=2.0, max_delay=None):
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    delays = [base * factor ** i for i in range(attempts - 1)]
    return delays if max_delay is None else [min(d, max_delay) for d in delays]


def retry_call(
    func,
    attempts=3,
    base=0.1,
    factor=2.0,
    max_delay=None,
    retry_on=(Exception,),
    sleep=time.sleep,
    on_retry=None,
):
    delays = backoff_delays(attempts, base, factor, max_delay)
    for i in range(attempts):
        try:
            return func()
        except retry_on as exc:
            if i == attempts - 1:
                raise
            if on_retry is not None:
                on_retry(i + 1, exc, delays[i])
            sleep(delays[i])


def retrying(**options):
    def decorate(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return retry_call(lambda: f(*args, **kwargs), **options)

        return wrapper

    return decorate
""",
    r"""
import unittest


class Flaky:
    def __init__(self, failures, exc=RuntimeError):
        self.failures, self.exc, self.calls = failures, exc, 0

    def __call__(self):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc(f"fail {self.calls}")
        return "ok"


class TestBackoff(unittest.TestCase):
    def test_delays_basic(self):
        self.assertEqual(backoff_delays(4, 1), [1, 2, 4])

    def test_delays_capped(self):
        self.assertEqual(backoff_delays(5, 1, 2, max_delay=5), [1, 2, 4, 5])

    def test_delays_factor(self):
        self.assertEqual(backoff_delays(4, 0.5, factor=3), [0.5, 1.5, 4.5])

    def test_delays_invalid_and_single(self):
        self.assertEqual(backoff_delays(1, 1), [])
        with self.assertRaises(ValueError):
            backoff_delays(0, 1)

    def test_success_first_try(self):
        slept = []
        self.assertEqual(retry_call(Flaky(0), sleep=slept.append), "ok")
        self.assertEqual(slept, [])

    def test_retries_then_succeeds(self):
        slept, f = [], Flaky(2)
        self.assertEqual(retry_call(f, attempts=4, base=1, sleep=slept.append), "ok")
        self.assertEqual(f.calls, 3)
        self.assertEqual(slept, [1, 2])

    def test_exhausted_reraises_last(self):
        slept, f = [], Flaky(10)
        with self.assertRaises(RuntimeError) as cm:
            retry_call(f, attempts=3, base=1, sleep=slept.append)
        self.assertEqual(str(cm.exception), "fail 3")
        self.assertEqual(f.calls, 3)
        self.assertEqual(slept, [1, 2])

    def test_non_retryable_propagates(self):
        slept, f = [], Flaky(5, exc=KeyError)
        with self.assertRaises(KeyError):
            retry_call(f, retry_on=(ValueError,), sleep=slept.append)
        self.assertEqual((f.calls, slept), (1, []))

    def test_max_delay_used(self):
        slept = []
        retry_call(Flaky(3), attempts=5, base=1, max_delay=2, sleep=slept.append)
        self.assertEqual(slept, [1, 2, 2])

    def test_on_retry_hook(self):
        events = []
        retry_call(Flaky(2), attempts=3, base=1, sleep=lambda d: events.append(("sleep", d)),
                   on_retry=lambda n, e, d: events.append((n, str(e), d)))
        self.assertEqual(events, [(1, "fail 1", 1), ("sleep", 1), (2, "fail 2", 2), ("sleep", 2)])

    def test_decorator(self):
        slept = []
        state = {"n": 0}

        @retrying(attempts=3, base=2, sleep=slept.append)
        def add(a, b=0):
            state["n"] += 1
            if state["n"] < 3:
                raise OSError("again")
            return a + b

        self.assertEqual(add(1, b=2), 3)
        self.assertEqual(slept, [2, 4])
        self.assertEqual(add.__name__, "add")

    def test_attempts_invalid(self):
        with self.assertRaises(ValueError):
            retry_call(lambda: 1, attempts=0)
""",
)

task(
    "fifo-inventory",
    "medium",
    "Inventory",
    r"""
Implement a class `Inventory` that tracks stock per SKU with FIFO (first in, first out) costing. Money is in integer cents.

Methods:
- `receive(sku, qty, unit_cost)`: add `qty` units of `sku` bought at `unit_cost` cents each, as a new batch (layer) behind any existing ones. `qty` must be a positive integer and `unit_cost` a non-negative integer, otherwise raise `ValueError`.
- `ship(sku, qty)`: remove `qty` units of `sku` taking from the oldest batches first, and return the total cost in cents of the units removed (sum of units taken times the cost of the batch they came from). `qty` must be a positive integer, otherwise `ValueError`. If fewer than `qty` units are in stock (including an unknown SKU), raise `ValueError` and change nothing. A batch that is used up is dropped.
- `stock(sku)`: units on hand; 0 for an unknown SKU.
- `value(sku=None)`: total cost in cents of the units on hand for that SKU, or for all SKUs if `sku` is None.
- `layers(sku)`: a list of `(qty, unit_cost)` tuples for the remaining batches, oldest first (an empty list for an unknown SKU). Modifying the returned list must not affect the inventory.

Example: receive("A", 10, 100); receive("A", 10, 150); `ship("A", 15)` returns 10*100 + 5*150 = 1750, and `layers("A")` is then `[(5, 150)]`.
""",
    r"""
from collections import deque


class Inventory:
    def __init__(self):
        self._batches = {}

    def receive(self, sku, qty, unit_cost):
        if qty <= 0 or unit_cost < 0:
            raise ValueError("invalid quantity or cost")
        self._batches.setdefault(sku, deque()).append([qty, unit_cost])

    def stock(self, sku):
        return sum(q for q, _ in self._batches.get(sku, ()))

    def ship(self, sku, qty):
        if qty <= 0 or qty > self.stock(sku):
            raise ValueError("invalid or insufficient quantity")
        cost, queue = 0, self._batches[sku]
        while qty:
            take = min(qty, queue[0][0])
            cost += take * queue[0][1]
            queue[0][0] -= take
            qty -= take
            if queue[0][0] == 0:
                queue.popleft()
        return cost

    def value(self, sku=None):
        skus = self._batches if sku is None else [sku]
        return sum(q * c for s in skus for q, c in self._batches.get(s, ()))

    def layers(self, sku):
        return [(q, c) for q, c in self._batches.get(sku, ())]
""",
    r"""
import unittest


class TestInventory(unittest.TestCase):
    def setUp(self):
        self.inv = Inventory()
        self.inv.receive("A", 10, 100)
        self.inv.receive("A", 10, 150)

    def test_stock(self):
        self.assertEqual(self.inv.stock("A"), 20)
        self.assertEqual(self.inv.stock("missing"), 0)

    def test_ship_across_batches(self):
        self.assertEqual(self.inv.ship("A", 15), 1750)
        self.assertEqual(self.inv.layers("A"), [(5, 150)])
        self.assertEqual(self.inv.stock("A"), 5)

    def test_ship_within_first_batch(self):
        self.assertEqual(self.inv.ship("A", 4), 400)
        self.assertEqual(self.inv.layers("A"), [(6, 100), (10, 150)])

    def test_ship_exact_batch_drops_it(self):
        self.assertEqual(self.inv.ship("A", 10), 1000)
        self.assertEqual(self.inv.layers("A"), [(10, 150)])

    def test_ship_everything(self):
        self.assertEqual(self.inv.ship("A", 20), 2500)
        self.assertEqual(self.inv.stock("A"), 0)
        self.assertEqual(self.inv.layers("A"), [])

    def test_insufficient_changes_nothing(self):
        with self.assertRaises(ValueError):
            self.inv.ship("A", 21)
        self.assertEqual(self.inv.layers("A"), [(10, 100), (10, 150)])
        with self.assertRaises(ValueError):
            self.inv.ship("zzz", 1)

    def test_value(self):
        self.inv.receive("B", 3, 1000)
        self.assertEqual(self.inv.value("A"), 2500)
        self.assertEqual(self.inv.value("B"), 3000)
        self.assertEqual(self.inv.value(), 5500)
        self.assertEqual(self.inv.value("nope"), 0)

    def test_value_after_ship(self):
        self.inv.ship("A", 12)
        self.assertEqual(self.inv.value("A"), 8 * 150)

    def test_new_batch_goes_to_back(self):
        self.inv.ship("A", 20)
        self.inv.receive("A", 2, 7)
        self.inv.receive("A", 2, 9)
        self.assertEqual(self.inv.ship("A", 3), 2 * 7 + 9)

    def test_layers_is_a_copy(self):
        self.inv.layers("A").clear()
        self.assertEqual(self.inv.stock("A"), 20)

    def test_invalid_arguments(self):
        for args in [("A", 0, 5), ("A", -1, 5), ("A", 1, -5)]:
            with self.assertRaises(ValueError):
                self.inv.receive(*args)
        for q in (0, -3):
            with self.assertRaises(ValueError):
                self.inv.ship("A", q)
        self.assertEqual(self.inv.stock("A"), 20)

    def test_free_goods(self):
        self.inv.receive("G", 5, 0)
        self.assertEqual(self.inv.ship("G", 5), 0)
""",
)

task(
    "config-merge",
    "medium",
    "deep_merge",
    r"""
Write four helper functions for nested configuration dictionaries. None of them may modify their inputs, and the dicts they return never share nested dicts or lists with the inputs.

`deep_merge(base, override)` returns a new dict. For every key in `override`: if both `base` and `override` hold a dict under that key, merge them recursively; if the override value is `None`, the key is removed from the result (even if it was not in `base`); otherwise the override value replaces the base value (lists are replaced, not concatenated). Keys only in `base` are kept.

`flatten(d, sep=".")` turns nested dicts into a single-level dict whose keys are the paths joined with `sep`: `{"a": {"b": 1}, "c": 2}` becomes `{"a.b": 1, "c": 2}`. An empty nested dict is kept as a leaf value `{}`. Non-dict values (including lists) are leaves.

`unflatten(flat, sep=".")` is the inverse: split each key on `sep` and build nested dicts. Raise `ValueError` if the keys conflict, that is, if a key is used both as a leaf value and as a prefix of another key (for example `"a"` and `"a.b"`).

`get_path(d, path, default=None, sep=".")` follows the `sep`-separated path through nested dicts and returns the value found, or `default` if any step is missing or hits a non-dict value.

Example: `deep_merge({"db": {"host": "x", "port": 1}}, {"db": {"port": 2}})` is `{"db": {"host": "x", "port": 2}}`.
""",
    r"""
import copy


def deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in override.items():
        if v is None:
            out.pop(k, None)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def flatten(d, sep="."):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and v:
            out.update({f"{k}{sep}{kk}": vv for kk, vv in flatten(v, sep).items()})
        else:
            out[k] = v
    return out


def unflatten(flat, sep="."):
    root = {}
    for key, value in flat.items():
        *parents, leaf = key.split(sep)
        cur = root
        for p in parents:
            cur = cur.setdefault(p, {})
            if not isinstance(cur, dict):
                raise ValueError(f"conflict at {key}")
        if leaf in cur:
            raise ValueError(f"conflict at {key}")
        cur[leaf] = value
    return root


def get_path(d, path, default=None, sep="."):
    cur = d
    for part in path.split(sep):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
""",
    r"""
import copy
import unittest


class TestConfig(unittest.TestCase):
    def test_merge_nested(self):
        self.assertEqual(
            deep_merge({"db": {"host": "x", "port": 1}, "debug": False}, {"db": {"port": 2}}),
            {"db": {"host": "x", "port": 2}, "debug": False},
        )

    def test_merge_lists_replaced(self):
        self.assertEqual(deep_merge({"a": [1, 2]}, {"a": [3]}), {"a": [3]})

    def test_merge_none_removes(self):
        self.assertEqual(deep_merge({"a": 1, "b": {"c": 2, "d": 3}}, {"a": None, "b": {"c": None}, "z": None}),
                         {"b": {"d": 3}})

    def test_merge_type_clash(self):
        self.assertEqual(deep_merge({"a": {"x": 1}}, {"a": 5}), {"a": 5})
        self.assertEqual(deep_merge({"a": 5}, {"a": {"x": 1}}), {"a": {"x": 1}})

    def test_merge_new_keys_and_no_mutation(self):
        base, over = {"a": {"b": 1}}, {"a": {"c": 2}, "d": {"e": 3}}
        b0, o0 = copy.deepcopy(base), copy.deepcopy(over)
        result = deep_merge(base, over)
        self.assertEqual(result, {"a": {"b": 1, "c": 2}, "d": {"e": 3}})
        self.assertEqual((base, over), (b0, o0))
        result["a"]["b"] = 99
        result["d"]["e"] = 99
        self.assertEqual((base, over), (b0, o0))

    def test_flatten(self):
        self.assertEqual(flatten({"a": {"b": 1, "c": {"d": 2}}, "e": [1, 2]}), {"a.b": 1, "a.c.d": 2, "e": [1, 2]})

    def test_flatten_empty_dict_and_sep(self):
        self.assertEqual(flatten({"a": {}, "b": {"c": 1}}, sep="/"), {"a": {}, "b/c": 1})

    def test_unflatten(self):
        self.assertEqual(unflatten({"a.b": 1, "a.c.d": 2, "e": 3}), {"a": {"b": 1, "c": {"d": 2}}, "e": 3})

    def test_roundtrip(self):
        d = {"x": {"y": {"z": 1}, "w": [1]}, "q": "s"}
        self.assertEqual(unflatten(flatten(d)), d)

    def test_unflatten_conflict(self):
        with self.assertRaises(ValueError):
            unflatten({"a": 1, "a.b": 2})
        with self.assertRaises(ValueError):
            unflatten({"a.b": 2, "a": 1})

    def test_get_path(self):
        d = {"a": {"b": {"c": 5}}, "x": 1}
        self.assertEqual(get_path(d, "a.b.c"), 5)
        self.assertEqual(get_path(d, "a.b"), {"c": 5})
        self.assertEqual(get_path(d, "a.q.c", default="no"), "no")
        self.assertIsNone(get_path(d, "x.y"))
        self.assertEqual(get_path(d, "a/b/c", sep="/"), 5)
""",
)

task(
    "order-state-machine",
    "medium",
    "OrderStateMachine",
    r"""
Implement an order lifecycle state machine. Provide an exception class `InvalidTransition` (a subclass of `Exception`) and a class `OrderStateMachine`.

States and events. The machine starts in state `"new"`. Allowed transitions, as `(current state, event) -> next state`:
- `("new", "pay")` -> `"paid"`
- `("paid", "ship")` -> `"shipped"`
- `("shipped", "deliver")` -> `"delivered"`
- `("delivered", "refund")` -> `"refunded"`
- `("new", "cancel")` -> `"cancelled"`
- `("paid", "cancel")` -> `"cancelled"`
No other combination is allowed; `"cancelled"` and `"refunded"` have no outgoing transitions.

`OrderStateMachine()` has:
- attribute `state`: the current state string.
- `can(event)`: True if `event` is allowed in the current state; False otherwise (including unknown event names).
- `fire(event)`: apply the event, record it in the history, and return the new state. Raise `InvalidTransition` if the event is not allowed now; the state and history must then be unchanged.
- `allowed_events()`: sorted list of the event names allowed in the current state (empty list when finished).
- `history()`: a list of `(event, from_state, to_state)` tuples in the order applied (a new list each call).
- `undo()`: revert the most recent transition, remove it from the history and return the restored state. Raise `InvalidTransition` if there is nothing to undo.

Example: fire("pay") returns "paid"; then allowed_events() is ["cancel", "ship"].
""",
    r"""
class InvalidTransition(Exception):
    pass


TRANSITIONS = {
    ("new", "pay"): "paid",
    ("paid", "ship"): "shipped",
    ("shipped", "deliver"): "delivered",
    ("delivered", "refund"): "refunded",
    ("new", "cancel"): "cancelled",
    ("paid", "cancel"): "cancelled",
}


class OrderStateMachine:
    def __init__(self):
        self.state = "new"
        self._history = []

    def can(self, event):
        return (self.state, event) in TRANSITIONS

    def fire(self, event):
        if not self.can(event):
            raise InvalidTransition(f"{event!r} not allowed in {self.state!r}")
        target = TRANSITIONS[(self.state, event)]
        self._history.append((event, self.state, target))
        self.state = target
        return target

    def allowed_events(self):
        return sorted(e for s, e in TRANSITIONS if s == self.state)

    def history(self):
        return list(self._history)

    def undo(self):
        if not self._history:
            raise InvalidTransition("nothing to undo")
        _, previous, _ = self._history.pop()
        self.state = previous
        return previous
""",
    r"""
import unittest


class TestOrder(unittest.TestCase):
    def setUp(self):
        self.m = OrderStateMachine()

    def test_initial(self):
        self.assertEqual(self.m.state, "new")
        self.assertEqual(self.m.allowed_events(), ["cancel", "pay"])
        self.assertEqual(self.m.history(), [])

    def test_happy_path(self):
        self.assertEqual(self.m.fire("pay"), "paid")
        self.assertEqual(self.m.fire("ship"), "shipped")
        self.assertEqual(self.m.fire("deliver"), "delivered")
        self.assertEqual(self.m.fire("refund"), "refunded")
        self.assertEqual(self.m.state, "refunded")
        self.assertEqual(self.m.allowed_events(), [])

    def test_allowed_events_by_state(self):
        self.m.fire("pay")
        self.assertEqual(self.m.allowed_events(), ["cancel", "ship"])
        self.m.fire("ship")
        self.assertEqual(self.m.allowed_events(), ["deliver"])
        self.m.fire("deliver")
        self.assertEqual(self.m.allowed_events(), ["refund"])

    def test_cancel_from_new_and_paid(self):
        self.assertEqual(self.m.fire("cancel"), "cancelled")
        other = OrderStateMachine()
        other.fire("pay")
        self.assertEqual(other.fire("cancel"), "cancelled")
        self.assertEqual(other.allowed_events(), [])

    def test_cannot_cancel_after_shipping(self):
        self.m.fire("pay")
        self.m.fire("ship")
        self.assertFalse(self.m.can("cancel"))
        with self.assertRaises(InvalidTransition):
            self.m.fire("cancel")

    def test_invalid_leaves_state_and_history(self):
        self.m.fire("pay")
        with self.assertRaises(InvalidTransition):
            self.m.fire("deliver")
        with self.assertRaises(InvalidTransition):
            self.m.fire("bogus")
        self.assertEqual(self.m.state, "paid")
        self.assertEqual(self.m.history(), [("pay", "new", "paid")])

    def test_can(self):
        self.assertTrue(self.m.can("pay"))
        self.assertFalse(self.m.can("ship"))
        self.assertFalse(self.m.can("nonsense"))

    def test_history_order_and_copy(self):
        self.m.fire("pay")
        self.m.fire("ship")
        h = self.m.history()
        self.assertEqual(h, [("pay", "new", "paid"), ("ship", "paid", "shipped")])
        h.clear()
        self.assertEqual(len(self.m.history()), 2)

    def test_undo(self):
        self.m.fire("pay")
        self.m.fire("ship")
        self.assertEqual(self.m.undo(), "paid")
        self.assertEqual(self.m.state, "paid")
        self.assertEqual(self.m.history(), [("pay", "new", "paid")])
        self.assertEqual(self.m.undo(), "new")
        self.assertEqual(self.m.state, "new")

    def test_undo_nothing(self):
        with self.assertRaises(InvalidTransition):
            self.m.undo()

    def test_undo_from_final_state(self):
        self.m.fire("cancel")
        self.assertEqual(self.m.undo(), "new")
        self.assertEqual(self.m.fire("pay"), "paid")

    def test_exception_type(self):
        self.assertTrue(issubclass(InvalidTransition, Exception))
""",
)

task(
    "meeting-slots",
    "medium",
    "find_slot",
    r"""
Write three scheduling helpers working on time intervals expressed as `(start, end)` tuples of numbers (for example minutes since midnight). An interval covers `start` up to but not including `end`.

`merge_intervals(intervals)` returns a sorted list of disjoint `(start, end)` tuples covering the same time. Intervals that overlap or merely touch (one ends exactly where the next starts) are merged into one. Empty or reversed intervals (`start >= end`) are ignored. The input is not modified.

`free_slots(busy, day_start, day_end)` takes a list of busy intervals (they may overlap, be unsorted, or stick out of the working window) and returns the sorted list of free `(start, end)` gaps inside the window `day_start..day_end`, each non-empty. Busy time outside the window is irrelevant.

`find_slot(calendars, duration, day_start, day_end)` takes a list of calendars (each a list of busy intervals, one per participant) and returns the earliest `(start, start + duration)` during which all participants are free and which lies entirely inside the window, or `None` if there is no such slot. Raise `ValueError` if `duration <= 0`.

Example: `free_slots([(9, 10), (12, 13)], 8, 17)` is `[(8, 9), (10, 12), (13, 17)]`.
""",
    r"""
def merge_intervals(intervals):
    out = []
    for s, e in sorted(i for i in intervals if i[0] < i[1]):
        if out and s <= out[-1][1]:
            start, end = out[-1]
            out[-1] = (start, max(end, e))
        else:
            out.append((s, e))
    return out


def free_slots(busy, day_start, day_end):
    slots, cur = [], day_start
    for s, e in merge_intervals(busy):
        if min(s, day_end) > cur:
            slots.append((cur, min(s, day_end)))
        cur = max(cur, e)
    if cur < day_end:
        slots.append((cur, day_end))
    return slots


def find_slot(calendars, duration, day_start, day_end):
    if duration <= 0:
        raise ValueError("duration must be positive")
    busy = []
    for calendar in calendars:
        busy.extend(calendar)
    for s, e in free_slots(busy, day_start, day_end):
        if e - s >= duration:
            return (s, s + duration)
    return None
""",
    r"""
import unittest


class TestSlots(unittest.TestCase):
    def test_merge_basic(self):
        self.assertEqual(merge_intervals([(5, 8), (1, 3), (2, 4), (8, 9)]), [(1, 4), (5, 9)])

    def test_merge_touching(self):
        self.assertEqual(merge_intervals([(1, 2), (2, 3), (3, 4)]), [(1, 4)])

    def test_merge_contained_empty_and_pure(self):
        self.assertEqual(merge_intervals([(1, 10), (2, 3), (4, 4), (7, 5)]), [(1, 10)])
        self.assertEqual(merge_intervals([]), [])
        src = [(5, 6), (1, 2)]
        merge_intervals(src)
        self.assertEqual(src, [(5, 6), (1, 2)])

    def test_free_basic(self):
        self.assertEqual(free_slots([(9, 10), (12, 13)], 8, 17), [(8, 9), (10, 12), (13, 17)])

    def test_free_outside_busy(self):
        self.assertEqual(free_slots([(0, 5), (20, 30)], 8, 17), [(8, 17)])

    def test_free_clipped(self):
        self.assertEqual(free_slots([(6, 10), (16, 20)], 8, 17), [(10, 16)])

    def test_free_fully_busy_or_empty(self):
        self.assertEqual(free_slots([(0, 30)], 8, 17), [])
        self.assertEqual(free_slots([(8, 12), (12, 17)], 8, 17), [])
        self.assertEqual(free_slots([], 8, 17), [(8, 17)])

    def test_free_unsorted_overlapping(self):
        self.assertEqual(free_slots([(12, 15), (9, 13), (10, 11)], 8, 17), [(8, 9), (15, 17)])

    def test_find_slot(self):
        cals = [[(9, 11)], [(10, 12), (14, 15)]]
        self.assertEqual(find_slot(cals, 2, 9, 17), (12, 14))
        self.assertEqual(find_slot(cals, 1, 9, 17), (12, 13))

    def test_find_slot_none(self):
        cals = [[(9, 11)], [(10, 12), (14, 15)]]
        self.assertIsNone(find_slot(cals, 3, 9, 17))
        self.assertIsNone(find_slot([[(0, 24)]], 1, 9, 17))

    def test_find_slot_at_window_start_and_floats(self):
        self.assertEqual(find_slot([[(10, 12)]], 1, 9, 17), (9, 10))
        self.assertEqual(find_slot([[(10, 12)]], 1.5, 9, 17), (12, 13.5))

    def test_find_slot_invalid_duration(self):
        with self.assertRaises(ValueError):
            find_slot([[]], 0, 9, 17)
""",
)

task(
    "text-template",
    "medium",
    "render_template",
    r"""
Implement a tiny template engine: `render_template(template, context)` returns the template string with every placeholder replaced.

A placeholder is written `{{ expression }}` (it never spans lines; spaces just inside the braces are optional). Text outside placeholders, including single braces, is copied unchanged. The expression is a path optionally followed by filters separated by `|`; whitespace around each part is ignored.

Paths. A path is names separated by dots, resolved against `context`: for a dict step, look the name up as a key; for a list or tuple step, the name must be a non-negative integer index. If a step cannot be resolved, or the final value is `None`, the value is considered missing. Example: `user.tags.1`.

Filters, applied left to right:
- `upper`, `lower`, `title`: apply `str.upper`, `str.lower`, `str.title` to `str(value)`.
- `length`: `len(value)`.
- `default:"text"`: if the value is missing, it becomes the given text; otherwise it is unchanged. The text is wrapped in double quotes and contains no `|` or `"`; if the argument is not wrapped in double quotes raise `ValueError`.
Applying `upper`, `lower`, `title` or `length` to a missing value raises `KeyError(path)`. An unknown filter name raises `ValueError`.

After all filters, a missing value raises `KeyError(path)`; otherwise the placeholder is replaced by `str(value)` (so `0`, `False` and the empty string are valid values, not missing).

Examples:
`render_template("Hi {{ user.name|title }}, {{ n|default:"zero" }}", {"user": {"name": "ann lee"}})` returns `"Hi Ann Lee, zero"`.
""",
    r"""
import re

FILTERS = {"upper": lambda v: str(v).upper(), "lower": lambda v: str(v).lower(),
           "title": lambda v: str(v).title(), "length": len}


def _lookup(context, path):
    cur = context
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, (list, tuple)) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return None
    return cur


def render_template(template, context):
    def replace(match):
        path, *filters = [p.strip() for p in match.group(1).split("|")]
        value = _lookup(context, path)
        for flt in filters:
            name, _, arg = (x.strip() for x in flt.partition(":"))
            if name == "default":
                if len(arg) < 2 or arg[0] != '"' or arg[-1] != '"':
                    raise ValueError(f"bad default: {flt}")
                value = arg[1:-1] if value is None else value
            elif name in FILTERS:
                if value is None:
                    raise KeyError(path)
                value = FILTERS[name](value)
            else:
                raise ValueError(f"unknown filter: {name}")
        if value is None:
            raise KeyError(path)
        return str(value)

    return re.sub(r"\{\{(.*?)\}\}", replace, template)
""",
    r"""
import unittest


class TestTemplate(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(render_template("Hello {{ name }}!", {"name": "Ann"}), "Hello Ann!")

    def test_dotted_and_index(self):
        ctx = {"user": {"name": "Bo", "tags": ["a", "b", "c"]}}
        self.assertEqual(render_template("{{ user.name }} {{ user.tags.1 }}", ctx), "Bo b")

    def test_tuple_index_and_tight_braces(self):
        self.assertEqual(render_template("{{p.0}},{{   p.1   }}", {"p": (3, 4)}), "3,4")

    def test_filters(self):
        ctx = {"s": "hello World", "items": [1, 2, 3]}
        self.assertEqual(render_template("{{ s|upper }}", ctx), "HELLO WORLD")
        self.assertEqual(render_template("{{ s|lower }}", ctx), "hello world")
        self.assertEqual(render_template("{{ s|title }}", ctx), "Hello World")
        self.assertEqual(render_template("{{ items|length }}", ctx), "3")

    def test_filter_chain(self):
        self.assertEqual(render_template("{{ s | lower | title }}", {"s": "aBC dEF"}), "Abc Def")

    def test_default_used_when_missing(self):
        self.assertEqual(render_template('{{ nope|default:"n/a" }}', {}), "n/a")
        self.assertEqual(render_template('{{ a.b.c|default:"-" }}', {"a": {"b": 1}}), "-")
        self.assertEqual(render_template('{{ v|default:"d" }}', {"v": None}), "d")

    def test_default_ignored_when_present(self):
        self.assertEqual(render_template('{{ v|default:"d" }}', {"v": "real"}), "real")

    def test_default_then_filter(self):
        self.assertEqual(render_template('{{ m|default:"x y"|upper }}', {}), "X Y")

    def test_falsy_values_are_not_missing(self):
        ctx = {"z": 0, "f": False, "e": ""}
        self.assertEqual(render_template("[{{ z }}][{{ f }}][{{ e }}]", ctx), "[0][False][]")
        self.assertEqual(render_template('[{{ z|default:"d" }}]', ctx), "[0]")

    def test_numbers_and_literal_text(self):
        self.assertEqual(render_template("{a} {{ n }} {b}\n{{ n }}", {"n": 42}), "{a} 42 {b}\n42")
        self.assertEqual(render_template("plain", {}), "plain")

    def test_missing_raises_keyerror(self):
        with self.assertRaises(KeyError):
            render_template("{{ x }}", {})
        with self.assertRaises(KeyError):
            render_template("{{ x|upper }}", {})
        with self.assertRaises(KeyError):
            render_template("{{ l.5 }}", {"l": [1]})
        with self.assertRaises(KeyError):
            render_template("{{ n }}", {"n": None})

    def test_bad_filters(self):
        with self.assertRaises(ValueError):
            render_template("{{ a|reverse }}", {"a": "x"})
        with self.assertRaises(ValueError):
            render_template("{{ a|default:x }}", {"a": "x"})
""",
)

task(
    "csv-summary",
    "medium",
    "summarize",
    r"""
Write three functions for summarising CSV data with the standard library.

`parse_rows(text)` parses CSV text whose first line is a header and returns a list of dicts (column name -> string value), one per data row, using normal CSV quoting rules (a quoted field may contain commas). Completely blank lines are skipped. Header only means an empty list.

`summarize(rows, group_col, value_col)` groups the rows by the text in `group_col` and returns a dict mapping each group to `{"count": int, "total": float, "mean": float}`. The text in `value_col` is stripped and converted with `float`; rows whose stripped value is empty are skipped entirely (they do not count). A non-empty value that is not a number raises `ValueError`. A group with no usable rows does not appear. `total` is the sum rounded to 2 decimals and `mean` is the unrounded sum divided by the count, rounded to 2 decimals. A missing column raises `KeyError`.

`format_summary(summary)` renders such a dict as a fixed-width text table and returns it as a string without a trailing newline. The first line is the header with columns `group`, `count`, `total`, `mean`, then one line per group sorted by total descending, ties by group name ascending. Column formats: group left-aligned in 10 characters, count right-aligned in 6, total right-aligned in 12, mean right-aligned in 10; the numbers in total and mean use 2 decimals. Longer group names are not truncated. Lines are joined with "\n".
""",
    r"""
import csv
import io


def parse_rows(text):
    return list(csv.DictReader(io.StringIO(text)))


def summarize(rows, group_col, value_col):
    totals, counts = {}, {}
    for row in rows:
        group = row[group_col]
        raw = row[value_col].strip()
        if not raw:
            continue
        totals[group] = totals.get(group, 0.0) + float(raw)
        counts[group] = counts.get(group, 0) + 1
    result = {}
    for group, total in totals.items():
        result[group] = {
            "count": counts[group],
            "total": round(total, 2),
            "mean": round(total / counts[group], 2),
        }
    return result


def format_summary(summary):
    pattern = "%-10s%6s%12s%10s"
    lines = [pattern % ("group", "count", "total", "mean")]
    order = sorted(summary, key=lambda g: (-summary[g]["total"], g))
    for group in order:
        s = summary[group]
        lines.append(
            "%-10s%6d%12.2f%10.2f" % (group, s["count"], s["total"], s["mean"])
        )
    return "\n".join(lines)
""",
    r"""
import unittest

CSV = "region,units,price\nnorth,3,10.5\nsouth,2,4\nnorth,1,2\n\nsouth,,9\n"


class TestCsvSummary(unittest.TestCase):
    def test_parse(self):
        rows = parse_rows(CSV)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0], {"region": "north", "units": "3", "price": "10.5"})
        self.assertEqual(rows[3], {"region": "south", "units": "", "price": "9"})

    def test_parse_quoted(self):
        rows = parse_rows('name,amount\n"Smith, J",5\n')
        self.assertEqual(rows, [{"name": "Smith, J", "amount": "5"}])

    def test_parse_header_only(self):
        self.assertEqual(parse_rows("a,b\n"), [])

    def test_summarize_price(self):
        s = summarize(parse_rows(CSV), "region", "price")
        self.assertEqual(s["north"], {"count": 2, "total": 12.5, "mean": 6.25})
        self.assertEqual(s["south"], {"count": 2, "total": 13.0, "mean": 6.5})

    def test_summarize_skips_empty_values(self):
        s = summarize(parse_rows(CSV), "region", "units")
        self.assertEqual(s["south"], {"count": 1, "total": 2.0, "mean": 2.0})
        self.assertEqual(s["north"]["count"], 2)

    def test_group_without_values_absent(self):
        s = summarize([{"g": "a", "v": " "}, {"g": "b", "v": "1"}], "g", "v")
        self.assertEqual(list(s), ["b"])
        self.assertEqual(summarize([{"g": "a", "v": " 3 "}], "g", "v")["a"]["total"], 3.0)

    def test_rounding(self):
        rows = [{"g": "a", "v": "0.1"}, {"g": "a", "v": "0.2"}, {"g": "a", "v": "0.2"}]
        s = summarize(rows, "g", "v")["a"]
        self.assertEqual((s["count"], s["total"], s["mean"]), (3, 0.5, 0.17))

    def test_bad_number(self):
        with self.assertRaises(ValueError):
            summarize([{"g": "a", "v": "abc"}], "g", "v")

    def test_missing_column(self):
        with self.assertRaises(KeyError):
            summarize([{"g": "a", "v": "1"}], "g", "nope")

    def test_format(self):
        s = summarize(parse_rows(CSV), "region", "price")
        expected = "\n".join([
            "%-10s%6s%12s%10s" % ("group", "count", "total", "mean"),
            "%-10s%6s%12s%10s" % ("south", "2", "13.00", "6.50"),
            "%-10s%6s%12s%10s" % ("north", "2", "12.50", "6.25"),
        ])
        self.assertEqual(format_summary(s), expected)

    def test_format_ties_and_empty(self):
        s = {"b": {"count": 1, "total": 5.0, "mean": 5.0}, "a": {"count": 1, "total": 5.0, "mean": 5.0}}
        lines = format_summary(s).split("\n")
        self.assertTrue(lines[1].startswith("a "))
        self.assertTrue(lines[2].startswith("b "))
        self.assertEqual(format_summary({}), "group      count       total      mean")

    def test_format_long_name(self):
        out = format_summary({"averyverylongname": {"count": 1, "total": 1.0, "mean": 1.0}})
        self.assertEqual(out.split("\n")[1], "averyverylongname" + "     1" + "        1.00" + "      1.00")
""",
)

task(
    "log-analysis",
    "medium",
    "summarize_logs",
    r"""
Write four functions for analysing application logs. A valid log line looks like

`2024-03-05T10:15:30 ERROR [db] connection lost`

that is: a timestamp `YYYY-MM-DDTHH:MM:SS` (digits only), one space, a level which is one of `DEBUG`, `INFO`, `WARN`, `ERROR`, one space, a component name in square brackets (letters, digits, `_`, `.`, `-`; at least one character), one space, and a non-empty message that runs to the end of the line. Trailing whitespace (including a newline) is ignored; leading whitespace makes the line invalid.

`parse_log_line(line)` returns `{"time": ..., "level": ..., "component": ..., "message": ...}` (all strings) for a valid line, or `None` for anything else.

`summarize_logs(lines)` takes an iterable of lines and returns `{"by_level": {level: count}, "by_component": {component: count}, "malformed": n}`. Only levels and components that occur appear as keys. Lines that are empty or whitespace-only are ignored altogether; other invalid lines are counted in `malformed`.

`top_errors(lines, n=3)` returns a list of at most `n` tuples `(message, count)` for the most frequent messages among valid `ERROR` lines, sorted by count descending and then by message ascending.

`busiest_hour(lines)` returns the hour prefix `"YYYY-MM-DDTHH"` that has the most valid lines (any level); ties go to the earliest hour; `None` if there are no valid lines.
""",
    r"""
import re
from collections import Counter

LINE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}) (DEBUG|INFO|WARN|ERROR) \[([A-Za-z0-9_.-]+)\] (.+)")


def parse_log_line(line):
    m = LINE.fullmatch(line.rstrip())
    if not m:
        return None
    return dict(zip(("time", "level", "component", "message"), m.groups()))


def summarize_logs(lines):
    levels, comps, bad = Counter(), Counter(), 0
    for line in lines:
        if not line.strip():
            continue
        rec = parse_log_line(line)
        if rec is None:
            bad += 1
        else:
            levels[rec["level"]] += 1
            comps[rec["component"]] += 1
    return {"by_level": dict(levels), "by_component": dict(comps), "malformed": bad}


def top_errors(lines, n=3):
    recs = (parse_log_line(line) for line in lines)
    counts = Counter(r["message"] for r in recs if r and r["level"] == "ERROR")
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]


def busiest_hour(lines):
    counts = Counter(r["time"][:13] for r in map(parse_log_line, lines) if r)
    return min(counts, key=lambda h: (-counts[h], h)) if counts else None
""",
    r"""
import unittest

LOG = [
    "2024-03-05T10:15:30 ERROR [db] connection lost",
    "2024-03-05T10:16:00 INFO [web] request ok\n",
    "",
    "   ",
    "garbage line",
    "2024-03-05T10:17:10 ERROR [db] connection lost",
    "2024-03-05T11:00:00 WARN [web.api] slow response  ",
    "2024-03-05T11:01:00 ERROR [web] timeout",
    "2024-03-05T11:02:00 ERROR [web] bad gateway",
    "2024-03-05T11:03:00 ERROR [db] connection lost",
    "2024-03-05 11:03:00 ERROR [db] wrong timestamp",
    "2024-03-05T11:03:00 FATAL [db] unknown level",
    "2024-03-05T11:03:00 INFO [] empty component",
    "2024-03-05T11:03:00 INFO [db]",
]


class TestLogs(unittest.TestCase):
    def test_parse_valid(self):
        self.assertEqual(
            parse_log_line("2024-03-05T10:15:30 ERROR [db] connection lost"),
            {"time": "2024-03-05T10:15:30", "level": "ERROR", "component": "db", "message": "connection lost"},
        )

    def test_parse_trailing_whitespace(self):
        r = parse_log_line("2024-03-05T10:15:30 INFO [a.b-c_d] hello   \n")
        self.assertEqual((r["component"], r["message"]), ("a.b-c_d", "hello"))

    def test_parse_invalid(self):
        for bad in [
            "",
            " 2024-03-05T10:15:30 INFO [db] leading space",
            "2024-03-05T10:15:30 info [db] lower level",
            "2024-03-05T10:15:30 INFO db no brackets",
            "2024-03-05T10:15:30 INFO [db] ",
            "2024-3-5T10:15:30 INFO [db] short date",
            "2024-03-05T10:15:30 INFO [d b] space in component",
        ]:
            self.assertIsNone(parse_log_line(bad), bad)

    def test_message_keeps_inner_spaces_and_brackets(self):
        r = parse_log_line("2024-03-05T10:15:30 WARN [x] a  [b] c")
        self.assertEqual(r["message"], "a  [b] c")

    def test_summarize(self):
        s = summarize_logs(LOG)
        self.assertEqual(s["by_level"], {"ERROR": 5, "INFO": 1, "WARN": 1})
        self.assertEqual(s["by_component"], {"db": 3, "web": 3, "web.api": 1})
        self.assertEqual(s["malformed"], 5)

    def test_summarize_empty(self):
        self.assertEqual(summarize_logs([]), {"by_level": {}, "by_component": {}, "malformed": 0})
        self.assertEqual(summarize_logs(["", "  \n"])["malformed"], 0)

    def test_summarize_accepts_generator(self):
        self.assertEqual(summarize_logs(iter(LOG[:2]))["by_level"], {"ERROR": 1, "INFO": 1})

    def test_top_errors(self):
        self.assertEqual(top_errors(LOG), [("connection lost", 3), ("bad gateway", 1), ("timeout", 1)])

    def test_top_errors_n(self):
        self.assertEqual(top_errors(LOG, 1), [("connection lost", 3)])
        self.assertEqual(top_errors(LOG, n=10)[-1], ("timeout", 1))
        self.assertEqual(len(top_errors(LOG, n=10)), 3)

    def test_top_errors_none(self):
        self.assertEqual(top_errors(["2024-03-05T10:15:30 INFO [x] fine"]), [])

    def test_busiest_hour(self):
        self.assertEqual(busiest_hour(LOG), "2024-03-05T11")

    def test_busiest_hour_tie_and_none(self):
        lines = ["2024-03-06T09:00:00 INFO [a] x", "2024-03-05T23:00:00 INFO [a] y"]
        self.assertEqual(busiest_hour(lines), "2024-03-05T23")
        self.assertIsNone(busiest_hour(["junk"]))
""",
)

task(
    "bill-split",
    "medium",
    "settle",
    r"""
Write three functions for sharing money, all in integer cents.

`split_evenly(total, n)` splits `total` cents among `n` people and returns a list of `n` ints that sum to `total` and differ by at most 1 cent; the first `total % n` people receive the extra cent. Raise `ValueError` if `n < 1` or `total < 0`.

`allocate(total, weights)` splits `total` cents proportionally to a list of non-negative integer weights using the largest remainder method: each share starts at `total * w // W` (where `W = sum(weights)`), and the cents still unassigned are handed out one each to the entries with the largest remainder `total * w % W`, ties going to the lower index. The result is a list of ints aligned with `weights` and summing to `total`. Raise `ValueError` if `weights` is empty, any weight is negative, `W` is zero, or `total < 0`.

`settle(balances)` takes a dict mapping person name to a net balance in cents (positive: is owed money; negative: owes money) and returns the list of payments `(debtor, creditor, amount)` that settles everyone. Raise `ValueError` if the balances do not sum to zero. People with a zero balance are ignored. Repeat until nobody is left: pick the debtor who owes the most (ties: the name that sorts first) and the creditor who is owed the most (ties: the name that sorts first); that debtor pays that creditor `min(debt, credit)` cents; the payment is appended to the result and the balances updated.

Example: `settle({"a": 50, "b": -30, "c": -20})` is `[("b", "a", 30), ("c", "a", 20)]`.
""",
    r"""
def split_evenly(total, n):
    if n < 1 or total < 0:
        raise ValueError("invalid arguments")
    return [total // n + (1 if i < total % n else 0) for i in range(n)]


def allocate(total, weights):
    whole = sum(weights)
    if total < 0 or not weights or min(weights) < 0 or whole == 0:
        raise ValueError("invalid arguments")
    shares = [total * w // whole for w in weights]
    order = sorted(range(len(weights)), key=lambda i: (-(total * weights[i] % whole), i))
    for i in order[: total - sum(shares)]:
        shares[i] += 1
    return shares


def settle(balances):
    if sum(balances.values()) != 0:
        raise ValueError("balances must sum to zero")
    bal = {k: v for k, v in balances.items() if v}
    payments = []
    while bal:
        debtor = min(bal, key=lambda k: (bal[k], k))
        creditor = min(bal, key=lambda k: (-bal[k], k))
        amount = min(-bal[debtor], bal[creditor])
        payments.append((debtor, creditor, amount))
        for k, delta in ((debtor, amount), (creditor, -amount)):
            bal[k] += delta
            if bal[k] == 0:
                del bal[k]
    return payments
""",
    r"""
import unittest


class TestSplit(unittest.TestCase):
    def test_even(self):
        self.assertEqual(split_evenly(100, 3), [34, 33, 33])
        self.assertEqual(split_evenly(10, 5), [2] * 5)
        self.assertEqual(split_evenly(0, 3), [0, 0, 0])
        self.assertEqual(split_evenly(2, 5), [1, 1, 0, 0, 0])

    def test_even_invalid(self):
        with self.assertRaises(ValueError):
            split_evenly(10, 0)
        with self.assertRaises(ValueError):
            split_evenly(-1, 2)

    def test_allocate_exact(self):
        self.assertEqual(allocate(100, [50, 30, 20]), [50, 30, 20])

    def test_allocate_ties_lower_index(self):
        self.assertEqual(allocate(100, [1, 1, 1]), [34, 33, 33])

    def test_allocate_largest_remainder(self):
        self.assertEqual(allocate(10, [1, 2]), [3, 7])
        self.assertEqual(allocate(5, [0, 1, 1]), [0, 3, 2])

    def test_allocate_sums(self):
        for total in (0, 1, 7, 99, 1001):
            shares = allocate(total, [3, 5, 11, 0, 2])
            self.assertEqual(sum(shares), total)
            self.assertEqual(shares[3], 0)

    def test_allocate_invalid(self):
        for total, w in [(10, []), (10, [0, 0]), (10, [1, -1]), (-5, [1])]:
            with self.assertRaises(ValueError):
                allocate(total, w)

    def test_settle_example(self):
        self.assertEqual(settle({"a": 50, "b": -30, "c": -20}), [("b", "a", 30), ("c", "a", 20)])

    def test_settle_ties(self):
        self.assertEqual(
            settle({"a": 10, "b": 10, "c": -15, "d": -5}),
            [("c", "a", 10), ("c", "b", 5), ("d", "b", 5)],
        )

    def test_settle_zero_ignored(self):
        self.assertEqual(settle({"x": 0, "y": 0}), [])
        self.assertEqual(settle({"x": 0, "y": 5, "z": -5}), [("z", "y", 5)])

    def test_settle_must_sum_to_zero(self):
        with self.assertRaises(ValueError):
            settle({"a": 5, "b": -4})

    def test_settle_clears_everyone(self):
        bal = {"a": 70, "b": -20, "c": -55, "d": 5}
        net = dict(bal)
        for debtor, creditor, amount in settle(bal):
            self.assertGreater(amount, 0)
            net[debtor] += amount
            net[creditor] -= amount
        self.assertEqual(set(net.values()), {0})
        self.assertEqual(bal, {"a": 70, "b": -20, "c": -55, "d": 5})
""",
)

# ---------------------------------------------------------------- LARGE

task(
    "cron-schedule",
    "large",
    "next_run",
    r"""
Build a small cron-expression library with four public names: `CronError`, `parse_cron`, `matches` and `next_run`. All datetimes are naive `datetime.datetime` objects; seconds and microseconds are ignored when matching.

`CronError` is an exception class and a subclass of `ValueError`; every problem with an expression raises it.

`parse_cron(expr)` parses a five-field expression `minute hour day-of-month month day-of-week` (fields separated by whitespace; leading and trailing whitespace is ignored) and returns a `collections.namedtuple` called `CronSpec` with the fields `minutes, hours, days, months, weekdays, dom_star, dow_star`. The first five are sorted tuples of ints; the last two are booleans that are True when the day-of-month (respectively day-of-week) field text starts with `*`.

Allowed ranges: minute 0-59, hour 0-23, day-of-month 1-31, month 1-12, day-of-week 0-7 where both 0 and 7 mean Sunday (after parsing, `weekdays` contains only 0-6, with 7 turned into 0; 0 is Sunday and 6 is Saturday). A field is a comma-separated list of parts, and each part is `*`, a single value `N`, or an inclusive range `A-B`, optionally followed by `/S` with an integer step `S >= 1` (take every S-th value starting from the first value of the part; a single value with a step is just that value). In the month field the values may also be written as three-letter names `JAN`..`DEC`, and in the day-of-week field as `SUN`..`SAT`, in any letter case, anywhere a number is allowed (including as range ends, such as `mon-fri`). Raise `CronError` for a wrong number of fields, non-numeric or unknown text, empty parts, values out of range, a range whose start is greater than its end, or a step below 1.

Macros: a whole expression may instead be one of `@hourly` (`0 * * * *`), `@daily` and `@midnight` (`0 0 * * *`), `@weekly` (`0 0 * * 0`), `@monthly` (`0 0 1 * *`), `@yearly` and `@annually` (`0 0 1 1 *`), in any letter case. Any other text starting with `@` is an error.

`matches(spec, dt)` is True when the minute, hour and month of `dt` are in the spec and the day rule holds. The day rule: if either `dom_star` or `dow_star` is True, then both the day of month and the weekday must be in the spec (a starred field accepts everything it lists, so this is the usual AND); if neither is starred, a date matches when either its day of month is in `days` or its weekday is in `weekdays` (OR). The weekday of a date is Sunday=0 .. Saturday=6.

`next_run(spec, after)` returns the earliest matching datetime that is strictly later than `after`, with zero seconds and microseconds. It must not step minute by minute through years; searching day by day is fine. If nothing matches within 8 years of `after` (for example `0 0 31 2 *`), raise `CronError`.

Examples: `parse_cron("*/15 0 1,15 jan-mar mon").minutes == (0, 15, 30, 45)`; `next_run(parse_cron("*/15 * * * *"), datetime(2024, 3, 5, 10, 15))` is `datetime(2024, 3, 5, 10, 30)`.
""",
    r"""
from collections import namedtuple
from datetime import timedelta


class CronError(ValueError):
    pass


CronSpec = namedtuple("CronSpec", "minutes hours days months weekdays dom_star dow_star")

MONTHS = {n: i for i, n in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split(), 1)}
WEEKDAYS = {n: i for i, n in enumerate("SUN MON TUE WED THU FRI SAT".split())}
MACROS = {
    "@hourly": "0 * * * *",
    "@daily": "0 0 * * *",
    "@midnight": "0 0 * * *",
    "@weekly": "0 0 * * 0",
    "@monthly": "0 0 1 * *",
    "@yearly": "0 0 1 1 *",
    "@annually": "0 0 1 1 *",
}
FIELDS = [(0, 59, {}), (0, 23, {}), (1, 31, {}), (1, 12, MONTHS), (0, 7, WEEKDAYS)]


def _number(text, names):
    if text.upper() in names:
        return names[text.upper()]
    if text.isdigit():
        return int(text)
    raise CronError(f"bad value {text!r}")


def _expand(field, lo, hi, names):
    values = set()
    for part in field.split(","):
        rng, slash, step = part.partition("/")
        if slash and not step.isdigit():
            raise CronError(f"bad step in {part!r}")
        step = int(step) if slash else 1
        if rng == "*":
            first, last = lo, hi
        else:
            a, dash, b = rng.partition("-")
            first = _number(a, names)
            last = _number(b, names) if dash else first
        if step < 1 or first < lo or last > hi or first > last:
            raise CronError(f"out of range: {part!r}")
        values.update(range(first, last + 1, step))
    return values


def parse_cron(expr):
    expr = expr.strip()
    if expr.startswith("@"):
        if expr.lower() not in MACROS:
            raise CronError(f"unknown macro {expr!r}")
        expr = MACROS[expr.lower()]
    parts = expr.split()
    if len(parts) != 5:
        raise CronError("expected 5 fields")
    sets = [_expand(p, lo, hi, names) for p, (lo, hi, names) in zip(parts, FIELDS)]
    weekdays = {v % 7 for v in sets[4]}
    columns = [sorted(s) for s in sets[:4]] + [sorted(weekdays)]
    return CronSpec(*map(tuple, columns), parts[2].startswith("*"), parts[4].startswith("*"))


def _day_matches(spec, day):
    if day.month not in spec.months:
        return False
    in_dom = day.day in spec.days
    in_dow = (day.weekday() + 1) % 7 in spec.weekdays
    if spec.dom_star or spec.dow_star:
        return in_dom and in_dow
    return in_dom or in_dow


def matches(spec, dt):
    return dt.minute in spec.minutes and dt.hour in spec.hours and _day_matches(spec, dt)


def next_run(spec, after):
    start = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
    day = start.replace(hour=0, minute=0)
    for _ in range(366 * 8):
        if _day_matches(spec, day):
            for hour in spec.hours:
                for minute in spec.minutes:
                    candidate = day.replace(hour=hour, minute=minute)
                    if candidate >= start:
                        return candidate
        day += timedelta(days=1)
    raise CronError("no run time within 8 years")
""",
    r"""
import unittest
from datetime import datetime

class TestParse(unittest.TestCase):

    def test_fields(self):
        s = parse_cron('*/15 0 1,15 jan-mar mon')
        self.assertEqual(s.minutes, (0, 15, 30, 45))
        self.assertEqual(s.hours, (0,))
        self.assertEqual(s.days, (1, 15))
        self.assertEqual(s.months, (1, 2, 3))
        self.assertEqual(s.weekdays, (1,))
        self.assertEqual((s.dom_star, s.dow_star), (False, False))

    def test_star_flags_and_ranges(self):
        s = parse_cron('0 */6 */10 * *')
        self.assertEqual(s.hours, (0, 6, 12, 18))
        self.assertEqual(s.days, (1, 11, 21, 31))
        self.assertEqual(s.months, tuple(range(1, 13)))
        self.assertEqual(s.weekdays, tuple(range(7)))
        self.assertEqual((s.dom_star, s.dow_star), (True, True))

    def test_whitespace_and_case_and_sunday_seven(self):
        self.assertEqual(parse_cron('  5   4 * DEC  SUN '), parse_cron('5 4 * 12 0'))
        self.assertEqual(parse_cron('0 0 * * 7'), parse_cron('0 0 * * 0'))
        self.assertEqual(parse_cron('0 0 * * 5-7').weekdays, (0, 5, 6))
        self.assertEqual(parse_cron('0 0 * * mon-fri').weekdays, (1, 2, 3, 4, 5))

    def test_macros_and_errors(self):
        self.assertEqual(parse_cron('@daily'), parse_cron('0 0 * * *'))
        self.assertEqual(parse_cron('@MIDNIGHT'), parse_cron('0 0 * * *'))
        self.assertEqual(parse_cron('@hourly'), parse_cron('0 * * * *'))
        self.assertEqual(parse_cron('@weekly'), parse_cron('0 0 * * 0'))
        self.assertEqual(parse_cron('@monthly'), parse_cron('0 0 1 * *'))
        self.assertEqual(parse_cron('@Yearly'), parse_cron('0 0 1 1 *'))
        self.assertEqual(parse_cron('@annually'), parse_cron('0 0 1 1 *'))
        for bad in ['', '* * * *', '* * * * * *', '60 * * * *', '* 24 * * *', '* * 0 * *', '* * * 13 *', '* * * * 8', '*/0 * * * *', 'a * * * *', '* * * foo *', '@reboot', '5-1 * * * *', '1,,2 * * * *', '*/x * * * *']:
            with self.assertRaises(CronError, msg=bad):
                parse_cron(bad)
        self.assertTrue(issubclass(CronError, ValueError))

class TestMatch(unittest.TestCase):

    def test_time_fields(self):
        s = parse_cron('30 14 * * *')
        self.assertTrue(matches(s, datetime(2024, 3, 5, 14, 30)))
        self.assertTrue(matches(s, datetime(2024, 3, 5, 14, 30, 59, 5)))
        self.assertFalse(matches(s, datetime(2024, 3, 5, 14, 31)))
        self.assertFalse(matches(s, datetime(2024, 3, 5, 13, 30)))

    def test_month_and_dom_or_dow_when_both_restricted(self):
        s = parse_cron('0 0 1 6 *')
        self.assertTrue(matches(s, datetime(2024, 6, 1)))
        self.assertFalse(matches(s, datetime(2024, 7, 1)))
        s = parse_cron('0 0 13 * 5')
        self.assertTrue(matches(s, datetime(2024, 3, 8)))
        self.assertTrue(matches(s, datetime(2024, 3, 13)))
        self.assertTrue(matches(s, datetime(2024, 9, 13)))
        self.assertFalse(matches(s, datetime(2024, 3, 12)))

    def test_and_when_one_starred(self):
        mondays = parse_cron('0 0 * * 1')
        self.assertTrue(matches(mondays, datetime(2024, 3, 4)))
        self.assertFalse(matches(mondays, datetime(2024, 3, 5)))
        odd_mondays = parse_cron('0 0 */2 * 1')
        self.assertFalse(matches(odd_mondays, datetime(2024, 3, 4)))
        self.assertTrue(matches(odd_mondays, datetime(2024, 3, 11)))

    def test_sunday_is_zero(self):
        self.assertTrue(matches(parse_cron('0 0 * * 0'), datetime(2024, 3, 3)))
        self.assertFalse(matches(parse_cron('0 0 * * 0'), datetime(2024, 3, 4)))
        self.assertTrue(matches(parse_cron('0 0 * * 6'), datetime(2024, 3, 2)))

class TestNextRun(unittest.TestCase):

    def test_every_quarter_hour(self):
        s = parse_cron('*/15 * * * *')
        self.assertEqual(next_run(s, datetime(2024, 3, 5, 10, 7, 45)), datetime(2024, 3, 5, 10, 15))
        self.assertEqual(next_run(s, datetime(2024, 3, 5, 10, 15)), datetime(2024, 3, 5, 10, 30))
        self.assertEqual(next_run(s, datetime(2024, 3, 5, 10, 14, 59)), datetime(2024, 3, 5, 10, 15))
        self.assertEqual(next_run(s, datetime(2024, 3, 5, 10, 45, 1)), datetime(2024, 3, 5, 11, 0))

    def test_next_day_and_year(self):
        self.assertEqual(next_run(parse_cron('30 2 * * *'), datetime(2024, 3, 5, 3, 0)), datetime(2024, 3, 6, 2, 30))
        self.assertEqual(next_run(parse_cron('@daily'), datetime(2024, 12, 31, 12, 0)), datetime(2025, 1, 1))
        self.assertEqual(next_run(parse_cron('@yearly'), datetime(2024, 6, 1)), datetime(2025, 1, 1))

    def test_short_months_skipped_and_leap_day(self):
        self.assertEqual(next_run(parse_cron('59 23 31 * *'), datetime(2024, 4, 30)), datetime(2024, 5, 31, 23, 59))
        self.assertEqual(next_run(parse_cron('0 0 29 2 *'), datetime(2023, 1, 1)), datetime(2024, 2, 29))
        self.assertEqual(next_run(parse_cron('0 0 29 2 *'), datetime(2024, 2, 29)), datetime(2028, 2, 29))

    def test_weekdays_and_or_rule_and_impossible(self):
        s = parse_cron('0 9 * * mon-fri')
        self.assertEqual(next_run(s, datetime(2024, 3, 1, 9, 0)), datetime(2024, 3, 4, 9, 0))
        self.assertEqual(next_run(s, datetime(2024, 3, 1, 8, 59)), datetime(2024, 3, 1, 9, 0))
        self.assertEqual(next_run(parse_cron('0 0 13 * 5'), datetime(2024, 3, 9)), datetime(2024, 3, 13))
        self.assertEqual(next_run(parse_cron('0 0 13 * 5'), datetime(2024, 3, 13)), datetime(2024, 3, 15))
        with self.assertRaises(CronError):
            next_run(parse_cron('0 0 31 2 *'), datetime(2024, 1, 1))
""",
)

task(
    "ini-config",
    "large",
    "parse_config",
    r"""
Write a configuration-file parser for an INI-like format with interpolation and typed getters. The public names are `ConfigError`, `parse_config` and the class `Config`.

`ConfigError` is a subclass of `ValueError` with an attribute `line` holding the 1-based line number of the offending line, or `None` when the error is not tied to one line (for example a bad reference found later at lookup time).

`parse_config(text)` parses the text and returns a `Config`. Rules, line by line:
- Blank lines, and lines whose first non-space character is `#` or `;`, are ignored.
- A section header is a line `[name]` (after stripping surrounding whitespace); the name is stripped and must be non-empty. Section names are case-sensitive. A malformed header (missing `]`, empty name, anything after `]`) or a repeated section raises `ConfigError`.
- Any other line must be `key = value`, split at the first `=`. The key is stripped and lower-cased (keys are case-insensitive) and must be non-empty. The value is everything after the `=`, stripped. A key line before any section header, a line with no `=`, or a key repeated within one section raises `ConfigError`.
- Comments inside values: a `#` or `;` that is the first character of the value or follows whitespace starts a comment that is removed (and the value stripped again); a `#` or `;` inside a word (`a#b`) is kept.
- Quoted values: if the value starts with `'` or `"`, the value is the text up to the next matching quote, taken verbatim (including `#`, `;` and spaces); after the closing quote only whitespace or a comment may follow, otherwise `ConfigError`; a missing closing quote is a `ConfigError`.
All parse errors carry the line number.

A section named `defaults` supplies fallback values for every other section: a key missing from a section is looked up in `defaults`.

Interpolation happens lazily when values are read, not at parse time. `${key}` is replaced by the value of `key` in the same section (including fallbacks from `defaults`), `${section.key}` by the value of `key` in that section (the part before the first dot is the section name), and `$$` stands for a literal `$`; a `$` followed by anything else is left alone. Referenced values are interpolated recursively. An unknown reference, or a circular chain, raises `ConfigError` with `line` equal to `None`.

`Config` methods: `sections()` returns the section names in file order, excluding `defaults`; `has(section, key)` tells whether the key exists in the section or in `defaults` (False for an unknown section); `get(section, key, default=<missing>)` returns the interpolated string, or `default` if the key is missing and a default was given, otherwise raises `KeyError`; `get_int`, `get_bool` and `get_list(section, key, default=<missing>, sep=",")` convert the interpolated value, with the same missing-key behaviour (a given default is returned unchanged). `get_int` uses `int()`, `get_bool` accepts `true/yes/on/1` and `false/no/off/0` in any letter case, `get_list` splits on `sep`, strips each item and drops empty items; a value that cannot be converted raises `ConfigError` with `line` None. `items(section)` returns a dict of all keys of the section, including those from `defaults`, mapped to interpolated values, and raises `KeyError` for an unknown section.
""",
    r"""
import re


class ConfigError(ValueError):
    def __init__(self, message, line=None):
        super().__init__(f"line {line}: {message}" if line else message)
        self.line = line


_MISSING = object()
_TRUE = {"true", "yes", "on", "1"}
_FALSE = {"false", "no", "off", "0"}


def _parse_value(raw, lineno):
    raw = raw.strip()
    if raw[:1] in ("'", '"'):
        end = raw.find(raw[0], 1)
        if end < 0:
            raise ConfigError("unterminated quote", lineno)
        rest = raw[end + 1:].strip()
        if rest and rest[0] not in "#;":
            raise ConfigError("text after closing quote", lineno)
        return raw[1:end]
    return re.sub(r"(?:^|\s)[#;].*$", "", raw).strip()


def parse_config(text):
    data, current = {}, None
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped[0] in "#;":
            continue
        if stripped[0] == "[":
            name = stripped[1:-1].strip()
            if not stripped.endswith("]") or not name or "]" in name:
                raise ConfigError("bad section header", lineno)
            if name in data:
                raise ConfigError(f"duplicate section {name}", lineno)
            data[name] = {}
            current = name
            continue
        if current is None:
            raise ConfigError("key outside of a section", lineno)
        key, eq, raw = stripped.partition("=")
        key = key.strip().lower()
        if not eq or not key:
            raise ConfigError("expected key = value", lineno)
        if key in data[current]:
            raise ConfigError(f"duplicate key {key}", lineno)
        data[current][key] = _parse_value(raw, lineno)
    return Config(data)


class Config:
    def __init__(self, data):
        self._data = data

    def sections(self):
        return [name for name in self._data if name != "defaults"]

    def _raw(self, section, key):
        key = key.lower()
        if section not in self._data:
            return _MISSING
        if key in self._data[section]:
            return self._data[section][key]
        return self._data.get("defaults", {}).get(key, _MISSING)

    def has(self, section, key):
        return self._raw(section, key) is not _MISSING

    def _expand(self, section, key, stack):
        if (section, key) in stack:
            raise ConfigError(f"circular reference at {section}.{key}")
        raw = self._raw(section, key)
        if raw is _MISSING:
            raise KeyError((section, key))

        def replace(match):
            if match.group(0) == "$$":
                return "$"
            ref = match.group(1).strip()
            target_section, target_key = ref.split(".", 1) if "." in ref else (section, ref)
            try:
                return self._expand(target_section, target_key.lower(), stack + ((section, key),))
            except KeyError:
                raise ConfigError(f"unknown reference {ref}") from None

        return re.sub(r"\$\$|\$\{([^}]*)\}", replace, raw)

    def get(self, section, key, default=_MISSING):
        if not self.has(section, key):
            if default is _MISSING:
                raise KeyError((section, key))
            return default
        return self._expand(section, key.lower(), ())

    def _typed(self, section, key, default, convert, what):
        if not self.has(section, key):
            return self.get(section, key, default)
        value = self.get(section, key)
        try:
            return convert(value)
        except ValueError:
            raise ConfigError(f"{section}.{key}: {value!r} is not {what}") from None

    def get_int(self, section, key, default=_MISSING):
        return self._typed(section, key, default, int, "an integer")

    def get_bool(self, section, key, default=_MISSING):
        def convert(value):
            if value.lower() in _TRUE:
                return True
            if value.lower() in _FALSE:
                return False
            raise ValueError(value)

        return self._typed(section, key, default, convert, "a boolean")

    def get_list(self, section, key, default=_MISSING, sep=","):
        return self._typed(
            section, key, default, lambda v: [p.strip() for p in v.split(sep) if p.strip()], "a list"
        )

    def items(self, section):
        if section not in self._data:
            raise KeyError(section)
        keys = {} if section == "defaults" else dict(self._data.get("defaults", {}))
        keys.update(self._data[section])
        return {key: self._expand(section, key, ()) for key in keys}
""",
    r"""
import unittest
SAMPLE = '\n# global comment\n[defaults]\nTimeout = 30\nmode = prod\n\n[db]\nhost = localhost   # trailing comment\nPORT = 5432\nurl = "postgres://${host}:${port}/app"   ; quoted, with comment\ndebug = Yes\ntags = a, b ,, c\npath = C:\\data;x\nlabel = \'x # not a comment\'\nhash = a#b\n\n[web]\ntimeout = 5\nhome = ${db.host}/home\nprice = $$5\n'

class TestConfig(unittest.TestCase):

    def setUp(self):
        self.cfg = parse_config(SAMPLE)

    def test_sections_and_has(self):
        self.assertEqual(self.cfg.sections(), ['db', 'web'])
        self.assertTrue(self.cfg.has('db', 'host'))
        self.assertTrue(self.cfg.has('db', 'TIMEOUT'))
        self.assertFalse(self.cfg.has('db', 'nope'))
        self.assertFalse(self.cfg.has('nosuch', 'host'))

    def test_basic_get_and_case_insensitive_keys(self):
        self.assertEqual(self.cfg.get('db', 'host'), 'localhost')
        self.assertEqual(self.cfg.get('db', 'port'), '5432')
        self.assertEqual(self.cfg.get('db', 'Port'), '5432')

    def test_defaults_fallback(self):
        self.assertEqual(self.cfg.get('db', 'timeout'), '30')
        self.assertEqual(self.cfg.get('web', 'timeout'), '5')
        self.assertEqual(self.cfg.get('web', 'mode'), 'prod')

    def test_comments_and_quotes(self):
        self.assertEqual(self.cfg.get('db', 'label'), 'x # not a comment')
        self.assertEqual(self.cfg.get('db', 'hash'), 'a#b')
        self.assertEqual(self.cfg.get('db', 'path'), 'C:\\data;x')

    def test_interpolation(self):
        self.assertEqual(self.cfg.get('db', 'url'), 'postgres://localhost:5432/app')
        self.assertEqual(self.cfg.get('web', 'home'), 'localhost/home')
        self.assertEqual(self.cfg.get('web', 'price'), '$5')

    def test_typed_getters(self):
        self.assertEqual(self.cfg.get_int('db', 'port'), 5432)
        self.assertEqual(self.cfg.get_int('web', 'timeout'), 5)
        self.assertIs(self.cfg.get_bool('db', 'debug'), True)
        self.assertEqual(self.cfg.get_list('db', 'tags'), ['a', 'b', 'c'])
        self.assertEqual(self.cfg.get_list('db', 'path', sep=';'), ['C:\\data', 'x'])

    def test_missing_and_defaults(self):
        with self.assertRaises(KeyError):
            self.cfg.get('db', 'nope')
        with self.assertRaises(KeyError):
            self.cfg.get_int('db', 'nope')
        self.assertEqual(self.cfg.get('db', 'nope', 'd'), 'd')
        self.assertEqual(self.cfg.get_int('db', 'nope', 7), 7)
        self.assertIsNone(self.cfg.get_bool('db', 'nope', None))
        self.assertEqual(self.cfg.get_list('db', 'nope', []), [])

    def test_conversion_errors(self):
        with self.assertRaises(ConfigError) as cm:
            self.cfg.get_int('db', 'host')
        self.assertIsNone(cm.exception.line)
        with self.assertRaises(ConfigError):
            self.cfg.get_bool('db', 'host')

    def test_bool_spellings_and_items(self):
        cfg = parse_config('[s]\na=ON\nb=off\nc=1\nd=0\ne=TRUE\nf=No\n')
        got = [cfg.get_bool('s', k) for k in 'abcdef']
        self.assertEqual(got, [True, False, True, False, True, False])
        self.assertEqual(self.cfg.items('web'), {'timeout': '5', 'mode': 'prod', 'home': 'localhost/home', 'price': '$5'})
        with self.assertRaises(KeyError):
            self.cfg.items('nosuch')

    def test_lazy_reference_errors(self):
        cfg = parse_config('[a]\nx = ${missing}\ny = ${b.z}\np = ${q}\nq = ${p}\nok = fine\n[b]\n')
        self.assertEqual(cfg.get('a', 'ok'), 'fine')
        for key in ('x', 'y', 'p'):
            with self.assertRaises(ConfigError) as cm:
                cfg.get('a', key)
            self.assertIsNone(cm.exception.line)

    def test_parse_errors_carry_line_numbers(self):
        cases = {'x = 1\n': 1, '[a]\nk = 1\nk = 2\n': 3, '[a]\n\njust words\n': 3, '[a]\n[a]\n': 2, '[a\nk=1\n': 1, '[]\n': 1, "[a]\nk = 'open\n": 2, "[a]\nk = 'x' y\n": 2, '[a]\n= 5\n': 2}
        for text, line in cases.items():
            with self.assertRaises(ConfigError, msg=text) as cm:
                parse_config(text)
            self.assertEqual(cm.exception.line, line, text)

    def test_empty_value_and_empty_text(self):
        cfg = parse_config('[a]\nk =\nq = ; only a comment\n')
        self.assertEqual(cfg.get('a', 'k'), '')
        self.assertEqual(cfg.get('a', 'q'), '')
        self.assertEqual(parse_config('').sections(), [])
""",
)

task(
    "cart-promotions",
    "large",
    "Cart",
    r"""
Implement a shopping cart with promotions. All money is integer cents. The public names are `format_money`, `PercentOff`, `BuyXGetY`, `SpendDiscount` and `Cart`.

`format_money(cents)` returns a string such as `"$1,234.56"`: a dollar sign, the dollars with thousands separators (commas), a dot and exactly two digits of cents. Negative amounts put the minus sign before the dollar sign: `format_money(-50)` is `"-$0.50"`; `format_money(5)` is `"$0.05"`.

Promotions are small classes with a method `discount(lines, running)` returning a non-negative int number of cents. `lines` is a list of `(sku, unit_price, qty)` tuples sorted by sku, and `running` is the cart total after all previously applied promotions.
- `PercentOff(pct, sku=None)`: `pct` percent (an integer) of the original line totals: of the line for `sku` if given (0 when the cart has no such line), otherwise of the whole original subtotal; the result is rounded down (`base * pct // 100`).
- `BuyXGetY(sku, buy, free)`: for every complete group of `buy + free` units of `sku` in the cart, `free` units are free; the discount is `(qty // (buy + free)) * free * unit_price`. A cart without that sku gets 0.
- `SpendDiscount(threshold, off)`: `off` cents if `running >= threshold`, else 0.

`Cart(shipping_cents=0, free_shipping_over=None)` has these methods:
- `add(sku, unit_price, qty=1)`: add units of a product. `unit_price` must be a non-negative int and `qty` a positive int, otherwise `ValueError`. Adding a sku that is already in the cart with a different unit price raises `ValueError`; with the same price the quantities add up.
- `remove(sku, qty=None)`: remove `qty` units, or the whole line if `qty` is None. An unknown sku raises `KeyError`; removing more units than the cart holds, or `qty < 1`, raises `ValueError`. A line that reaches zero units is deleted.
- `item_count()`: total number of units in the cart.
- `add_promo(promo)`: append a promotion; promotions are applied in the order added.
- `quote()`: returns a dict with `"subtotal"` (sum of unit price times qty), `"discounts"` (a list of `(promotion class name, cents)` tuples, in application order, only for promotions whose discount is above 0), `"shipping"` and `"total"`. Each promotion is computed with the current running total, capped at the running total so it never goes below zero, and then subtracted from it. Shipping is 0 for an empty cart, 0 if `free_shipping_over` is not None and the running total after discounts is at least that value, and `shipping_cents` otherwise. The total is the running total plus shipping.
- `receipt()`: a multi-line string joined by `"\n"`: one line per cart line sorted by sku, formatted `"<sku> <qty> x <unit price> = <line total>"` with money formatted by `format_money`; then `"Subtotal: <money>"`; then for each discount `"Discount (<name>): -<money>"`; then `"Shipping: <money>"`; then `"Total: <money>"`.

Example: a cart with 3 of "A" at 1000 and 2 of "B" at 250, promotions `PercentOff(10)`, `BuyXGetY("A", 2, 1)` and `SpendDiscount(2000, 300)`, shipping 500 free over 2000, quotes subtotal 3500, discounts 350, 1000 and 300, shipping 500, total 2350.
""",
    r"""
def format_money(cents):
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}${cents // 100:,}.{cents % 100:02d}"


class PercentOff:
    def __init__(self, pct, sku=None):
        self.pct, self.sku = pct, sku

    def discount(self, lines, running):
        base = sum(p * q for s, p, q in lines if self.sku is None or s == self.sku)
        return base * self.pct // 100


class BuyXGetY:
    def __init__(self, sku, buy, free):
        self.sku, self.buy, self.free = sku, buy, free

    def discount(self, lines, running):
        for sku, price, qty in lines:
            if sku == self.sku:
                return (qty // (self.buy + self.free)) * self.free * price
        return 0


class SpendDiscount:
    def __init__(self, threshold, off):
        self.threshold, self.off = threshold, off

    def discount(self, lines, running):
        return self.off if running >= self.threshold else 0


class Cart:
    def __init__(self, shipping_cents=0, free_shipping_over=None):
        self.shipping_cents = shipping_cents
        self.free_shipping_over = free_shipping_over
        self._items = {}
        self._promos = []

    def add(self, sku, unit_price, qty=1):
        if unit_price < 0 or qty < 1:
            raise ValueError("invalid price or quantity")
        if sku in self._items:
            price, held = self._items[sku]
            if price != unit_price:
                raise ValueError(f"price mismatch for {sku}")
            qty += held
        self._items[sku] = (unit_price, qty)

    def remove(self, sku, qty=None):
        price, held = self._items[sku]
        if qty is None:
            qty = held
        if qty < 1 or qty > held:
            raise ValueError("invalid quantity")
        if qty == held:
            del self._items[sku]
        else:
            self._items[sku] = (price, held - qty)

    def item_count(self):
        return sum(q for _, q in self._items.values())

    def add_promo(self, promo):
        self._promos.append(promo)

    def _lines(self):
        return [(sku, p, q) for sku, (p, q) in sorted(self._items.items())]

    def quote(self):
        lines = self._lines()
        subtotal = sum(p * q for _, p, q in lines)
        running, discounts = subtotal, []
        for promo in self._promos:
            amount = min(promo.discount(lines, running), running)
            if amount > 0:
                discounts.append((type(promo).__name__, amount))
                running -= amount
        shipping = self.shipping_cents
        if not lines or (self.free_shipping_over is not None and running >= self.free_shipping_over):
            shipping = 0
        return {"subtotal": subtotal, "discounts": discounts, "shipping": shipping, "total": running + shipping}

    def receipt(self):
        q = self.quote()
        out = [f"{s} {n} x {format_money(p)} = {format_money(p * n)}" for s, p, n in self._lines()]
        out.append(f"Subtotal: {format_money(q['subtotal'])}")
        out += [f"Discount ({name}): -{format_money(amount)}" for name, amount in q["discounts"]]
        out.append(f"Shipping: {format_money(q['shipping'])}")
        out.append(f"Total: {format_money(q['total'])}")
        return "\n".join(out)
""",
    r"""
import unittest

def sample_cart(**kwargs):
    cart = Cart(**kwargs)
    cart.add('B', 250, 2)
    cart.add('A', 1000, 3)
    return cart

class TestMoney(unittest.TestCase):

    def test_format(self):
        self.assertEqual(format_money(0), '$0.00')
        self.assertEqual(format_money(5), '$0.05')
        self.assertEqual(format_money(123456789), '$1,234,567.89')
        self.assertEqual(format_money(-50), '-$0.50')
        self.assertEqual(format_money(100000), '$1,000.00')

class TestCart(unittest.TestCase):

    def test_add_merge_and_count(self):
        cart = sample_cart()
        cart.add('A', 1000, 2)
        self.assertEqual(cart.item_count(), 7)
        self.assertEqual(cart.quote()['subtotal'], 5500)

    def test_add_validation(self):
        cart = sample_cart()
        for args in [('A', 999), ('C', -1), ('C', 5, 0)]:
            with self.assertRaises(ValueError):
                cart.add(*args)
        self.assertEqual(cart.item_count(), 5)

    def test_remove(self):
        cart = sample_cart()
        cart.remove('A', 1)
        self.assertEqual(cart.item_count(), 4)
        cart.remove('B')
        self.assertEqual(cart.quote()['subtotal'], 2000)
        with self.assertRaises(KeyError):
            cart.remove('B')
        with self.assertRaises(ValueError):
            cart.remove('A', 3)
        with self.assertRaises(ValueError):
            cart.remove('A', 0)
        cart.remove('A', 2)
        self.assertEqual(cart.item_count(), 0)

    def test_quote_without_promos(self):
        cart = sample_cart(shipping_cents=500)
        self.assertEqual(cart.quote(), {'subtotal': 3500, 'discounts': [], 'shipping': 500, 'total': 4000})

    def test_full_example(self):
        cart = sample_cart(shipping_cents=500, free_shipping_over=2000)
        cart.add_promo(PercentOff(10))
        cart.add_promo(BuyXGetY('A', 2, 1))
        cart.add_promo(SpendDiscount(2000, 300))
        self.assertEqual(cart.quote(), {'subtotal': 3500, 'discounts': [('PercentOff', 350), ('BuyXGetY', 1000), ('SpendDiscount', 300)], 'shipping': 500, 'total': 2350})

    def test_free_shipping_threshold_and_empty_cart_has_no_shipping(self):
        cart = sample_cart(shipping_cents=500, free_shipping_over=3500)
        self.assertEqual(cart.quote()['shipping'], 0)
        cart.add_promo(PercentOff(1))
        self.assertEqual(cart.quote()['shipping'], 500)
        self.assertEqual(Cart(shipping_cents=500).quote(), {'subtotal': 0, 'discounts': [], 'shipping': 0, 'total': 0})

    def test_percent_off_sku_and_percent_rounds_down(self):
        cart = sample_cart()
        cart.add_promo(PercentOff(25, sku='B'))
        cart.add_promo(PercentOff(50, sku='ZZ'))
        self.assertEqual(cart.quote()['discounts'], [('PercentOff', 125)])
        cart = Cart()
        cart.add('A', 99)
        cart.add_promo(PercentOff(15))
        self.assertEqual(cart.quote()['total'], 99 - 14)

    def test_buy_x_get_y(self):
        cart = Cart()
        cart.add('A', 400, 7)
        cart.add_promo(BuyXGetY('A', 2, 1))
        cart.add_promo(BuyXGetY('Q', 1, 1))
        self.assertEqual(cart.quote()['discounts'], [('BuyXGetY', 800)])

    def test_spend_discount_uses_running_total(self):
        cart = sample_cart()
        cart.add_promo(PercentOff(50))
        cart.add_promo(SpendDiscount(2000, 100))
        self.assertEqual(cart.quote()['discounts'], [('PercentOff', 1750)])
        cart.add_promo(SpendDiscount(1700, 100))
        self.assertEqual(cart.quote()['discounts'], [('PercentOff', 1750), ('SpendDiscount', 100)])

    def test_discount_capped_at_running_total(self):
        cart = sample_cart(shipping_cents=500)
        cart.add_promo(SpendDiscount(0, 9999))
        cart.add_promo(SpendDiscount(0, 10))
        q = cart.quote()
        self.assertEqual(q['discounts'], [('SpendDiscount', 3500)])
        self.assertEqual(q['total'], 500)

    def test_receipt(self):
        cart = sample_cart(shipping_cents=500, free_shipping_over=2000)
        cart.add_promo(PercentOff(10))
        cart.add_promo(BuyXGetY('A', 2, 1))
        cart.add_promo(SpendDiscount(2000, 300))
        self.assertEqual(cart.receipt(), 'A 3 x $10.00 = $30.00\nB 2 x $2.50 = $5.00\nSubtotal: $35.00\nDiscount (PercentOff): -$3.50\nDiscount (BuyXGetY): -$10.00\nDiscount (SpendDiscount): -$3.00\nShipping: $5.00\nTotal: $23.50')
""",
)

task(
    "text-layout",
    "large",
    "wrap_text",
    r"""
Write a small plain-text layout module with seven public functions: `wrap_text`, `justify_text`, `align`, `truncate`, `table_row`, `indent_block` and `bullet_list`.

`wrap_text(text, width)` returns a list of lines. The text is first split on `"\n"` into paragraphs. Each paragraph is split into words with `str.split()` and wrapped greedily: a word goes on the current line if the line plus one space plus the word is at most `width` characters, otherwise it starts a new line. A paragraph with no words (empty or whitespace only) yields one empty line `""`, so blank lines are preserved. A word longer than `width` is cut into pieces of exactly `width` characters, the last piece possibly shorter; every piece except the last occupies a line of its own, and the last piece is then treated like a normal word that following words may join. Raise `ValueError` if `width < 1`.

`justify_text(text, width)` wraps the text the same way and returns one string (lines joined with `"\n"`) in which every line is stretched to exactly `width` characters by widening the gaps between words, except the last line of each paragraph and lines that hold a single word, which are left as they are (words joined by single spaces). The missing spaces are spread over the gaps as evenly as possible; if they do not divide evenly, the leftmost gaps get one extra space each. For example `justify_text("a bb ccc dddd", 11)` is `"a   bb  ccc\ndddd"`.

`align(line, width, mode)` pads `line` with spaces to `width`: `mode` `"left"` pads on the right, `"right"` pads on the left, `"center"` puts `(width - len(line)) // 2` spaces on the left and the rest on the right. A line already at least `width` long is returned unchanged. Any other mode raises `ValueError`.

`truncate(text, width, ellipsis="...")` returns `text` if it is at most `width` long; otherwise `text[:width - len(ellipsis)] + ellipsis`; if `width` is smaller than the ellipsis, it returns `ellipsis[:width]`. Raise `ValueError` if `width < 0`.

`table_row(cells, widths, aligns=None)` converts each cell with `str`, truncates it to its width (with the default ellipsis), aligns it with the matching entry of `aligns` (default: all `"left"`), and returns `"| "` + the cells joined by `" | "` + `" |"`.

`indent_block(lines, prefix, first_prefix=None)` returns a new list in which every non-empty line gets `prefix` in front, except the line at index 0, which gets `first_prefix` instead if it is given. Empty strings stay empty.

`bullet_list(items, width, bullet="* ")` wraps each item with `wrap_text` to the width `width - len(bullet)` and returns the combined list of lines, where the first line of every item starts with `bullet` and the other lines of that item are indented with as many spaces as the bullet is long (use the same rule as `indent_block` for empty lines). Raise `ValueError` if `width - len(bullet) < 1`.

Example: `table_row(["name", 42, "x"], [6, 5, 3], ["left", "right", "center"])` is `"| name   |    42 |  x  |"`.
""",
    r"""
def _wrap_words(paragraph, width):
    lines, current = [], []
    for word in paragraph.split():
        while len(word) > width:
            if current:
                lines.append(current)
                current = []
            lines.append([word[:width]])
            word = word[width:]
        if current and len(" ".join(current + [word])) > width:
            lines.append(current)
            current = []
        current.append(word)
    if current or not lines:
        lines.append(current)
    return lines


def wrap_text(text, width):
    if width < 1:
        raise ValueError("width must be positive")
    return [" ".join(words) for p in text.split("\n") for words in _wrap_words(p, width)]


def justify_text(text, width):
    if width < 1:
        raise ValueError("width must be positive")
    out = []
    for paragraph in text.split("\n"):
        lines = _wrap_words(paragraph, width)
        for i, words in enumerate(lines):
            if i == len(lines) - 1 or len(words) < 2:
                out.append(" ".join(words))
                continue
            gaps = len(words) - 1
            spaces = width - sum(map(len, words))
            base, extra = divmod(spaces, gaps)
            line = words[0]
            for j, word in enumerate(words[1:]):
                line += " " * (base + (1 if j < extra else 0)) + word
            out.append(line)
    return "\n".join(out)


def align(line, width, mode):
    if mode not in ("left", "right", "center"):
        raise ValueError(f"unknown mode {mode!r}")
    gap = max(0, width - len(line))
    if mode == "left":
        return line + " " * gap
    if mode == "right":
        return " " * gap + line
    return " " * (gap // 2) + line + " " * (gap - gap // 2)


def truncate(text, width, ellipsis="..."):
    if width < 0:
        raise ValueError("width must not be negative")
    if len(text) <= width:
        return text
    if width < len(ellipsis):
        return ellipsis[:width]
    return text[: width - len(ellipsis)] + ellipsis


def table_row(cells, widths, aligns=None):
    aligns = aligns or ["left"] * len(cells)
    parts = [align(truncate(str(c), w), w, a) for c, w, a in zip(cells, widths, aligns)]
    return "| " + " | ".join(parts) + " |"


def indent_block(lines, prefix, first_prefix=None):
    out = []
    for i, line in enumerate(lines):
        lead = first_prefix if i == 0 and first_prefix is not None else prefix
        out.append(lead + line if line else line)
    return out


def bullet_list(items, width, bullet="* "):
    inner = width - len(bullet)
    if inner < 1:
        raise ValueError("width too small for the bullet")
    out = []
    for item in items:
        out += indent_block(wrap_text(item, inner), " " * len(bullet), bullet)
    return out
""",
    r"""
import unittest

class TestWrap(unittest.TestCase):

    def test_greedy_and_exact_fit(self):
        self.assertEqual(wrap_text('the quick brown fox jumps over the lazy dog', 10), ['the quick', 'brown fox', 'jumps over', 'the lazy', 'dog'])
        self.assertEqual(wrap_text('abc def', 7), ['abc def'])
        self.assertEqual(wrap_text('abc def', 6), ['abc', 'def'])

    def test_paragraphs_and_blank_lines_and_extra_whitespace_collapsed(self):
        self.assertEqual(wrap_text('a b\n\nc', 10), ['a b', '', 'c'])
        self.assertEqual(wrap_text('', 5), [''])
        self.assertEqual(wrap_text('   \nx', 5), ['', 'x'])
        self.assertEqual(wrap_text('  a    b  ', 10), ['a b'])

    def test_long_word_split_and_invalid_width(self):
        self.assertEqual(wrap_text('abcdefghij kl', 4), ['abcd', 'efgh', 'ij', 'kl'])
        self.assertEqual(wrap_text('abcdefghij kl', 7), ['abcdefg', 'hij kl'])
        self.assertEqual(wrap_text('xy abcdefghij', 4), ['xy', 'abcd', 'efgh', 'ij'])
        with self.assertRaises(ValueError):
            wrap_text('a', 0)
        with self.assertRaises(ValueError):
            justify_text('a', 0)

class TestJustify(unittest.TestCase):

    def test_example_and_sentence(self):
        self.assertEqual(justify_text('a bb ccc dddd', 11), 'a   bb  ccc\ndddd')
        self.assertEqual(justify_text('the quick brown fox jumps over the lazy dog', 12), 'the    quick\nbrown    fox\njumps   over\nthe lazy dog')

    def test_last_and_single_word_lines_untouched_and_paragraphs(self):
        self.assertEqual(justify_text('abcdefghij kl', 4), 'abcd\nefgh\nij\nkl')
        self.assertEqual(justify_text('hello', 12), 'hello')
        self.assertEqual(justify_text('a bb ccc dddd\n\nxx yy', 11), 'a   bb  ccc\ndddd\n\nxx yy')

    def test_line_lengths(self):
        out = justify_text('one two three four five six seven eight nine ten', 15).split('\n')
        self.assertTrue(all((len(line) == 15 for line in out[:-1])))
        self.assertEqual(out[-1], 'nine ten')
        self.assertEqual(' '.join(' '.join(out).split()), 'one two three four five six seven eight nine ten')

class TestAlignTruncate(unittest.TestCase):

    def test_align(self):
        self.assertEqual(align('ab', 5, 'left'), 'ab   ')
        self.assertEqual(align('ab', 5, 'right'), '   ab')
        self.assertEqual(align('ab', 7, 'center'), '  ab   ')
        self.assertEqual(align('ab', 6, 'center'), '  ab  ')

    def test_align_too_long_and_invalid(self):
        self.assertEqual(align('abcdef', 3, 'right'), 'abcdef')
        with self.assertRaises(ValueError):
            align('a', 3, 'middle')

    def test_truncate(self):
        self.assertEqual(truncate('hello world', 8), 'hello...')
        self.assertEqual(truncate('hello', 5), 'hello')
        self.assertEqual(truncate('hello world', 3), '...')
        self.assertEqual(truncate('hello world', 2), '..')
        self.assertEqual(truncate('hello world', 0), '')
        self.assertEqual(truncate('hello world', 7, '~'), 'hello ~')
        with self.assertRaises(ValueError):
            truncate('x', -1)

class TestRowsAndIndent(unittest.TestCase):

    def test_table_row_and_table_row_default_align_and_truncate(self):
        self.assertEqual(table_row(['name', 42, 'x'], [6, 5, 3], ['left', 'right', 'center']), '| name   |    42 |  x  |')
        self.assertEqual(table_row(['abcdefghij', 'z'], [6, 2]), '| abc... | z  |')

    def test_indent_block(self):
        self.assertEqual(indent_block(['a', '', 'b'], '> '), ['> a', '', '> b'])
        self.assertEqual(indent_block(['a', '', 'b'], '> ', '- '), ['- a', '', '> b'])
        self.assertEqual(indent_block([], '> '), [])

    def test_bullet_list(self):
        self.assertEqual(bullet_list(['the quick brown fox', 'hi'], 12), ['* the quick', '  brown fox', '* hi'])
        self.assertEqual(bullet_list(['abcdef'], 6, '-- '), ['-- abc', '   def'])
        self.assertEqual(bullet_list([], 10), [])
        with self.assertRaises(ValueError):
            bullet_list(['x'], 2)
""",
)

task(
    "expression-evaluator",
    "large",
    "evaluate",
    r"""
Write a safe arithmetic expression evaluator without using `eval`, `exec` or `ast`. Public names: `ExprError`, `tokenize` and `evaluate`.

`ExprError` is an exception class (a subclass of `ValueError`) raised for every problem: bad characters, bad syntax, unknown names, wrong argument counts and arithmetic errors.

`tokenize(expr)` returns a list of `(kind, text)` tuples. Whitespace is skipped. The kinds are: `"num"` for an unsigned number written as digits optionally followed by a dot and digits (`12`, `3.5`; not `.5` or `3.`); `"name"` for an identifier (a letter or underscore followed by letters, digits or underscores); and `"op"` for one of the operators `+ - * / // % **` or the punctuation `( ) ,`. `//` and `**` are single tokens. Any other character raises `ExprError`.

`evaluate(expr, variables=None)` evaluates the expression and returns an int or a float. `variables` maps names to numbers. Grammar, from lowest to highest precedence:
- `expr`: terms joined by binary `+` and `-`, left associative.
- `term`: factors joined by `*`, `/`, `//` and `%`, left associative.
- `factor`: a unary `-` or `+` applied to a factor, or a power.
- `power`: an atom, optionally followed by `**` and a factor (so `**` is right associative and its exponent may carry a unary sign: `2 ** -1`). Because the unary sign sits below `**`, `-2 ** 2` is `-(2 ** 2)`, i.e. `-4`.
- `atom`: a number, a variable name, a function call `name(arg, arg, ...)`, or a parenthesised `expr`.

Semantics follow Python: numbers written without a dot are ints, with a dot are floats; `/` is true division and always gives a float; `//` and `%` use Python's floor semantics (`-7 // 2 == -4`, `-7 % 3 == 2`); `**` is Python's power. Division, floor division or modulo by zero raises `ExprError`.

Functions: `abs(x)`, `min(a, b, ...)` and `max(a, b, ...)` with at least two arguments, `round(x)` or `round(x, n)` (Python's `round`), and `sqrt(x)` (float result via `math.sqrt`; a negative argument raises `ExprError`). A call with the wrong number of arguments, or to an unknown function, raises `ExprError`. A name that is not a function but is used as one, or a name missing from `variables` (including a function name used as a plain variable), raises `ExprError`. Empty input, unbalanced parentheses, a missing operand and leftover tokens after a complete expression (`1 2`) raise `ExprError`.

Examples: `evaluate("2 + 3 * 4")` is `14`; `evaluate("max(x, 3) ** 2", {"x": 5})` is `25`; `evaluate("7 / 2")` is `3.5`.
""",
    r"""
import math
import re

TOKEN = re.compile(r"\s*(?:(\d+(?:\.\d+)?)|([A-Za-z_]\w*)|(//|\*\*|[-+*/%(),]))")


class ExprError(ValueError):
    pass


def tokenize(expr):
    tokens, pos = [], 0
    while True:
        if not expr[pos:].strip():
            return tokens
        m = TOKEN.match(expr, pos)
        if not m:
            raise ExprError(f"unexpected character at {pos}")
        number, name, op = m.groups()
        tokens.append(("num", number) if number else ("name", name) if name else ("op", op))
        pos = m.end()


FUNCTIONS = {
    "abs": (1, 1, abs),
    "min": (2, None, min),
    "max": (2, None, max),
    "round": (1, 2, round),
    "sqrt": (1, 1, None),
}


class _Parser:
    def __init__(self, tokens, variables):
        self.tokens, self.pos, self.variables = tokens, 0, variables

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else (None, None)

    def take(self):
        token = self.peek()
        if token[0] is None:
            raise ExprError("unexpected end of expression")
        self.pos += 1
        return token

    def accept(self, *ops):
        kind, text = self.peek()
        if kind == "op" and text in ops:
            self.pos += 1
            return text
        return None

    def expr(self):
        value = self.term()
        while op := self.accept("+", "-"):
            right = self.term()
            value = value + right if op == "+" else value - right
        return value

    def term(self):
        value = self.factor()
        while op := self.accept("*", "/", "//", "%"):
            right = self.factor()
            if op == "*":
                value *= right
                continue
            if right == 0:
                raise ExprError("division by zero")
            value = value / right if op == "/" else value // right if op == "//" else value % right
        return value

    def factor(self):
        op = self.accept("-", "+")
        if op:
            value = self.factor()
            return -value if op == "-" else value
        base = self.atom()
        if self.accept("**"):
            return base ** self.factor()
        return base

    def atom(self):
        kind, text = self.take()
        if kind == "num":
            return float(text) if "." in text else int(text)
        if kind == "name":
            if self.accept("("):
                return self.call(text)
            if text not in self.variables:
                raise ExprError(f"unknown variable {text}")
            return self.variables[text]
        if text == "(":
            value = self.expr()
            if not self.accept(")"):
                raise ExprError("missing )")
            return value
        raise ExprError(f"unexpected {text!r}")

    def call(self, name):
        if name not in FUNCTIONS:
            raise ExprError(f"unknown function {name}")
        args = []
        if not self.accept(")"):
            args.append(self.expr())
            while self.accept(","):
                args.append(self.expr())
            if not self.accept(")"):
                raise ExprError("missing )")
        low, high, func = FUNCTIONS[name]
        if len(args) < low or (high is not None and len(args) > high):
            raise ExprError(f"wrong argument count for {name}")
        if name == "sqrt":
            if args[0] < 0:
                raise ExprError("sqrt of negative number")
            return math.sqrt(args[0])
        return func(*args)


def evaluate(expr, variables=None):
    parser = _Parser(tokenize(expr), variables or {})
    value = parser.expr()
    if parser.peek()[0] is not None:
        raise ExprError(f"unexpected {parser.peek()[1]!r}")
    return value
""",
    r"""
import unittest

class TestTokenize(unittest.TestCase):

    def test_tokens(self):
        self.assertEqual(tokenize('3.5*(x+1)'), [('num', '3.5'), ('op', '*'), ('op', '('), ('name', 'x'), ('op', '+'), ('num', '1'), ('op', ')')])

    def test_double_char_operators(self):
        self.assertEqual(tokenize('2**3//4'), [('num', '2'), ('op', '**'), ('num', '3'), ('op', '//'), ('num', '4')])

    def test_names_commas_whitespace(self):
        self.assertEqual(tokenize(' max( a_1 , 2 ) '), [('name', 'max'), ('op', '('), ('name', 'a_1'), ('op', ','), ('num', '2'), ('op', ')')])

    def test_empty_and_bad_character(self):
        self.assertEqual(tokenize('   '), [])
        with self.assertRaises(ExprError):
            tokenize('3 $ 4')

class TestEvaluate(unittest.TestCase):

    def test_precedence(self):
        self.assertEqual(evaluate('2 + 3 * 4'), 14)
        self.assertEqual(evaluate('(2 + 3) * 4'), 20)
        self.assertEqual(evaluate('10 - 4 - 3'), 3)
        self.assertEqual(evaluate('100 / 10 / 5'), 2.0)

    def test_unary_and_power(self):
        self.assertEqual(evaluate('-2 ** 2'), -4)
        self.assertEqual(evaluate('2 ** 3 ** 2'), 512)
        self.assertEqual(evaluate('2 ** -1'), 0.5)
        self.assertEqual(evaluate('2 * -3'), -6)
        self.assertEqual(evaluate('2 - -3'), 5)
        self.assertEqual(evaluate('+5'), 5)
        self.assertEqual(evaluate('--4'), 4)

    def test_division_family(self):
        self.assertEqual(evaluate('7 / 2'), 3.5)
        v = evaluate('6 / 3')
        self.assertEqual(v, 2.0)
        self.assertIsInstance(v, float)
        self.assertEqual(evaluate('7 // 2'), 3)
        self.assertEqual(evaluate('-7 // 2'), -4)
        self.assertEqual(evaluate('-7 % 3'), 2)
        self.assertEqual(evaluate('7.5 % 2'), 1.5)

    def test_number_types(self):
        self.assertIsInstance(evaluate('2 + 3'), int)
        self.assertIsInstance(evaluate('2 + 3.0'), float)
        self.assertEqual(evaluate('0.1 + 0.2'), 0.1 + 0.2)

    def test_variables(self):
        self.assertEqual(evaluate('x * y + 1', {'x': 3, 'y': 4}), 13)
        self.assertEqual(evaluate('max(x, 3) ** 2', {'x': 5}), 25)
        self.assertEqual(evaluate('rate * 2', {'rate': 1.5}), 3.0)

    def test_functions(self):
        self.assertEqual(evaluate('abs(-3)'), 3)
        self.assertEqual(evaluate('min(4, 2, 9)'), 2)
        self.assertEqual(evaluate('round(2.567, 2)'), 2.57)
        self.assertEqual(evaluate('round(3.7)'), 4)
        self.assertEqual(evaluate('sqrt(16)'), 4.0)
        self.assertEqual(evaluate('max(abs(-4), min(2, 3) * 3)'), 6)

    def test_errors(self):
        for bad in ['', '   ', '1 +', '* 2', '(1 + 2', '1 + 2)', '1 2', '2 $ 3', 'foo', 'foo(1)', 'abs()', 'abs(1, 2)', 'round(1, 2, 3)', 'min()', 'max(7)', 'sqrt(-1)', '1 / 0', '1 // 0', '1 % 0', 'max(1,)', '(', '3 ** ', 'x(2)', 'abs']:
            with self.assertRaises(ExprError, msg=bad):
                evaluate(bad, {'x': 1})

    def test_error_is_value_error_and_function_name_not_a_variable(self):
        self.assertTrue(issubclass(ExprError, ValueError))
        with self.assertRaises(ExprError):
            evaluate('max + 1')
""",
)

task(
    "version-constraints",
    "large",
    "resolve",
    r"""
Write a small dependency-version toolkit. Public names: `Version`, `satisfies`, `max_satisfying`, `resolve` and `ResolutionError`.

`Version(text)` parses `MAJOR.MINOR.PATCH` with an optional `-prerelease` and an optional `+build`, for example `1.4.2`, `2.0.0-rc.1` or `1.0.0+abc`. The three numbers are required and consist of digits only. The pre-release is one or more non-empty dot-separated identifiers made of letters, digits and `-` (the core ends at the first `-`). The build part (letters, digits, `.`, `-`) is accepted but ignored: it does not affect equality, ordering or `str()`. Anything else raises `ValueError`. The object has the attributes `major`, `minor`, `patch` (ints) and `prerelease` (a tuple of strings, empty for a release), supports `==`, `<`, `<=`, `>`, `>=` and hashing, and `str(v)` gives `"MAJOR.MINOR.PATCH"` plus `"-" + ".".join(prerelease)` when there is a pre-release. Ordering: compare the three numbers; if equal, a pre-release is lower than the release; two pre-releases compare identifier by identifier, numeric identifiers numerically, a numeric identifier is lower than a non-numeric one, non-numeric ones as plain strings, and if all compared identifiers are equal the one with fewer identifiers is lower (so `1.0.0-alpha < 1.0.0-alpha.1 < 1.0.0-beta.2 < 1.0.0-beta.11 < 1.0.0-rc.1 < 1.0.0`).

Constraints are strings of comma-separated clauses, all of which must hold; whitespace is ignored. An empty string or `*` accepts everything. A clause is an optional operator and a version of one to three numeric components (missing ones are 0; an optional `-prerelease` is allowed): `==`, `!=`, `>=`, `<=`, `>`, `<` compare directly, and no operator means `==`. `~V` means `>= V` and below the next minor (`~1.2.3` and `~1.2` mean `>=1.2.x, <1.3.0`) except that `~1` (one component) means `<2.0.0`. `^V` means `>= V` and below the next breaking version: `<(major+1).0.0` when major is above 0, else `<0.(minor+1).0` when minor is above 0, else `<0.0.(patch+1)`. These upper bounds are release versions. A malformed clause raises `ValueError`.

`satisfies(version, constraint)` (version as a string or `Version`) returns a bool. A version with a pre-release satisfies a constraint only if at least one clause of the constraint itself names a pre-release (then the normal clause rules apply); otherwise it does not.

`max_satisfying(versions, constraint)` returns the highest satisfying version from an iterable of strings or `Version` objects as a string via `str`, or `None` if none match.

`resolve(requirements, available)`: `requirements` maps package name to a constraint string, and `available` maps package name to `{version string: {dependency name: constraint}}`. Work in rounds, starting from no picks: in each round gather for every needed package the list of constraints (its entry in `requirements`, if any, plus the constraints from the dependencies of the versions picked in the previous round), then pick for each package the highest available version that satisfies all of its gathered constraints. Raise `ResolutionError` (a subclass of `Exception`) whose message contains the package name if a package is not in `available` or no version fits. Stop and return the picks as a dict `{package: version string}` ordered by package name as soon as a round gives exactly the same picks as the previous round; after 20 rounds without that, raise `ResolutionError`.
""",
    r"""
import re
from functools import total_ordering

_VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?")
_CLAUSE = re.compile(r"(==|!=|>=|<=|>|<|~|\^)?\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:-([0-9A-Za-z.-]+))?")


class ResolutionError(Exception):
    pass


@total_ordering
class Version:
    def __init__(self, text):
        m = _VERSION.fullmatch(text)
        if not m:
            raise ValueError(f"bad version {text!r}")
        self.major, self.minor, self.patch = int(m.group(1)), int(m.group(2)), int(m.group(3))
        self.prerelease = tuple(m.group(4).split(".")) if m.group(4) else ()
        if "" in self.prerelease:
            raise ValueError(f"bad pre-release in {text!r}")

    def _key(self):
        pre = tuple((0, int(p), "") if p.isdigit() else (1, 0, p) for p in self.prerelease)
        return (self.major, self.minor, self.patch, not self.prerelease, pre)

    def __eq__(self, other):
        return isinstance(other, Version) and self._key() == other._key()

    def __lt__(self, other):
        return self._key() < other._key()

    def __hash__(self):
        return hash(self._key())

    def __str__(self):
        text = f"{self.major}.{self.minor}.{self.patch}"
        return text + "-" + ".".join(self.prerelease) if self.prerelease else text


def _clauses(constraint):
    text = constraint.strip()
    if text in ("", "*"):
        return []
    clauses = []
    for part in text.split(","):
        m = _CLAUSE.fullmatch(part.strip())
        if not m:
            raise ValueError(f"bad clause {part!r}")
        op, *numbers, pre = m.groups()
        numbers = [int(n) for n in numbers if n is not None]
        padded = numbers + [0] * (3 - len(numbers))
        base = Version("%d.%d.%d" % tuple(padded) + (f"-{pre}" if pre else ""))
        clauses.append((op or "==", base, len(numbers)))
    return clauses


def _holds(version, op, base, width):
    if op == "~":
        upper = Version(f"{base.major + 1}.0.0" if width == 1 else f"{base.major}.{base.minor + 1}.0")
        return base <= version < upper
    if op == "^":
        if base.major > 0:
            upper = Version(f"{base.major + 1}.0.0")
        elif base.minor > 0:
            upper = Version(f"0.{base.minor + 1}.0")
        else:
            upper = Version(f"0.0.{base.patch + 1}")
        return base <= version < upper
    return {
        "==": version == base,
        "!=": version != base,
        ">=": version >= base,
        "<=": version <= base,
        ">": version > base,
        "<": version < base,
    }[op]


def satisfies(version, constraint):
    version = version if isinstance(version, Version) else Version(version)
    clauses = _clauses(constraint)
    if version.prerelease and not any(base.prerelease for _, base, _ in clauses):
        return False
    return all(_holds(version, *clause) for clause in clauses)


def max_satisfying(versions, constraint):
    parsed = [v if isinstance(v, Version) else Version(v) for v in versions]
    matching = [v for v in parsed if satisfies(v, constraint)]
    return str(max(matching)) if matching else None


def resolve(requirements, available):
    picks = {}
    for _ in range(20):
        needed = {name: [c] for name, c in requirements.items()}
        for name, version in picks.items():
            for dep, constraint in available[name][version].items():
                needed.setdefault(dep, []).append(constraint)
        new = {}
        for name in sorted(needed):
            if name not in available:
                raise ResolutionError(f"unknown package {name}")
            fits = [v for v in available[name] if all(satisfies(v, c) for c in needed[name])]
            if not fits:
                raise ResolutionError(f"no version of {name} satisfies {needed[name]}")
            new[name] = str(max(Version(v) for v in fits))
        if new == picks:
            return new
        picks = new
    raise ResolutionError("no stable resolution")
""",
    r"""
import unittest
AVAILABLE = {'app-lib': {'1.0.0': {}, '1.1.0': {'util': '^1.0.0'}, '2.0.0': {'util': '^2.0.0'}}, 'util': {'1.0.0': {}, '1.5.0': {}, '2.0.0': {}, '2.1.0': {}}, 'extra': {'1.0.0': {'util': '<1.5.0'}}}

class TestVersion(unittest.TestCase):

    def test_parse_attributes(self):
        v = Version('2.10.3-rc.1+build.7')
        self.assertEqual((v.major, v.minor, v.patch, v.prerelease), (2, 10, 3, ('rc', '1')))
        self.assertEqual(str(v), '2.10.3-rc.1')
        self.assertEqual(Version('1.0.0').prerelease, ())
        self.assertEqual(Version('1.0.0-alpha-1').prerelease, ('alpha-1',))

    def test_invalid_and_numeric_ordering(self):
        for bad in ['1.0', '1.0.0.0', 'v1.0.0', '1.0.x', '-1.0.0', '1.0.0-', '1.0.0-a..b', '', '1.0.0-a.', '1.0.0+']:
            with self.assertRaises(ValueError, msg=bad):
                Version(bad)
        order = ['0.9.9', '1.0.0', '1.0.1', '1.1.0', '1.10.0', '2.0.0']
        self.assertEqual(sorted(reversed([Version(x) for x in order])), [Version(x) for x in order])
        self.assertTrue(Version('1.10.0') > Version('1.9.0'))

    def test_prerelease_ordering(self):
        order = ['1.0.0-alpha', '1.0.0-alpha.1', '1.0.0-alpha.beta', '1.0.0-beta', '1.0.0-beta.2', '1.0.0-beta.11', '1.0.0-rc.1', '1.0.0']
        versions = [Version(x) for x in order]
        for lo, hi in zip(versions, versions[1:]):
            self.assertTrue(lo < hi, (str(lo), str(hi)))
            self.assertTrue(hi >= lo and hi > lo and (lo <= hi) and (lo != hi))

    def test_build_ignored_and_hash(self):
        self.assertEqual(Version('1.0.0+abc'), Version('1.0.0+xyz'))
        self.assertEqual(len({Version('1.0.0+a'), Version('1.0.0')}), 1)
        self.assertEqual(str(Version('1.0.0+abc')), '1.0.0')

class TestSatisfies(unittest.TestCase):

    def check(self, constraint, yes, no):
        for v in yes:
            self.assertTrue(satisfies(v, constraint), (constraint, v))
        for v in no:
            self.assertFalse(satisfies(v, constraint), (constraint, v))

    def test_comparison_operators(self):
        self.check('>=1.2.0,<2.0.0', ['1.2.0', '1.9.9'], ['2.0.0', '1.1.9'])
        self.check('==1.2.3', ['1.2.3'], ['1.2.4'])
        self.check('1.2.3', ['1.2.3'], ['1.2.2'])
        self.check('!=1.2.3', ['1.2.4', '0.1.0'], ['1.2.3'])
        self.check('>1.0.0', ['1.0.1'], ['1.0.0'])
        self.check('<=1.0.0', ['1.0.0', '0.5.0'], ['1.0.1'])

    def test_tilde_and_caret(self):
        self.check('~1.2.3', ['1.2.3', '1.2.9'], ['1.2.2', '1.3.0'])
        self.check('~1.2', ['1.2.0', '1.2.7'], ['1.3.0', '1.1.9'])
        self.check('~1', ['1.0.0', '1.9.0'], ['2.0.0', '0.9.0'])
        self.check('^1.2.3', ['1.2.3', '1.9.0'], ['1.2.2', '2.0.0'])
        self.check('^0.2.3', ['0.2.3', '0.2.9'], ['0.3.0', '0.2.2'])
        self.check('^0.0.3', ['0.0.3'], ['0.0.4', '0.0.2'])

    def test_any_and_whitespace_and_prerelease_rules(self):
        self.check('*', ['0.0.1', '9.9.9'], [])
        self.check('', ['1.0.0'], [])
        self.check(' >= 1.2 , < 2 ', ['1.2.0', '1.99.0'], ['2.0.0', '1.1.0'])
        self.check('>=1.0.0', ['1.5.0'], ['2.0.0-rc.1'])
        self.check('>=2.0.0-rc.1', ['2.0.0-rc.1', '2.0.0-rc.2', '2.0.0'], ['2.0.0-beta'])
        self.check('>=1.0.0-alpha', ['1.5.0', '1.0.0-beta'], [])

    def test_accepts_version_objects_and_invalid_constraints(self):
        self.assertTrue(satisfies(Version('1.2.3'), '^1.0.0'))
        for bad in ['>=', '~~1', '1.2.3.4', 'abc', '>=1.0,,<2', '=>1.0']:
            with self.assertRaises(ValueError, msg=bad):
                satisfies('1.0.0', bad)

class TestMaxAndResolve(unittest.TestCase):

    def test_max_satisfying(self):
        vs = ['1.0.0', '1.2.0', '1.10.0', '2.0.0', '1.11.0-beta']
        self.assertEqual(max_satisfying(vs, '^1.0.0'), '1.10.0')
        self.assertEqual(max_satisfying(vs, '*'), '2.0.0')
        self.assertIsNone(max_satisfying(vs, '>=3.0.0'))
        self.assertEqual(max_satisfying([Version('1.0.0+x'), Version('1.1.0')], '<1.1.0'), '1.0.0')
        self.assertIsNone(max_satisfying([], '*'))

    def test_resolve_transitive_and_resolve_multiple_constraints(self):
        self.assertEqual(resolve({'app-lib': '^1.0.0'}, AVAILABLE), {'app-lib': '1.1.0', 'util': '1.5.0'})
        out = resolve({'app-lib': '^1.0.0', 'extra': '*'}, AVAILABLE)
        self.assertEqual(out, {'app-lib': '1.1.0', 'extra': '1.0.0', 'util': '1.0.0'})
        self.assertEqual(list(out), ['app-lib', 'extra', 'util'])

    def test_resolve_conflict_and_resolve_unknown_and_unsatisfiable(self):
        with self.assertRaises(ResolutionError) as cm:
            resolve({'app-lib': '^2.0.0', 'extra': '*'}, AVAILABLE)
        self.assertIn('util', str(cm.exception))
        with self.assertRaises(ResolutionError) as cm:
            resolve({'nope': '*'}, AVAILABLE)
        self.assertIn('nope', str(cm.exception))
        with self.assertRaises(ResolutionError):
            resolve({'util': '>=9.0.0'}, AVAILABLE)

    def test_resolve_repicks_when_new_constraint_appears_and_resolve_empty_and_resolve_oscillation_fails(self):
        avail = {'a': {'1.0.0': {}, '2.0.0': {'b': '<2.0.0'}}, 'b': {'1.0.0': {}, '2.0.0': {}}}
        self.assertEqual(resolve({'a': '*', 'b': '*'}, avail), {'a': '2.0.0', 'b': '1.0.0'})
        self.assertEqual(resolve({}, AVAILABLE), {})
        avail = {'a': {'1.0.0': {}, '2.0.0': {'b': '<2.0.0'}}, 'b': {'1.0.0': {'a': '<2.0.0'}, '2.0.0': {}}}
        with self.assertRaises(ResolutionError):
            resolve({'a': '*', 'b': '*'}, avail)
""",
)

task(
    "job-scheduler",
    "large",
    "schedule",
    r"""
Write a deterministic job scheduler for jobs with dependencies. Public names: `CycleError`, `ready_jobs`, `topological_order`, `schedule`, `makespan`, `utilization` and `critical_path`.

A job set is a dict mapping job name to a dict with the key `"duration"` (a non-negative int, required) and the optional keys `"deps"` (a list of job names that must finish first; default empty) and `"priority"` (an int; higher runs first; default 0).

`CycleError` is a subclass of `ValueError` raised when the dependencies contain a cycle (including a job depending on itself). Every public function that takes a job set also raises `ValueError` (not a `CycleError`) if a dependency names a job that does not exist or if a duration is negative.

`ready_jobs(jobs, finished)` returns the list of names of jobs that are not in the collection `finished` and whose dependencies are all in `finished`, sorted by `(-priority, name)`.

`topological_order(jobs)` returns a list of all job names in a valid execution order: repeatedly, among the not-yet-listed jobs whose dependencies are all listed, take the one with the smallest `(-priority, name)`. Raises `CycleError` if some jobs can never become ready.

`schedule(jobs, workers)` simulates `workers` identical workers (numbered from 0) and returns a list of tuples `(name, worker, start, end)` sorted by `(start, worker)` (jobs that share both keys keep the order in which they were started). Raise `ValueError` if `workers < 1`. The simulation starts at time 0 and repeats the following until all jobs are done: (1) give ready jobs to idle workers: a job is ready when all its dependencies have finished by the current time; ready jobs are taken in order of `(-priority, name)` and each one is given to the lowest-numbered idle worker, starting at the current time and ending `duration` later; (2) if nothing is running but jobs remain, raise `CycleError`; (3) advance the clock to the earliest end time among running jobs and mark every running job ending at that time as finished, which frees its worker. A job with duration 0 therefore starts and finishes at the same instant, and jobs depending on it may start at that same instant in the next round of step (1).

`makespan(schedule_result)` returns the largest end time in a schedule, or 0 for an empty schedule.

`utilization(schedule_result, workers)` returns the share of worker time spent running jobs: the sum of `end - start` over all entries divided by `workers * makespan`, rounded to 4 decimals; 0.0 if the makespan is 0.

`critical_path(jobs)` returns `(length, path)` for the longest chain of dependent jobs by total duration. The earliest finish time of a job is its duration plus the largest earliest finish among its dependencies (0 if none). `length` is the largest earliest finish; the path ends at the job achieving it (ties: smallest name) and walks backwards, each time moving to the dependency with the largest earliest finish (ties: smallest name), and is returned in execution order. An empty job set gives `(0, [])`.

Example: for jobs `fetch(3)`, `parse(2, after fetch)` and `lint(1, after fetch)`, with two workers, `schedule` gives `[("fetch", 0, 0, 3), ("lint", 0, 3, 4), ("parse", 1, 3, 5)]`.
""",
    r"""
class CycleError(ValueError):
    pass


def _validate(jobs):
    for name, spec in jobs.items():
        if spec["duration"] < 0:
            raise ValueError(f"negative duration for {name}")
        for dep in spec.get("deps", []):
            if dep not in jobs:
                raise ValueError(f"{name} depends on unknown job {dep}")


def _rank(jobs, name):
    return (-jobs[name].get("priority", 0), name)


def ready_jobs(jobs, finished):
    _validate(jobs)
    ready = (
        name
        for name, spec in jobs.items()
        if name not in finished and all(dep in finished for dep in spec.get("deps", []))
    )
    return sorted(ready, key=lambda name: _rank(jobs, name))


def topological_order(jobs):
    order = []
    while len(order) < len(jobs):
        ready = ready_jobs(jobs, order)
        if not ready:
            raise CycleError("dependency cycle")
        order.append(ready[0])
    return order


def schedule(jobs, workers):
    if workers < 1:
        raise ValueError("workers must be >= 1")
    topological_order(jobs)
    now, running, done = 0, [], set()
    pending, result = set(jobs), []
    while pending or running:
        busy = {w for _, w, _ in running}
        idle = [w for w in range(workers) if w not in busy]
        ready = [n for n in ready_jobs(jobs, done) if n in pending]
        for worker, name in zip(idle, ready):
            end = now + jobs[name]["duration"]
            result.append((name, worker, now, end))
            running.append((end, worker, name))
            pending.remove(name)
        if not running:
            raise CycleError("dependency cycle")
        now = min(end for end, _, _ in running)
        finished = [r for r in running if r[0] == now]
        running = [r for r in running if r[0] != now]
        done.update(name for _, _, name in finished)
    return sorted(result, key=lambda r: (r[2], r[1]))


def makespan(schedule_result):
    ends = [end for _, _, _, end in schedule_result]
    return max(ends) if ends else 0


def utilization(schedule_result, workers):
    span = makespan(schedule_result)
    if span == 0:
        return 0.0
    busy = sum(end - start for _, _, start, end in schedule_result)
    return round(busy / (workers * span), 4)


def critical_path(jobs):
    finish = {}
    for name in topological_order(jobs):
        deps = jobs[name].get("deps", [])
        finish[name] = jobs[name]["duration"] + max((finish[d] for d in deps), default=0)
    if not finish:
        return 0, []
    current = min(finish, key=lambda n: (-finish[n], n))
    path = [current]
    while jobs[current].get("deps"):
        current = min(jobs[current]["deps"], key=lambda n: (-finish[n], n))
        path.append(current)
    return finish[path[0]], path[::-1]
""",
    r"""
import unittest
JOBS = {'fetch': {'duration': 3}, 'parse': {'duration': 2, 'deps': ['fetch']}, 'lint': {'duration': 1, 'deps': ['fetch']}, 'build': {'duration': 4, 'deps': ['parse', 'lint']}, 'docs': {'duration': 2, 'deps': ['parse']}, 'test': {'duration': 3, 'deps': ['build']}}

class TestOrder(unittest.TestCase):

    def test_default_order_and_priority(self):
        self.assertEqual(topological_order(JOBS), ['fetch', 'lint', 'parse', 'build', 'docs', 'test'])
        jobs = {k: dict(v) for k, v in JOBS.items()}
        jobs['docs']['priority'] = 5
        self.assertEqual(topological_order(jobs), ['fetch', 'lint', 'parse', 'docs', 'build', 'test'])

    def test_cycle(self):
        cyc = {'a': {'duration': 1, 'deps': ['b']}, 'b': {'duration': 1, 'deps': ['a']}}
        with self.assertRaises(CycleError):
            topological_order(cyc)
        with self.assertRaises(CycleError):
            topological_order({'a': {'duration': 1, 'deps': ['a']}})
        self.assertTrue(issubclass(CycleError, ValueError))

    def test_validation(self):
        with self.assertRaises(ValueError) as cm:
            topological_order({'a': {'duration': 1, 'deps': ['zzz']}})
        self.assertNotIsInstance(cm.exception, CycleError)
        with self.assertRaises(ValueError):
            topological_order({'a': {'duration': -1}})
        self.assertEqual(topological_order({}), [])

class TestSchedule(unittest.TestCase):

    def test_single_worker_and_two_workers(self):
        s = schedule(JOBS, 1)
        self.assertEqual(s, [('fetch', 0, 0, 3), ('lint', 0, 3, 4), ('parse', 0, 4, 6), ('build', 0, 6, 10), ('docs', 0, 10, 12), ('test', 0, 12, 15)])
        self.assertEqual(makespan(s), 15)
        s = schedule(JOBS, 2)
        self.assertEqual(s, [('fetch', 0, 0, 3), ('lint', 0, 3, 4), ('parse', 1, 3, 5), ('build', 0, 5, 9), ('docs', 1, 5, 7), ('test', 0, 9, 12)])
        self.assertEqual(makespan(s), 12)

    def test_more_workers_than_needed_and_example_and_priority_with_one_worker(self):
        self.assertEqual(schedule(JOBS, 10), schedule(JOBS, 2))
        jobs = {k: JOBS[k] for k in ('fetch', 'parse', 'lint')}
        self.assertEqual(schedule(jobs, 2), [('fetch', 0, 0, 3), ('lint', 0, 3, 4), ('parse', 1, 3, 5)])
        jobs = {'x': {'duration': 1, 'priority': 1}, 'y': {'duration': 1}, 'z': {'duration': 1, 'priority': 1}}
        self.assertEqual(schedule(jobs, 1), [('x', 0, 0, 1), ('z', 0, 1, 2), ('y', 0, 2, 3)])

    def test_independent_jobs_spread_over_workers_and_zero_duration_and_errors_and_empty(self):
        jobs = {n: {'duration': d} for n, d in [('a', 4), ('b', 2), ('c', 2), ('d', 1)]}
        self.assertEqual(schedule(jobs, 2), [('a', 0, 0, 4), ('b', 1, 0, 2), ('c', 1, 2, 4), ('d', 0, 4, 5)])
        jobs = {'a': {'duration': 0}, 'b': {'duration': 2, 'deps': ['a']}}
        self.assertEqual(schedule(jobs, 1), [('a', 0, 0, 0), ('b', 0, 0, 2)])
        with self.assertRaises(ValueError):
            schedule(JOBS, 0)
        with self.assertRaises(CycleError):
            schedule({'a': {'duration': 1, 'deps': ['b']}, 'b': {'duration': 1, 'deps': ['a']}}, 2)
        self.assertEqual(schedule({}, 3), [])
        self.assertEqual(makespan([]), 0)

class TestReadyAndUtilization(unittest.TestCase):

    def test_ready_jobs(self):
        self.assertEqual(ready_jobs(JOBS, set()), ['fetch'])
        self.assertEqual(ready_jobs(JOBS, {'fetch'}), ['lint', 'parse'])
        self.assertEqual(ready_jobs(JOBS, ['fetch', 'parse', 'lint']), ['build', 'docs'])
        self.assertEqual(ready_jobs(JOBS, set(JOBS)), [])

    def test_ready_jobs_priority(self):
        jobs = {k: dict(v) for k, v in JOBS.items()}
        jobs['docs']['priority'] = 5
        self.assertEqual(ready_jobs(jobs, ['fetch', 'parse', 'lint']), ['docs', 'build'])

    def test_utilization(self):
        self.assertEqual(utilization(schedule(JOBS, 2), 2), 0.625)
        self.assertEqual(utilization(schedule(JOBS, 1), 1), 1.0)
        self.assertEqual(utilization([], 3), 0.0)
        self.assertEqual(utilization(schedule(JOBS, 3), 3), round(15 / 36, 4))

class TestCriticalPath(unittest.TestCase):

    def test_main(self):
        self.assertEqual(critical_path(JOBS), (12, ['fetch', 'parse', 'build', 'test']))

    def test_tie_breaks(self):
        jobs = {'a': {'duration': 2}, 'b': {'duration': 2}, 'c': {'duration': 1, 'deps': ['a', 'b']}}
        self.assertEqual(critical_path(jobs), (3, ['a', 'c']))
        self.assertEqual(critical_path({'x': {'duration': 5}, 'y': {'duration': 5}}), (5, ['x']))

    def test_empty_and_cycle(self):
        self.assertEqual(critical_path({}), (0, []))
        with self.assertRaises(CycleError):
            critical_path({'a': {'duration': 1, 'deps': ['a']}})
""",
)

task(
    "line-diff",
    "large",
    "diff_lines",
    r"""
Write a line-based diff module built on the longest common subsequence (LCS). Public names: `diff_lines`, `apply_diff`, `invert_diff`, `diff_stats`, `changed_ranges`, `similarity` and `format_diff`. Inputs are lists of strings (lines).

`diff_lines(a, b)` returns a list of `(op, line)` tuples where `op` is `"="` (line kept), `"-"` (line only in `a`) or `"+"` (line only in `b`). Reading the `"="` and `"-"` entries in order gives back `a`; reading the `"="` and `"+"` entries gives `b`. The result must be built deterministically so that every implementation gives the same list. Let `L(i, j)` be the length of the LCS of `a[i:]` and `b[j:]`. Start at `i = j = 0` and repeat: if both lists have remaining lines and `a[i] == b[j]`, emit `("=", a[i])` and advance both; otherwise, if `b` is exhausted, or `a` is not exhausted and `L(i + 1, j) >= L(i, j + 1)`, emit `("-", a[i])` and advance `i`; otherwise emit `("+", b[j])` and advance `j`. Stop when both are exhausted. So when removing a line and adding a line are equally good, the removal comes first.

`apply_diff(a, ops)` replays an operation list against the list `a` and returns the resulting list of lines. Walk through `ops` with a position pointer into `a`: for `"="` and `"-"` the line must equal `a` at the pointer (otherwise raise `ValueError`, also when the pointer is past the end) and the pointer advances; `"="` also appends the line to the output; `"+"` appends the line to the output. After the last op, the pointer must be exactly at the end of `a`, otherwise raise `ValueError`.

`invert_diff(ops)` returns a new list in which `"+"` and `"-"` are swapped and `"="` is kept, in the same order, so that applying it to `b` gives `a`.

`diff_stats(ops)` returns `{"added": n, "removed": n, "unchanged": n}` counting the `"+"`, `"-"` and `"="` entries.

`changed_ranges(ops)` returns a list of `(start, end)` half-open index ranges into `ops`, one for each maximal run of consecutive entries whose op is not `"="`, in order. For the example below it returns `[(1, 2), (4, 5)]`; with no changes it returns `[]`.

`similarity(a, b)` returns `2 * lcs / (len(a) + len(b))` rounded to 4 decimals, where `lcs` is the LCS length; two empty lists give `1.0`.

`format_diff(ops, context=None)` renders the operations as text, one line per op, joined by `"\n"`: `"  " + line` for `"="`, `"- " + line` for `"-"`, `"+ " + line` for `"+"`. If `context` is an int (it must be at least 0, otherwise raise `ValueError`), unchanged entries are hidden unless they are within `context` positions (in the ops list) of an entry that is not `"="`; every maximal run of hidden entries is replaced by one line `"..."`. An empty ops list renders as the empty string.

Example: `diff_lines(["a", "b", "c", "d"], ["a", "c", "d", "e"])` is `[("=", "a"), ("-", "b"), ("=", "c"), ("=", "d"), ("+", "e")]`.
""",
    r"""
def _lcs_table(a, b):
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) - 1, -1, -1):
        for j in range(len(b) - 1, -1, -1):
            if a[i] == b[j]:
                table[i][j] = table[i + 1][j + 1] + 1
            else:
                table[i][j] = max(table[i + 1][j], table[i][j + 1])
    return table


def diff_lines(a, b):
    table = _lcs_table(a, b)
    ops, i, j = [], 0, 0
    while i < len(a) or j < len(b):
        if i < len(a) and j < len(b) and a[i] == b[j]:
            ops.append(("=", a[i]))
            i += 1
            j += 1
        elif j == len(b) or (i < len(a) and table[i + 1][j] >= table[i][j + 1]):
            ops.append(("-", a[i]))
            i += 1
        else:
            ops.append(("+", b[j]))
            j += 1
    return ops


def apply_diff(a, ops):
    out, pos = [], 0
    for op, line in ops:
        if op == "+":
            out.append(line)
            continue
        if pos >= len(a) or a[pos] != line:
            raise ValueError("patch does not match the input")
        pos += 1
        if op == "=":
            out.append(line)
    if pos != len(a):
        raise ValueError("patch does not cover the whole input")
    return out


def invert_diff(ops):
    swap = {"+": "-", "-": "+", "=": "="}
    return [(swap[op], line) for op, line in ops]


def diff_stats(ops):
    names = {"+": "added", "-": "removed", "=": "unchanged"}
    stats = dict.fromkeys(names.values(), 0)
    for op, _ in ops:
        stats[names[op]] += 1
    return stats


def changed_ranges(ops):
    ranges, start = [], None
    for i, (op, _) in enumerate(ops + [("=", "")]):
        if op != "=" and start is None:
            start = i
        elif op == "=" and start is not None:
            ranges.append((start, i))
            start = None
    return ranges


def similarity(a, b):
    if not a and not b:
        return 1.0
    return round(2 * _lcs_table(a, b)[0][0] / (len(a) + len(b)), 4)


def format_diff(ops, context=None):
    if context is not None and context < 0:
        raise ValueError("context must not be negative")
    prefixes = {"=": "  ", "-": "- ", "+": "+ "}
    lines, hidden = [], False
    for i, (op, line) in enumerate(ops):
        near = context is None or op != "=" or any(
            ops[k][0] != "=" for k in range(max(0, i - context), min(len(ops), i + context + 1))
        )
        if near:
            lines.append(prefixes[op] + line)
            hidden = False
        elif not hidden:
            lines.append("...")
            hidden = True
    return "\n".join(lines)
""",
    r"""
import unittest
A = ['a', 'b', 'c', 'd']
B = ['a', 'c', 'd', 'e']

class TestDiff(unittest.TestCase):

    def test_example_and_identical_and_empty(self):
        self.assertEqual(diff_lines(A, B), [('=', 'a'), ('-', 'b'), ('=', 'c'), ('=', 'd'), ('+', 'e')])
        self.assertEqual(diff_lines(A, A), [('=', x) for x in A])
        self.assertEqual(diff_lines([], []), [])
        self.assertEqual(diff_lines([], ['x', 'y']), [('+', 'x'), ('+', 'y')])
        self.assertEqual(diff_lines(['x', 'y'], []), [('-', 'x'), ('-', 'y')])

    def test_removal_before_addition_on_ties_and_repeated_lines(self):
        self.assertEqual(diff_lines(['x'], ['y']), [('-', 'x'), ('+', 'y')])
        self.assertEqual(diff_lines(['a', 'x', 'c'], ['a', 'y', 'c']), [('=', 'a'), ('-', 'x'), ('+', 'y'), ('=', 'c')])
        self.assertEqual(diff_lines(['a', 'a', 'b'], ['a', 'b', 'a']), [('=', 'a'), ('-', 'a'), ('=', 'b'), ('+', 'a')])

    def test_reconstructs_both_sides(self):
        cases = [(A, B), ([], ['z']), (['q', 'r', 's'], ['s', 'r', 'q']), (list('abcabba'), list('cbabac'))]
        for a, b in cases:
            ops = diff_lines(a, b)
            self.assertEqual([x for op, x in ops if op != '+'], a)
            self.assertEqual([x for op, x in ops if op != '-'], b)
            self.assertEqual(apply_diff(a, ops), b)
            self.assertEqual(apply_diff(b, invert_diff(ops)), a)

    def test_lcs_is_longest(self):
        ops = diff_lines(list('abcabba'), list('cbabac'))
        self.assertEqual(sum((op == '=' for op, _ in ops)), 4)

class TestApplyAndStats(unittest.TestCase):

    def test_apply_mismatch_and_apply_incomplete_and_apply_pure_additions(self):
        with self.assertRaises(ValueError):
            apply_diff(['z'], [('=', 'a')])
        with self.assertRaises(ValueError):
            apply_diff([], [('-', 'a')])
        with self.assertRaises(ValueError):
            apply_diff(['a'], [('-', 'b')])
        with self.assertRaises(ValueError):
            apply_diff(['a', 'b'], [('=', 'a')])
        self.assertEqual(apply_diff([], [('+', 'x'), ('+', 'y')]), ['x', 'y'])

    def test_invert_and_stats(self):
        ops = diff_lines(A, B)
        self.assertEqual(invert_diff(ops), [('=', 'a'), ('+', 'b'), ('=', 'c'), ('=', 'd'), ('-', 'e')])
        self.assertEqual(invert_diff(invert_diff(ops)), ops)
        self.assertEqual(diff_stats(diff_lines(A, B)), {'added': 1, 'removed': 1, 'unchanged': 3})
        self.assertEqual(diff_stats([]), {'added': 0, 'removed': 0, 'unchanged': 0})

    def test_changed_ranges(self):
        self.assertEqual(changed_ranges(diff_lines(A, B)), [(1, 2), (4, 5)])
        self.assertEqual(changed_ranges(diff_lines(['a', 'x', 'c'], ['a', 'y', 'c'])), [(1, 3)])
        self.assertEqual(changed_ranges(diff_lines(A, A)), [])
        self.assertEqual(changed_ranges(diff_lines([], ['p', 'q'])), [(0, 2)])
        self.assertEqual(changed_ranges([]), [])

    def test_similarity(self):
        self.assertEqual(similarity(A, B), 0.75)
        self.assertEqual(similarity([], []), 1.0)
        self.assertEqual(similarity(['a'], []), 0.0)
        self.assertEqual(similarity(['a', 'b'], ['a', 'b']), 1.0)
        self.assertEqual(similarity(['a', 'b', 'c'], ['a', 'x', 'y', 'z']), 0.2857)
        self.assertEqual(similarity(['a'], ['a', 'b', 'c']), 0.5)

class TestFormat(unittest.TestCase):

    def test_full(self):
        self.assertEqual(format_diff(diff_lines(A, B)), '  a\n- b\n  c\n  d\n+ e')
        self.assertEqual(format_diff([]), '')

    def test_context_zero_and_context_one_keeps_everything_here(self):
        self.assertEqual(format_diff(diff_lines(A, B), context=0), '...\n- b\n...\n+ e')
        self.assertEqual(format_diff(diff_lines(A, B), context=1), '  a\n- b\n  c\n  d\n+ e')

    def test_context_windows(self):
        a = list('abcdefghij')
        b = list('abcdEfghij')
        ops = diff_lines(a, b)
        self.assertEqual(format_diff(ops, 1), '...\n  d\n- e\n+ E\n  f\n...')
        self.assertEqual(format_diff(ops, 2), '...\n  c\n  d\n- e\n+ E\n  f\n  g\n...')

    def test_context_without_changes_and_context_invalid(self):
        self.assertEqual(format_diff([('=', 'a'), ('=', 'b')], context=1), '...')
        with self.assertRaises(ValueError):
            format_diff([], context=-1)
""",
)

task(
    "circuit-breaker",
    "large",
    "CircuitBreaker",
    r"""
Write resilience helpers driven by an injected clock (a zero-argument callable returning the current time in seconds), so they can be tested without real time. Public names: `CircuitOpenError`, `SlidingWindowCounter`, `CircuitBreaker` and `ResilientClient`.

`CircuitOpenError` is a subclass of `Exception`, raised when a call is rejected because the circuit is open.

`SlidingWindowCounter(window, clock)` counts events in the most recent `window` seconds. `window` must be positive (else `ValueError`). `record(n=1)` registers `n` events at the current time (`n` must be a positive int, else `ValueError`). `count()` returns the number of events recorded at times `t` with `t > now - window` (an event exactly `window` seconds old no longer counts) and may discard expired events. `rate()` returns `count() / window`.

`CircuitBreaker(failure_threshold, recovery_timeout, clock, success_threshold=1)` protects a flaky function. It has three states reported by the property `state` as the strings `"closed"`, `"open"` and `"half_open"`:
- It starts closed. While closed, each failed call increments the attribute `failures` (consecutive failures); a successful call resets `failures` to 0. When `failures` reaches `failure_threshold` the circuit opens: the opening time is remembered, and `failures` is reset to 0.
- While open, calls are rejected. Once `recovery_timeout` seconds have passed since the opening time (`now - opened_at >= recovery_timeout`) the circuit counts as half open; reading `state` performs this transition lazily.
- While half open, calls are allowed through. A failed call opens the circuit again right away (with a new opening time). `success_threshold` successful calls in a row close it again, resetting `failures` to 0.
`call(func, *args, **kwargs)` first checks the state: if open, it raises `CircuitOpenError` without calling `func` (and without counting a failure). Otherwise it calls `func(*args, **kwargs)`; if that raises an `Exception`, the failure is recorded as above and the exception is re-raised; otherwise the success is recorded and the result returned. `reset()` closes the circuit and clears all counters. `stats()` returns `{"state": ..., "failures": ..., "opened_at": ...}` where `opened_at` is the time of the most recent opening, or `None` if it never opened (it is set back to `None` when the circuit closes again).

`ResilientClient(fetch, breaker, fallback=None)` wraps a function `fetch(key)`. Its attribute `counters` is a dict with the integer keys `"ok"`, `"failed"`, `"rejected"` and `"fallback"`, all starting at 0. `get(key)` calls `breaker.call(fetch, key)`. On success it increments `"ok"` and returns the value. If the breaker raises `CircuitOpenError` it increments `"rejected"`; if `fetch` raises any other `Exception` it increments `"failed"`. In both failure cases, if a `fallback` function was given, it increments `"fallback"` and returns `fallback(key)`; otherwise the original exception propagates.
""",
    r"""
from collections import deque


class CircuitOpenError(Exception):
    pass


class SlidingWindowCounter:
    def __init__(self, window, clock):
        if window <= 0:
            raise ValueError("window must be positive")
        self.window, self.clock = window, clock
        self._events = deque()

    def record(self, n=1):
        if n < 1:
            raise ValueError("n must be positive")
        self._events.append((self.clock(), n))

    def count(self):
        cutoff = self.clock() - self.window
        while self._events and self._events[0][0] <= cutoff:
            self._events.popleft()
        return sum(n for _, n in self._events)

    def rate(self):
        return self.count() / self.window


class CircuitBreaker:
    def __init__(self, failure_threshold, recovery_timeout, clock, success_threshold=1):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.clock = clock
        self.reset()

    def reset(self):
        self._state = "closed"
        self.failures = 0
        self._successes = 0
        self._opened_at = None

    @property
    def state(self):
        if self._state == "open" and self.clock() - self._opened_at >= self.recovery_timeout:
            self._state = "half_open"
            self._successes = 0
        return self._state

    def _open(self):
        self._state = "open"
        self._opened_at = self.clock()
        self.failures = 0

    def _record_failure(self):
        if self.state == "half_open":
            self._open()
            return
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self._open()

    def _record_success(self):
        if self.state == "half_open":
            self._successes += 1
            if self._successes >= self.success_threshold:
                self._state = "closed"
                self._opened_at = None
        self.failures = 0

    def call(self, func, *args, **kwargs):
        if self.state == "open":
            raise CircuitOpenError("circuit is open")
        try:
            result = func(*args, **kwargs)
        except Exception:
            self._record_failure()
            raise
        self._record_success()
        return result

    def stats(self):
        return {"state": self.state, "failures": self.failures, "opened_at": self._opened_at}


class ResilientClient:
    def __init__(self, fetch, breaker, fallback=None):
        self.fetch, self.breaker, self.fallback = fetch, breaker, fallback
        self.counters = {"ok": 0, "failed": 0, "rejected": 0, "fallback": 0}

    def get(self, key):
        try:
            value = self.breaker.call(self.fetch, key)
        except CircuitOpenError:
            self.counters["rejected"] += 1
            if self.fallback is None:
                raise
        except Exception:
            self.counters["failed"] += 1
            if self.fallback is None:
                raise
        else:
            self.counters["ok"] += 1
            return value
        self.counters["fallback"] += 1
        return self.fallback(key)
""",
    r"""
import unittest

class FakeClock:

    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t

class Service:

    def __init__(self):
        self.calls = 0
        self.fail = False

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError('down')
        return ('ok', args, kwargs)

def trip(breaker, service):
    service.fail = True
    for _ in range(breaker.failure_threshold):
        with unittest.TestCase().assertRaises(RuntimeError):
            breaker.call(service)

class TestCounter(unittest.TestCase):

    def test_window(self):
        clock = FakeClock()
        c = SlidingWindowCounter(10, clock)
        c.record()
        clock.t = 5
        c.record(2)
        clock.t = 9
        self.assertEqual(c.count(), 3)
        clock.t = 10
        self.assertEqual(c.count(), 2)
        clock.t = 14.9
        self.assertEqual(c.count(), 2)
        clock.t = 15
        self.assertEqual(c.count(), 0)

    def test_rate_and_validation(self):
        clock = FakeClock()
        c = SlidingWindowCounter(4, clock)
        c.record(6)
        self.assertEqual(c.rate(), 1.5)
        with self.assertRaises(ValueError):
            SlidingWindowCounter(0, clock)
        with self.assertRaises(ValueError):
            c.record(0)

class TestBreaker(unittest.TestCase):

    def setUp(self):
        self.clock = FakeClock()
        self.svc = Service()
        self.b = CircuitBreaker(3, 10, self.clock)

    def test_closed_passes_through_and_opens_after_threshold(self):
        self.assertEqual(self.b.call(self.svc, 1, k=2), ('ok', (1,), {'k': 2}))
        self.assertEqual(self.b.state, 'closed')
        trip(self.b, self.svc)
        self.assertEqual(self.b.state, 'open')
        calls = self.svc.calls
        with self.assertRaises(CircuitOpenError):
            self.b.call(self.svc)
        self.assertEqual(self.svc.calls, calls)

    def test_success_resets_failure_count(self):
        self.svc.fail = True
        for _ in range(2):
            with self.assertRaises(RuntimeError):
                self.b.call(self.svc)
        self.assertEqual(self.b.failures, 2)
        self.svc.fail = False
        self.b.call(self.svc)
        self.assertEqual(self.b.failures, 0)
        self.svc.fail = True
        for _ in range(2):
            with self.assertRaises(RuntimeError):
                self.b.call(self.svc)
        self.assertEqual(self.b.state, 'closed')

    def test_half_open_after_timeout_and_half_open_success_closes(self):
        self.clock.t = 2
        trip(self.b, self.svc)
        self.clock.t = 11.9
        self.assertEqual(self.b.state, 'open')
        self.clock.t = 12
        self.assertEqual(self.b.state, 'half_open')
        self.b = CircuitBreaker(3, 10, self.clock)
        self.clock.t = 0
        trip(self.b, self.svc)
        self.clock.t = 10
        self.svc.fail = False
        self.b.call(self.svc)
        self.assertEqual(self.b.state, 'closed')
        self.assertEqual(self.b.stats(), {'state': 'closed', 'failures': 0, 'opened_at': None})

    def test_half_open_failure_reopens_with_new_time(self):
        trip(self.b, self.svc)
        self.clock.t = 10
        with self.assertRaises(RuntimeError):
            self.b.call(self.svc)
        self.assertEqual(self.b.state, 'open')
        self.clock.t = 19
        self.assertEqual(self.b.state, 'open')
        self.clock.t = 20
        self.assertEqual(self.b.state, 'half_open')

    def test_success_threshold(self):
        b = CircuitBreaker(1, 5, self.clock, success_threshold=2)
        self.svc.fail = True
        with self.assertRaises(RuntimeError):
            b.call(self.svc)
        self.clock.t = 5
        self.svc.fail = False
        b.call(self.svc)
        self.assertEqual(b.state, 'half_open')
        b.call(self.svc)
        self.assertEqual(b.state, 'closed')

    def test_stats_when_open_and_rejections_are_not_failures(self):
        self.clock.t = 2
        trip(self.b, self.svc)
        self.assertEqual(self.b.stats(), {'state': 'open', 'failures': 0, 'opened_at': 2})
        for _ in range(5):
            with self.assertRaises(CircuitOpenError):
                self.b.call(self.svc)
        self.assertEqual(self.b.failures, 0)
        self.clock.t = 12
        self.assertEqual(self.b.state, 'half_open')

    def test_reset_and_error_type(self):
        trip(self.b, self.svc)
        self.b.reset()
        self.assertEqual(self.b.stats(), {'state': 'closed', 'failures': 0, 'opened_at': None})
        self.assertTrue(issubclass(CircuitOpenError, Exception))

class TestClient(unittest.TestCase):

    def test_counters_with_fallback(self):
        clock = FakeClock()
        svc = Service()
        client = ResilientClient(svc, CircuitBreaker(2, 10, clock), fallback=lambda k: f'cached:{k}')
        self.assertEqual(client.get('a')[0], 'ok')
        svc.fail = True
        self.assertEqual(client.get('a'), 'cached:a')
        self.assertEqual(client.get('b'), 'cached:b')
        self.assertEqual(client.get('c'), 'cached:c')
        self.assertEqual(client.counters, {'ok': 1, 'failed': 2, 'rejected': 1, 'fallback': 3})
        self.assertEqual(svc.calls, 3)

    def test_without_fallback_propagates(self):
        clock = FakeClock()
        svc = Service()
        svc.fail = True
        client = ResilientClient(svc, CircuitBreaker(1, 10, clock))
        with self.assertRaises(RuntimeError):
            client.get('a')
        with self.assertRaises(CircuitOpenError):
            client.get('a')
        self.assertEqual(client.counters, {'ok': 0, 'failed': 1, 'rejected': 1, 'fallback': 0})

    def test_recovers(self):
        clock = FakeClock()
        svc = Service()
        client = ResilientClient(svc, CircuitBreaker(1, 10, clock), fallback=lambda k: None)
        svc.fail = True
        client.get('x')
        svc.fail = False
        clock.t = 10
        self.assertEqual(client.get('x')[0], 'ok')
        self.assertEqual(client.counters['ok'], 1)
""",
)

task(
    "recurrence-rules",
    "large",
    "occurrences",
    r"""
Write a small calendar recurrence library using `datetime.date`. Public names: `parse_rule`, `occurrences`, `next_occurrence` and `conflicts`.

`parse_rule(text)` parses a rule string of `KEY=VALUE` pairs separated by `;`, for example `FREQ=WEEKLY;INTERVAL=2;BYDAY=MO,WE;COUNT=5`, into a dict with exactly the keys `"freq"` (`"daily"`, `"weekly"` or `"monthly"`), `"interval"` (int, default 1), `"byweekday"` (a sorted list of unique ints with Monday = 0 .. Sunday = 6, or `None` if absent), `"count"` (int or `None`) and `"until"` (a `datetime.date` or `None`). Keys and values are written in upper case. `FREQ` is required and is `DAILY`, `WEEKLY` or `MONTHLY`. `INTERVAL` and `COUNT` are positive integers. `BYDAY` is a comma-separated list of `MO TU WE TH FR SA SU` and is only allowed with `FREQ=WEEKLY`. `UNTIL` is a date written `YYYYMMDD`. `COUNT` and `UNTIL` cannot both appear. Raise `ValueError` for an empty string, a missing `FREQ`, unknown or repeated keys, a pair without `=`, or any invalid value.

`occurrences(start, rule, window_start, window_end)` returns the sorted list of dates of the series that fall inside the inclusive window; `rule` is a dict as returned by `parse_rule`. The series is defined from the first date `start` as follows. Daily: `start + k * interval` days for `k = 0, 1, 2, ...`. Weekly: weeks start on Monday; take the week containing `start` as week 0 and then every `interval`-th week after it; in each such week the series contains the weekdays listed in `byweekday` (default: just the weekday of `start`), except that dates before `start` are not part of the series. Monthly: the same day of the month as `start`, in every `interval`-th month counted from the month of `start`; a month that does not have that day (for example the 31st in April) is skipped and is not counted as an occurrence. `until`, when present, is an inclusive last date. `count`, when present, limits the series to its first `count` dates counted from `start` (whether or not they are inside the window). If the window is empty or reversed, the result is empty.

`next_occurrence(start, rule, after)` returns the first date of the series strictly later than `after`, looking at most 3660 days ahead of `after`, or `None`.

`conflicts(events)` takes a list of `(name, date, start_minute, end_minute)` tuples and returns a list of `(name_a, name_b)` pairs, one for each pair of events on the same date whose half-open minute intervals `[start, end)` overlap (events that merely touch do not conflict), where `a` comes earlier in the input than `b`; the pairs are ordered by the position of `a`, then of `b`. Raise `ValueError` if an event has `end_minute <= start_minute`.

Example: `occurrences(date(2024, 1, 31), parse_rule("FREQ=MONTHLY"), date(2024, 1, 1), date(2024, 6, 30))` is the dates January 31, March 31 and May 31.
""",
    r"""
from datetime import date, timedelta

FREQS = {"DAILY": "daily", "WEEKLY": "weekly", "MONTHLY": "monthly"}
DAYS = {name: i for i, name in enumerate("MO TU WE TH FR SA SU".split())}


def parse_rule(text):
    pairs = {}
    for part in text.split(";"):
        key, eq, value = part.partition("=")
        if not eq or key in pairs or key not in ("FREQ", "INTERVAL", "BYDAY", "COUNT", "UNTIL"):
            raise ValueError(f"bad rule part {part!r}")
        pairs[key] = value
    if pairs.get("FREQ") not in FREQS:
        raise ValueError("FREQ is required and must be DAILY, WEEKLY or MONTHLY")
    if "COUNT" in pairs and "UNTIL" in pairs:
        raise ValueError("COUNT and UNTIL are exclusive")
    if "BYDAY" in pairs and pairs["FREQ"] != "WEEKLY":
        raise ValueError("BYDAY needs WEEKLY")
    rule = {"freq": FREQS[pairs["FREQ"]], "interval": 1, "byweekday": None, "count": None, "until": None}
    for key in ("INTERVAL", "COUNT"):
        if key in pairs:
            if not pairs[key].isdigit() or int(pairs[key]) < 1:
                raise ValueError(f"bad {key}")
            rule[key.lower()] = int(pairs[key])
    if "BYDAY" in pairs:
        names = pairs["BYDAY"].split(",")
        if any(n not in DAYS for n in names):
            raise ValueError("bad BYDAY")
        rule["byweekday"] = sorted({DAYS[n] for n in names})
    if "UNTIL" in pairs:
        raw = pairs["UNTIL"]
        if len(raw) != 8 or not raw.isdigit():
            raise ValueError("bad UNTIL")
        rule["until"] = date(int(raw[:4]), int(raw[4:6]), int(raw[6:]))
    return rule


def _series(start, rule, limit):
    step = rule["interval"]
    if rule["freq"] == "daily":
        d = start
        while d <= limit:
            yield d
            d += timedelta(days=step)
    elif rule["freq"] == "weekly":
        days = rule["byweekday"] or [start.weekday()]
        monday = start - timedelta(days=start.weekday())
        while monday <= limit:
            for wd in days:
                d = monday + timedelta(days=wd)
                if d >= start:
                    yield d
            monday += timedelta(weeks=step)
    else:
        index = start.year * 12 + start.month - 1
        while date(index // 12, index % 12 + 1, 1) <= limit:
            try:
                yield date(index // 12, index % 12 + 1, start.day)
            except ValueError:
                pass
            index += step


def occurrences(start, rule, window_start, window_end):
    limit = min(window_end, rule["until"]) if rule["until"] else window_end
    found = []
    for d in _series(start, rule, limit):
        if rule["count"] is not None and len(found) >= rule["count"]:
            break
        if d > limit:
            break
        found.append(d)
    return [d for d in found if window_start <= d <= window_end]


def next_occurrence(start, rule, after):
    first = after + timedelta(days=1)
    found = occurrences(start, rule, first, after + timedelta(days=3660))
    return found[0] if found else None


def conflicts(events):
    for name, _, begin, end in events:
        if end <= begin:
            raise ValueError(f"bad interval for {name}")
    pairs = []
    for i, (name_a, day_a, begin_a, end_a) in enumerate(events):
        for name_b, day_b, begin_b, end_b in events[i + 1:]:
            if day_a == day_b and begin_a < end_b and begin_b < end_a:
                pairs.append((name_a, name_b))
    return pairs
""",
    r"""
import unittest
from datetime import date

def d(day, month=3, year=2024):
    return date(year, month, day)

class TestParse(unittest.TestCase):

    def test_full(self):
        self.assertEqual(parse_rule('FREQ=WEEKLY;INTERVAL=2;BYDAY=MO,WE;COUNT=5'), {'freq': 'weekly', 'interval': 2, 'byweekday': [0, 2], 'count': 5, 'until': None})

    def test_defaults(self):
        self.assertEqual(parse_rule('FREQ=DAILY'), {'freq': 'daily', 'interval': 1, 'byweekday': None, 'count': None, 'until': None})

    def test_until_and_sorted_byday(self):
        self.assertEqual(parse_rule('FREQ=MONTHLY;UNTIL=20240630')['until'], date(2024, 6, 30))
        self.assertEqual(parse_rule('FREQ=WEEKLY;BYDAY=SU,FR,FR,MO')['byweekday'], [0, 4, 6])

    def test_errors(self):
        for bad in ['', 'INTERVAL=2', 'FREQ=YEARLY', 'FREQ=DAILY;INTERVAL=0', 'FREQ=DAILY;BYDAY=MO', 'FREQ=DAILY;COUNT=2;UNTIL=20240101', 'FREQ=DAILY;FOO=1', 'FREQ=DAILY;FREQ=DAILY', 'FREQ=WEEKLY;BYDAY=XX', 'FREQ=DAILY;UNTIL=2024-01-01', 'FREQ=DAILY;COUNT=0', 'FREQ', 'FREQ=DAILY;', 'FREQ=DAILY;INTERVAL=x']:
            with self.assertRaises(ValueError, msg=bad):
                parse_rule(bad)

class TestOccurrences(unittest.TestCase):

    def test_daily_and_daily_count_and_until(self):
        rule = parse_rule('FREQ=DAILY;INTERVAL=3')
        self.assertEqual(occurrences(d(1), rule, d(1), d(12)), [d(1), d(4), d(7), d(10)])
        self.assertEqual(occurrences(d(1), rule, d(5), d(12)), [d(7), d(10)])
        self.assertEqual(occurrences(d(1), parse_rule('FREQ=DAILY;INTERVAL=3;COUNT=3'), d(1), d(31)), [d(1), d(4), d(7)])
        self.assertEqual(occurrences(d(1), parse_rule('FREQ=DAILY;INTERVAL=3;COUNT=2'), d(5), d(31)), [])
        self.assertEqual(occurrences(d(1), parse_rule('FREQ=DAILY;INTERVAL=3;UNTIL=20240307'), d(1), d(31)), [d(1), d(4), d(7)])

    def test_weekly_default_day_and_weekly_byday_excludes_dates_before_start_and_weekly_interval(self):
        self.assertEqual(occurrences(d(1), parse_rule('FREQ=WEEKLY'), d(1), d(25)), [d(1), d(8), d(15), d(22)])
        rule = parse_rule('FREQ=WEEKLY;BYDAY=MO,WE')
        self.assertEqual(occurrences(d(6), rule, d(1), d(20)), [d(6), d(11), d(13), d(18), d(20)])
        rule = parse_rule('FREQ=WEEKLY;INTERVAL=2;BYDAY=MO,FR')
        self.assertEqual(occurrences(d(4), rule, d(1), d(31)), [d(4), d(8), d(18), d(22)])
        counted = parse_rule('FREQ=WEEKLY;INTERVAL=2;BYDAY=MO,FR;COUNT=3')
        self.assertEqual(occurrences(d(4), counted, d(1), d(31)), [d(4), d(8), d(18)])

    def test_monthly_skips_short_months_and_monthly_interval_and_year_rollover(self):
        rule = parse_rule('FREQ=MONTHLY')
        jan31 = date(2024, 1, 31)
        self.assertEqual(occurrences(jan31, rule, date(2024, 1, 1), date(2024, 6, 30)), [date(2024, 1, 31), date(2024, 3, 31), date(2024, 5, 31)])
        self.assertEqual(occurrences(jan31, parse_rule('FREQ=MONTHLY;COUNT=2'), date(2024, 1, 1), date(2025, 1, 1)), [date(2024, 1, 31), date(2024, 3, 31)])
        self.assertEqual(occurrences(date(2024, 1, 15), parse_rule('FREQ=MONTHLY;INTERVAL=5'), date(2024, 1, 1), date(2025, 1, 1)), [date(2024, 1, 15), date(2024, 6, 15), date(2024, 11, 15)])
        self.assertEqual(occurrences(date(2024, 11, 30), parse_rule('FREQ=MONTHLY'), date(2024, 11, 1), date(2025, 3, 31)), [date(2024, 11, 30), date(2024, 12, 30), date(2025, 1, 30), date(2025, 3, 30)])

    def test_leap_day_and_empty_windows(self):
        self.assertEqual(occurrences(date(2024, 2, 29), parse_rule('FREQ=MONTHLY;INTERVAL=12'), date(2024, 1, 1), date(2029, 1, 1)), [date(2024, 2, 29), date(2028, 2, 29)])
        rule = parse_rule('FREQ=DAILY')
        self.assertEqual(occurrences(d(10), rule, d(1), d(9)), [])
        self.assertEqual(occurrences(d(1), rule, d(10), d(5)), [])

class TestNextAndConflicts(unittest.TestCase):

    def test_next_occurrence(self):
        rule = parse_rule('FREQ=DAILY;INTERVAL=3')
        self.assertEqual(next_occurrence(d(1), rule, d(4)), d(7))
        self.assertEqual(next_occurrence(d(1), rule, d(1, 2)), d(1))
        self.assertIsNone(next_occurrence(d(1), parse_rule('FREQ=DAILY;INTERVAL=3;UNTIL=20240307'), d(10)))

    def test_next_occurrence_leap_and_conflicts(self):
        rule = parse_rule('FREQ=MONTHLY;INTERVAL=12')
        self.assertEqual(next_occurrence(date(2024, 2, 29), rule, date(2024, 3, 1)), date(2028, 2, 29))
        events = [('standup', d(5), 540, 555), ('review', d(5), 550, 600), ('sync', d(5), 600, 630), ('lunch', d(5), 720, 780), ('other', d(6), 540, 600)]
        self.assertEqual(conflicts(events), [('standup', 'review')])

    def test_conflict_order(self):
        events = [('a', d(5), 0, 60), ('b', d(5), 30, 90), ('c', d(5), 60, 120)]
        self.assertEqual(conflicts(events), [('a', 'b'), ('b', 'c')])
        wide = [('x', d(5), 0, 100), ('y', d(5), 10, 20), ('z', d(5), 15, 30)]
        self.assertEqual(conflicts(wide), [('x', 'y'), ('x', 'z'), ('y', 'z')])

    def test_conflicts_none_and_invalid(self):
        self.assertEqual(conflicts([]), [])
        self.assertEqual(conflicts([('a', d(5), 0, 10), ('b', d(6), 0, 10)]), [])
        with self.assertRaises(ValueError):
            conflicts([('a', d(5), 10, 10)])
""",
)


# ---------------------------------------------------------------- BUILD

REF_LINES = {"small": (1, 12), "medium": (25, 50), "large": (70, 130)}
TEST_RANGE = (6, 12)
TAIL = "\n\nWrite the complete solution in Python in one ```python code block."


def count_tests(source):
    import ast

    return sum(
        isinstance(item, ast.FunctionDef) and item.name.startswith("test")
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef)
        for item in node.body
    )


def ref_lines(source):
    return sum(1 for line in source.splitlines() if line.strip())


def build(root=ROOT, write=True):
    import ast
    from concurrent.futures import ThreadPoolExecutor

    from transformers import AutoTokenizer

    from anthropochrony.artifacts import digest, write_jsonl
    from anthropochrony.cli import reference_card
    from anthropochrony.gen.engine import render, system_token_count
    from anthropochrony.prompts import system_prompt
    from anthropochrony.sandbox import Sandbox
    from anthropochrony.tasks.pipeline import verify_reference
    from anthropochrony.tasks.schema import Task

    tokenizer = AutoTokenizer.from_pretrained(root / "pins/tiny-aya-global", local_files_only=True)
    card = reference_card(root)
    sandbox = Sandbox()
    system = system_prompt(card, estimator="agent")
    system_tokens = system_token_count(tokenizer, system)
    revision = digest([{k: t[k] for k in ("slug", "spec", "reference", "tests")} for t in TASKS])[
        :12
    ]

    def make(index, spec):
        task_id = f"FRESH-{index:02d}-{spec['slug']}"
        row = Task(
            task_id=task_id,
            family="FRESH",
            source="anthropochrony/fresh",
            source_revision=revision,
            license="Apache-2.0",
            task_turn=f"Task:\n{spec['spec']}{TAIL}",
            harness="unittest",
            entry_point=spec["entry_point"],
            reference_solution=spec["reference"],
            test_code=spec["tests"],
            io_examples=None,
            public_examples={},
            split="fresh",
            time_limits={"per_test_s": 2, "per_task_s": 30},
        ).as_dict()
        row["ref_tokens"] = len(
            tokenizer.encode(row["reference_solution"], add_special_tokens=False)
        )
        row["prompt_tokens"] = len(
            render(
                tokenizer,
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": row["task_turn"]},
                ],
            )
        )
        row["uncached_task_tokens"] = row["prompt_tokens"] - system_tokens
        row["tests_ref"] = digest({"test": row["test_code"], "io": None})
        problems = []
        names = {
            n.name
            for n in ast.parse(row["reference_solution"]).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        }
        if spec["entry_point"] not in names:
            problems.append("entry point not defined")
        low, high = REF_LINES[spec["size"]]
        lines = ref_lines(row["reference_solution"])
        if not low <= lines <= high:
            problems.append(f"reference lines {lines} outside {low}-{high}")
        tests = count_tests(row["test_code"])
        if not TEST_RANGE[0] <= tests <= TEST_RANGE[1]:
            problems.append(f"{tests} tests")
        if spec["size"] == "large" and len(spec["spec"].split()) < 250:
            problems.append("large spec under 250 words")
        verified, flag = verify_reference(row, sandbox)
        if flag:
            problems.append(f"verify: {flag}")
        verified["time_limits"] = {"per_test_s": 2, "per_task_s": 30}
        slow = max(verified["gt"]["times_s"])
        if slow > 2:
            problems.append(f"slow {slow:.2f}s")
        verified["flags"] = {"size": spec["size"], "ref_lines": lines, "tests": tests}
        return verified, problems

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda a: make(*a), enumerate(TASKS, 1)))
    good = []
    for row, problems in results:
        if problems:
            print("DROP", row["task_id"], problems)
        else:
            good.append(row)
    print(f"kept {len(good)} of {len(TASKS)}")
    for size in SIZES:
        lines = [r["flags"]["ref_lines"] for r in good if r["flags"]["size"] == size]
        if lines:
            print(
                size,
                len(lines),
                "ref lines min/median/max",
                min(lines),
                statistics.median(lines),
                max(lines),
            )
    if write and len(good) == len(TASKS):
        out = root / "data/fresh"
        out.mkdir(parents=True, exist_ok=True)
        write_jsonl(out / "tasks.jsonl", good)
        print("wrote", out / "tasks.jsonl")
    return good


if __name__ == "__main__":
    build(write="--dry" not in sys.argv)
