# CPU side of the benchmark head-to-head: scoring, tables, figure, tests.
SHELL := /bin/bash
.ONESHELL:
# Where the public MetaTime artifact is, or should be downloaded to.
SUITE ?= data/metatime-artifact

.PHONY: setup test lint score report figure notebook

setup:
	uv sync --locked

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

# A full rescore needs the local run files in data/metatime/ (see docs/METATIME.md).
score:
	set -e
	. ./env.sh
	uv run python scripts/metatime/score.py --suite "$(SUITE)"

report:
	uv run python scripts/metatime/report.py

figure:
	uv run python scripts/metatime/figure.py

notebook:
	uv run python scripts/run_notebook.py
