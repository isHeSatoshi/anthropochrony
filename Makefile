SHELL := /bin/bash
.ONESHELL:
ROOT := $(abspath .)
RUN ?= base
VARIANT ?= m1
SEED ?= 0
ROUND ?= 1
PHASE ?= pilot
LOCAL_DATA ?= ../scratch/data
GENERATION ?=

.PHONY: setup exec-env test lint smoke preflight tasks calibrate measure labels baselines sft-data train remeasure eval extra-analyses simulate figures demo release status prepare-round execute-round release-data

setup:
	set -e
	. "$(ROOT)/env.sh"
	uv sync --locked

# Download the public Hub release (pinned revisions) into the gitignored `data/` layout.
release-data:
	set -e
	. "$(ROOT)/env.sh"
	uv run python scripts/fetch_release.py

exec-env:
	set -e
	. "$(ROOT)/env.sh"
	uv venv --python 3.10 "$$HOME/.local/share/anthropochrony-envs/exec"
	uv pip sync --python "$$ANTHROPOCHRONY_EXEC_PYTHON" src/anthropochrony/execenv/requirements.lock

test:
	set -e
	. "$(ROOT)/env.sh"
	uv run pytest

lint:
	set -e
	. "$(ROOT)/env.sh"
	uv run ruff check .
	uv run ruff format --check .
	uv run anthropochrony prompts --check

smoke preflight tasks calibrate labels baselines status release:
	set -e
	. "$(ROOT)/env.sh"
	uv run anthropochrony $@ $(if $(filter tasks,$@),--local-data "$(abspath $(LOCAL_DATA))",)

measure remeasure:
	set -e
	. "$(ROOT)/env.sh"
	uv run anthropochrony measure --run "$(RUN)" --phase "$(PHASE)"

prepare-round execute-round:
	set -e
	. "$(ROOT)/env.sh"
	uv run anthropochrony $@ --run "$(RUN)" --phase "$(PHASE)" --round "$(ROUND)" $(if $(GENERATION),--generation "$(abspath $(GENERATION))",)

sft-data:
	set -e
	. "$(ROOT)/env.sh"
	uv run anthropochrony sft-data --variant "$(VARIANT)"

train:
	set -e
	. "$(ROOT)/env.sh"
	uv run anthropochrony train --variant "$(VARIANT)" --seed "$(SEED)"

eval:
	set -e
	. "$(ROOT)/env.sh"
	uv run anthropochrony eval
	uv run anthropochrony extra-analyses
	uv run anthropochrony simulate

extra-analyses simulate:
	set -e
	. "$(ROOT)/env.sh"
	uv run anthropochrony $@

figures:
	set -e
	. "$(ROOT)/env.sh"
	uv run anthropochrony plots

demo:
	set -e
	. "$(ROOT)/env.sh"
	uv run --extra demo python "$(ROOT)/space/app.py"
