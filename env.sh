#!/bin/sh
# Source from the repository root. Do not set HF_HOME: it hides the existing login.
ANTHROPOCHRONY_ROOT="${ANTHROPOCHRONY_ROOT:-$(pwd)}"
export ANTHROPOCHRONY_ROOT
export HF_HUB_CACHE="${ANTHROPOCHRONY_ROOT}/../.hf/hub"
export HF_DATASETS_CACHE="${ANTHROPOCHRONY_ROOT}/../.hf/datasets"
export HF_XET_CACHE="${ANTHROPOCHRONY_ROOT}/../.hf/xet"
export UV_PROJECT_ENVIRONMENT="${HOME}/.local/share/anthropochrony-envs/stack"
export TMPDIR="${HOME}/.cache/anthropochrony-tmp"
export ANTHROPOCHRONY_EXEC_PYTHON="${HOME}/.local/share/anthropochrony-envs/exec/bin/python"
mkdir -p "$TMPDIR"
