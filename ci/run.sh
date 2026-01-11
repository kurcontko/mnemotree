#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
VENV="${VENV:-.venv}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON=python3
fi

"$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/pip" install -e ".[dev]"

# Produce an artifact-friendly coverage report in CI.
export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} --cov-report=xml"

make lint typecheck test VENV="$VENV"
