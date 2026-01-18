.PHONY: help venv install install-dev lint format typecheck test test-fast precommit precommit-install build clean

PYTHON ?= python
VENV ?= .venv

help:
	@echo "Common targets:"
	@echo "  make venv         Create virtualenv"
	@echo "  make install      Install runtime deps"
	@echo "  make install-dev  Install dev deps"
	@echo "  make lint         Run ruff lint"
	@echo "  make format       Run ruff format"
	@echo "  make typecheck    Run mypy"
	@echo "  make test         Run pytest"
	@echo "  make precommit    Run pre-commit on all files"
	@echo "  make precommit-install  Install git hook"
	@echo "  make build        Build package with python -m build"
	@echo "  make clean        Remove build outputs"

venv:
	$(PYTHON) -m venv $(VENV)

install:
	$(VENV)/bin/pip install -e .

install-dev:
	$(VENV)/bin/pip install -e ".[dev]"

lint:
	$(VENV)/bin/ruff check src tests

format:
	$(VENV)/bin/ruff format src tests

typecheck:
	$(VENV)/bin/mypy

test:
	$(VENV)/bin/pytest

precommit:
	$(VENV)/bin/pre-commit run --all-files --config pre-commit.yaml

precommit-install:
	$(VENV)/bin/pre-commit install --config pre-commit.yaml

build:
	$(VENV)/bin/python -m build

clean:
	rm -rf build dist __pycache__ .pytest_cache .ruff_cache
