# Run from WSL/Linux (Kaggle and Colab are Linux too).
# The venv lives outside the Windows-mounted repo for speed.
export UV_PROJECT_ENVIRONMENT ?= $(HOME)/.venvs/open-nla-audits
UV := uv
PY := $(UV) run python

.PHONY: install lint format test test-offline check-providers baseline secret-scan prepush hooks

install:
	$(UV) sync --python 3.11

lint:
	$(UV) run ruff check src tests scripts
	$(UV) run ruff format --check src tests scripts

format:
	$(UV) run ruff format src tests scripts
	$(UV) run ruff check --fix src tests scripts

test:
	$(UV) run pytest -q

test-offline:
	$(UV) run pytest -q -m "not network"

# Day-1 check: which free providers/models answer with your keys (needs .env)
check-providers:
	$(PY) scripts/check_providers.py --config configs/llm.yaml

# Step 1 baseline FVE (GPU: run via notebooks/step1_{kaggle,colab}.ipynb). PROFILE=kaggle|colab
PROFILE ?= kaggle
baseline:
	$(PY) -m nla_audit.experiments.step1_baseline run --config configs/step1.yaml --profile $(PROFILE) --sync

# Scan everything git would push for secrets / forbidden files
secret-scan:
	$(PY) scripts/secret_scan.py

prepush: lint test-offline secret-scan

# Install a git pre-push hook that runs the secret scan
hooks:
	printf '#!/bin/sh\nexec sh scripts/pre-push.sh "$$@"\n' > .git/hooks/pre-push
	chmod +x .git/hooks/pre-push
