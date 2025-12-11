#!/usr/bin/env bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install typer inquirerpy
uv pip install --no-deps -e .
uv pip show histology_features
uv run python .venv/bin/histology_features slurm-submit $1
deactivate