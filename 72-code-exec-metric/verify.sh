#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

uv sync --frozen --no-progress
uv run --frozen pytest -q
uv run --frozen mypy main.py
uv run --frozen python main.py demo
