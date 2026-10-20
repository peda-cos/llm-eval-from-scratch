#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

uv sync --frozen --no-progress
uv run --frozen pytest -q
uv run --frozen mypy main.py eval_runner

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

uv run --frozen python main.py demo --json-out "$tmp/report.json" > "$tmp/table.md"
uv run --frozen python - "$tmp/report.json" "$tmp/table.md" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
table = Path(sys.argv[2]).read_text(encoding="utf-8")
assert report["status"] == "ok"
assert report["schema_version"] == 1
assert report["summary"]["tasks"] == 10
assert report["summary"]["models"] == 3
assert len(report["results"]) == 30
assert set(report["calibration"]) == {"rule_based", "biased", "noisy"}
assert all(item["n_samples"] == 10 for item in report["calibration"].values())
assert report["leaderboard"][0]["model_id"] == "rule_based"
means = {row["model_id"]: row["mean_score"] for row in report["leaderboard"]}
assert means["rule_based"] > means["noisy"]
assert "| 1 | rule_based |" in table
print("demo gate: rule_based={:.3f} biased={:.3f} noisy={:.3f}".format(
    means["rule_based"], means["biased"], means["noisy"]
))
PY

if uv run --frozen python main.py demo \
    --tasks fixtures/tasks_invalid.jsonl \
    --json-out "$tmp/error.json" > "$tmp/error.out" 2> "$tmp/error.err"; then
    echo "expected the invalid fixture to exit non-zero" >&2
    exit 1
fi

uv run --frozen python - "$tmp/error.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert payload["status"] == "error"
assert payload["error"]["stage"] == "validate"
assert payload["error"]["type"] == "FixtureValidationError"
assert len(payload["error"]["details"]) == 3
print("error path: structured validate failure")
PY

echo "verify: all checks passed"
