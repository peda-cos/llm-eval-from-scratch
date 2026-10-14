# Code Exec Metric

Generated code is right when it passes the tests. This module implements the code-execution
surface of an LLM eval harness: extract one fenced block from a free-form generation, run it
in a fresh Python subprocess with a wall-clock deadline, an output cap, and an import
denylist, then score the fraction of supplied assertions that pass. It also implements the
unbiased pass-at-k estimator used by HumanEval-style benchmarks.

The sandbox is deliberately educational. It is a subprocess boundary with a backstop
denylist, not a container or a security boundary. The capstone stops at this point too;
production evals add seccomp, network isolation, read-only filesystems, and container or
microVM isolation.

## Quickstart

Prerequisites: Python 3.11 or newer and [uv](https://docs.astral.sh/uv/). From a fresh clone:

```bash
cd llm-eval-from-scratch/72-code-exec-metric
./verify.sh
```

`verify.sh` installs the pinned dev toolchain into the ignored `.venv`, runs the tests and
strict mypy, then runs the deterministic demo. It is safe to run repeatedly. The first run may
download the pinned dev dependencies; tests and the demo perform no network I/O.

Manual commands:

```bash
uv sync --frozen
uv run python main.py demo
uv run pytest -q
uv run mypy main.py
```

## Architecture

```text
generation text
      |
      v
 extract_code  ------------------------------+
      |                                      |
      v                                      |
 run_candidate(code, targets, timeout_s)     |
      |                                      |
      +-- build child runner (-c)             |
      |     injected denylist + targets       |
      +-- code on child stdin                 |
      +-- stream stdout, 256 KiB cap          |
      +-- deadline kill (process group)       |
      +-- parse last result envelope          |
      |                                      |
      v                                      |
 ExecResult(exit_code, score, assertions)    |
      |                                      |
      +--> pass_at_k(n, c, k) <--------------+
```

| Path | Role |
|---|---|
| `main.py` | Extraction, child runner template, Popen sandbox, exit-code normalization, pass-at-k, fixture loader, demo CLI. |
| `fixtures/demo_cases.jsonl` | Two handcrafted tasks and 11 generations covering every exit code. |
| `tests/test_extract.py` | Fenced-block extraction contract, including lesson 70 compatibility cases. |
| `tests/test_sandbox.py` | Real subprocess tests for pass, partial, syntax, timeout, overflow, denylist, and envelope edge cases. |
| `tests/test_pass_at_k.py` | Worked estimator vectors and invalid-domain cases. |
| `tests/test_cli.py` | Demo output, fixture validation, mismatch handling, and CLI exit codes. |
| `verify.sh` | Frozen install, pytest, strict mypy, demo smoke test. |
| `pyproject.toml`, `uv.lock` | Pinned dev toolchain; no runtime dependencies. |

## Wire protocol

The parent builds a child script from `RUNNER_TEMPLATE` and passes it as `python -I -c
<script>`. The runner contains three injected values: the denylist, a result marker, and the
task's assertion strings as a JSON literal. The candidate **code** itself travels on the
child's stdin as UTF-8; the parent closes stdin after writing.

The child evaluates the candidate once, then evaluates each assertion independently in the
same namespace. Its last protocol write is:

```text
__code_exec_result__:<JSON>\n
```

The JSON envelope is one of:

```json
{"status": "ok", "assertions": [{"passed": true, "detail": ""}], "error": null}
{"status": "syntax_error", "assertions": [], "error": "SyntaxError: ..."}
{"status": "error", "assertions": [], "error": "ImportError: denied: subprocess"}
```

Candidate `print` output is **not** redirected: it goes to the same stdout pipe, counts toward
the 256 KiB cap, and is ignored by the parent. Before writing the envelope, the child flushes
the current and original stdout/stderr so buffered prints do not land after the result. The
parent locates the last marker and uses `json.JSONDecoder().raw_decode`, which tolerates
trailing shutdown output such as an `atexit` print. A fixed marker is sufficient for this
educational protocol; it is not an authenticity mechanism against adversarial code.

## Exit codes

| Exit code | Meaning | Score |
|---|---|---|
| `pass` | The candidate imported and every supplied assertion passed. | `1.0` |
| `assertion_fail` | The candidate ran; at least one assertion failed. | `passed / total` |
| `syntax_error` | Compilation or import-time execution raised `SyntaxError`/`IndentationError`. | `0.0` |
| `timeout` | The wall-clock deadline expired. | `0.0` |
| `error` | Any other crash, denied import, output overflow, missing/malformed envelope, or non-zero runner exit. | `0.0` |

A generation with no fenced block, or a whitespace-only block, is an `error` with detail
`"no code block found"` at the `score_code_exec` layer. `run_candidate` is the lower-level
entry point and can execute an explicitly empty string if a caller really wants that.

Output overflow always uses the exact detail string `"output overflow"`. A candidate's own
top-level `assert` is an `error`, not an `assertion_fail`; only the supplied assertions produce
the fractional score. A syntax error inside a supplied assertion is a failed assertion.

## Denylist and its limits

The import denylist is:

```text
os.system  subprocess  socket  requests  urllib  urllib.request  urllib.error
urllib.parse  ctypes  shutil  http.client  asyncio.subprocess
```

A `MetaPathFinder` at the front of `sys.meta_path` raises
`ImportError("denied: <module>")` for a denied name or a submodule of one. This is an
import-resolution stub rather than a textual rewrite: it also catches aliases, `from x import
y`, `__import__`, and `importlib` calls. `os.system` is monkey-patched to raise
`ImportError("denied: os.system")`, so `from os import system` receives the patched function.

This is a backstop, not a boundary. Candidate code can still use unblocked file APIs,
`posix`, `sys.meta_path` manipulation, or a process-group escape; it also has no memory cap.
Do not run untrusted open-web code through this module.

## Timeout and output cap

- Default wall-clock budget: **3.0 s** (`DEFAULT_TIMEOUT_S`).
- `run_candidate` clamps every requested timeout to **0.1–30.0 s**.
- Output cap: **256 KiB** (262,144 bytes), including protocol bytes and candidate prints.
- On timeout or overflow the parent kills the child's process group, waits for it, and
  returns a normalized result instead of a traceback.

The parent uses `subprocess.Popen` and a `selectors` loop rather than `subprocess.run`. That is
intentional: `run` buffers stdout until the child exits, so a printing loop could exhaust host
memory before the cap is enforced. Streaming lets the parent kill as soon as the running total
crosses the cap.

## Pass-at-k

For `n` independent samples, `c` passing samples, and `k` draws, the unbiased estimator is:

```text
pass_at_k(n, c, k) = 1 - C(n - c, k) / C(n, k)
```

When `n - c < k`, the numerator is zero by convention, so the result is `1.0`. Worked values:

```text
pass_at_k(10, 3, 1) = 0.300
pass_at_k(10, 3, 5) = 231/252 ≈ 0.9166666666666666
pass_at_k(10, 3, 10) = 1.000
pass_at_k(10, 0, 5) = 0.000
```

The estimator is exposed for a future leaderboard layer. The demo counts a sample as passing
only when the whole task passes (`score == 1.0`); partial assertion credit does not count as a
task pass.

## Demo fixture

`fixtures/demo_cases.jsonl` is handcrafted, not model output. It has two tasks:

- `add_two`: five samples, three full passes. Includes a partial solution (first assertion
  passes, second fails), a syntax error, and two correct variants.
- `failure_modes`: six samples covering pass, `assertion_fail`, `syntax_error`, `timeout`,
  denied import (`error`), and output flood (`error` / `"output overflow"`).

The demo verifies every expected exit code and score, then prints `pass@1`, `pass@3`, and
`pass@n` for each task. Fixture-derived numbers are not benchmark results.

## Design decisions

- **Stdlib only at runtime.** The sandbox needs `subprocess`, `selectors`, `json`, and `math`;
  adding a dependency would not make it safer.
- **Duplicate lesson 70's extraction rule.** This module does not import sibling modules, so
  `extract_code` repeats the first-fence / next-fence rule and its tests.
- **Fresh interpreter per candidate.** State, monkey patches, and imports cannot leak between
  candidates. `-I` isolates the child from user site and environment configuration.
- **One stdout channel.** Candidate output and the protocol share a pipe so the cap covers all
  bytes. The marker plus `raw_decode` separates the result from noise.
- **Independent assertion evaluation.** Every target gets a result, even after earlier
  failures. This is what makes the score a fraction rather than a stop-at-first-failure bit.
- **Normalize every failure.** Timeout, overflow, syntax, denied import, and crashes all return
  the same `ExecResult` shape; downstream callers branch on `exit_code`, not on tracebacks.
- **No-code guard.** A generation with no fenced block scores zero instead of letting a
  vacuous `assert True` pass an empty program.
- **Toy tests are still assertions.** Target strings are not pytest files; they have toy
  failure modes and no fixtures, setup, teardown, or parallelism.

## What runs offline, and what a real provider would need

Everything here runs offline after the pinned dev dependencies are installed: extraction,
subprocess execution, scoring, pass-at-k, tests, type checking, and the demo. There is no
model call, network client, dataset download, GPU kernel, or cloud sandbox. No external
service is simulated because none is part of the execution metric.

There is no real-provider/GPU mode to enable. Generations are plain text; to evaluate real
model output, save each generation and call
`score_code_exec(generation_text, task_targets, timeout_s)`, or write the same shape used by
the demo fixture. Any provider adapter belongs upstream. This repository does not claim model
accuracy or benchmark numbers from any provider.

## Measured results

Measured locally with `./verify.sh` and the manual commands:

- **86 pytest cases passed** (`86 passed in 2.22s` on the local machine).
- **mypy strict:** `Success: no issues found in 1 source file`.
- **Demo:** all 11 samples matched their expected exit code and score, then printed:
  - `add_two: n=5 c=3 pass@1=0.600`, `pass@3=1.000`, `pass@5=1.000`
  - `failure_modes: n=6 c=1 pass@1=0.167`, `pass@3=0.500`, `pass@6=1.000`
  - `demo: ok`
- **`verify.sh`:** exits 0, including on a second consecutive run.

These are local fixture and test measurements, not a model benchmark.

## Limitations

- Not a real sandbox: no seccomp, container, network namespace, read-only filesystem, memory
  cap, or user separation. Candidate code runs as the host user in the host working directory.
- The denylist is bypassable (`os.remove`, `posix.system`, clearing `sys.meta_path`, a
  deliberate `setsid` escape). It is a backstop behind the timeout and cap, not a security
  boundary. Do not execute untrusted open-web code.
- The fixed result marker is not authenticated; adversarial candidate output could forge it.
- Stateful file/network tasks are unsupported. This module is for pure-function assertion
  tasks, not production CI suites.
- Assertion strings are toy tests: no pytest fixtures, parameterization, setup/teardown, or
  isolation between assertions.
- The tested kill path is POSIX process-group kill. The non-POSIX fallback kills only the
  direct child, not descendants.
- Each candidate pays Python interpreter startup cost. Batching would change the isolation
  contract.
- Targets are embedded in the `-c` argument, so very large assertion sets are bounded by the
  OS argument limit.
- `pass_at_k` uses the direct combination formula; it is intended for small sample counts and
  does not compute confidence intervals.
- No Docker/microVM, real provider, GPU, leaderboard aggregation, or production hardening.

## References

Sources consulted for the subprocess contract, the estimator, and the security caveats. No
code was copied from them.

1. AI Engineering From Scratch, **Code Exec Metric** capstone:
   <https://aiengineeringfromscratch.com/learn/capstone-projects/code-exec-metric>
2. Chen, M., Tworek, J., Jun, H., et al. (2021). *Evaluating Large Language Models Trained on
   Code.* arXiv:2107.03374. <https://arxiv.org/abs/2107.03374>
3. OpenAI **human-eval**, `human_eval/execution.py` (fresh-process execution and reliability
   guard):
   <https://github.com/openai/human-eval/blob/master/human_eval/execution.py>
4. Pigment Engineering (2026). *Sandboxing LLM-generated code execution* (defense-in-depth
   and why regex/denylist vetting is not a boundary):
   <https://engineering.pigment.com/2026/06/10/sandbox-for-llm-generated-code-execution>
