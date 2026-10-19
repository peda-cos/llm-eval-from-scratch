from __future__ import annotations

import json
import math

import pytest

from eval_runner import Generation, RunnerError, run_eval
from eval_runner.runner import _correct_from_score
from eval_runner.tasks import TaskSpec, load_fixtures
from main import (
    BiasedAdapter,
    NoisyAdapter,
    RuleBasedAdapter,
    check_demo_clean,
    demo_adapters,
)


def _task(
    task_id: str,
    target: str,
    *,
    prompt: str | None = None,
) -> TaskSpec:
    return {
        "task_id": task_id,
        "category": "arithmetic",
        "prompt": prompt or f"Question: 1 + 1\nAnswer:",
        "targets": [target],
        "metric_name": "exact_match",
        "post_process": "strip_whitespace",
    }


class ConstantAdapter:
    def __init__(
        self,
        model_id: str,
        answers: dict[str, str],
        confidence: float = 0.9,
    ) -> None:
        self.model_id = model_id
        self.answers = answers
        self.confidence = confidence

    def generate(self, prompt: str, task: TaskSpec) -> Generation:
        return Generation(text=self.answers[task["task_id"]], confidence=self.confidence)


class RaisingAdapter:
    model_id = "raising"

    def generate(self, prompt: str, task: TaskSpec) -> Generation:
        raise RuntimeError("boom")


class BadReturnAdapter:
    model_id = "bad_return"

    def generate(self, prompt: str, task: TaskSpec) -> Generation:
        return "not a generation"  # type: ignore[return-value]


class ConfidenceAdapter:
    model_id = "confidence"

    def __init__(self, values: dict[str, tuple[str, float]]) -> None:
        self.values = values

    def generate(self, prompt: str, task: TaskSpec) -> Generation:
        text, confidence = self.values[task["task_id"]]
        return Generation(text=text, confidence=confidence)


def test_generation_validation() -> None:
    generation = Generation("answer", 0.5, token_nll=0.2, token_count=2)
    assert generation.text == "answer"
    assert generation.token_nll == 0.2
    assert generation.token_count == 2

    for confidence in (-0.1, 1.1, math.nan, True):
        with pytest.raises(ValueError, match="confidence"):
            Generation("x", confidence)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="text"):
        Generation(123, 0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="token_nll"):
        Generation("x", 0.5, token_nll=-1.0)
    with pytest.raises(ValueError, match="token_count"):
        Generation("x", 0.5, token_count=-1)
    with pytest.raises(ValueError, match="token_count"):
        Generation("x", 0.5, token_count=True)  # type: ignore[arg-type]


def test_single_pass_result_leaderboard_and_envelope_shape() -> None:
    tasks = [_task("t1", "2"), _task("t2", "3")]
    adapter = ConstantAdapter("perfect", {"t1": "2", "t2": "wrong"})
    report = run_eval([adapter], tasks, parallel=False, bootstrap_samples=20, seed=3)

    assert len(report.results) == 2
    assert [result.score for result in report.results] == [1.0, 0.0]
    assert [result.correct for result in report.results] == [True, False]
    assert report.leaderboard[0].model_id == "perfect"
    assert report.leaderboard[0].mean_score == 0.5
    assert report.calibration["perfect"].n_samples == 2
    assert report.calibration["perfect"].ece == pytest.approx(0.4)
    assert report.workers == 1
    assert report.parallel is False
    assert report.tasks == 2

    payload = report.to_dict()
    assert payload["schema_version"] == 1
    assert payload["status"] == "ok"
    assert payload["summary"]["models"] == 1  # type: ignore[index]
    assert payload["summary"]["model_ids"] == ["perfect"]  # type: ignore[index]
    assert set(payload["results"][0]) == {  # type: ignore[index]
        "model_id",
        "task_id",
        "category",
        "metric_name",
        "score",
        "correct",
        "confidence",
        "prediction",
        "duration_ms",
        "token_nll",
        "token_count",
        "metric_detail",
    }
    assert payload["leaderboard"][0]["rank"] == 1  # type: ignore[index]
    assert json.loads(report.to_json()) == payload
    markdown = report.render_markdown()
    assert "| Model | ECE | Brier | Populated bins | Samples |" in markdown
    assert "perfect" in markdown


def test_parallel_matches_sequential_for_scores_and_calibration() -> None:
    tasks = [_task("t1", "2"), _task("t2", "3"), _task("t3", "4")]
    adapter = ConstantAdapter("same", {"t1": "2", "t2": "wrong", "t3": "4"})
    sequential = run_eval([adapter], tasks, parallel=False, bootstrap_samples=30, seed=11)
    parallel = run_eval([adapter], tasks, parallel=True, workers=2, bootstrap_samples=30, seed=11)
    assert [(r.task_id, r.score, r.prediction, r.correct) for r in sequential.results] == [
        (r.task_id, r.score, r.prediction, r.correct) for r in parallel.results
    ]
    assert sequential.leaderboard == parallel.leaderboard
    assert sequential.calibration["same"] == parallel.calibration["same"]
    assert parallel.workers == 2
    assert sequential.workers == 1


def test_calibration_buffer_uses_every_prediction_in_task_order() -> None:
    tasks = [_task("t1", "2"), _task("t2", "3"), _task("t3", "4"), _task("t4", "5")]
    adapter = ConfidenceAdapter(
        {
            "t1": ("2", 0.9),
            "t2": ("wrong", 0.8),
            "t3": ("4", 0.7),
            "t4": ("wrong", 0.6),
        }
    )
    report = run_eval([adapter], tasks, parallel=False, bootstrap_samples=20, seed=9)
    calibration = report.calibration["confidence"]
    assert calibration.n_samples == 4
    assert sum(calibration.counts) == 4
    assert calibration.mean_confidence == pytest.approx(
        tuple([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.7, 0.8, 0.9])
    )
    assert calibration.populated_bins == 4
    assert [result.correct for result in report.results] == [True, False, True, False]


def test_correct_from_score_thresholds() -> None:
    assert _correct_from_score("exact_match", 1.0) is True
    assert _correct_from_score("exact_match", 0.999) is False
    assert _correct_from_score("accuracy", 1.0) is True
    assert _correct_from_score("code_exec", 1.0) is True
    assert _correct_from_score("code_exec", 0.5) is False
    assert _correct_from_score("f1", 0.5) is True
    assert _correct_from_score("bleu_4", 0.5) is True
    assert _correct_from_score("rouge_l", 0.49) is False
    with pytest.raises(ValueError, match="unknown metric"):
        _correct_from_score("not-a-metric", 1.0)


def test_validation_and_adapter_failures_are_structured() -> None:
    task = _task("t1", "2")
    with pytest.raises(RunnerError, match="adapter"):
        run_eval([], [task])
    with pytest.raises(RunnerError, match="task"):
        run_eval([ConstantAdapter("a", {})], [])
    with pytest.raises(RunnerError, match="duplicate"):
        run_eval([ConstantAdapter("a", {}), ConstantAdapter("a", {})], [task])
    with pytest.raises(RunnerError, match="invalid task"):
        run_eval([ConstantAdapter("a", {})], [{"task_id": "bad"}])  # type: ignore[list-item]

    with pytest.raises(RunnerError) as caught:
        run_eval([RaisingAdapter()], [task], parallel=False)
    assert caught.value.stage == "generate"
    assert caught.value.model_id == "raising"
    assert caught.value.task_id == "t1"
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert caught.value.to_dict()["stage"] == "generate"

    with pytest.raises(RunnerError, match="must return Generation"):
        run_eval([BadReturnAdapter()], [task], parallel=False)


def test_demo_gate_orders_models_and_calibration() -> None:
    tasks = load_fixtures()
    report = run_eval(
        demo_adapters(),
        tasks,
        parallel=False,
        bootstrap_samples=50,
        seed=7,
    )
    check_demo_clean(report, len(tasks))
    assert len(report.results) == 30
    means = {row.model_id: row.mean_score for row in report.leaderboard}
    assert means["rule_based"] == 1.0
    assert means["biased"] == pytest.approx(0.4)
    assert means["noisy"] < 0.1
    assert means["rule_based"] > means["biased"] > means["noisy"]
    ece = {model_id: report.calibration[model_id].ece for model_id in report.model_ids}
    assert ece["noisy"] > ece["biased"] > ece["rule_based"]
    assert {type(adapter) for adapter in demo_adapters()} == {
        RuleBasedAdapter,
        BiasedAdapter,
        NoisyAdapter,
    }

    with pytest.raises(RunnerError, match="unexpected type"):
        check_demo_clean("not-a-report", len(tasks))
