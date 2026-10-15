import json
import math

import numpy as np

from main import (
    CalibrationReport,
    ModelReport,
    PerplexityResult,
    brier_decomposition,
    brier_score,
    expected_calibration_error,
    reliability_diagram,
)


def test_calibration_report_matches_the_primitive_functions() -> None:
    confidences = [0.1, 0.2, 0.2, 0.7, 0.9]
    correct = [0, 1, 0, 1, 1]
    report = CalibrationReport.from_predictions(confidences, correct, bins=5)
    assert math.isclose(
        report.ece,
        expected_calibration_error(confidences, correct, bins=5),
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        report.brier,
        brier_score(confidences, correct),
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    expected_conf, expected_acc, expected_counts = reliability_diagram(
        confidences, correct, bins=5
    )
    assert np.array_equal(report.reliability[0], expected_conf)
    assert np.array_equal(report.reliability[1], expected_acc)
    assert np.array_equal(report.reliability[2], expected_counts)
    assert report.decomposition == brier_decomposition(confidences, correct)
    assert report.populated_bins == 4
    assert report.n_samples == 5
    assert report.bins == 5


def test_calibration_report_records_custom_bins_and_unpopulated_bins() -> None:
    report = CalibrationReport.from_predictions(
        [0.05, 0.05, 0.55, 0.95], [1, 0, 1, 1], bins=15
    )
    assert report.bins == 15
    assert report.n_samples == 4
    assert report.populated_bins == 3
    counts = report.reliability[2]
    assert counts.shape == (15,)
    assert counts[0] == 2
    assert counts[8] == 1
    assert counts[14] == 1
    assert counts[1] == 0


def test_model_report_exposes_exactly_the_three_scalars() -> None:
    calibration = CalibrationReport.from_predictions([0.9, 0.9], [1, 0])
    held_out = PerplexityResult.from_token_nll([1.0], [1])
    report = ModelReport("synthetic", calibration, held_out)
    metrics = report.metrics()
    assert set(metrics) == {"perplexity", "ece", "brier"}
    assert math.isclose(metrics["ece"], 0.4, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(metrics["brier"], 0.41, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(metrics["perplexity"], math.e, rel_tol=0.0, abs_tol=1e-12)


def test_model_report_to_dict_is_json_safe() -> None:
    calibration = CalibrationReport.from_predictions([0.9, 0.9], [1, 0])
    held_out = PerplexityResult.from_token_nll([1.0], [1])
    report = ModelReport("synthetic", calibration, held_out)
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["model"] == "synthetic"
    assert payload["metrics"]["ece"] == calibration.ece
    assert payload["metrics"]["brier"] == calibration.brier
    assert payload["metrics"]["perplexity"] == held_out.perplexity
    assert payload["calibration"]["populated_bins"] == 1
    assert payload["calibration"]["reliability"]["counts"] == [0] * 9 + [2]
    assert payload["calibration"]["decomposition"]["base_rate"] == 0.5
    assert payload["perplexity"]["total_tokens"] == 1
    assert isinstance(payload["calibration"]["reliability"]["counts"], list)


def test_model_report_without_artifacts_is_empty_but_serializable() -> None:
    report = ModelReport("unmeasured")
    assert report.metrics() == {}
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["metrics"] == {}
    assert payload["calibration"] is None
    assert payload["perplexity"] is None
