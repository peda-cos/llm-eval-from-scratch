from main import main


def test_demo_prints_expected_synthetic_metric_lines(capsys) -> None:
    assert main(["demo"]) == 0
    captured = capsys.readouterr()
    assert (
        "model=well_calibrated n=1000 ece=0.000000 brier=0.167500 "
        "populated_bins=10 reliability=0.000000 resolution=0.082500 "
        "uncertainty=0.250000"
    ) in captured.out
    assert (
        "model=overconfident n=100 ece=0.300000 brier=0.330000 "
        "populated_bins=1 reliability=0.090000 resolution=0.000000 "
        "uncertainty=0.240000"
    ) in captured.out
    assert (
        "model=underconfident n=100 ece=0.300000 brier=0.180000 "
        "populated_bins=1 reliability=0.090000 resolution=0.000000 "
        "uncertainty=0.090000"
    ) in captured.out
    assert (
        "perplexity corpus=held_out_synthetic tokens=30 mean_nll=1.000000 "
        "ppl=2.718282"
    ) in captured.out
    assert captured.out.strip().endswith("demo: ok")
    assert captured.err == ""


def test_invalid_cli_argument_returns_usage_error(capsys) -> None:
    assert main(["bogus"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "usage: python main.py [demo]\n"
