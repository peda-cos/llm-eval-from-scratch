import json

from main import main


def test_demo_is_deterministic_and_emits_tables_and_json(capsys) -> None:
    assert main(["demo"]) == 0
    first = capsys.readouterr().out

    assert main([]) == 0
    second = capsys.readouterr().out
    assert first == second

    assert "| 1 | atlas-7b | 0.82 | 0.79-0.85 | 0.67 | 12 |" in first
    assert "| 3 | helios-2026-experime | 0.49 | 0.44-0.53 | 0.00 | 12 |" in first
    assert "| atlas-7b | beacon-13b | 0.017 | -0.009 to 0.039 | no | 12 |" in first

    json_text = first.split("## JSON report\n", 1)[1]
    payload = json.loads(json_text)
    assert len(payload["models"]) == 3
    assert payload["seed"] == 7
    assert [pair["significant"] for pair in payload["pairwise"]] == [
        False,
        True,
        True,
    ]
    assert payload["models"][0]["model_id"] == "atlas-7b"
    assert payload["models"][0]["categories"] == {
        "code": 0.82,
        "math": 0.85,
        "reasoning": 0.7466666666666667,
        "safety": 0.86,
    }


def test_unknown_command_returns_usage_error(capsys) -> None:
    assert main(["not-a-command"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "usage: python main.py [demo]\n"
