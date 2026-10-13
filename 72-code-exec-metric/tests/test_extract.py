import pytest

from main import extract_code


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("before\n```python\nprint(1)\n```\nafter", "print(1)"),
        ("```\n\nx = 1\n\n```", "\nx = 1\n"),
        ("  ```python\nx = 1\n  ```", "x = 1"),
        ("```python\n    x = 1\n```", "    x = 1"),
        ("first\n```\none\n```\nsecond\n```\ntwo\n```", "one"),
        ("a\r\n```python\r\nx = 1\r\n```\r\nb", "x = 1"),
        ("```extra\nvalue = 2\n```", "value = 2"),
    ],
)
def test_extract_code_matches_lesson_70_rule(text: str, expected: str) -> None:
    assert extract_code(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "no fence here",
        "```python\nprint(1)\n",
        "~~~python\nprint(1)\n~~~",
    ],
)
def test_extract_code_returns_empty_without_a_complete_backtick_block(
    text: str,
) -> None:
    assert extract_code(text) == ""


def test_extract_code_only_keeps_fenced_content_not_prose() -> None:
    generation = (
        "I would write it like this.\n\n"
        "```python\n"
        "def double(x):\n"
        "    return x * 2\n"
        "```\n\n"
        "That should be correct."
    )

    assert extract_code(generation) == "def double(x):\n    return x * 2"
