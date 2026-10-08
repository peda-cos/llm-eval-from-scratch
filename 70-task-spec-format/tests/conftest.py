from typing import Any, Callable

import pytest


@pytest.fixture
def task_factory() -> Callable[..., dict[str, Any]]:
    def make_task(**overrides: Any) -> dict[str, Any]:
        task: dict[str, Any] = {
            "task_id": "base_001",
            "category": "arithmetic",
            "prompt": "Question: 1 + 1\nAnswer:",
            "targets": ["2"],
            "metric_name": "exact_match",
            "post_process": "strip_whitespace",
            "metadata": {"difficulty": "easy"},
        }
        task.update(overrides)
        return task

    return make_task
