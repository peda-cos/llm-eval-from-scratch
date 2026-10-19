"""The adapter seam between the runner and a model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from eval_runner.tasks import TaskSpec


@dataclass(frozen=True, slots=True)
class Generation:
    """One model generation plus the self-reported signals used for calibration."""

    text: str
    confidence: float
    token_nll: float | None = None
    token_count: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise ValueError("confidence must be a number in [0, 1]")
        if not math.isfinite(float(self.confidence)) or not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be a finite number in [0, 1]")
        if self.token_nll is not None:
            if isinstance(self.token_nll, bool) or not isinstance(self.token_nll, (int, float)):
                raise ValueError("token_nll must be a finite non-negative number")
            if not math.isfinite(float(self.token_nll)) or float(self.token_nll) < 0.0:
                raise ValueError("token_nll must be a finite non-negative number")
        if self.token_count is not None:
            if isinstance(self.token_count, bool) or not isinstance(self.token_count, int):
                raise ValueError("token_count must be a non-negative integer")
            if self.token_count < 0:
                raise ValueError("token_count must be a non-negative integer")


class ModelAdapter(Protocol):
    """Any object with a stable model id and one generation method."""

    model_id: str

    def generate(self, prompt: str, task: TaskSpec) -> Generation: ...
