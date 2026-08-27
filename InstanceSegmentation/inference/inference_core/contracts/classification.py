"""Classifier result contract independent of feature implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Classification:
    class_id: int
    class_name: str
    score: float
    probabilities: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.class_id < 0:
            raise ValueError("classification class_id must be non-negative")
        if not self.class_name:
            raise ValueError("classification class_name must not be empty")
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("classification score must be in [0, 1]")
        if self.probabilities is not None:
            if any(
                not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in self.probabilities
            ):
                raise ValueError("classification probabilities must be in [0, 1]")


__all__ = ["Classification"]
