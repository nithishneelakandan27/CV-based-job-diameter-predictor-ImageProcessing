"""Shared result objects for the inspection pipeline.

The result types deliberately distinguish an exploratory pixel measurement from a
validated, calibrated inspection result. A phone image without a fixture and
calibration must never be represented as an industrial PASS/FAIL decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class InspectionState(str, Enum):
    """How far a result has progressed through the inspection workflow."""

    EXPERIMENT = "EXPERIMENT"
    VALIDATED = "VALIDATED"
    INCOMPLETE = "INCOMPLETE"


@dataclass
class PipelineTimings:
    """Per-stage timings plus a total for one image."""

    stages: dict[str, float] = field(default_factory=dict)

    def record(self, name: str, milliseconds: float) -> None:
        self.stages[name] = milliseconds

    @property
    def total_milliseconds(self) -> float:
        return sum(self.stages.values())

    def as_dict(self) -> dict[str, float]:
        return {**self.stages, "complete_pipeline": self.total_milliseconds}


@dataclass(frozen=True)
class ExperimentResult:
    """Output of an uncalibrated image-processing experiment."""

    source: str
    state: InspectionState
    edge_metrics: dict[str, dict[str, float | int]]
    timings_ms: dict[str, float]
    notes: tuple[str, ...] = ()
    measurement: Any | None = None
