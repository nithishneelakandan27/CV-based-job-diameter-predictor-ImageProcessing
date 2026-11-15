"""Engineering tolerance checks, separate from pixel processing and calibration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite


class ToleranceStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


def _positive_finite(value: float, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (not allow_zero and value == 0):
        comparator = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {comparator}")
    return float(value)


@dataclass(frozen=True)
class SymmetricTolerance:
    """Nominal dimension with a symmetric plus/minus tolerance, in mm."""

    nominal_mm: float
    plus_minus_mm: float

    def __post_init__(self) -> None:
        _positive_finite(self.nominal_mm, "nominal_mm")
        _positive_finite(self.plus_minus_mm, "plus_minus_mm", allow_zero=True)

    @property
    def lower_limit_mm(self) -> float:
        return self.nominal_mm - self.plus_minus_mm

    @property
    def upper_limit_mm(self) -> float:
        return self.nominal_mm + self.plus_minus_mm


@dataclass(frozen=True)
class ToleranceResult:
    nominal_mm: float
    measured_mm: float
    lower_limit_mm: float
    upper_limit_mm: float
    status: ToleranceStatus

    def as_dict(self) -> dict[str, float | str]:
        return {
            "nominal_mm": self.nominal_mm,
            "measured_mm": self.measured_mm,
            "lower_limit_mm": self.lower_limit_mm,
            "upper_limit_mm": self.upper_limit_mm,
            "status": self.status.value,
        }


def check_tolerance(measured_mm: float, specification: SymmetricTolerance) -> ToleranceResult:
    """Return PASS inclusively at both engineering tolerance boundaries."""

    measured = _positive_finite(measured_mm, "measured_mm", allow_zero=True)
    lower, upper = specification.lower_limit_mm, specification.upper_limit_mm
    status = ToleranceStatus.PASS if lower <= measured <= upper else ToleranceStatus.FAIL
    return ToleranceResult(
        nominal_mm=specification.nominal_mm,
        measured_mm=measured,
        lower_limit_mm=lower,
        upper_limit_mm=upper,
        status=status,
    )
