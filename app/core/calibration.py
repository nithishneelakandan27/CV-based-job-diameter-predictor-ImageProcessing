"""Calibration abstractions for converting pixel measurements to millimetres.

``LinearScaleCalibration`` is deliberately small: it is valid only where a
constant mm-per-pixel relationship has been demonstrated for the actual camera,
lens, working distance, and measurement plane. It does not correct perspective
or lens distortion. A future camera calibration implementation can implement
``CalibrationModel`` and provide undistortion/position-dependent conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Protocol, runtime_checkable


@runtime_checkable
class CalibrationModel(Protocol):
    """Interface used by geometric measurements, independent of calibration type."""

    def pixels_to_mm(self, pixels: float) -> float:
        """Convert a non-negative distance in pixels to millimetres."""


@dataclass(frozen=True)
class LinearScaleCalibration:
    """A constant scale calibration for a calibrated measurement plane."""

    mm_per_pixel: float
    reference_description: str = ""

    def __post_init__(self) -> None:
        if (
            isinstance(self.mm_per_pixel, bool)
            or not isinstance(self.mm_per_pixel, (int, float))
            or not isfinite(self.mm_per_pixel)
            or self.mm_per_pixel <= 0
        ):
            raise ValueError("mm_per_pixel must be a positive number")

    def pixels_to_mm(self, pixels: float) -> float:
        if (
            isinstance(pixels, bool)
            or not isinstance(pixels, (int, float))
            or not isfinite(pixels)
            or pixels < 0
        ):
            raise ValueError("pixels must be a non-negative number")
        return float(pixels) * self.mm_per_pixel
