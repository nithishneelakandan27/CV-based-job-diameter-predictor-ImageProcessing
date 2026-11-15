"""Geometric measurement primitives independent of image acquisition details."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .calibration import CalibrationModel


def _finite_number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


@dataclass(frozen=True)
class PixelDiameterMeasurement:
    """A diameter measured between two localized edges in pixel coordinates."""

    left_edge_x: float
    right_edge_x: float
    method: str = "horizontal_edge_distance"
    region_id: str | None = None

    def __post_init__(self) -> None:
        left = _finite_number(self.left_edge_x, "left_edge_x")
        right = _finite_number(self.right_edge_x, "right_edge_x")
        if right <= left:
            raise ValueError("right_edge_x must be greater than left_edge_x")

    @property
    def diameter_pixels(self) -> float:
        return self.right_edge_x - self.left_edge_x

    def calibrated_mm(self, calibration: CalibrationModel) -> "CalibratedDiameterMeasurement":
        """Convert only through an explicit calibration model."""

        return CalibratedDiameterMeasurement(
            diameter_pixels=self.diameter_pixels,
            diameter_mm=calibration.pixels_to_mm(self.diameter_pixels),
            method=self.method,
            region_id=self.region_id,
        )


@dataclass(frozen=True)
class CircleDiameterMeasurement:
    """Diameter derived from a fitted circle radius, still in pixel units."""

    center_x: float
    center_y: float
    radius_pixels: float
    method: str = "circle_fit"
    region_id: str | None = None

    def __post_init__(self) -> None:
        _finite_number(self.center_x, "center_x")
        _finite_number(self.center_y, "center_y")
        radius = _finite_number(self.radius_pixels, "radius_pixels")
        if radius <= 0:
            raise ValueError("radius_pixels must be greater than zero")

    @property
    def diameter_pixels(self) -> float:
        return self.radius_pixels * 2.0


@dataclass(frozen=True)
class CalibratedDiameterMeasurement:
    """A pixel diameter after a specified calibration model converted it to mm."""

    diameter_pixels: float
    diameter_mm: float
    method: str
    region_id: str | None = None

    def __post_init__(self) -> None:
        if _finite_number(self.diameter_pixels, "diameter_pixels") <= 0:
            raise ValueError("diameter_pixels must be greater than zero")
        if _finite_number(self.diameter_mm, "diameter_mm") <= 0:
            raise ValueError("diameter_mm must be greater than zero")
