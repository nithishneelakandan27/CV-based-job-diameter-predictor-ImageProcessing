"""Configurable four-diameter measurement definitions.

Positions are normalized to an aligned *job ROI*, not hard-coded image rows.
That keeps drawing/fixture knowledge in configuration and out of algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping


@dataclass(frozen=True)
class MeasurementRegion:
    """Vertical band of a diameter section, expressed as fractions of the job ROI.

    The ROI must have already been located and alignment-normalized such that the
    cylindrical axis is vertical.  Fractions therefore remain portable across
    pixel resolutions but need validation for each fixture/orientation.
    """

    top_fraction: float
    bottom_fraction: float

    def __post_init__(self) -> None:
        for name, value in (("top_fraction", self.top_fraction), ("bottom_fraction", self.bottom_fraction)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                raise ValueError(f"{name} must be a number between 0 and 1")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.top_fraction >= self.bottom_fraction:
            raise ValueError("top_fraction must be less than bottom_fraction")

    def pixel_rows(self, image_height: int) -> tuple[int, int]:
        """Return non-empty [start, end) rows for a concrete job ROI."""

        if not isinstance(image_height, int) or image_height < 2:
            raise ValueError("image_height must be an integer of at least 2")
        start = int(round(self.top_fraction * image_height))
        end = int(round(self.bottom_fraction * image_height))
        start = min(max(start, 0), image_height - 1)
        end = min(max(end, start + 1), image_height)
        if end <= start:
            raise ValueError("measurement region resolves to no image rows")
        return start, end


@dataclass(frozen=True)
class DiameterSpec:
    """Configuration for one independently measured diameter section."""

    name: str
    region: MeasurementRegion
    nominal_mm: float | None = None
    tolerance_mm: float | None = None
    edge_strategy: str = "sobel_x"

    def __post_init__(self) -> None:
        if self.name not in {"D1", "D2", "D3", "D4"}:
            raise ValueError("name must be one of D1, D2, D3, D4")
        if not isinstance(self.region, MeasurementRegion):
            raise ValueError("region must be a MeasurementRegion")
        if (self.nominal_mm is None) != (self.tolerance_mm is None):
            raise ValueError("nominal_mm and tolerance_mm must be configured together")
        if self.nominal_mm is not None:
            if (
                isinstance(self.nominal_mm, bool)
                or isinstance(self.tolerance_mm, bool)
                or not isinstance(self.nominal_mm, (int, float))
                or not isinstance(self.tolerance_mm, (int, float))
                or not isfinite(self.nominal_mm)
                or not isfinite(self.tolerance_mm)
                or self.nominal_mm <= 0
                or self.tolerance_mm < 0
            ):
                raise ValueError("nominal_mm must be positive and tolerance_mm non-negative")

    @property
    def is_tolerance_configured(self) -> bool:
        return self.nominal_mm is not None and self.tolerance_mm is not None


@dataclass(frozen=True)
class FourDiameterSpec:
    """A complete, uniquely named D1/D2/D3/D4 specification for one job type."""

    diameters: tuple[DiameterSpec, ...]
    job_type: str = "stepped_cylindrical_component"

    def __post_init__(self) -> None:
        names = tuple(spec.name for spec in self.diameters)
        if set(names) != {"D1", "D2", "D3", "D4"} or len(names) != 4:
            raise ValueError("FourDiameterSpec requires exactly one each of D1, D2, D3 and D4")

    def ordered(self) -> tuple[DiameterSpec, ...]:
        by_name = {spec.name: spec for spec in self.diameters}
        return tuple(by_name[name] for name in ("D1", "D2", "D3", "D4"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FourDiameterSpec":
        """Load a JSON-compatible manual/fixture configuration.

        Example diameter entry: ``{"name": "D1", "region":
        {"top_fraction": 0.0, "bottom_fraction": 0.2}}``. Nominal and
        tolerance remain optional so experimental configs cannot imply PASS/FAIL.
        """

        entries = data.get("diameters")
        if not isinstance(entries, list):
            raise ValueError("configuration requires a 'diameters' list")
        specs: list[DiameterSpec] = []
        for entry in entries:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("region"), Mapping):
                raise ValueError("each diameter requires a region object")
            region_data = entry["region"]
            specs.append(
                DiameterSpec(
                    name=entry.get("name", ""),
                    region=MeasurementRegion(
                        top_fraction=region_data.get("top_fraction"),
                        bottom_fraction=region_data.get("bottom_fraction"),
                    ),
                    nominal_mm=entry.get("nominal_mm"),
                    tolerance_mm=entry.get("tolerance_mm"),
                    edge_strategy=entry.get("edge_strategy", "sobel_x"),
                )
            )
        return cls(tuple(specs), job_type=data.get("job_type", "stepped_cylindrical_component"))
