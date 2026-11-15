"""Job localization implementations independent of diameter measurement.

The BMP fixture images contain dark jobs on a red backlight.  The geometric
detector below deliberately uses that stable, classical silhouette contrast; it
does not require manual ROIs or a learned model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import cv2
import numpy as np


@dataclass(frozen=True)
class JobROI:
    """One rectangular candidate job location in source-image pixels."""

    job_id: str
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if not self.job_id:
            raise ValueError("job_id must not be empty")
        if self.x < 0 or self.y < 0 or self.width <= 1 or self.height <= 1:
            raise ValueError("ROI x/y must be non-negative and width/height must exceed 1 pixel")

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract this ROI, rejecting an ROI that is outside the image."""

        if image is None or image.ndim < 2:
            raise ValueError("image must be a non-empty array")
        image_height, image_width = image.shape[:2]
        if self.x + self.width > image_width or self.y + self.height > image_height:
            raise ValueError(f"ROI {self.job_id} is outside the image bounds")
        return image[self.y : self.y + self.height, self.x : self.x + self.width].copy()


@runtime_checkable
class JobDetector(Protocol):
    """Replaceable source of job ROIs, independent of downstream measurement."""

    def locate(self, image: np.ndarray) -> tuple[JobROI, ...]:
        """Return candidate jobs in source-image coordinates."""


@dataclass(frozen=True)
class FixedRoiJobDetector:
    """Fixture/manual job locations supplied by validated configuration."""

    rois: tuple[JobROI, ...]

    def __post_init__(self) -> None:
        if not self.rois:
            raise ValueError("at least one job ROI is required")
        ids = [roi.job_id for roi in self.rois]
        if len(set(ids)) != len(ids):
            raise ValueError("job ROI identifiers must be unique")

    def locate(self, image: np.ndarray) -> tuple[JobROI, ...]:
        # Bounds are checked during extraction so errors identify the exact ROI.
        if image is None or image.size == 0:
            raise ValueError("image must not be empty")
        return self.rois


@dataclass(frozen=True)
class DarkRedSilhouetteJobDetector:
    """Locate the horizontal jobs in the supplied red-backlit BMP fixture.

    Jobs are darker than the background in the BGR red channel.  Otsu supplies
    a per-image exposure adaptation.  The safety ceiling ``red_ceiling_fraction``
    prevents background texture from being included in an unusually bright image
    by limiting the threshold to a fraction of the red-channel peak intensity.
    Connected components are filtered by their expected fixture-scale area and
    aspect ratio, padded for outer-edge measurements, then given stable
    row/column identifiers.

    Background
    ----------
    The original implementation used a hard ``max_red_value = 35`` clamp that
    caused the Otsu threshold to be artificially suppressed on brighter images
    (observed Otsu values of 50 on images where the backlight peak was ~125).
    This truncated the outer-flange silhouette of D1 by up to 26 pixels,
    introducing a systematic error of roughly 2.6 mm on affected images.

    The fix replaces the arbitrary constant with a physics-grounded relative
    ceiling: ``floor(red_ceiling_fraction * red_channel_peak)``.  Across all
    observed fixture images, the Otsu threshold is consistently ~40 % of the
    red-channel peak intensity, well below the default 55 % ceiling.  The
    ceiling therefore never fires on correctly captured images and only guards
    against pathological cases where no clear bimodal valley exists.

    The ``max_red_value`` field is preserved for backward compatibility with
    existing callers that pass it explicitly (e.g., test code), but it is no
    longer used in the threshold computation.  Pass ``red_ceiling_fraction``
    instead to control the adaptive guard.
    """

    max_red_value: int = 35          # retained for API compatibility; not used internally
    close_kernel_size: int = 9
    minimum_area: int = 50_000
    minimum_width: int = 400
    minimum_height: int = 200
    padding: int = 12
    red_ceiling_fraction: float = 0.55  # Otsu capped at this fraction of the red-channel peak

    def __post_init__(self) -> None:
        if self.max_red_value <= 0 or self.close_kernel_size < 1 or self.close_kernel_size % 2 == 0:
            raise ValueError("red threshold must be positive and close_kernel_size must be positive and odd")
        if self.minimum_area <= 0 or self.minimum_width <= 1 or self.minimum_height <= 1 or self.padding < 0:
            raise ValueError("component limits must be positive and padding must be non-negative")
        if not isinstance(self.red_ceiling_fraction, float) or not (0.0 < self.red_ceiling_fraction <= 1.0):
            raise ValueError("red_ceiling_fraction must be a float in (0, 1]")

    def _adaptive_threshold(self, red: np.ndarray) -> int:
        """Return a per-image silhouette threshold using adaptive Otsu with a
        physics-grounded relative safety ceiling.

        Strategy
        --------
        1. Compute the global Otsu threshold on the red channel.  Otsu
           maximises inter-class variance between the dark job silhouettes
           and the bright red backlight, so it tracks illumination changes
           automatically without any hard-coded constant.
        2. Clamp the result at ``floor(red_ceiling_fraction * peak_red)``
           where ``peak_red`` is the 99th-percentile of the red channel
           (robust against isolated saturated pixels).  This ceiling is
           physically grounded: job material always appears below 55 % of
           the backlight peak, so a threshold above that level would indicate
           a pathological image (no clear dark/bright separation) rather than
           a legitimately bright backlight.
        3. The safety ceiling is also bounded at 90 absolute counts so that
           even a very high-dynamic-range image cannot accidentally include
           background texture in the silhouette mask.
        """
        otsu_val = int(round(float(cv2.threshold(red, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])))
        peak_red = int(np.percentile(red, 99))
        ceiling = min(int(self.red_ceiling_fraction * max(peak_red, 1)), 90)
        return min(otsu_val, ceiling)

    def locate(self, image: np.ndarray) -> tuple[JobROI, ...]:
        if image is None or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise ValueError("image must be a non-empty BGR image")

        red = image[:, :, 2]
        threshold = self._adaptive_threshold(red)
        silhouette = cv2.inRange(red, 0, threshold)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.close_kernel_size, self.close_kernel_size)
        )
        silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_CLOSE, kernel)
        _, _, statistics, centroids = cv2.connectedComponentsWithStats(silhouette, connectivity=8)

        candidates: list[tuple[int, int, int, int, float, float]] = []
        for (x, y, width, height, area), (center_x, center_y) in zip(statistics[1:], centroids[1:]):
            if area < self.minimum_area or width < self.minimum_width or height < self.minimum_height:
                continue
            # The fixture parts are horizontal; reject border/background blobs.
            if width <= height:
                continue
            candidates.append((int(x), int(y), int(width), int(height), float(center_x), float(center_y)))
        if not candidates:
            raise ValueError("No job silhouettes matched the configured fixture geometry")

        rows = self._group_rows(candidates)
        image_height, image_width = image.shape[:2]
        rois: list[JobROI] = []
        for row_index, row in enumerate(rows, start=1):
            for column_index, (x, y, width, height, _, _) in enumerate(sorted(row, key=lambda item: item[4]), start=1):
                left = max(0, x - self.padding)
                top = max(0, y - self.padding)
                right = min(image_width, x + width + self.padding)
                bottom = min(image_height, y + height + self.padding)
                rois.append(JobROI(f"job_r{row_index:02d}_c{column_index:02d}", left, top, right - left, bottom - top))
        return tuple(rois)

    @staticmethod
    def _group_rows(candidates: list[tuple[int, int, int, int, float, float]]) -> list[list[tuple[int, int, int, int, float, float]]]:
        """Group components by fixture row before sorting columns.

        Sorting raw top-left coordinates alone is not deterministic by physical
        column when a slightly tilted job has a different top edge.  Component
        centroids and the median job height make the grid assignment stable.
        """

        row_gap = float(np.median([candidate[3] for candidate in candidates])) * 0.5
        rows: list[list[tuple[int, int, int, int, float, float]]] = []
        for candidate in sorted(candidates, key=lambda item: item[5]):
            if not rows or candidate[5] - float(np.mean([item[5] for item in rows[-1]])) > row_gap:
                rows.append([candidate])
            else:
                rows[-1].append(candidate)
        return rows
