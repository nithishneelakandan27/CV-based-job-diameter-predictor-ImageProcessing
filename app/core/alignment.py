"""Job ROI alignment contract.

The current implementation is deliberately identity-only. A future fixture or
geometric alignment stage can implement ``JobNormalizer`` without changing the
measurement-region or edge-localization interfaces.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import cv2


@runtime_checkable
class JobNormalizer(Protocol):
    def normalize(self, job_image: np.ndarray) -> np.ndarray:
        """Return an upright job ROI suitable for normalized measurement bands."""


class IdentityJobNormalizer:
    """No-op placeholder until fixture orientation/alignment is validated."""

    def normalize(self, job_image: np.ndarray) -> np.ndarray:
        if job_image is None or job_image.size == 0:
            raise ValueError("job_image must not be empty")
        return job_image


class HorizontalFlangeLeftJobNormalizer:
    """Turn a horizontal, flange-left job upright for the existing D1-D4 engine.

    The measurement engine defines D1 in the bottom region and D4 at the top.
    A counter-clockwise rotation maps a source job whose largest flange is on
    the left into that physical order: D1 bottom, then D2/D3, then D4 top.
    """

    def normalize(self, job_image: np.ndarray) -> np.ndarray:
        if job_image is None or job_image.ndim != 3 or job_image.shape[2] != 3 or job_image.size == 0:
            raise ValueError("job_image must be a non-empty BGR image")
        return cv2.rotate(job_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
