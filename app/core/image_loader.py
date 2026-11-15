"""Image loading with explicit validation and timing-friendly semantics."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class ImageLoadError(ValueError):
    """Raised when an inspection image cannot be read."""


def load_bgr_image(path: str | Path) -> np.ndarray:
    """Load a colour image in OpenCV's BGR channel order.

    Images are intentionally not resized: pixel measurements only remain
    meaningful when the original acquisition resolution is preserved.
    """

    image_path = Path(path)
    if not image_path.is_file():
        raise ImageLoadError(f"Image does not exist: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ImageLoadError(f"OpenCV could not decode image: {image_path}")
    return image
