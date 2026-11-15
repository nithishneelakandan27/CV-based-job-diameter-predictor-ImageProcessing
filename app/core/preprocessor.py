"""Deterministic preprocessing primitives for inspection experiments."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessedImage:
    """Common image representations retained for fair edge-method comparison."""

    original_bgr: np.ndarray
    grayscale: np.ndarray
    red: np.ndarray
    blue: np.ndarray
    hsv_value: np.ndarray


def prepare_channels(image_bgr: np.ndarray) -> PreprocessedImage:
    """Derive grayscale, BGR channels and HSV value without assuming a winner."""

    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected a non-empty BGR image with three channels")
    if image_bgr.size == 0:
        raise ValueError("Image must not be empty")
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return PreprocessedImage(
        original_bgr=image_bgr,
        grayscale=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY),
        red=image_bgr[:, :, 2],
        blue=image_bgr[:, :, 0],
        hsv_value=hsv[:, :, 2],
    )
