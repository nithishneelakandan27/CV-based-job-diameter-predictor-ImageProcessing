"""Classical edge-detection experiments for reflective machined components.

No channel is assumed to be best. Each method returns both the binary edge image
and simple density/contour metrics so images can be compared under a controlled
lighting setup before a production choice is made.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import cv2
import numpy as np

from .preprocessor import PreprocessedImage


@dataclass(frozen=True)
class EdgeDetectionConfig:
    name: str
    source: str
    method: str
    blur: str = "gaussian"
    canny_low: int = 50
    canny_high: int = 150


@dataclass(frozen=True)
class EdgeDetectionResult:
    config: EdgeDetectionConfig
    edge_image: np.ndarray
    contour_count: int
    edge_pixel_count: int
    edge_coverage_percent: float

    def metrics(self) -> dict[str, int | float]:
        return {
            "edge_pixel_count": self.edge_pixel_count,
            "edge_coverage_percent": self.edge_coverage_percent,
            "contour_count": self.contour_count,
        }


DEFAULT_EXPERIMENTS: tuple[EdgeDetectionConfig, ...] = (
    EdgeDetectionConfig("grayscale_gaussian_canny", "grayscale", "canny", "gaussian"),
    EdgeDetectionConfig("grayscale_median_canny", "grayscale", "canny", "median"),
    EdgeDetectionConfig("red_gaussian_canny", "red", "canny", "gaussian"),
    EdgeDetectionConfig("blue_gaussian_canny", "blue", "canny", "gaussian"),
    EdgeDetectionConfig("hsv_value_gaussian_canny", "hsv_value", "canny", "gaussian"),
    EdgeDetectionConfig("grayscale_sobel", "grayscale", "sobel", "gaussian"),
    EdgeDetectionConfig("grayscale_otsu", "grayscale", "otsu", "gaussian"),
)


class EdgeDetector:
    """Run comparable Canny, Sobel and threshold experiments."""

    def detect(self, channels: PreprocessedImage, config: EdgeDetectionConfig) -> EdgeDetectionResult:
        source = self._source_image(channels, config.source)
        filtered = self._filter(source, config.blur)

        if config.method == "canny":
            edges = cv2.Canny(filtered, config.canny_low, config.canny_high, L2gradient=True)
        elif config.method == "sobel":
            horizontal = cv2.Sobel(filtered, cv2.CV_16S, 1, 0, ksize=3)
            vertical = cv2.Sobel(filtered, cv2.CV_16S, 0, 1, ksize=3)
            edges = cv2.convertScaleAbs(cv2.addWeighted(horizontal, 0.5, vertical, 0.5, 0))
            _, edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif config.method == "otsu":
            _, edges = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            raise ValueError(f"Unsupported edge method: {config.method}")

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        edge_pixel_count = int(np.count_nonzero(edges))
        return EdgeDetectionResult(
            config=config,
            edge_image=edges,
            contour_count=len(contours),
            edge_pixel_count=edge_pixel_count,
            edge_coverage_percent=(edge_pixel_count / edges.size) * 100.0,
        )

    def run_experiments(
        self,
        channels: PreprocessedImage,
        configs: tuple[EdgeDetectionConfig, ...] = DEFAULT_EXPERIMENTS,
    ) -> Mapping[str, EdgeDetectionResult]:
        return {config.name: self.detect(channels, config) for config in configs}

    @staticmethod
    def contours(edge_image: np.ndarray, minimum_area_px: float = 100.0) -> list[np.ndarray]:
        """Return visible candidate contours, filtering small edge fragments only.

        These contours are visual aids, not validated part locations or diameter
        regions. Engineering drawing data is required for that decision.
        """

        if minimum_area_px < 0:
            raise ValueError("minimum_area_px must be non-negative")
        contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [contour for contour in contours if cv2.contourArea(contour) >= minimum_area_px]

    @staticmethod
    def _source_image(channels: PreprocessedImage, name: str) -> np.ndarray:
        try:
            return getattr(channels, name)
        except AttributeError as exc:
            raise ValueError(f"Unknown channel source: {name}") from exc

    @staticmethod
    def _filter(image: np.ndarray, blur: str) -> np.ndarray:
        if blur == "gaussian":
            return cv2.GaussianBlur(image, (5, 5), 0)
        if blur == "median":
            return cv2.medianBlur(image, 5)
        if blur == "none":
            return image
        raise ValueError(f"Unsupported blur method: {blur}")
