"""Four-diameter pixel measurement using replaceable classical edge strategies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from time import perf_counter

import cv2
import numpy as np

from .calibration import CalibrationModel
from .edge_detector import EdgeDetectionConfig, EdgeDetector
from .measurement import PixelDiameterMeasurement
from .measurement_spec import DiameterSpec, FourDiameterSpec, MeasurementRegion
from .preprocessor import prepare_channels
from .tolerance_checker import SymmetricTolerance, check_tolerance


class DiameterStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    MEASUREMENT_FAILED = "MEASUREMENT_FAILED"


@dataclass(frozen=True)
class EdgeLocalization:
    """Localized left/right section boundaries and their one-dimensional scores."""

    left_edge_x: float
    right_edge_x: float
    left_strength: float
    right_strength: float
    strategy: str
    left_confidence: float = 1.0
    right_confidence: float = 1.0


@dataclass(frozen=True)
class DiameterResult:
    """Complete result for one location; absent values are intentionally explicit."""

    name: str
    region: MeasurementRegion
    left_edge_x: float | None
    right_edge_x: float | None
    diameter_pixels: float | None
    diameter_mm: float | None
    nominal_mm: float | None
    tolerance_mm: float | None
    lower_limit_mm: float | None
    upper_limit_mm: float | None
    status: DiameterStatus
    edge_strategy: str
    edge_confidence: dict[str, float] | None = None
    message: str = ""


@dataclass(frozen=True)
class JobMeasurementResult:
    """Independent D1/D2/D3/D4 results and per-diameter elapsed timings."""

    diameters: tuple[DiameterResult, ...]
    timings_ms: dict[str, float]
    status: DiameterStatus


class BoundaryEdgeLocator:
    """Locate opposing side boundaries in a configured horizontal band.

    ``sobel_x`` emphasizes vertical boundaries and is the default for a side-view
    cylinder. ``canny`` is retained as a switchable experimental strategy. Both
    assume the job ROI is aligned upright and closely cropped enough that the
    intended two outer boundaries fall on opposite image halves.
    """

    def __init__(self, edge_detector: EdgeDetector | None = None) -> None:
        self.edge_detector = edge_detector or EdgeDetector()

    def locate(self, job_image: np.ndarray, region: MeasurementRegion, strategy: str = "sobel_x") -> EdgeLocalization:
        if job_image is None or job_image.ndim != 3 or job_image.shape[2] != 3 or job_image.size == 0:
            raise ValueError("job_image must be a non-empty BGR image")
        row_start, row_end = region.pixel_rows(job_image.shape[0])
        channels = prepare_channels(job_image)
        source = channels.grayscale

        if strategy == "sobel_x":
            filtered = cv2.GaussianBlur(source, (5, 5), 0)
            response = np.abs(cv2.Sobel(filtered, cv2.CV_32F, 1, 0, ksize=3))
        elif strategy == "canny":
            edges = self.edge_detector.detect(
                channels, EdgeDetectionConfig("measurement_canny", "grayscale", "canny", "gaussian")
            ).edge_image
            response = edges.astype(np.float32)
        else:
            raise ValueError(f"Unsupported measurement edge strategy: {strategy}")

        # Median aggregation suppresses isolated reflection pixels while retaining
        # a persistent vertical material/background boundary through the band.
        scores = np.median(response[row_start:row_end, :], axis=0)
        scores = cv2.GaussianBlur(scores.reshape(1, -1), (0, 0), 1.5).reshape(-1)
        if scores.size < 4 or float(np.max(scores)) <= 0:
            raise ValueError("No usable edge response in measurement region")

        middle = scores.size // 2
        left_scores = scores[:middle]
        right_scores = scores[middle:]

        left_idx, left_strength, left_conf = self._find_outer_edge(left_scores, is_left=True)
        right_idx, right_strength, right_conf = self._find_outer_edge(right_scores, is_left=False)
        right_index_absolute = right_idx + middle

        if right_index_absolute <= left_idx + 1:
            raise ValueError("Localized edges do not form a valid diameter")

        return EdgeLocalization(
            left_edge_x=self._subpixel_peak(scores, int(round(left_idx))) if not isinstance(left_idx, float) else self._subpixel_peak_val(scores, left_idx),
            right_edge_x=self._subpixel_peak(scores, int(round(right_index_absolute))) if not isinstance(right_index_absolute, float) else self._subpixel_peak_val(scores, right_index_absolute),
            left_strength=left_strength,
            right_strength=right_strength,
            strategy=strategy,
            left_confidence=left_conf,
            right_confidence=right_conf,
        )

    @staticmethod
    def _find_outer_edge(side_scores: np.ndarray, is_left: bool) -> tuple[float, float, float]:
        """Find the true outer silhouette edge by scanning from the outside in, suppressing internal reflections."""
        if side_scores.size < 2:
            return 0.0, 0.0, 0.0

        max_val = float(np.max(side_scores))
        if max_val <= 0:
            return 0.0, 0.0, 0.0

        median_val = float(np.median(side_scores))
        std_val = float(np.std(side_scores))
        threshold = max(2.0, median_val + 0.5 * std_val, 0.15 * max_val)

        if is_left:
            # Scan from outside (index 0) inward
            for i in range(1, len(side_scores) - 1):
                if side_scores[i] >= side_scores[i - 1] and side_scores[i] >= side_scores[i + 1]:
                    if side_scores[i] >= threshold:
                        subpixel_idx = BoundaryEdgeLocator._subpixel_peak(side_scores, i)
                        confidence = float(side_scores[i] / max_val)
                        # A tiny exterior reflection can occur before the real
                        # silhouette.  It should not outrank the persistent
                        # material/background edge merely because it is first.
                        if confidence < 0.5:
                            best_idx = int(np.argmax(side_scores))
                            return BoundaryEdgeLocator._subpixel_peak(side_scores, best_idx), float(side_scores[best_idx]), 1.0
                        return subpixel_idx, float(side_scores[i]), confidence
            best_idx = int(np.argmax(side_scores))
            return float(best_idx), float(side_scores[best_idx]), 0.5
        else:
            # Scan from outside (end of array) inward
            for i in range(len(side_scores) - 2, 0, -1):
                if side_scores[i] >= side_scores[i - 1] and side_scores[i] >= side_scores[i + 1]:
                    if side_scores[i] >= threshold:
                        subpixel_idx = BoundaryEdgeLocator._subpixel_peak(side_scores, i)
                        confidence = float(side_scores[i] / max_val)
                        if confidence < 0.5:
                            best_idx = int(np.argmax(side_scores))
                            return BoundaryEdgeLocator._subpixel_peak(side_scores, best_idx), float(side_scores[best_idx]), 1.0
                        return subpixel_idx, float(side_scores[i]), confidence
            best_idx = int(np.argmax(side_scores))
            return float(best_idx), float(side_scores[best_idx]), 0.5

    @staticmethod
    def _subpixel_peak(scores: np.ndarray, index: int) -> float:
        """Parabolic local peak interpolation; falls back safely at endpoints."""

        if index <= 0 or index >= len(scores) - 1:
            return float(index)
        left, center, right = (float(scores[index - 1]), float(scores[index]), float(scores[index + 1]))
        denominator = left - (2.0 * center) + right
        if abs(denominator) < 1e-9:
            return float(index)
        offset = 0.5 * (left - right) / denominator
        return float(index) + max(-0.5, min(0.5, offset))

    @staticmethod
    def _subpixel_peak_val(scores: np.ndarray, float_index: float) -> float:
        """Fallback for float index."""
        idx = int(round(float_index))
        return BoundaryEdgeLocator._subpixel_peak(scores, max(0, min(len(scores) - 1, idx)))


class FourDiameterMeasurementEngine:
    """Measure configured D1-D4 bands while preserving missing-data states."""

    def __init__(self, edge_locator: BoundaryEdgeLocator | None = None) -> None:
        self.edge_locator = edge_locator or BoundaryEdgeLocator()

    def measure_diameter(
        self,
        job_image: np.ndarray,
        specification: DiameterSpec,
        calibration: CalibrationModel | None = None,
    ) -> DiameterResult:
        try:
            edges = self.edge_locator.locate(job_image, specification.region, specification.edge_strategy)
            pixel_measurement = PixelDiameterMeasurement(edges.left_edge_x, edges.right_edge_x, specification.edge_strategy, specification.name)
        except ValueError as exc:
            return DiameterResult(
                name=specification.name, region=specification.region, left_edge_x=None, right_edge_x=None,
                diameter_pixels=None, diameter_mm=None, nominal_mm=specification.nominal_mm,
                tolerance_mm=specification.tolerance_mm, lower_limit_mm=None, upper_limit_mm=None,
                status=DiameterStatus.MEASUREMENT_FAILED, edge_strategy=specification.edge_strategy, message=str(exc),
            )

        diameter_mm: float | None = None
        lower: float | None = None
        upper: float | None = None
        tolerance: SymmetricTolerance | None = None
        if specification.is_tolerance_configured:
            # Limits are specification data, so retain them even if calibration
            # is absent and therefore a PASS/FAIL decision is not permitted.
            tolerance = SymmetricTolerance(specification.nominal_mm, specification.tolerance_mm)  # type: ignore[arg-type]
            lower, upper = tolerance.lower_limit_mm, tolerance.upper_limit_mm
        status = DiameterStatus.NOT_CONFIGURED
        message = "Pixel measurement only; calibration and tolerance specification are required for PASS/FAIL."
        if calibration is not None:
            diameter_mm = pixel_measurement.calibrated_mm(calibration).diameter_mm
        if diameter_mm is not None and tolerance is not None:
            decision = check_tolerance(diameter_mm, tolerance)
            status = DiameterStatus(decision.status.value)
            message = "Calibrated measurement evaluated against configured tolerance."
        elif calibration is None:
            message = "Pixel measurement only; no calibration is configured."
        elif not specification.is_tolerance_configured:
            message = "Calibrated measurement available; no nominal/tolerance specification is configured."

        return DiameterResult(
            name=specification.name, region=specification.region,
            left_edge_x=edges.left_edge_x, right_edge_x=edges.right_edge_x,
            diameter_pixels=pixel_measurement.diameter_pixels, diameter_mm=diameter_mm,
            nominal_mm=specification.nominal_mm, tolerance_mm=specification.tolerance_mm,
            lower_limit_mm=lower, upper_limit_mm=upper, status=status,
            edge_strategy=specification.edge_strategy,
            edge_confidence={"left": edges.left_confidence, "right": edges.right_confidence},
            message=message,
        )

    def measure_job(
        self,
        job_image: np.ndarray,
        specification: FourDiameterSpec,
        calibration: CalibrationModel | None = None,
    ) -> JobMeasurementResult:
        results: list[DiameterResult] = []
        timings: dict[str, float] = {}
        for diameter_spec in specification.ordered():
            start = perf_counter()
            results.append(self.measure_diameter(job_image, diameter_spec, calibration))
            timings[f"{diameter_spec.name}_measurement"] = (perf_counter() - start) * 1000.0
        return JobMeasurementResult(tuple(results), timings, self._job_status(results))

    @staticmethod
    def _job_status(results: list[DiameterResult]) -> DiameterStatus:
        """Aggregate four required diameter outcomes without hiding their detail."""

        statuses = {result.status for result in results}
        if DiameterStatus.MEASUREMENT_FAILED in statuses:
            return DiameterStatus.MEASUREMENT_FAILED
        if DiameterStatus.NOT_CONFIGURED in statuses:
            return DiameterStatus.NOT_CONFIGURED
        if DiameterStatus.FAIL in statuses:
            return DiameterStatus.FAIL
        return DiameterStatus.PASS
