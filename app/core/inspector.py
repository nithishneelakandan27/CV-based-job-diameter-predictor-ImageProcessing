"""Orchestration layer for the explicit dimensional-inspection architecture.

Image processing -> edge detection -> geometric measurement -> calibration ->
tolerance check -> PASS/FAIL. The supplied phone-image experiment intentionally
stops before the final two stages because its measurement regions, calibration,
and engineering specification are unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .alignment import IdentityJobNormalizer, JobNormalizer
from .calibration import CalibrationModel
from .diameter_engine import FourDiameterMeasurementEngine, JobMeasurementResult
from .edge_detector import EdgeDetector
from .image_loader import load_bgr_image
from .job_detector import JobDetector, JobROI
from .measurement import CalibratedDiameterMeasurement, PixelDiameterMeasurement
from .measurement_spec import FourDiameterSpec
from .preprocessor import PreprocessedImage, prepare_channels
from .result import ExperimentResult, InspectionState, PipelineTimings
from .tolerance_checker import SymmetricTolerance, ToleranceResult, check_tolerance


class DiameterInspectionEngine:
    """Coordinates independent processing stages without embedding drawing knowledge."""

    def __init__(self, edge_detector: EdgeDetector | None = None, job_normalizer: JobNormalizer | None = None) -> None:
        self.edge_detector = edge_detector or EdgeDetector()
        self.four_diameter_engine = FourDiameterMeasurementEngine()
        self.job_normalizer = job_normalizer or IdentityJobNormalizer()

    def experiment(self, image_path: str | Path) -> tuple[ExperimentResult, PreprocessedImage, dict]:
        """Run edge experiments only; no dimensions or PASS/FAIL are inferred."""

        timings = PipelineTimings()
        start = perf_counter()
        image = load_bgr_image(image_path)
        timings.record("image_loading", (perf_counter() - start) * 1000.0)

        start = perf_counter()
        channels = prepare_channels(image)
        timings.record("preprocessing", (perf_counter() - start) * 1000.0)

        start = perf_counter()
        edge_results = dict(self.edge_detector.run_experiments(channels))
        timings.record("edge_detection", (perf_counter() - start) * 1000.0)
        # No valid region exists yet, but make the absent stage observable.
        timings.record("measurement", 0.0)

        result = ExperimentResult(
            source=str(image_path),
            state=InspectionState.EXPERIMENT,
            edge_metrics={name: outcome.metrics() for name, outcome in edge_results.items()},
            timings_ms=timings.as_dict(),
            notes=(
                "Exploratory pixel/edge experiment only; no diameter region was specified.",
                "No camera calibration, physical reference, nominal dimension, or tolerance was supplied.",
                "Therefore no millimetre measurement or PASS/FAIL result has been generated.",
            ),
        )
        return result, channels, edge_results

    def inspect_configured_image(
        self,
        image_path: str | Path,
        job_detector: JobDetector,
        specification: FourDiameterSpec,
        calibration: CalibrationModel | None = None,
    ) -> "ConfiguredImageInspectionResult":
        """Inspect fixture/configured jobs and time localization, ROI, and D1-D4.

        This method deliberately needs an explicit detector/configuration. It does
        not infer the nine phone-image locations or pretend to detect 50 jobs.
        """

        overall_start = perf_counter()
        timings = PipelineTimings()
        start = perf_counter()
        image = load_bgr_image(image_path)
        timings.record("image_loading", (perf_counter() - start) * 1000.0)

        start = perf_counter()
        candidates = job_detector.locate(image)
        timings.record("job_localization", (perf_counter() - start) * 1000.0)

        jobs: list[ConfiguredJobInspectionResult] = []
        roi_total = 0.0
        normalization_total = 0.0
        diameter_totals = {f"{name}_measurement": 0.0 for name in ("D1", "D2", "D3", "D4")}
        for candidate in candidates:
            start = perf_counter()
            job_image = candidate.extract(image)
            roi_total += (perf_counter() - start) * 1000.0
            start = perf_counter()
            normalized_job_image = self.job_normalizer.normalize(job_image)
            normalization_total += (perf_counter() - start) * 1000.0
            job_result = self.four_diameter_engine.measure_job(normalized_job_image, specification, calibration)
            jobs.append(ConfiguredJobInspectionResult(candidate.job_id, candidate, job_result))
            for stage, duration in job_result.timings_ms.items():
                diameter_totals[stage] += duration
        timings.record("roi_extraction", roi_total)
        timings.record("normalization", normalization_total)
        for stage, duration in diameter_totals.items():
            timings.record(stage, duration)
        complete_inspection = (perf_counter() - overall_start) * 1000.0
        return ConfiguredImageInspectionResult(
            tuple(jobs), {**timings.stages, "complete_image_inspection": complete_inspection}
        )

    @staticmethod
    def evaluate_diameter(
        measurement: PixelDiameterMeasurement,
        calibration: CalibrationModel,
        tolerance: SymmetricTolerance,
    ) -> tuple[CalibratedDiameterMeasurement, ToleranceResult]:
        """Complete downstream stages only when explicit calibration/specs exist."""

        calibrated = measurement.calibrated_mm(calibration)
        return calibrated, check_tolerance(calibrated.diameter_mm, tolerance)


@dataclass(frozen=True)
class ConfiguredJobInspectionResult:
    job_id: str
    roi: JobROI
    result: JobMeasurementResult


@dataclass(frozen=True)
class ConfiguredImageInspectionResult:
    jobs: tuple[ConfiguredJobInspectionResult, ...]
    timings_ms: dict[str, float]
