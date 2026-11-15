#!/usr/bin/env python3
"""Automatically inspect every stepped job in the red-backlit BMP fixture.

Without ``--measurement-config`` the pipeline remains pixel-only.  A supplied
configuration can explicitly add a validated linear scale and D1-D4 drawing
tolerances; the image code itself never supplies either value.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.alignment import HorizontalFlangeLeftJobNormalizer
from app.core.calibration import LinearScaleCalibration
from app.core.image_loader import load_bgr_image
from app.core.inspector import ConfiguredImageInspectionResult, DiameterInspectionEngine
from app.core.job_detector import DarkRedSilhouetteJobDetector
from app.core.measurement_spec import DiameterSpec, FourDiameterSpec, MeasurementRegion


# Fractions apply after a counter-clockwise rotation.  They select flat
# silhouette sections, away from the four physical shoulder transitions.
BMP_FOUR_DIAMETER_SPEC = FourDiameterSpec((
    DiameterSpec("D1", MeasurementRegion(0.84, 0.92)),  # largest, source-left flange -> bottom
    DiameterSpec("D2", MeasurementRegion(0.67, 0.76)),
    DiameterSpec("D3", MeasurementRegion(0.35, 0.58)),
    DiameterSpec("D4", MeasurementRegion(0.20, 0.27)),  # source-right top section
), job_type="horizontal_flange_left_stepped_component")

_COLORS = {"D1": (0, 255, 255), "D2": (255, 255, 0), "D3": (0, 255, 0), "D4": (255, 0, 255)}
_CSV_FIELDS = (
    "image", "job_id", "row", "column", "roi_x", "roi_y", "roi_width", "roi_height",
    "job_status", "diameter", "left_edge_x", "right_edge_x", "diameter_pixels", "diameter_mm",
    "nominal_mm", "tolerance_mm", "lower_limit_mm", "upper_limit_mm", "status", "edge_strategy",
    "left_confidence", "right_confidence", "message",
)


def load_measurement_configuration(path: Path) -> tuple[FourDiameterSpec, LinearScaleCalibration | None]:
    """Merge optional calibration/tolerances with the fixed Phase 3 BMP bands.

    The config intentionally cannot redefine localization, orientation, or
    regions.  Its supported JSON shape is:

    ``{"calibration": {"mm_per_pixel": 0.1}, "diameters": [
    {"name": "D1", "nominal_mm": 10.0, "tolerance_mm": 0.1}]}``

    ``calibration`` and ``diameters`` are independently optional.  A tolerance
    without calibration therefore remains explicitly NOT_CONFIGURED.
    """

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read measurement configuration: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Measurement configuration is not valid JSON: {path}") from exc
    if not isinstance(data, Mapping):
        raise ValueError("measurement configuration must be a JSON object")

    calibration_data = data.get("calibration")
    calibration: LinearScaleCalibration | None = None
    if calibration_data is not None:
        if not isinstance(calibration_data, Mapping) or "mm_per_pixel" not in calibration_data:
            raise ValueError("calibration requires a mm_per_pixel value")
        reference_description = calibration_data.get("reference_description", "")
        if not isinstance(reference_description, str):
            raise ValueError("calibration reference_description must be a string")
        calibration = LinearScaleCalibration(
            mm_per_pixel=calibration_data["mm_per_pixel"],
            reference_description=reference_description,
        )

    entries = data.get("diameters", [])
    if not isinstance(entries, list):
        raise ValueError("diameters must be a list when configured")
    configured: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("name"), str):
            raise ValueError("each configured diameter requires a name")
        name = entry["name"]
        if name not in {"D1", "D2", "D3", "D4"}:
            raise ValueError("configured diameter name must be D1, D2, D3, or D4")
        if name in configured:
            raise ValueError(f"configured diameter {name} appears more than once")
        configured[name] = entry

    specifications: list[DiameterSpec] = []
    for base in BMP_FOUR_DIAMETER_SPEC.ordered():
        entry = configured.get(base.name)
        nominal_mm = None if entry is None else entry.get("nominal_mm")
        tolerance_mm = None if entry is None else entry.get("tolerance_mm")
        specifications.append(DiameterSpec(
            name=base.name,
            region=base.region,
            nominal_mm=nominal_mm,
            tolerance_mm=tolerance_mm,
            edge_strategy=base.edge_strategy,
        ))
    return FourDiameterSpec(tuple(specifications), job_type=BMP_FOUR_DIAMETER_SPEC.job_type), calibration


def _diameter_dict(result: Any) -> dict[str, Any]:
    return {
        "name": result.name,
        "region": {"top_fraction": result.region.top_fraction, "bottom_fraction": result.region.bottom_fraction},
        "left_edge_x": result.left_edge_x,
        "right_edge_x": result.right_edge_x,
        "diameter_pixels": result.diameter_pixels,
        "diameter_mm": result.diameter_mm,
        "nominal_mm": result.nominal_mm,
        "tolerance_mm": result.tolerance_mm,
        "lower_limit_mm": result.lower_limit_mm,
        "upper_limit_mm": result.upper_limit_mm,
        "status": result.status.value,
        "edge_strategy": result.edge_strategy,
        "edge_confidence": result.edge_confidence,
        "message": result.message,
    }


def _status_counts(inspection: ConfiguredImageInspectionResult) -> dict[str, int]:
    counts = {status: 0 for status in ("PASS", "FAIL", "NOT_CONFIGURED", "MEASUREMENT_FAILED")}
    for job in inspection.jobs:
        counts[job.result.status.value] += 1
    return counts


def inspection_report(
    image_path: Path,
    inspection: ConfiguredImageInspectionResult,
    calibration: LinearScaleCalibration | None = None,
) -> dict[str, Any]:
    """Build a JSON-compatible, per-image inspection result."""

    return {
        "image": image_path.name,
        "orientation": "source horizontal, flange-left; normalized 90 degrees counter-clockwise",
        "measurement_units": "pixels_and_mm" if calibration is not None else "pixels",
        "calibration": None if calibration is None else {
            "type": "linear_scale",
            "mm_per_pixel": calibration.mm_per_pixel,
            "reference_description": calibration.reference_description,
        },
        "timings_ms": inspection.timings_ms,
        "job_count": len(inspection.jobs),
        "status_counts": _status_counts(inspection),
        "jobs": [
            {
                "job_id": job.job_id,
                "roi": {"x": job.roi.x, "y": job.roi.y, "width": job.roi.width, "height": job.roi.height},
                "status": job.result.status.value,
                "diameters": [_diameter_dict(result) for result in job.result.diameters],
            }
            for job in inspection.jobs
        ],
        "notes": [
            "D1-D4 are independent outer-silhouette pixel measurements.",
            "Absent calibration or drawing tolerances leave the corresponding status NOT_CONFIGURED.",
        ],
    }


def _job_grid_coordinates(job_id: str) -> tuple[int, int]:
    """Extract row/column values from the detector's deterministic identifier."""

    _, row, column = job_id.split("_")
    return int(row[1:]), int(column[1:])


def csv_rows(image_path: Path, inspection: ConfiguredImageInspectionResult) -> Iterable[dict[str, Any]]:
    for job in inspection.jobs:
        row, column = _job_grid_coordinates(job.job_id)
        for result in job.result.diameters:
            confidence = result.edge_confidence or {}
            yield {
                "image": image_path.name,
                "job_id": job.job_id,
                "row": row,
                "column": column,
                "roi_x": job.roi.x,
                "roi_y": job.roi.y,
                "roi_width": job.roi.width,
                "roi_height": job.roi.height,
                "job_status": job.result.status.value,
                "diameter": result.name,
                "left_edge_x": result.left_edge_x,
                "right_edge_x": result.right_edge_x,
                "diameter_pixels": result.diameter_pixels,
                "diameter_mm": result.diameter_mm,
                "nominal_mm": result.nominal_mm,
                "tolerance_mm": result.tolerance_mm,
                "lower_limit_mm": result.lower_limit_mm,
                "upper_limit_mm": result.upper_limit_mm,
                "status": result.status.value,
                "edge_strategy": result.edge_strategy,
                "left_confidence": confidence.get("left"),
                "right_confidence": confidence.get("right"),
                "message": result.message,
            }


def annotate(image: Any, inspection: ConfiguredImageInspectionResult) -> Any:
    """Draw source ROIs and rotate each upright measurement line back to source."""

    annotated = image.copy()
    for job in inspection.jobs:
        roi = job.roi
        cv2.rectangle(annotated, (roi.x, roi.y), (roi.x + roi.width - 1, roi.y + roi.height - 1), (255, 255, 255), 2)
        cv2.putText(annotated, job.job_id, (roi.x + 5, roi.y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        # For ROTATE_90_COUNTERCLOCKWISE: rotated x == source y and
        # rotated y == source_width - 1 - source x.
        normalized_height = roi.width
        for result in job.result.diameters:
            if result.left_edge_x is None or result.right_edge_x is None or result.diameter_pixels is None:
                continue
            start, end = result.region.pixel_rows(normalized_height)
            source_x = roi.x + roi.width - 1 - ((start + end) // 2)
            source_top = roi.y + int(round(result.left_edge_x))
            source_bottom = roi.y + int(round(result.right_edge_x))
            color = _COLORS[result.name]
            cv2.line(annotated, (source_x, source_top), (source_x, source_bottom), color, 2)
            cv2.circle(annotated, (source_x, source_top), 3, color, -1)
            cv2.circle(annotated, (source_x, source_bottom), 3, color, -1)
            cv2.putText(
                annotated, f"{result.name} {result.diameter_pixels:.1f}px", (source_x + 5, max(16, source_top - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )
    return annotated


def inspect_image(
    image_path: Path,
    engine: DiameterInspectionEngine | None = None,
    specification: FourDiameterSpec = BMP_FOUR_DIAMETER_SPEC,
    calibration: LinearScaleCalibration | None = None,
) -> ConfiguredImageInspectionResult:
    engine = engine or DiameterInspectionEngine(job_normalizer=HorizontalFlangeLeftJobNormalizer())
    return engine.inspect_configured_image(image_path, DarkRedSilhouetteJobDetector(), specification, calibration)


def write_image_outputs(
    image_path: Path,
    output_dir: Path,
    inspection: ConfiguredImageInspectionResult,
    calibration: LinearScaleCalibration | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image = load_bgr_image(image_path)
    annotation_path = output_dir / f"{image_path.stem}_annotated.png"
    json_path = output_dir / f"{image_path.stem}_results.json"
    csv_path = output_dir / f"{image_path.stem}_results.csv"
    if not cv2.imwrite(str(annotation_path), annotate(image, inspection)):
        raise OSError(f"Could not write {annotation_path}")
    json_path.write_text(json.dumps(inspection_report(image_path, inspection, calibration), indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows(image_path, inspection))
    return {"annotation": annotation_path, "json": json_path, "csv": csv_path}


def benchmark(
    images: Iterable[Path],
    repetitions: int,
    specification: FourDiameterSpec = BMP_FOUR_DIAMETER_SPEC,
    calibration: LinearScaleCalibration | None = None,
) -> dict[str, Any]:
    """Measure complete full-image inspections, including image loading."""

    engine = DiameterInspectionEngine(job_normalizer=HorizontalFlangeLeftJobNormalizer())
    results: list[dict[str, Any]] = []
    for image_path in images:
        timings: list[float] = []
        job_count = 0
        for _ in range(repetitions):
            start = perf_counter()
            inspection = inspect_image(image_path, engine, specification, calibration)
            timings.append((perf_counter() - start) * 1000.0)
            job_count = len(inspection.jobs)
        sorted_timings = sorted(timings)
        results.append({
            "image": image_path.name,
            "jobs": job_count,
            "runs": repetitions,
            "median_full_image_pipeline_ms": statistics.median(timings),
            "max_full_image_pipeline_ms": max(timings),
            "min_full_image_pipeline_ms": min(timings),
            "runs_ms": sorted_timings,
        })
    return {
        "benchmark": "automatic localization + ROI extraction + horizontal normalization + D1-D4 measurement",
        "results": results,
        "note": "These BMPs contain nine jobs each; this benchmark does not establish the approximately 50-job/one-second production target.",
    }


def input_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    images = sorted(input_path.glob("*.bmp"))
    if not images:
        raise ValueError(f"No BMP files found in {input_path}")
    return images


def run(
    input_path: Path,
    output_dir: Path,
    benchmark_runs: int = 5,
    measurement_config: Path | None = None,
) -> dict[str, Any]:
    images = input_images(input_path)
    specification, calibration = (
        load_measurement_configuration(measurement_config)
        if measurement_config is not None else (BMP_FOUR_DIAMETER_SPEC, None)
    )
    engine = DiameterInspectionEngine(job_normalizer=HorizontalFlangeLeftJobNormalizer())
    output_paths: dict[str, dict[str, str]] = {}
    counts: dict[str, int] = {}
    for image_path in images:
        inspection = inspect_image(image_path, engine, specification, calibration)
        paths = write_image_outputs(image_path, output_dir, inspection, calibration)
        output_paths[image_path.name] = {name: str(path) for name, path in paths.items()}
        counts[image_path.name] = len(inspection.jobs)
    benchmark_report = benchmark(images, benchmark_runs, specification, calibration)
    benchmark_path = output_dir / "benchmark.json"
    benchmark_path.write_text(json.dumps(benchmark_report, indent=2), encoding="utf-8")
    return {
        "job_counts": counts,
        "outputs": output_paths,
        "benchmark": benchmark_report,
        "benchmark_path": str(benchmark_path),
        "measurement_config": None if measurement_config is None else str(measurement_config),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automatically inspect horizontal stepped jobs in BMP fixture images.")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "input", help="One BMP or a directory of BMPs")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "output" / "bmp_inspection")
    parser.add_argument(
        "--measurement-config", type=Path,
        help="Optional JSON with explicit linear calibration and D1-D4 nominal/tolerance values",
    )
    parser.add_argument("--benchmark-runs", type=int, default=5, help="Complete pipeline repetitions per input image")
    arguments = parser.parse_args()
    if arguments.benchmark_runs < 1:
        parser.error("--benchmark-runs must be at least one")
    return arguments


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(run(arguments.input, arguments.output, arguments.benchmark_runs, arguments.measurement_config), indent=2))
