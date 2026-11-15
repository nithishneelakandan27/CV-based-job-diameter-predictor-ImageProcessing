#!/usr/bin/env python3
"""Run manual/fixture-configured four-diameter pixel experiments.

This script requires a user-supplied JSON config because the phone images do not
define production job ROIs or D1-D4 bands. It can draw pixel measurements but
never invents a millimetre value, tolerance, or PASS/FAIL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.inspector import DiameterInspectionEngine
from app.core.job_detector import FixedRoiJobDetector, JobROI
from app.core.measurement_spec import FourDiameterSpec


def load_config(path: Path) -> tuple[FixedRoiJobDetector, FourDiameterSpec]:
    """Read manual job ROIs and normalized D1-D4 bands from a JSON file."""

    data = json.loads(path.read_text(encoding="utf-8"))
    rois = tuple(
        JobROI(
            job_id=entry["job_id"], x=entry["x"], y=entry["y"],
            width=entry["width"], height=entry["height"],
        )
        for entry in data.get("job_rois", [])
    )
    return FixedRoiJobDetector(rois), FourDiameterSpec.from_dict(data)


def _serialize_result(result) -> dict:
    return {
        "name": result.name,
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
        "message": result.message,
    }


def annotate(image, inspection) -> object:
    """Overlay configurable D1-D4 pixel measurements; no fake mm labels."""

    annotated = image.copy()
    colors = {"D1": (0, 255, 255), "D2": (255, 255, 0), "D3": (0, 255, 0), "D4": (255, 0, 255)}
    for job in inspection.jobs:
        roi = job.roi
        cv2.rectangle(annotated, (roi.x, roi.y), (roi.x + roi.width, roi.y + roi.height), (255, 255, 255), 1)
        for result in job.result.diameters:
            if result.diameter_pixels is None or result.left_edge_x is None or result.right_edge_x is None:
                continue
            top, bottom = result.region.pixel_rows(roi.height)
            y = roi.y + (top + bottom) // 2
            left, right = int(round(roi.x + result.left_edge_x)), int(round(roi.x + result.right_edge_x))
            color = colors[result.name]
            cv2.line(annotated, (left, y), (right, y), color, 2)
            cv2.circle(annotated, (left, y), 3, color, -1)
            cv2.circle(annotated, (right, y), 3, color, -1)
            cv2.putText(
                annotated, f"{result.name}: {result.diameter_pixels:.2f} px", (left, max(15, y - 7)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )
    return annotated


def run(image_path: Path, config_path: Path, output_dir: Path) -> None:
    detector, specification = load_config(config_path)
    engine = DiameterInspectionEngine()
    inspection = engine.inspect_configured_image(image_path, detector, specification)
    from app.core.image_loader import load_bgr_image

    output_dir.mkdir(parents=True, exist_ok=True)
    image = load_bgr_image(image_path)
    visualization_path = output_dir / f"{image_path.stem}_four_diameter_experiment.png"
    if not cv2.imwrite(str(visualization_path), annotate(image, inspection)):
        raise OSError(f"Could not write {visualization_path}")
    report = {
        "image": image_path.name,
        "timings_ms": inspection.timings_ms,
        "jobs": [
            {"job_id": job.job_id, "diameters": [_serialize_result(item) for item in job.result.diameters]}
            for job in inspection.jobs
        ],
        "notes": [
            "This is a configuration-driven pixel experiment, not validated industrial inspection.",
            "Millimetres and PASS/FAIL remain NOT_CONFIGURED without an explicit calibration and drawing tolerance.",
        ],
    }
    report_path = output_dir / "four_diameter_experiment_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {visualization_path} and {report_path}")
    print(json.dumps(inspection.timings_ms, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a manual four-diameter pixel experiment.")
    parser.add_argument("image", type=Path, help="Input source image")
    parser.add_argument("--config", type=Path, required=True, help="JSON with job_rois and D1-D4 normalized regions")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "output" / "four_diameter_experiments")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.image, arguments.config, arguments.output)
