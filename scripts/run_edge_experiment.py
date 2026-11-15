#!/usr/bin/env python3
"""Create visual edge-detection comparisons for every input image.

Usage:
    python scripts/run_edge_experiment.py
    python scripts/run_edge_experiment.py --input data/input --output data/output/edge_experiments

The output is an exploratory aid. It never identifies an engineering diameter,
converts pixels to mm, or issues PASS/FAIL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.edge_detector import EdgeDetector
from app.core.inspector import DiameterInspectionEngine


def _label(image: np.ndarray, text: str) -> np.ndarray:
    labelled = image.copy()
    cv2.rectangle(labelled, (0, 0), (labelled.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(labelled, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return labelled


def _as_bgr(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def make_visualization(channels, primary_edge: np.ndarray) -> np.ndarray:
    """Produce original/channels/edge/contour panels at a consistent display size."""

    contour_overlay = channels.original_bgr.copy()
    contours = EdgeDetector.contours(primary_edge, minimum_area_px=100.0)
    cv2.drawContours(contour_overlay, contours, -1, (0, 255, 0), 2)
    # Green contours are exploratory edge fragments; they are not jobs or ROIs.
    panels = (
        _label(_as_bgr(channels.original_bgr), "Original"),
        _label(_as_bgr(channels.grayscale), "Grayscale"),
        _label(_as_bgr(channels.red), "Red channel"),
        _label(_as_bgr(channels.blue), "Blue channel"),
        _label(_as_bgr(primary_edge), "Canny: grayscale + Gaussian"),
        _label(_as_bgr(contour_overlay), "Candidate contours (not measurements)"),
    )
    width = 480
    height = max(1, int(channels.original_bgr.shape[0] * width / channels.original_bgr.shape[1]))
    rendered = [cv2.resize(panel, (width, height), interpolation=cv2.INTER_AREA) for panel in panels]
    return np.vstack((np.hstack(rendered[:3]), np.hstack(rendered[3:])))


def run(input_dir: Path, output_dir: Path) -> list[dict]:
    images = sorted(
        path for extension in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
        for path in input_dir.glob(extension)
    )
    if not images:
        raise FileNotFoundError(f"No supported images found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    engine = DiameterInspectionEngine()
    summaries: list[dict] = []
    for image_path in images:
        result, channels, edge_results = engine.experiment(image_path)
        primary = edge_results["grayscale_gaussian_canny"].edge_image
        visualization_path = output_dir / f"{image_path.stem}_edge_experiment.png"
        if not cv2.imwrite(str(visualization_path), make_visualization(channels, primary)):
            raise OSError(f"Could not write {visualization_path}")

        summary = {
            "image": image_path.name,
            "state": result.state.value,
            "visualization": visualization_path.name,
            "edge_metrics": result.edge_metrics,
            "timings_ms": result.timings_ms,
            "notes": list(result.notes),
        }
        summaries.append(summary)
        print(
            f"{image_path.name}: {result.timings_ms['complete_pipeline']:.2f} ms; "
            f"Canny edge coverage {result.edge_metrics['grayscale_gaussian_canny']['edge_coverage_percent']:.2f}%"
        )

    report_path = output_dir / "experiment_report.json"
    report_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Saved {len(summaries)} visualization(s) and {report_path}")
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exploratory classical edge-detection experiments.")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "input")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "output" / "edge_experiments")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.output)
