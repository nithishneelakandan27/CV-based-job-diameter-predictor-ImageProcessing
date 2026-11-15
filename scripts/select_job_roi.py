#!/usr/bin/env python3
"""Interactive or configuration-driven utility to select one job ROI and define D1-D4 measurement bands.

This script does not perform automated detection, invent dimensions/tolerances,
convert pixels to mm, or issue PASS/FAIL. It establishes sensible measurement
regions on real phone images prior to production fixture calibration.

Usage:
    python scripts/select_job_roi.py
    python scripts/select_job_roi.py --image data/input/WhatsApp Image 2026-09-07 at 10.40.46 AM.jpeg --interactive
    python scripts/select_job_roi.py --x 150 --y 200 --width 180 --height 320
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

from app.core.image_loader import load_bgr_image


def select_roi_interactive(image: np.ndarray) -> tuple[int, int, int, int]:
    """Allow user to select one ROI interactively using OpenCV mouse callbacks."""
    clone = image.copy()
    drawing = False
    ix, iy, ex, ey = -1, -1, -1, -1

    def draw_rectangle(event, x, y, flags, param):
        nonlocal ix, iy, ex, ey, drawing, clone
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            ex, ey = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                ex, ey = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            ex, ey = x, y

    window_name = "Select Job ROI (Drag & Release Mouse, Press ENTER/SPACE to confirm, ESC to cancel)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, draw_rectangle)

    print("Instructions:")
    print("1. Click and drag mouse over ONE cylindrical job to define its ROI.")
    print("2. Press ENTER or SPACE to confirm selection.")
    print("3. Press ESC or 'q' to abort and use default coordinates.")

    while True:
        display = clone.copy()
        if ix != -1 and iy != -1:
            cv2.rectangle(display, (ix, iy), (ex, ey), (0, 255, 0), 2)
            cv2.putText(
                display,
                f"ROI: x={min(ix, ex)}, y={min(iy, ey)}, w={abs(ex-ix)}, h={abs(ey-iy)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (13, 32):  # ENTER or SPACE
            break
        if key in (27, ord('q')):  # ESC or 'q'
            cv2.destroyAllWindows()
            raise RuntimeError("Interactive ROI selection cancelled by user.")

    cv2.destroyAllWindows()

    x = min(ix, ex)
    y = min(iy, ey)
    w = abs(ex - ix)
    h = abs(ey - iy)
    if w < 5 or h < 5:
        raise ValueError("Selected ROI is too small or invalid.")
    return x, y, w, h


def annotate_preview(
    image: np.ndarray,
    roi_x: int,
    roi_y: int,
    roi_width: int,
    roi_height: int,
    bands: list[dict],
) -> np.ndarray:
    """Draw job ROI and D1-D4 normalized measurement bands on the image."""
    annotated = image.copy()
    # Draw job ROI box
    cv2.rectangle(
        annotated,
        (roi_x, roi_y),
        (roi_x + roi_width, roi_y + roi_height),
        (255, 255, 255),
        2,
    )
    cv2.putText(
        annotated,
        "Job ROI",
        (roi_x, max(20, roi_y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    colors = {
        "D1": (0, 255, 255),   # Yellow
        "D2": (255, 255, 0),   # Cyan
        "D3": (0, 255, 0),     # Green
        "D4": (255, 0, 255),   # Magenta
    }

    for band in bands:
        name = band["name"]
        region = band["region"]
        top_frac = region["top_fraction"]
        bot_frac = region["bottom_fraction"]

        y_top = roi_y + int(round(top_frac * roi_height))
        y_bot = roi_y + int(round(bot_frac * roi_height))
        color = colors.get(name, (0, 0, 255))

        # Draw horizontal measurement band region across ROI width
        cv2.rectangle(
            annotated,
            (roi_x, y_top),
            (roi_x + roi_width, y_bot),
            color,
            1,
        )
        # Draw center line for the band
        y_mid = (y_top + y_bot) // 2
        cv2.line(annotated, (roi_x, y_mid), (roi_x + roi_width, y_mid), color, 2)
        cv2.putText(
            annotated,
            f"{name} ({top_frac:.2f}-{bot_frac:.2f})",
            (roi_x + roi_width + 5, y_mid + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return annotated


def run(
    image_path: Path,
    config_output: Path,
    preview_output: Path,
    interactive: bool,
    manual_roi: tuple[int, int, int, int] | None,
) -> None:
    image = load_bgr_image(image_path)
    height, width = image.shape[:2]
    print(f"Loaded image: {image_path.name} ({width} x {height} pixels)")

    # Default fallback ROI if non-interactive and no manual ROI provided
    # (Sensible estimated region on phone image for demonstration/testing)
    default_roi = (150, 180, 160, 320)

    if interactive:
        try:
            roi_x, roi_y, roi_w, roi_h = select_roi_interactive(image)
        except Exception as exc:
            print(f"Interactive selection failed or was cancelled: {exc}")
            print(f"Falling back to default ROI: {default_roi}")
            roi_x, roi_y, roi_w, roi_h = default_roi
    elif manual_roi is not None:
        roi_x, roi_y, roi_w, roi_h = manual_roi
    else:
        roi_x, roi_y, roi_w, roi_h = default_roi

    # Define standard four vertical normalized measurement bands (regions)
    # Note: These are calibration fractions for the job ROI; no tolerances/nominals are invented.
    diameters_config = [
        {"name": "D4", "region": {"top_fraction": 0.00, "bottom_fraction": 0.25}},
        {"name": "D3", "region": {"top_fraction": 0.25, "bottom_fraction": 0.45}},
        {"name": "D2", "region": {"top_fraction": 0.45, "bottom_fraction": 0.52}},
        {"name": "D1", "region": {"top_fraction": 0.70, "bottom_fraction": 1.00}},
    ]

    config_data = {
        "job_type": "stepped_cylindrical_component",
        "source_image": image_path.name,
        "image_resolution": {"width": width, "height": height},
        "job_rois": [
            {
                "job_id": "job-01",
                "x": roi_x,
                "y": roi_y,
                "width": roi_w,
                "height": roi_h,
            }
        ],
        "diameters": diameters_config,
        "notes": [
            "Manual ROI and measurement band calibration for real phone image.",
            "No nominal dimensions, tolerances, calibration scale, or PASS/FAIL are defined.",
        ],
    }

    config_output.parent.mkdir(parents=True, exist_ok=True)
    config_output.write_text(json.dumps(config_data, indent=2), encoding="utf-8")
    print(f"Saved job ROI and band configuration to {config_output}")

    preview = annotate_preview(image, roi_x, roi_y, roi_w, roi_h, diameters_config)
    preview_output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(preview_output), preview):
        raise OSError(f"Could not save preview image to {preview_output}")
    print(f"Saved annotated preview visualization to {preview_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select ONE job ROI and configure D1-D4 measurement bands."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=PROJECT_ROOT / "data" / "input" / "WhatsApp Image 2026-09-07 at 10.40.46 AM.jpeg",
        help="Input phone image path",
    )
    parser.add_argument(
        "--config-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "output" / "manual_roi_config.json",
        help="Path to save output JSON configuration",
    )
    parser.add_argument(
        "--preview-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "output" / "manual_roi_preview.png",
        help="Path to save annotated preview image",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive mouse selection of job ROI using OpenCV GUI window",
    )
    parser.add_argument("--x", type=int, help="Manual ROI x coordinate")
    parser.add_argument("--y", type=int, help="Manual ROI y coordinate")
    parser.add_argument("--width", type=int, help="Manual ROI width")
    parser.add_argument("--height", type=int, help="Manual ROI height")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    manual_roi = None
    if all(v is not None for v in (args.x, args.y, args.width, args.height)):
        manual_roi = (args.x, args.y, args.width, args.height)
    run(args.image, args.config_output, args.preview_output, args.interactive, manual_roi)
