# Project Overview: Computer Vision-Based Automated Job Diameter Measurement

## 1. Executive Summary

This project is an industrial computer-vision system developed in Python using classical image processing techniques (OpenCV and NumPy). Its primary objective is to automate the dimensional inspection of cylindrical machined components (referred to in manufacturing as "jobs") held in a multi-part fixture.

In modern manufacturing lines (turning, grinding, CNC machining), cylindrical components such as shafts, pins, bushings, and axles must adhere to strict dimensional tolerances. Traditional dimensional quality control relies on manual tools such as vernier calipers, external micrometers, height gauges, or slow offline Coordinate Measuring Machines (CMMs). Manual inspection is labor-intensive, subject to operator fatigue and bias, and creates bottlenecks in high-volume production.

This system demonstrates that a single high-resolution image of a 9-component fixture (3×3 grid) can be captured, segmented into individual component Regions of Interest (ROIs), normalized for orientation, and measured across four distinct stepped diameters ($D_1, D_2, D_3, D_4$) in under **90 milliseconds per frame** ($\approx 9.6\text{ ms per job}$) on standard CPU hardware.

---

## 2. Industrial Context and Problem Statement

### 2.1 The Workpiece: Stepped Cylindrical Component
The components inspected in this project are multi-diameter stepped metallic cylinders. Looking at the profile of a stepped shaft:

```text
               ┌──────────┐ ── D4 (Smallest shaft end)
               │          │
         ┌─────┴──────────┴─────┐ ── D3 (Main bearing journal)
         │                      │
         │                      │
      ┌──┴──────────────────────┴──┐ ── D2 (Intermediate step)
      │                            │
   ┌──┴────────────────────────────┴──┐ ── D1 (Large mounting flange)
   │                                  │
   └──────────────────────────────────┘
```

- **$D_1$ (Mounting Flange):** The largest diameter, located at one end of the cylinder.
- **$D_2$ (Intermediate Shoulder):** A step transition smaller than $D_1$.
- **$D_3$ (Bearing Journal):** A precision cylindrical section.
- **$D_4$ (Shaft End):** The smallest diameter at the opposite end.

Even though some stepped cylinders may have similar nominal dimensions on certain sections (such as $D_3$ and $D_4$), an industrial inspection system must measure each section independently so that local tool wear, taper, or machining flaws in one section can be isolated.

### 2.2 The Inspection Challenge
1. **Multi-Object Handling:** The fixture holds nine jobs simultaneously arranged in a $3 \times 3$ grid. The system must automatically detect, segment, and index all nine jobs without manual user intervention.
2. **Surface Specularity and Glare:** Machined steel and aluminum components reflect ambient light. Under direct top lighting, specular highlights create false edges and gradient spikes that confuse standard edge detectors.
3. **Subpixel Precision:** At a typical optical working distance, a single image pixel might represent $0.1\text{ mm}$ ($100\ \mu\text{m}$). Standard engineering tolerances are often $\pm 0.05\text{ mm}$ ($\pm 50\ \mu\text{m}$) or tighter. Therefore, standard discrete pixel counts are insufficient; the system requires continuous subpixel edge localization.
4. **Illumination Fluctuations:** Industrial factory lighting and backlighting units experience voltage variations, LED thermal drift, and ambient daylight shifts. The segmentation stage must remain robust and avoid clipping component boundaries when illumination shifts.

---

## 3. High-Level System Architecture

The project adheres to strict separation of concerns across multiple independent layers:

```
+-------------------------------------------------------------------------------+
|                             IMAGE ACQUISITION                                 |
|                       app/core/image_loader.py                                |
|  - Validates file existence and BGR 3-channel structure                       |
|  - Preserves native resolution (2448 x 2048) without downsampling            |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                       COMPONENT SEGMENTATION & ROI                            |
|                        app/core/job_detector.py                               |
|  - Red channel isolation (dark silhouette against red backlight)              |
|  - Adaptive Otsu thresholding with relative ceiling: min(T_Otsu, 0.55*P99)    |
|  - Morphological closing (9x9 rectangular kernel)                             |
|  - Connected components analysis (area, aspect ratio, minimum size filtering) |
|  - Grid clustering into 3 rows x 3 columns; deterministic naming              |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                         ORIENTATION NORMALIZATION                             |
|                          app/core/alignment.py                                |
|  - Rotates horizontal job 90 deg counter-clockwise (cv2.ROTATE_90_COUNTER...)|
|  - Maps flange (D1) to bottom, D4 to top; aligns cylinder axis vertically     |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                       SUBPIXEL DIAMETER MEASUREMENT                           |
|                      app/core/diameter_engine.py                              |
|  - Extracts configurable vertical bands (D1, D2, D3, D4)                      |
|  - Computes horizontal Sobel gradient (Sobel-X)                               |
|  - Median column-wise projection (suppresses scratches and specular points)   |
|  - Inward outer-edge search above adaptive threshold                          |
|  - 3-point parabolic peak interpolation for subpixel coordinates              |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                         METROLOGY & DECISION                                  |
|              app/core/calibration.py & tolerance_checker.py                   |
|  - Pixel-to-mm conversion via explicit CalibrationModel protocol              |
|  - Bilateral symmetric tolerance evaluation (Nominal +/- Tolerance)          |
|  - Generates PASS, FAIL, NOT_CONFIGURED, or MEASUREMENT_FAILED verdicts       |
+-------------------------------------------------------------------------------+
```

---

## 4. Repository Structure and Inventory

```text
Machine_vision/
├── app/
│   ├── __init__.py
│   ├── main.py                          # Convenience CLI entry point for edge experiments
│   ├── core/
│   │   ├── __init__.py
│   │   ├── alignment.py                 # Job normalizer protocol & 90 deg CCW rotation
│   │   ├── calibration.py               # Linear scale calibration protocol & model
│   │   ├── diameter_engine.py           # Sobel-X, median projection, parabolic subpixel fit
│   │   ├── edge_detector.py             # Multi-channel Canny, Sobel, Otsu benchmarking
│   │   ├── image_loader.py              # Unscaled BGR loader enforcing resolution integrity
│   │   ├── inspector.py                 # High-level pipeline coordinator & stage timer
│   │   ├── job_detector.py              # Adaptive red silhouette & fixed ROI detectors
│   │   ├── measurement.py               # Pixel & calibrated diameter dataclasses
│   │   ├── measurement_spec.py          # Dimensionless vertical measurement bands [top, bot]
│   │   ├── preprocessor.py              # Multi-channel extraction (BGR, Gray, R, B, HSV-V)
│   │   ├── result.py                    # InspectionState enum & structured result containers
│   │   └── tolerance_checker.py         # Bilateral symmetric engineering tolerance checks
│   └── utils/
│       ├── __init__.py
│       └── timing.py                    # High-resolution wall-clock profiling context manager
├── data/
│   ├── config/
│   │   └── bmp_measurement_config.synthetic.example.json  # Reference specification schema
│   ├── input/                           # 4 industrial BMP images + 9 smartphone test images
│   └── output/                          # Annotated PNGs, structured JSONs, and CSV reports
├── docs/                                # Technical documentation, math guides, and viva prep
├── scripts/
│   ├── run_bmp_inspection.py            # Primary production script for 9-job BMP fixture
│   ├── run_edge_experiment.py           # Multi-method edge extraction runner
│   ├── run_four_diameter_experiment.py  # Configuration-driven manual ROI test runner
│   └── select_job_roi.py                # Interactive/CLI tool to select ROIs on single images
├── tests/
│   ├── test_adaptive_segmentation.py    # 20 regression tests for Phase 2.1 adaptive detector
│   ├── test_bmp_inspection.py           # Integration tests for 3x3 fixture pipeline
│   ├── test_four_diameter.py            # Unit tests for D1-D4 measurement engine
│   ├── test_measurement.py              # Geometry, subpixel arithmetic & calibration tests
│   └── test_tolerance_checker.py        # Bilateral tolerance evaluation tests
├── requirements.txt                     # numpy>=1.26, opencv-python>=4.8
└── README.md                            # High-level architectural overview
```

---

## 5. Technology Stack & Specifications

| Component | Technology | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Language** | Python | 3.13.5 | Core development language |
| **Vision Library** | OpenCV (`opencv-python`) | 5.0.0.93 | Gradients, thresholding, connected components, morphology |
| **Numerical Engine** | NumPy | 2.5.3 | Array slicing, median projection, parabolic interpolation |
| **Testing Framework**| Pytest | 9.1.1 | Regression and unit testing (46 tests total) |
| **Camera Hardware** | Industrial GigE (Red Backlit)| 2448 × 2048 px | Uncompressed monochrome/color fixture images (14.34 MB each) |
| **Mobile Hardware** | Smartphone Camera | 960 × 1280 to 1200 × 1600 px | Ambient exploratory test images |

---

## 6. Key Design Philosophies

1. **Metrological Truth over Fabricated Results:**  
   The codebase never outputs synthetic millimeter values or PASS/FAIL verdicts unless an explicit physical calibration model and validated engineering tolerances are injected. Without calibration, the system outputs raw pixel measurements with a status of `NOT_CONFIGURED`.
2. **Preservation of Raw Pixel Scale:**  
   `image_loader.py` deliberately forbids downsampling or resizing. Resizing an image interpolates pixel values, creating artificial edge gradients and permanently destroying dimensional calibration.
3. **Directional Separation:**  
   Instead of isotropic gradient operators (which respond equally to horizontal shoulders, vertical sides, and surface scratches), the pipeline uses directional Sobel-X filtering to isolate the vertical physical boundaries of the cylinders.
4. **Statistical Outlier Rejection:**  
   By collapsing the 2D measurement band into a 1D profile using the column-wise **median** rather than the mean, scratches, pinholes, and specular glints are completely rejected.
