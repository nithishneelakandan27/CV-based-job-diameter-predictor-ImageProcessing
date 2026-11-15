# Project Presentation Outline & Slide Deck Guide

This document outlines a professional, 12-slide technical presentation for project defense, committee evaluation, or conference presentation. Every slide includes slide bullet points, recommended visual diagrams, and a verbal presentation script.

---

## Slide 1: Title & Project Overview

### Slide Content
- **Project Title:** Computer Vision-Based Automated Job Diameter Measurement
- **Sub-title:** High-Speed Multi-Part Subpixel Dimensional Metrology for Precision Manufacturing
- **Domain:** Industrial Machine Vision, Digital Metrology, Classical Computer Vision
- **Technology Stack:** Python 3.13, OpenCV 5.0, NumPy 2.5
- **Presenter:** [Your Name / Department]

### Suggested Visual Diagram
A split graphic showing:
- Left: An industrial $3 \times 3$ fixture holding 9 dark cylindrical stepped components against a bright red background.
- Right: The automated output showing color-coded subpixel measurement lines and green PASS indicators.

### Verbal Script (30 seconds)
> "Good morning, respected committee members and reviewers. Today I am presenting my project: Computer Vision-Based Automated Job Diameter Measurement.
> In high-precision manufacturing, validating cylindrical components against tight dimensional tolerances is essential. Our system automates the multi-object inspection of nine stepped shafts simultaneously from a single high-resolution image, achieving subpixel measurement precision in under 90 milliseconds per frame."

---

## Slide 2: Industrial Problem & Motivation

### Slide Content
- **The Workpiece:** Multi-diameter stepped shafts with critical bearing journals and mounting flanges.
- **The Current Industry Bottleneck:**
  - Manual inspection using vernier calipers and micrometers is slow, operator-biased, and error-prone.
  - Offline Coordinate Measuring Machines (CMMs) take minutes per part and cannot keep up with high-speed CNC turning lines.
- **Key Challenges in Optical Inspection:**
  - Specular glare and surface reflections on turned steel.
  - Multi-part handling: detecting and indexing 9 components simultaneously in a single frame.
  - Optical exposure variations: industrial backlights drift with temperature and line voltage.

### Suggested Visual Diagram
A flow comparison diagram:
`Manual Gauging (30-60 sec/part, human fatigue) vs. Automated Vision (< 10 ms/part, 100% inline)`

### Verbal Script (45 seconds)
> "In precision mechanical engineering, components like stepped axles and shafts require multi-diameter quality checks. Traditional quality control relies on manual micrometers or offline CMM machines. Manual gauging is labor-intensive and suffers from human bias, while CMMs create severe production bottlenecks.
> Optical vision offers high speed, but optical inspection of metallic parts faces serious hurdles: metal surfaces produce harsh glare, multiple components must be tracked without confusion, and factory lighting variations can distort component boundaries. This project solves these challenges using a deterministic, classical computer vision architecture."

---

## Slide 3: Proposed Solution & Core Innovations

### Slide Content
- **Monochromatic Silhouette Optical Geometry:**
  - Telecentric diffuse red LED backlighting transforms reflective cylinders into sharp, high-contrast silhouettes.
- **Adaptive Relative-Ceiling Segmentation:**
  - Dynamic Otsu thresholding coupled with a 99th-percentile relative ceiling to prevent edge clipping under illumination shifts.
- **Robust Directional Gradient Differential Profiling:**
  - Directional horizontal Sobel-X filtering combined with vertical column-wise median projection to eliminate scratches and specular glints.
- **Three-Point Parabolic Subpixel Localization:**
  - Continuous curve-fitting for true subpixel edge coordinates down to fractions of a pixel.

### Suggested Visual Diagram
Block diagram illustrating the four pillars:
`Optical Backlighting ──> Adaptive Segmentation ──> Median Gradient Projection ──> Subpixel Metrology`

### Verbal Script (45 seconds)
> "Our solution replaces unreliable top lighting with a monochromatic red diffuse backlighting setup. This eliminates surface reflections and provides crisp silhouette edges.
> We built a modular, deterministic pipeline featuring three core innovations: first, an adaptive segmentation algorithm that prevents ROI clipping when backlight intensity shifts; second, directional Sobel-X filtering paired with column-wise median projection, which makes edge extraction impervious to surface scratches; and third, parabolic subpixel interpolation that extracts continuous physical coordinates."

---

## Slide 4: End-to-End System Architecture

### Slide Content
- **Layered Architecture with Strict Separation of Concerns:**
  1. `image_loader`: Enforces 1:1 raw sensor resolution without lossy resizing.
  2. `job_detector`: Locates components via adaptive red silhouette segmentation.
  3. `alignment`: Rotates components $90^\circ$ CCW to normalize the longitudinal axis.
  4. `diameter_engine`: Slices configured vertical bands and localizes subpixel edges.
  5. `calibration & tolerance_checker`: Injects physical scale models and evaluates bilateral tolerances.

### Suggested Visual Diagram
Complete pipeline flow chart:
```text
[Input Image: 2448x2048] ──> [Red Extraction] ──> [Adaptive Otsu] ──> [Connected Components]
         ──> [9x ROIs] ──> [90° CCW Rotation] ──> [Sobel-X + Median] ──> [Parabolic Subpixel]
         ──> [D1-D4 Pixel Difference] ──> [Linear Calibration] ──> [PASS/FAIL Verdict]
```

### Verbal Script (60 seconds)
> "This slide illustrates our system architecture. The pipeline enforces a clean separation of concerns.
> `image_loader` reads the raw 5-megapixel sensor array, strictly forbidding resizing to protect optical calibration.
> `job_detector` isolates the red channel, performs adaptive thresholding, and groups connected components into a 3x3 fixture grid.
> `alignment` normalizes component orientation by rotating them 90 degrees counter-clockwise so their cylinder axis is vertical.
> Next, `diameter_engine` slices specific vertical measurement bands, computes horizontal gradients, projects column medians, and fits parabolic curves to find subpixel boundaries.
> Finally, `calibration` and `tolerance_checker` convert pixel measurements to millimeters and verify engineering tolerances."

---

## Slide 5: Image Preprocessing & Adaptive Segmentation

### Slide Content
- **Red Channel Isolation:**
  - $R \in [60, 130]$, while $B, G < 3\text{ counts}$. Directly extracting channel 2 maximizes dynamic range.
- **The Illumination Problem (Phase 2.0 Limitation):**
  - The previous implementation clamped the threshold at $35$. On brighter images where the true valley was at $50$, this truncated the component flange by $31\text{ pixels}$.
- **Phase 2.1 Adaptive Threshold Formulation:**
  $$T_{final} = \min\left(T_{Otsu}, \; \min(\lfloor 0.55 \cdot P_{99} \rfloor, \; 90)\right)$$
- **Morphological Closing:** $9 \times 9$ kernel bridges internal reflection holes.
- **Connected Components:** Filters by area ($\ge 50{,}000$), aspect ratio ($w > h$), and groups into a $3 \times 3$ grid.

### Suggested Visual Diagram
Before/After thresholding histograms showing the Otsu threshold moving from $30 \to 50$ with changing backlight, and the $0.55 \cdot P_{99}$ protective ceiling.

### Verbal Script (60 seconds)
> "Let us look at the preprocessing and segmentation stage. By isolating the red channel, we leverage the full dynamic range of the red backlight while ignoring dark green and blue channels.
> In our engineering audit, we discovered that the previous code hard-clamped the threshold at 35. When the backlight brightened, this clamp truncated the outer flange by 31 pixels.
> In Phase 2.1, we introduced an adaptive ceiling based on the 99th percentile of red intensity. This tracks illumination changes automatically while capping anomalous noise. A 9x9 morphological closing operation bridges tiny reflection pinholes, and connected components analysis filters out clamps and non-job shadows based on area and horizontal aspect ratio."

---

## Slide 6: Edge Detection & Directional Gradient Filtering

### Slide Content
- **Directional Sobel-X Convolution:**
  $$K_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix}, \quad G_x = |K_x * I|$$
  - High sensitivity to vertical boundaries; zero response to horizontal shoulders or lathe turning marks.
- **Why Not Canny?**
  - Canny outputs a binary 1-pixel edge map, destroying continuous gradient intensity. Sobel-X retains floating-point responses required for subpixel curve fitting.
- **Low-Pass Gaussian Regularization:**
  - $5 \times 5$ Gaussian kernel ($\sigma \approx 0.85$) eliminates camera electronic shot noise without shifting edge inflection points.

### Suggested Visual Diagram
Comparison showing:
- Raw image with horizontal lathe tool marks.
- Directional Sobel-X response showing only two bright vertical lines at the outer boundaries, with horizontal tool marks completely suppressed.

### Verbal Script (45 seconds)
> "For edge detection, we chose directional horizontal Sobel filtering rather than standard Canny.
> Canny binarizes edges into discrete pixels, destroying the continuous gradient profile needed for subpixel interpolation. Sobel-X preserves the floating-point derivative profile.
> Furthermore, because our cylinder axis is vertical, the boundaries we need to measure are vertical. Sobel-X responds strongly to vertical boundaries while completely ignoring horizontal shoulder transitions, chamfers, and circumferential lathe turning grooves. A 5x5 Gaussian blur suppresses high-frequency sensor noise before computing gradients."

---

## Slide 7: Subpixel Edge Localization & Median Projection

### Slide Content
- **Statistical Median Column Projection:**
  $$S(x) = \operatorname{median}_{y \in [y_{start}, y_{end})} G_x(y, x)$$
  - Breakdown point of $50\%$: completely rejects isolated surface scratches and specular glints affecting $< 50\%$ of rows.
- **Inward Outer-Edge Scan:**
  - Left half scans $0 \to \text{mid}$; Right half scans $W \to \text{mid}$.
  - Guarantees detection of the outer physical boundary rather than internal shoulder reflections.
- **Three-Point Parabolic Interpolation:**
  $$\delta^* = \frac{1}{2} \cdot \frac{S_{-1} - S_{+1}}{S_{-1} - 2S_0 + S_{+1}}, \quad x_{\text{subpixel}} = x^* + \delta^*$$
  - Resolves continuous subpixel edge coordinates down to fractions of a pixel ($0.05\text{ px}$).

### Suggested Visual Diagram
Graphic showing a 1D bell-shaped curve with three discrete sample points $S_{-1}, S_0, S_{+1}$, and a continuous red parabola fitted through them with the peak offset $\delta^*$ clearly marked.

### Verbal Script (60 seconds)
> "To collapse the 2D measurement band into a 1D profile, we use column-wise median projection. In statistics, the median has a 50% breakdown point. If a scratch or dust speck affects a few rows, the mean would be distorted, but the median is mathematically immune.
> Next, our algorithm scans from the outside inward to locate the outer silhouette boundary, avoiding internal step reflections.
> Finally, we apply 3-point parabolic interpolation. Digital pixels are discrete steps of 100 micrometers. By fitting a quadratic curve to the peak and its two neighbors, we find the continuous analytical maximum. This achieves subpixel repeatability down to fractions of a pixel."

---

## Slide 8: Diameter Measurement & Metrology Engine

### Slide Content
- **Dimensionless Measurement Bands:**
  - $D_1 \in [0.84, 0.92]$ (Mounting flange)
  - $D_2 \in [0.67, 0.76]$ (Intermediate step)
  - $D_3 \in [0.35, 0.58]$ (Bearing journal)
  - $D_4 \in [0.20, 0.27]$ (Shaft end)
- **Dimensional Formula:** $D_{px} = x_{\text{subpixel, right}} - x_{\text{subpixel, left}}$
- **Decoupled Calibration:** $D_{mm} = D_{px} \cdot s_{cal}$
- **Bilateral Tolerance Checking:**
  - $\text{PASS} \iff D_{nom} - t \le D_{mm} \le D_{nom} + t$
  - Overall Job Status: `PASS` only if all 4 diameters pass.

### Suggested Visual Diagram
Illustration of the normalized upright stepped shaft with four colored measurement bands ($D_1$ yellow, $D_2$ cyan, $D_3$ green, $D_4$ magenta) and horizontal measurement callouts.

### Verbal Script (45 seconds)
> "Here you see how diameters are extracted. Each diameter is defined as a dimensionless fractional band along the normalized cylinder height. This avoids chamfers and fillets.
> The diameter is computed as the horizontal distance between right and left subpixel boundaries.
> Our architecture strictly decouples image processing from calibration. If calibration is provided, pixel values are converted to millimeters and evaluated against symmetric drawing tolerances. If calibration is absent, the system outputs raw pixel measurements with a status of NOT_CONFIGURED, preventing fabricated or uncalibrated inspection verdicts."

---

## Slide 9: Experimental Results & D1 Recovery

### Slide Content
- **Dataset Evaluated:** 4 full-resolution industrial BMP images ($2448 \times 2048$, 9 jobs/image, 144 total diameter measurements).
- **100% Detection Rate:** All 36 job instances successfully located and indexed.
- **Verification of D1 Flange Recovery:**
  - Before Fix: Images 3 & 4 suffered D1 drops of up to $38.25\text{ px}$ ($\sigma_{repeat} = 15.73\text{ px}$).
  - After Phase 2.1: $D_1$ completely recovered ($\sigma_{repeat} = \mathbf{0.12 - 0.56\text{ px}}$).
- **Subpixel Measurement Repeatability Across All Jobs:**
  - $D_1$ Mean Repeatability: **$0.255\text{ px}$** ($\approx 25.5\ \mu\text{m}$)
  - $D_2$ Mean Repeatability: **$0.169\text{ px}$** ($\approx 16.9\ \mu\text{m}$)
  - $D_3$ Mean Repeatability: **$0.179\text{ px}$** ($\approx 17.9\ \mu\text{m}$)
  - $D_4$ Mean Repeatability: **$0.222\text{ px}$** ($\approx 22.2\ \mu\text{m}$)
- **Execution Speed:** $\mathbf{\approx 86.6\text{ ms per frame}}$ ($\approx 9.6\text{ ms per job}$).

### Suggested Visual Diagram
A bar chart or table comparing the D1 cross-image standard deviation before ($15.73\text{ px}$) versus after ($0.20\text{ px}$).

### Verbal Script (60 seconds)
> "This slide presents our verified empirical results. We tested across all four industrial fixture images, measuring 36 component instances and 144 total diameters.
> In Phase 2.0, D1 on Images 3 and 4 was severely clipped by up to 38 pixels due to threshold clamping. After our Phase 2.1 adaptive thresholding, D1 fully recovered, and cross-capture standard deviation dropped from over 15 pixels down to between 0.12 and 0.56 pixels.
> Across all four diameters, our mean repeatability standard deviation is between 0.17 and 0.25 pixels—well within subpixel precision. The entire 9-job inspection pipeline runs in just 86.6 milliseconds on a standard CPU."

---

## Slide 10: Technical Limitations & Metrology Gaps

### Slide Content
- **Center-to-Corner Measurement Discrepancy ($\Delta \approx 4.5\text{ px}$):**
  - Center jobs measure $D_1 \approx 388.9\text{ px}$, whereas corner jobs measure $D_1 \approx 384.4\text{ px}$.
  - Cannot be attributed exclusively to lens distortion without position-swapping experiments; it is confounded with real part manufacturing variations and perspective parallax.
- **Absence of Physical Calibration Artifacts:**
  - The current calibration uses synthetic scale values; physical calibration with traceable optical gauge pins remains pending.
- **Rigid 90-Degree Alignment Assumption:**
  - Tilted parts ($\theta \ge 1.5^\circ$) cause median projection peak broadening.
- **Ambient Light Vulnerability:**
  - High ambient front lighting contaminates silhouette contrast, causing the relative ceiling to clamp.

### Suggested Visual Diagram
Graphic showing the $3 \times 3$ grid with corner diameters ($384.4\text{ px}$) vs center diameter ($388.9\text{ px}$), illustrating potential optical lens distortion vectors.

### Verbal Script (45 seconds)
> "A responsible engineering project must honestly state its limitations.
> First, we observed a 4.5-pixel difference between center jobs and corner jobs. While optical lens distortion is a suspect, we cannot claim this conclusively because the 9 jobs are physically different parts. Part manufacturing variation is currently confounded with lens distortion.
> Second, our calibration remains synthetic pending physical gauge pin tests.
> Third, our rotation normalizer assumes near-zero part tilt in the fixture.
> And fourth, our adaptive ceiling is robust against backlight scaling, but can be degraded if ambient white room light reflects off the front of the components."

---

## Slide 11: Future Enhancements & Industrial Roadmap

### Slide Content
- **Phase 2.2: Camera Intrinsic Calibration:**
  - Acquire ChArUco target images to estimate focal length, principal point, and radial distortion coefficients ($k_1, k_2, p_1, p_2$), undistorting the sensor field.
- **Physical Gauge Pin Traceability:**
  - Measure certified Class XX optical reference cylinders in all 9 fixture slots to prove absolute millimeter accuracy.
- **PCA / RANSAC Part Tilt Alignment:**
  - Fit parallel boundary lines directly in 2D to measure perpendicular diameter on tilted components.
- **Optical Bandpass Filter:**
  - Install a $630\text{ nm}$ optical bandpass filter on the camera lens to block ambient factory lighting.
- **Multi-Core Pipeline Scaling:**
  - Distribute 9 job inspections across CPU worker processes to reach 50+ jobs per second.

### Suggested Visual Diagram
A roadmap diagram showing Phase 2.1 (Completed) -> Phase 2.2 (ChArUco Calibration) -> Phase 3.0 (Inline Hardware Deployment).

### Verbal Script (45 seconds)
> "To transition this system into factory production, we have defined a clear roadmap.
> In Phase 2.2, we will perform camera intrinsic calibration using a ChArUco board to remove optical distortion and perspective error.
> We will calibrate physical scale using certified optical gauge pins swapped across all 9 fixture pockets.
> We will implement PCA-based line fitting to dynamically compensate for part tilt, and add a 630 nm optical bandpass filter to completely eliminate ambient room light interference."

---

## Slide 12: Conclusion & Summary

### Slide Content
- **Summary of Accomplishments:**
  - Automated 9-component detection and 4-diameter subpixel inspection pipeline.
  - Eliminated Phase 2.0 ROI clipping via adaptive Otsu segmentation with relative safety ceiling.
  - Achieved verified subpixel repeatability ($\sigma \approx 0.17 - 0.25\text{ px}$) across 144 measurements.
  - Validated execution speed ($\approx 9.6\text{ ms per component}$ on CPU).
  - 46 comprehensive automated unit and regression tests (100% pass rate).
- **Core Engineering Takeaway:**
  - Classical computer vision delivers explainable, deterministic, high-speed metrology when guided by robust optical physics and rigorous statistical filtering.

### Suggested Visual Diagram
Summary card displaying key metrics:
`46/46 Tests Passed | 86.6 ms Full Pipeline | 0.17 px Repeatability | 100% Detection Rate`

### Verbal Script (30 seconds)
> "In conclusion, this project establishes a robust, highly repeatable, and computationally efficient baseline for automated dimensional inspection. By combining classical optical backlighting, adaptive silhouette segmentation, directional Sobel-X filtering, and subpixel parabolic interpolation, we achieved subpixel repeatability under 0.25 pixels and an inspection time of 86 milliseconds per frame, backed by 46 automated regression tests.
> Thank you for your time and attention. I am now ready to answer your questions."
