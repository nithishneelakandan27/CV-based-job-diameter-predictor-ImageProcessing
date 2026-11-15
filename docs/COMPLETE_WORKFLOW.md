# Complete Technical Workflow: End-to-End Processing Pipeline

This document walks through every single processing stage of the dimensional inspection pipeline in strict execution order.

---

## High-Level Execution Sequence

```text
[Input Image: 2448x2048 BMP]
       │
       ▼
[Stage 1: Image Acquisition & Ingestion] ───> load_bgr_image()
       │
       ▼
[Stage 2: Red Channel Extraction] ──────────> image[:, :, 2]
       │
       ▼
[Stage 3: Adaptive Otsu Segmentation] ──────> _adaptive_threshold() & cv2.inRange()
       │
       ▼
[Stage 4: Morphological Closing] ───────────> cv2.morphologyEx(..., MORPH_CLOSE)
       │
       ▼
[Stage 5: Connected Components & Filtering] > cv2.connectedComponentsWithStats()
       │
       ▼
[Stage 6: Grid Grouping & ROI Extraction] ──> _group_rows() & JobROI.extract()
       │
       ▼  (Loop over each of the 9 extracted Job ROIs)
[Stage 7: Orientation Normalization] ───────> cv2.rotate(..., ROTATE_90_COUNTERCLOCKWISE)
       │
       ▼  (Loop over each diameter band: D1, D2, D3, D4)
[Stage 8: Band Extraction & Preprocessing] ─> MeasurementRegion.pixel_rows() & Grayscale
       │
       ▼
[Stage 9: Gaussian Smoothing] ──────────────> cv2.GaussianBlur(5x5)
       │
       ▼
[Stage 10: Directional Sobel-X Gradient] ───> cv2.Sobel(dx=1, dy=0)
       │
       ▼
[Stage 11: Median Column Projection] ───────> np.median(..., axis=0) + 1D Gaussian
       │
       ▼
[Stage 12: Inward Outer-Edge Search] ───────> _find_outer_edge() with adaptive threshold
       │
       ▼
[Stage 13: 3-Point Parabolic Interpolation] > _subpixel_peak()
       │
       ▼
[Stage 14: Diameter & Metric Computation] ──> right_edge - left_edge & LinearScaleCalibration
       │
       ▼
[Stage 15: Tolerance Evaluation] ───────────> check_tolerance() (Nominal ± Tolerance)
       │
       ▼
[Stage 16: Annotation & Export] ────────────> annotate(), write_image_outputs() (PNG/JSON/CSV)
```

---

## Detailed Stage-by-Stage Breakdown

### Stage 1: Image Acquisition & Ingestion
- **What happens:** The uncompressed BMP image (or test JPEG/PNG) is read from disk into an 8-bit unsigned integer NumPy array with shape `(2048, 2448, 3)` in OpenCV's standard BGR order.
- **Why it is required:** Dimensional inspection requires raw, uncompressed optical sensor data.
- **Algorithm used:** Direct memory file decoding.
- **Input:** File path on disk (string or `Path`).
- **Output:** NumPy array `np.ndarray` of shape `(2048, 2448, 3)`, dtype `uint8`.
- **What happens if removed:** No downstream processing can occur; the system crashes with file I/O errors.
- **Python / OpenCV code:**  
  `app/core/image_loader.py` -> `load_bgr_image()` using `cv2.imread(str(image_path), cv2.IMREAD_COLOR)`.  
  *Crucial Rule:* Deliberately avoids any resizing (`cv2.resize`) to preserve the 1:1 pixel scale.

---

### Stage 2: Red Channel Extraction
- **What happens:** The 3-channel BGR image is indexed to extract only channel index 2: `red = image[:, :, 2]`.
- **Why it is required:** The inspection fixture uses an intense **red diffuse LED backlight** behind the parts. To the camera, the red backlight appears very bright ($R \in [60, 130]$ counts), while the opaque metallic cylinders block the red light and appear almost completely dark ($R \in [1, 10]$ counts). In the blue and green channels, overall intensity is near zero ($< 3$ counts), offering poor contrast. The red channel provides the highest signal-to-noise ratio (SNR) for silhouette segmentation.
- **Algorithm used:** Matrix channel slicing.
- **Input:** 3D array `(2048, 2448, 3)`.
- **Output:** 2D single-channel array `(2048, 2448)`, dtype `uint8`.
- **What happens if removed:** If a standard grayscale conversion ($0.299R + 0.587G + 0.114B$) were used instead, the weak green and blue channels would dilute the bright red backlight signal, reducing the dynamic range by roughly $70\%$.
- **Python / OpenCV code:**  
  `app/core/job_detector.py` -> `DarkRedSilhouetteJobDetector.locate()`:
  ```python
  red = image[:, :, 2]
  ```

---

### Stage 3: Adaptive Otsu Thresholding
- **What happens:** An adaptive binary threshold $T_{final}$ is computed and applied to the red channel to separate the dark component silhouettes from the bright backlight.
- **Why it is required:** Backlight brightness varies due to power fluctuations and temperature drift. A fixed threshold (like the old `max_red_value = 35`) fails when the backlight brightens, causing component clipping.
- **Algorithm used:** Otsu's bimodal inter-class variance maximization combined with a physics-grounded relative safety ceiling:
  $$T_{Otsu} = \arg\max_t \sigma_b^2(t)$$
  $$\text{Ceiling} = \min\left(\lfloor 0.55 \cdot P_{99} \rfloor, \; 90\right)$$
  $$T_{final} = \min(T_{Otsu}, \text{Ceiling})$$
  Pixels with values in $[0, T_{final}]$ are mapped to 255 (white/foreground), and pixels $> T_{final}$ are mapped to 0 (black/background).
- **Input:** 2D red channel array `(2048, 2448)`.
- **Output:** Binary silhouette mask `(2048, 2448)`, values $\in \{0, 255\}$.
- **What happens if removed:** Component boundaries cannot be separated from the background; connected components cannot run.
- **Python / OpenCV code:**  
  `app/core/job_detector.py` -> `_adaptive_threshold()` & `locate()`:
  ```python
  otsu_val = int(round(float(cv2.threshold(red, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])))
  peak_red = int(np.percentile(red, 99))
  ceiling = min(int(self.red_ceiling_fraction * max(peak_red, 1)), 90)
  threshold = min(otsu_val, ceiling)
  silhouette = cv2.inRange(red, 0, threshold)
  ```

---

### Stage 4: Morphological Closing
- **What happens:** A morphological dilation followed by erosion is performed on the binary silhouette using a $9 \times 9$ rectangular structuring element.
- **Why it is required:** Machined metal components often have tiny surface reflections, oil spots, or chamfer glints near their edges that could create internal black pinholes or edge notches in the binary mask. Morphological closing bridges small gaps and smooths outer boundary noise.
- **Algorithm used:** Mathematical Morphology: $(A \oplus B) \ominus B$.
- **Input:** Binary mask `(2048, 2448)`.
- **Output:** Cleaned binary mask `(2048, 2448)`.
- **What happens if removed:** Internal pinholes or broken edges can cause components to split into multiple fragmented blobs or fail the solid area threshold.
- **Python / OpenCV code:**  
  `app/core/job_detector.py` -> `locate()`:
  ```python
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.close_kernel_size, self.close_kernel_size))
  silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_CLOSE, kernel)
  ```

---

### Stage 5: Connected Components Analysis & Filtering
- **What happens:** The binary mask is scanned for 8-connected clusters of foreground pixels. Each distinct blob is indexed, and its bounding box $(x, y, w, h)$, area, and centroid $(\bar{x}, \bar{y})$ are computed.
- **Why it is required:** The image contains nine components plus border artifacts (such as fixture clamps, outer frame shadows, and dust specks). Geometric filters discard non-component blobs.
- **Algorithm used:** Two-pass connected-component labeling with 8-way connectivity, followed by geometric predicate filtering:
  $$\text{Keep blob } \iff (\text{Area} \ge 50{,}000) \land (\text{Width} \ge 400) \land (\text{Height} \ge 200) \land (\text{Width} > \text{Height})$$
- **Input:** Morphologically closed binary mask.
- **Output:** List of filtered candidate tuples: `(x, y, width, height, center_x, center_y)`.
- **What happens if removed:** The pipeline would attempt to measure fixture screws, cable shadows, and dust flecks as components, leading to measurement failure.
- **Python / OpenCV code:**  
  `app/core/job_detector.py` -> `locate()`:
  ```python
  _, _, statistics, centroids = cv2.connectedComponentsWithStats(silhouette, connectivity=8)
  ```

---

### Stage 6: Grid Grouping & ROI Extraction
- **What happens:** The candidate components are clustered into three horizontal rows by their $y$-centroids, sorted from left to right within each row by their $x$-centroids, assigned deterministic IDs (`job_r01_c01` to `job_r03_c03`), padded by 12 pixels on all four sides, and cropped from the source image.
- **Why it is required:** Sorting by raw top-left coordinates is unstable if parts sit with slight tilt. Centroid-based clustering with a dynamic row gap ($0.5 \times \text{median height}$) guarantees deterministic indexing. Padding ensures the outer silhouette edge is never clipped.
- **Algorithm used:** Hierarchical clustering by centroid distance:
  $$\text{Same row } \iff |y_i - \bar{y}_{row}| \le 0.5 \cdot \text{median}(h)$$
- **Input:** List of candidates and full BGR image.
- **Output:** Tuple of nine `JobROI` objects and extracted sub-images of shape $\approx (408, 612, 3)$.
- **What happens if removed:** Components cannot be individually isolated or linked to traceability logs.
- **Python / OpenCV code:**  
  `app/core/job_detector.py` -> `_group_rows()`, `JobROI.extract()`.

---

### Stage 7: Orientation Normalization
- **What happens:** The cropped horizontal job sub-image is rotated $90^\circ$ counter-clockwise (`ROTATE_90_COUNTERCLOCKWISE`).
- **Why it is required:** In the fixture, components lie horizontally with their large flange ($D_1$) on the left. The downstream measurement engine expects an upright cylinder whose axis is vertical, with $D_1$ at the bottom ($y \approx 0.84 - 0.92$) and $D_4$ at the top ($y \approx 0.20 - 0.27$).
- **Algorithm used:** Matrix transposition followed by row reversal (90° CCW rotation).
- **Input:** Horizontal sub-image of shape $(H_{src}, W_{src}, 3) \approx (408, 612, 3)$.
- **Output:** Upright sub-image of shape $(W_{src}, H_{src}, 3) \approx (612, 408, 3)$.
- **What happens if removed:** The measurement bands would slice along the length of the cylinder rather than across its diameter, failing completely.
- **Python / OpenCV code:**  
  `app/core/alignment.py` -> `HorizontalFlangeLeftJobNormalizer.normalize()`:
  ```python
  cv2.rotate(job_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
  ```

---

### Stage 8: Band Extraction & Preprocessing
- **What happens:** For each diameter ($D_1, D_2, D_3, D_4$), the normalized height $H$ is multiplied by configured fractional boundaries $[\alpha, \beta]$ to determine the pixel rows $[y_{start}, y_{end})$. The sub-image is converted to grayscale.
- **Why it is required:** Stepped cylinders have fillets, chamfers, and transition shoulders. Slicing specific vertical intervals isolates flat cylindrical bearing bands.
- **Algorithm used:** Dimensionless fractional scaling:
  $$y_{start} = \operatorname{round}(\alpha \cdot H), \quad y_{end} = \operatorname{round}(\beta \cdot H)$$
- **Input:** Upright ROI and `MeasurementRegion`.
- **Output:** Row bounds $[y_{start}, y_{end})$ and 2D grayscale array.
- **What happens if removed:** Measurements would sample corner chamfers and fillets, yielding incorrect and inconsistent diameter values.
- **Python / OpenCV code:**  
  `app/core/measurement_spec.py` -> `MeasurementRegion.pixel_rows()`, `app/core/preprocessor.py` -> `prepare_channels()`.

---

### Stage 9: Gaussian Smoothing
- **What happens:** The grayscale ROI is convolved with a 2D Gaussian kernel of size $5 \times 5$ ($\sigma \approx 0.85$).
- **Why it is required:** Raw camera sensors introduce high-frequency electronic shot noise. Taking numerical derivatives (gradients) on noisy data amplifies noise spikes into false edge peaks.
- **Algorithm used:** 2D Gaussian spatial convolution:
  $$I_{smooth}(x, y) = I(x, y) * G_{5\times 5}(x, y)$$
- **Input:** 2D grayscale ROI.
- **Output:** Smoothed 2D float/uint8 image.
- **What happens if removed:** Edge gradient profiles become jagged with multiple local maxima, causing erratic edge peak detection.
- **Python / OpenCV code:**  
  `app/core/diameter_engine.py` -> `BoundaryEdgeLocator.locate()`:
  ```python
  filtered = cv2.GaussianBlur(source, (5, 5), 0)
  ```

---

### Stage 10: Directional Sobel-X Gradient
- **What happens:** The horizontal derivative is computed using a $3 \times 3$ Sobel kernel:
  $$K_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$$
  The absolute value of the response is stored as single-precision float (`CV_32F`).
- **Why it is required:** Since the cylinder axis is vertical, its outer physical boundaries run vertically. Sobel-X isolates transitions in the horizontal ($x$) direction while producing near-zero response to horizontal transitions, tool marks, or grinding lines.
- **Algorithm used:** Discrete convolution with horizontal central difference kernel: $G_x = |\frac{\partial I}{\partial x}|$.
- **Input:** Smoothed grayscale ROI.
- **Output:** 2D float array $G_x(y, x)$ of shape $(H, W)$.
- **What happens if removed:** Without edge gradients, boundary transitions cannot be localized. If an isotropic filter (like Laplacian or Canny magnitude) were used, horizontal scratches and grinding marks would create strong false edges.
- **Python / OpenCV code:**  
  `app/core/diameter_engine.py` -> `locate()`:
  ```python
  response = np.abs(cv2.Sobel(filtered, cv2.CV_32F, 1, 0, ksize=3))
  ```

---

### Stage 11: Median Column Projection
- **What happens:** Across the selected row interval $[y_{start}, y_{end})$, the 2D gradient band is collapsed into a 1D horizontal vector using the **column-wise median**:
  $$S(x) = \operatorname{median}_{y \in [y_{start}, y_{end})} G_x(y, x)$$
  This 1D signal is then smoothed with a 1D Gaussian kernel ($\sigma = 1.5$).
- **Why it is required:** A 2D band contains hundreds of rows. Taking the median across rows eliminates localized flaws (scratches, dust specks, specularity glints) because an anomaly affecting $< 50\%$ of the rows has zero impact on the median.
- **Algorithm used:** Statistical 50th-percentile order statistics along column slices, followed by 1D Gaussian regularization.
- **Input:** 2D gradient sub-array of shape $(y_{end} - y_{start}, W)$.
- **Output:** 1D array $S(x)$ of length $W$ (approx. 408 values).
- **What happens if removed:** If individual rows are analyzed independently, localized surface defects cause random measurement jitter. If an arithmetic mean is used, a single bright reflection spike skews the edge location.
- **Python / OpenCV code:**  
  `app/core/diameter_engine.py` -> `locate()`:
  ```python
  scores = np.median(response[row_start:row_end, :], axis=0)
  scores = cv2.GaussianBlur(scores.reshape(1, -1), (0, 0), 1.5).reshape(-1)
  ```

---

### Stage 12: Inward Outer-Edge Search
- **What happens:** The 1D signal is split at its midpoint into Left and Right halves. Scanning proceeds from the outside inward (Left: $0 \to \text{mid}$; Right: $W \to \text{mid}$). The algorithm finds the first prominent local maximum that exceeds an adaptive threshold:
  $$\text{Threshold} = \max\left(2.0, \; \operatorname{median}(S) + 0.5 \cdot \operatorname{std}(S), \; 0.15 \cdot \max(S)\right)$$
  If the detected outer peak has confidence $< 0.5$ relative to the maximum peak, the algorithm falls back to the global maximum to avoid stopping on faint external dust particles.
- **Why it is required:** Internal reflections or step transitions inside the component silhouette can produce strong gradient peaks. Scanning from the outside inward ensures the system locks onto the true physical boundary.
- **Algorithm used:** Constrained 1D directional peak finding with adaptive noise floor estimation.
- **Input:** 1D profile halves.
- **Output:** Discrete integer peak indices $x_{left}^*$ and $x_{right}^*$ with confidence scores.
- **What happens if removed:** The algorithm might select an internal step edge or specular reflection line inside the cylinder, yielding an artificially small diameter.
- **Python / OpenCV code:**  
  `app/core/diameter_engine.py` -> `_find_outer_edge()`.

---

### Stage 13: Three-Point Parabolic Interpolation
- **What happens:** Around each discrete integer peak $x^*$, a continuous quadratic curve $f(\delta) = a\delta^2 + b\delta + c$ is fitted to the three discrete points: $S(x^* - 1)$, $S(x^*)$, and $S(x^* + 1)$. The continuous analytical vertex offset $\delta^* \in [-0.5, 0.5]$ is computed:
  $$\delta^* = \frac{1}{2} \cdot \frac{S(x^* - 1) - S(x^* + 1)}{S(x^* - 1) - 2S(x^*) + S(x^* + 1)}$$
  $$x_{\text{subpixel}} = x^* + \delta^*$$
- **Why it is required:** Digital pixels are discrete steps ($1\text{ px} \approx 0.1\text{ mm}$). Parabolic fitting interpolates continuous subpixel edge coordinates down to fractions of a pixel ($0.05 - 0.1\text{ px}$ repeatability).
- **Algorithm used:** Second-order Taylor series / parabolic vertex fitting.
- **Input:** Integer peak index $x^*$ and 1D scores $S$.
- **Output:** Continuous float edge coordinate $x_{\text{subpixel}}$ (e.g. $8.792\text{ px}$).
- **What happens if removed:** Measurements become quantized to integer pixel values, creating artificial step jumps of $\pm 1\text{ pixel}$ ($\approx \pm 0.1\text{ mm}$), which violates precision tolerances.
- **Python / OpenCV code:**  
  `app/core/diameter_engine.py` -> `_subpixel_peak()`.

---

### Stage 14: Diameter & Metric Computation
- **What happens:** The pixel diameter is computed as:
  $$D_{px} = x_{\text{subpixel, right}} - x_{\text{subpixel, left}}$$
  If an explicit `CalibrationModel` is provided, the physical millimeter diameter is evaluated:
  $$D_{mm} = D_{px} \cdot s_{\text{cal}}$$
  If no calibration model is injected, $D_{mm}$ is set to `None` and the status remains `NOT_CONFIGURED`.
- **Why it is required:** Transforms edge locations into dimensional values.
- **Algorithm used:** 1D Euclidean distance and linear scale transformation.
- **Input:** Left and right subpixel edge floats, calibration model.
- **Output:** `DiameterResult` object with $D_{px}$ and optional $D_{mm}$.
- **What happens if removed:** Raw coordinates remain unconnected to physical dimensions.
- **Python / OpenCV code:**  
  `app/core/measurement.py` -> `PixelDiameterMeasurement`, `app/core/calibration.py` -> `LinearScaleCalibration.pixels_to_mm()`.

---

### Stage 15: Tolerance Evaluation
- **What happens:** For each diameter with a defined nominal dimension and tolerance ($D_{nom} \pm t$), the measured value $D_{mm}$ is checked:
  $$\text{Status} = \begin{cases} \text{PASS}, & D_{nom} - t \le D_{mm} \le D_{nom} + t \\ \text{FAIL}, & \text{otherwise} \end{cases}$$
  The overall job status is evaluated: a job is `PASS` if and only if all four diameters pass.
- **Why it is required:** Automates the quality control acceptance/rejection decision.
- **Algorithm used:** Inclusive interval comparison: $[D_{nom} - t, \; D_{nom} + t]$.
- **Input:** $D_{mm}$ and `SymmetricTolerance` specification.
- **Output:** `ToleranceResult` (`PASS` or `FAIL`).
- **What happens if removed:** Human operators would have to manually compare values against engineering drawings.
- **Python / OpenCV code:**  
  `app/core/tolerance_checker.py` -> `check_tolerance()`, `app/core/diameter_engine.py` -> `_job_status()`.

---

### Stage 16: Annotation & Export
- **What happens:** The upright measurement endpoints are rotated back to the original source image coordinate system using the inverse coordinate mapping:
  $$\text{source}_x = \text{ROI}_x + \text{ROI}_{width} - 1 - \operatorname{round}\left(\frac{y_{start} + y_{end}}{2}\right)$$
  $$\text{source}_{y, top} = \text{ROI}_y + \operatorname{round}(x_{left})$$
  $$\text{source}_{y, bottom} = \text{ROI}_y + \operatorname{round}(x_{right})$$
  Bounding boxes, color-coded measurement lines ($D_1$: yellow, $D_2$: cyan, $D_3$: green, $D_4$: magenta), and labels are drawn on the full image. Three files are generated:
  1. Annotated PNG image (`*_annotated.png`)
  2. Structured JSON report (`*_results.json`)
  3. Machine-readable CSV table (`*_results.csv`)
- **Why it is required:** Provides visual feedback for machine operators and tabular data for factory ERP/SPC databases.
- **Algorithm used:** Inverse coordinate rotation and formatted file export.
- **Input:** Full BGR image and `ConfiguredImageInspectionResult`.
- **Output:** Saved PNG, JSON, and CSV files on disk.
- **What happens if removed:** No visual or digital record of the inspection is persisted.
- **Python / OpenCV code:**  
  `scripts/run_bmp_inspection.py` -> `annotate()`, `write_image_outputs()`.
