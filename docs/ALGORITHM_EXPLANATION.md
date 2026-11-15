# Algorithm Explanation & Design Decisions

This document provides a deep, technical dive into every major algorithm implemented in the inspection system, explaining what each algorithm does, why it was chosen over alternatives, where it lives in the codebase, and its practical engineering limitations.

---

## 1. Adaptive Red-Channel Otsu Thresholding

### What it does
Converts the continuous 8-bit red channel of the input image into a binary silhouette mask where component silhouettes are represented as white ($255$) and the backlit background is represented as black ($0$). It calculates an optimal threshold $T_{Otsu}$ that maximizes the statistical variance between foreground and background pixels, and clamps it using a relative safety ceiling based on the 99th-percentile of red intensity:
$$T_{final} = \min\left(T_{Otsu}, \; \min(\lfloor 0.55 \cdot P_{99} \rfloor, \; 90)\right)$$

### Why it is used (Rationale & Alternatives)
- **Why Otsu instead of fixed thresholding?** A fixed threshold fails when backlight voltage fluctuates or LED panels age. In our dataset, images 1 and 2 required thresholds of 30–39, while images 3 and 4 required 50. Otsu automatically tracks this shift.
- **Why Otsu instead of adaptive local thresholding (`cv2.adaptiveThreshold`)?** Adaptive local thresholding computes thresholds in small sliding windows (e.g., $15 \times 15$). Inside large solid regions of a component, local windows have near-zero variance, causing adaptive thresholding to detect false internal noise edges and hollow out the silhouette. Otsu is global and designed specifically for bimodal images.
- **Why the red channel?** The diffuse LED backlight is monochromatic red ($\approx 630\text{ nm}$). The red channel provides a dynamic range of $\approx 100\text{ counts}$, while blue and green have ranges $< 5\text{ counts}$.

### Implementation Reference
- **File:** [`app/core/job_detector.py`](file:///Users/neelakandan/Machine_vision/app/core/job_detector.py)
- **Functions:** `DarkRedSilhouetteJobDetector._adaptive_threshold()`, `locate()`

### Limitations
- Assumes a bimodal histogram. If significant ambient white light reflects off the front of the components, the histogram develops a third intermediate peak, causing Otsu to place the threshold inside the component body.
- Assumes spatially uniform backlighting across the $2448 \times 2048$ sensor.

---

## 2. Morphological Closing

### What it does
Applies mathematical morphology—specifically dilation followed by erosion—using a $9 \times 9$ rectangular structuring element:
$$\text{Closed Mask} = (A \oplus B) \ominus B$$

### Why it is used (Rationale & Alternatives)
- **Why Closing?** Machined steel components can have tiny specular glints, dust flecks, or surface oil dots that appear slightly brighter than the surrounding metallic body. A pure threshold would punch black holes inside the silhouette. Closing fills holes smaller than $9 \times 9$ pixels and smooths fine boundary notches without altering the overall macroscopic dimensions of the component.
- **Why not Opening?** Morphological opening (erosion then dilation) removes small foreground protrusions and would thin the silhouette, potentially eroding fine outer edge details.

### Implementation Reference
- **File:** [`app/core/job_detector.py`](file:///Users/neelakandan/Machine_vision/app/core/job_detector.py)
- **Function:** `DarkRedSilhouetteJobDetector.locate()`:
  ```python
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.close_kernel_size, self.close_kernel_size))
  silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_CLOSE, kernel)
  ```

### Limitations
- A $9 \times 9$ kernel slightly rounds very sharp exterior $90^\circ$ corners. This is acceptable here because outer corners are located outside the measurement bands.
- If two components are placed closer than 9 pixels apart, closing will fuse them into a single connected blob.

---

## 3. Connected Components Analysis with Geometric Filtering

### What it does
Scans the binary silhouette mask using 8-way pixel connectivity, groups adjacent foreground pixels into distinct labeled objects, and outputs their bounding rectangle coordinates $(x, y, w, h)$, pixel area, and centroid coordinates $(\bar{x}, \bar{y})$. Candidate objects are then subjected to geometric rejection rules:
$$\text{Accept Candidate} \iff (\text{Area} \ge 50{,}000) \land (w \ge 400) \land (h \ge 200) \land (w > h)$$

### Why it is used (Rationale & Alternatives)
- **Why Connected Components instead of Contour Detection (`cv2.findContours`)?** `cv2.connectedComponentsWithStats` returns bounding boxes, areas, and exact subpixel centroids in a single efficient pass without the overhead of constructing hierarchical polygonal contour trees.
- **Why the Aspect Ratio Filter ($w > h$)?** In the mechanical fixture, the cylindrical components are loaded horizontally. Clamp shadows, vertical fixture pillars, and border artifacts are vertical ($h > w$) or small ($< 50{,}000\text{ px}$). The filter completely eliminates non-job background blobs.

### Implementation Reference
- **File:** [`app/core/job_detector.py`](file:///Users/neelakandan/Machine_vision/app/core/job_detector.py)
- **Function:** `DarkRedSilhouetteJobDetector.locate()`

### Limitations
- If two components physically touch each other, connected components treats them as a single object with $w \approx 1200$ or $h \approx 800$, causing both to be rejected.
- Requires components to be loaded in the expected horizontal orientation.

---

## 4. Centroid-Based Grid Clustering & Row Grouping

### What it does
Groups the nine detected candidates into a structured $3 \times 3$ grid and assigns stable, deterministic identifiers:
$$\begin{matrix}
\text{job\_r01\_c01} & \text{job\_r01\_c02} & \text{job\_r01\_c03} \\
\text{job\_r02\_c01} & \text{job\_r02\_c02} & \text{job\_r02\_c03} \\
\text{job\_r03\_c01} & \text{job\_r03\_c02} & \text{job\_r03\_c03}
\end{matrix}$$

### Why it is used (Rationale & Alternatives)
- **Why not sort by bounding box top-left $(x, y)$?** If a component in column 2 is slightly tilted or machined with a slightly larger flange, its top edge $y$ might be a few pixels higher than the component in column 1. Lexicographical sorting on $(y, x)$ would place column 2 before column 1, scrambling the indexing.
- **The Clustering Algorithm:** `_group_rows()` sorts candidates by centroid $y$, computes the median component height, and establishes a dynamic row gap:
  $$\text{Row Gap} = 0.5 \cdot \operatorname{median}(\text{Height})$$
  A candidate starts a new row only if its centroid $y$ exceeds the current row's average centroid by more than `Row Gap`. Once grouped into rows, each row is sorted by centroid $x$ from left to right.

### Implementation Reference
- **File:** [`app/core/job_detector.py`](file:///Users/neelakandan/Machine_vision/app/core/job_detector.py)
- **Function:** `DarkRedSilhouetteJobDetector._group_rows()`

### Limitations
- Assumes all 9 fixture slots are filled. If slot `(row 1, col 2)` is empty, the third part in row 1 will be incorrectly labeled as `c02`.

---

## 5. Orientation Normalization ($90^\circ$ Counter-Clockwise Rotation)

### What it does
Applies an exact $90^\circ$ counter-clockwise rotation to each cropped job sub-image:
$$\text{Upright Image} = \operatorname{rotate90CCW}(\text{Source ROI})$$

```text
Horizontal (Source ROI):            Rotated Upright:
┌────────────────────────────┐      ┌──────────┐ ── D4 (top, y ~ 0.23 H)
│ [Flange]   [Shaft]         │ ===> │          │
│ D1         D2   D3   D4    │      │          │ ── D3 (y ~ 0.45 H)
└────────────────────────────┘      │          │ ── D2 (y ~ 0.71 H)
                                    ├──────────┤
                                    │ [Flange] │ ── D1 (bottom, y ~ 0.88 H)
                                    └──────────┘
```

### Why it is used (Rationale & Alternatives)
- **Why rotate?** The downstream measurement engine is configured around an upright cylindrical model where the cylinder axis is vertical ($y$-axis) and diameters are horizontal distances ($x_{right} - x_{left}$).
- Rotating $90^\circ$ CCW maps the source image's left side (where the large flange sits) to the bottom of the upright image, standardizing the order: $D_1$ at bottom, $D_4$ at top.

### Implementation Reference
- **File:** [`app/core/alignment.py`](file:///Users/neelakandan/Machine_vision/app/core/alignment.py)
- **Function:** `HorizontalFlangeLeftJobNormalizer.normalize()`

### Limitations
- Only corrects multiples of $90^\circ$. It does not compensate for slight mechanical tilt (e.g., $1.5^\circ$ part skew).

---

## 6. Directional Horizontal Sobel Gradient (Sobel-X)

### What it does
Computes the first spatial derivative in the horizontal ($x$) direction using a $3 \times 3$ convolution kernel:
$$G_x(y, x) = \left| \sum_{u=-1}^1 \sum_{v=-1}^1 K_x(u, v) \cdot I(y+u, x+v) \right|, \quad K_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$$

### Why it is used (Rationale & Alternatives)
- **Why Sobel-X instead of Canny?** Canny applies non-maximum suppression (NMS) and hysteresis thresholding, converting continuous gradient magnitudes into a 1-pixel-wide binary edge map. Binary edges discard gradient magnitude information, making true subpixel peak interpolation impossible. Sobel preserves the continuous bell-shaped gradient profile.
- **Why directional Sobel-X instead of isotropic Sobel ($G = \sqrt{G_x^2 + G_y^2}$)?** The cylindrical axis is vertical, so the outer physical boundaries are strictly vertical lines. Sobel-X responds strongly to vertical edges while producing zero response to horizontal shoulder steps, chamfers, or horizontal machining grooves.

### Implementation Reference
- **File:** [`app/core/diameter_engine.py`](file:///Users/neelakandan/Machine_vision/app/core/diameter_engine.py)
- **Function:** `BoundaryEdgeLocator.locate()`

### Limitations
- Assumes the edge is approximately vertical. If the component is tilted by $\theta$, the true orthogonal gradient vector is rotated by $\theta$, slightly reducing the Sobel-X response.

---

## 7. Column-Wise Median Projection

### What it does
For each vertical measurement band $[y_{start}, y_{end})$, the 2D gradient array is collapsed along the vertical axis into a 1D horizontal vector by evaluating the median value of every column:
$$S(x) = \operatorname{median}_{y \in [y_{start}, y_{end})} G_x(y, x)$$

### Why it is used (Rationale & Alternatives)
- **Why Median instead of Mean?** In classical statistics, the arithmetic mean has a breakdown point of $0\%$: a single extreme outlier can distort the mean arbitrarily. In industrial machine vision, reflective cylinders suffer from isolated specular highlights (bright glints) and surface scratches. These glints affect only a few rows. Because the median has a **$50\%$ breakdown point**, any artifact affecting fewer than half the rows has **zero impact** on the resulting 1D profile.
- **Dimensional Reduction:** Reduces a 2D image analysis problem of size $(50 \times 400)$ into a fast, highly stable 1D peak-finding problem of size $400$.

### Implementation Reference
- **File:** [`app/core/diameter_engine.py`](file:///Users/neelakandan/Machine_vision/app/core/diameter_engine.py)
- **Function:** `BoundaryEdgeLocator.locate()`

### Limitations
- Requires the cylinder boundary to be perfectly straight and vertical. If the cylinder is tilted, the edge boundary shifts horizontally across different rows. Taking the median across a tilted edge blurs the 1D peak, reducing peak sharpness.

---

## 8. Inward Outer-Edge Localization

### What it does
Splits the 1D projected gradient vector at its midpoint $x_{mid} = W / 2$ into Left and Right halves:
- **Left half:** Scans from the outside inward: $x = 1, 2, \dots, x_{mid}-1$.
- **Right half:** Scans from the outside inward: $x = W-2, W-3, \dots, x_{mid}$.
Finds the first local maximum satisfying $S(x) \ge S(x-1)$ and $S(x) \ge S(x+1)$ that exceeds an adaptive noise threshold:
$$\text{Threshold} = \max\left(2.0, \; \operatorname{median}(S) + 0.5 \cdot \operatorname{std}(S), \; 0.15 \cdot \max(S)\right)$$

### Why it is used (Rationale & Alternatives)
- **Why scan from outside inward?** Stepped cylinders often have internal shoulder edges, chamfer transitions, or central keyway reflections inside the silhouette. If the algorithm simply took `argmax(S)`, an intense internal reflection could outscore the outer boundary. Scanning from the outside inward guarantees that the first significant peak encountered is the true outer physical silhouette boundary.
- **Fallback Guard:** If the outer peak's strength is $< 50\%$ of the maximum peak in that half, it is treated as an external dust particle or shadow artifact, and the algorithm falls back to the global maximum peak.

### Implementation Reference
- **File:** [`app/core/diameter_engine.py`](file:///Users/neelakandan/Machine_vision/app/core/diameter_engine.py)
- **Function:** `BoundaryEdgeLocator._find_outer_edge()`

### Limitations
- If a heavy external fixture clamp or wire encroaches into the measurement band outside the cylinder, the inward scan will lock onto the clamp edge instead of the cylinder.

---

## 9. Three-Point Parabolic Subpixel Interpolation

### What it does
Given an integer discrete peak index $x^*$ where $S(x^*) \ge S(x^* \pm 1)$, fits a continuous parabola through the three points $(x^*-1, S_{-1})$, $(x^*, S_0)$, and $(x^*+1, S_{+1})$, and computes the analytical maximum offset $\delta^* \in [-0.5, 0.5]$:
$$\delta^* = \frac{1}{2} \cdot \frac{S_{-1} - S_{+1}}{S_{-1} - 2S_0 + S_{+1}}$$
$$x_{\text{subpixel}} = x^* + \delta^*$$

### Why it is used (Rationale & Alternatives)
- **Why Parabolic Interpolation?** Camera sensor pixels sample a continuous physical optical blur (Point Spread Function convolved with the edge step). Discrete pixel indices jump in integer increments of $1.0\text{ px}$ ($\approx 100\ \mu\text{m}$). Parabolic interpolation estimates the true continuous peak with a theoretical precision of $0.02 - 0.05\text{ pixels}$ ($2 - 5\ \mu\text{m}$).
- **Why not Gaussian fitting?** For symmetric peaks, parabolic interpolation is computationally negligible (only 4 subtractions and 1 division) and yields nearly identical subpixel estimates to logarithmic Gaussian peak fitting without requiring costly `log` or nonlinear curve fitting routines.

### Implementation Reference
- **File:** [`app/core/diameter_engine.py`](file:///Users/neelakandan/Machine_vision/app/core/diameter_engine.py)
- **Function:** `BoundaryEdgeLocator._subpixel_peak()`

### Limitations
- Assumes the discrete gradient peak is locally symmetric. If an edge has asymmetric illumination (e.g., backlight falloff on one side), parabolic fitting can exhibit slight systematic subpixel bias (known in metrology as pixel locking / peak bias).

---

## 10. Linear Scale Calibration & Symmetric Tolerance Checking

### What it does
Converts the pixel diameter $D_{px} = x_{\text{right}} - x_{\text{left}}$ into millimeters via a scale factor $s_{cal}$:
$$D_{mm} = D_{px} \cdot s_{cal}$$
Then checks whether $D_{mm}$ falls within bilateral symmetric tolerance limits:
$$\text{PASS} \iff D_{nom} - t \le D_{mm} \le D_{nom} + t$$

### Why it is used (Rationale & Alternatives)
- **Architecture Separation:** Keeps physical calibration models (`CalibrationModel`) strictly decoupled from image processing algorithms. If the calibration changes from a simple linear scale to an undistorted polynomial camera model, zero lines of code in `diameter_engine.py` need to change.
- **Metrological Discipline:** If either the calibration scale or the tolerance specification is missing, the system outputs explicit `NOT_CONFIGURED` statuses rather than fabricating guesses.

### Implementation Reference
- **Files:** [`app/core/calibration.py`](file:///Users/neelakandan/Machine_vision/app/core/calibration.py), [`app/core/tolerance_checker.py`](file:///Users/neelakandan/Machine_vision/app/core/tolerance_checker.py)
- **Functions:** `LinearScaleCalibration.pixels_to_mm()`, `check_tolerance()`

### Limitations
- A constant linear scale assumes zero lens distortion (no barrel/pincushion distortion) and perfect optical perpendicularity (no perspective foreshortening).
