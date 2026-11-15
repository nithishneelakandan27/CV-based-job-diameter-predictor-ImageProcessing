# Technical Viva Questions & Answers: Project Defense Guide

This document contains 42 comprehensive technical viva and interview questions with detailed, defensible answers designed for final-year engineering project examinations, technical interviews, and industry audits.

---

## Section 1: Basic Computer Vision & System Design

### Q1: Why did you choose classical computer vision instead of Deep Learning (YOLO, Mask R-CNN, Segment Anything)?
> **Answer:**  
> Classical computer vision was chosen for three fundamental industrial metrology reasons:
> 1. **Determinism and Explainability:** In dimensional quality control, an inspection decision must be 100% mathematically traceable. Deep learning segmentation masks have stochastic, fuzzy boundaries governed by millions of unexplainable weights, making them unsuitable for micrometer-level metrology.
> 2. **Subpixel Metrology Capability:** Neural network bounding boxes and masks operate at integer pixel resolutions. Classical gradient differential operators (Sobel, parabolic interpolation) permit continuous subpixel edge localization down to fractions of a pixel ($0.05\text{ px}$).
> 3. **Computational Efficiency:** Our classical pipeline processes 9 components across a 5-megapixel ($2448 \times 2048$) uncompressed image in **$86\text{ ms}$ on a standard CPU** without requiring expensive, power-hungry industrial GPUs.

### Q2: What is the primary objective of this project?
> **Answer:**  
> The objective is automated dimensional inspection of multi-diameter stepped cylindrical components held in a multi-part fixture. The system automatically segments multiple components from a single frame, aligns them, and measures four distinct stepped diameters ($D_1, D_2, D_3, D_4$) with subpixel precision, comparing them against engineering drawings to issue automated PASS/FAIL quality verdicts.

### Q3: Why is downsampling or resizing forbidden in dimensional metrology?
> **Answer:**  
> Resizing an image via bilinear, bicubic, or area interpolation computes weighted averages of neighboring pixels. This synthetically creates artificial gray levels and smooths out sharp edge transitions, permanently corrupting the native optical scale ($mm/\text{pixel}$) and introducing subpixel position bias. In metrology, images must be processed at raw sensor resolution.

### Q4: What type of lighting is used, and why is lighting so critical for reflective metals?
> **Answer:**  
> The system utilizes a **diffuse red LED backlight** (telecentric or diffuse illumination panel) located behind the parts. Machined steel and aluminum components have high surface specularity; top-down or front lighting causes harsh reflection glints, shadows, and internal glare that overwhelm edge detectors. Backlighting turns the problem into high-contrast silhouette metrology, where the component appears as a crisp dark silhouette against a bright background.

---

## Section 2: Image Preprocessing

### Q5: Why did you extract only the red channel instead of converting the image to standard grayscale?
> **Answer:**  
> The backlight emits monochromatic red light ($\approx 630\text{ nm}$). In the camera sensor's Bayer filter, the red pixels receive maximum photon flux (intensities up to $100 - 130$ counts), while green and blue pixels receive negligible stray light ($< 3$ counts). Converting to standard luminance grayscale uses the ITU-R formula:
> $$Y = 0.299R + 0.587G + 0.114B$$
> Weighting green by $58.7\%$ and blue by $11.4\%$ severely attenuates the backlight intensity by roughly $70\%$, unnecessarily crushing the image dynamic range. Directly isolating `image[:, :, 2]` preserves the maximum signal-to-noise ratio.

### Q6: What is the purpose of Gaussian smoothing before computing gradients?
> **Answer:**  
> Numerical differentiation (gradient calculation) is an inherently high-pass filtering operation that amplifies high-frequency noise. Sensor shot noise and thermal noise create localized single-pixel intensity spikes whose derivatives look like sharp false edges. Applying a $5 \times 5$ Gaussian filter acts as a low-pass filter, smoothing high-frequency noise while preserving macroscopic edge steps.

### Q7: Why use a $5 \times 5$ kernel rather than a larger kernel like $15 \times 15$ or $21 \times 21$?
> **Answer:**  
> A larger kernel has a larger spatial standard deviation $\sigma$, which excessively blurs the physical edge boundary. Over-blurring reduces the gradient peak magnitude, flattens the peak, and risks merging adjacent step transitions (such as a corner chamfer into a bearing diameter), degrading localization accuracy. A $5 \times 5$ kernel ($\sigma \approx 0.85$) provides the optimal balance between noise suppression and boundary sharpness.

### Q8: Does Gaussian blur shift the physical location of a straight edge?
> **Answer:**  
> Mathematically, no. For a linear step edge, Gaussian convolution is symmetric. The maximum of the derivative of a convolved symmetric step edge remains exactly at the inflection point of the original edge.

---

## Section 3: Component Segmentation & Morphological Operations

### Q9: Why did you choose Otsu's thresholding instead of adaptive local thresholding (`cv2.adaptiveThreshold`)?
> **Answer:**  
> Adaptive local thresholding computes thresholds in small local windows (e.g., $15 \times 15$ pixels). Inside the solid dark interior of a cylindrical component, all pixels have identical low intensity, meaning local variance is near zero. Adaptive thresholding amplifies camera sensor noise inside the component body, creating false edges and hollow silhouettes. Otsu is a global statistical technique designed specifically for bimodal images where a clear foreground/background separation exists.

### Q10: What is the mathematical objective of Otsu's thresholding?
> **Answer:**  
> Otsu's method exhaustively searches for the threshold $t^*$ that maximizes the **between-class variance** $\sigma_b^2(t)$:
> $$\sigma_b^2(t) = \omega_0(t) \omega_1(t) [\mu_0(t) - \mu_1(t)]^2$$
> Maximizing the distance between class means while weighting by class probabilities is mathematically equivalent to minimizing the intra-class variance (the spread of gray levels within the component and background classes).

### Q11: What was the root cause of the segmentation failure in Phase 2.0?
> **Answer:**  
> The original implementation hard-clamped the Otsu threshold: `threshold = min(otsu, 35)`. In Images 3 and 4, the backlight was brighter (mean red $\approx 65$ vs $51$), shifting the true bimodal valley to intensity $50$. Clamping the threshold at $35$ classified peripheral component boundary pixels (intensities $36 - 50$) as background, artificially shrinking the silhouette height by $31\text{ pixels}$ and causing an erroneous $26.6\text{ pixel}$ ($2.66\text{ mm}$) drop on $D_1$.

### Q12: How does the Phase 2.1 adaptive threshold ceiling work?
> **Answer:**  
> Phase 2.1 computes a dynamic relative ceiling based on the 99th percentile of the red channel:
> $$T_{final} = \min\left(T_{Otsu}, \; \min(\lfloor 0.55 \cdot P_{99} \rfloor, \; 90)\right)$$
> Across our optical dataset, the true valley consistently sits at $\approx 51.5\%$ of $P_{99}$. The $55\%$ ceiling provides safety margin without clipping, while capping the threshold if lighting shifts or non-bimodal noise enters the frame.

### Q13: Why is morphological closing applied after thresholding?
> **Answer:**  
> Morphological closing consists of a dilation followed by an erosion: $(A \oplus B) \ominus B$. It bridges small internal holes and gaps smaller than the $9 \times 9$ structuring element (caused by surface reflections, dust specks, or oil spots) and smooths boundary noise without altering the macroscopic size of the component.

### Q14: How does the system filter out fixture clamps and background shadows?
> **Answer:**  
> `cv2.connectedComponentsWithStats` extracts morphological properties for every blob. Four strict geometric criteria filter out non-component objects:
> 1. $\text{Area} \ge 50{,}000\text{ px}$ (rejects dust, screws, and small shadows).
> 2. $\text{Width} \ge 400\text{ px}$ and $\text{Height} \ge 200\text{ px}$ (enforces component scale).
> 3. $\text{Width} > \text{Height}$ (since components lie horizontally, this rejects vertical fixture bars and wire shadows).

---

## Section 4: Edge Detection & Directional Gradients

### Q15: Why use Sobel-X instead of the Canny edge detector?
> **Answer:**  
> Canny applies Non-Maximum Suppression (NMS) and hysteresis thresholding, converting continuous gradients into a 1-pixel-wide binary edge map (0 or 255). A binary edge discards intensity gradient distribution, making continuous subpixel curve fitting impossible. Sobel-X retains floating-point gradient magnitudes ($G_x \in \mathbb{R}$), which form smooth bell-shaped curves suitable for subpixel interpolation.

### Q16: Why use directional Sobel-X instead of isotropic gradient magnitude $\sqrt{G_x^2 + G_y^2}$?
> **Answer:**  
> The component is oriented such that its cylindrical axis is vertical. The outer physical boundaries are strictly vertical lines, producing gradients exclusively in the horizontal ($x$) direction. Sobel-X detects these vertical boundaries while completely ignoring horizontal transitions, step fillets, chamfers, and circumferential lathe turning marks.

### Q17: What are the coefficients of the Sobel-X operator, and why do they sum to zero?
> **Answer:**  
> The $3 \times 3$ Sobel-X kernel is:
> $$K_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix}$$
> The sum of coefficients is $(-1+1) + (-2+2) + (-1+1) = 0$. In a uniform region where all pixel values are equal ($I(x, y) = c$), convolving with $K_x$ yields $c \cdot \sum K_x = 0$. A derivative operator must produce zero response in flat, constant regions.

---

## Section 5: Subpixel Localization & Median Projection

### Q18: What is the significance of subpixel edge localization?
> **Answer:**  
> At an optical scale where $1\text{ pixel} = 100\ \mu\text{m}$, integer pixel coordinates can only measure in discrete $100\ \mu\text{m}$ steps. Many industrial tolerances are $\pm 50\ \mu\text{m}$ or tighter. Subpixel localization estimates the continuous mathematical center of the optical edge blur, providing edge repeatability down to $0.05\text{ pixels}$ ($5\ \mu\text{m}$), enabling bilateral tolerance verification.

### Q19: Does subpixel localization guarantee high measurement accuracy?
> **Answer:**  
> **No.** This is a critical distinction in metrology. Subpixel interpolation provides high **precision (repeatability)**, meaning repeated measurements yield nearly identical numbers ($\sigma \approx 0.1\text{ px}$). However, it does not guarantee **accuracy (trueness)**. If the camera has uncorrected lens distortion, optical tilt, or an uncalibrated scale factor, the subpixel measurements will be repeatable but systematically biased.

### Q20: How does three-point parabolic interpolation derive the subpixel peak?
> **Answer:**  
> Around the discrete integer maximum $S(x^*)$, the continuous profile is modeled locally as a parabola: $f(\delta) = a\delta^2 + b\delta + c$. Sampling at discrete offsets $\delta \in \{-1, 0, +1\}$ yields three equations with three unknowns. Solving $\frac{df}{d\delta} = 2a\delta + b = 0$ yields the vertex offset:
> $$\delta^* = \frac{1}{2} \cdot \frac{S_{-1} - S_{+1}}{S_{-1} - 2S_0 + S_{+1}}$$
> The continuous coordinate is $x_{\text{subpixel}} = x^* + \delta^*$, where $\delta^* \in [-0.5, 0.5]$.

### Q21: Why use column-wise median projection instead of the arithmetic mean?
> **Answer:**  
> Across a vertical band of 50 rows, metallic cylinders frequently exhibit tiny specular glints, dust particles, or micro-scratches. The arithmetic mean has a breakdown point of $0\%$: a single bright reflection spike in one row skews the mean across all rows. The median has a **$50\%$ breakdown point**: as long as a scratch or glare point affects fewer than half the rows in the band, it has zero mathematical effect on the median profile.

### Q22: What happens if a component in the fixture is physically tilted by $2^\circ$?
> **Answer:**  
> If the cylinder axis is tilted by $\theta = 2^\circ$, its vertical edge sweeps horizontally across $\Delta x = H \cdot \tan(2^\circ)$ pixels over the height of the band. Projecting across rows convolves the sharp edge profile with a boxcar filter of width $\Delta x$, broadening and flattening the gradient peak. This reduces peak sharpness and slightly degrades subpixel precision.

---

## Section 6: Diameter Measurement & Geometry

### Q23: Why does the system rotate the component $90^\circ$ counter-clockwise?
> **Answer:**  
> In the physical fixture, jobs lie horizontally with their large flange on the left. Rotating $90^\circ$ CCW maps the left side to the bottom, standardizing the component so that its longitudinal axis is vertical, with $D_1$ (flange) at the bottom and $D_4$ at the top. This aligns the cylinder diameter with horizontal pixel rows.

### Q24: What distance formula is used to compute diameter?
> **Answer:**  
> Because the cylinder is aligned upright and diameters are sampled along horizontal bands, diameter is the 1D difference between right and left subpixel edge coordinates:
> $$D_{px} = x_{\text{subpixel, right}} - x_{\text{subpixel, left}}$$
> It is **not** a 2D Euclidean distance ($\sqrt{\Delta x^2 + \Delta y^2}$) because left and right edges are evaluated across the same horizontal band.

### Q25: Why are the four diameter bands defined as dimensionless fractions rather than pixel rows?
> **Answer:**  
> Hardcoding pixel rows (e.g., rows 100 to 150) breaks if the camera distance changes or if component length varies slightly. Defining bands as fractions of normalized height (e.g., $D_1 \in [0.84, 0.92]$) keeps the specification scale-independent and drawing-driven, ensuring bands adapt to the segmented component bounds.

### Q26: Why are $D_3$ and $D_4$ measured independently even if they have the same nominal dimension?
> **Answer:**  
> $D_3$ and $D_4$ represent distinct physical axial positions on the workpiece. If a lathe cutting tool chipped or experienced chatter during machining of section $D_4$, $D_3$ might pass while $D_4$ fails. Measuring them as independent `DiameterSpec` entries allows local defects to be isolated.

---

## Section 7: Calibration, Tolerances & Decision Logic

### Q27: How does the system convert pixel measurements to millimeters?
> **Answer:**  
> Via an explicit `LinearScaleCalibration` model:
> $$D_{mm} = D_{px} \cdot s_{cal}$$
> The scale factor $s_{cal}$ represents physical millimeters per pixel. If calibration is absent, the pipeline preserves pixel values and sets $D_{mm} = \text{None}$ with status `NOT_CONFIGURED`.

### Q28: How can you establish a valid physical calibration factor in production?
> **Answer:**  
> By placing a traceable reference artifact (such as a Class XX certified optical gauge pin with diameter certified to $\pm 0.5\ \mu\text{m}$) into the fixture plane, measuring its diameter in pixels $D_{px}^*$, and calculating:
> $$s_{cal} = \frac{D_{\text{true}}}{D_{px}^*} \quad [\text{mm/pixel}]$$

### Q29: What is the tolerance evaluation rule?
> **Answer:**  
> Bilateral symmetric tolerance evaluation:
> $$\text{PASS} \iff D_{nom} - t \le D_{mm} \le D_{nom} + t$$
> The check is inclusive at boundaries. A component passes if and only if all four diameters pass:
> $$\text{JobStatus} = \text{PASS} \iff (D_1 = \text{PASS}) \land (D_2 = \text{PASS}) \land (D_3 = \text{PASS}) \land (D_4 = \text{PASS})$$

### Q30: What is the hierarchy of overall job statuses?
> **Answer:**  
> The system evaluates status precedence strictly:
> $$\text{MEASUREMENT\_FAILED} > \text{NOT\_CONFIGURED} > \text{FAIL} > \text{PASS}$$
> If any diameter fails to measure (e.g. edge not found), the job is `MEASUREMENT_FAILED`. If any diameter lacks calibration or drawing limits, it is `NOT_CONFIGURED`. If any calibrated diameter violates limits, it is `FAIL`.

---

## Section 8: Testing & Verification

### Q31: How many automated tests exist, and what do they verify?
> **Answer:**  
> There are **46 automated tests** (all passing):
> - `test_measurement.py` (7 tests): Geometry arithmetic, subpixel coordinate preservation, scale validation.
> - `test_tolerance_checker.py` (4 tests): Tolerance boundary limits, inclusive bounds, invalid inputs.
> - `test_four_diameter.py` (11 tests): Stepped cylinder synthetic tests, band isolation, status aggregation.
> - `test_bmp_inspection.py` (4 tests): 3×3 grid layout, deterministic naming, CSV/JSON schema integrity.
> - `test_adaptive_segmentation.py` (20 tests): Adaptive thresholding, illumination scaling, ROI clipping guards.

### Q32: What is the difference between part-to-part variation and measurement repeatability?
> **Answer:**  
> - **Part-to-Part Variation:** The dimensional spread among different physical components in one image ($\sigma \approx 1.4\text{ px}$ in our project). It reflects manufacturing tolerances in the workshop.
> - **Measurement Repeatability:** The variation when measuring the **exact same physical component** across repeated image captures ($\sigma \approx 0.12 - 0.38\text{ px}$). It reflects vision system precision.

---

## Section 9: Metrological Limitations & Optical Physics

### Q33: Why do center jobs measure larger than corner jobs by $\approx 4.5\text{ pixels}$?
> **Answer:**  
> The center component (`job_r02_c02`) measures $D_1 \approx 388.9\text{ px}$, while corner components measure $\approx 384.4\text{ px}$. This discrepancy cannot be attributed solely to lens distortion without experimental proof. It is confounded by:
> 1. True part-to-part manufacturing variations among the 9 parts.
> 2. Perspective foreshortening (varying distance to optical axis).
> 3. Optical radial lens distortion (barrel/pincushion distortion).
> 4. Non-uniform diffuse backlight emission angles at image corners.

### Q34: What experiment would isolate lens distortion from physical part variation?
> **Answer:**  
> A **position-permutation experiment**: take one single reference component, place it in the center slot (`job_r02_c02`), measure it, then move the exact same part to the corner slot (`job_r01_c01`) and measure it again. Any dimensional difference on the same physical part proves optical field distortion.

### Q35: When is camera intrinsic calibration mandatory?
> **Answer:**  
> Whenever standard entocentric C-mount lenses are used across wide fields of view ($2448 \times 2048$) requiring tolerances tighter than $\pm 0.1\text{ mm}$. Calibration estimates the focal lengths $(f_x, f_y)$, optical center $(c_x, c_y)$, and radial/tangential distortion coefficients $(k_1, k_2, p_1, p_2)$ using a ChArUco target.

### Q36: Why can a telecentric lens eliminate perspective distortion?
> **Answer:**  
> Standard lenses have an angular field of view: objects further away appear smaller, and rays enter at an angle. Telecentric lenses have an optical stop that accepts only parallel chief rays. Magnification is completely independent of distance over the depth of field, eliminating perspective error and parallax.

---

## Section 10: Future Enhancements & Industrial Deployment

### Q37: How would you handle an empty slot in the fixture?
> **Answer:**  
> In the current implementation, `_group_rows()` assumes all 9 slots contain parts. To support partial loading, the system should be upgraded to template-match known fixture pocket fiducials, defining 9 static bounding search windows so each slot is verified independently.

### Q38: How would you improve part tilt compensation?
> **Answer:**  
> Instead of collapsing 2D bands into a 1D median projection, the system can extract subpixel boundary points along individual scanlines and fit two parallel lines using RANSAC or Total Least Squares (PCA axis alignment). The true diameter is the orthogonal perpendicular distance between the parallel lines:
> $$D = |d_{right} - d_{left}|$$

### Q39: How would you scale the system to 50 jobs per second?
> **Answer:**  
> Current single-core CPU execution requires $\approx 86\text{ ms}$ for 9 jobs ($\approx 9.6\text{ ms/job}$). By dispatching the 9 independent ROI measurement tasks across multiple CPU cores using `concurrent.futures.ProcessPoolExecutor` or OpenMP, 50 jobs can be processed in parallel in under $300\text{ ms}$.

### Q40: What happens if ambient white room light contaminates the red backlight?
> **Answer:**  
> Ambient white light reflects off the cylindrical metal surface, raising foreground pixel counts from $\approx 5$ to $30 - 40$. In Phase 2.1 stress testing, this reduced contrast and caused the $0.55 \cdot P_{99}$ ceiling to clamp prematurely. To eliminate this in factory deployment, a **narrowband optical bandpass filter** ($630\text{ nm} \pm 10\text{ nm}$) should be attached to the camera lens to block all ambient room light.

### Q41: How would you perform Gauge R&R (Repeatability & Reproducibility)?
> **Answer:**  
> Using standard AIAG metrology protocols: take 10 parts, have 3 different operators load and unload the parts into the fixture across 3 separate trials, and compute ANOVA variance components. A measurement system is acceptable if $\%GRR < 10\%$ of process tolerance.

### Q42: What is the most important lesson learned from this project?
> **Answer:**  
> In dimensional computer vision metrology, **algorithms cannot compensate for poor optical physics**. High subpixel repeatability does not equal accuracy. Reliable industrial inspection requires controlled backlighting, rigid part fixturing, validated calibration standards, and strict separation between image processing geometry and quality control decision logic.
