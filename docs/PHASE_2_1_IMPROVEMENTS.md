# Phase 2.1 Technical Report: Adaptive Segmentation & ROI Reliability

## 1. The Original Problem & Root Cause

### 1.1 The Original Implementation
In the original version of [`app/core/job_detector.py`](file:///Users/neelakandan/Machine_vision/app/core/job_detector.py), the red-channel segmentation threshold was computed using a hard scalar ceiling:

```python
otsu_threshold, _ = cv2.threshold(red, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
threshold = min(int(round(otsu_threshold)), self.max_red_value)  # max_red_value = 35
silhouette = cv2.inRange(red, 0, threshold)
```

### 1.2 Why the Fixed Clamp Failed
The four industrial BMP images in `data/input/` were captured under slightly varying diffuse red backlighting conditions:

| Image File | Mean Red | Max Red | Red $P_{99}$ | True Otsu Threshold | Clamped Threshold ($T \le 35$) | Deficit |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Image_20260826172434771.bmp` | $51.2$ | $98$ | $76$ | $39$ | $35$ | $-4$ counts |
| `Image_20260826172506544.bmp` | $38.8$ | $75$ | $58$ | $30$ | $30$ | $0$ (unaffected) |
| `Image_20260826172520410.bmp` | $64.9$ | $125$ | $96$ | $50$ | $35$ | **$-15$ counts** |
| `Image_20260826172941994.bmp` | $65.5$ | $126$ | $97$ | $50$ | $35$ | **$-15$ counts** |

On Images 3 and 4, the backlight was brighter (mean red $\approx 65$, peak $\approx 125$). The true bimodal separation valley between dark components and bright backlight sat at intensity **$50$**.

Because the threshold was clamped at $35$, any pixel on the component boundary with a red value between $36$ and $50$ was classified as background. 

### 1.3 The Cascading Measurement Failure
1. The horizontal components had an outer physical flange height of $\approx 383\text{ px}$.
2. Under the $T=35$ clamp on Images 3 and 4, the segmented silhouette height collapsed from $\approx 406\text{ px}$ down to $375\text{ px}$ (a vertical loss of $31\text{ pixels}$).
3. When `HorizontalFlangeLeftJobNormalizer` rotated the sub-image $90^\circ$ counter-clockwise, the source image height became the **horizontal width** of the normalized ROI.
4. When `BoundaryEdgeLocator._find_outer_edge()` scanned from the right edge inward to find the $D_1$ flange boundary, the true outer boundary at $x \approx 393\text{ px}$ was completely clipped outside the image window. The detector locked onto an internal step edge at $x \approx 355 - 369\text{ px}$.
5. **The Measurement Error:** $D_1$ measured $346.84\text{ px}$ instead of $385.09\text{ px}$—an artificial drop of **$38.25\text{ pixels}$ ($\approx 3.82\text{ mm}$)** on `job_r03_c01`!

---

## 2. The Phase 2.1 Adaptive Segmentation Design

### 2.1 The Mathematical Formulation
To solve this without substituting another arbitrary constant, Phase 2.1 implemented an adaptive threshold with a physics-grounded relative safety ceiling:

$$T_{final} = \min\left(T_{Otsu}, \; \min(\lfloor 0.55 \cdot P_{99} \rfloor, \; 90)\right)$$

Where:
- $T_{Otsu}$ is computed via OpenCV's `cv2.THRESH_BINARY + cv2.THRESH_OTSU` on the red channel.
- $P_{99} = \operatorname{percentile}_{99}(\text{Red Channel})$ is the 99th percentile intensity (robust against dead saturated pixels).
- `red_ceiling_fraction = 0.55` sets the relative ceiling at $55\%$ of the backlight peak.
- $90$ is an absolute upper bound ensuring the threshold never reaches ambient room lighting.

### 2.2 Why the Relative Ceiling is Used
- **Why not unconstrained Otsu?** If an image is captured with the backlight turned off or severely degraded, Otsu will still split the unimodal dark image in half, segmenting random dark noise into giant phantom objects. The ceiling acts as a protective guard.
- **Why $0.55$?** Across all four fixture images, the ratio of the true Otsu threshold to the backlight peak $P_{99}$ is remarkably constant:
  - Image 1: $39 / 76 = 0.513$
  - Image 2: $30 / 58 = 0.517$
  - Image 3: $50 / 96 = 0.521$
  - Image 4: $50 / 97 = 0.515$
  The bimodal valley consistently sits at $\approx 51.5\%$ of $P_{99}$. Setting the ceiling at $55\%$ provides a $+3.5\%$ operating buffer while preventing the threshold from floating into the background mode.

---

## 3. Empirical Verification & Measurement Recovery

### 3.1 Quantitative D1 Measurement Comparison
The table below shows the exact D1 diameter measurement (in pixels) for all nine jobs across all four captures before and after Phase 2.1:

| Job Grid ID | Image 1 (Before) | Image 1 (After) | Image 2 (Before) | Image 2 (After) | Image 3 (Before) | Image 3 (After) | Image 4 (Before) | Image 4 (After) | StdDev (Before) | **StdDev (After)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `job_r01_c01` | 384.38 | 384.31 | 385.31 | 385.31 | **365.49** | **384.08** | **357.72** | **383.81** | $11.94\text{ px}$ | **$0.56\text{ px}$** |
| `job_r01_c02` | 387.04 | 387.04 | 387.21 | 387.21 | **386.04** | **387.07** | **380.43** | **386.86** | $2.78\text{ px}$ | **$0.12\text{ px}$** |
| `job_r01_c03` | 387.91 | 387.91 | 387.84 | 387.84 | 387.16 | 387.63 | 387.25 | 387.54 | $0.34\text{ px}$ | **$0.15\text{ px}$** |
| `job_r02_c01` | 385.72 | 385.72 | 386.47 | 386.47 | **360.63** | **386.32** | **368.90** | **386.27** | $11.06\text{ px}$ | **$0.28\text{ px}$** |
| `job_r02_c02` | 388.95 | 388.95 | 389.10 | 389.10 | **379.02** | **389.03** | **367.56** | **388.50** | $8.85\text{ px}$ | **$0.24\text{ px}$** |
| `job_r02_c03` | 387.90 | 387.89 | 388.65 | 388.65 | **378.55** | **388.19** | 385.44 | 387.82 | $3.98\text{ px}$ | **$0.32\text{ px}$** |
| `job_r03_c01` | 383.41 | 384.80 | 385.09 | 385.09 | **346.84** | **384.70** | **363.19** | **384.54** | $15.73\text{ px}$ | **$0.20\text{ px}$** |
| `job_r03_c02` | 386.98 | 386.91 | 386.85 | 386.85 | **355.80** | **387.07** | **358.87** | **386.53** | $14.83\text{ px}$ | **$0.20\text{ px}$** |
| `job_r03_c03` | 386.84 | 386.86 | 387.21 | 387.21 | **384.10** | **386.62** | **371.59** | **386.90** | $6.38\text{ px}$ | **$0.21\text{ px}$** |

### 3.2 Key Findings
1. **Full Recovery:** The $20 - 38\text{ pixel}$ clipping deficits on Images 3 and 4 were completely eliminated.
2. **Subpixel Repeatability:** Across all nine jobs, the cross-capture measurement repeatability standard deviation dropped from up to $15.73\text{ px}$ down to **$0.12 - 0.56\text{ pixels}$** ($\approx 12 - 56\ \mu\text{m}$).
3. **Zero Regression on D2, D3, D4:** Shaft diameters $D_2, D_3, D_4$ were unaffected, maintaining their baseline repeatability ($\sigma \le 0.38\text{ px}$).

---

## 4. Stress Testing & Failure Modes (Honest Engineering Critique)

To establish the limits of this solution, artificial stress testing was conducted:

| Test Condition | Otsu $T$ | Peak $P_{99}$ | Ceiling | Selected $T$ | Result | Analysis |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Gain $0.5\times$ (Dimmed)** | $19$ | $38$ | $20$ | $19$ | **Pass** ($9/9$) | Proportional downscaling works. |
| **Gain $2.0\times$ (Bright)** | $78$ | $152$ | $83$ | $78$ | **Pass** ($9/9$) | Otsu scales up; ceiling permits it. |
| **Low Contrast ($0.6\times + 30$)** | $53$ | $75$ | **$41$** | **$41$** | **Failed (Clipped)** | Ambient white offset raised dark levels to 30. Otsu was 53, but ceiling clamped at 41, re-inducing flange clipping! |
| **Weak Separation ($+40$ offset)** | $57$ | $78$ | **$42$** | **$42$** | **Failed (0 jobs)** | Clamp caused components to fail area thresholds. |

### Engineering Takeaway
The Phase 2.1 adaptive ceiling is **not universally illumination-invariant**. It is robust against **linear scaling of the backlight intensity** (dimming/brightening the red LEDs), but it remains vulnerable to **ambient white light contamination** (e.g., room lights reflecting off the component body), which shifts the foreground distribution upward relative to the background peak.
