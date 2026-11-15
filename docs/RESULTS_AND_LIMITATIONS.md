# Experimental Results, Performance Benchmarks & Technical Limitations

## 1. Verified Dataset & Execution Inventory

The empirical findings documented here derive exclusively from executing the inspection pipeline on the actual project files:

| Category | Specification | Details |
| :--- | :--- | :--- |
| **Industrial BMP Dataset** | 4 files (`Image_202608261724...bmp` to `...41994.bmp`) | $2448 \times 2048$ resolution, uncompressed BGR, $14.34\text{ MB}$ each |
| **Exploratory Phone Dataset**| 9 files (JPEG/PNG) | $960 \times 1280$ to $1200 \times 1600$, ambient indoor lighting |
| **Detected Components** | Exactly **9 jobs per image** in a $3 \times 3$ grid | 100% detection rate across all four industrial BMP captures |
| **Total Jobs Analyzed** | **36 component instances** across 4 captures | Identifiers: `job_r01_c01` through `job_r03_c03` |
| **Diameter Measurements** | **4 diameters per component** ($D_1, D_2, D_3, D_4$) | **36 measurements per image**, **144 total measurements** |
| **Automated Test Suite** | **46 passing tests** (28 subtests) | Executed via Pytest in $3.78\text{ seconds}$ |

---

## 2. Before-and-After Segmentation & Measurement Results

### 2.1 The D1 Recovery Benchmark
Prior to Phase 2.1, the hard clamp at threshold 35 caused significant bounding box truncation on Images 3 and 4. The table below illustrates the verified before-and-after measurements:

| Image File | Backlight State | ROI Height (Before) | ROI Height (After) | Mean $D_1$ (Before) | Mean $D_1$ (After) | Intra-Image $D_1$ Spread ($\sigma$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Image_...1724.bmp` | Baseline Red | $389 - 410\text{ px}$ | $400 - 413\text{ px}$ | $386.57\text{ px}$ | **$386.71\text{ px}$** | $1.43\text{ px}$ |
| `Image_...06544.bmp` | Dimmer Red | $401 - 412\text{ px}$ | $401 - 412\text{ px}$ | $387.08\text{ px}$ | **$387.08\text{ px}$** | $1.28\text{ px}$ |
| `Image_...20410.bmp` | Brighter Red | **$360 - 405\text{ px}$** | **$401 - 411\text{ px}$** | **$371.51\text{ px}$** | **$386.75\text{ px}$** | **$1.48\text{ px}$** (was $13.87$) |
| `Image_...41994.bmp` | Brighter Red | **$369 - 403\text{ px}$** | **$401 - 412\text{ px}$** | **$371.22\text{ px}$** | **$386.53\text{ px}$** | **$1.42\text{ px}$** (was $10.34$) |

### 2.2 Stability of $D_2, D_3, D_4$
Because $D_2, D_3, D_4$ are smaller shaft diameters located in the interior sections of the cylinder, their measurements were completely unaffected by outer flange boundary changes:
- **$D_2$:** Mean across all images remained $299.4\text{ px} \pm 0.1\text{ px}$.
- **$D_3$:** Mean across all images remained $272.3\text{ px} \pm 0.1\text{ px}$.
- **$D_4$:** Mean across all images remained $226.3\text{ px} \pm 0.1\text{ px}$.

---

## 3. High-Precision Measurement Repeatability Statistics

### Rigorous Distinction of Terms
In dimensional metrology, three terms are frequently confused but represent fundamentally different concepts:

1. **Part-to-Part Variation ($\sigma_{part} \approx 1.3 - 2.8\text{ px}$):**  
   The dimensional spread among the nine physically different parts manufactured in the machine shop. A high spread indicates part manufacturing variation, not vision system error.
2. **Measurement Repeatability ($\sigma_{repeat} \approx 0.07 - 0.38\text{ px}$):**  
   The variation of the vision system when measuring the **exact same physical component** across repeated image captures under the same fixturing. This reflects the camera noise floor, illumination stability, and algorithmic consistency.
3. **Measurement Accuracy (Currently Unverified):**  
   The difference between the vision system's measurement and the true physical dimension (traceable to a national metrology institute / SI meter). Accuracy cannot be proven without physical calibration targets and certified micrometer measurements.

### Per-Job Measurement Repeatability Across All 4 Captures
$$\sigma = \sqrt{\frac{1}{4 - 1} \sum_{k=1}^4 (D_k - \bar{D})^2}$$

| Job Identifier | Grid Position | $D_1$ Repeatability ($\sigma$) | $D_2$ Repeatability ($\sigma$) | $D_3$ Repeatability ($\sigma$) | $D_4$ Repeatability ($\sigma$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `job_r01_c01` | Top-Left Corner | $0.565\text{ px}$ | $0.207\text{ px}$ | $0.150\text{ px}$ | $0.326\text{ px}$ |
| `job_r01_c02` | Top-Center | $0.125\text{ px}$ | $0.072\text{ px}$ | $0.244\text{ px}$ | $0.192\text{ px}$ |
| `job_r01_c03` | Top-Right Corner | $0.151\text{ px}$ | $0.187\text{ px}$ | $0.088\text{ px}$ | $0.208\text{ px}$ |
| `job_r02_c01` | Middle-Left | $0.285\text{ px}$ | $0.086\text{ px}$ | $0.118\text{ px}$ | $0.092\text{ px}$ |
| `job_r02_c02` | Optical Center | $0.236\text{ px}$ | $0.206\text{ px}$ | $0.198\text{ px}$ | $0.258\text{ px}$ |
| `job_r02_c03` | Middle-Right | $0.325\text{ px}$ | $0.137\text{ px}$ | $0.151\text{ px}$ | $0.210\text{ px}$ |
| `job_r03_c01` | Bottom-Left Corner | $0.198\text{ px}$ | $0.187\text{ px}$ | $0.184\text{ px}$ | $0.149\text{ px}$ |
| `job_r03_c02` | Bottom-Center | $0.196\text{ px}$ | $0.187\text{ px}$ | $0.284\text{ px}$ | $0.376\text{ px}$ |
| `job_r03_c03` | Bottom-Right Corner| $0.211\text{ px}$ | $0.253\text{ px}$ | $0.190\text{ px}$ | $0.190\text{ px}$ |

### Repeatability Summary Table
| Diameter | Minimum $\sigma$ | **Mean Repeatability $\sigma$** | Maximum $\sigma$ | Equivalent Physical Spread (at $0.1\text{ mm/px}$) |
| :---: | :---: | :---: | :---: | :---: |
| **$D_1$** | $0.125\text{ px}$ | **$0.255\text{ px}$** | $0.565\text{ px}$ | $\approx 25.5\ \mu\text{m}$ |
| **$D_2$** | $0.072\text{ px}$ | **$0.169\text{ px}$** | $0.253\text{ px}$ | $\approx 16.9\ \mu\text{m}$ |
| **$D_3$** | $0.088\text{ px}$ | **$0.179\text{ px}$** | $0.284\text{ px}$ | $\approx 17.9\ \mu\text{m}$ |
| **$D_4$** | $0.092\text{ px}$ | **$0.222\text{ px}$** | $0.376\text{ px}$ | $\approx 22.2\ \mu\text{m}$ |

---

## 4. Processing Speed & Throughput Benchmarks

Benchmarked across multiple consecutive pipeline iterations on standard macOS CPU hardware (Python 3.13):

```text
Full Pipeline Breakdown (per 2448x2048 image, 9 jobs):
┌────────────────────────────┬──────────────┬────────────┐
│ Pipeline Stage             │ Latency (ms) │ Percentage │
├────────────────────────────┼──────────────┼────────────┤
│ 1. Image Loading (I/O)     │ 26.5 ms      │ 30.6 %     │
│ 2. Silhouette Segmentation │ 37.9 ms      │ 43.8 %     │
│ 3. ROI Extraction          │  1.5 ms      │  1.7 %     │
│ 4. CCW Normalization       │  2.7 ms      │  3.1 %     │
│ 5. D1-D4 Subpixel Metrology│ 18.0 ms      │ 20.8 %     │
├────────────────────────────┼──────────────┼────────────┤
│ Total Execution Time       │ 86.6 ms      │ 100.0 %    │
└────────────────────────────┴──────────────┴────────────┘
```

- **Per-Image Inspection Time:** $\mathbf{\approx 86.6\text{ milliseconds}}$ (over 11 full frames per second).
- **Per-Job Processing Time:** $\mathbf{\approx 9.6\text{ milliseconds per component}}$.
- **Throughput Capability:** Well within standard industrial inline cycle times ($< 1.0\text{ second}$ per assembly station).

---

## 5. Technical Limitations & Gaps (Honest Metrology Assessment)

1. **Center-to-Corner Measurement Discrepancy ($\Delta \approx 4.5\text{ px}$):**  
   The center component (`job_r02_c02`) measures $D_1 \approx 388.9\text{ px}$, whereas corner components (`job_r01_c01`, `job_r03_c01`) measure $D_1 \approx 384.4\text{ px}$.  
   *Critical Metrology Warning:* We **cannot** attribute this difference exclusively to lens distortion without position-permutation tests. The difference is confounded by:
   - Genuine part-to-part manufacturing variations among the 9 parts.
   - Non-telecentric perspective parallax.
   - Optical radial distortion (barrel/pincushion).
   - Non-uniform backlight diffusion angle at the image corners.
2. **Pending Physical Calibration:**  
   The workspace currently uses synthetic calibration scales ($0.1\text{ mm/pixel}$). No traceable gauge block or certified calibration master pin has been photographed in the fixture.
3. **Rigid Orientation Normalization:**  
   The system rotates sub-images by exactly $90^\circ$ CCW. If a component sits slightly askew in the fixture ($\theta \ge 1.5^\circ$), vertical 1D median projection convolves the tilted boundary across multiple columns, blurring the peak and slightly degrading edge sharpness.
4. **Fixture Completeness Assumption:**  
   The row-grouping logic assumes all 9 fixture slots contain a component. If a slot is left empty, the clustering logic will mislabel subsequent column assignments.
5. **Sensitivity to Ambient Front Lighting:**  
   The adaptive thresholding works reliably when backlight intensity changes proportionally. However, if strong white room lighting reflects off the front of the components, the bimodal separation collapses, causing segmentation errors.
