# Mathematical Foundations & Numerical Examples

This document explains the mathematical theory behind every equation implemented in the project. Every concept includes a step-by-step numerical example using small, easily verifiable numbers.

---

## 1. Otsu's Bimodal Thresholding

### Mathematical Formulation
Otsu's method finds a global threshold $t^* \in [0, L-1]$ that maximizes the inter-class variance $\sigma_b^2(t)$ between two classes:
- **Class 0 (Foreground / Dark Jobs):** Gray levels $[0, t]$
- **Class 1 (Background / Red Backlight):** Gray levels $[t+1, L-1]$

Let the normalized histogram probability of gray level $i$ be $p_i = \frac{n_i}{N}$.

For any candidate threshold $t$:
1. **Class Probabilities (Weights):**
   $$\omega_0(t) = \sum_{i=0}^t p_i, \quad \omega_1(t) = \sum_{i=t+1}^{L-1} p_i = 1 - \omega_0(t)$$

2. **Class Means:**
   $$\mu_0(t) = \frac{\sum_{i=0}^t i \cdot p_i}{\omega_0(t)}, \quad \mu_1(t) = \frac{\sum_{i=t+1}^{L-1} i \cdot p_i}{\omega_1(t)}$$

3. **Global Mean:**
   $$\mu_T = \sum_{i=0}^{L-1} i \cdot p_i = \omega_0(t)\mu_0(t) + \omega_1(t)\mu_1(t)$$

4. **Between-Class Variance:**
   $$\sigma_b^2(t) = \omega_0(t) \cdot (\mu_0(t) - \mu_T)^2 + \omega_1(t) \cdot (\mu_1(t) - \mu_T)^2 = \omega_0(t) \omega_1(t) \left[\mu_0(t) - \mu_1(t)\right]^2$$

The optimal threshold maximizes this variance:
$$t^* = \arg\max_t \sigma_b^2(t)$$

### Step-by-Step Numerical Example
Consider a tiny 6-pixel red channel image with pixel values:
$$\text{Pixels} = [10, \; 10, \; 10, \; 80, \; 80, \; 80]$$
Here, $N = 6$. The values are strictly bimodal (three pixels at intensity 10, three at intensity 80).

- **Global Mean:**
  $$\mu_T = \frac{10 + 10 + 10 + 80 + 80 + 80}{6} = \frac{270}{6} = 45$$

**Test Candidate $t = 10$:**
- Class 0 pixels ($\le 10$): three pixels with value 10.
  $$\omega_0 = \frac{3}{6} = 0.5, \quad \mu_0 = 10$$
- Class 1 pixels ($> 10$): three pixels with value 80.
  $$\omega_1 = \frac{3}{6} = 0.5, \quad \mu_1 = 80$$
- Between-Class Variance:
  $$\sigma_b^2(10) = \omega_0 \cdot \omega_1 \cdot (\mu_0 - \mu_1)^2 = 0.5 \cdot 0.5 \cdot (10 - 80)^2 = 0.25 \cdot (-70)^2 = 0.25 \cdot 4900 = \mathbf{1225}$$

**Test Candidate $t = 5$ (too low):**
- Class 0 ($\le 5$): 0 pixels $\implies \omega_0 = 0 \implies \sigma_b^2(5) = \mathbf{0}$.

**Test Candidate $t = 45$ (in between):**
- Class 0 ($\le 45$): three pixels at 10. $\omega_0 = 0.5, \mu_0 = 10$.
- Class 1 ($> 45$): three pixels at 80. $\omega_1 = 0.5, \mu_1 = 80$.
- $\sigma_b^2(45) = 0.25 \cdot (10 - 80)^2 = \mathbf{1225}$.

Both $t=10$ and $t=45$ yield the maximum variance $\sigma_b^2 = 1225$. Otsu selects the lower boundary ($t^* = 10$). Any threshold between 10 and 79 separates the dark jobs from the backlight.

---

## 2. Gaussian Smoothing (Spatial Low-Pass Filtering)

### Mathematical Formulation
The 2D Gaussian kernel is defined by the bivariate normal distribution:
$$G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$$
Discrete convolution filters the input image $I(x, y)$:
$$I_{smooth}(x, y) = \sum_{u=-k}^k \sum_{v=-k}^k G(u, v) \cdot I(x - u, y - v)$$
To prevent changing the image's overall brightness, the kernel is normalized so its elements sum to $1$:
$$\sum_{u=-k}^k \sum_{v=-k}^k G(u, v) = 1.0$$

### Step-by-Step Numerical Example
Consider a 1D Gaussian kernel with weights:
$$K = \begin{bmatrix} 0.25 & 0.50 & 0.25 \end{bmatrix}$$
Suppose a scanline has an isolated noise spike:
$$I = \begin{bmatrix} 20 & 20 & \mathbf{100} & 20 & 20 \end{bmatrix}$$
Filter the central spike at index 2:
$$I_{smooth}(2) = (0.25 \cdot 20) + (0.50 \cdot 100) + (0.25 \cdot 20) = 5 + 50 + 5 = \mathbf{60}$$
The noise spike is attenuated from 100 down to 60, preventing it from producing an extreme gradient false edge.

---

## 3. Sobel-X Horizontal Gradient Operator

### Mathematical Formulation
The horizontal gradient approximates the partial derivative $\frac{\partial I}{\partial x}$ by combining central difference differentiation horizontally with triangular smoothing vertically:
$$K_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix} = \underbrace{\begin{bmatrix} 1 \\ 2 \\ 1 \end{bmatrix}}_{\text{Vertical Blur}} \cdot \underbrace{\begin{bmatrix} -1 & 0 & +1 \end{bmatrix}}_{\text{Horizontal Difference}}$$
For an image neighborhood $I$, the horizontal gradient response is:
$$G_x = |K_x * I|$$

### Step-by-Step Numerical Example
Consider a vertical edge transition across three columns:
$$I = \begin{bmatrix} 10 & 50 & 90 \\ 10 & 50 & 90 \\ 10 & 50 & 90 \end{bmatrix}$$
Compute the Sobel-X response for the center pixel:
$$\begin{aligned}
G_x &= (-1 \cdot 10) + (0 \cdot 50) + (+1 \cdot 90) \\
    &\quad + (-2 \cdot 10) + (0 \cdot 50) + (+2 \cdot 90) \\
    &\quad + (-1 \cdot 10) + (0 \cdot 50) + (+1 \cdot 90) \\
    &= (-10 + 90) + (-20 + 180) + (-10 + 90) \\
    &= 80 + 160 + 80 = \mathbf{320}
\end{aligned}$$
Now consider a horizontal edge (where values vary only vertically):
$$I_{horiz} = \begin{bmatrix} 10 & 10 & 10 \\ 50 & 50 & 50 \\ 90 & 90 & 90 \end{bmatrix}$$
$$\begin{aligned}
G_x &= (-1 \cdot 10) + (0 \cdot 10) + (+1 \cdot 10) \\
    &\quad + (-2 \cdot 50) + (0 \cdot 50) + (+2 \cdot 50) \\
    &\quad + (-1 \cdot 90) + (0 \cdot 90) + (+1 \cdot 90) \\
    &= 0 + 0 + 0 = \mathbf{0}
\end{aligned}$$
**Metrological Consequence:** Sobel-X produces a strong response of 320 on the vertical cylinder boundary, but exactly 0 on horizontal shoulder steps and grinding lines.

---

## 4. Median Column Projection (Rank Order Statistics)

### Mathematical Formulation
Given a 2D gradient band of height $M = y_{end} - y_{start}$ and width $W$:
$$G_x \in \mathbb{R}^{M \times W}$$
For each column $x \in [0, W-1]$, extract the column vector:
$$\mathbf{g}_x = \begin{bmatrix} G_x(0, x), & G_x(1, x), & \dots, & G_x(M-1, x) \end{bmatrix}^T$$
Sort the elements in ascending order:
$$\mathbf{g}_{(x)} = \left[ g_{(1)}, \; g_{(2)}, \; \dots, \; g_{(M)} \right], \quad g_{(1)} \le g_{(2)} \le \dots \le g_{(M)}$$
The 1D projected score $S(x)$ is the 50th percentile (median):
$$S(x) = \begin{cases} g_{\left(\frac{M+1}{2}\right)}, & M \text{ is odd} \\ \frac{1}{2}\left(g_{\left(\frac{M}{2}\right)} + g_{\left(\frac{M}{2}+1\right)}\right), & M \text{ is even} \end{cases}$$

### Step-by-Step Numerical Example (Breakdown Point Demonstration)
Suppose a vertical band has $M = 5$ rows at column $x=10$.
Four rows observe the normal dark-to-light boundary ($G_x = 40$), but one row has an intense specular reflection spike ($G_x = 250$):
$$\mathbf{g} = [40, \; 40, \; \mathbf{250}, \; 40, \; 40]$$

- **Arithmetic Mean:**
  $$\bar{g} = \frac{40 + 40 + 250 + 40 + 40}{5} = \frac{410}{5} = \mathbf{82.0} \quad (\text{Skews peak by } +105\%)$$

- **Median:**
  $$\text{Sort } \mathbf{g} \implies [40, \; 40, \; \mathbf{40}, \; 40, \; 250]$$
  $$\operatorname{median}(\mathbf{g}) = \mathbf{40.0} \quad (\text{Outlier has } \mathbf{0\%} \text{ impact!})$$

---

## 5. Three-Point Parabolic Subpixel Interpolation

### Mathematical Derivation
Let the discrete integer peak occur at index $x^*$. The true continuous peak lies at $x^* + \delta$, where $\delta \in [-0.5, 0.5]$.
Model the continuous curve locally as a second-order polynomial (parabola):
$$f(\delta) = a\delta^2 + b\delta + c$$
Evaluate $f(\delta)$ at three discrete offsets $\delta \in \{-1, 0, +1\}$:
1. $\delta = -1 \implies S_{-1} = a(-1)^2 + b(-1) + c = a - b + c$
2. $\delta = 0 \implies S_0 = a(0)^2 + b(0) + c = c$
3. $\delta = +1 \implies S_{+1} = a(1)^2 + b(1) + c = a + b + c$

Solve for coefficients $a, b, c$:
- From (2): $c = S_0$
- Subtract (1) from (3):
  $$S_{+1} - S_{-1} = 2b \implies b = \frac{S_{+1} - S_{-1}}{2}$$
- Add (1) and (3):
  $$S_{+1} + S_{-1} = 2a + 2c = 2a + 2S_0 \implies a = \frac{S_{-1} - 2S_0 + S_{+1}}{2}$$

To find the continuous vertex (maximum), take the derivative and set to zero:
$$\frac{df}{d\delta} = 2a\delta + b = 0 \implies \delta^* = -\frac{b}{2a}$$
Substitute $a$ and $b$:
$$\delta^* = -\frac{\frac{S_{+1} - S_{-1}}{2}}{2 \cdot \left(\frac{S_{-1} - 2S_0 + S_{+1}}{2}\right)} = -\frac{1}{2} \cdot \frac{S_{+1} - S_{-1}}{S_{-1} - 2S_0 + S_{+1}} = \frac{1}{2} \cdot \frac{S_{-1} - S_{+1}}{S_{-1} - 2S_0 + S_{+1}}$$

The continuous subpixel edge location is:
$$x_{\text{subpixel}} = x^* + \delta^*$$

### Step-by-Step Numerical Example
Suppose the discrete 1D gradient profile has values around discrete peak $x^* = 100$:
$$S(99) = 30.0, \quad S(100) = 80.0, \quad S(101) = 60.0$$
Notice that $S(101) = 60.0$ is much higher than $S(99) = 30.0$. This indicates that the true continuous peak was pulled to the right of $x^* = 100$.

1. Compute numerator:
   $$\text{Num} = S_{-1} - S_{+1} = 30.0 - 60.0 = -30.0$$

2. Compute denominator:
   $$\text{Den} = S_{-1} - 2S_0 + S_{+1} = 30.0 - 2(80.0) + 60.0 = 30.0 - 160.0 + 60.0 = -70.0$$

3. Compute offset:
   $$\delta^* = \frac{1}{2} \cdot \frac{-30.0}{-70.0} = 0.5 \cdot (+0.4286) = +\mathbf{0.2143}\text{ pixels}$$

4. Continuous Edge Coordinate:
   $$x_{\text{subpixel}} = 100 + 0.2143 = \mathbf{100.2143}\text{ pixels}$$

Instead of reporting the crude integer 100, the algorithm reports $100.214\text{ px}$.

---

## 6. Distance Formula & Pixel-to-Millimeter Conversion

### Actual Distance Formula Used
Because the job sub-image has been normalized upright (cylinder axis along $y$), the diameter is the 1D horizontal distance between the left and right subpixel boundaries:
$$D_{px} = x_{\text{subpixel, right}} - x_{\text{subpixel, left}}$$
*Note:* The code does **not** compute 2D Euclidean distance ($\sqrt{\Delta x^2 + \Delta y^2}$) because $x_{\text{right}}$ and $x_{\text{left}}$ represent boundary coordinates across the same vertical band after alignment.

### Linear Scale Transformation
Given scale factor $s_{cal} = \text{mm\_per\_pixel}$:
$$D_{mm} = D_{px} \cdot s_{cal}$$

### Numerical Example
Suppose:
- $x_{\text{subpixel, left}} = 8.792\text{ px}$
- $x_{\text{subpixel, right}} = 393.169\text{ px}$
- Calibration scale $s_{cal} = 0.100\text{ mm/pixel}$

$$\begin{aligned}
D_{px} &= 393.169 - 8.792 = \mathbf{384.377}\text{ pixels} \\
D_{mm} &= 384.377 \cdot 0.100 = \mathbf{38.438}\text{ mm}
\end{aligned}$$

---

## 7. Statistical Metrics: Part-to-Part Spread vs. Measurement Repeatability

### Sample Mean ($\bar{x}$)
$$\bar{x} = \frac{1}{N} \sum_{i=1}^N x_i$$

### Sample Standard Deviation ($\sigma$)
$$\sigma = \sqrt{\frac{1}{N - 1} \sum_{i=1}^N (x_i - \bar{x})^2}$$

### Distinct Metrological Applications in This Project

#### 1. Part-to-Part Spread (Single Image, 9 Different Jobs)
Measures physical manufacturing variance among different components:
- $N = 9$ components.
- $x_i = D_1$ measurement of component $i$ in Image 1.
- In our project: $\sigma_{\text{part-to-part}} \approx \mathbf{1.43}\text{ px}$.

#### 2. Measurement Repeatability (4 Captures of the Exact Same Component)
Measures precision/reproducibility of the vision system:
- $N = 4$ captures of `job_r01_c02`.
- $x = [387.04, \; 387.21, \; 387.07, \; 386.86]\text{ px}$.
- Mean $\bar{x} = 387.05\text{ px}$.
- Repeatability standard deviation:
  $$\sigma_{\text{repeat}} = \sqrt{\frac{(387.04-387.05)^2 + (387.21-387.05)^2 + (387.07-387.05)^2 + (386.86-387.05)^2}{3}} = \mathbf{0.125}\text{ px}$$

**Crucial Metrological Takeaway:**  
- Repeatability standard deviation is **$0.125\text{ px}$** ($\approx 12.5\ \mu\text{m}$).
- Part-to-part manufacturing spread is **$1.43\text{ px}$** ($\approx 143\ \mu\text{m}$).
- They must **never** be conflated.
