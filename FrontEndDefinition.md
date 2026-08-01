# The front end, defined

Every feature the front end produces, with the arithmetic written out. Companion to the
plain-language version in `SimpleStrokeTests/RESULTS.md`.

Source of truth: `RationalGaborFeatures/{GaborStack,AndLayer,RayHarmonics,Pooling}.module.jl`
and `SimpleStrokeTests/Frontend.module.jl`.

---

## Symbols

### The image and the filter bank

| symbol | meaning |
|:--|:--|
| $I(p)$ | the image, at pixel $p=(y,x)$; $W$ is its width in pixels (112 throughout) |
| $s = 1 \dots S$ | **scale** index. Production: $S=3$ |
| $\rho_s$ | scale $s$ in **cycles across the image**. Production: $\rho = 2.00,\ 3.742,\ 7.00$ |
| $\lambda_s = W/\rho_s$ | the same scale as a **wavelength in pixels**: $56.0,\ 29.9,\ 16.0$ |
| $\beta_s$ | radial bandwidth of scale $s$ in octaves: $2.0,\ 1.6,\ 1.2$ |
| $n_s$ | number of **directions** at scale $s$: $8,\ 12,\ 16$ |
| $k = 1 \dots n_s$ | direction index |
| $\theta_{s,k} = \dfrac{(k-1)\pi}{n_s}$ | the direction of channel $k$. Spans $[0,\pi)$: a line and the same line turned by $180°$ are the same line |
| $t = 0.75$ | `dtheta_on_sigma`, one global constant |
| $\sigma_{\varphi,s} = \dfrac{\pi/n_s}{t}$ | angular width of each channel: $30°,\ 20°,\ 15°$ |
| $G_{s,k}$ | the filter for scale $s$, direction $k$, held in the frequency domain |
| $G_{\text{lp}}$ | one extra non-directional low-pass filter, at $\rho_{\text{lp}} = \min_s \rho_s / 2 = 1.0$ |

### The responses everything is built from

The image is padded by copying its border outward, transformed once, multiplied by each filter, and
transformed back:

$$r_{s,k}(p) \;=\; \mathcal{F}^{-1}\!\big[\, \mathcal{F}[I]\cdot G_{s,k} \,\big](p)
\qquad\text{(complex)}$$

$$E_{s,k}(p) \;=\; \big|\, r_{s,k}(p) \,\big|^{2}
\qquad\qquad
E_{\text{lp}}(p) \;=\; \big|\, \mathcal{F}^{-1}[\, \mathcal{F}[I]\cdot G_{\text{lp}} \,](p) \,\big|^{2}$$

$E_{s,k}(p)$ is the **filter response at a pixel**: how strongly a stroke of size $\lambda_s$
running in direction $\theta_{s,k}$ is present at $p$. It is real and non-negative.

#### Why $r$ is complex, and why that matters

For a **real** image the Fourier transform is redundant: the value at frequency $-f$ is the complex
conjugate of the value at $+f$, so the negative half of the frequency plane mirrors the positive
half and carries nothing new.

An ordinary filter — one you could write down as a real array of numbers in the image — keeps
**both** halves and produces a real output. $G_{s,k}$ is **one-sided**: it is zero on one half of
the frequency plane and keeps only the other, which is why $r$ comes out complex. The half is
picked by direction — writing $\varphi$ for the direction of a frequency and $\theta_0$ for the
channel's own direction,

$$G_{s,k}(f) \;=\; \underbrace{\exp\!\left(-\frac{\log^{2}(\rho/\rho_s)}{2\log^{2}\kappa_\beta}\right)}_{\text{radial}}
\;\cdot\;
\underbrace{\begin{cases}
\exp\!\left(-\dfrac{\Delta\varphi^{2}}{2\sigma_{\varphi,s}^{2}}\right), & |\Delta\varphi| \le \pi/2\\[2mm]
0, & |\Delta\varphi| > \pi/2
\end{cases}}_{\text{angular — this is the one-sidedness}}$$

with $\Delta\varphi = \operatorname{atan2}\!\big(\sin(\varphi-\theta_0),\, \cos(\varphi-\theta_0)\big)$.
The angular factor is **zero wherever the frequency's direction is more than $90°$ from
$\theta_0$**, which is what confines the filter to a half-plane.

The complex result is the **analytic signal**, Gabor's own term from 1946. In polar form
$r(p) = a(p)\,e^{i\phi(p)}$, its modulus $a(p) = |r(p)|$ is the **envelope** — how much of that
frequency band is present — and $\phi(p)$ is the **local phase**, whereabouts you sit within the
oscillation.

Three consequences, all load-bearing:

**$|r|$ does not oscillate.** A real bandpass filter's output swings positive and negative as the
underlying wave rises and falls, so "how much structure is here" flickers with exactly where you
sample. $|r|$ is the smooth outline of that swing rather than the swing itself.

**Exact quadrature, for free.** The classic construction uses two filters — an even-symmetric one
responding to bar-like features and an odd-symmetric one responding to edge-like features, $90°$
out of phase — and forms $\text{even}^2 + \text{odd}^2$. Here $\operatorname{Re}(r)$ *is* the even
response and $\operatorname{Im}(r)$ *is* the odd one, exactly in quadrature by construction rather
than approximately so because two spatial kernels were built separately and hoped to match.

**Contrast-polarity invariance is exact.** Invert the image, $I \to -I$; then $r \to -r$, so

$$E \;=\; |r|^{2} \;\longrightarrow\; |-r|^{2} \;=\; |r|^{2}$$

identically. Not approximately — identically. This is the property behind the front end scoring
$-0.06$ on polarity and transferring intact across a polarity flip, where frozen ConvNeXt reads
polarity at $0.998$ and collapses.

### Averaging

$w_c(p)$ is a Gaussian window for cell $c$, normalised so that $\sum_p w_c(p) = 1$. With a
$g \times g$ grid there are $n_c = g^2$ cells; at $g=1$ the single window covers the whole picture.

$$\langle f \rangle_c \;=\; \sum_p w_c(p)\, f(p)$$

This is the **only** averaging operator below. Where the subscript is dropped, read it as applying
to every cell.

### Constants

| symbol | value | used in |
|:--|--:|:--|
| $\kappa$ | $0.5$ | stroke-end stabiliser |
| $\varepsilon$ | $10^{-12}$ | guard where a denominator can vanish |
| $\gamma$ | $1.0$ | probe distance multiplier (`d_factor`) |
| $\sigma_{\parallel,s} = \dfrac{W}{2\pi \rho_s \sigma_{\varphi,s}}$ | $17.0,\ 13.6,\ 9.7$ px | the filter's extent **along** a contour |
| $d_s = \gamma\,\sigma_{\parallel,s}$ | $17.0,\ 13.6,\ 9.7$ px | how far the stroke-end and branching probes step out |
| $c_s$ | $6.64\times10^{-3},\ 2.40\times10^{-5},\ 9.12\times10^{-9}$ | corner-strength floor, §7 |

---

## 1. Orientation summary — 5 numbers per scale per cell

**Average first, then combine.**

$$\bar{E}_{s,k,c} \;=\; \langle E_{s,k} \rangle_c$$

$$T_{s,c} \;=\; \sum_{k=1}^{n_s} \bar{E}_{s,k,c}
\qquad
Z^{(2)}_{s,c} \;=\; \sum_{k=1}^{n_s} \bar{E}_{s,k,c}\, e^{\,i\,2\theta_{s,k}}
\qquad
Z^{(4)}_{s,c} \;=\; \sum_{k=1}^{n_s} \bar{E}_{s,k,c}\, e^{\,i\,4\theta_{s,k}}$$

$$z_2 = \frac{Z^{(2)}}{T}, \qquad z_4 = \frac{Z^{(4)}}{T} \qquad (\text{both } 0 \text{ if } T=0)$$

Features: $\quad \sqrt{T}, \quad \operatorname{Re} z_2, \quad \operatorname{Im} z_2, \quad |z_2|, \quad |z_4|$

$e^{i2\theta}$ rather than $e^{i\theta}$ because direction is defined modulo $\pi$: doubling the
angle makes a direction and its opposite land on the same point of the circle. $|z_2| = 0$ when all
directions are equally present and $1$ when only one is; $\arg(z_2)/2$ is the dominant direction.

## 2. Overall brightness — 1 number per cell

$$\text{feature} \;=\; \sqrt{\big\langle E_{\text{lp}} \big\rangle_c}$$

## 3. Corner strength $A_1$ — 1 number per scale per cell

**Combine at the pixel, then average.** With $h = n_s/2$, shifting the direction index by $h$ is
exactly a $90°$ turn; indices wrap modulo $n_s$.

$$C_{0,s}(p) \;=\; \sum_{k=1}^{n_s} E_{s,k}(p)
\qquad\qquad
S_s(p) \;=\; \sum_{k=1}^{n_s} E_{s,k}(p)\, E_{s,\,k+h}(p)$$

$$A_{1,s}(p) \;=\; \max\!\left(0,\; \frac{S_s(p)}{C_{0,s}(p)} \;-\; c_s\, C_{0,s}(p)\right)
\quad\text{if } C_{0,s}(p) > \varepsilon,\ \text{else } 0$$

$$\text{feature} \;=\; \sqrt{\big\langle A_{1,s} \big\rangle_c}$$

The product $E_k E_{k+h}$ is large only if **both** factors are large, which is what makes this a
statement about a single pixel rather than about a region. Dividing by $C_0$ turns a squared
quantity back into an energy. The $-c_s C_0$ term is the correction of §7.

## 4. Stroke-end strength $A_2$ — 1 number per scale per cell

**Combine at the pixel, then average.**

$$k^{*}(p) \;=\; \operatorname*{arg\,max}_{k}\, E_{s,k}(p)
\qquad\text{(ties resolve to the lowest } k)$$

$$\psi(p) \;=\; \theta_{s,k^{*}(p)} + \tfrac{\pi}{2}
\qquad\qquad
\mathbf{u}(p) \;=\; \big(\sin\psi(p),\, \cos\psi(p)\big)$$

The stroke runs at right angles to the filter's carrier, so $\mathbf{u}$ points **along** it.
Writing $e(p) = E_{s,k^{*}(p)}(p)$ and sampling bilinearly (zero outside the image),

$$E_{\pm}(p) \;=\; E_{s,k^{*}(p)}\big(p \pm d_s\,\mathbf{u}(p)\big)$$

$$A_{2,s}(p) \;=\; e(p)\;\frac{\big|E_{+}(p) - E_{-}(p)\big|}{E_{+}(p) + E_{-}(p) + \kappa\, e(p)}
\quad\text{if } e(p) > \varepsilon,\ \text{else } 0$$

$$\text{feature} \;=\; \sqrt{\big\langle A_{2,s} \big\rangle_c}$$

The ratio is $0$ where the stroke continues both ways and near $1$ at a termination. $\kappa e$ is a
*relative* stabiliser: an absolute one collapsed $A_2$ to plain energy. The leading $e(p)$ makes the
result an energy, so averaging it is meaningful.

## 5. Branching — 3 numbers per scale per cell

**Average first, then divide.** With $K = 2 n_s$ probe directions,

$$\varphi_j \;=\; \frac{2\pi (j-1)}{K}, \qquad j = 1 \dots K
\qquad\text{— a full turn, not a half turn}$$

$$\text{ch}(\varphi) \;=\; \text{the } k \text{ whose } \theta_{s,k} \text{ is closest to }
(\varphi + \tfrac{\pi}{2}) \bmod \pi$$

$$R_s(p, \varphi_j) \;=\; E_{s,\,\text{ch}(\varphi_j)}\Big(p + d_s\big(\sin\varphi_j,\, \cos\varphi_j\big)\Big)$$

$R$ asks: *is there a contour at distance $d_s$ in direction $\varphi$, oriented along $\varphi$?*
The spatial step is what recovers a full turn from directions that only span half of one — east and
west read different pixels.

$$a_{0,s}(p) = \sum_{j=1}^{K} R_s(p,\varphi_j)
\qquad
a_{1,s}(p) = \left|\sum_{j=1}^{K} R_s(p,\varphi_j)\, e^{\,i\varphi_j}\right|
\qquad
a_{2,s}(p) = \left|\sum_{j=1}^{K} R_s(p,\varphi_j)\, e^{\,i2\varphi_j}\right|$$

$$f_s \;=\; 10^{-3}\cdot \frac{1}{n_c}\sum_{c} \big\langle a_{0,s} \big\rangle_c
\qquad\text{a floor set by the image itself}$$

$$\text{features} \;=\; \big\langle a_{0,s}\big\rangle_c,
\qquad
\frac{\big\langle a_{1,s}\big\rangle_c}{\big\langle a_{0,s}\big\rangle_c + f_s},
\qquad
\frac{\big\langle a_{2,s}\big\rangle_c}{\big\langle a_{0,s}\big\rangle_c + f_s}$$

The two ratios are formed **after** averaging, not per pixel. Per pixel they are bounded quantities
with no limit as $a_0 \to 0$, so writing $0$ there asserted "perfectly symmetric" wherever there was
no evidence, and averaging those zeros made the result scale with how much of the picture was blank.

Idealised values for a point with $m$ equally spaced branches: $a_1/a_0 = 0$ for two opposite
branches or for four; $= 1/\sqrt{2}$ for two branches at $90°$; $= 1/3$ for a T-junction.

## 6. Strongest-anywhere — 3 numbers per scale

**Maximum instead of average**, over the whole picture, independent of the grid.

$$\text{features} \;=\; \max_p A_{1,s}(p), \qquad \max_p A_{2,s}(p), \qquad \max_p a_{0,s}(p)$$

Note the absence of $\sqrt{\cdot}$ and of any cell index: these are single numbers per scale.

## 7. The corner-strength floor $c_s$

A perfectly straight line should give $A_1 = 0$, and does not, because the channels have angular
width. For a single-direction input at angle $\theta_0$ the radial part of the filter is common to
every channel and cancels in the ratio, so each channel's response is fixed by angle alone:

$$\Delta_k \;=\; \min\big(|\theta_{s,k}-\theta_0|,\; \pi - |\theta_{s,k}-\theta_0|\big)
\qquad
\tilde{E}_k \;=\; \exp\!\left(-\left(\frac{\Delta_k}{\sigma_{\varphi,s}}\right)^{2}\right)$$

$$\tilde{C}_0 = \sum_k \tilde{E}_k
\qquad
\tilde{S} = \sum_k \tilde{E}_k \tilde{E}_{k+h}
\qquad
\boxed{\;c_s \;=\; \max_{\theta_0}\; \frac{\tilde{S}}{\tilde{C}_0^{\,2}}\;}$$

The exponent is squared rather than halved because energy is amplitude squared.

This gives $6.64\times10^{-3},\ 2.40\times10^{-5},\ 9.12\times10^{-9}$. The dominant term is **not**
the channel $90°$ from the line — that sits $3\sigma_\varphi$ away and contributes
$e^{-9} \approx 1.2\times10^{-4}$ — but the **pair straddling the line at $\pm 45°$**, each
$1.5\sigma_\varphi$ away with $e^{-2.25} = 0.105$ of the energy and exactly $90°$ apart, so their
product is $0.105^2 = 1.1\times10^{-2}$, two orders larger. The closed form matches measurement to
within 2 % at every scale (`RationalGaborFeatures/Validate_i1D.jl`).

---

## Feature count

Per cell, per scale: 5 orientation $+$ 1 corner $+$ 1 stroke-end $+$ 3 branching $=$ **10**. Plus
one brightness number per cell, and three strongest-anywhere numbers per scale.

$$\text{total} \;=\; n_c\,(10 S + 1) \;+\; 3S$$

| configuration | scales $S$ | grid $g$ | total |
|:--|--:|--:|--:|
| baseline | 3 | 1 | 31 |
| baseline | 3 | 3 | 279 |
| $+$ strongest-anywhere | 3 | 1 | 40 |
| $+\ \lambda = 8$ px scale | 4 | 1 | 41 |
| $+\ \lambda = 8$ px and strongest-anywhere | 4 | 1 | 53 |

---

## Where each operation happens

| block | at the pixel | then |
|:--|:--|:--|
| orientation summary | — | average, **then** combine across directions, then divide |
| brightness | — | average |
| corner strength | multiply pairs at $90°$, divide by the pixel total, subtract floor | average |
| stroke-end | pick strongest direction, compare two probes, divide, multiply by centre | average |
| branching | build the ring, take three circular sums | average each, **then** divide |
| strongest-anywhere | (maps as above) | **maximum** |

Two orderings are in tension, and the difference between them is the project's central claim:

$$\big\langle f g \big\rangle \;\neq\; \big\langle f \big\rangle \big\langle g \big\rangle,
\qquad\qquad
\big\langle f g \big\rangle - \big\langle f \big\rangle\big\langle g \big\rangle
\;=\; \operatorname{Cov}_p(f, g)$$

That covariance within the window is direct evidence that $f$ and $g$ were large *at the same
place*, and it is destroyed by averaging first. Corner strength, stroke-end strength and the
branching ring all take the left-hand form. The orientation summary takes the right-hand one, and
has never been compared against the alternative.
