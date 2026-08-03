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
| $\rho_s$ | scale $s$ in **cycles across the image**. Production: $\rho = 2.00, 3.742, 7.00$ |
| $\lambda_s = W/\rho_s$ | the same scale as a **wavelength in pixels**: $56.0, 29.9, 16.0$ |
| $\beta_s$ | radial bandwidth of scale $s$ in octaves: $2.0, 1.6, 1.2$ |
| $n_s$ | number of **directions** at scale $s$: $8, 12, 16$ |
| $k = 1 \dots n_s$ | direction index |
| $\theta_{s,k} = \dfrac{(k-1)\pi}{n_s}$ | the direction of channel $k$. Spans $[0,\pi)$: a line and the same line turned by $180°$ are the same line |
| $t = 0.75$ | `dtheta_on_sigma`, one global constant |
| $\sigma_{\varphi,s} = \dfrac{\pi/n_s}{t}$ | angular width of each channel: $30°, 20°, 15°$ |
| $G_{s,k}$ | the filter for scale $s$, direction $k$, held in the frequency domain |
| $G_{\text{lp}}$ | one extra non-directional low-pass filter, at $\rho_{\text{lp}} = \min_s \rho_s / 2 = 1.0$ |

### The responses everything is built from

The image is padded by copying its border outward, transformed once, multiplied by each filter, and
transformed back:

```math
r_{s,k}(p) = \mathcal{F}^{-1}\left( \mathcal{F}(I) \cdot G_{s,k} \right)(p)
\qquad\text{(complex)}
```

```math
E_{s,k}(p) = \big| r_{s,k}(p) \big|^{2}
\qquad\qquad
E_{\text{lp}}(p) = \big| \mathcal{F}^{-1}\left( \mathcal{F}(I) \cdot G_{\text{lp}} \right)(p) \big|^{2}
```

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

```math
G_{s,k}(f) = \underbrace{\exp\left(-\frac{\log^{2}(\rho/\rho_s)}{2\log^{2}\kappa_\beta}\right)}_{\text{radial}}
\cdot
\underbrace{\begin{cases}
\exp\left(-\dfrac{\Delta\varphi^{2}}{2\sigma_{\varphi,s}^{2}}\right), & |\Delta\varphi| \le \pi/2\\
0, & |\Delta\varphi| > \pi/2
\end{cases}}_{\text{angular — this is the one-sidedness}}
```

with $\Delta\varphi = \mathrm{atan2}\big(\sin(\varphi-\theta_0), \cos(\varphi-\theta_0)\big)$.
The angular factor is **zero wherever the frequency's direction is more than $90°$ from
$\theta_0$**, which is what confines the filter to a half-plane.

The complex result is the **analytic signal**, Gabor's own term from 1946. In polar form
$r(p) = a(p) e^{i\phi(p)}$, its modulus $a(p) = |r(p)|$ is the **envelope** — how much of that
frequency band is present — and $\phi(p)$ is the **local phase**, whereabouts you sit within the
oscillation.

Three consequences, all load-bearing:

**$|r|$ does not oscillate.** A real bandpass filter's output swings positive and negative as the
underlying wave rises and falls, so "how much structure is here" flickers with exactly where you
sample. $|r|$ is the smooth outline of that swing rather than the swing itself.

**Exact quadrature, for free.** The classic construction uses two filters — an even-symmetric one
responding to bar-like features and an odd-symmetric one responding to edge-like features, $90°$
out of phase — and forms $\text{even}^2 + \text{odd}^2$. Here $\mathrm{Re}(r)$ *is* the even
response and $\mathrm{Im}(r)$ *is* the odd one, exactly in quadrature by construction rather
than approximately so because two spatial kernels were built separately and hoped to match.

**Contrast-polarity invariance is exact.** Invert the image, $I \to -I$; then $r \to -r$, so

```math
E = |r|^{2} \longrightarrow |-r|^{2} = |r|^{2}
```

identically. Not approximately — identically. This is the property behind the front end scoring
$-0.06$ on polarity and transferring intact across a polarity flip, where frozen ConvNeXt reads
polarity at $0.998$ and collapses.

### Averaging

$w_c(p)$ is a Gaussian window for cell $c$, normalised so that $\sum_p w_c(p) = 1$. With a
$g \times g$ grid there are $n_c = g^2$ cells; at $g=1$ the single window covers the whole picture.

```math
\langle f \rangle_c = \sum_p w_c(p) f(p)
```

This is the **only** averaging operator below. Where the subscript is dropped, read it as applying
to every cell.

### Constants

| symbol | value | used in |
|:--|--:|:--|
| $\kappa$ | $0.5$ | stroke-end stabiliser |
| $\varepsilon$ | $10^{-12}$ | guard where a denominator can vanish |
| $\gamma$ | $1.0$ | probe distance multiplier (`d_factor`) |
| $\sigma_{\parallel,s} = \dfrac{W}{2\pi \rho_s \sigma_{\varphi,s}}$ | $17.0, 13.6, 9.7$ px | the filter's extent **along** a contour |
| $d_s = \gamma \sigma_{\parallel,s}$ | $17.0, 13.6, 9.7$ px | how far the stroke-end and branching probes step out |
| $c_s$ | $6.64\times10^{-3}, 2.40\times10^{-5}, 9.12\times10^{-9}$ | corner-strength floor, §7 |

---

## 1. Orientation summary — 5 numbers per scale per cell

**The idea.** Within a cell, ask one question: *how is energy distributed across directions?* The
answer is a function of orientation, sampled at the $n_s$ directions the bank provides. These five
numbers are the first few terms of its Fourier series.

**Average first, then combine.**

### Step 1 — average each direction separately

```math
\bar{E}_{s,k,c} = \langle E_{s,k} \rangle_c
```

The mean over cell $c$ of the squared modulus of the Gabor response at scale $s$, direction $k$.
This gives the cell's orientation profile for each scale and each direction.

There is a separate profile for every scale, because a cell can be dominated by one direction at a
coarse scale and another at a fine one. Steps 2 and 3 are applied to each profile independently. At
grid 1 with the production bank that is $8 + 12 + 16 = 36$ averages — three profiles, of 8, 12 and
16 values — each reduced to 5 features, giving the 15 orientation features.

### Step 2 — expand the profile as a Fourier series

Orientation is **$\pi$-periodic**: a line at $\theta$ and the same line at $\theta + \pi$ are
identical. So the profile must repeat every half-turn, and its Fourier series runs in
$e^{i2\theta},\ e^{i4\theta},\ e^{i6\theta} \dots$ rather than $e^{i\theta},\ e^{i2\theta} \dots$

**The exponential form.** Any $\pi$-periodic function has the expansion

```math
\bar{E}_{s,c}(\theta) \;=\; \sum_{m=-\infty}^{\infty} c_m \, e^{i 2 m \theta},
\qquad
c_m \;=\; \frac{1}{n_s}\sum_{k=1}^{n_s} \bar{E}_{s,k,c}\, e^{-i 2 m \theta_{s,k}}
```

Two properties of it matter here:

* $c_0$ is real and equals the mean of the profile.
* Because $\bar{E}$ is **real**, $c_{-m} = \overline{c_m}$. The negative-index coefficients are not
  independent information — they are the conjugates of the positive ones — so everything is
  contained in $m = 0, 1, 2, \dots$

That second fact is what turns the complex series into a real one. Pairing each $m$ with its
partner $-m$:

```math
c_m e^{i2m\theta} + c_{-m} e^{-i2m\theta}
\;=\; c_m e^{i2m\theta} + \overline{c_m e^{i2m\theta}}
\;=\; 2\,\mathrm{Re}\big(c_m e^{i2m\theta}\big)
\;=\; 2\,|c_m|\cos\big(2m\theta + \arg c_m\big)
```

so the cosine form below is not a different expansion — it is this one with the conjugate pairs
collected.

**What we actually compute.**

```math
T_{s,c} = \sum_{k=1}^{n_s} \bar{E}_{s,k,c}
\qquad
Z^{(2)}_{s,c} = \sum_{k=1}^{n_s} \bar{E}_{s,k,c}\, e^{+i 2 \theta_{s,k}}
\qquad
Z^{(4)}_{s,c} = \sum_{k=1}^{n_s} \bar{E}_{s,k,c}\, e^{+i 4 \theta_{s,k}}
```

**How these relate to the $c_m$.** Compare the two definitions directly. The sum defining
$Z^{(2m)}$ is the sum defining $c_m$ with the $1/n_s$ dropped and the sign of the exponent
flipped — and flipping that sign on a real-weighted sum is exactly conjugation:

```math
Z^{(2m)}_{s,c} \;=\; n_s\,\overline{c_m} \;=\; n_s\, c_{-m}
```

So our $Z^{(2)}$ is $n_s$ times the textbook coefficient at index $-1$, and $Z^{(4)}$ is $n_s$
times the one at index $-2$. Two differences from the textbook: a **factor $n_s$**, and a
**conjugation**.

Substituting back gives the profile in cosine form, and the minus signs inside the cosines are
precisely where the conjugation surfaces:

```math
\bar{E}_{s,c}(\theta) \;=\; \frac{1}{n_s}\Big(\;
T_{s,c}
\;+\; 2\,\big|Z^{(2)}_{s,c}\big| \cos\!\big(2\theta - \arg Z^{(2)}_{s,c}\big)
\;+\; 2\,\big|Z^{(4)}_{s,c}\big| \cos\!\big(4\theta - \arg Z^{(4)}_{s,c}\big)
\;+\; \text{higher terms} \;\Big)
```

Reading the three terms off the expansion:

| term | what it is |
|:--|:--|
| $T/n_s$ | the **flat part** — average energy across all directions, with no dependence on $\theta$ |
| $2\lvert Z^{(2)}\rvert / n_s$ | the size of the ripple going **once round per half-turn**. Its peak sits at $\theta = \arg(Z^{(2)})/2$, which is therefore the **dominant orientation** |
| $2\lvert Z^{(4)}\rvert / n_s$ | the size of the ripple going **twice round per half-turn**, peaking at $\arg(Z^{(4)})/4$ |

The series is truncated there; only these three are kept.

Two checks, which are what make the features interpretable.

**A single straight line** puts all its energy at one direction $\theta_0$. Then $T = E$ and
$Z^{(2)} = E e^{i2\theta_0}$, so the once-round ripple has full amplitude and peaks exactly at
$\theta_0$.

**Two lines crossing at $90°$** give
$Z^{(2)} = E e^{i2\theta_0} + E e^{i2\theta_0 + i\pi} = 0$ — the once-round ripple **vanishes
identically** — while
$Z^{(4)} = E e^{i4\theta_0} + E e^{i4\theta_0 + i2\pi} = 2E e^{i4\theta_0}$, at full amplitude.
That is why a crossing is invisible to the first harmonic and maximal in the second.

#### Why the two departures from the textbook convention are harmless

**The factor $n_s$.** It appears in $T$ and in every $Z$ alike — $T = n_s c_0$ and
$Z^{(2m)} = n_s \overline{c_m}$ — and every downstream use is a ratio of one to the other:

```math
z_2 \;=\; \frac{Z^{(2)}}{T} \;=\; \frac{n_s \overline{c_1}}{n_s c_0} \;=\; \frac{\overline{c_1}}{c_0}
```

The $n_s$ cancels identically and never reaches a feature. The one exception is $\sqrt{T}$, which
is reported directly and so carries an uncancelled $\sqrt{n_s}$ — a fixed scale factor on a single
feature, which the first weight of a linear readout absorbs.

**The conjugation.** It costs nothing, since conjugation preserves modulus and $\lvert z_2\rvert$
and $\lvert z_4\rvert$ are therefore unchanged — and it buys a sign. Take a cell containing a
single line at $\theta_0$, so all the energy $E$ sits at one direction:

| | our convention | textbook |
|:--|:--|:--|
| coefficient | $Z^{(2)} = E e^{+i2\theta_0}$ | $c_1 = \frac{E}{n_s}e^{-i2\theta_0}$ |
| its argument | $\arg Z^{(2)} = +2\theta_0$ | $\arg c_1 = -2\theta_0$ |
| recovering $\theta_0$ | $\theta_0 = \arg(Z^{(2)})/2$ | $\theta_0 = -\arg(c_1)/2$ |

Both work; ours reads the orientation off directly instead of requiring a sign flip, and it is
what puts the cosine peak at $\theta = \arg(Z^{(2)})/2$ rather than at $-\arg(c_1)/2$.

The only visible consequence is that $\mathrm{Im}(z_2)$ carries the opposite sign to the textbook
convention. Since $\mathrm{Re}(z_2)$ and $\mathrm{Im}(z_2)$ are handed to a learned readout as a
pair, a global sign on one component is absorbed by the first weight it meets.

### Step 3 — normalise by the total

```math
z_2 = \frac{Z^{(2)}}{T}, \qquad z_4 = \frac{Z^{(4)}}{T} \qquad (\text{both } 0 \text{ if } T = 0)
```

$Z^{(2)}$ and $Z^{(4)}$ have units of energy, so they grow both with image contrast and with how
much stroke happens to be in the cell. Dividing by $T$ removes both influences: $z_2$ and $z_4$ are
**dimensionless and contrast-invariant**, with magnitude bounded in $[0,1]$. They describe the
*shape* of the orientation profile rather than its size.

### The five reported numbers

| feature | what it is |
|:--|:--|
| $\sqrt{T}$ | **how much** oriented structure there is. The square root converts energy back to amplitude, so it scales linearly with image contrast rather than quadratically |
| $\mathrm{Re}(z_2), \mathrm{Im}(z_2)$ | the **orientation vector**. Its angle gives the dominant direction, $\arg(z_2)/2$; its length gives how dominant that direction is. Reported as two components rather than as angle-plus-length because an angle wraps from $\pi$ back to $0$, and a linear readout cannot follow a discontinuity |
| $\lvert z_2 \rvert$ | **coherence**: $0$ if all directions are equally present, $1$ if all the energy is in one. Rotation-invariant — the same for a line at any angle |
| $\lvert z_4 \rvert$ | the second harmonic's size. Rotation-invariant, and responds to profiles with **two lobes $90°$ apart** |

**Why both $\lvert z_2\rvert$ and $\lvert z_4\rvert$ are needed:**

| what is in the cell | $\lvert z_2\rvert$ | $\lvert z_4\rvert$ |
|:--|--:|--:|
| a single straight line | 1 | 1 |
| two lines crossing at $90°$ | **0** | 1 |
| no dominant direction | 0 | **0** |

Neither number alone separates the three cases. The pair does.

### Two consequences worth flagging

$\lvert z_4\rvert$ is a **corner-and-crossing measure computed at the cell level** — which is what
$A_1$ (§3) computes at the pixel level. They are the same idea in opposite orders:
average-then-combine here, combine-then-average there. The difference between them is exactly the
covariance discussed at the end of this document, and it has never been measured.

**Why the series stops at the second harmonic.** With $n_s$ directions sampled over $\pi$, the
harmonic $e^{i2m\theta}$ is estimable without aliasing only while $2m \le n_s - 2$, so $m \le 3$ at
$n_s = 8$. Keeping $m = 0, 1, 2$ is therefore a choice rather than a limit: the third harmonic is
available even at the coarsest scale. It was tested in the capacity sweep and is worth **+0.026**
on curvedness — which fits, since curvature is about how the profile *spreads*, and two harmonics
describe a shape only crudely.

## 2. Low-frequency contrast — 1 number per cell

```math
\text{feature} = \sqrt{\big\langle E_{\text{lp}} \big\rangle_c}
```

**Not brightness.** The low-pass filter has its DC term set to zero
(`GaborStack.module.jl:285`, "keep it DC-free like the rest"), so the image's mean level is removed
before anything else happens. What survives is how much the picture *varies* at very coarse scale —
the filter sits at $\rho_{\text{lp}} = 1.0$, that is $\lambda = 112$ px, about the width of the
whole image.

**A black stroke on white and a white stroke on black give the identical value.** The filter is
radially symmetric and real, so the filtered image is real, and squaring discards its sign. This
channel is therefore **exactly as polarity-invariant as the oriented ones**, which measurement
confirms: the `lowpass` block alone predicts polarity at $-0.000$, the same as `orient`.

Worth stating plainly, because the project long assumed the opposite. A comment in the Phase 9
harness predicted that `lowpass`, "which carries mean level, should get it easily" — it does not,
because the mean level is precisely what was thrown away.

**What it does carry** is coarse layout: where the ink is and roughly how much, at a scale far
below any stroke detail. Alone it reaches $0.659$ on closedness and $0.309$ on thickness against
$0.036$ on curvedness — it sees the gross shape of the figure and nothing about its contours.

## 3. Corner strength $A_1$ — 1 number per scale per cell

**The idea.** §1 asked which directions are present *in a region*. That cannot tell a corner from
two separate strokes that happen to share a cell — both give a profile with two lobes. $A_1$ asks
the stricter question: are two perpendicular directions present **at the same pixel**?

**Combine at the pixel, then average.**

### Step 1 — pair each direction with the one at right angles

With $h = n_s/2$, shifting the direction index by $h$ moves $\theta$ by exactly $90°$, since
$\theta_{s,k+h} = \theta_{s,k} + \tfrac{n_s}{2}\cdot\tfrac{\pi}{n_s} = \theta_{s,k} + \tfrac{\pi}{2}$.
Indices wrap modulo $n_s$, and $n_s$ is even at every scale so the pairing is exact.

```math
C_{0,s}(p) = \sum_{k=1}^{n_s} E_{s,k}(p)
\qquad\qquad
S_s(p) = \sum_{k=1}^{n_s} E_{s,k}(p)\, E_{s, k+h}(p)
```

$C_0$ is the total energy at this pixel across all directions. $S$ is the sum of **products** of
perpendicular pairs. Each pair appears twice, once from each end, so $S$ carries a factor of 2 —
harmless, since it is a constant multiplier on the whole feature.

**The product is the entire point.** $E_k E_{k+h}$ is large only if **both** factors are large.
Adding them, $E_k + E_{k+h}$, would be large if *either* were — which a single straight line would
satisfy. Multiplying demands both at once, at one pixel, which is what distinguishes a corner from
a pair of strokes lying in the same neighbourhood.

### Step 2 — normalise, and subtract the floor

```math
A_{1,s}(p) = \max\left(0,\; \frac{S_s(p)}{C_{0,s}(p)} \;-\; c_s\, C_{0,s}(p)\right)
\qquad\text{if } C_{0,s}(p) > \varepsilon, \text{ else } 0
```

**Why divide by $C_0$.** $S$ is a sum of products of energies, so it has units of energy *squared*
and grows as the fourth power of image contrast. Dividing by $C_0$ brings it back to units of
energy, matching $A_2$ and the orientation block, so the features stay on comparable scales and a
doubling of contrast has a consistent effect across all of them.

Writing $p_k = E_k / C_0$ for the normalised profile at that pixel makes the meaning plain:

```math
\frac{S_s(p)}{C_{0,s}(p)} \;=\; C_{0,s}(p) \sum_{k} p_k(p)\, p_{k+h}(p)
```

— that is, **the shape measure $\sum_k p_k p_{k+h}$ multiplied back by the total energy**. The shape
measure alone would be meaningless where there is no ink, since a normalised profile of noise is
still a profile. Multiplying by $C_0$ makes the result an energy, which is why averaging it is
meaningful and why $0$ is the correct value in blank regions.

**Why subtract $c_s C_0$.** A perfectly straight line should give $A_1 = 0$ and does not, because
the channels have angular width and a line leaks into directions either side of its own. §7 derives
the size of that leak in closed form: it is a fixed fraction $c_s$ of $C_0$, depending only on the
bank's angular tuning and never on the image. Subtracting it and clamping at zero removes the floor
exactly. The clamp is needed because $c_s$ is the *maximum* leak over line orientations, so at other
orientations the subtraction slightly overshoots.

### Step 3 — average and report

```math
\text{feature} = \sqrt{\big\langle A_{1,s} \big\rangle_c}
```

The square root converts energy to amplitude, matching $\sqrt{T}$ in §1.

**What it cannot do.** $A_1$ is built on the orientation profile, which is $\pi$-periodic, so it
cannot count branches. A T-junction and an X-crossing have identical direction content —
$0°$ and $90°$ — and $A_1$ gives them the same answer. Distinguishing them is a $2\pi$ question, and
that is what §5 exists for.

## 4. Stroke-end strength $A_2$ — 1 number per scale per cell

**The idea.** Walk along a stroke and ask at each pixel: *does the stroke continue in both
directions, or does it stop here?* In the middle of a stroke both sides look the same. At a
termination one side has a stroke and the other has nothing. The asymmetry between the two sides is
the signal.

**Combine at the pixel, then average.**

### Step 1 — find which way the stroke runs

```math
k^{*}(p) = \arg\max_{k}\, E_{s,k}(p)
\qquad\text{(ties resolve to the lowest } k)
```

```math
\psi(p) = \theta_{s,k^{*}(p)} + \tfrac{\pi}{2}
\qquad\qquad
\mathbf{u}(p) = \big(\sin\psi(p),\, \cos\psi(p)\big)
```

$\theta$ is the **carrier** direction — the direction in which the filter's stripes alternate, which
is *across* the stroke, not along it. So the stroke itself runs at $\theta + 90°$, and $\mathbf{u}$
is the unit step in that direction, written $(\Delta y, \Delta x)$ to match image indexing.

**Why the single strongest direction, rather than all of them.** Taking a maximum over directions
lets weakly-responding off-orientation channels contribute: their flanks are nearly empty, so their
ratio below is ill-conditioned and produces spurious asymmetry. Restricting to the locally dominant
channel — as an end-stopped cell in cortex does — gave a measured $10.4\times$ separation between a
line end and a line interior, against $2.5\times$ for the maximum-over-directions version.

### Step 2 — compare the two flanks

With $e(p) = E_{s,k^{*}(p)}(p)$ the response at the pixel itself, and sampling bilinearly with zero
outside the image,

```math
E_{\pm}(p) = E_{s,k^{*}(p)}\big(p \pm d_s\, \mathbf{u}(p)\big)
```

```math
A_{2,s}(p) = e(p) \cdot \frac{\big|E_{+}(p) - E_{-}(p)\big|}{E_{+}(p) + E_{-}(p) + \kappa\, e(p)}
\qquad\text{if } e(p) > \varepsilon, \text{ else } 0
```

Both probes read **the same channel** $k^*$ as the centre pixel — the question is whether *this*
stroke continues, not whether anything at all is over there.

**The ratio.** In a stroke's interior $E_+ \approx E_-$, the numerator vanishes and $A_2 \approx 0$.
At a termination one flank is empty, the numerator is as large as the denominator can make it, and
the ratio approaches $1$. For an isolated blob *both* flanks are empty and it returns to $0$ — which
is correct, since a blob is not a stroke end.

**Why $\kappa e$ and not a fixed constant.** The denominator must be stabilised where both flanks
are near zero. An absolute $\varepsilon$ there made the ratio saturate at $1$ wherever there was any
energy at all, collapsing $A_2$ to a copy of plain energy. Scaling the stabiliser by the centre
response instead means the conditioning is always in proportion to the signal, at every contrast.

**Why the leading $e(p)$.** Without it $A_2$ would be a bare ratio, bounded and scale-free, and
averaging it over a region would weight blank pixels as heavily as strong contours — the same defect
that had to be fixed in §5. Multiplying by the centre response makes $A_2$ an energy, so its average
is meaningful and $0$ is its correct value in blank regions.

**Where the probe distance comes from.** $d_s = \gamma \sigma_{\parallel,s}$ places the flanks one
along-contour envelope out, so they sit just beyond the filter's own excitatory region. A fixed
pixel offset would be degenerate at coarse scales, where both flanks would fall inside a single
envelope and the asymmetry would vanish by construction. This anchoring reproduces the published
tables, but it is **not scale-free**: measured in dimensionless units the best offset tracks the
stroke width rather than the filter's envelope, so `d_anchor = :structure` exists as the alternative
for new datasets.

### Step 3 — average and report

```math
\text{feature} = \sqrt{\big\langle A_{2,s} \big\rangle_c}
```

## 5. Branching — 3 numbers per scale per cell

**The idea.** $A_1$ can say *two directions meet here* but not *how many strokes leave here*,
because direction is $\pi$-periodic and a T and an X have identical direction content. Counting
branches is a $2\pi$ question. The trick is to **step away from the point**: east and west read
different pixels even though they are the same direction.

**Average first, then divide.**

### Step 1 — build a ring of probes

```math
\varphi_j = \frac{2\pi (j-1)}{K}, \qquad j = 1 \dots K, \qquad K = 2 n_s
\qquad\text{— a full turn, not a half turn}
```

```math
\text{ch}(\varphi) = \text{the } k \text{ whose } \theta_{s,k} \text{ is closest to }
(\varphi + \tfrac{\pi}{2}) \bmod \pi
```

```math
R_s(p, \varphi_j) = E_{s,\, \text{ch}(\varphi_j)}\Big(p + d_s\big(\sin\varphi_j,\, \cos\varphi_j\big)\Big)
```

$R$ asks: *is there a contour at distance $d_s$ in direction $\varphi$, and is it oriented along
$\varphi$?* Both halves matter. Reading the response at an offset recovers the full turn; requiring
the orientation there to match the direction of travel is what makes it a **branch** rather than any
passing contour. The $+\tfrac{\pi}{2}$ in $\text{ch}$ converts a direction of travel into the
carrier that a stroke running that way would excite.

$R$ therefore has one lobe per branch leaving the point.

### Step 2 — summarise the ring by its circular harmonics

```math
a_{0,s}(p) = \sum_{j=1}^{K} R_s(p,\varphi_j)
\qquad
a_{1,s}(p) = \left|\sum_{j=1}^{K} R_s(p,\varphi_j)\, e^{i\varphi_j}\right|
\qquad
a_{2,s}(p) = \left|\sum_{j=1}^{K} R_s(p,\varphi_j)\, e^{i2\varphi_j}\right|
```

This is the same Fourier idea as §1, on the ring instead of the orientation profile — but here the
domain really is a full $2\pi$, so the harmonics run in $\varphi, 2\varphi$ rather than
$2\theta, 4\theta$.

$a_0$ is the total, roughly proportional to how many branches there are. $a_1$ measures **one-sided
asymmetry**: it is large when the branches are lopsided and zero when they balance. $a_2$ measures
two-fold structure. Magnitudes are taken per pixel, so the phases — which depend on how the figure
happens to be rotated — are discarded and what remains is rotation-invariant.

Idealised values, for a point with $m$ equally spaced branches:

| configuration | $a_0$ | $a_1/a_0$ | $a_2/a_0$ |
|:--|--:|--:|--:|
| endpoint, 1 branch | 1 | 1 | 1 |
| straight line, 2 opposite | 2 | 0 | 1 |
| corner, 2 branches at $90°$ | 2 | $1/\sqrt{2}$ | 0 |
| T-junction, 3 branches | 3 | $1/3$ | $1/3$ |
| X-crossing, 4 branches | 4 | 0 | 0 |

The first column is the count; the two ratios are the type signature. Note that a straight line and
an X-crossing agree on $a_1/a_0$ and differ only in $a_0$ — the count really is doing work.

### Step 3 — average, then divide

```math
f_s = 10^{-3}\cdot \frac{1}{n_c}\sum_{c} \big\langle a_{0,s} \big\rangle_c
\qquad\text{a floor set by the image itself}
```

```math
\text{features} = \big\langle a_{0,s}\big\rangle_c,
\qquad
\frac{\big\langle a_{1,s}\big\rangle_c}{\big\langle a_{0,s}\big\rangle_c + f_s},
\qquad
\frac{\big\langle a_{2,s}\big\rangle_c}{\big\langle a_{0,s}\big\rangle_c + f_s}
```

**The ordering here was a bug and is now the fix.** Forming $a_1/a_0$ at each pixel and averaging
the result seems natural and is wrong. The ratio is a *bounded, scale-free* quantity with no limit
as $a_0 \to 0$: where there is no contour, there is no asymmetry to measure, and any value written
there is fiction. The original code wrote $0$, which asserts "perfectly symmetric" — a strong claim
— at every blank pixel. Averaging those zeros then made the feature scale with **how much of the
picture was blank**, so it reported ink coverage as much as branch structure.

Dividing after averaging avoids all of this: both numerator and denominator are energies, both
average meaningfully, and the quotient is defined everywhere.

**Contrast this with $A_1$ and $A_2$**, which divide *at the pixel*. The difference is that those
two multiply by an energy afterwards, so the result is energy-like and tends to zero where there is
nothing. A quantity is safe to average when zero is its true limit in empty regions; it is not safe
when it is a bounded ratio. That distinction — **energy-like versus scale-free** — is the general
rule, and it is why the same treatment is not needed in §3 and §4.

**The floor $f_s$** is relative, a thousandth of the image's own mean $a_0$, so an empty cell tends
smoothly to zero instead of through a branch. An absolute floor would mean something different at
every contrast.

## 6. Strongest-anywhere — 3 numbers per scale

**Maximum instead of average**, over the whole picture, independent of the grid.

```math
\text{features} = \max_p A_{1,s}(p), \qquad \max_p A_{2,s}(p), \qquad \max_p a_{0,s}(p)
```

Note the absence of $\sqrt{\cdot}$ and of any cell index: these are single numbers per scale.

## 7. The corner-strength floor $c_s$

A perfectly straight line should give $A_1 = 0$, and does not, because the channels have angular
width. For a single-direction input at angle $\theta_0$ the radial part of the filter is common to
every channel and cancels in the ratio, so each channel's response is fixed by angle alone:

```math
\Delta_k = \min\big(|\theta_{s,k}-\theta_0|, \pi - |\theta_{s,k}-\theta_0|\big)
\qquad
\tilde{E}_k = \exp\left(-\left(\frac{\Delta_k}{\sigma_{\varphi,s}}\right)^{2}\right)
```

```math
\tilde{C}_0 = \sum_k \tilde{E}_k
\qquad
\tilde{S} = \sum_k \tilde{E}_k \tilde{E}_{k+h}
\qquad
c_s = \max_{\theta_0} \frac{\tilde{S}}{\tilde{C}_0^{2}}
```

The exponent is squared rather than halved because energy is amplitude squared.

This gives $6.64\times10^{-3}, 2.40\times10^{-5}, 9.12\times10^{-9}$. The dominant term is **not**
the channel $90°$ from the line — that sits $3\sigma_\varphi$ away and contributes
$e^{-9} \approx 1.2\times10^{-4}$ — but the **pair straddling the line at $\pm 45°$**, each
$1.5\sigma_\varphi$ away with $e^{-2.25} = 0.105$ of the energy and exactly $90°$ apart, so their
product is $0.105^2 = 1.1\times10^{-2}$, two orders larger. The closed form matches measurement to
within 2 % at every scale (`RationalGaborFeatures/Validate_i1D.jl`).

---

## Feature count

Per cell, per scale: 5 orientation $+$ 1 corner $+$ 1 stroke-end $+$ 3 branching $=$ **10**. Plus
one brightness number per cell, and three strongest-anywhere numbers per scale.

```math
\text{total} = n_c (10 S + 1) + 3S
```

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

```math
\big\langle f g \big\rangle \neq \big\langle f \big\rangle \big\langle g \big\rangle,
\qquad\qquad
\big\langle f g \big\rangle - \big\langle f \big\rangle\big\langle g \big\rangle
= \mathrm{Cov}_p(f, g)
```

That covariance within the window is direct evidence that $f$ and $g$ were large *at the same
place*, and it is destroyed by averaging first. Corner strength, stroke-end strength and the
branching ring all take the left-hand form. The orientation summary takes the right-hand one, and
has never been compared against the alternative.
