# What the orientation block actually is

*A note on the relation between the front end's orientation summary and a plain 2D Fourier
analysis of the same cell. Nothing here is a new experiment; it is an identity, verified
numerically, plus what follows from it.*

---

## The claim

The five numbers the front end reports per scale per cell —

```math
\sqrt{T_{s,c}}, \quad \mathrm{Re}(z_2), \quad \mathrm{Im}(z_2), \quad |z_2|, \quad |z_4|
```

— **are the zeroth, first and second angular harmonics of the local power spectrum in a radial
annulus.** They are a polar decomposition of $|\hat I(f)|^2$: radial bins × angular harmonics. No
Gabor bank is needed to obtain them.

## Why

Parseval. With $r_{s,k} = \mathcal{F}^{-1}[\hat I \cdot G_{s,k}]$,

```math
\big\langle |r_{s,k}|^2 \big\rangle \;=\; \frac{1}{n^2}\sum_f |\hat I(f)|^2\,|G_{s,k}(f)|^2
```

Summing over the orientations at one scale,

```math
T_s \;=\; \sum_k \big\langle |r_{s,k}|^2 \big\rangle
      \;=\; \frac{1}{n^2}\sum_f |\hat I(f)|^2 \left[\sum_k |G_{s,k}(f)|^2\right]
```

and the bracket is the bank's **radial annulus**: the angular windows tile, so their squared sum is
very nearly independent of angle. The same argument on the harmonics gives

```math
Z^{(2m)}_s \;\approx\; v_m \sum_f |\hat I(f)|^2\, A_s(|f|)^2\, e^{i 2 m \varphi_f}
```

where $v_m$ is the angular window's own $m$-th Fourier coefficient. So $Z^{(2)}$ and $Z^{(4)}$ are
the first and second angular harmonics of the annular power, scaled by a known constant.

## Verified

`T` computed two ways on the same image — dense Gabor responses then spatial mean, versus one FFT
weighted by the annulus:

| scale | T from dense Gabors | T from the spectrum | rel. diff |
|:--|--:|--:|--:|
| λ=56 | 110.61465 | 110.61465 | 0.00e+00 |
| λ=30 | 40.046665 | 40.046669 | 9.5e−08 |
| λ=16 | 12.455797 | 12.455799 | 1.5e−07 |

Float32 precision. It is an identity, not an approximation.

```julia
F, (oy, ox) = GaborStack.embed(img, bank.size; mode=:replicate); Ff = fft(F)
ks  = [k for k in 1:length(bank.meta) if bank.meta[k].rho0 == ρ]
tg  = sum(mean(abs2, ifft(Ff .* bank.filters[k])) for k in ks)      # what the front end does
ann = sum(abs2.(bank.filters[k]) for k in ks)
tf  = sum(abs2.(Ff) .* ann) / length(F)^2                           # one FFT, no Gabors
```

The spectral route is also far cheaper: **1 forward FFT and some weighted sums, against 1 forward
plus 57 inverse FFTs.** Roughly 50× for that block.

---

## So why not take the shortcut

**The cell is smaller than the coarsest wavelength.** At grid 3 a cell is 112/3 ≈ **37 px** while
λ = 56 px. A 37-px patch's DFT has fundamental frequency 1/37 cycles per patch; a 56-px structure
does not fit inside it and cannot be recovered from it. The Gabor route convolves over the *whole*
image and only then averages over the cell, so a pixel's response legitimately reflects structure
extending beyond the cell boundary. This is not a technicality — it is the entire coarse half of
the ladder.

**A local average of energy is not the spectrum of a truncated patch.** The identity above holds
over the full domain. Restricted to a sub-region you are transforming a rectangularly windowed
patch, with the leakage that implies, whereas $\langle |r|^2 \rangle_c$ is a genuine local energy
average with no cell-boundary artefact.

**Everything else needs per-pixel maps.** `A₁`, `A₂`, the ray transform and the spatial max all
operate *pointwise*, before pooling — the "combine before averaging" ordering the project rests on.
A cell-level FFT produces no per-pixel map, so none of those operators can exist.

---

## What this reframes

**The Gabor machinery is paid for by the conjunction, ray and max layers. The orientation summary
is a by-product.** Once the 57 inverse FFTs have been done for those, $\sqrt{T}, z_2, z_4$ cost
essentially nothing. If the orientation block were all we wanted, ~98 % of the compute would be
waste.

And it is worth naming what that block *is*, because this project already met it. A polar-binned
local power spectrum — radial annuli × angular harmonics — is very nearly the 3×3 "tic-tac-toe"
Fourier signature of `P0.4_ExptsWithGlobalFourier/`, which the root `README.md` records as *"the
reference feature set that `P0-8_RationalGaborFeatures` had to beat"*. Polar rather than Cartesian
bins, and multi-scale rather than single-scale, but the same object.

**So the orientation block is not where this front end differs from the Fourier baseline it was
built to beat.** The difference is entirely in the pointwise nonlinear layer above it — which is
precisely the part Phases 5–8 found adds ≈ 0 on EMNIST, and Phase 9 found does earn its place on
graded properties.

That is a sharper statement of what the front end's actual contribution is than anything currently
written down in `PHASES.md` or the phase `RESULTS.md` files.

---

## What the higher harmonics measure

We keep $m = 0, 1, 2$ — that is $T$, $Z^{(2)}$ and $Z^{(4)}$. What would $Z^{(6)}$ and above add?

Not "more detail". The harmonics work by **cancellation**. For $K$ orientations equally spaced by
$\pi/K$,

```math
c_m \;\propto\; \sum_{j=0}^{K-1} e^{-i2m(\theta_0 + j\pi/K)} \;=\; 0
\quad\text{unless}\quad m \equiv 0 \pmod K
```

so harmonic $m$ is silent on any arrangement whose symmetry order does not divide it. Measured at
the junction point of ray figures, $|c_m|/|c_0|$ (one scale, 36 orientations):

| figure | orientations | m=1 ($Z^{(2)}$) | m=2 ($Z^{(4)}$) | **m=3 ($Z^{(6)}$)** | m=4 | m=5 | m=6 |
|:--|:--|--:|--:|--:|--:|--:|--:|
| 1 ray (stroke end) | {0} | 0.766 | 0.590 | 0.403 | 0.261 | 0.175 | 0.143 |
| straight line | {0} | 0.985 | 0.941 | 0.871 | 0.781 | 0.678 | 0.567 |
| L-corner, 90° | {0,90} | 0.133 | 0.606 | 0.104 | 0.253 | 0.034 | 0.137 |
| T-junction | {0,90} | 0.429 | 0.826 | 0.493 | 0.621 | 0.444 | 0.452 |
| X-crossing, 90° | {0,90} | **0.000** | 0.832 | **0.000** | 0.740 | **0.000** | 0.585 |
| **Y-junction, 120°** | {0,60,120} | 0.031 | **0.005** | **0.365** | 0.045 | 0.001 | 0.076 |
| 6-ray star, 60° | {0,60,120} | 0.002 | 0.059 | **0.477** | 0.002 | 0.028 | 0.543 |

**They are junction-*angle* detectors, not junction-*count* detectors.**

* $Z^{(2)}$ — is there a single dominant orientation. Vanishes as soon as two orientations are
  equally represented: 0.000 for the X.
* $Z^{(4)}$ — is there a **90°** arrangement. Large for L, T and X (0.61–0.83); **0.005 for the Y**.
* $Z^{(6)}$ — is there a **60°** arrangement. 0.365 for the Y; **0.000 for the X**. The exact
  complement of $Z^{(4)}$.

So $Z^{(4)}$ and $Z^{(6)}$ separate right-angle structure from Y-structure with no overlap, and
nothing below $m=3$ can express that distinction. It is genuinely new information rather than a
re-encoding.

**They also read orientation-peak width, which is curvature.** A single straight line excites
*every* harmonic with slow decay (0.985, 0.941, 0.871, 0.781 …), because a narrow bump in $\theta$
has broad Fourier content. A curved contour has a broadened peak, so its harmonics fall off faster,
and the ratio between successive harmonics reads the spread directly. That is why the Phase 12
capacity sweep found $C_6/C_8$ worth **+0.026 on curvedness and nothing elsewhere**: the stroke
dataset has almost no 60° junctions, so the angle-detection half of their value was never
exercised. The harmonics were tested on a dataset that could not show what they are for.

**Two things the table also shows.**

The **T breaks the pattern** — $m=1$ is 0.429 where L and X give ≈ 0, despite the same orientation
content {0°, 90°}. Its stem extends only one way, so the local energy is asymmetric. π-periodicity
constrains the orientation *labels*, not the *magnitudes*, and finite filters put genuine 2π
information into the magnitudes.

**Y and the 6-ray star are indistinguishable** — both peak at $m=3$ with everything else suppressed,
because three rays at 120° and six at 60° have identical orientation content mod π. That is the
π-periodicity limit in its purest form, and it is precisely what the ray transform exists to break.

---

## Caveats

The angular equivalence for $Z^{(2m)}$ is stated as $\approx$ rather than $=$: it is exact only if
the angular windows tile perfectly and the annulus is angle-independent, which holds to the accuracy
of the tiling rather than to machine precision. Only the $T$ identity was verified numerically here.

None of this has been tested as an alternative *implementation*. The claim is about what the
features are, not a proposal to compute them differently — and the three obstacles above say the
shortcut is not available at the grids and scales actually in use.
