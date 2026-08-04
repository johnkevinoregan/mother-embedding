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

## Caveats

The angular equivalence for $Z^{(2m)}$ is stated as $\approx$ rather than $=$: it is exact only if
the angular windows tile perfectly and the annulus is angle-independent, which holds to the accuracy
of the tiling rather than to machine precision. Only the $T$ identity was verified numerically here.

None of this has been tested as an alternative *implementation*. The claim is about what the
features are, not a proposal to compute them differently — and the three obstacles above say the
shortcut is not available at the grids and scales actually in use.
