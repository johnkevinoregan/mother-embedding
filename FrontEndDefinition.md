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
| `I(p)` | the image, at pixel `p = (y, x)`; `W` is its width in pixels (112 throughout) |
| `s = 1 … S` | **scale** index. Production: `S = 3` |
| `ρ_s` | scale `s` in **cycles across the image**. Production: `ρ = 2.00, 3.742, 7.00` |
| `λ_s = W / ρ_s` | the same scale as a **wavelength in pixels**: `56.0, 29.9, 16.0` |
| `β_s` | radial bandwidth of scale `s` in octaves: `2.0, 1.6, 1.2` |
| `n_s` | number of **directions** at scale `s`: `8, 12, 16` |
| `k = 1 … n_s` | direction index |
| `θ_{s,k} = (k−1)·π / n_s` | the direction of channel `k`. Spans `[0, π)`: a line and the same line turned by 180° are the same line |
| `t = 0.75` | `dtheta_on_sigma`, one global constant |
| `σφ_s = (π / n_s) / t` | angular width of each channel: `30°, 20°, 15°` |
| `G_{s,k}` | the filter for scale `s`, direction `k`, held in the frequency domain |
| `G_lp` | one extra non-directional low-pass filter, at `ρ_lp = min(ρ_s)/2 = 1.0` |

### The responses everything is built from

The image is padded to a field of `H×W` by copying its border outward, transformed once, multiplied
by each filter, and transformed back:

```
r_{s,k}(p)  =  F⁻¹[ F[I] · G_{s,k} ] (p)              complex
E_{s,k}(p)  =  | r_{s,k}(p) |²                        real, ≥ 0
E_lp(p)     =  | F⁻¹[ F[I] · G_lp ] (p) |²
```

`E_{s,k}(p)` is the **filter response at a pixel**: how strongly a stroke of size `λ_s` running in
direction `θ_{s,k}` is present at `p`.

#### Why `r` is complex, and why that matters

For a **real** image the Fourier transform is redundant: the value at frequency `−f` is the complex
conjugate of the value at `+f`, so the negative half of the frequency plane mirrors the positive
half and carries nothing new.

An ordinary filter — one you could write down as a real array of numbers in the image — keeps
**both** halves and produces a real output. `G_{s,k}` is **one-sided**: it is zero on one half of
the frequency plane and keeps only the other, which is why `r` comes out complex. The half is
picked by direction, in `GaborStack.module.jl`:

```julia
dφ      = atan(sin(PHI - θ0), cos(PHI - θ0))
angular = abs(dφ) > π/2 ? 0.0 : exp(-dφ^2 / (2σφ^2))
```

— zero wherever the frequency's direction is more than 90° away from `θ0`.

The complex result is the **analytic signal**, Gabor's own term from 1946. In polar form
`r(p) = a(p)·exp(i·ϕ(p))`, its modulus `a(p) = |r(p)|` is the **envelope** — how much of that
frequency band is present — and `ϕ(p)` is the **local phase**, whereabouts you sit within the
oscillation.

Three consequences, and all three are load-bearing:

**`|r|` does not oscillate.** A real bandpass filter's output swings positive and negative as the
underlying wave rises and falls, so "how much structure is here" flickers with exactly where you
sample. `|r|` is the smooth outline of that swing rather than the swing itself.

**Exact quadrature, for free.** The classic construction uses two filters — an even-symmetric one
responding to bar-like features and an odd-symmetric one responding to edge-like features, 90° out
of phase — and forms `even² + odd²`. Here `Re(r)` *is* the even response and `Im(r)` *is* the odd
one, exactly 90° apart by construction, rather than approximately so because two spatial kernels
were built separately and hoped to match.

**Contrast-polarity invariance is exact.** Invert the image, `I → −I`, and `r → −r`, so `E = |r|²`
is identically unchanged. Not approximately — identically. This is the property behind the
front end scoring −0.06 on polarity and transferring intact across a polarity flip, where frozen
ConvNeXt reads polarity at 0.998 and collapses.

### Averaging

`w_c(p)` is a Gaussian window for cell `c`, normalised so `Σ_p w_c(p) = 1`. With a `g × g` grid
there are `n_c = g²` cells; at `g = 1` the single window covers the whole picture.

```
⟨f⟩_c  =  Σ_p  w_c(p) · f(p)                   weighted average of a map over cell c
```

`⟨·⟩` is the only averaging operator below. Where a subscript is dropped, read `⟨·⟩` for every cell.

### Constants

| symbol | value | used in |
|:--|--:|:--|
| `κ` | 0.5 | stroke-end stabiliser |
| `ε` | 10⁻¹² | guard where a denominator can vanish |
| `d_factor` | 1.0 | probe distance multiplier |
| `σ_along,s = W / (2π ρ_s σφ_s)` | 17.0, 13.6, 9.7 px | the filter's extent **along** a contour |
| `d_s = d_factor · σ_along,s` | 17.0, 13.6, 9.7 px | how far the stroke-end and branching probes step out |
| `c_s` | 6.64e−3, 2.40e−5, 9.12e−9 | corner-strength floor, defined at the end |

---

## 1. Orientation summary — 5 numbers per scale per cell

**Average first, then combine.**

```
Ē_{s,k,c}  =  ⟨ E_{s,k} ⟩_c                                    average each direction

T_{s,c}    =  Σ_{k=1}^{n_s}  Ē_{s,k,c}                         total over directions

Z₂_{s,c}   =  Σ_{k=1}^{n_s}  Ē_{s,k,c} · exp( i·2·θ_{s,k} )    twice round per turn
Z₄_{s,c}   =  Σ_{k=1}^{n_s}  Ē_{s,k,c} · exp( i·4·θ_{s,k} )    four times round per turn

z₂ = Z₂ / T        z₄ = Z₄ / T                                 (zero if T = 0)
```

Features: `√T`, `Re z₂`, `Im z₂`, `|z₂|`, `|z₄|`.

`exp(i·2θ)` rather than `exp(iθ)` because direction is defined modulo `π`: doubling the angle makes
a direction and its opposite land on the same point of the circle. `|z₂|` is 0 when all directions
are equally present and 1 when only one is; `arg z₂ / 2` is the dominant direction.

## 2. Overall brightness — 1 number per cell

```
feature  =  √( ⟨ E_lp ⟩_c )
```

## 3. Corner strength `A₁` — 1 number per scale per cell

**Combine at the pixel, then average.** `h = n_s / 2`, so shifting the direction index by `h` is
exactly a 90° turn. Indices wrap modulo `n_s`.

```
C₀_s(p)  =  Σ_{k=1}^{n_s}  E_{s,k}(p)                          all directions at this pixel

S_s(p)   =  Σ_{k=1}^{n_s}  E_{s,k}(p) · E_{s,k+h}(p)           each direction × the one at 90°

A₁_s(p)  =  max( 0 ,  S_s(p)/C₀_s(p)  −  c_s · C₀_s(p) )       if C₀_s(p) > ε, else 0

feature  =  √( ⟨ A₁_s ⟩_c )
```

The product `E_k · E_{k+h}` is large only if **both** are large, which is what makes this a
statement about one pixel rather than about a region. Dividing by `C₀` turns a squared quantity
back into an energy. The `− c_s·C₀` term is the correction of §7.

## 4. Stroke-end strength `A₂` — 1 number per scale per cell

**Combine at the pixel, then average.**

```
k*(p)     =  argmax_k  E_{s,k}(p)                      strongest direction; ties → lowest k

ψ(p)      =  θ_{s,k*(p)} + π/2                         the stroke runs at right angles
                                                       to the filter's carrier
u(p)      =  ( sin ψ(p) , cos ψ(p) )                   step vector, (Δy, Δx)

e(p)      =  E_{s,k*(p)}(p)
E₊(p)     =  E_{s,k*(p)}( p + d_s · u(p) )             bilinear; 0 outside the image
E₋(p)     =  E_{s,k*(p)}( p − d_s · u(p) )

A₂_s(p)   =  e(p) · | E₊(p) − E₋(p) |
                    ─────────────────────────────      if e(p) > ε, else 0
                    E₊(p) + E₋(p) + κ · e(p)

feature   =  √( ⟨ A₂_s ⟩_c )
```

The ratio is 0 where the stroke continues both ways and near 1 at a termination. `κ·e` is a
*relative* stabiliser: an absolute one collapsed `A₂` to plain energy. The leading `e(p)` makes the
result an energy, so averaging it is meaningful.

## 5. Branching — 3 numbers per scale per cell

**Average first, then divide.** `K = 2·n_s` probe directions.

```
φ_j       =  2π (j−1) / K ,          j = 1 … K         a full turn, not a half turn

ch(φ)     =  the k with θ_{s,k} closest to (φ + π/2) mod π

R_s(p, φ_j)  =  E_{s, ch(φ_j)}( p + d_s · ( sin φ_j , cos φ_j ) )       bilinear
```

`R` asks: *is there a contour at distance `d_s` in direction `φ`, oriented along `φ`?* The spatial
step is what recovers a full turn from directions that only span half of one — east and west read
different pixels.

```
a₀_s(p)   =  Σ_{j=1}^{K}  R_s(p, φ_j)                                  ring total
a₁_s(p)   =  | Σ_{j=1}^{K}  R_s(p, φ_j) · exp( i·φ_j ) |               varies once round
a₂_s(p)   =  | Σ_{j=1}^{K}  R_s(p, φ_j) · exp( i·2φ_j ) |              varies twice round

f_s       =  10⁻³ · mean over cells of ⟨a₀_s⟩          a floor set by the image itself

features  =  ⟨a₀_s⟩_c ,   ⟨a₁_s⟩_c / (⟨a₀_s⟩_c + f_s) ,   ⟨a₂_s⟩_c / (⟨a₀_s⟩_c + f_s)
```

The two ratios are formed **after** averaging, not per pixel. Per pixel they are bounded quantities
with no limit as `a₀ → 0`, so writing 0 there asserted "perfectly symmetric" wherever there was no
evidence, and averaging those zeros made the result scale with how much of the picture was blank.

Idealised values, for a point with `m` equally spaced branches: `a₁/a₀ = 0` for `m = 2` opposite or
`m = 4`; `= 1/√2` for two branches at 90°; `= 1/3` for a T.

## 6. Strongest-anywhere — 3 numbers per scale

**Max instead of average**, over the whole picture, independent of the grid.

```
features  =  max_p A₁_s(p) ,    max_p A₂_s(p) ,    max_p a₀_s(p)
```

Note the absence of `√·` and of any cell index: these are single numbers per scale.

## 7. The corner-strength floor `c_s`

A perfectly straight line should give `A₁ = 0`, and does not, because the channels have angular
width. For a single-direction input at angle `θ₀` the radial part of the filter is common to every
channel and cancels, so the response of channel `k` is fixed by angle alone:

```
Δ_k       =  circular distance on [0, π) between θ_{s,k} and θ₀
Ẽ_k       =  exp( − (Δ_k / σφ_s)² )                    energy, so amplitude squared

C̃₀        =  Σ_k Ẽ_k
S̃         =  Σ_k Ẽ_k · Ẽ_{k+h}

c_s       =  max over θ₀ of   S̃ / C̃₀²
```

Giving `6.64e−3, 2.40e−5, 9.12e−9`. The dominant term is **not** the channel 90° from the line —
that sits `3σφ` away and contributes `exp(−9) ≈ 1.2e−4` — but the **pair straddling the line at
±45°**, each `1.5σφ` away with `exp(−2.25) = 0.105` of the energy and exactly 90° apart, so the
product is `0.105² = 1.1e−2`. Closed form matches measurement to within 2 % at every scale
(`RationalGaborFeatures/Validate_i1D.jl`).

---

## Feature count

Per cell, per scale: 5 orientation + 1 corner + 1 stroke-end + 3 branching = **10**.
Plus one brightness number per cell, and three strongest-anywhere numbers per scale.

```
total  =  n_c · ( 10·S + 1 )  +  3·S
```

| configuration | scales | grid | total |
|:--|--:|--:|--:|
| baseline | 3 | 1 | 31 |
| baseline | 3 | 3 | 279 |
| + strongest-anywhere | 3 | 1 | 40 |
| + λ = 8 px scale | 4 | 1 | 41 |
| + λ = 8 px and strongest-anywhere | 4 | 1 | 53 |

---

## Where each operation happens

| block | at the pixel | then |
|:--|:--|:--|
| orientation summary | — | average, **then** combine across directions, then divide |
| brightness | — | average |
| corner strength | multiply pairs at 90°, divide by the pixel total, subtract floor | average |
| stroke-end | pick strongest direction, compare two probes, divide, multiply by centre | average |
| branching | build the ring, take three circular sums | average each, **then** divide |
| strongest-anywhere | (maps as above) | **maximum** |

Two orderings are in tension and the difference is the project's central claim: `⟨f·g⟩ ≠ ⟨f⟩·⟨g⟩`,
and the gap between them is `Cov_p(f, g)` within the window — direct evidence that `f` and `g`
were large *at the same place*. Corner strength, stroke-end strength and the branching ring all
take the left-hand form. The orientation summary takes the right-hand one, and has never been
compared against the alternative.
