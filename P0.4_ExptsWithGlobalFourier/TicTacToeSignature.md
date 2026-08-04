# A tic-tac-toe Fourier signature — findings

*Does the first two or three 2D Fourier coefficients of each cell of an N×N grid give
a usable signature of "vertical / horizontal / diagonal stroke", "blob", "loop" in
that area? Companion to `TicTacToeFourierSignature.jl`, which reproduces every number
below.*

---

## The construction

The 112×112 letter is tiled by an N×N grid (3×3 by default, cell 37 px, stroke width
≈ 13 px). Each cell is windowed (Gaussian, σ = P/4, patch P = 1.3 × cell so windows
overlap slightly) and its low-order spectrum summarised by **moments**, not read
coefficient by coefficient:

| symbol | definition | reads as |
|:--|:--|:--|
| `a₀` | `F(0,0)` | ink fraction |
| `ac` | `√Σ_{≠0}\|F\|²` | how much structure |
| `E₂` | `Σ\|F\|²e^{2iθ} / Σ\|F\|²` | `\|E₂\|` anisotropy, `arg(E₂)/2 + 90°` stroke angle |
| `E₄` | `Σ\|F\|²e^{4iθ} / Σ\|F\|²` | intended "two strokes ⟂"; **fails** |
| `e₁,e₂,e₃` | AC power share in rings `ω = 1,2,3` | radial profile, loop score `e₂/(e₁+e₂)` |

`E₂` is the orientation tensor of the power spectrum — the π-periodic (encode-2θ)
quantity the project's design rules ask for. `(Re E₂, Im E₂)` is the continuous
2-vector; the letter-shaped labels in the notebook are display only.

**Two samplings of the same coefficients.** `lattice` = the literal DFT orders
`(v,u) ∈ [−K,K]²`. `polar` = the same integral evaluated at `ω ∈ {1,2,3}` × 12
orientations. They classify equally well, but only `polar` reads angles correctly.

---

## 1. Orientation — works

A bar swept through 12 angles in the centre cell, true vs. reported:

| sampling | max angular error | `\|E₂\|` of one bar |
|:--|--:|:--|
| polar (ω×θ) | **0.16°** | 0.51 – 0.57, flat in angle |
| lattice (v,u), K=3 | **3.48°** | 0.61 – 0.72, peaks at 0/90° |

The lattice error is systematic and pulls toward **0/45/90/135°** — the square
sampling grid's own 4-fold symmetry leaking into the 2nd harmonic. Use polar to read
an angle.

`|E₂|` behaves as an "is there one dominant stroke here" gauge (polar, centre cell):

| figure | `a₀` | `\|E₂\|` | `\|E₄\|` | `e₁` | `e₂` | stroke° |
|:--|--:|--:|--:|--:|--:|--:|
| vertical bar | 0.424 | 0.511 | 0.179 | 0.811 | 0.168 | 90 |
| horizontal bar | 0.424 | 0.511 | 0.179 | 0.811 | 0.168 | 180 |
| diagonal `/` | 0.409 | 0.572 | 0.233 | 0.783 | 0.185 | 135 |
| diagonal `\` | 0.409 | 0.572 | 0.233 | 0.783 | 0.185 | 45 |
| plus | 0.668 | **0.000** | 0.085 | 0.928 | 0.055 | — |
| X | 0.662 | **0.000** | 0.142 | 0.898 | 0.080 | — |
| disc r=15 | 0.580 | **0.000** | 0.000 | 0.970 | 0.015 | — |
| ring R=15 | 0.439 | **0.000** | 0.009 | 0.429 | 0.570 | — |

## 2. `|E₄|` does not detect crossings

The hope was that two strokes ~90° apart would ring the 4th harmonic. Measured
(polar): **single bar 0.18–0.23, plus 0.085, X 0.142** — a *single* stroke has more
`|E₄|` than a crossing. The crossing region radiates broadband isotropic energy that
dilutes the harmonic faster than the second stroke adds to it. Use `|E₂| ≈ 0` **with**
high `ac` as the "something non-oriented is here" cue instead. On the lattice `|E₄|`
looks more informative (η² ≈ 0.30) but most of that is *lattice alignment*, not shape.

## 3. Loop vs blob — only when the cell matches the figure

With a disc and a ring of the **same outer size**, both filling the cell, the radial
profile separates them cleanly — `e₂/e₁`: **disc 0.015, ring 1.33**, a ×90 difference.
That is the `J₀` zero of a ring's transform falling between rings 1 and 2, exactly as
Bessel predicts (ring first zero at `ωR ≈ 0.38`, disc at `0.61`).

But the ratio confounds *hollowness* with *object size relative to the cell*:

- at N=1 (patch = whole image) the **same** disc scores 0.52 and the **same** ring
  0.39 — **inverted**;
- ring stroke radius 4 → loop score 0.57; radius 6 → **0.41**. Same ring, different
  answer;
- on real EMNIST at 3×3 the centre-cell loop score does **not** pick out `O`
  (mean 0.20, vs `C` 0.26 and `F` 0.26) — an EMNIST `O`'s loop spans the whole grid,
  not one cell;
- at N=1 the whole-image score *does* separate `O` from the rest (**AUC 0.80**) but
  with the **opposite sign** to the loop hypothesis: it is reading object scale and
  smoothness, not enclosure.

**Conclusion:** loops are not available from a low-order radial profile unless the
cell is sized to the loop. Enclosure is a topological property; this is a scale
measure that happens to correlate with it when everything else is held fixed.

## 4. The 9-cell signature is strongly diagnostic of letter identity

Protocol identical to `P0.3_New_Gabor_FPE/KeyPointDiagnosticity.md`: 360 EMNIST instances,
12 classes (`O C I L T X K A H Y E F`), 30 each, leave-one-out **nearest-class-mean**
on the standardised vector. Chance = 8.3 %. Lattice sampling, K=3, 3×3 unless noted.

| descriptor | numbers | LOO accuracy |
|:--|--:|--:|
| ink only (`a₀` per cell) | 9 | 56.7 % |
| **orientation only (`Re E₂, Im E₂` per cell)** | **18** | **75.3 %** |
| **ink + orientation** | **27** | **76.1 %** |
| ink + orientation + loop score | 36 | **76.4 %** |
| all 9 features per cell | 81 | 74.2 % |
| all 9, but **no grid** (1×1) | 9 | 49.7 % |
| all 9, 2×2 grid | 36 | 71.4 % |
| all 9, 4×4 grid | 144 | 77.8 % |
| all 9, 3×3, **polar** sampling | 81 | 73.3 % |
| all 9, 3×3, bounding-box normalised | 81 | 72.8 % |
| *previous best in this project (global shape harmonics `\|M1..6\|` + radial)* | *~20* | *≈ 61 %* |

Per-feature η² (3×3, lattice, mean over the 9 cells): `ReE2` 0.426 · `ac` 0.409 ·
`e1` 0.404 · `a0` 0.389 · `ImE2` 0.361 · `|E4|` 0.300 · `e2` 0.270 · `e3` 0.262 ·
`|E2|` 0.235.

Three things to read off this:

- **The grid does the work.** The same nine numbers over the whole image score
  49.7 %; over a 3×3 grid, 74.2 %.
- **Two numbers per cell are enough.** Adding `ac`, `|E₄|` and the ring profile
  *lowers* the unweighted nearest-mean score by diluting good dimensions with noisy
  ones — the same artifact documented in `KeyPointDiagnosticity.md`; η²-weighting
  recovers it (74.2 % → 77.2 %).
- **Bounding-box normalising does not help** (72.8 % vs 74.2 %) — EMNIST is already
  size-normalised, so the extra crop only adds jitter.

## 5. Answer to the original question

*Vertical / horizontal / diagonal per area*: **yes** — cleanly, from two numbers per
cell, and those two numbers are already a π-periodic orientation code ready to bind
as FPE with integer frequencies.

*Blob vs loop*: **no**, not at this grid scale and not from a low-order radial
profile in general.

The unplanned result is §4: a crude 3×3 orientation signature (18–27 numbers)
outperforms every local descriptor tried in this project so far (≈ 61 % → ≈ 76 %).
That says the information separating letters is **where the oriented strokes are**,
not what type of junction sits at each keypoint.

## 6. Relation to the Gabor front end

`F(ω,θ)` under a Gaussian window **is** a Gabor response (see
`LocalFourierPatches.jl`), so the whole construction is: run a Gabor bank at 3 scales
× 12 orientations, pool energy over 9 large regions, take the 2nd circular harmonic
over orientation in each. Nothing here *needs* the Fourier framing — but the Fourier
framing is what makes "keep the first two or three coefficients" the natural move.
