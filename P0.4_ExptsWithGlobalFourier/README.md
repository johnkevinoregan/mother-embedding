# P0.4_ExptsWithGlobalFourier

Experiments with Fourier descriptions of a character, as an alternative to the
Gabor / ray-harmonic front end in `P0.3_New_Gabor_FPE/`. Same image format as the rest
of the project: EMNIST 28×28 upsampled to a 112×112 letter, embedded in a
**224×224 field** (border 56), stroke width ≈ 13 px; synthetic figures (bar,
diagonal, plus, T, X, L, ring) drawn on the same 112 patch.

## `LocalFourierPatches.jl`

The first few **2D Fourier coefficients of a small local patch** (P = 5 … 40 px),
at any location in the image.

```
F(v, u) = (1/Σw) · Σ_{y,x}  w[y,x] · p[y,x] · exp( −2πi ( v(y−1) + u(x−1) ) / P )
```

for `|v|, |u| ≤ K`, with an analysis window `w` (box / Hann / Gaussian σ=P/4). The
`1/Σw` makes `F(0,0)` the windowed mean intensity, so magnitudes are comparable
across patch sizes and windows. `K` is clamped to `(P−1)÷2` — beyond that a P-point
DFT aliases.

Four panels, all live on sliders (source, instance, patch size, centre, K, window,
DC removal, polarity):

1. **The patch and its coefficients** — field with the patch box, patch, windowed
   patch, band-limited reconstruction, and the `|F|` and `arg F` grids over `(v,u)`.
2. **Coefficient table** — wavelength, wave direction, the stroke orientation each
   coefficient prefers, magnitude, phase, and magnitude relative to DC.
3. **Patch-size sweep** — P = 5 … 40 at the same centre, drawn at true relative
   scale, raw vs. reconstruction.
4. **Sliding-patch maps** — one map per order, `|F(v,u)|` at every location.

### What it shows (measured)

- **A windowed low-order coefficient *is* a Gabor filter response.** `F(v,u)` with a
  Hann/Gaussian window is the inner product of the image with a windowed complex
  sinusoid at frequency `(u,v)/P`. The sliding-patch maps are Gabor channel moduli,
  and they behave that way: `|F(0,2)|` (wave along x) lights the *vertical* stem of a
  T; `|F(2,0)|` lights the *horizontal* crossbar. This notebook and
  `CreateGaborLifting.module.jl` compute the same thing from opposite ends — a Gabor bank
  fixes `(ω, θ)` and slides, here the patch is fixed and the whole `(v,u)` grid is
  read at once.
- **Energy fraction is the wrong thing to watch.** AC energy captured by orders `≤ 3`
  on an EMNIST `K` (Hann, mean over nine nearby centres) is **78 / 74 / 82 / 92 / 93
  / 94 / 96 %** at `P = 5 / 9 / 13 / 19 / 25 / 32 / 40` — it *rises* with `P`, because
  a bigger patch of a blurred letter is relatively smoother. What degrades is
  **absolute resolution** `λ = P/K`: at `P = 40, K = 3` the finest describable detail
  is 13 px = one whole stroke width, and the reconstruction merges neighbouring
  strokes. The useful band is `P ≈` **1–2 stroke widths (13–26 px)**.
- **Polarity invariance is *not* free here.** Under `I → 1 − I`, `F → W − F` where `W`
  is the transform of the window itself. For the **box** window `W` is an exact delta
  at DC, so every AC `|F|` is exactly polarity-invariant. For Hann/Gaussian it is
  not: at `P = 21`, `|W|` relative to its DC value is **0.54 / 0.34** at order ±1 but
  only **0.019 / 0.011** at ±2 and ≤ 0.007 at ±3. So first-order coefficients of a
  windowed patch are substantially polarity-*dependent*; orders ≥ 2 are invariant for
  practical purposes. The notebook prints the measured AC polarity check live.
- **Verified numerically** (`|F|` shift-invariance to 4e-8 for the box window; exact
  reconstruction and 100 % energy at `K = (P−1)/2`; `F(0,0)` = windowed mean).
- **Not rotation-invariant** — rotating the letter rotates the whole `(v,u)` grid.
  Recovering that is what the ring/harmonic construction in `P0.3_New_Gabor_FPE/` does.

## `TicTacToeFourierSignature.jl` + `TicTacToeSignature.md`

The motivating question: tile the character with an **N×N grid** (tic-tac-toe) and
ask whether the first two or three Fourier coefficients of each cell give a signature
of *"vertical / horizontal / diagonal stroke", "blob", "loop"* in that area.

Each cell is summarised by moments of its low-order power spectrum — ink `a₀`,
structure `ac`, orientation tensor `E₂` (anisotropy + stroke angle), `E₄`, and the
radial ring profile `e₁,e₂,e₃`. Sampling is selectable: the literal DFT lattice
`(v,u)`, or the same integral on a polar `ω×θ` grid.

**Findings (all measured; full detail in `TicTacToeSignature.md`):**

- **Orientation: works.** Bar swept through 12 angles is read to **0.16°** with polar
  sampling; the square DFT lattice is biased toward 0/45/90/135° by up to **3.5°**
  (its own 4-fold symmetry leaking into the 2nd harmonic). `|E₂|` ≈ 0.51–0.57 for one
  clean stroke, **0.000** for a plus, X, disc or ring.
- **Crossings: `|E₄|` fails.** Single bar 0.18–0.23, plus 0.085, X 0.142 — a single
  stroke rings the 4th harmonic *more* than a crossing does.
- **Loop vs blob: only if the cell is sized to the loop.** Disc vs ring of equal outer
  size gives `e₂/e₁` = 0.015 vs 1.33 (the `J₀` zero, as Bessel predicts), but the
  ratio confounds hollowness with object-size-relative-to-cell: at N=1 the ordering
  inverts, and on real EMNIST at 3×3 the loop score does not pick out `O`.
- **The 9-cell signature is strongly diagnostic of letter identity** — 360 EMNIST
  instances, 12 classes, LOO nearest-class-mean, chance 8.3 %:
  **ink+orientation (27 numbers) = 76.1 %**, orientation alone (18) = 75.3 %, ink
  alone = 56.7 %, no grid at all = 49.7 %, 4×4 = 77.8 %. The previous best descriptor
  in this project (global shape harmonics) reached ≈ 61 %.

So: the answer to the original question is *yes* for stroke orientation, *no* for
loops — and the unplanned result is that the crude 3×3 orientation signature beats
every local descriptor tried in this project so far.

**Follow-up:** `../P0.5_ExptsWithZernike/CombinedZernikeFourier.jl` concatenates this 3×3
grid with global Zernike moments and reaches **82.8 % (84.2 % η²-weighted)** — the two
are complementary because this descriptor is translation-invariant per cell while
Zernike is rotation-invariant about a centre, so they fail on different letters.

## Running

```bash
cd /home/kevin/claude-code/mother-embedding
julia --project=. -e 'using Pluto; Pluto.run(host="0.0.0.0", port=1235, launch_browser=false, notebook="P0.4_ExptsWithGlobalFourier/TicTacToeFourierSignature.jl")'
```
