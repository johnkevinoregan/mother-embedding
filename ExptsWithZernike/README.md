# ExptsWithZernike

Zernike-moment description of a character — the disc-supported counterpart of
`ExptsWithGlobalFourier/`. Same image format as the rest of the project: EMNIST 28×28
upsampled to a 112×112 letter (stroke ≈ 13 px), or a synthetic figure on the same
frame.

## `ZernikeCharacterMoments.jl`

**One global Zernike description of the whole character** (no spatial grid). The
in-disc mean is subtracted first, so `A₀₀ = 0` and every moment is a statement about
structure rather than ink quantity.

```
V_nm(ρ,θ) = R_n^{|m|}(ρ)·e^{imθ}        ρ ≤ 1,  n ≥ 0,  |m| ≤ n,  n−|m| even
A_nm      = (n+1)/π · ∫∫_{ρ≤1} f · R_n^{|m|}(ρ)·e^{−imθ} dA
```

`m` is angular frequency (variation *around* the disc), `n` radial order (variation
*outward*). `n_max = 8` gives **25 moments** (`m ≥ 0`; `A_{n,−m} = conj A_nm`).

Panels, all live on sliders (source, instance, `n_max`, zero-DC, disc fit, rotation):
input and its zero-DC disc field, band-limited reconstruction and residual; the
`|A_nm|` pyramid over `(m, n)` plus a ranked bar chart; a table of the largest
moments; images of the basis functions `Re V_nm`; a reconstruction sweep over
`n_max`; the rotation test; and the diagnosticity table.

### Findings (all measured; full detail in the notebook's Notes)

- **Rotation invariance is real.** `|A_nm|` drifts ≤ **1.9 %** over 0–180° on the
  inscribed disc (exactly 0 at 90°, a pixel-grid symmetry) while the complex moments
  move **77–170 %**. This is precisely what `|F(v,u)|` on a square DFT lattice could
  not deliver.
- **Fitting the unit disc to the ink matters more than anything else** — centroid +
  98th-percentile ink radius, rather than the disc inscribed in the frame:
  **64.7 % → 73.1 %** (Re/Im) and **53.1 % → 65.8 %** (`|A|`). Most of an unfitted
  basis is spent describing empty corners.
- **…but the fit costs invariance ~5×** (drift rises to 1.0–10.2 %), because the
  radius is re-estimated from each rotated copy. Accuracy and invariance are in
  tension here; pick per use case.
- **Rotation invariance costs accuracy**: discarding the phase drops 73.1 % → 65.8 %.
  Orientation is where letter identity lives, so an invariant that throws it away
  throws away signal.
- **Zero-DC costs ≈ 1 point** (64.7 % vs 65.6 %) — the overall ink level barely
  separates letters.
- **Accuracy peaks at `n_max ≈ 8`**: 35.6 / 56.9 / 63.9 / **64.7** / 61.1 / 59.2 % at
  `n_max` = 2/4/6/8/10/12. The decline past 8 is nearest-class-mean dilution by noisy
  high-order moments.
- **Reconstruction stays poor and that is fine** — relative in-disc RMS error 0.91 →
  0.51 as `n_max` goes 2 → 12. 25 moments cannot *draw* a 13 px stroke; they can
  still tell 12 letters apart at 73 % against 8.3 % chance.

### Against the Fourier work

| descriptor | numbers | LOO accuracy |
|:--|--:|--:|
| global Zernike, `\|A\|` + Re/Im, fitted disc, `n_max`=8 | 75 | **76.4 %** |
| global Zernike, Re/Im, fitted disc, `n_max`=8 | 50 | 73.1 % |
| global Zernike, `\|A\|` only (rotation-invariant) | 25 | 65.8 % |
| *Fourier: same 9 features over the whole image, no grid* | *9* | *49.7 %* |
| *Fourier: 3×3 tic-tac-toe grid* | *27–81* | *74.2–76.1 %* |

Zernike is a far better **global** descriptor than global Fourier moments — the
orthogonal disc basis genuinely buys something — but it lands level with, not ahead
of, simply cutting the image into nine boxes and measuring stroke orientation in each.
Spatial partitioning is worth about as much as a better global basis, and the two are
complementary rather than competing.

## `TicTacToeZernike.jl`

The grid counterpart: an N×N grid (3×3 default) over the character, each cell
described by Zernike moments on **its own disc** — the direct Zernike answer to
`ExptsWithGlobalFourier/TicTacToeFourierSignature.jl`. Per cell: ink `a₀`, structure
`ac`, orientation `O₂` (pooled `m=2`), crossing `f₄` (`m=4` power share), symmetric
content `m₀`, and `loop = −Re A₄₀/(|A₂₀|+|A₄₀|)`.

**Better features, worse classification — and not for the reason you'd guess.**

- **Orientation to 0.24°** from `−arg(A₂₂)/2` (0.64° pooled), vs 3.48° for the square
  DFT lattice. Zernike's angular basis *is* `e^{imθ}`, so no lattice bias.
- **`f₄` detects crossings, where the Fourier `|E₄|` failed**: plus 0.699, X 0.689 vs a
  single bar 0.187 — a 3.6× separation. (Fourier had a single bar at 0.179 *beating* a
  plus at 0.085.)
- **Loops are read off the *sign* of `A₄₀`**: rings +0.75/+0.81, disc −0.29, bar −0.45 —
  and robust to ring thickness, where the Fourier `e₂/e₁` flipped its verdict. On real
  EMNIST, `O`-vs-rest AUC **0.839** at 1×1 in the *right* direction (Fourier managed
  0.80 but sign-inverted), decaying to 0.627 at 2×2 and 0.545 at 3×3 — the loop must
  fit inside a cell.
- **But it classifies at only ~66 %** (3×3: 64.4–65.6 %, 4×4: 66.9 %) against 76.1 % for
  the Fourier grid and 76.4 % for global Zernike.
- **The cause is a position/orientation confound, not the hard disc edge.** Giving the
  *Fourier* descriptor a hard disc window costs nothing (74.7 → 73.9 %), so truncation
  is not it. Zernike moments are taken **about the disc centre**, so position leaks
  into shape: an isotropic blob — no orientation at all — slid off-centre reports its
  **position angle** (30°→31.9°, 60°→58.1°, 90°→90.0°, 120°→121.9°). The Fourier cell
  descriptor is immune because it is built from `|F|²`, which discards phase. Across
  749 inked EMNIST cells the two orientation estimates differ by a median **43.3°**
  (45° = chance), despite agreeing to <1° on a clean centred bar.
- This retroactively explains why fitting the disc was worth 8 points in
  `ZernikeCharacterMoments.jl`: fitting **centres the object**, which is exactly the
  condition Zernike needs. A grid cell cannot.

The transferable lesson: **tiling space wants a translation-invariant cell descriptor**
(Fourier power spectrum), **describing one centred object wants a rotation-invariant
one** (Zernike). Using either for the other's job costs about 10 points.

## Running

```bash
cd /home/kevin/claude-code/mother-embedding
julia --project=. -e 'using Pluto; Pluto.run(host="0.0.0.0", port=1235, launch_browser=false, notebook="ExptsWithZernike/ZernikeCharacterMoments.jl")'
```
