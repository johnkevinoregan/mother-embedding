# P0.5_ExptsWithZernike

Zernike-moment description of a character — the disc-supported counterpart of
`P0.4_ExptsWithGlobalFourier/`. Same image format as the rest of the project: EMNIST 28×28
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
`P0.4_ExptsWithGlobalFourier/TicTacToeFourierSignature.jl`. Per cell: ink `a₀`, structure
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

## `CombinedZernikeFourier.jl`

Tests the prediction from `TicTacToeZernike.jl` §6 — that global Zernike and the
Fourier 3×3 grid are complementary — by concatenating them. **They are, and the gain
is large.**

| descriptor | numbers | LOO | η²-weighted |
|:--|--:|--:|--:|
| Z — global Zernike (`n≤8`, fitted disc, `\|A\|`+Re/Im) | 75 | 76.4 % | 77.5 % |
| F — Fourier 3×3 (all 9 features per cell) | 81 | 74.2 % | 77.2 % |
| **Z + F concatenated** | **156** | **82.8 %** | **84.2 %** |

**+6.4 points** over the better block alone (84.4 % with polar sampling for F, so the
sampling choice barely matters). The previous best anywhere in this project was ≈ 61 %.

- **Not a dimensionality artifact.** Shuffling block F's rows — same columns, same
  marginals, correspondence destroyed — gives 69.2 / 71.4 / 71.7 %, i.e. *below* block
  Z alone. Uninformative columns actively hurt the equal-weighted nearest-mean
  classifier, so the real +6.4 is complementarity.
- **The two fail on different images**: both correct 65.3 %, only Zernike 11.1 %, only
  Fourier 8.9 %, neither 14.7 %. A magic per-image *chooser* would reach 85.3 % — the
  **best-of-two selection bound** — and the concatenation gets 82.8 %. That bound does
  not constrain concatenation, though: selecting between two classifiers can never fix
  an image both get wrong, whereas a joint feature space can. Measured: of the 53
  images neither block gets right, the concatenation recovers 6 (10 with η² weighting),
  while losing 15 that one block had. What actually limits the pair is the 14.7 % that
  *both* miss.
- **Per class:** `E` 66.7 (Z) / 76.7 (F) → **93.3 %**; `K` 76.7 / 63.3 → **86.7 %**.
  `O` is the exception where Fourier alone (96.7 %) beats the pair (90.0 %). `L` stays
  weak (36.7 / 50.0 → 56.7 %).
- **Weighting is a minor effect**: η² weighting adds ~1.4 points; per-block rebalancing
  matters only when the blocks are lopsided (with the 27-number F subset, 80.6 → 82.5 %).

Why it was predictable: Zernike's free invariance is rotation about a centre and it is
*not* translation-invariant; the Fourier cell descriptor is built from `|F|²` and so is
translation-invariant per cell and not rotation-invariant. Each is blind exactly where
the other sees.

## `AllClassesDiagnosticity.jl`

Every other number in this project is measured on the same **12 uppercase letters**
inherited from `P0.3_New_Gabor_FPE/KeyPointDiagnosticity.md`. This notebook re-runs the same
two descriptors on the **full EMNIST-balanced 47-class set** (10 digits, 26 uppercase,
11 lowercase), chance **2.13 %**. Class set, instances/class, `n_max`, grid size and
sampling are all on sliders.

| descriptor | 47 classes | *12 classes* |
|:--|--:|--:|
| Z — global Zernike | 55.5 % (η² 57.4 %) | *76.4 % (77.5 %)* |
| F — Fourier 3×3 | 58.2 % (η² 60.4 %) | *74.2 % (77.2 %)* |
| **Z + F** | **62.5 % (η² 65.3 %)** | ***82.8 % (84.2 %)*** |
| best-of-two selection bound | 68.4 % | *85.3 %* |

- **The ranking of the two blocks flips.** Zernike leads on 12 uppercase letters; the
  Fourier grid leads on all 47 (58.2 vs 55.5, and 60.4 vs 57.4 weighted). Digits and
  lowercase add classes that differ by *where* strokes are rather than by global shape,
  which is what a spatial grid is for. **Spatial layout scales with class count better
  than global shape does.**
- **Complementarity survives but shrinks**: +4.3 points from concatenation vs +6.4 at
  12 classes. "Neither correct" more than doubles (14.7 % → 31.6 %), so more of the
  problem is simply beyond both descriptors. In information terms the harder task is
  handled better though — 62.5 % is **29× chance**, against 10× on the 12-class set.
- **η²-weighting matters more here** (+2.8 points vs +1.4): with 47 class means
  estimated from 29 examples each, the equal-weighted nearest-mean classifier dilutes
  more easily.
- **A quarter of the remaining error is the label set.** Top confusions are `F`→`f`,
  `2`→`Z`, `0`→`O`, `q`→`9`, `O`→`0`, `L`→`1`, `1`→`I` — mostly pairs that are the same
  handwritten shape.

### Merging the homoglyph classes

The *label set* control re-runs everything with classes merged **before**
classification, 30 instances per original class throughout:

| label set | classes | chance | Z | F | **Z + F** |
|:--|--:|--:|--:|--:|--:|
| strict | 47 | 2.13 % | 55.5 / 57.4 % | 58.2 / 60.4 % | **62.5 / 65.3 %** |
| **homoglyphs merged** | **40** | **2.50 %** | 59.4 / 62.0 % | 62.7 / 65.2 % | **67.8 / 70.6 %** |
| homoglyphs + case merged | 31 | 3.23 % | 57.4 / 59.4 % | 61.7 / 63.6 % | 66.9 / 67.7 % |

*(LOO / η²-weighted; homoglyph groups are `0/O`, `1/I/L`, `2/Z`, `5/S`, `9/g/q`)*

- Merging the five homoglyph groups is worth **+5.3 points** (62.5 → 67.8, 65.3 → 70.6
  weighted); "neither correct" falls 31.6 → 27.5 % and the best-of-two selection bound rises to
  72.5 %.
- **Build the groups disjointly.** Listing confusable pairs and taking the transitive
  closure is wrong: `6≡G`, `G≡g`, `9≡g` chains `6` to `9` and collapses `6/9/G/Q/g/q`
  into one class. The notebook asserts disjointness.
- **Merging the case pairs makes it worse** — 40 → 31 classes drops Z+F from 67.8 % to
  66.9 % and η²-weighted 70.6 → 67.7 %, *while chance rises*. EMNIST-balanced already
  merged the case pairs that look alike; the 11 lowercase classes it keeps are exactly
  those whose shape differs, so merging them creates **bimodal classes** a
  nearest-class-mean cannot represent. Within-class scatter, merged ÷ mean of parts:
  `D/d` 1.28, `T/t` 1.22, `H/h` 1.21 … `F/f` 1.04 — every pair worse merged, and `D/d`
  duly becomes the worst class (35 %). So `F`→`f` is a real descriptor failure, not a
  label artifact.
- **Merging vs rescoring**: forgiving homoglyph errors from the strict 47-way model
  gives 71.6 %, training on 40 merged labels gives 70.6 % — rescoring wins by 1.0 point
  because the 47-way model keeps one tight mean per glyph while a merged class pools
  60–90 instances into one mean. Quote either, but not the strict 62.5 % alone.

## Running

```bash
cd /home/kevin/claude-code/mother-embedding
julia --project=. -e 'using Pluto; Pluto.run(host="0.0.0.0", port=1235, launch_browser=false, notebook="P0.5_ExptsWithZernike/ZernikeCharacterMoments.jl")'
```
