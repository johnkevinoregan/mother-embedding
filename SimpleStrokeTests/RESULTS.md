# SimpleStrokeTests — results

Everything this directory has produced, in the order it happened. `RESULTSexpanded.md` is a
separate, longer commentary on Phase 9 and is left as it was.

---

## Summary

This directory holds a **synthetic dataset of single strokes on a grey field**, labelled with
**eight graded properties** rather than class labels, and the experiments run on it. The change
of question is the point: instead of *"can a classifier separate these"* — which on synthetic
data is nearly always yes, and which is why the preceding phase produced nothing — it asks
**"fit a readout and measure how much of each property it recovers"**.

**What it established (Phase 9).** A linear readout on 31 hand-designed features beats a
two-hidden-layer MLP on 12,544 raw pixels on every structural property. **The conjunction layer
finally pays** — +0.16 to +0.21 over a shuffle control, against +0.01 on EMNIST. Trained on light
strokes and tested on dark, the features are unchanged while both pixel arms fall below chance.
And the dataset exposed a real bug: the front end **was not polarity invariant**, which EMNIST
could never have revealed because its background is zero.

**What it then established (Phase 12).** Four dials that had never been turned — number of
scales, number of orientations, order of the orientation harmonics, number of ray probe
distances. **Adopted: higher harmonics plus crossed ray offsets, 54 features.** Its value is
**robustness** — gains over the 31-feature baseline grow with distribution shift (~+0.02 i.i.d.,
+0.04–0.06 under blur, +0.08 under a thickness shift). Judged on the i.i.d. split alone it would
have been rejected. More scales were rejected outright.

**What remains broken.** The front end **confuses thickness with blurriness**: train on sharp
edges, test on blurred, and thickness readings collapse to R² ≈ −2. Nothing has fixed it. A
cross-scale feature appeared to, was published, and turned out to be an implementation artefact.

**Three method lessons, each paid for.** Reduced-n selection inflates gains 2–5× and reversed one
sign. The i.i.d. split alone picks the wrong configuration. And: check that the code being run is
the code that was named.

---

## Contents

**Part 1 — Phase 9: the dataset and what it showed**

| | |
|:--|:--|
| [The target vector](#the-target-vector) | the eight graded properties |
| [The five arms](#the-five-arms) | what is being compared |
| [i.i.d. results](#iid--12000-training-images-3000-test) | the headline table |
| [Extrapolation splits](#extrapolation-splits) | train on one range, test on another |
| [`closedness` is confounded](#closedness-is-confounded--read-that-row-as-nothing) | why that row means nothing |
| [Block attribution](#block-attribution--and-the-control-that-decides-it) | which features carry what, and the shuffle control |
| [Sample efficiency](#sample-efficiency) | how the arms scale with data |
| [The pooling grid](#the-pooling-grid--and-a-configuration-that-beats-everything-here) | why grid 1 wins here |
| [Ray transform not thickness-invariant](#the-ray-transform-is-not-thickness-invariant) | a known limitation |
| [Ratios after pooling](#ratios-are-formed-after-pooling-not-per-pixel) | a conceptual bug and its fix |
| [The bug this found](#the-bug-this-found) | polarity invariance |
| [Corrections](#corrections-to-earlier-write-ups) · [Caveats](#caveats) · [Open](#open) | |

**Part 2 — Phase 12: turning the front end's own dials**

| | |
|:--|:--|
| [Plain-language account](#plain-language-account) | **start here for the gist** |
| [Full-n confirmation](#full-n-confirmation--the-authoritative-numbers) | the numbers to quote |
| [Cross-scale, retracted](#cross-scale--proposed-published-and-retracted) | what went wrong and how |
| [Appendix: reduced-n pass](#appendix--the-reduced-n-selection-pass-superseded) | superseded, kept for the lesson |

---

# Part 1 — Phase 9: the dataset and what it showed

`SimpleStrokeTests` asks a different question from every phase before it. Phases 5–8 asked
whether a classifier could separate some categories; the answer was almost always yes, which
is why Phase 8 produced no information at all. Here the question is **how much of each
geometric property a *linear* readout can recover from a representation** — because a
property that needs a hidden layer to extract is present in the representation but has not
been made available by it, and making it available is the entire job of a front end.

The dataset is a single stroke on a uniform grey field, described by a vector of eight
graded properties rather than a class label. See `Contours.module.jl` for its construction
and the couplings that could not be removed; `RESULTSexpanded.md` explains everything here
from scratch, including how the splits guarantee the comparisons are fair.

---

## The target vector

*Eight graded numbers per image instead of a class label — five geometric, three describing the stroke's appearance. The three appearance rows double as controls on the front end itself: `polarity` in particular **should** be unreadable from our features, and reading it would falsify the invariance claim.*

Eight properties per image, nothing masked. Rows 1–5 are geometry; rows 6–8 are controls on
the front end itself.

| # | property | range | |
|--:|:--|:--|:--|
| 1 | `curvedness` | 0–1 | mean \|κ\| of the base curve, squashed; 0 = straight |
| 2 | `brokenness` | 0–1 | gap width **in units of stroke width**; 0 = continuous |
| 3 | `closedness` | 0 / 1 | is it a closed loop |
| 4 | `vangle` | 32–180° | angle at the vertex; 180 = passes straight through |
| 5 | `arms` | 2 / 3 / 4 | arms meeting at a point; 2 = no junction |
| 6 | `thickness` | 3–12 px | stroke width |
| 7 | `fuzziness` | 0.8–20 px | edge ramp width |
| 8 | `polarity` | ±1 | light or dark stroke |

---

## The five arms

*What is being compared, and why the comparison is stacked against us: the CNN builds its filters *for these eight targets* while ours were fixed before the dataset existed, and the pixel probe gets 45× more free parameters than our feature probe.*

| | representation | readout | parameters |
|--:|:--|:--|--:|
| 1 | raw pixels, **fixed** | linear (ridge) | 12,544 |
| 2 | raw pixels, **fixed** | MLP, 2×256 hidden | 3.3 M |
| 3 | CNN, **learned end-to-end** | trained on these targets | 0.9 M |
| 4 | **our features, fixed** | **linear (ridge)** | **279** |
| 5 | our features, fixed | MLP, 2×256 hidden | 0.2 M |

Row 4 minus row 1 is what the front end made explicit that was not already. Row 4 against
row 3 is the harder comparison, and **every asymmetry in it runs against us**: the CNN
builds its filters *for these eight targets* while ours were fixed before the dataset
existed, and the pixel probe has 45× more free parameters than our feature probe.

---

## i.i.d. — 12,000 training images, 3,000 test

*The headline table. A linear readout on 279 designed features beats a 3.3 M-parameter MLP on raw pixels on every structural row, and beats the trained CNN on four of five. The best configuration is 31 globally pooled features. **Caveat added later:** the CNN arm did not converge, so it is a reference point and not a ceiling.*

![arms](phase9_arms.png)

| arm | curvedness | brokenness | closedness | vangle | arms | thickness | fuzziness | polarity |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| *trivial baseline* | 0.009 | 0.008 | 0.457 | 0.066 | 0.029 | 0.195 | 0.152 | 0.063 |
| pixels·linear | −0.000 | −0.001 | −0.000 | −0.000 | 0.000 | −0.002 | −0.000 | **0.709** |
| pixels·MLP | −0.183 | −0.404 | 0.721 | −0.137 | 0.108 | 0.007 | −0.222 | **0.787** |
| CNN | 0.411 | 0.298 | 0.962 | 0.290 | **0.859** | **0.893** | **0.936** | **0.990** |
| **ours·linear** | **0.694** | **0.318** | **0.986** | **0.580** | 0.848 | 0.582 | 0.618 | −0.000 |
| ours·MLP | **0.835** | **0.592** | **0.993** | **0.835** | **0.905** | 0.635 | 0.657 | −0.169 |
| **ours·MLP, grid 1 (31 features)** | **0.925** | **0.737** | **0.998** | **0.938** | **0.954** | 0.687 | 0.714 | −0.099 |

*Every number in this document comes from one canonical run under the current front end —
`results_canon3` for grid 3, `results_canon1` for the grid-1 row, `results_canon_g*` for the
pooling sweep — so the tables and figures are internally consistent rather than assembled from
runs at different code states. CNN = full-resolution, four conv stages with pooling and batch
norm, 60 epochs on a GPU.
The earlier CPU-era numbers, from two strided convolutions trained for 12 epochs, are in
"What the weaker CNN cost" below — the difference is large and the caveat was justified.*

**The best configuration is the last row — 31 globally pooled features with a small MLP**,
which beats the CNN on every structural property, by 0.938 to 0.291 on `vangle` and 0.737 to
0.254 on `brokenness`. See the pooling-grid section for why coarser pooling wins.

**The CNN row has ~0.03 of run-to-run variance.** Two runs at identical settings gave `arms`
0.874 and 0.843, presumably non-deterministic cuDNN kernels. Differences in the CNN column
below about 0.05 should not be read as real, and the earlier description of `arms` as "a dead
heat at 0.842 against 0.843" was over-reading the third decimal.

**The result splits along the structural/photometric line.** On geometry our fixed features
with a *linear* readout still beat a CNN trained end-to-end on these targets — on four of
the five rows. The exception is `arms`, where the CNN edges ahead 0.874 to 0.859; with an
MLP readout on the same features we lead there too, 0.916. On the three photometric rows the
CNN wins clearly — thickness, blur and contrast sign are local statistics a convolutional net
learns directly, and two of them our representation deliberately discards.

**`pixels·linear` at exactly zero is the sanity check, not a failure.** A fixed weighted sum
of pixels is a template at a fixed location, and position and rotation are randomised, so no
template can work. That it nonetheless scores **0.709 on polarity** — the one thing linear
in pixels can see, via mean level — shows the pipeline is measuring what it should.

**`ours` at −0.000 on polarity is the correct result**, not a failure. Quadrature energy
discards the sign of contrast by construction. See the block table: *every* block is at
−0.000, so the representation throws contrast sign away entirely.

---

## Extrapolation splits

*Train on one range of a nuisance and test on another — light strokes → dark, sharp edges → blurred, thin → thick. This is where designed invariance earns its keep: under a polarity flip our numbers are unchanged while both pixel arms fall below chance. It is also where the thickness/blur weakness first appears.*

The i.i.d. table mostly measures capacity. Invariance only shows when the test set contains
nuisance values training never held. In each split the held-out nuisance's own row is
dropped — a model cannot be scored on predicting something that was constant while it
trained.

### Polarity — trained on light strokes, tested on dark

| arm | curvedness | brokenness | closedness | vangle | arms | thickness | fuzziness |
|:--|--:|--:|--:|--:|--:|--:|--:|
| *trivial* | 0.008 | 0.009 | 0.437 | 0.070 | 0.031 | 0.204 | 0.184 |
| pixels·linear | −0.130 | −0.112 | −2.352 | −0.532 | −1.773 | −2.931 | −0.889 |
| pixels·MLP | −2.747 | −1.491 | −5.025 | −3.668 | **−6.981** | −8.319 | −4.060 |
| CNN | 0.116 | −0.262 | 0.841 | **−2.107** | −0.057 | −1.485 | −0.108 |
| **ours·linear** | **0.689** | **0.322** | **0.985** | **0.598** | **0.842** | 0.642 | 0.627 |
| ours·MLP | **0.859** | **0.616** | **0.998** | **0.873** | **0.921** | 0.669 | 0.678 |

**This is where the properly trained CNN separates from us most sharply, and not in its
favour.** It is the strongest arm on `arms` i.i.d. (0.874) and lands at **−0.415** on the
same property when contrast is inverted. On `vangle` it goes 0.291 → **−2.225**. Whatever it
learned about geometry was entangled with which way the contrast ran.

`ours·linear` here is **identical to its i.i.d. row to three decimals** — perfect transfer to
a contrast polarity it never saw. Both pixel arms do not merely fail; they land far below
predicting the mean, the MLP worst at −6.04, because they have learned that dark-here means
one thing and the test set inverts it.

### Fuzziness — trained sharp (ramp ≤ 3 px), tested blurred (ramp ≥ 8 px)

| arm | curvedness | brokenness | closedness | vangle | arms | thickness |
|:--|--:|--:|--:|--:|--:|--:|
| *trivial* | 0.000 | 0.005 | 0.378 | 0.071 | 0.024 | 0.019 |
| pixels·linear | −0.001 | −0.001 | −0.004 | −0.001 | −0.004 | −0.003 |
| pixels·MLP | −0.074 | −0.196 | 0.772 | −0.083 | −0.003 | −0.124 |
| CNN | 0.277 | −0.071 | 0.917 | 0.103 | 0.675 | −0.620 |
| **ours·linear** | **0.681** | −0.029 | **0.979** | **0.580** | **0.823** | −2.370 |
| ours·MLP | **0.807** | **0.387** | **0.986** | **0.816** | **0.884** | −1.799 |

Blur costs the CNN its corner readout entirely — `vangle` 0.291 i.i.d. to **−0.012** — while
ours drops only 0.567 → 0.525. It keeps `arms` (0.715), which is the row it was strongest on.

### Thickness — trained thin (≤ 6 px), tested thick (≥ 8 px)

| arm | curvedness | brokenness | closedness | vangle | arms | fuzziness |
|:--|--:|--:|--:|--:|--:|--:|
| *trivial* | −0.001 | −0.006 | −0.435 | −0.086 | −0.035 | −0.911 |
| pixels·linear | −0.002 | 0.003 | −0.001 | −0.000 | −0.004 | −0.004 |
| pixels·MLP | −0.232 | −0.171 | 0.527 | −0.230 | −0.016 | −0.543 |
| CNN | 0.148 | 0.035 | 0.569 | 0.169 | 0.571 | 0.239 |
| **ours·linear** | **0.634** | **0.262** | **0.978** | **0.504** | **0.783** | −2.157 |
| ours·MLP | **0.751** | **0.408** | **0.974** | **0.703** | **0.869** | −2.650 |

`arms` is the clearest case in the whole experiment: the CNN leads it i.i.d. at 0.874, and
under the three nuisance shifts it goes to **0.715**, **0.389** and **−0.415** while ours
holds 0.845, 0.776 and 0.857. **The i.i.d. lead was not a representation of junction order;
it was a representation of junction order *as drawn in the training set*.**

**The structural rows transfer; the coupled photometric row breaks, and should.** Blur makes
a stroke look thicker, so a thickness estimate calibrated on sharp edges is biased when
tested on blurred ones (−2.55), and vice versa (−2.05). That is a real property of the
measurement, not a failure of transfer — and it is *informative*: it says our thickness
estimate is a joint function of width and blur rather than of width alone.

`brokenness` also drops under blur (0.340 → 0.107), which is honest: a 20 px ramp fills a
small gap.

---

## `closedness` is confounded — read that row as nothing

*A negative result about our own dataset. Open and closed contours have non-overlapping arclengths, so a length threshold classifies every image correctly and the 0.99 on that row measures nothing. Geometric, not a coding error, and not fixable without redesigning the generator.*

Its trivial baseline is 0.457, by far the highest of the eight, which was the first sign.
On investigation the row is **over-determined by three local cues, none of which is closure**:

| cue | open | closed | R² for `closedness` alone |
|:--|--:|--:|--:|
| arclength | 77 px | 245 px | **0.898** |
| orientation anisotropy \|C₂\| | 0.555 | 0.148 | 0.327 |
| \|C₄\| | 0.331 | 0.034 | 0.353 |
| \|C₂\| + \|C₄\| | | | **0.490** |
| endpoint gap / stroke width | 8.5 | 0.0 | 0.295 |

**Arclength ranges are disjoint** — open 68–90 px, closed 225–283 px — so a threshold at
150 px labels every image in the dataset correctly.

None of these requires following the contour, which is the point. **Length is a sum**: total
energy is `length × width × contrast`, and since the bank measures three scales, width is
recoverable from the ratio across them and the sum can be normalised — a linear readout
reaches length without tracing anything. **Orientation isotropy is a per-cell histogram
statistic**: a closed loop turns through 2π so its tangents cover every orientation, while an
open arc capped at 2π/3 covers 120° and is strongly anisotropic. And **curvature** adds a
third, since every closed loop here has R ≤ 45 while open arcs run to R = 600.

### Why it is geometry, not a coding error

In a frame of diameter D a closed contour can be π·D long, while an open arc whose turn is
capped at 2π/3 is limited to ~1.2·D. **Closure buys length**, unavoidably. Exempting closed
loops from the arclength cap (added to stop kinked figures being clipped) widened the ratio
from ~2.6× to 3.5×, but the confound is intrinsic to putting a long contour in a small box.

### The genuine signal is present but swamped

An open arc has exactly two line terminations; a closed loop has none. That *is* local, and
detecting terminations is what `A2` end-stopping is for — **`A2` alone scores 0.783 on
`closedness`**. The right mechanism works; it is simply not needed when three shortcuts are
available.

### Not fixed, deliberately

Removing all three cues at once requires long *open* contours — spirals and serpentines,
which are long, orientation-isotropic and tightly curved while still having two free ends.
That would make `closedness` a real test of end-stopping, at the cost of stimuli that look
like scribbles rather than simple strokes. Judged not worth the change to the dataset's
character. **The row stays in the tables and should be read as "long and isotropic versus
short and anisotropic", not as closure detection.**

---

## Block attribution — and the control that decides it

*Which parts of the representation carry which property, with a shuffle control that holds column count and marginals fixed and destroys only the correspondence with the image. This is the section that establishes the conjunction layer pays **+0.16 to +0.21** here against +0.01 on EMNIST.*

![blocks](phase9_blocks.png)

| block | curvedness | brokenness | closedness | vangle | arms | thickness | fuzziness |
|:--|--:|--:|--:|--:|--:|--:|--:|
| `orient` (135 cols) | 0.635 | 0.159 | 0.949 | 0.394 | 0.667 | 0.508 | 0.455 |
| `lowpass` (9) | 0.044 | 0.051 | 0.669 | 0.047 | 0.238 | 0.326 | 0.050 |
| `A1+A2` (54) | 0.174 | 0.065 | 0.783 | 0.265 | 0.515 | 0.508 | 0.446 |
| `rays` (81) | 0.310 | 0.116 | 0.917 | 0.303 | 0.553 | 0.412 | 0.359 |
| **control: `orient`+`lowpass` intact, `A`/`rays` shuffled** (279) | 0.632 | 0.174 | 0.953 | 0.403 | 0.727 | 0.514 | 0.463 |
| **`all`** (279) | **0.682** | **0.340** | **0.985** | **0.567** | **0.859** | **0.576** | **0.625** |
| **real gain** | +0.050 | **+0.166** | +0.032 | **+0.164** | **+0.132** | +0.062 | **+0.162** |

**`all` has 279 columns against `orient`'s 135, so `all` − `orient` is not the claim.** The
control permutes the `A1`/`A2`/ray columns *across samples* — identical column count,
identical marginals, correspondence with the image destroyed — while **leaving `orient` and
`lowpass` untouched**. It is therefore "the conventional statistics plus 135 columns of
noise", and it *should* score close to `orient`: a control that collapsed to zero would be
testing nothing. Whatever it scores **above** `orient` is what capacity alone buys, and the
gap from it **up to** `all` is the conjunction layer's real contribution.

**It buys between 0.003 and 0.06.** The shuffled row sits on top of `orient` — so the gain
from the conjunction and ray blocks is information, not parameters, and the honest number is
`all` − `all·SHUFFLED`.

**This is the first positive result for the conjunction layer in the project.** Four
converging lines on EMNIST put it at +0.01, and the Phase 8 benchmark could not construct a
task where it mattered. Here it is worth **+0.166 on `brokenness`, +0.164 on `vangle`,
+0.132 on `arms`**.

**And it lands where the theory predicted.** `vangle` and `arms` are i2D by construction — a
corner angle and a ray count are not functions of an orientation histogram — and they carry
the largest gains. `closedness`, which is global and needs long-range integration rather
than local conjunction, gains almost nothing (+0.032). The operators help exactly where they
were argued to help.

---

## Sample efficiency

*How each arm scales from 500 to 12,000 training images. The designed features are far ahead when data is scarce, which is the inductive-bias claim stated as a curve rather than a single number.*

![samples](phase9_samples.png)

`ours·linear`, test R² by training-set size. Subsets are nested, so the curve moves smoothly
rather than jittering from resampling.

| k | curvedness | brokenness | closedness | vangle | arms | thickness | fuzziness |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 500 | 0.617 | 0.122 | 0.971 | 0.360 | 0.782 | 0.443 | 0.470 |
| 2,000 | 0.648 | 0.283 | 0.982 | 0.486 | 0.833 | 0.516 | 0.565 |
| 6,000 | 0.674 | 0.325 | 0.985 | 0.548 | 0.855 | 0.568 | 0.615 |
| 12,000 | 0.682 | 0.340 | 0.985 | 0.567 | 0.859 | 0.576 | 0.625 |
| **% of ceiling at k=500** | **90 %** | 36 % | **99 %** | 64 % | **91 %** | 77 % | 75 % |

And the CNN over the same nested subsets — the comparison that makes the point:

| k | curvedness | brokenness | closedness | vangle | arms |
|--:|--:|--:|--:|--:|--:|
| 500 | −0.092 | −0.180 | 0.712 | −0.037 | 0.056 |
| 2,000 | 0.022 | −0.127 | 0.831 | 0.036 | 0.400 |
| 6,000 | 0.154 | 0.032 | 0.931 | 0.170 | 0.633 |
| 12,000 | 0.455 | 0.289 | 0.953 | 0.277 | 0.863 |
| **% of *its own* ceiling at k=500** | — | — | 75 % | — | **6 %** |

**At 500 images the properly trained CNN is at or below zero on four of the five structural
rows.** Ours is at 0.617 / 0.122 / 0.360 / 0.782 on the same four. The better architecture
raised the CNN's ceiling substantially without making it any less dependent on data to reach
it — at k = 500 it is *worse* than the weak strided net was, presumably because a larger
model overfits harder on 500 images.

**A fixed representation starts where it will finish; a learned one has to buy its
representation with data.** On `arms`, ours moves 0.782 → 0.859 across a 24× increase in
data while the CNN moves 0.089 → 0.650. At 500 images the gap is **8.8×**; by 12,000 it is
1.3× and still closing.

**That is the honest shape of the result.** Two of the CNN's curves are climbing steeply at
the right-hand edge — `arms` gained +0.146 over the last doubling — so with substantially
more data, more epochs, or both, it would likely close the gap on `curvedness` and `arms`.
What it does *not* do is learn `brokenness` at all: flat within noise of zero at every size,
where ours reaches 0.340. A 3 px gap in a 112 px image appears to be below what this
architecture extracts at any sample size tested.

`pixels·linear` is at 0.000 ± 0.02 for every property at every k, except polarity where it
sits at 0.689–0.709 throughout. **More data does not help a representation that does not
contain the property**, which is the cleanest statement of what a front end is for.

Two rows are still climbing at k = 12,000 — `brokenness` (36 % of ceiling at k=500) and
`vangle` (64 %). Those are the two finest-grained properties, and the two most likely to be
limited by the 3×3 pooling grid rather than by data.

---

## The pooling grid — and a configuration that beats everything here

*Sweeping the spatial grid from 1 to 5. Grid 1 — global pooling, 31 numbers — wins on every property, because stroke position is randomised so a fixed grid is pure liability. Fashion-MNIST later reverses this, which is what shows the result was about *position randomisation* and not about pooling.*

![grid sweep](phase9_grid.png)

`grid = 5` was run to ask whether `brokenness` and `vangle` were limited by the operators or
by pooling a 3 px gap inside a 37 px cell. The answer is the operators — but sweeping the
grid the *other* way turned up something better.

| grid | columns | linear `vangle` | linear `arms` | **MLP `vangle`** | **MLP `arms`** |
|--:|--:|--:|--:|--:|--:|
| **1×1** | **31** | 0.543 | 0.729 | **0.938** | **0.954** |
| 2×2 | 124 | 0.562 | 0.775 | 0.883 | 0.923 |
| 3×3 | 279 | **0.580** | **0.848** | 0.835 | 0.905 |
| 4×4 | 496 | 0.556 | 0.840 | 0.800 | 0.890 |
| 5×5 | 775 | 0.544 | 0.836 | 0.772 | 0.895 |

**The two readouts want opposite things, monotonically and without exception.** The linear
readout peaks at 3×3; the MLP improves all the way down to a single cell, and **31 globally
pooled numbers beat 775 on every row**.

**Why.** A linear map cannot form position-dependent combinations for itself, so the grid is
how it gets them — the cells *are* its nonlinearity. An MLP builds those internally, and what
it would rather have is what the grid destroys: **translation invariance**. Position is
randomised in every image, so a fixed grid makes the same shape at a different place produce
different numbers, and any readout on it must learn to undo that from data. Global pooling
has it for free.

**The best configuration in this experiment is therefore 31 features, globally pooled, with a
small MLP** — and it beats the CNN by a wide margin:

| | `curvedness` | `brokenness` | `closedness` | `vangle` | `arms` |
|:--|--:|--:|--:|--:|--:|
| CNN (full-res, 60 epochs, GPU) | 0.410 | 0.236 | 0.959 | 0.248 | 0.843 |
| **grid 1 + MLP (31 features)** | **0.903** | **0.663** | **0.996** | **0.898** | **0.930** |

> ### ⚠ Correction — "properly trained" is not supportable, and this table is not a ceiling
>
> This file described the CNN arm as *properly trained* throughout. Reading the saved
> per-epoch history in `results_canon3/history.jls` shows it **did not converge**, and not in
> the benign sense of a curve still climbing. Mean validation R² over the 60 epochs:
>
> ```
> e5  0.343   e20 0.444   e35  0.647   e50 0.695
> e10 0.543   e25 0.457   e40  0.558   e55 0.668
> e15 0.587   e30 −0.536  e45  0.334   e60 0.667
> ```
>
> It swings from **−0.536 at epoch 30** to 0.647 five epochs later, and `curvedness` reads
> −0.010 at epoch 40 against 0.414 at epoch 50. Best-epoch-on-validation then selects a spike
> out of a noisy trajectory, so the CNN's numbers here are **where an unstable run happened to
> be sampled, not what the architecture can represent.** Ours over the same run is smooth and
> saturated — argmax at epoch 20, then a gentle monotone decline.
>
> The likely cause is in `cnn()`: Adam at a constant `1f-3`, batch 64, BatchNorm, no
> learning-rate schedule, on a regression target. A cosine decay would probably stabilise it
> and might raise these numbers materially.
>
> **What survives:** the CNN arm is a fair *reference point* for a conventional network trained
> under this budget, and the rebuild from the strided CPU-era `:small` was a real improvement.
> **What is withdrawn:** any reading of this table as a statement about what a CNN can learn
> on this task. `ConVNextTest/RESULTS.md` makes the point independently — a *frozen* ConvNeXt
> that never saw a stroke reaches 0.976 on `vangle`, so 0.248 is plainly not a ceiling for
> convolutional architectures.

**Two things this costs us.** The "explicit under a *linear* readout" framing — the whole
premise of the phase — describes the 3×3 configuration, not the best one. And the
translation-invariance handicap that has been noted throughout as running against us turns
out to have been **self-inflicted**: the front end can be translation invariant simply by not
imposing a grid.

**And one caveat.** These stimuli contain a single figure on an empty field, so global
pooling loses nothing — there is only one thing to pool. On an image with several objects it
would confound them, and the grid would start earning its place again. This result is about
this dataset, not about pooling in general.

---

## The ray transform is not thickness-invariant

*A limitation found by direct measurement: a 4 px bar crossing a 15 px bar reads as two rays rather than four. Neither dataset in the project can reveal it, since both hold stroke width constant within an image.*

A limitation of the operator, found by asking what happens when the arms of a junction differ
in width. `|c₂|/c₀` should be ~0 for a four-ray crossing and ~1 for a straight line:

| stimulus | \|c₂\|/c₀ at ρ = 2.00 / 3.74 / 7.00 | reads as |
|:--|:--|:--|
| X, both bars 13 px | 0.05 / 0.00 / 0.16 | 4 rays ✓ |
| **X, 4 px × 15 px** | **0.70 / 0.52 / 0.71** | **2 rays ✗** |
| X, 6 px × 15 px | 0.57 / **0.29** / 0.73 | mixed |
| straight, 13 px | 0.78 / 0.88 / 0.90 | 2 rays ✓ |

**A crossing of unequal bars reads as a straight line.** The cause is not scale selection:
`|c₂|/c₀ = (a−b)/(a+b)` for lobes `a` and `b`, so 0.70 means the thin arm contributes 22 % of
what the thick one does. The harmonics compare lobes on **raw magnitude**, and a 15 px bar
simply produces more energy than a 4 px bar at any scale.

Note the middle scale reads the 6/15 case at **0.29** while the coarsest reads 0.73 — the
information is present, in one scale, which is an argument for keeping the scales separate.

*This cannot show on the stroke dataset, where every image has a single stroke width by
construction, and it did not show on EMNIST either. It needs a stimulus built for it.*

### Two designs tried against it, and what the dataset said

**Max over scale** — one profile per direction, taking the strongest scale, each probed at its
own offset. Attractive because `dₛ ∝ λₛ`, so the winning scale brings its own offset and the
probe radius tracks local stroke width **with no preliminary measurement of the image** — the
objection to anchoring `d` to a measured structure scale, since nothing in biology performs a
global pass before setting a receptive field.

On the dataset it **loses on every row**:

| grid 1 | features | curvedness | brokenness | closedness | vangle | arms |
|:--|--:|--:|--:|--:|--:|--:|
| **per-scale** | 31 | **0.925** | **0.737** | **0.998** | **0.938** | **0.954** |
| max over scale | 26 | 0.912 | 0.698 | 0.997 | 0.912 | 0.935 |

Collapsing the scales discards something real, and saves five features out of 31. Kept in the
code behind `scale_mode = :max`, off by default.

**Divisive normalisation** — `R′(φ) = R(φ)/(R(φ) + κ·maxφ R)`, which must *saturate*: dividing
every lobe by the same number leaves their ratio unchanged, so no linear rescaling can rescue
a thin arm swamped by a thick one. On the five-stimulus diagnostic it took the 4/15 crossing
from 0.638 to 0.298 at κ = 0.10.

**On the dataset it loses, and so does the max. The full 2×2, grid 1, `ours·MLP`:**

| | curvedness | brokenness | closedness | vangle | arms |
|:--|--:|--:|--:|--:|--:|
| **per-scale, no norm** (default) | **0.925** | **0.737** | **0.998** | **0.938** | **0.954** |
| per-scale, divisive κ=0.10 | 0.902 | 0.650 | 0.997 | 0.901 | 0.934 |
| max over scale, no norm | 0.912 | 0.687 | 0.996 | 0.915 | 0.936 |
| max over scale, divisive | 0.899 | 0.646 | 0.995 | 0.894 | 0.916 |

The two costs are separate and roughly additive: collapsing the scales costs ~0.05 on
`brokenness`, normalising costs ~0.09, both cost ~0.09. The existing default wins every cell
here, so both changes are kept in the code and left off **for this dataset**.

> **This is not a general verdict, and should not be read as one.** Divisive normalisation
> exists to handle spatially varying local contrast, and these stimuli have essentially none:
> one stroke, one width, one contrast, on a flat field. The mechanism is being scored on a
> problem the benchmark does not contain while paying its cost on every image.
>
> The project's target is a general front end for greyscale images, where local contrast
> varies enormously within a single frame, several scales coexist at the same location, and
> there is no empty background. Divisive normalisation is standard in V1 models for exactly
> those conditions. **The right reading is "not needed on single strokes", not "not needed".**
> It stays behind `normalize = :divisive` for that reason.
>
> The same caution applies to the pooling-grid result below — grid 1 wins here because there
> is one object per image — and, more weakly, to per-scale beating max, which was tested only
> on stimuli with a single stroke width.

**Why the diagnostic pointed the wrong way.** It measured one mixed-thickness crossing, and
this dataset contains none — every image has a single stroke width by construction. So
divisive normalisation was being scored on a problem that does not occur here, while paying
its cost (compressed dynamic range: the straight line drops from 0.78 to 0.57 on
`|c₂|/c₀`) on every image that does. That is the third time in this phase a five-stimulus
single-pixel probe has disagreed with the dataset, and the dataset has won each time.

κ also remains a fitted constant of exactly the kind `d_factor` turned out to be — chosen by
eye on five noiseless images. If the normalisation is ever revisited, that has to be settled
first.

---

## Ratios are formed after pooling, not per pixel

*A conceptual bug and its fix. Dividing per pixel and then averaging weights empty background equally with contour and makes the pooled value scale with ink coverage. The general rule this produced — a fill value is safe only when it is the quantity's true limit — is what later distinguished the safe epsilon guards from the unsafe one.*

`ray_maps` used to divide at every pixel — `|c₁|/c₀`, `|c₂|/c₀` — guarded by `c₀ > 1e-12`,
writing `0` where that failed. Both halves were wrong.

**The guard asserted something it had no evidence for.** `0` means *perfectly symmetric*, a
positive claim at a location with no signal. On a stroke drawing the branch fires at most
pixels; on a photograph `c₀ > 0` everywhere and it never fires. An operator that behaves
differently *in kind* between line art and natural images is not a general front end, which
is the whole aim.

**And pooling a ratio is wrong whatever the background.** A ratio where `c₀` is small is
numerically fine and statistically meaningless, and a spatial mean gave it the same weight as
a ratio measured on a strong contour. Because the guard wrote zeros, the pooled value came
out as *the true ratio × the fraction of the window containing contour* — a shape descriptor
multiplied by ink coverage, which depends on thickness, length and blur.

**Now:** `ray_maps` returns `c₀`, `|c₁|`, `|c₂|` unnormalised — three energies — and the
ratio is formed from the *pooled* numerator and denominator, with a **relative** floor (a
thousandth of the image's own mean `c₀`) instead of a branch. Defined everywhere, no fill
value, energy-weighted by construction.

`Validate_RayHarmonics` passes all four gates afterwards and the signature table is
unchanged — endpoint 0.699, straight 0.054, L 0.578, T 0.313, X 0.062 — so the operator still
measures what it did; only the normalisation moved.

### What it changed

| | before | after |
|:--|--:|--:|
| `rays` block alone, `arms` | 0.536 | **0.690** |
| `rays` block alone, `vangle` | 0.297 | **0.367** |
| grid 1 + MLP, `brokenness` | 0.663 | **0.737** |
| grid 1 + MLP, `vangle` | 0.898 | **0.938** |
| grid 3 linear, `vangle` | 0.552 | **0.580** |

**Two earlier conclusions are withdrawn.** `|c₁|/c₀` was reported as near-dead weight at
≤ 0.015 unique contribution; it is **+0.021** once pooled correctly, and standalone it goes
0.467 → 0.539 on `arms`. And the clean split "A₁+A₂ own `vangle`, rays own `arms`" only half
survives: rays still own `arms` (+0.113 against +0.022), but `vangle` is now a **tie**
(A +0.059, rays +0.066).

**A standing rule falls out of this.** `RESULTS.md` already recorded that A₂'s *absolute*-ε
conditioning collapsed it to plain energy, fixed by the relative `κ·E(x)`. The ray transform
kept the absolute form and nobody revisited it. That is the same error twice, so: **no
absolute epsilon anywhere in this front end.** Any conditioning constant must be relative to
a local energy.

---

## The bug this found

*Polarity invariance was broken, and this dataset is what exposed it: zero-padding a **non-zero** background puts a full-contrast step round the image whose cross term flips sign with contrast. EMNIST could never have shown it, because its background is zero.*

On its first real run the block table showed `orient` predicting **polarity at R² 0.65**,
when quadrature energy should predict nothing at all. Rendering the same shape at both
polarities showed features differing by **29 %** between a stroke and its exact
contrast-reverse. **The front end was not polarity invariant.**

The cause is padding. The bank zero-pads the image into a larger field. On EMNIST the
background *is* zero, so the padding was seamless and this could never appear. Here the
background is ≈ 0.5, so zero-padding puts a full-contrast step around the whole border, and
the response becomes `R_border + pol·R_stroke`, whose squared magnitude carries a cross term
`2·pol·Re(R_border · conj(R_stroke))` that **flips sign with polarity**.

Subtracting the median before filtering makes the padded zeros continuous with the
background. Invariance is now exact to float precision — **2.4 × 10⁻⁷** relative, from 29 %.

It also cost real accuracy on every other row, because that spurious border edge was noise
in every feature. On 1,000 training images: `closedness` 0.63 → 0.93, `vangle` 0.04 → 0.38,
`arms` 0.23 → 0.57.

**This matters beyond this experiment.** Polarity invariance is a stated requirement of the
project and has been claimed throughout. It was silently false for any image without a zero
background. Only EMNIST's black background hid it — which is precisely the argument for
testing on something that is not characters.

### A prediction it corrected

On record before the run: *"`orient` ≈ 0, `lowpass` ≈ 1"* — the oriented channels blind to
polarity, the lowpass block reading it perfectly. **Wrong on the second half.** `lowpass` is
the magnitude of a DC-centred channel, and magnitudes discard sign, so it is equally blind.
The representation throws contrast sign away *entirely*, which is stronger and cleaner than
claimed — but it does mean nothing downstream can ever recover contrast sign from these
features.

---

## Corrections to earlier write-ups

*Claims this project published and later withdrew, kept in place rather than deleted.*

`RationalGaborFeatures/RESULTS.md` states:

> no task has been constructed on which the conjunction layer beats the orientation
> statistics — not EMNIST, not the binary `F`/`f` probe, and not a synthetic benchmark built
> specifically to require co-location.

**That sentence is now false.** Phase 9 is such a task. The EMNIST conclusion stands
unchanged — the layer really does add ≈ 0 there, for the reason Phase 7 gave, that A is 93 %
linearly recoverable from `orient` on handwriting. What Phase 9 shows is that this was a
statement about **EMNIST**, not about the operators: on stimuli where corner angle and
junction order actually vary independently of everything else, the conjunction blocks carry
information the orientation statistics do not.

---

## Caveats

*What these numbers do not license — chiefly that every image is a single stroke of one width and one contrast on a flat field, which makes the dataset poor for any design choice depending on image statistics.*

**The CNN is undertrained.** 12 epochs on CPU, validation R² still rising at the last epoch
(0.549 → 0.556). More training would raise it, possibly a lot. Its numbers are a floor, not
a demonstrated ceiling, and should be read that way.

**The shuffle control is one permutation**, not a distribution over permutations. Worth
repeating across seeds before this is published.

**The pooling grid is 3×3** — cells ~37 px on a 112 px image, while figures are 68–96 px
long. `brokenness` at 0.340 is the row most likely limited by that: a 3 px gap is a small
perturbation inside a 37 px cell. `grid=5` is a one-parameter change and the obvious next
run.

**The representation is not translation invariant** — a fixed spatial grid against
randomised position, where a CNN is translation-equivariant for free. That handicap runs
against us throughout.

**`closedness` measures the wrong thing**, as set out above. Every number in that column,
for every arm, should be discounted.

**One seed per arm.** No error bars. Differences of 0.02 should not be read as real;
differences of 0.4 should.

---

## Open

*What Phase 9 left unfinished.*

1. **Repeat the shuffle control across seeds**, and add a matched-column control (a random
   subset of 135 `all` columns against 135 `orient` columns).
2. **`grid=5`**, to see whether `brokenness` and `vangle` are operator-limited or
   pooling-limited.
3. **Train the CNN to convergence**, on a GPU if one is available, so the comparison is
   against a real ceiling.
4. **Re-check the EMNIST numbers with the polarity fix in place.** EMNIST's background is
   zero so the border artefact should be absent, but "should be" is not "was measured".
5. **A closed curve carrying a corner or a branch**, which would remove the
   `closedness`–`vangle`–`arms` coupling of ~0.22 that this dataset could not avoid.

---

# Part 2 — Phase 12: turning the front end's own dials

Phase 9 tests the front end **as built**. This asks whether it improves when given more to work
with. The pooling grid had already been swept; the number of scales, the number of orientations,
the order of the orientation harmonics and the number of ray probe distances had not.

## Plain-language account

***Start here.** What the front end is, what the eight properties mean, what R² means, which four dials were turned, what happened, and what stayed broken — written for a reader who has not followed the project.*

`SWEEP_FULLN.md` (the proper version) and `XSCALE.md` (a follow-up). Read this first.*

---

### What the front end is, and what we were asking

The **front end** is a fixed recipe for measuring an image. It looks at every pixel through a
bank of filters tuned to different **orientations** (which way an edge runs) and different
**scales** (how coarse or fine the detail is), and boils the result down to **31 numbers** per
image. Nothing about it is learned — the recipe was designed, and the same recipe is applied to
every image.

To test whether those 31 numbers are any good, we generate pictures of a single stroke on a grey
background and ask a small network to read eight things off them:

| | in plain terms |
|:--|:--|
| **curvedness** | how bent is the stroke |
| **brokenness** | is there a gap in it, and how big |
| **closedness** | is it a closed loop or an open arc |
| **vangle** | if it has a kink, how sharp is the kink |
| **arms** | is it a plain stroke, a T-junction, or a crossing |
| **thickness** | how wide is the stroke |
| **fuzziness** | how blurry are its edges |
| **polarity** | is it lighter or darker than the background |

Scores are **R²**: **1.0** means the property is read off perfectly, **0.0** means no better than
always guessing the average, and **negative** means worse than guessing.

**The question in this round:** the front end has several dials that had never been turned. Would
turning them up make it better?

---

### The four dials we turned

**1. More scales.** Use five filter sizes instead of three, so the range from coarse to fine is
sampled more finely.

**2. Higher harmonics.** At each point the front end summarises *which directions have edges* —
like a compass rose of edge strength. It used to keep only a coarse summary of that rose. We let
it keep a more detailed one. Costs five extra numbers.

**3. More orientations.** Use 16/24/32 filter directions instead of 8/12/16 — a finer compass —
while keeping each filter's tuning just as broad as before.

**4. More probe distances.** To decide whether something is a T-junction or a crossing, the front
end reaches out a fixed distance from each point and asks "is there a stroke over there?". It only
ever reached out *one* distance per scale. We let it reach three.

---

### What happened

**More scales made things worse.** This was the one we most expected to help, and it hurt — every
time. Rejected.

**Higher harmonics helped, cheaply.** Better readings of how *bent* a stroke is, consistently, for
five extra numbers. Sensible in hindsight: curvature is about how the compass rose *spreads out*,
and the coarse summary couldn't describe that shape.

**More orientations did nothing useful — until the test got harder** (see below). On ordinary
tests it bought nothing at all, for twice the computation.

**More probe distances helped the most, and broadly** — better at spotting gaps, kinks and
junctions. This was also the cheapest to justify: we'd checked earlier that two of the three
existing probes were reading something other than what the theory said they should.

---

### The harder test, which changed the answers

Reading properties off images drawn from the *same* pool you trained on is easy. The demanding
test is **training on one range and testing on another** — for example, train only on thin
strokes, then test on thick ones. That asks whether the measurements really capture the property,
or merely memorised the range they saw.

**Almost everything looked different under that test**, and mostly in our favour:

- Gains that seemed large on the easy test **shrank by two to five times**.
- The extra orientations, worthless on the easy test, became one of the **best** dials.
- Combining higher harmonics with more probe distances was slightly *worse* than probe distances
  alone on the easy test, but **clearly better** on the hard one.

**The practical lesson: judge on the hard test.** Judged on the easy test alone we would have
picked the wrong configuration twice over.

---

### The one thing that stayed broken

There is a specific weakness we already knew about. **The front end confuses thickness with
blurriness.** Train it on sharp-edged strokes and show it blurry ones, and its thickness readings
collapse to far worse than guessing (R² ≈ −2). Train it on thin strokes and show it thick ones,
and its blurriness readings collapse the same way.

The reason is intuitive: a **thick sharp** stroke and a **thin blurry** one produce very similar
patterns of filter response. Both put a lot of energy into the coarse filters. Telling them apart
needs something the front end didn't measure.

**So we tried to measure it.** A thick sharp stroke has energy in the coarse filters *and* the
fine ones, because its edges are crisp. A thin blurry stroke has energy only in the coarse ones.
So we added a number recording **what fraction of the energy sits in the finer filters** — two
extra numbers in total.

**It did not work.** A first attempt appeared to improve the problem by about 0.16 in both
directions, and that was written up as the first real progress on it. Re-running with the
operator implemented correctly, the improvement disappears: +0.02 in one direction and −0.27 in
the other. The apparent success came from a mistake in our own code — we had written a new
version of a measurement the codebase already contained, and the two differed in a small way
(one takes a square root, the other does not) that turned out to matter.

**So nothing yet fixes this weakness.** The measurement we added does capture something — it
improves ordinary accuracy on thickness and blurriness — but it does not help when the range
shifts, which is the actual problem.

The scores remain deeply negative (−2.04 and −2.72), so a readout trained on one range is still
badly wrong on the other.

---

### Where this leaves the front end

| if you care most about | use | numbers |
|:--|:--|--:|
| shape under changing conditions | harmonics + probe distances | 54 |
| ordinary accuracy | the above **plus** cross-scale | 58 |

**No single setting wins everything** — the second is better on ordinary accuracy but worse on the
shape readings when conditions change, and neither repairs the thickness/blur weakness. Which to prefer depends on the eventual application, not on the front
end.

Both are better than the 31-number original on essentially every measure.

---

### Two mistakes worth recording

**We overstated an early result.** A first, cheaper pass (on a fifth of the data) suggested two
dials had solved the thickness/blur problem outright. Repeated properly on the full data, that
gain **vanished entirely**. Small-scale trials systematically exaggerate the benefit of adding
capacity, because extra capacity helps most when data is scarce. They are useful for deciding
*what to test properly*, never for the size of an effect.

**We declared an idea dead too early, then revived it on bad evidence.** The cross-scale
measurement was tested alone, failed, and was written off — then appeared to work in combination,
so the write-off was withdrawn. Both moves were wrong: the combination result came from a coding
mistake. The real lesson is narrower and duller than either — **check that you are running the
thing you say you are running.** The codebase already contained this measurement; we wrote a
second version without noticing, and the two were not identical.

---

### What we would do next

The remaining weakness needs a different kind of answer than "measure more of the same". The plan
is to look at a large pretrained network that *doesn't* have this weakness, find which of its
internal measurements distinguish thickness from blurriness when the range shifts, and see what
those measurements actually respond to — rather than guessing at another formula.

---

## Full-n confirmation — the authoritative numbers

*The numbers to quote. Four arms across four splits at 16,000/4,000. Establishes the adopted configuration — higher harmonics plus crossed ray offsets, 54 features — whose value is **robustness**: its advantage over the baseline grows with distribution shift and would have been missed on the i.i.d. split alone. Also contains two independent cross-checks that passed.*

`Sweep_Capacity.jl` with `SW_NTRAIN=16000 SW_NTEST=4000 SW_EPOCHS=100`, log in `confirm.log`.
Grid 1, MLP readout, four splits. Supersedes the reduced-n numbers in `SWEEP.md`.

**Two cross-checks passed.** The baseline reproduces `ConVNextTest`'s independently-computed
`ours·MLP` to **≤ 0.006** on both the i.i.d. and polarity splits — two harnesses, same images,
same protocol. That run also predates the A₁ analytic-floor default, so it doubles as a
measurement of that change: **the floor subtraction moves the published tables by ≤ 0.006**, far
inside run-to-run noise. The reproduction flag is still right to keep, but the risk was much
smaller than the warnings claimed.

### Δ from baseline, all four splits

| arm | nfeat | split | curved | broken | vangle | arms | thick | fuzzy |
|:--|--:|:--|--:|--:|--:|--:|--:|--:|
| harmonics +C₆C₈ | 36 | i.i.d. | +0.026 | −0.005 | +0.007 | +0.001 | −0.015 | −0.017 |
| | | polarity | +0.031 | +0.002 | +0.002 | 0.000 | −0.019 | −0.019 |
| | | blur | +0.028 | +0.036 | +0.030 | +0.002 | −0.051 | — |
| | | thickness | **+0.075** | +0.010 | +0.011 | +0.003 | — | −0.005 |
| offsets ×3 | 49 | i.i.d. | +0.018 | +0.026 | +0.018 | +0.015 | +0.029 | +0.032 |
| | | polarity | +0.017 | +0.025 | +0.011 | +0.012 | +0.028 | +0.031 |
| | | blur | +0.003 | +0.047 | +0.045 | +0.026 | −0.018 | — |
| | | thickness | +0.065 | +0.044 | +0.024 | 0.000 | — | **−0.151** |
| **harmonics+offsets** | 54 | i.i.d. | +0.027 | +0.022 | +0.016 | +0.011 | +0.017 | +0.012 |
| | | polarity | +0.033 | +0.020 | +0.016 | +0.014 | +0.022 | +0.031 |
| | | blur | +0.037 | +0.060 | +0.049 | +0.018 | +0.009 | — |
| | | thickness | **+0.080** | **+0.075** | +0.030 | −0.002 | — | −0.012 |

### Verdict: adopt `harmonics + offsets` (54 features)

**Its value is robustness, not i.i.d. accuracy.** Gains against baseline grow monotonically with
distribution shift — ~+0.02 i.i.d., +0.04–0.06 under blur, +0.08 under thickness shift. On the
i.i.d. split alone `offsets` by itself is marginally better, and judged there the combination
would have been rejected.

**And the combination is only additive under shift.** On i.i.d. it is slightly *worse* than
`offsets` alone on five of six rows — the two axes are largely the same information by two routes.
Under shift it becomes genuinely additive: it keeps offsets' geometry gains **without offsets'
fuzziness penalty** (−0.012 against −0.151 under thickness shift). That is the single strongest
reason to take the combination over `offsets` alone.

**Polarity invariance is untouched** by either axis: every arm's polarity-split numbers are ≥ its
own i.i.d. numbers.

### What did NOT happen

**The thickness/fuzziness confound is unmoved.** *(This statement was briefly contradicted by
`XSCALE.md` and then restored: the contradiction came from an implementation error, and it stands
as originally written.)* Every arm sits at −2.05 to −2.11 on the fuzziness
split and −2.45 to −2.60 on the thickness split, against baselines of −2.060 and −2.448. The
reduced-n sweep's +0.25 was an artefact.

This strengthens rather than weakens the case for a genuinely different operator. Both properties
are about **how energy is distributed across scale at one location**, and no amount of finer
sampling along the existing axes separates them — 5 scales made it worse, more offsets did nothing.
`AndLayer` already implements `:A3`, cross-scale conjunction, currently **off by default** with the
comment *"not because any argument requires it"*. There is now an argument requiring it.

### Method note worth carrying forward

Reduced-n selection inflated every gain 2–5× and reversed one sign. If a sweep of this kind is run
again, treat reduced n as a filter for *which arms to confirm*, never as a source of effect sizes —
and confirm on the extrapolation splits, since the i.i.d. split alone would have picked the wrong
configuration here.

---

## Cross-scale — proposed, published, and retracted

*A feature designed specifically to fix the thickness/blur weakness. It appeared to work, was published, and turned out to be an implementation artefact — an inline reimplementation of an operator the codebase already contained, differing by a square root. Kept in full because the retraction is the useful part.*

`Sweep_Capacity.jl` with `cross_scale=`, log in `xscale.log`. 16,000/4,000, grid 1, 100 epochs.

**The hypothesis.** Thickness and fuzziness are confounded because both are read from how energy
distributes across scale, and nothing in the feature set encodes the *relationship between* scales
at a point — only per-scale amounts, which pooling then averages. A thick sharp stroke has energy
at coarse **and** fine scales; a thin blurred one has coarse only.

**Two forms.** `:product` is the existing `a3_maps`, `C₀(k)·C₀(k+1)` — an energy, so mean pooling
is valid. `:ratio` is `C₀(k+1)/(C₀(k)+C₀(k+1))`, the fraction of energy at the finer scale — a
**bounded, scale-free** quantity, so numerator and denominator are pooled *separately* and divided
afterwards with a relative floor, by the rule established when the ray ratios turned out wrong.
Two features each at grid 1.

**Predictions, recorded before running:** product moves the confound by < 0.05; ratio moves it by
> 0.2 on both splits. Falsification stated in advance: if the ratio failed, stop proposing
operators and go empirical.

### The confound rows — corrected

Re-run with `:product` routed through the real `a3_maps`/`assemble` path.

| arm | nfeat | thickness (blur split) | Δ | fuzziness (thickness split) | Δ |
|:--|--:|--:|--:|--:|--:|
| baseline | 31 | −2.060 | — | −2.448 | — |
| A₃ alone | 33 | −2.219 | **−0.159** | −2.644 | **−0.196** |
| xscale ratio alone | 33 | −2.086 | −0.026 | −2.338 | +0.110 |
| harmonics+offsets | 54 | −2.051 | +0.009 | −2.460 | −0.012 |
| **adopted + A₃ + ratio** | 58 | −2.041 | **+0.019** | −2.718 | **−0.270** |

**Nothing moves the confound**, and `A₃` alone makes it worse in both directions.

### What the two forms are

`:product` is `AndLayer.a3_maps` — `C₀(k)·C₀(k+1)`, emitted through `assemble` like A₁ and A₂,
so it arrives as `sqrt(pooled)`. `:ratio` is `C₀(k+1)/(C₀(k)+C₀(k+1))`, the fraction of energy at
the finer scale — **bounded and scale-free**, so numerator and denominator are pooled *separately*
and divided afterwards, which `assemble` cannot express and which is why it stays inline.

### Predictions, scored

Recorded before running: product moves the confound by < 0.05; ratio moves it by > 0.2 on both
splits; and if the ratio failed, stop proposing operators.

| | outcome |
|:--|:--|
| product < 0.05 | **wrong** — −0.159 and −0.196, i.e. it makes things worse |
| ratio > 0.2 both splits | **wrong** — −0.026 and +0.110 |

Both predictions failed. The falsification condition triggered and stands this time.

### The i.i.d. side, which did work

On the ordinary split the cross-scale features are the best thing tried: `adopted + A₃ + ratio`
reaches **thickness 0.762 and fuzziness 0.781**, against 0.714 and 0.734 for the 31-feature
baseline and 0.731 / 0.746 for `harmonics+offsets`. So the information is real and useful — it
simply does not survive a shift in the range, which is the actual problem.

### What was published and withdrawn

The first version of this page reported **+0.161 and +0.160**, replicated in both directions, and
concluded that cross-scale features move the confound in combination. That came from an inline
reimplementation of `a3_maps` in `Frontend._feat` which omitted the `sqrt` that `assemble` applies
to every A-block, making it energy-like where A₁ and A₂ are amplitude-like. Through the real path
the effect is +0.019 and −0.270.

The variant's effect was real and did replicate — but it is a **raw pooled cross-scale product**,
not `A₃`, and dropping the `sqrt` for one block while keeping it elsewhere is an inconsistency
rather than a design. It is recoverable from commit `094ed0e` if anyone wants to justify and test
it on its own terms.

**Two further claims collapse with it.** "An ingredient that does nothing alone can still matter
in company" was inferred from this artefact and has no support here. And the earlier decision to
stop proposing operators, withdrawn on the strength of the combination result, should not have
been withdrawn.

---

## Appendix — the reduced-n selection pass (superseded)

*The first, cheaper sweep at a fifth of the data. Its **rankings** mostly held; its **magnitudes** were inflated 2–5× and one sign reversed. Kept because the size of those errors is the evidence for the method lesson.*

Kept because its *rankings* mostly held, and because the size of its errors is the evidence for
the method lesson above. **Its magnitudes are wrong** — see the full-n section.

`Sweep_Capacity.jl`, log in `sweep.log`. 3,000 train / 1,000 test, grid 1, MLP readout, 60 epochs,
one seed. Stimuli are a **prefix of the files in `ConVNextTest/data`**, so every arm sees
byte-identical images.

**Why these axes and not the grid.** The pooling grid was swept in Phase 9 (1–5) and grid 1 won on
every property, because stroke position is randomised and a fixed grid is pure liability. These
three axes carry no such penalty, so they can add capacity without it. They had never been swept.

**Reduced n for selection.** Absolute numbers are well below the published tables — baseline
`brokenness` is 0.463 here against 0.730 at 16,000 images — so these are for *ranking arms*, not
for quoting. With one seed, |Δ| below ~0.05 is at the noise floor.

### Δ from baseline, all three splits

| arm | nfeat | | curved | broken | vangle | arms | thick | fuzzy |
|:--|--:|:--|--:|--:|--:|--:|--:|--:|
| **harmonics +C₆C₈** | 36 | i.i.d. | **+0.058** | +0.043 | +0.031 | +0.006 | −0.021 | −0.028 |
| | | blur | **+0.060** | −0.021 | +0.041 | −0.023 | −0.077 | — |
| | | thick | **+0.045** | +0.029 | **+0.120** | 0.000 | — | −0.013 |
| **orient ×2 +C₆C₈** | 37 | i.i.d. | +0.058 | +0.019 | +0.019 | 0.000 | −0.025 | −0.031 |
| | | blur | **+0.078** | **+0.062** | +0.036 | −0.006 | **+0.237** | — |
| | | thick | +0.054 | **+0.116** | +0.076 | −0.044 | — | **+0.073** |
| **offsets ×3 crossed** | 49 | i.i.d. | +0.002 | **+0.130** | **+0.048** | **+0.030** | **+0.038** | **+0.036** |
| | | blur | 0.000 | +0.020 | +0.021 | +0.032 | **+0.275** | — |
| | | thick | +0.014 | **+0.154** | **+0.137** | −0.007 | — | −0.166 |
| scales 5 | 51 | i.i.d. | +0.002 | +0.054 | +0.008 | +0.016 | −0.028 | −0.022 |
| | | blur | −0.005 | −0.025 | +0.023 | −0.004 | −0.103 | — |
| | | thick | −0.001 | +0.120 | +0.068 | −0.010 | — | −0.547 |

### Verdicts

**Harmonics C₆/C₈ — accept.** Five extra features, and the curvedness gain replicates across all
three conditions (+0.058 / +0.060 / +0.045) with vangle behind it. The most solid result in the
sweep, and the cheapest. Mechanistically expected: curvature is about how the orientation profile
*spreads*, and C₂/C₄ cannot describe two-lobe structure with independent amplitudes.

**Ray offsets, crossed d × λ — accept for structure.** Largest gains on brokenness (+0.130 i.i.d.,
+0.154 under thickness shift) and vangle, and best on thickness-under-blur (+0.275). Two caveats:
its headline i.i.d. brokenness gain **shrinks from +0.130 to +0.020 under blur**, so part of it was
fitted to the nuisance distribution; and it makes fuzziness-under-thickness-shift worse (−0.166).

**Orientation count ×2 at constant σφ — accept, but for robustness only.** Buys **nothing i.i.d.**
(identical to harmonics-alone on curvedness, slightly worse elsewhere) for 2× the channels and 2×
the extraction time. Under *both* shifts it is the broadest gainer, and the only arm that improves
the confound pair in both directions (+0.237 thickness-under-blur, +0.073 fuzziness-under-thickness).
Plausibly because σφ is fixed, so the extra channels buy finer angular *sampling* and better
harmonic estimates — in-distribution the readout absorbs sampling error, out of distribution it
cannot.

**Scales 3 → 5 — reject.** Inconsistent brokenness gains, and it makes the thickness/fuzziness pair
worse in **all four** tests, catastrophically so on fuzziness-under-thickness-shift (−0.547). It is
also the least trustworthy arm: its `betas` were interpolated by me rather than derived from the
data's spectrum, against Phase 0's discipline. If revisited, derive them with `scale_ladder`.

### Predictions, scored

| prediction | outcome |
|:--|:--|
| offsets → `arms` and `brokenness` | **correct**, brokenness by a wide margin |
| harmonics → `vangle` | **correct in direction**, though curvedness gained more |
| scales → `thickness`/`fuzziness` | **wrong, and consistently backwards** — worse in 4/4 tests |
| orientations → further `vangle` gain | **wrong i.i.d.**; right that they help, but on robustness and on other rows |

### The one thing that moved the confound

Phase 11 found thickness and fuzziness confounded in our representation and nothing had touched it.
**Orientation count and ray offsets both move it by ~+0.25.** Scales — the axis predicted to fix
it — makes it worse, which argues the confound is **not** about scale-sampling resolution. A
blurred thin stroke and a sharp thick stroke may produce near-identical oriented-energy scale
distributions at any sampling density, in which case separating them needs something sensitive to
the *edge profile* itself: phase congruency, or the energy ratio across the stroke's two edges.

### Next

**Confirm `harmonics +C₆C₈` combined with `offsets ×3` at full n** — they gain on disjoint rows
(curvedness/vangle versus brokenness/arms), ~54 features, and neither needs the extra channel cost.
Then test whether `orient ×2` on top is worth 2× extraction for its robustness gain. Any winner
must hold on the extrapolation splits, not just i.i.d.
---

# Improving the front end: a fine scale and a spatial max

*Added 2026-07-31. Figures: `figures_predictions/pred_*.png` — predicted against true for
seven properties, six arms, at epochs 5/15/25/35/60, with binned medians and IQR overlaid.
Scripts: `Add_DeepHead.jl` (arms), `Plot_Predictions.jl` / `Replot_Predictions.jl` (figures).*

Two changes to the front end, arrived at from opposite directions, which turn out to be additive.

## Final table — ordinary test, epoch 60, 16,000 train / 4,000 test

| arm | nfeat | curved | broken | closed | vangle | arms | thick | fuzzy | polarity |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| our CNN, end to end | — | 0.509 | 0.106 | 0.985 | 0.548 | 0.222 | 0.722 | 0.911 | 0.749 |
| ours, baseline | 31 | 0.931 | 0.723 | **0.998** | 0.949 | 0.962 | 0.713 | 0.738 | −0.109 |
| ours + λ=8 | 41 | 0.932 | 0.764 | 0.997 | 0.942 | 0.961 | 0.858 | 0.886 | −0.167 |
| ours + λ=8,4 | 51 | 0.922 | 0.766 | **0.998** | 0.933 | 0.957 | 0.871 | 0.920 | −0.234 |
| ours + spatial max | 40 | 0.937 | 0.743 | **0.998** | 0.954 | 0.966 | 0.826 | 0.853 | −0.138 |
| **ours + λ=8 + spatial max** | **53** | **0.938** | **0.785** | 0.997 | **0.959** | **0.969** | **0.907** | **0.928** | −0.216 |
| frozen ConvNeXt | 1024 | 0.981 | 0.854 | **0.998** | 0.970 | 0.985 | 0.947 | 0.979 | 0.996 |

`polarity` is the row where **low is the goal** — our features are built to carry no
contrast-sign information, which is what lets them survive a polarity flip while the others
collapse. It is not a loss.

**Gap to frozen ConvNeXt, before and after:**

| | 31 features | 53 features |
|:--|--:|--:|
| curvedness | 0.050 | 0.043 |
| brokenness | 0.131 | **0.069** |
| vangle | 0.021 | **0.011** |
| arms | 0.023 | **0.016** |
| **thickness** | 0.234 | **0.040** |
| **fuzziness** | 0.241 | **0.051** |

53 numbers with zero learned parameters, against 1024 numbers from 88.6 M parameters fitted to
1.28 M photographs.

---

## Why the extra scale helped

**The finest filter was too coarse to see an edge.** The ladder was λ = 56 / 29.9 / 16 px, with
stroke widths of 3–12 px. The finest channel, at λ = 16 px with σ_x = 7.6 px, is *wider than most
of the strokes it is looking at*. It can tell you a stroke is there and roughly how much energy it
carries; it cannot resolve the stroke's own edge profile.

**And that is exactly what separates thickness from blur.** A thick sharp stroke and a thin
blurred one produce nearly identical responses in coarse channels — both dump energy at low
spatial frequency. What distinguishes them is what happens at *fine* scales: a sharp edge has
energy there and a soft one does not. With no channel finer than 16 px, the two were mapped to
overlapping regions of feature space — a **degeneracy**, distinct causes giving the same
measurement. The readout compensated with a rule calibrated on the training range, which is why
the failure appeared under changed conditions (thickness → −2.06 when trained sharp and tested
blurred) rather than as poor ordinary accuracy.

λ = 8 px (ρ = 14, σ_x = 4.5 px) is the first channel small enough to sit *inside* a stroke and
report on its edge. Result: thickness +0.145, fuzziness +0.148 — the largest movement any change
has produced on those rows.

**λ = 8 is also where the ladder's own spacing points.** 56 / 29.9 / 16 has a ratio of 1.87 per
step, so the next rung is 8.55 px. This is an extension of the existing geometric progression, not
a new kind of thing wedged in.

**λ = 4 was tried too** and adds essentially nothing beyond λ = 8 on brokenness (+0.002 after
+0.041), though it keeps helping fuzziness (+0.034). Below λ = 8 there is little left to see: the
generator's sharpest edge is a 0.8 px ramp, so the extra channels are resolving detail the stimuli
do not contain.

**Caveat for other datasets.** λ = 8 is real signal here because these images are natively 112 px.
EMNIST and Fashion-MNIST are 28×28 upsampled 4×, so λ = 8 sits exactly at their original Nyquist
limit and would mostly measure bilinear interpolation. Not a safe default without checking.

---

## Why the spatial max helped

**Every spatial statistic in this front end was a weighted mean.** No max, no quantile, nothing
sparse, anywhere.

**A mean of an energy-like map is roughly `signal strength × ink coverage`.** Half of a 112×112
image containing a stroke will have a mean roughly twice that of an image where a quarter does,
*even if the local structure is identical*. So every pooled feature silently multiplied "how
strong is the local structure" by "how much stroke is in the frame".

**That is a confound for exactly the properties that were failing.** Thickness correlates strongly
with ink coverage — a thicker stroke covers more pixels — so the mean-pooled features were
reporting a mixture of local structure and global coverage, and the readout could not separate
them. A max reports **peak local structure regardless of how much stroke there is**, removing the
coverage factor entirely.

Result, with no new filters at all: thickness +0.113, fuzziness +0.115, and small gains on every
other row — about 78 % of what the extra scale bought, for **nine numbers and no extra extraction
cost**.

**This is the same bug the project already found once and misdiagnosed.** Phase 9 discovered that
the ray ratios were being formed per pixel and averaged, so the pooled value scaled with ink
coverage, and fixed it by pooling numerator and denominator separately. That was recorded as a
property of *bounded ratios*. It was never specific to ratios — **it affects every mean-pooled
feature in the front end**, and went unnoticed for two years because there was no max to compare a
mean against.

**Why a max and not a finer grid.** A finer pooling grid also localises the signal, but it makes
the representation translation-variant, and stroke position is randomised here — which is why grid
3 and grid 5 score *worse* than grid 1 on every property. A spatial max is local **and**
translation invariant, which is the combination a grid cannot provide.

**It also explains how frozen ConvNeXt copes with global average pooling.** Its late units are
sparse and selective, so a unit's spatial mean already behaves like a detector. `A₁` and `A₂` are
dense energy-like maps whose mean is dominated by everything that is *not* the feature of
interest. Same pooling operation, very different consequences, depending on what is pooled.

---

## Why they add rather than overlap

The two fixes are different in kind, and the numbers show it. On brokenness the combination is
almost exactly the sum of the parts — `0.723 + 0.041 + 0.020 = 0.784` against **0.785** measured.

* The **fine scale adds information** the bank could not previously see: no filter was small
  enough to resolve an edge, so that measurement did not exist.
* The **spatial max stops discarding information** that was already there: the maps contained the
  signal, and the mean was averaging it away against a coverage-dependent background.

One extends what is measured; the other extracts what was measured more faithfully. There is no
reason for them to overlap, and empirically they do not.

---

## What is still unexplained: brokenness

Gap detection is the one row nothing has properly moved. It sits at 0.785 against ConvNeXt's
0.854, and the binned medians show every arm — ConvNeXt included — with a **dead zone** below a
true brokenness of about 0.4, where predictions stay pinned near zero. ConvNeXt lifts off roughly
one bin earlier than we do (0.233 at true 0.46, where we read 0.095).

Three candidates have been eliminated or nearly so:

| candidate | evidence |
|:--|:--|
| wavelength | two halvings of λ gave **+0.041 then +0.002** — exhausted |
| pooling statistic | spatial max gave only **+0.020** |
| pooling grid | finer grids make it **worse** (0.737 → 0.592 → 0.560), though confounded with translation variance |

**The live candidate is `σ_along`** — the filter's extent *along* the contour, which is what
bridges a gap. It has barely shrunk as scales were added, because `n_orient` rises with `ρ` and
fights the reduction: `σ_along = W·n·dts/(2π²ρ)`, so λ fell 14× across the ladder while σ_along
fell only 4.7× (17.0 → 3.6 px). At λ = 8 the filter still reaches 6.1 px along the stroke, and a
gap shorter than that is bridged whatever the wavelength.

**A free test:** hold the ladder fixed and use `nori = [8,12,16,8]`. Same wavelengths, same σφ,
same feature count, but σ_along at the finest scale drops from 9.7 px to about 2.4 px.

Also unqueued, and worth trying together:

* **spatial variance** as a third summary — a broken stroke's `A₂` map is two isolated spikes and
  an unbroken one's is not, so unevenness is a more natural gap detector than either mean or max;
* **coverage-normalised means**, `mean(A₂)/mean(C₀)` — the same pool-then-divide the ray block
  already uses, applied to the conjunction layer. More robust than a max, and it removes the
  confound explicitly rather than sidestepping it;
* **max applied to the remaining blocks** — `C₀` and `lowpass` are trivial; the ray *ratios* need
  the value at the argmax of `c₀` rather than a max of the ratio; the orientation harmonics have
  no per-pixel map to max, because `assemble` pools before forming them.

---

## Summarising a map: what we do now, and what else we could do

*Written out at length because the spatial max turned out to matter more than expected, and the
same reasoning applies to several other places in the front end that still use averages.*

### The problem every summary is solving

The front end measures things **at every pixel**. On a 112×112 image that is 12,544 numbers per
map, and there are dozens of maps — one per orientation per scale, plus a corner map, a stroke-end
map, and the ray maps. Handing all of that to a readout is hopeless, so each map gets **squeezed
down to a single number** (or nine, on a 3×3 grid).

**Until now that squeeze has always been an average.** Every one of them.

An average answers one question: *"across the whole picture, how much of this is there?"* That is
the right question for some properties and the wrong question for others, and nothing in the front
end could ask a different one.

### Two things an average throws away

**It dilutes anything that happens in only a few places.** A 3-pixel gap in a 68-pixel stroke makes
the stroke-end detector fire hard at two points and do nothing anywhere else. Average that over
12,544 pixels and a large response at two of them becomes a rounding error — sitting on top of the
response from the stroke's own two ends, which are always there. The smaller the gap, the more
hopeless the ratio. That is the "dead zone" visible in the brokenness scatters, where predictions
stay pinned near zero until the gap is fairly large.

**It multiplies in how much stroke is in the picture.** This one is subtler and turned out to be
the bigger problem. Take two images with *identical* local structure — same width, same sharpness —
but one has a long stroke and the other a short one. The long one covers more pixels, so its
average is higher, purely because more of the image is stroke. Every averaged feature is therefore
reporting **local structure × how much ink there is**, mixed together, with no way for the readout
to separate them. Since thickness correlates strongly with ink coverage, the features meant to
report thickness were partly reporting stroke length instead.

A **maximum** fixes both at once. It asks *"is there a strong response anywhere?"* — which does not
care how much of the rest of the image is blank, and does not dilute a small bright spot.

---

### Where the front end still uses averages, and what a max would mean there

**Already done** — max of the corner detector `A₁`, the stroke-end detector `A₂`, and the branch
count `c₀`, one per scale. Nine numbers. This is what produced the gains above.

**Trivial to add next.**

* **Total oriented energy** (`C₀`) — currently "how much oriented structure on average". A max
  would say **"how strong is the strongest piece of structure anywhere"**, which is a much more
  direct read of local contrast and stroke width, and is not diluted by empty background.
* **The lowpass channel** — the blurred, non-oriented channel carrying overall brightness. A max
  would report the strongest local departure from the background rather than the average one.

**Needs care: the ray ratios.** The ray block reports `c₀` (roughly, how many branches leave this
point) and two *ratios*, `|c₁|/c₀` and `|c₂|/c₀`, describing how lopsided the branching is. A
ratio is only meaningful where there is something to divide — where `c₀` is small, the ratio is
noise. **Taking the maximum of a ratio would therefore find the emptiest, noisiest patch of
background in the image**, which is exactly the bug Phase 9 fixed once already. The correct version
is to find **where `c₀` is largest and report the ratio at that place** — locate the strongest
junction, then describe it.

**Needs a design change: the orientation summary.** The `orient` block reports which directions
carry energy, as a handful of harmonic numbers per cell. But it computes those numbers **from the
already-averaged energies** — average first, then summarise the directions. So there is no
per-pixel "direction summary" map lying around to take a maximum of. Producing one means computing
the direction summary at every pixel and *then* squeezing it, which is a real change to how the
block works.

It is also worth doing for a reason beyond the max. **The whole conjunction layer exists because
combining before averaging is not the same as averaging before combining** — that difference is
the co-location signal `A₁` was built to capture. The `orient` block does the second ordering for
its own harmonics, and nobody has ever compared it against the first. Both are defensible; only
one has been measured.

---

### Three alternatives to a max, and why each might be better

**1. Divide the coverage back out.** If the diagnosis is right — that an average is *structure ×
ink coverage* — then the direct fix is to divide by the coverage rather than dodge it:

```
average corner strength  ÷  average total energy      =  "corner strength per unit ink"
```

This removes the confound explicitly, and unlike a max it keeps the averaging, so it is far
steadier — a max can be set by one odd pixel, an average cannot. **The machinery already exists**:
this is precisely the divide-after-averaging fix applied to the ray ratios in Phase 9. It was
simply never applied to the corner and stroke-end detectors. Of everything on this list, this is
the one I would try first.

**2. Unevenness across the image.** How *variable* is the map, rather than how large? An unbroken
stroke produces a smooth, even stroke-end map; a broken one produces the same map with **two extra
isolated spikes**. That difference is enormous in the variability and small in both the average and
the maximum. For gap detection specifically this looks like the natural measurement, and it is the
one I would expect to move brokenness — the single row nothing has shifted.

**3. A high percentile instead of the outright maximum.** The maximum is the single most extreme
pixel in the image, so one strange pixel — a rasterisation artefact, a filter ringing at the
border — sets the value. Asking instead for the **95th or 99th percentile** ("how strong are the
strongest few percent") is nearly as sensitive to a local event and much harder to fool. Worth
having as a robustness check on the max results above, since those are currently taking the
literal maximum.

**And a fourth, different in kind: counting.** *"How many places have a strong response?"* — rather
than how strong, or how uneven. That is much closer to what some of the properties actually are:
arm count is literally a count of branches, and brokenness is about whether there are two extra
stroke-ends. Neither an average nor a maximum can express "there are two of these"; a
count-above-threshold can. This has never been tried and is cheap.

**The general point.** The front end currently has exactly one way of turning a map into a number.
Every property is read through that single lens, whether or not it suits them. Adding two or three
more summaries costs a handful of features and lets the readout pick whichever one matches each
property — and the evidence so far is that they match different ones.

---

## Every feature, and exactly how it is computed

*Two terms, defined once, and then no others.*

**Filter response at a pixel** — for each *direction* (8–16 of them) and each *size* (3 filter
sizes), a number at every pixel saying how strongly a stroke of that size, running in that
direction, is present there. Always zero or positive. Everything below is built from these.

**Average over the picture** — add the value at every pixel, weighted by a smooth window, and
divide by the total weight. At grid 1 the window covers the whole image, so it really is the
average over the picture. This is the step called *pooling* elsewhere in these notes.

### The 31 baseline features

| # | feature | what it is meant to mean | exactly how it is computed, in order | ordering |
|--:|:--|:--|:--|:--|
| 15 | **orientation summary** (5 × 3 sizes) | how much structure there is, and which directions it runs in | **1.** average each direction's response over the picture → one number per direction. **2.** add those up → *total*; report **√total**. **3.** combine the same averages with weights going round twice per full turn → a two-lobed direction summary; **divide by *total***; report its two parts and its size. **4.** same with weights going round four times; report its size only. | **average first, combine second** |
| 1 | **overall brightness** | how much the picture departs from flat grey | **1.** blur the picture heavily. **2.** average over the picture. **3.** report the square root. | average is the whole thing |
| 3 | **corner strength `A₁`** (1 × 3 sizes) | are two directions at right angles present **at the same pixel** | **1.** at each pixel multiply each direction's response by the response at right angles to it, and add the products. **2.** at each pixel divide by the sum of all direction responses there. **3.** subtract a fixed known amount — what a plain straight line produces anyway — and clamp at zero. **4.** average over the picture; report the square root. | **multiply first, average last** |
| 3 | **stroke-end strength `A₂`** (1 × 3 sizes) | does the stroke stop here | **1.** at each pixel find the strongest direction. **2.** look a fixed distance forward and backward *along* the stroke and read that direction's response at both places. **3.** take the size of the difference, divided by their sum plus a stabiliser. **4.** multiply by the response at the pixel itself. **5.** average over the picture; report the square root. | **compare first, average last** |
| 9 | **branching** (3 × 3 sizes) | how many strokes leave this point, and how lopsidedly | **1.** at each pixel step out a fixed distance in many directions and read the response there — a ring of values. **2.** three summaries of that ring: its total, and how much it varies once and twice around the circle. **3.** average each of the three over the picture **separately**. **4.** report the averaged total raw, and the other two **divided by the averaged total** plus a small floor. | **average first, divide second** |

### The 9 added by the spatial max

| # | feature | how it is computed | difference |
|--:|:--|:--|:--|
| 3 | strongest corner, per size | the **largest** value of the corner map anywhere in the picture | max instead of average |
| 3 | strongest stroke-end, per size | the **largest** value of the stroke-end map anywhere | max instead of average |
| 3 | strongest branching, per size | the **largest** ring-total anywhere | max instead of average |

### What the ordering column is saying

Three blocks **combine at the pixel and average afterwards** — corner strength, stroke-end
strength, and the ring of branch values. That ordering is the project's central design claim:
multiplying two things at a pixel and *then* averaging is not the same as averaging each and
multiplying, and the difference is precisely the evidence that they happened *at the same place*.

Two blocks do the **opposite**. The branching ratios divide only after averaging — a deliberate
fix, because dividing at each pixel produced nonsense wherever there was nothing to divide. The
orientation summary averages each direction first and combines afterwards, and that one has simply
never been compared against the alternative.

And until this week **every one of them ended in an average**, which is what the spatial max
changed.
