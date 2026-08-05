# A third route to i2D — and the two invariances neither of us has both of

*On Osório, Bernardino & Wichert, "Shape Matters: Few-Shot Object Classification From
High-Information Contour Features", Neural Computation 38:1–22 (2026),
[doi:10.1162/NECO.a.1563](https://doi.org/10.1162/NECO.a.1563). Code at
[github.com/mosorio/shape-matters](https://github.com/mosorio/shape-matters). Local copy:
`OsarioSpaceMatters.pdf` in the repo root — the filename transposes the title, and the author is
Osório — excluded from the repo via `.git/info/exclude`.*

*Nothing here is a new experiment. It is a comparison, one experiment it suggests, and two things
worth stealing.*

---

## What they built

Four stages, none of them learned in the gradient sense:

1. **Marching squares** isocontour on the greyscale image → a dense contour point set `S`.
2. **Ramer–Douglas–Peucker** polyline simplification with tolerance `ε` → the keypoints `K ⊆ S`,
   which by construction are the points of high curvature. This is their Attneave step.
3. At each keypoint, a **raw `s × s` image patch** `v` (they use 5×5), plus that point's
   **object-centred normalised polar coordinates** `(r, θ)` — origin at the contour centroid,
   radius divided by `r_max` over the figure. An image is the unordered set `{(vᵢ, rᵢ, θᵢ)}`.
   They call `v` the *what* and `(r, θ)` the *where*.
4. **Prototype matching.** A pair matches if `cos(v, vᶜ) > τ_cosine`, `|Δr| < τ_r` and
   `|Δθ| < τ_θ`; greedy, one-to-one. Score the class by the Jaccard index over matched pairs,
   `|M| / (|F| + |P| − |M|)`, and take the argmax. Prototypes come from K-means.

Plus an uncertainty cascade: if the standard deviation of the per-class Jaccard scores falls below
`τ_std`, redo everything at a coarser wavelet resolution and blend, `J̄ = αJ + (1−α)J′`, `α = 0.6`.

## Why this is worth our attention

Their thesis is ours. Fixed, structurally-grounded features survive a distribution shift that a
CNN does not. The MNIST → ETL1 transfer is the same shape of argument as Phase 11's polarity flip
and Phase 13's few-shot-on-unseen-classes arm, arrived at by a completely different mechanism.
That is corroboration of a kind we have not had — Grüning & Barth ([[CompareWithGrueningBarth]])
share our starting theorem but never test an invariance at all.

The numbers, so the claim can be weighed:

| prototypes/class | theirs, MNIST | CNN, MNIST | theirs, ETL1 | CNN, ETL1 (centred) |
|--:|--:|--:|--:|--:|
| 1 | 83.1 | 65.2 ± 2.7 | 75.0 | 40.5 ± 2.9 |
| 25 | 96.1 | 90.9 ± 0.3 | 90.2 | 66.6 ± 1.1 |
| 200 | 97.6 | 96.8 ± 0.2 | 93.9 | 86.2 ± 1.1 |
| all 60 k | 98.4 | 97.8 ± 0.15 | — | 90.0 ± 0.57 |

ETL1 prototypes are drawn **only** from MNIST in every row.

## The three routes to i2D

We now have three distinct mechanisms for finding where a contour turns or branches:

| | mechanism | operator class | what it needs of the input |
|:--|:--|:--|:--|
| Zetzsche & Barth → Min-Nets, our `A₁` | AND of two oriented filters | bilinear in the image | nothing; runs on any greyscale |
| our ray transform | offset sampling of the energy field | **linear in `E`**, quadratic in `I` | nothing; runs on any greyscale |
| **this paper** | trace the boundary, simplify the polyline, keep the corners | not a filter at all | a **cleanly thresholdable figure** |

Route 3 does not contradict the Zetzsche & Barth impossibility result. It **opts out of the regime
the result applies to.** RDP is a symbolic algorithm on an already-segmented boundary, so "no linear
filter is i2D-selective" simply does not bind — there is no filter.

The price is stated in the paper's own discussion: the approach assumes discriminative information
is carried by contour structure, and they concede it "may limit performance in domains where
appearance-based cues are essential". They binarise ETL1 with Otsu to make the assumption hold. Our
front end never binarises, which is exactly why it can be pointed at Phase 10's Fashion-MNIST
silhouettes and theirs cannot without a segmentation stage in front of it.

## The two invariances, split between us

This is the substantive part. Each project solves the half the other has ignored.

### Their *where* solves a hole in ours

Our spatial pooling is a fixed retinotopic `g × g` grid of Gaussian windows
(`FrontEndDefinition.md` §"Averaging"). It has **no translation invariance and no scale
invariance**; we have been relying throughout on EMNIST and the Phase 9 stimuli being centred and
size-normalised for us. Their `(r/r_max, θ)` buys both, for the price of a centroid and a max over
the contour points — about six lines of arithmetic, equations 3.3–3.7.

Table 2 quantifies what the invariance is worth. Centring the ETL1 digits before classification
moves their model by **0.0–0.3 points** and the CNN by **12 to 31 points**. That is the cleanest
single measurement in the paper.

### Our *what* solves a hole in theirs

Their `v` is a raw 5×5 patch compared by cosine similarity, and the consequences follow directly:

| property | their `v` | our per-cell block |
|:--|:--|:--|
| contrast **scale** | invariant — the norms divide out of the cosine | invariant — `z₂`, `z₄` are ratios; `√T` scales linearly |
| contrast **polarity** | **fails.** `I → −I` sends `cos(v, vᶜ) → −1`, so every match is rejected | exact, by `E = \|r\|²` |
| rotation | no | `\|z₂\|`, `\|z₄\|`, `a₁/a₀`, `a₂/a₀` all invariant |
| scale of the local structure | no — `s` is fixed at 5×5 | 3-scale ladder, though with the thickness/fuzziness degeneracy of [[FourierGaborBankRelations]] |

Their **contour** stage, oddly, *is* polarity invariant almost by accident: the level set
`{I = 0.5}` and the level set `{1 − I = 0.5}` are literally the same point set, so marching squares
and RDP return identical keypoints on an inverted image. So on the Phase 11 polarity test they
would keep the *where* exactly and lose the *what* entirely. They never run such a test.

### The experiment this suggests

**Replace their `v` with our block.** Compute the 10 numbers per scale in a Gaussian window centred
on each RDP keypoint, keep their `(r, θ)`, keep the Jaccard matcher, keep their thresholds as the
only thing that needs retuning. That is a direct test of whether our descriptor beats raw pixels at
the one job it was designed for, against a published protocol with published numbers — and it would
be the first time our features have been evaluated by anything other than a linear or MLP readout.

The converse is equally cheap and probably worth doing first: **bolt their object-centring onto our
pooling**, replacing the `g × g` retinotopic grid with cells binned in `(r, θ)` about the ink
centroid. It is the smallest available fix to a real gap, and it is testable on the Phase 9 stimuli
by translating and rescaling them, which the current harness does not do.

## Where to be sceptical

**Table 1 is a weak-baseline comparison.** The CNN is a shallow 3-layer net trained on 5 images per
class, and they say outright that "the goal is not to optimize CNN performance". That is precisely
the failure mode the root `README.md` already warns about — leave-one-out NCM understating features
by ~24 points, and `P0.6_TestFeaturesWithMLP/` overturning a qualitative conclusion once a real
classifier was used. Table 1 is not evidence about representations. **Table 2's last row is** — a
CNN trained on all of MNIST reaching 67.2 % uncentred / 90.0 % centred on ETL1 against their
93.9 % — and that comparison is fair and the result is good.

**"No training" is doing some work.** `τ_cosine`, `τ_r`, `τ_θ`, `τ_std`, `α`, `ε` and the patch
size `s` are all hand-set, and `τ_cosine` is retuned from 0.8 to 0.75 specifically for ETL1. Seven
hyperparameters chosen against test performance is not zero fitting; it is fitting with a very
small parameter count, which is a different and weaker claim than the one the paper makes.

**Prototypes are selected in the wrong space.** Equation 3.8 K-means over "flattened images" —
raw pixels — and then the selected images are represented by shape features. The clustering metric
and the classification metric are different metrics. They do not remark on this.

**No curves.** The CNN arms report a final number plus "in all cases, the CNN achieved 100 %
training accuracy, which reflects overfitting rather than true generalization". A per-epoch
validation curve is exactly what distinguishes overfitting from an unstable run, and it is not
there. This project has been bitten by that specific gap before.

## Two things worth taking

**ETL1 as an out-of-distribution test set.** 14,416 handwritten digits from the ETL Character
Database, a *real* corpus shift rather than a synthetic transform of our own devising, with a
published preprocessing recipe — pad 63×64 to 64×64, 2×2 mean-pool to 32×32, Otsu binarise, drop
images with fewer than 10 active pixels — and now published baselines to hit. That is a better
probe for Phase 13's generalisation question than anything currently in the repo, and it is the
same digit vocabulary EMNIST already uses.

**Accuracy as a function of prototypes per class.** Their entire evaluation is a curve in the
low-data direction rather than a point at full data, which is what makes the CNN comparison
informative at all. It is the same instinct as Phase 9's graded properties — turn a yes/no into a
gradient — applied to sample count instead of to stimulus geometry, and it composes directly with
Phase 13's few-shot arm.

## What we could tell them that they do not know

**Their descriptor throws away an invariance their own front end already has.** The contour stage
survives contrast inversion exactly; the patch descriptor destroys it. They have not noticed
because they never test polarity — a fix as crude as taking `|cos(v, vᶜ)|` would recover it, and a
proper oriented-energy descriptor would recover rotation and scale with it.

**Object-centring and offset sampling are orthogonal.** Their `(r, θ)` is a *global* frame for
*where the keypoints are*; nothing in their representation says what kind of point each one is.
A corner and a T-junction at the same `(r, θ)` are told apart only by their 5×5 patches, i.e. by
appearance, which is why the matcher needs three thresholds. The ray harmonics give the type
directly and would let `τ_cosine` do less work.
