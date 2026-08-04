# Phase 10 — Fashion-MNIST

The first dataset in this project that is not a line drawing. Filled silhouettes with texture
rather than strokes on an empty field, and **published baselines** to calibrate against instead
of only our own arms. See `README.md` here for the benchmark literature and its caveats.

60,000 train / 10,000 test, 10 classes, chance 10.0 %. Features extracted at 112×112 after
bilinear upsampling from 28×28, exactly as the EMNIST phases do. One hidden layer of 256, best
epoch chosen on a validation slice carved out of training. The CNN is trained at **native
28×28** — the published baselines are at that resolution, and upsampling gives a convolutional
net no information it did not already have.

Full log: `phase10_full.log`. Extracted features are cached under `cache/`, keyed by grid and
split sizes, so the figures below can be refitted without repeating the 29 minutes of
extraction.

---

## Results

| arm | features | accuracy |
|:--|--:|--:|
| **pixels + MLP** *(calibration)* | 12,544 | **87.70 %** |
| **CNN trained here**, 28×28, no augmentation | — | **93.10 %** |
| | | |
| **grid 3** — orient+lowpass | 144 | 89.03 % |
| grid 3 — + A₁+A₂ | 198 | **89.70 %** |
| grid 3 — **+ A SHUFFLED** *(control)* | 198 | **88.08 % ± 0.14** |
| grid 3 — + rays | 225 | 89.57 % |
| grid 3 — everything | 279 | 89.66 % |
| grid 3 — A₁+A₂ alone | 54 | 86.51 % |
| grid 3 — rays alone | 81 | 86.26 % |
| | | |
| **grid 1** — orient+lowpass | 16 | 78.31 % |
| grid 1 — + A₁+A₂ | 22 | 80.69 % |
| grid 1 — **+ A SHUFFLED** *(control)* | 22 | **77.84 % ± 0.27** |
| grid 1 — + rays | 25 | 81.50 % |
| grid 1 — everything | 31 | 81.63 % |
| grid 1 — A₁+A₂ alone | 6 | 55.39 % |
| grid 1 — rays alone | 9 | 65.20 % |

Published for comparison: linear on pixels 85.2 %, MLP ≈ 88 %, the CNN cluster that replicates
93.7–94.6 %, hybrids 95–96.6 %.

**The calibration arm validates the harness.** Our own pixels+MLP gives 87.70 % against a
published ≈ 88 %, so the pipeline is not quietly losing or gaining points somewhere. Our CNN's
93.10 % lands just under the replicating cluster, which is where an unaugmented net under a
fixed protocol should sit — so it is a comparison our other arms can actually be scored against,
which the literature figures are not.

---

## The main result: the conjunction layer pays on someone else's dataset

`+ A SHUFFLED` permutes the A₁+A₂ block's rows across samples. Same column count, same
marginals per column, only the correspondence with the image destroyed. Whatever the real block
scores above it is what conjunction contributes, with capacity held fixed.

| | grid 1 | grid 3 |
|:--|--:|--:|
| orient+lowpass | 78.31 % | 89.03 % |
| + A shuffled *(control)* | 77.84 % | 88.08 % |
| + A₁+A₂ *(real)* | 80.69 % | 89.70 % |
| **conjunction, vs control** | **+2.85** | **+1.62** |
| σ over 5 permutations | 0.27 | 0.14 |
| cost of the shuffled columns alone | −0.47 | −0.95 |

Both gains are more than ten times the permutation spread.

**The control changes the conclusion rather than merely confirming it.** Read naively against
the base, A₁+A₂ is worth +0.67 at grid 3 — small enough to look like the ≈ 0 that EMNIST gave,
which is what prediction 2 said. But the 54 extra columns are themselves *harmful* here, worth
−0.95, so the real contribution is +1.62. Without the shuffled twin we would have recorded
"≈ 0 again, as on EMNIST" and been wrong by a factor of two and a half.

That −0.95 is worth noting on its own: EMNIST's Phase 5b measured −0.75 for the same 54
shuffled columns. Two datasets, independent runs, the same penalty to within 0.2 — an external
check on the control machinery.

**Why this matters more than Phase 9's +0.16 to +0.21.** Phase 9 found the conjunction layer
paying for the first time, but on stimuli we designed ourselves, which always leaves open the
objection that the task was built to suit the operator. Fashion-MNIST is someone else's dataset
with published baselines, and the layer earns +1.62 to +2.85 on it. The EMNIST result
(+0.01 across five lines of evidence) now looks clearly like a fact about handwriting rather
than about the operators.

---

## Phase 10b — why? Not for the reason Phase 7 gave

`Phase10b_Collinearity.jl`, log in `collinearity2.log`. Phase 7 explained the EMNIST null with
`R²(A ← orient) = 0.933` — A and orientation energy near-collinear *on handwriting*, so "real
letters do not produce the configurations that separate them". If that is the explanation,
garments must come out markedly **lower**.

Same estimator as Phase 7, both halves recomputed by the same code on the same day, grid 3:

| dataset | median R²(A ← orient) | mean | cols > 0.9 | conjunction gain vs control |
|:--|--:|--:|--:|--:|
| EMNIST balanced, 20k/10k | 0.931 | 0.923 | 42 / 54 | **+0.01** |
| Fashion-MNIST, 60k/10k | **0.943** | 0.933 | 44 / 54 | **+1.62** |

**Fashion-MNIST is *more* collinear, and the layer is worth 160× more there.** The hypothesis is
refuted, and cleanly: `R²(A ← orient)` does not predict whether conjunction helps.

The EMNIST figure reproduces Phase 7 to 0.002, with the column counts matching to the integer,
so this is a correction to the **inference** and not to the measurement. The error is a standard
one — R² measures how much of A's *variance* is linearly predictable, and variance is not
information about the label. A 5.7 % residual can carry all of the class-relevant signal, and
on garments it evidently does.

### Nor is it A₁ leaking on straight lines

The second candidate, from Zetzsche & Barth's i2D criterion — if A₁ responds to i1D input it is
partly a function of orientation energy by construction. `P0-8_RationalGaborFeatures/Validate_i1D.jl`
measures it, and A₁ **does** leak at the coarsest scale (4.6 × 10⁻² of the crossing response at
ρ = 2, against 1.6 × 10⁻⁴ and 6.1 × 10⁻⁸ at ρ = 3.74 and ρ = 7). But per-scale R² settles it:

| | ρ = 2.0 | ρ = 3.74 | ρ = 7.0 |
|:--|--:|--:|--:|
| i1D leakage of A₁ | 4.6e−02 | 1.6e−04 | 6.1e−08 |
| median R²(A₁ ← orient), Fashion | 0.957 | 0.913 | **0.883** |
| median R²(A₁ ← orient), EMNIST | 0.951 | 0.909 | **0.856** |

Leakage spans **six orders of magnitude** across the three scales; R² moves by 0.07. The scale
with effectively no leakage is still 86–88 % predictable. So leakage contributes at most a few
points at ρ = 2 and is not what makes A predictable.

### What is left standing

The surviving explanation is the plain one: **how much i2D structure the images contain.**
EMNIST strokes are i1D almost everywhere with a handful of junctions per character; woven fabric
is i2D nearly everywhere. A₁ simply has more to report on garments — consistent with A₁+A₂ alone
reaching 86.51 % here, and with the gain being larger at grid 1 (+2.85), where the base
representation is most starved. That is a hypothesis with a measurement attached to it, not yet
a result.

---

## The three predictions

Recorded in `README.md` before the run.

### 1. Features between the published MLP and CNN numbers, ≈ 88–91 % — **correct**

**89.70 %** at grid 3, from 198 numbers with no learned parameters in the representation. Above
the published MLP on 12,544 pixels, and **3.40 points below our own CNN** at 93.10 %.

Worth noting what does the work: **A₁+A₂ alone reach 86.51 % from 54 numbers**, and the ray
block alone 86.26 % from 81. Neither was designed for texture.

The CNN winning here is the mirror image of Phase 9, where a properly trained CNN lost to a
linear readout on our features on four of five geometric properties. Geometry is what the front
end is built for; texture discrimination at 28×28 is what a learned filter bank is good at.
Both results are what the inductive-bias argument predicts, in opposite directions.

### 2. The AND layer adds ≈ 0, as on EMNIST — **wrong**

**+1.62 at grid 3 and +2.85 at grid 1** against the shuffle control, at σ ≤ 0.27. On EMNIST the
same layer was +0.01. See the section above; the reasoning behind the prediction — silhouettes
contain few junctions — was sound and simply not what the operators respond to. A₁ is an
i2D-selective operator, and fabric texture is full of i2D structure that is not a junction.

The gain is larger at grid 1 (+2.85) than at grid 3 (+1.62), consistent with a 16-column base
being starved: independent information is worth more when there is less of it.

### 3. Grid 3 beats grid 1 — **correct, and by a wide margin**

| | grid 1 | grid 3 | difference |
|:--|--:|--:|--:|
| orient+lowpass | 78.31 % | 89.03 % | **+10.7** |
| everything | 81.63 % | 89.66 % | **+8.0** |

This is the reverse of Phase 9, where 31 globally pooled features beat 775 gridded ones on
every property, and it settles what that result meant.

**In `P9_P12_SimpleStrokeTests` position is randomised**, so a fixed spatial grid is pure liability —
the same shape at a different location produces different numbers and the readout must learn to
undo it. **Here garments are centred with their parts in consistent places** — sleeves up, soles
down, straps at the top of a bag — so the grid carries real information.

So the Phase 9 grid result was about **position randomisation in that dataset**, not about
pooling in general, and the caution attached to it in `P9_P12_SimpleStrokeTests/RESULTS.md` was
correct. The right grid is a function of whether object parts land in predictable places, which
for natural images they largely do.

---

## The ray block does not earn its columns here

At grid 3, adding rays to the base is worth +0.54 (89.03 → 89.57), and adding them on top of
A₁+A₂ is worth **−0.04** (89.70 → 89.66). That is 81 columns for nothing.

This is the expected outcome on filled silhouettes, which contain few junctions — the ray
transform's whole purpose is converting a mod-π orientation reading into a mod-2π ray count.
It is also concrete support for the standing complaint that the ray harmonics spend a large
number of features for little return: on this dataset they measurably do, while A₁+A₂ get more
from 54 columns than rays get from 81 (86.51 % vs 86.26 % standalone).

No conclusion about the ray transform in general follows. Nothing here has junctions to count.

---

## What this does and does not establish

**Establishes:** the front end works off line drawings. Multi-scale oriented energy is a usable
texture descriptor — 89.70 % on a texture-and-silhouette task, beating an MLP on raw pixels
from 63× fewer numbers, with nothing in the representation fitted to the data. And the
conjunction layer contributes on a dataset this project did not construct.

**Does not establish** anything about the design questions left open by Phase 9. The background
is black and uniform, there is one centred object, contrast barely varies within a frame, and
the resolution is 28×28 upsampled. **Divisive normalisation, whose whole purpose is handling
spatially varying local contrast, still has no test here.** That needs natural greyscale images;
BSDS boundary detection remains the intended target.

## Open

1. **Measure i2D content directly** and check it against the conjunction gain. That is now the
   only surviving explanation for +0.01 on EMNIST against +1.62 here, and unlike the two
   hypotheses Phase 10b eliminated it has not been tested at all. The natural statistic is the
   fraction of above-threshold energy at which A₁/C₀ exceeds its i1D floor, per dataset.
2. **A₁'s i1D floor at ρ = 2 is 4.6 %**, from the ±45° channel pair. Two fixes, both with costs
   — more orientations lengthens the coarsest filter from σ_along 17 → 26 px, and subtracting
   the closed-form floor `c(n,σφ)·C₀` invalidates every table in this project. Left as a
   recorded decision; nothing here turns on it.
3. **The near-duplicate correction.** About 6 % of the test set are near-duplicates of training
   images (arXiv:1906.08255), so every number on this page, ours included, is slightly
   optimistic. It costs ~0.4 points where it has been measured, and it applies to all arms, so
   no comparison here is affected.

---

## Convergence — the curves this phase originally did not keep

Phase 10 first reported final numbers and saved **no accuracy curves at all**. The only per-epoch
record anywhere was four sampled lines for the CNN arm in `phase10_full.log`, and nothing for the
feature arms — so none of the numbers below could be checked for convergence. Re-run 2026-08-04
with per-epoch validation *and* test accuracy recorded for every arm.

Recording only. Selection is still best-epoch-on-validation, unchanged, so the numbers stay
comparable — and they are: **all fourteen feature-arm accuracies reproduce exactly**, to the digit.
Only the CNN moved, 93.10 → 92.98, which is cuDNN nondeterminism in the convolution backward pass.

`Phase10_FashionMNIST.jl` → `curves.jls` → `Plot_Phase10.jl` → `figures/phase10_curves.png`
(one panel per arm, with the selected epoch marked) and `figures/phase10_curves_compare.png`.

**The feature arms are converged.** Every one has a last-five-epoch slope under 0.08 points/epoch
and a last-five standard deviation of 0.22 or less, and every reported number sits within 0.31
points of its own plateau. Best-epoch-on-validation is doing essentially nothing for them — which
is the result we wanted and could not previously demonstrate.

**The pixel calibration arm is the exception, and it is the one that matters for the external
comparison.**

| | last-5 sd | last-5 slope | selected epoch | reported − plateau |
|:--|--:|--:|--:|--:|
| `pixels + MLP` | **0.85** | **+0.479 /epoch** | **24 of 25** | **+0.70** |
| every feature arm | ≤ 0.22 | ≤ 0.08 | 15–25 | −0.37 … +0.31 |

That arm is four times noisier than any other, still climbing when training stops, and its reported
87.70 % sits 0.70 points above its own last-five mean — so best-epoch selection *is* picking a
favourable point off a noisy curve, and 25 epochs is not enough for it.

**What this changes.** The arm-versus-arm and arm-versus-control comparisons are unaffected: those
are all between feature arms, all converged, all reproduced exactly. What it qualifies is
**Prediction 1**, which calibrates our features against "the published MLP ≈ 88 %" using this arm.
The comparison is between a published, presumably converged figure and one of ours that is neither
converged nor stable. It is not wrong, but it is softer than it reads, and it should be quoted with
that attached. Re-running the pixel arm to 60+ epochs would settle it and costs minutes.

**The `g1: + A SHUFFLED` control selected epoch 25 of 25**, its last. Its end slope is +0.036
points/epoch, so it is flat rather than truncated, and the control conclusion stands — but it is
worth noting as the other place the run brushed its epoch budget.
