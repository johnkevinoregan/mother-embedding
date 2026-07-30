# The phases — what each one asked, and what it found

A map of the phased work. Phases **0–4** are design and validation, carried out in the
validator notebooks rather than as experiment scripts; **5 onward** are experiments, one
script each.

**What came before Phase 0.** The numbering starts with `RationalGaborFeatures/`, so it does
not cover the earlier lines of work: the dense-Gabor keypoint detector (`Dense_Gabors/`,
superseded), junction type by linear projection (`New_Gabor_FPE/`), the Fourier grid
(`ExptsWithGlobalFourier/`), Zernike moments (`ExptsWithZernike/`), and the MLP re-evaluation
in `TestFeaturesWithMLP/` that overturned the leave-one-out nearest-class-mean conclusions
used throughout the earlier work. Those are documented in the root `README.md` and the
`PROGRESS_*.md` journal. Two results from that period matter for reading what follows: the
leave-one-out protocol **understated features by ~24 points** and in one case manufactured a
qualitative conclusion a stronger classifier does not reproduce; and §7.11 of
`TestFeaturesWithMLP/README_MLP_FPE_Experiment.md` is the few-shot comparison whose missing
control became Phase 6.

Directories: `RationalGaborFeatures/` (the front end and the EMNIST phases),
`SimpleStrokeTests/` (Phase 9), `FashionMNIST/` (Phase 10). Each has its own `README.md` for
how to run it and `RESULTS.md` for the tables; this file is the map across all of them.

---

## 0–4 — building the front end and proving it computes what it claims

| | question | verdict | where |
|:--|:--|:--|:--|
| **0** | Can every parameter be fixed by *measurement* rather than convention? | Yes. Scale ladder from EMNIST's measured spectrum — ρ = 2.00 / 3.74 / 7.00, so λ = 56 / 30 / 16 px at 112. Stroke width measured at 12.67 px. Angular tuning from the along-contour extent. | `RationalGaborFeatures/README.md` §4.1 |
| **1–2** | Does the bank do what a bank should? | Exact quadrature, zero DC by construction, RMS normalisation so scales are comparable, polarity invariance, correct EMNIST counts. | `Validate_GaborStack.jl` |
| **3** | Does the conjunction layer detect co-location rather than co-occurrence? | Yes on synthetic ground truth. A₁ separates a corner from two disjoint strokes with the same orientation content, and orders junctions straight < L < T < X. | `Validate_AndLayer.jl` |
| **4** | Does pooling preserve it, and is there a control? | Soft 3×3 Gaussian pooling, with `shuffle_block!` — same columns, same marginals, correspondence destroyed — built in from the start. | `Validate_Pooling.jl` |

Later additions: `Validate_ScaleFree.jl`, `Validate_RayHarmonics.jl`, `Validate_Convention.jl`
(that `theta` is the carrier everywhere, checked to 1.0° on a full 2π ray-direction recovery),
and `Validate_i1D.jl` (Phase 10b — that A₁ really does vanish on i1D input; it does at two
scales out of three).

---

## 5–8 — does any of it help on EMNIST?

### Phase 5a — does the new front end reach a number we already have?

Reference arm reproduces the previously published **92.31 %** exactly. The new bank gives
**93.66 %**, so **+1.35 points** from the ladder, the orientation count and correct padding
alone. **The AND layer adds +0.06, then −0.06.**

> Re-verified 2026-07-29. The originally published figures were 93.71 / 93.78 / 93.71 and
> could not be reproduced within 0.06 by the current code, though the reference arm is
> bit-exact and two identical re-runs agree to four significant figures. No code change
> accounts for it; the original run's conditions were not captured in the repository.

### Phase 5b — is 3×3 pooling destroying a point property?

Partly yes and it does not matter. A₁+A₂ alone gain **+3.1 points** at a 6×6 grid, so signal
*was* being discarded — but the baseline does not improve either, so **the task is saturated
at 3×3**. That is a fact about EMNIST, not about conjunctions.

### Phase 5c — does the AND layer pay when data is scarce?

No. **Flat and indistinguishable from zero at every sample size from 200 to 112,800 images.**
The shuffled twin earns its place here: at 200 images, adding 54 *shuffled* columns costs
−10.10 points, so `+A` costing only −0.44 is itself evidence those columns are informative.

### Phase 6 — does augmentation help the pixels but not the features?

**The cleanest positive result in the project.** Rotation, scale and translation augmentation
buys a small CNN **+3.9 to +12.8 points** and buys the features **−1.2 to +0.4**. The designed
features already carry the invariances augmentation supplies — a much stronger claim than
"features win by 7.7 points".

### Phase 7 — the `F`/`f` probe, and how correlated A really is

`F`/`f` is **near-undecidable**: pixels 66.4 %, CNN 67.5 %, CNN with augmentation 69.9 %, our
features 69.9 %. Two unrelated routes converging says ceiling, not failure.

And the explanation for 5a–5c: **`R²(A ← orient) = 0.933`** median across 40 classes. Different
operators, near-collinear *on handwriting*.

### Phase 8 — a task where co-location is decisive

**Attempted three times, all three failed.** Everything solves it — including plain pooled
orientation energy — down to a 2 px gap against a 9–15 px stroke.

The failure is the finding: **any co-location difference also changes local ink density**, and
a classifier trained on thousands of examples finds that instead. It also exposed a conflation
in reading Phase 3 — cosine 0.97 between two feature vectors means they are *close*, not that
a trained classifier cannot separate them.

---

## 9 — changing the question

`SimpleStrokeTests/`. Phases 5–8 asked *can a classifier separate these*, and on synthetic
stimuli the answer is nearly always yes, which is why Phase 8 produced no information.

Phase 9 asks instead: **fit a linear readout and see how much of each property it recovers** —
because a property needing a hidden layer to extract is present in the representation but has
not been made *available* by it. Eight graded properties per image, five arms, three
extrapolation splits, everything scored against a trivial three-scalar baseline.

**What it found:**

* A linear readout on our features beats a two-hidden-layer MLP on 12,544 raw pixels on every
  structural property, and a CNN trained on the task on four of five — but that CNN arm did not
  converge (see Phase 11), so it bounds nothing.
* **The conjunction layer finally pays** — +0.16 to +0.21 over a shuffle control whose spread
  across five permutations is ≤ 0.002. On EMNIST the same layer was worth +0.01.
* **Perfect transfer to inverted polarity**, where both pixel arms fall below chance and the
  CNN reaches −2.1 on corner angle.
* **The best configuration is 31 globally pooled features with a small MLP** — 0.925 / 0.737 /
  0.938 / 0.954 against the CNN's 0.411 / 0.298 / 0.290 / 0.859.
* And it exposed a real bug: **the front end was not polarity invariant**, because zero-padding
  a non-zero background puts a full-contrast step round the image. Invisible on EMNIST, whose
  background *is* zero.

**Its limitation, which is why Phase 10 exists:** every image is a single stroke of one width
and one contrast on a flat field. Good for asking whether a property is explicit; poor for any
design choice that depends on image statistics.

---

## 10 — something that is not characters

`FashionMNIST/`. Silhouettes with texture rather than line drawings, and **published
baselines** to calibrate against instead of only our own arms.

**The front end works off line drawings: 89.70 % from 198 numbers**, above an MLP on 12,544
raw pixels (87.70 %, matching the published ≈ 88 % and so validating the harness) and 3.4
points below a CNN trained here under the same protocol (93.10 %). A₁+A₂ alone reach 86.51 %
from 54 numbers, on a task neither was designed for.

**And the conjunction layer pays on a dataset this project did not construct** — the strongest
form of the Phase 9 result. Against a 5-permutation shuffle control, A₁+A₂ are worth **+1.62 at
grid 3 and +2.85 at grid 1** (σ ≤ 0.27), where EMNIST gave +0.01. The control is what makes the
number: read naively against the base the layer looks worth +0.67, but 54 *shuffled* columns
cost −0.95 here (EMNIST measured −0.75 for the same block), so the naive reading understates
conjunction by a factor of 2.5 and would have confirmed the wrong prediction.

Of the three predictions recorded beforehand, two held and one was wrong. The features landed
in the predicted 88–91 % band. **Grid 3 beat grid 1 by 8 points** — the reverse of Phase 9,
which settles what that result meant: it was about *position randomisation* in the stroke
dataset, not about pooling in general. The prediction that the AND layer would add ≈ 0 failed;
its reasoning (silhouettes have few junctions) was sound, but A₁ is i2D-selective and fabric
texture is full of i2D structure that is not a junction.

The **ray block, by contrast, does not earn its columns here**: +0.54 over the base alone, and
−0.04 when added on top of A₁+A₂. Eighty-one features for nothing — expected on silhouettes with
no junctions to count, and concrete support for the standing complaint that the ray harmonics
are expensive relative to their return.

**Phase 10b asked why, and eliminated both available answers.** Phase 7 had explained the
EMNIST null with `R²(A ← orient) = 0.933`. Recomputed by one script across both datasets:
EMNIST 0.931, **Fashion-MNIST 0.943** — *more* collinear, with 160× the conjunction gain. So
that number does not predict whether conjunction helps, and Phase 7's inference is withdrawn
(the measurement reproduces exactly; R² is variance, and variance is not label information).

The second candidate came from Zetzsche & Barth's i2D criterion, and produced a genuine finding
about the front end: `Validate_i1D.jl` shows **A₁ does leak on exactly-i1D input at the coarsest
scale** — 4.6 × 10⁻² of its crossing response at ρ = 2, against 1.6 × 10⁻⁴ and 6.1 × 10⁻⁸ at
ρ = 3.74 and ρ = 7. The culprit is not the 90° partner but the **±45° pair straddling the line**,
predicted in closed form to within 2 %. It is not the explanation either: ρ = 7 has no leakage
and is still 0.856 predictable. What is left is the plain hypothesis — EMNIST strokes are i1D
almost everywhere, fabric is i2D almost everywhere — and that has not been measured.

**What it still will not settle:** black uniform background, one centred object, contrast
barely varying within a frame, 28×28 upsampled. Those are exactly the conditions under which
divisive normalisation would matter, so that question needs natural greyscale images —
BSDS boundary detection is the intended target.

---

## 11 — is the front end doing anything scale has not already done?

`ConVNextTest/`. The standing objection to Phase 9 was that its CNN lost because it had only
12,000 stroke images. A **frozen ImageNet ConvNeXt** answers that directly: 28.6 M / 88.6 M
parameters fitted to 1.28 M photographs, never trained on a stroke, scored by an *identical*
readout on *byte-identical* images.

**Four questions, four different answers — which is the finding.**

**i.i.d.: ConvNeXt wins everything**, 0.984 / 0.865 / 0.998 / 0.976 / 0.987 against our
0.930 / 0.730 / 0.998 / 0.944 / 0.958. Worse for the project's method, **explicitness goes the
wrong way too** — one *linear* map on frozen features gets 0.882 on `vangle` against our linear
arm's 0.549. "Our features make geometry explicit in a way learned representations do not" is
**retired**.

**Polarity flip: ours wins every structural property**, brokenness by 0.68 (0.776 vs 0.098), and
the linear ConvNeXt arms fall *below* the trivial baseline (−2.1, −3.7, −4.8). Transfer cost:
ours +0.016, every ConvNeXt arm −0.24 to −1.16, with Base degrading *more* than Tiny.

**But two other shifts disagree.** Under edge blur ours wins all five structural rows; under
stroke thickness ConvNeXt wins three of five and the mean. The reason is a new finding:
**thickness and fuzziness are confounded in our representation** — blur breaks our thickness
readout (−2.70) and thickening breaks our blur readout (−2.98), where ConvNeXt gets −0.56 and
−0.09. Both are read from the scale distribution of oriented energy, which a wider stroke and a
softer edge move the same way.

**Random-init control: the architecture buys nothing.** Same ConvNeXt with `weights=None` scores
−0.014 / 0.031 / 0.028 — indistinguishable from raw pixels, and `closedness` *below* the trivial
baseline. So ImageNet pretraining does all the work, and specifically it makes channels whose
*spatial average* is informative, which random channels are not.

**What survives:** one exact, constructed invariance (contrast polarity) that 1.28 M photographs
cannot buy and that more scale makes worse, plus better structural transfer under blur — against
a thickness/fuzziness confound ConvNeXt does not have.

---

## The shape of it

**0–4** establish that the operators do what they claim on synthetic ground truth. **5–7** ask
whether that helps on EMNIST; it does not, and Phase 7 explains why. **8** tries to construct a
task where it should help, and fails. **9** changes the question from separability to
explicitness and gets the first positive result. **10** asks whether any of it survives off
line drawings.

Two threads run throughout.

**Every phase carries a control that could have killed it** — the shuffled twin, the grid
control, the trivial baseline, the energy-matched stimulus. Three apparent results *were*
killed by them: total ink faking a ray-count ordering, unmatched orientation energy faking a
T-versus-X result, and 279 columns beating 135 partly on capacity.

**And the negative results are the substance.** 5b, 5c, 7 and 8 are all failures to find an
effect, and it was those failures — particularly Phase 8's — that forced the change of
question which made Phase 9 work.
