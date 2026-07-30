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
(that `theta` is the carrier everywhere, checked to 1.0° on a full 2π ray-direction recovery).

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
  structural property, and a properly trained CNN on four of five.
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
baselines** to calibrate against instead of only our own arms. See `FashionMNIST/README.md`
for the benchmark numbers — including which of the published figures are not credible — and
the three predictions recorded before the run.

**What it still will not settle:** black uniform background, one centred object, contrast
barely varying within a frame, 28×28 upsampled. Those are exactly the conditions under which
divisive normalisation would matter, so that question needs natural greyscale images —
BSDS boundary detection is the intended target.

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
