# The phases — what each one asked, and what it found

A map of the phased work. Phases **0–4** are design and validation, carried out in the
validator notebooks rather than as experiment scripts; **5 onward** are experiments, one
script each.

**What came before Phase 0.** The numbering starts with `P0-8_RationalGaborFeatures/`, so it does
not cover the earlier lines of work: the dense-Gabor keypoint detector (`P0.2_Dense_Gabors/`,
superseded), junction type by linear projection (`P0.3_New_Gabor_FPE/`), the Fourier grid
(`P0.4_ExptsWithGlobalFourier/`), Zernike moments (`P0.5_ExptsWithZernike/`), and the MLP re-evaluation
in `P0.6_TestFeaturesWithMLP/` that overturned the leave-one-out nearest-class-mean conclusions
used throughout the earlier work. Those are documented in the root `README.md` and the
`PROGRESS_*.md` journal. Two results from that period matter for reading what follows: the
leave-one-out protocol **understated features by ~24 points** and in one case manufactured a
qualitative conclusion a stronger classifier does not reproduce; and §7.11 of
`P0.6_TestFeaturesWithMLP/README_MLP_FPE_Experiment.md` is the few-shot comparison whose missing
control became Phase 6.

**Every section below opens with a *Source* line** naming the script that produced its numbers and
the figure, table or log they landed in. Paths are relative to the phase's own directory. Where a
phase has an accuracy- or R²-per-epoch curve, that is named first: a final number cannot distinguish
a converged run from a sample of a trajectory, and this project has been caught by that.

Directories: `P0-8_RationalGaborFeatures/` (the front end and the EMNIST phases),
`P9_P12_SimpleStrokeTests/` (Phase 9), `P10_FashionMNIST/` (Phase 10). Each has its own `README.md` for
how to run it and `RESULTS.md` for the tables; this file is the map across all of them.

---

## 0–4 — building the front end and proving it computes what it claims

*Source — `P0-8_RationalGaborFeatures/`: the `Validate_*.jl` gates, which run headless and print `ALL GATES PASSED` or name what failed. Captured output: `validate_i1d.log`, `validate_gpu.log`. No figures — these phases assert properties, not measurements.*

| | question | verdict | where |
|:--|:--|:--|:--|
| **0** | Can every parameter be fixed by *measurement* rather than convention? | Yes. Scale ladder from EMNIST's measured spectrum — ρ = 2.00 / 3.74 / 7.00, so λ = 56 / 30 / 16 px at 112. Stroke width measured at 12.67 px. Angular tuning from the along-contour extent. | `P0-8_RationalGaborFeatures/README.md` §4.1 |
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

*Source — `P0-8_RationalGaborFeatures/`: `Phase5a_EMNIST.jl` → accuracy curves `figures/phase5a_curves.png`; table in `RESULTS.md` §Phase 5a; the 2026-07-29 re-verification in `phase5a_recheck.log` and `phase5a_recheck2.log`.*

Reference arm reproduces the previously published **92.31 %** exactly. The new bank gives
**93.66 %**, so **+1.35 points** from the ladder, the orientation count and correct padding
alone. **The AND layer adds +0.06, then −0.06.**

> Re-verified 2026-07-29. The originally published figures were 93.71 / 93.78 / 93.71 and
> could not be reproduced within 0.06 by the current code, though the reference arm is
> bit-exact and two identical re-runs agree to four significant figures. No code change
> accounts for it; the original run's conditions were not captured in the repository.

### Phase 5b — is 3×3 pooling destroying a point property?

*Source: `Phase5b_FinerGrid.jl` → `figures/phase5b_curves.png`; table in `RESULTS.md` §Phase 5b.*

Partly yes and it does not matter. A₁+A₂ alone gain **+3.1 points** at a 6×6 grid, so signal
*was* being discarded — but the baseline does not improve either, so **the task is saturated
at 3×3**. That is a fact about EMNIST, not about conjunctions.

### Phase 5c — does the AND layer pay when data is scarce?

*Source: `Phase5c_FewShot.jl` → `figures/phase5c_fewshot.png`; table in `RESULTS.md` §Phase 5c.*

No. **Flat and indistinguishable from zero at every sample size from 200 to 112,800 images.**
The shuffled twin earns its place here: at 200 images, adding 54 *shuffled* columns costs
−10.10 points, so `+A` costing only −0.44 is itself evidence those columns are informative.

### Phase 6 — does augmentation help the pixels but not the features?

*Source: `Phase6_Augmentation.jl` → `figures/phase6_augmentation.png`; table in `RESULTS.md` §Phase 6.*

**The cleanest positive result in the project.** Rotation, scale and translation augmentation
buys a small CNN **+3.9 to +12.8 points** and buys the features **−1.2 to +0.4**. The designed
features already carry the invariances augmentation supplies — a much stronger claim than
"features win by 7.7 points".

### Phase 7 — the `F`/`f` probe, and how correlated A really is

*Source: `Phase7_FfProbe.jl` → tables in `RESULTS.md` §Phase 7 and its four subsections (the R²(A ← orient) numbers are in §"R²(A ← orient) puts a number on the correlation"). No figure.*

`F`/`f` is **near-undecidable**: pixels 66.4 %, CNN 67.5 %, CNN with augmentation 69.9 %, our
features 69.9 %. Two unrelated routes converging says ceiling, not failure.

And the explanation for 5a–5c: **`R²(A ← orient) = 0.933`** median across 40 classes. Different
operators, near-collinear *on handwriting*.

### Phase 8 — a task where co-location is decisive

*Source: `Phase8_JunctionBenchmark.jl` → the stimuli `figures/phase8_stimuli.png` and the gap sweep `figures/phase8_gapsweep.png`; table in `RESULTS.md` §Phase 8.*

**The stimuli.** Three designs, each built so that the *only* difference between classes is
whether two orientations meet **at a point** or merely fall in the same pooling window — which
is precisely what `A₁` claims to detect and what pooled orientation energy provably cannot:

| design | positive | negative | intended contrast |
|:--|:--|:--|:--|
| **crossing vs. near-miss** | two bars intersecting | the same two bars, one displaced so they pass without touching | co-location, orientation content identical |
| **gap sweep** | a corner with the arms joined | the same corner opened by a gap of 2–12 px | how close is "at a point" |
| **junction type** | L / T / X at matched total ink | one another | ray count at fixed energy |

Stroke width 9–15 px, and in the third design total ink was **held constant across classes**,
because Phase 3's junction ordering had already turned out to track ink rather than ray count.

**Attempted three times, all three failed.** Everything solves it — including plain pooled
orientation energy — down to a 2 px gap against a 9–15 px stroke.

The failure is the finding: **any co-location difference also changes local ink density**, and
a classifier trained on thousands of examples finds that instead. It also exposed a conflation
in reading Phase 3 — cosine 0.97 between two feature vectors means they are *close*, not that
a trained classifier cannot separate them.

---

## 9 — changing the question

*Source — `P9_P12_SimpleStrokeTests/`: `Phase9_Readouts.jl` runs the arms; `Plot_Phase9.jl` draws `phase9_arms.png` (per-property R²), `phase9_blocks.png` (block attribution), `phase9_learning.png`, `phase9_learning_detail.png` and `phase9_learning_splits.png` (R²-per-epoch curves), `phase9_grid.png` and `phase9_samples.png`. Predicted-vs-true scatters with binned medians: `Plot_Predictions.jl` / `Replot_Predictions.jl` → `figures_predictions/pred_*.png`, redrawable from `figures_predictions/predictions.jls` without retraining. Tables in `RESULTS.md` Part 1.*

`P9_P12_SimpleStrokeTests/`. Phases 5–8 asked *can a classifier separate these*, and on synthetic
stimuli the answer is nearly always yes, which is why Phase 8 produced no information.

Phase 9 asks instead: **fit a linear readout and see how much of each property it recovers** —
because a property needing a hidden layer to extract is present in the representation but has
not been made *available* by it. Eight graded properties per image, five arms, three
extrapolation splits, everything scored against a trivial three-scalar baseline.

### The stimuli

One item on a uniform grey field, 112×112, generated parametrically by `Contours.module.jl`, so
a test image is a **fresh draw** rather than a held-out slice — leakage is impossible by
construction. Each image carries **eight graded values, not a class label**:

| property | range | what varies |
|:--|:--|:--|
| `curvedness` | 0–1 | straight through arc, capped at 2π/3 of total turn |
| `brokenness` | 0–1 | gap size as a fraction of arclength |
| `closedness` | 0/1 | open arc or closed oval (aspect 0.62–1.0) |
| `vangle` | 32–170° | the kink angle, if there is a kink |
| `arms` | 2–4 | plain stroke, T-junction, or crossing |
| `thickness` | 3–12 px | stroke width, **log-uniform** |
| `fuzziness` | 0.8–20 px | edge ramp, **log-uniform** |
| `polarity` | ±1 | lighter or darker than the background |

Position and rotation are randomised; background level varies 0.40–0.60; arclength has a 68 px
floor. For T-junctions and crossings both strokes share the same curvature type.

### Why each of those choices

**Graded values rather than classes** — because Phase 8 showed a classifier will find *any* cue
that separates classes, and local ink density always co-varies. A graded target with a linear
readout measures how much of the property is *available*, which a class boundary cannot.

**A uniform grey background, not black** — the single most consequential choice. It is what
exposed the polarity-invariance bug: zero-padding a **non-zero** background puts a full-contrast
step round the image, and EMNIST could never have revealed it because its background *is* zero.
The project's target is general greyscale images, so an empty background would have tested the
wrong thing.

**Polarity varied**, so invariance is measurable rather than assumed — and so an extrapolation
split can train on light strokes and test on dark.

**Contrast set as a fraction of the available headroom**, `0.88·min(bg, 1−bg)`, so nothing
clips and mean luminance does not give the answer away. An earlier version had darker
backgrounds for dark strokes, which was visible in the contact sheet and would have made
`polarity` readable from the mean.

**Thickness and edge ramp log-uniform**, because both act multiplicatively on scale; uniform
sampling would have concentrated the range at the coarse end.

**Position randomised** — which is why grid 1 beats every larger pooling grid here, and why
Fashion-MNIST reverses that: a fixed grid is a liability only when parts do not land in
consistent places.

**Curvature capped at 2π/3 of turn**, so an "open arc" cannot close on itself and be
indistinguishable from an oval with two gaps.

**The vertex angle measured from the geometry** (`excess_turn`, the signed-turn difference
against the un-kinked base) rather than from the construction parameter — the parameter was
wrong by 28° on curved bases, putting 33 % of images in the wrong angle band.

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

*Source — `P10_FashionMNIST/`: `Phase10_FashionMNIST.jl` → tables in `RESULTS.md`, curves in `curves.jls`; `Plot_Phase10.jl` → `figures/phase10_curves.png` (one panel per arm, selected epoch marked) and `figures/phase10_curves_compare.png`. `Phase10b_Collinearity.jl` → `collinearity2.log`. Captured runs: `phase10_curves.log` (the 2026-08-04 re-run that added the curves; all fourteen feature-arm numbers reproduced exactly) and `phase10_full.log` (the original, which kept no curves). See `RESULTS.md` §Convergence — the feature arms are converged, the pixel calibration arm is not.*

`P10_FashionMNIST/`. Silhouettes with texture rather than line drawings, and **published
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

*Source — `P11_ConVNextTest/`: `ConvNextStimuli.jl` → `extract_convnext.py` → `ConvNextReadout.jl` (three steps, in that order). `Plot_Curves.jl` → `figures/curves_iid.png`, the R²-per-epoch curves that show these are ceilings rather than a sampled trajectory. `Deltas.jl` → `deltas.log`; `PartialOut.jl` → `partialout.log`; the stage probe → `stageprobe.log`; the random-init control → `run_random.log`. Tables in `RESULTS.md`.*

`P11_ConVNextTest/`. The standing objection to Phase 9 was that its CNN lost because it had only
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

## 12 — turning the front end's own dials

*Source — `P9_P12_SimpleStrokeTests/`: `Sweep_Capacity.jl` (the dials) and `Add_DeepHead.jl` (readout architectures, and the `fine`/`smax`/`both` arms) → tables in `RESULTS.md` Part 2, with the authoritative full-n numbers in §"Full-n confirmation" and the retracted reduced-n pass kept in the appendix below it. Plain-language account in `RESULTS.md` Part 2 §"Plain-language account". The λ=8 and spatial-max scatters are `figures_predictions/pred_*.png`, from `Replot_Predictions.jl`.*

`P9_P12_SimpleStrokeTests/`, on the Phase 9 stimuli. Everything before this tests the front end **as
built**; this asks whether it improves when given more to work with. The pooling grid had been
swept; the number of scales, the number of orientations, the order of the orientation harmonics
and the number of ray probe distances had not.

**Plain-language account: [`RESULTS.md` Part 2](P9_P12_SimpleStrokeTests/RESULTS.md).**
Detail in `RESULTS.md` Part 2, which absorbed the four files this was originally spread over:
§"Full-n confirmation" (the authoritative numbers), §"Cross-scale — proposed, published, and
retracted", and the reduced-n selection pass kept as an appendix because it inflated every gain
2–5× and reversed one sign.

**Adopted: `harmonics + offsets`, 54 features.** Higher orientation harmonics (C₆/C₈) buy
curvedness for five extra numbers; the crossed `d × λ` ray offsets — the off-diagonal that was
discussed years ago and never built — buy gaps, kinks and junctions. **Its value is robustness:**
gains over the 31-feature baseline grow monotonically with distribution shift, ~+0.02 i.i.d.,
+0.04–0.06 under blur, +0.08 under a thickness shift. Judged on the i.i.d. split alone it would
have been rejected.

**Rejected: more scales.** Predicted to fix the thickness/fuzziness confound; made it worse in
4/4 tests.

**Rejected as a standalone: more orientations.** Nothing i.i.d. for twice the compute — but the
broadest gainer under shift, which is only visible if you look there.

**The thickness/fuzziness confound remains unfixed.** A cross-scale feature appeared to move it
~+0.16 in both directions; that was an implementation artefact and is retracted. `A₃`, the
operator that already existed for this, makes it worse.

**Three method lessons**, each paid for:
* **Reduced-n selection inflates gains 2–5×** and reversed one sign. Use it to choose what to
  confirm, never for effect sizes.
* **Judge on the extrapolation splits.** The i.i.d. split alone picked the wrong configuration
  twice.
* **Check that the code being run is the code that was named.** The cross-scale operator already
  existed; a second version was written without noticing, and they were not identical.

---

## 13 — how much of the fit is memorised, and does the representation change that?

*Source — `P13_CurriculumEMNIST/`: features from `Extract_Ours.jl` and `extract_convnext_emnist.py`; frozen arms from `Curriculum.jl`, end-to-end arms from `train_convnext_scratch.py`. `Plot_Curriculum.jl` → `figures/curriculum_compare.png` (all four arms) and one panel per arm, `figures/curriculum_{ours,convnext,scratch_tiny,scratch_base}.png`. The lookup baseline is `KNN_Baseline.jl` → `knn.log`; the few-shot test is `fewshot_train.py` → `FewShot_Eval.jl` → `Plot_FewShot.jl` → `figures/fewshot.png`, with the per-class check in `PerClass_FewShot.jl`. Tables in `RESULTS.md`.*

`P13_CurriculumEMNIST/`. EMNIST's training set cut into **4 disjoint subsets of 28,200**, with training
switched to the next subset every 15 epochs and the optimiser never reset. The partition is i.i.d.,
so a learner that had extracted the task rather than memorised examples should not be able to tell
the data changed — every discontinuity measures how much of the fit was example-specific, taken
*during* training rather than inferred from a train/test gap. Four arms: our 381 frozen features,
frozen ImageNet ConvNeXt, and ConvNeXt-tiny and -base **trained end to end on EMNIST from random
initialisation**.

**Memorisation is large, uniform and useless.** Every arm holds a 10–14 point margin on whatever it
is currently looking at. The from-scratch arms interpolate their subset outright — 100.000 % on all
28,200 images in one block. And the from-scratch *control*, which never switches, drove training
accuracy from 94.4 % to a literal 100 % while its held-out score moved 83.10 → 85.06, most of that
the learning-rate schedule. **Perfecting the memorisation bought almost nothing.**

**For frozen representations, forgetting is a readout property.** Subset 1's advantage over held-out
decays 10.95 → 2.43 → 1.31 → 0.84 (ours) and 10.34 → 2.48 → 1.31 → 0.51 (frozen ConvNeXt) — agreeing
to within 0.1 points after the first block, despite one being 381 hand-designed numbers and the
other 1024 learned from 1.28 M photographs. The front end is not the lever for it.

**Training the representation is what changes that.** The frozen arms cross the switches invisibly
(−1.3 and −0.2 sd). The from-scratch arms **jump up** at every switch, +3.3 and +4.9 sd, and retain
about double the advantage on old data one block after leaving it. The total benefit of fresh data
is the same for everyone (~1.3–2.0 points); what differs is whether it arrives as a step or as a
slow accumulation.

**And the from-scratch advantage is not memorisation.** Hold out 10 of the 47 classes entirely,
train on the other 37, then classify the withheld ones few-shot from the penultimate features. At
**1-shot: 74.5 % against 60.2 (frozen ImageNet), 57.9 (ours) and 48.0 (raw pixels)** — on characters
it has never seen, where stored base-class images can contribute nothing directly. The gain is
spread over 9 of 10 classes, so it is not leakage from near-duplicate glyphs.

**kNN calibrates all of it.** Pure lookup against 28,200 stored images reaches 80.89 % on our
features and 78.78 % on frozen ConvNeXt's, against trained readouts at 85.18 and 84.90 — so ~95 % of
EMNIST accuracy needs no learning at all, and 73.7 points of it no representation at all. Our 381
designed features are a **better** nearest-neighbour space than 1024 ImageNet features.

**Rank at epoch 60:** ConvNeXt-base scratch 88.06, tiny scratch 87.01, ours 85.97, frozen ConvNeXt
85.70. Our features **lead at epoch 15** and are passed only once the networks have seen more than
one subset. Training the representation on the task is worth ~2 points over hand-designing it.

Caveats on record: the from-scratch arms use a different optimiser and schedule, so cross-arm
*levels* are confounded (within-arm switching-vs-control contrasts are bit-exact and clean); no
augmentation anywhere, so this is not ConvNeXt at its best; λ = 8 px sits at EMNIST's original
Nyquist limit and largely measures interpolation here.

---

## The shape of it

*This section is self-contained: every term it uses is defined here, so it can be read without
the rest of the document.*

**0–4 — the operators compute what they claim, on stimuli whose answer is known in advance.**
Not "work well" — specific properties, asserted and then gated. The bank is in exact quadrature
with its DC term zeroed, so inverting the image leaves every feature bit-identical (measured
change: exactly 0). And A₁, the conjunction operator, separates a **corner** from **two disjoint
strokes carrying the same orientation content** — the distinction that matters, because the two
have identical orientation histograms and differ only in whether the orientations *meet* at a
point. These are proofs on synthetic figures, not accuracy on a dataset.

**5–7 — none of it helps on EMNIST, and Phase 7 says why.** "Helps" has a precise meaning here:
added to the 144 orientation-and-lowpass columns, do the 54 conjunction columns raise 40-way
character classification? They add **+0.06 points, then −0.06** — nothing. Not because they
compute nothing: the 54 conjunction columns **alone** reach 88.45 %, against 93.59 % for the 135
orientation columns alone. The reason is measured in Phase 7: fit each conjunction column on the
orientation columns by least squares, and **93 % of it is linearly recoverable** (median R² 0.933;
42 of 54 columns above 0.9, all 54 above 0.75). The right word is **correlated, not redundant** —
they measure something real, and almost all of it was already present in a cheaper form.

**8 — a task built specifically to need co-location, and its own control killed it.** Junction
figures were classified with and without a small gap at the meeting point, so that only "do the
strokes meet" varies. The conjunction layer won — but so did total ink, because **any
co-location difference necessarily changes local ink density**: a gap removes ink next to the
junction, and with 37 px pooling cells that is a measurable energy change a classifier will find.
There may be no stimulus pair that differs in *meeting* without differing in density. Verdict at
the time: **no task existed on which the conjunction layer beat plain orientation statistics** —
not EMNIST, not the F/f probe, not a benchmark built to require it.

**9 — changing the question, and the first positive result.** Every phase up to here asked *can a
classifier separate these classes*. Phase 9 asks instead: *how much of a **graded** property is
**linearly** available in the representation* — fit a readout to predict curvedness, brokenness,
V-angle, arm count, stroke thickness and edge fuzziness on synthetic contours, and report R²
(1.0 = perfect, 0 = no better than always guessing the average, negative = worse than that).
Under that question the conjunction and ray blocks finally earn their columns.

**10 — off line drawings for the first time.** Fashion-MNIST: silhouettes with weave, ribbing and
sole tread rather than strokes on an empty background, and with published baselines so the result
is calibrated against something outside this project. The conjunction layer **pays on someone
else's data** — +0.67 points at a 3×3 grid, against a control of the same columns shuffled, which
*costs* 0.95. And a 3×3 grid beats a single global pool by 8 points, the reverse of Phase 9,
because objects here are centred while Phase 9's strokes were randomly placed.

**11 — the comparison that retired a central claim.** A **frozen** ImageNet ConvNeXt — 88.6 M
parameters, trained on 1.28 M photographs, never shown a stroke, weights never updated — read out
by an identical head on byte-identical images. It beat our 31 features on every property, and beat
them *more linearly*, which retires "our designed features make geometry explicit in a way learned
representations do not". What survived is narrower and real: an **exact polarity invariance** that
1.28 M photographs cannot buy, and better transfer when edges are blurred.

**12 — turning the front end's own dials.** Everything before this tested the front end *as built*;
this asks whether it improves when given more — more scales, more orientations, higher orientation
harmonics, more ray probe distances. Two changes closed most of the ConvNeXt gap: a **fine scale at
λ = 8 px** (the finest filter had been wider than the strokes it was measuring, so it could not
resolve a stroke's own edge) and a **spatial maximum** alongside the existing spatial averages (an
average of an energy map is *signal × ink coverage*, so every pooled feature silently multiplied
"how strong is the structure" by "how much ink is in the frame"). Together: 31 features → 53, and
the deficit on thickness and fuzziness from 0.24 to 0.04.

**13 — is the accuracy memorisation?** EMNIST's training set cut into four disjoint subsets, with
training switched to the next one every 15 epochs and the optimiser never reset. The subsets are a
random partition, so a learner that had extracted the task rather than memorised examples should
not detect the change. Held-out accuracy indeed walks through the switches untouched — while
accuracy on the subset being trained on **collapses ~7 points at each one**, which measures
memorisation directly rather than inferring it from a train/test gap. A second test settles the
harder question: hold out 10 of the 47 character classes entirely, train on the other 37, and
classify the withheld ones from one example each. A network trained end to end on EMNIST scores
**74.5 %** against 60.2 % (frozen ImageNet), 57.9 % (ours) and 48.0 % (raw pixels) — on characters
it has never seen, where remembered training images cannot help. So its advantage is extracted
structure, not stored examples.

---

Three threads run throughout.

**Every phase carries a control that could have killed it.** The four used most often:

* **The shuffled twin.** Take the block under test and permute it *across images*, so the number
  of columns and each column's distribution survive and only the correspondence with the image is
  destroyed. Whatever the real block scores above its shuffled twin is what it contributes as
  **information**; the rest was capacity. Extra columns are not free in either direction — on
  EMNIST at a 3×3 grid, real conjunction columns cost +0.01 while the same count shuffled costs
  −0.81.
* **The grid control.** Whenever a block is re-pooled at a finer resolution, the baseline is
  re-pooled at the same resolutions too, so a gain that comes from finer pooling is not credited
  to the operator. This is what settled Phase 5b: the baseline gives 93.71 → 93.73 → 93.10 across
  3×3, 6×6 and 11×11, so **the task is saturated at 3×3** — a fact about EMNIST, not about
  conjunctions.
* **The trivial baseline.** Predict the training-set average for every image. It scores R² = 0 by
  construction, so any arm scoring below it is doing worse than a constant — which several
  otherwise respectable arms do under changed conditions.
* **The energy-matched stimulus.** Figures drawn so total oriented energy is held constant, so a
  difference between a T and an X cannot be a difference in how much ink is present.

**Three apparent results were killed by exactly these controls.**

1. **A₁ appeared to order junctions by ray count** — straight 6.3e4 < L 9.5e4 < T 1.15e5 < X 1.58e5
   — and this went into the README and a commit message. It was tracking **total ink**
   (980 / 1052 / 1359 / 1708 px). With ink held constant a T and an X give 1.15e5 against 1.16e5,
   and normalised by energy the ordering *inverts*: an L-corner (0.0415) outranks a T (0.0391).
   Theory says it must be so — A₁ reads the orientation profile, which is **π-periodic**, while ray
   count is a **2π** property, and a T and an X have identical orientation content.
2. **A T-versus-X separation** that came from unmatched orientation energy rather than from
   junction type.
3. **Blocks appearing to win on merit when part of the win was capacity** — visible only once the
   shuffled twin was run alongside.

**And the negative results are the substance.** Phases 5b, 5c, 7 and 8 are all failures to find an
effect, and it was those failures — Phase 8's above all — that forced the change of question which
made Phase 9 work. Phase 11 is a fourth: it retired a claim this project had been making for
months. The pattern is consistent enough to be worth stating plainly — **the results that moved
this work forward were mostly the ones that went the wrong way.**

**Several published numbers here have been retracted**, and the retractions are kept in place
rather than deleted, because the reason each was wrong is reusable. The largest: a cross-scale
operator reported as moving the thickness/fuzziness confound by ~+0.16 in both directions turned
out to be an inline reimplementation missing a square root that the real pooling path applies —
the true effect is +0.019 and −0.270. And a reduced-sample selection pass **inflated every gain
2–5× and reversed one sign**, which is why selection is now done on reduced samples but effect
sizes never are.
