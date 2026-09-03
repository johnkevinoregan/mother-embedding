# P0–8 Rational Gabor Features — how the hand-designed visual front end was built, tested, corrected, and narrowed down

## Executive summary

This subproject develops the first mature version of the `mother-embedding` visual front end. The aim is to replace raw pixels with a **small, fixed, interpretable set of visual measurements** before any learned classifier is applied. The guiding idea is biologically inspired: early vision should first measure local orientation and contour structure, and only then should a trainable system learn how to use those measurements for a task.

The work starts from a conventional idea — a bank of oriented Gabor-like filters — but makes two important changes. First, the filter bank itself is made **data-rational**: its scales are chosen from the measured spatial-frequency spectrum of the images rather than from arbitrary constants. Second, the project introduces explicit nonlinear operators intended to capture local two-dimensional structure that ordinary pooled orientation measurements lose: **A₁**, which detects perpendicular orientations occurring at the same location, and **A₂**, an end-stopping measure that responds to line terminations.

The central architectural principle is:

`oriented filtering → local nonlinear conjunction → spatial pooling`

rather than pooling orientation measurements first and trying to reconstruct junction information afterwards.

The synthetic validation experiments show that this principle is real. A₁ strongly distinguishes two perpendicular strokes that actually meet from the same strokes pulled apart, while ordinary pooled orientation statistics barely change. A₂ strongly distinguishes a line ending from the interior of a line. Thus the new operators genuinely compute information that the conventional pooled orientation representation can discard.

However, when the system is tested on EMNIST handwritten characters, the surprising result is that **the conjunction operators add essentially no classification accuracy**. The large improvement over the previous front end — about **+1.4 percentage points** — comes instead from the redesigned log-Gabor bank, better scale placement, more appropriate orientation sampling, and correct padding. The conjunction signals are informative by themselves, but on handwritten characters they are so strongly correlated with the ordinary orientation measurements that they provide almost no additional usable information.

A series of follow-up experiments tests alternative explanations for this null result. Finer spatial pooling does recover much stronger conjunction signals, but still does not improve classification. The conjunctions do not become useful when training data are scarce. Real-image blur does not explain the failure. A direct `F`-versus-`f` test does not rescue them. A ray-based operator designed to distinguish T-junctions from X-crossings also fails to beat the orientation baseline on that pair. Finally, synthetic Phase 8 tasks intended to make co-location decisive are themselves solved almost perfectly by the orientation baseline because local ink layout gives the answer away.

The strongest positive result of P0–8 comes from a different experiment: **data augmentation helps a small CNN by roughly 4–13 percentage points but helps the designed features essentially not at all**. This is direct evidence that the front end already contains much of the translation/rotation/scale tolerance that the CNN otherwise has to acquire from augmented examples.

The endpoint of P0–8 is therefore not “the conjunction layer solves EMNIST.” It is:

1. a much better, validated, scale-rational oriented-energy front end;
2. evidence that this hand-designed representation carries useful invariances before learning;
3. two validated i2D operators, A₁ and A₂, whose distinctive information is real but adds almost nothing on EMNIST;
4. several rejected interpretations and parameter choices that were exposed by targeted controls;
5. a clearer separation between what should be **built into the front end** and what should be left for later learning.

---

## 1. What problem is this front end trying to solve?

A conventional CNN receives pixels and learns its early filters, pooling rules, invariances, and higher-order combinations from data. P0–8 asks whether some of those early visual computations can instead be specified explicitly.

The motivation is both computational and biological.

Computationally, a fixed front end may improve **sample efficiency**. A CNN trained on a few examples has to discover that a character remains the same character after a small shift, a slight rotation, or a small change of scale. A hand-designed front end can build some of those tolerances in from the beginning.

Biologically, early visual cortex is often described in terms of oriented simple and complex cells, with further nonlinear mechanisms such as end-stopping. P0–8 tries to construct an explicit computational analogue rather than assuming that all such structure should emerge from task-specific training.

The particular representational problem that motivated the subproject is **co-location**.

Suppose one image contains a horizontal and a vertical stroke that meet, while another contains exactly the same two strokes but with a small gap between them. Both images contain almost the same amount of horizontal and vertical structure. If orientation energy is first averaged over a spatial region, the two cases can look nearly identical.

Yet for shape, the distinction “the strokes meet” versus “the strokes merely occur nearby” can matter greatly.

This motivates a key distinction:

- ordinary orientation summaries tell us **what orientations occur in a region**;
- the desired nonlinear representation should also tell us **whether those orientations occur at the same place**.

In the language of intrinsic dimensionality, straight edges and stripes are largely **i1D** structures, whereas corners, line ends, and junctions are **i2D** structures. The project deliberately formulates the central question more narrowly as co-location because this yields a precise mathematical and experimental test.

The repository’s standalone explanation is in [RESULTSexpanded.md](https://github.com/johnkevinoregan/mother-embedding/blob/main/P0-8_RationalGaborFeatures/RESULTSexpanded.md).

---

## 2. The task used to test the front end

Most of P0–8 evaluates the representation on **EMNIST Balanced**, a dataset of 28 × 28 greyscale handwritten characters.

EMNIST Balanced contains 47 labelled classes and an official split of 112,800 training images and 18,800 test images. Earlier work in the project had established that several nominally distinct classes are visually indistinguishable from the image alone — for example `0/O` and `1/I/L`. P0–8 therefore merges five homoglyph groups, leaving **40 effective classes**.

The purpose is not to maximise an EMNIST leaderboard score. EMNIST is being used as a controlled test bed for asking whether particular fixed visual measurements are useful.

The learned classifier after the front end is intentionally modest but not deliberately weak: for the main Phase 5 tests it is a one-hidden-layer MLP,

`features → 256 ReLU units → 40 classes`

trained with Adam.

This choice is important because earlier work in the project had used a nearest-class-mean classifier and discovered that it could give badly misleading conclusions about representations. A weak classifier may fail to exploit a good representation and therefore make two feature families look complementary when they are not. P0–8 therefore adopts the methodological rule that **representations must be compared using a classifier strong enough to exploit each one**.

---

## 3. The starting point that was rejected

Before P0–8, the project used an 88-dimensional front end based on **windowed Fourier features**. It achieved about **92.30%** on the merged-class EMNIST task.

P0–8 does not simply add more features to this representation. It re-examines the assumptions underneath it.

The most serious problem was scale choice. The old bank used fixed wavelength values inherited from previous experimentation. Once the spatial-frequency content of EMNIST was measured directly, it became clear that part of that old bank was badly placed: one channel lay beyond the effective Nyquist limit of the source images and another lay in a frequency region containing only about one percent of the useful signal.

In other words, a substantial part of the old front end was spending capacity analysing frequencies that were mostly interpolation artefact or nearly absent from the data.

The old front end therefore became the **reference to beat**, not the architecture to preserve.

---

## 4. Phase 0: measure the image statistics before choosing filters

The redesign begins by refusing to choose spatial scales by convention.

Two quantities were measured on 20,000 EMNIST training images.

### 4.1 Stroke width

The median stroke width was estimated at about **3.17 pixels in the native 28 × 28 image**, or **12.67 pixels after 4× upsampling to 112 × 112**.

This gives a physically meaningful scale for asking whether a filter is too small, too large, or appropriately localised relative to the structures actually present.

### 4.2 Radial power spectrum

The average two-dimensional Fourier power spectrum of the images was measured and collapsed into radial frequency bands.

The result showed that most useful energy lies at relatively low spatial frequencies. The adopted usable band was approximately

\[
\rho \in [2,7]
\]

cycles per image width.

Three log-spaced centre frequencies were then placed at approximately

\[
\rho = 2.00,\;3.74,\;7.00,
\]

corresponding on the 112 × 112 working grid to wavelengths of about

\[
56,\;30,\;16\ \text{pixels}.
\]

This is one of the main ideas that survives P0–8: **the scale ladder is derived from the measured spectrum of the input domain rather than hard-coded**.

The implementation is in [GaborStack.module.jl](https://github.com/johnkevinoregan/mother-embedding/blob/main/P0-8_RationalGaborFeatures/GaborStack.module.jl), particularly `radial_spectrum` and `scale_ladder`.

---

## 5. Phases 1–2: the retained oriented-energy bank

The basic front end that emerges is a dense **log-Gabor oriented-energy stack**.

The 28 × 28 input is first bilinearly upsampled to 112 × 112. Upsampling does not add information; it gives a finer computational grid on which filters and spatial offsets can be placed. The image is then padded before FFT filtering to avoid circular wraparound.

The final Phase 0–8 bank uses three scales and increasing numbers of orientation channels:

- coarse scale: 8 orientations;
- middle scale: 12 orientations;
- fine scale: 16 orientations.

### Why log-Gabor rather than ordinary spatial Gabor kernels?

Several choices were deliberately retained.

**Frequency-domain construction.** The filters are constructed directly as frequency-domain bumps and convolution is done by FFT. This avoids arbitrary truncation of a spatial kernel and makes the bank naturally adaptable to image size.

**Analytic, one-sided filters.** Real and imaginary responses form an exact quadrature pair. Squaring and summing them gives oriented energy:

\[
E = \mathrm{Re}(r)^2 + \mathrm{Im}(r)^2.
\]

This makes the representation exactly insensitive to contrast polarity: a black stroke on white and the corresponding white stroke on black produce the same energy.

**Log-Gabor radial tuning.** A log-Gabor has exactly zero DC response and allows bandwidth and wavelength to be controlled more independently than a conventional Gaussian in linear frequency. That permits relatively coarse wavelength tuning without forcing the spatial receptive field to become excessively broad.

**Correct padding.** FFT convolution is circular. P0–8 explicitly calculates enough padding from the spatial extent of the filters instead of assuming that a fixed border is sufficient.

These choices sound technical, but they mattered empirically. Validation uncovered two serious implementation errors: a mismatch between frequency units defined relative to the padded field versus the original image, and a normalisation scheme that caused feature amplitudes to change merely when the padding size changed. Both were fixed before downstream accuracy was trusted.

That validation philosophy — test the operator on synthetic stimuli with known answers before testing a classifier — is one of the most important methodological products of this subproject.

---

## 6. What the ordinary orientation block represents

After oriented filtering, one could simply pool each orientation channel over coarse spatial regions and give those pooled values to a classifier.

P0–8 uses a somewhat more compact version. Within each spatial pooling cell and scale it represents the orientation profile by low-order circular harmonics. In the current implementation the emitted quantities include total energy and low-order orientation harmonics such as `Re C₂`, `Im C₂`, `|C₂|`, and `|C₄|`.

With a 3 × 3 spatial grid, the orientation block contributes **135 numbers**, and a separate low-pass/DC block contributes **9 numbers**, giving a **144-dimensional orientation + low-pass baseline**.

The crucial limitation is that these harmonics are formed **after spatial pooling**. They describe which orientations occur inside the cell, but not necessarily whether different orientations occurred at the same pixel.

The implementation of this baseline is in [Pooling.module.jl](https://github.com/johnkevinoregan/mother-embedding/blob/main/P0-8_RationalGaborFeatures/Pooling.module.jl).

---

## 7. The central architectural idea: multiply before pooling

The distinctive hypothesis in P0–8 is that the order of operations matters.

For two local energy maps \(e_1(x)\) and \(e_2(x)\), compare:

\[
\left(\sum_x w(x)e_1(x)\right)
\left(\sum_x w(x)e_2(x)\right)
\]

with

\[
\sum_x w(x)e_1(x)e_2(x).
\]

The first says that both orientations exist somewhere in the region. The second is large only when they tend to occur at the **same locations**.

Their difference is related to the spatial covariance between the two maps.

Therefore:

**pool → combine** loses information about precise co-location;

**combine locally → pool** can preserve it while still gaining positional tolerance.

This leads to the design rule:

`select → multiply → pool`

or, in biological language,

`orientation-selective response → local nonlinear conjunction → tolerant spatial summary`.

This is the conceptual reason for introducing A₁ and A₂.

---

## 8. Phase 3: A₁ — local conjunction of perpendicular orientations

A₁ is designed to respond where two approximately perpendicular orientations coexist locally.

At each pixel the oriented-energy profile is normalised across orientation channels and correlated with itself at a 90° shift. Conceptually:

\[
A_1(x) =
C_0(x)\sum_k p_k(x)p_{k+90^\circ}(x),
\]

where \(C_0\) is total local energy.

A single straight contour is dominated by one orientation and produces relatively little A₁. A crossing, corner, or junction containing strong perpendicular components at the same location produces more.

### The decisive synthetic control

The most important validation uses two perpendicular bars whose relative separation is varied.

At zero gap the bars cross. As the gap grows, the bars contain essentially the same orientation content but cease to be co-located.

On this sweep:

- the dense A₁ response falls from its maximum to roughly **6%**;
- the conventional pooled orientation representation remains almost unchanged, around **97%** similar.

This is strong evidence that A₁ really is measuring something that the pooled orientation statistic can discard.

However, one later correction is essential: **A₁ is not a ray counter**. A T-junction and an X-crossing can have the same set of undirected orientations, so an operator built only from a \(\pi\)-periodic orientation profile cannot in principle distinguish “three rays” from “four rays.” Early apparent ordering of straight/L/T/X stimuli was therefore over-interpreted. Later remeasurement suggests total A₁ is related more closely to the number of local i2D events than to ray count itself.

A further later validation found a small i1D leakage at the coarsest scale. The current code subtracts an analytically predicted i1D floor by default, reducing that leakage by roughly 50×. The published P0–8 EMNIST tables, however, were computed with the earlier unfloored A₁ and were not re-run after this correction.

The implementation is in [AndLayer.module.jl](https://github.com/johnkevinoregan/mother-embedding/blob/main/P0-8_RationalGaborFeatures/AndLayer.module.jl).

---

## 9. Phase 3: A₂ — end-stopping

A₂ targets another form of local i2D structure: **line termination**.

At each pixel, the locally dominant orientation is found. The algorithm then probes the same orientation channel at two locations displaced along the direction of the stroke. If the stroke continues equally on both sides, the responses are similar. At an endpoint, one side has continuation energy and the other does not.

Schematically:

\[
A_2(x)
=
E(x)
\frac{|E_+ - E_-|}
{E_+ + E_- + \kappa E(x)}.
\]

On synthetic bars, the final implementation gives an end-versus-interior contrast of about **10.4×** at the finest scale.

Several variants were explicitly rejected during development:

- **maximum over all orientations** was rejected in favour of probing the locally dominant orientation; this improved end/interior contrast from roughly 2.5× to 10.4×;
- an **absolute epsilon** in the denominator was rejected because it allowed noise in weak channels to produce spurious asymmetry and made A₂ degenerate toward ordinary energy;
- a standard angular-tuning value inherited from the literature was rejected because it produced filters longer than the strokes being analysed;
- early guesses for the probe displacement were rejected after sweeps showed they were too large or too small.

A later scale-free analysis also showed that the optimal probe distance should not really be anchored to the filter envelope. It tends to track the **stroke width being measured**. Thus even the apparently successful EMNIST value turned out to be partly accidental.

This is an important example of the subproject’s general pattern: a useful operator survives, but its original parameter justification does not.

---

## 10. Phase 4: spatial pooling — retained, but only after local detection

After the dense feature maps have been computed, they are pooled using overlapping Gaussian-weighted windows on a coarse spatial grid.

The default grid is **3 × 3**.

This intentionally throws away exact position. A junction that moves a few pixels between two handwritten versions of the same letter should not become a completely different feature vector.

A 4-pixel shift leaves the 3 × 3 representation with cosine similarity about **0.998**. Finer grids preserve more spatial detail but slightly reduce this tolerance.

The critical conclusion is therefore not “pooling is bad.” It is:

**detect the local property first, then pool it.**

The default P0–8 vector was:

| block | dimensions |
| --- | ---: |
| pooled orientation statistics | 135 |
| low-pass | 9 |
| A₁ | 27 |
| A₂ | 27 |
| **total** | **198** |

The project also built in a useful dimensionality control: feature blocks can be **shuffled across samples**. A shuffled block has exactly the same number of columns and the same marginal distributions, but its values no longer correspond to the correct images. This separates “the new information helped” from “adding more numerical dimensions changed optimisation.”

---

## 11. Phase 5a: the first decisive EMNIST result

The old 88-feature system was first re-run in the new experimental harness and reproduced its previous result almost exactly: about **92.3%**. This was an important sanity check.

The new representation was then tested incrementally.

| representation | dimensions | final accuracy |
| --- | ---: | ---: |
| previous 88-feature system | 88 | 92.31% |
| new orientation + low-pass | 144 | 93.71% |
| + A₁ | 171 | 93.78% |
| + A₁ + A₂ | 198 | 93.71% |

The main positive result is the jump from about **92.3% to 93.7%**.

But the source of that gain is not the conjunction layer. It is already present in the 144-dimensional orientation + low-pass representation.

The interpretation is that the **new bank itself** is better: scales are placed where the images actually contain energy, orientation sampling is more appropriate, and boundary handling is correct.

By contrast, adding A₁ changes accuracy by only about +0.07 point, and adding A₂ removes that tiny difference. At the scale of this test, the net contribution is effectively zero.

This is visible in the repository’s [Phase 5a learning-curves figure](https://github.com/johnkevinoregan/mother-embedding/blob/main/P0-8_RationalGaborFeatures/figures/phase5a_curves.png).

Importantly, A₁ and A₂ are not useless measurements. Trained alone, the 54 conjunction features achieve about **88.45%** accuracy. They contain a large amount of character information. The negative result is narrower: **given the orientation baseline, they add almost nothing further on EMNIST**.

---

## 12. Phase 5b: was coarse pooling destroying the conjunction signal?

One obvious explanation for the null result was that A₁ is sharply local but its values were being averaged over very large 3 × 3 cells.

This was tested by making the conjunction grid progressively finer while keeping the orientation baseline controlled.

The conjunction representation by itself does improve strongly when pooling is refined: moving from 3 × 3 to 6 × 6 raises its classification accuracy by about three points.

But when the same features are added to the orientation baseline, overall accuracy does not improve. At 11 × 11 it actually falls.

Thus two facts are simultaneously true:

1. coarse pooling **does weaken** the conjunction signal;
2. recovering that signal **still does not make it useful beyond the baseline** on EMNIST.

This matters because it rules out a simple implementation excuse. The null result is not merely because A₁ was averaged away.

See the [Phase 5b pooling-grid figure](https://github.com/johnkevinoregan/mother-embedding/blob/main/P0-8_RationalGaborFeatures/figures/phase5b_curves.png).

---

## 13. Phase 5c: perhaps conjunctions help only with very little data?

A central motivation for a designed front end is sample efficiency. Perhaps a learned classifier can infer the useful conjunction-like information from ordinary orientation statistics when given enough examples, while an explicit A₁/A₂ signal would help in the few-shot regime.

This was tested with only 5, 10, 20, 50, 100, or 400 training examples per class, using five seeds and a fixed number of optimisation steps.

The result is essentially flat. The difference between the baseline and baseline + conjunctions stays near zero at every sample size.

For example:

- 5 examples/class: conjunctions are about **0.44 point worse**;
- 10 examples/class: about **0.22 point better**;
- 20 examples/class: about **0.20 point better**;
- 100 examples/class: essentially identical;
- 400 examples/class: about **0.12 point worse**.

All of these differences are within their uncertainty.

So the hypothesis “A₁/A₂ become particularly useful when data are scarce” was rejected.

The shuffled-feature controls are informative here: at 5 examples per class, adding an equally large shuffled block costs roughly ten points, whereas adding the real conjunction block costs almost nothing. The A features are therefore meaningful; they are simply not supplying a new discriminative advantage beyond what the orientation representation already carries.

See the [Phase 5c few-shot figure](https://github.com/johnkevinoregan/mother-embedding/blob/main/P0-8_RationalGaborFeatures/figures/phase5c_fewshot.png).

---

## 14. Phase 6: the strongest positive result — designed invariance versus augmentation

This experiment asks a different and more revealing question.

Earlier results had shown a large advantage of the designed features over a small CNN when training data were scarce. But that comparison was unfair in an important sense: the CNN was asked to learn translation, rotation, and scale tolerance from raw examples, whereas the fixed front end had those tolerances partly built into its filtering and pooling.

Data augmentation provides exactly the missing control.

Training images were randomly transformed by approximately:

- ±10° rotation;
- ±10% scale;
- ±2 pixels translation.

Ten augmented versions were generated per original image.

The key prediction was:

**if the front end already encodes these invariances, augmentation should help the CNN strongly but should help the fixed features very little.**

That is exactly what happens.

At small sample sizes, augmentation improves the CNN by roughly **9–13 percentage points**, whereas the designed-feature system changes by approximately zero. At 50 examples per class, the CNN still gains nearly four points while the features slightly decline under the fixed update budget.

This is stronger evidence than simply saying that the front end “beats the CNN.” It identifies a mechanism: the hand-designed representation already provides much of what augmentation is trying to teach the CNN.

The practical advantage is less dramatic, because augmentation largely closes the CNN gap. By 50 examples per class, the augmented CNN and the fixed-feature system are approximately level.

So the claim retained from Phase 6 is not that designed features are universally superior. It is that **they start with useful invariances already encoded**, whereas a raw-pixel CNN needs either data or augmentation to acquire comparable tolerance.

See the [Phase 6 augmentation figure](https://github.com/johnkevinoregan/mother-embedding/blob/main/P0-8_RationalGaborFeatures/figures/phase6_augmentation.png).

---

## 15. Blur was tested and rejected as the explanation

Synthetic validation stimuli have crisp binary edges, whereas EMNIST characters are anti-aliased and then enlarged by interpolation. It was therefore plausible that the conjunction operators worked on synthetic bars only because the edges were unrealistically sharp.

The repository measured the actual amount of blur in EMNIST and then blurred the synthetic stimuli to comparable levels.

The result rejects the hypothesis.

At realistic EMNIST blur, pooled A₁ contrast is essentially unchanged. The much larger loss occurs earlier when dense A₁ maps are pooled over the 3 × 3 grid: the crossing/separated contrast drops from roughly **17.6× to 4.9×** even before realistic blur is added.

A₂ is somewhat more blur-sensitive than A₁, as expected for a sharp termination signal, but blur still does not explain why the conjunction block adds no classification accuracy.

Thus “synthetic edges are too clean” was tried and rejected.

---

## 16. Phase 7: direct tests of the remaining explanation

### 16.1 The `F` versus `f` test

The largest residual confusion in EMNIST involved `F` and `f`. This seemed like an ideal case for junction-sensitive features: an uppercase F contains T-like three-ray structure, whereas a typical lowercase f can contain an X-like four-ray crossing.

Because this pair constitutes only a small fraction of all test images, a benefit might be diluted in the 40-class accuracy. Phase 7 therefore trains a classifier **only on F versus f**.

Several feature families were compared:

| features | F/f accuracy |
| --- | ---: |
| orientation + low-pass | 69.88% |
| A₁ + A₂ | 67.54% |
| ray harmonics | 67.50% |
| orientation + A₁ + A₂ | 69.62% |
| orientation + rays | 68.88% |
| everything | 69.29% |

The conjunctions do not win even on the pair they were expected to help.

### 16.2 Why introduce ray harmonics?

The failure of A₁ on `F/f` led to an important conceptual correction.

A₁ uses an orientation profile defined modulo 180°. It can detect that perpendicular orientations coexist, but it cannot distinguish the number of directed arms emerging from a junction. A T and an X both contain horizontal and vertical orientations.

A **ray transform** was therefore brought in from an earlier line of the project. Instead of asking only what orientation is present at the centre, it samples oriented energy around a ring. Because east and west correspond to different spatial locations, the resulting angular function is defined over 360°, not just 180°.

Fourier coefficients of that ray function can in principle encode properties such as ray number and directional asymmetry.

This was a better operator for the theoretical `F/f` distinction, but on the actual EMNIST pair it still did not improve classification.

The implementation is in [RayHarmonics.module.jl](https://github.com/johnkevinoregan/mother-embedding/blob/main/P0-8_RationalGaborFeatures/RayHarmonics.module.jl).

### 16.3 Is `F/f` simply hard in the images themselves?

To check whether the fixed front end was discarding useful pixel information, Phase 7 also gave raw pixels directly to conventional learners.

A pixel MLP, a small CNN, and the same CNN with augmentation were tested. The augmented CNN reaches about **69.9%**, essentially the same level as the orientation front end.

That convergence suggests that a substantial fraction of the `F/f` labels are visually ambiguous in this dataset rather than cleanly separable by a better descriptor.

Thus the error concentration that originally motivated the conjunction layer was partly misleading: the pair is theoretically distinguishable, but many actual handwritten instances do not reliably instantiate that distinction.

### 16.4 How correlated are conjunctions with orientation statistics?

A final regression asks how well each A₁/A₂ feature can be predicted from the ordinary orientation + low-pass features.

On held-out EMNIST test data, the median \(R^2\) is about **0.933**. Forty-two of the 54 conjunction columns have \(R^2 > 0.9\).

This explains why adding the conjunction block changes little on EMNIST: although A₁/A₂ are mathematically different operators, **their values are almost linearly recoverable from the orientation features on this particular dataset**.

The correct conclusion is therefore not “A₁ is mathematically redundant.” Synthetic controls prove that it is not. The narrower conclusion is that handwritten characters usually do not contain enough configurations that dissociate A₁/A₂ from ordinary orientation statistics.

---

## 17. Phase 8: trying to build a task where co-location must matter

Phase 8 tries to create a synthetic positive control in which touching versus non-touching structure should force a benefit from the conjunction layer.

Three versions were attempted.

The first used several classes such as crossing, tee, corner, and separated bars. The orientation baseline reached 100% because the 3 × 3 spatial layout itself gave away the class.

A second version reduced the task to touching versus gapped structures. Again the orientation baseline solved it.

A third version systematically reduced the gap. Even with only a **2-pixel gap** against strokes roughly 9–15 pixels wide, the conventional orientation representation remained about 99.9% accurate.

The reason is subtle but important: changing whether two strokes meet also changes the local spatial distribution of ink, and a spatially resolved orientation representation can exploit that change even without an explicit co-location operator.

The stimuli are shown in the [Phase 8 stimulus figure](https://github.com/johnkevinoregan/mother-embedding/blob/main/P0-8_RationalGaborFeatures/figures/phase8_stimuli.png), and the result across gap sizes is shown in the [Phase 8 gap-sweep figure](https://github.com/johnkevinoregan/mother-embedding/blob/main/P0-8_RationalGaborFeatures/figures/phase8_gapsweep.png).

One historical qualification is important. The repository now marks the broad Phase 8 generalisation as **superseded by later Phase 9 work**, which eventually constructed graded contour tasks on which conjunction and ray features did add substantial predictive value. That later result does **not** change the P0–8 conclusion about EMNIST; it shows instead that the P0–8 negative was specific to the task distribution, not proof that the operators were useless in general.

---

## 18. What was rejected, what was retained, and what remained provisional

| idea or component | status after P0–8 | reason |
| --- | --- | --- |
| old 88-dimensional windowed-Fourier front end | **rejected as the preferred front end** | reproduced at 92.3%, but the new scale-rational bank reaches about 93.7% |
| arbitrary fixed scale ladder | **rejected** | measured spectrum showed some old channels were outside useful signal bandwidth |
| data-derived scale ladder | **retained** | principled, portable, and directly tied to measured image statistics |
| frequency-domain analytic log-Gabor bank | **retained** | exact quadrature, exact polarity invariance, zero DC, flexible bandwidth/localisation |
| padding treated as an implementation detail | **rejected** | insufficient/incorrect padding measurably contaminated responses |
| pooled orientation + low-pass representation | **strongly retained** | produces nearly all of the EMNIST gain and becomes the strongest baseline |
| “pool first, then recover co-location downstream” | **rejected in principle** | local multiplication before pooling carries information that pooled summaries can erase |
| A₁ as a co-location/i2D operator | **retained as a validated operator** | synthetic tests clearly dissociate it from pooled orientation statistics |
| A₁ as a ray-count or junction-order detector | **rejected** | a π-periodic orientation profile cannot distinguish T from X by ray number |
| original unfloored A₁ | **corrected later** | small i1D leakage at the coarse scale; current code subtracts an analytic floor |
| A₂ using max across orientations | **rejected** | locally dominant orientation gives much cleaner end-stopping |
| A₂ with absolute denominator epsilon | **rejected** | caused weak/noisy channels to masquerade as end-stopping |
| fixed A₂ probe distance tied to filter envelope | **rejected as a general rule** | optimal distance tracks stroke/structure scale more closely |
| 3 × 3 pooling | **retained as the EMNIST default** | strong translation tolerance and no accuracy gain from finer baseline grids |
| claim that finer pooling would rescue A₁/A₂ on EMNIST | **rejected** | A signal strengthens, but total classification does not |
| claim that A₁/A₂ help especially with few examples | **rejected** | effect remains approximately zero from 5 to 400 examples/class |
| blur as explanation for A₁/A₂’s EMNIST null | **rejected** | realistic blur barely changes pooled A₁ contrast |
| ray harmonics as the missing F/f solution | **rejected for this EMNIST test** | theoretically more appropriate than A₁ for ray number, but no accuracy gain |
| designed invariance as a useful property of the front end | **strongly retained** | augmentation gives CNNs 4–13 points and gives the fixed features essentially nothing |
| claim that conjunctions are mathematically redundant with orientation statistics | **rejected** | synthetic dissociations prove they are different operators |
| narrower claim that conjunctions are highly correlated with orientation statistics on EMNIST | **retained** | median held-out linear predictability around \(R^2 = 0.93\) |
| strong Phase 8 claim that no task can require the conjunction layer | **later superseded** | Phase 9 eventually found tasks where conjunction/ray features matter |

---

## 19. What front end should one regard as the main P0–8 outcome?

There are two answers, depending on whether one means the **empirically sufficient EMNIST representation** or the **full experimental front-end architecture**.

For EMNIST classification, the decisive representation is the **144-dimensional orientation + low-pass front end**. It contains essentially the full +1.4-point improvement over the old 88-dimensional system. Adding A₁ and A₂ does not improve ordinary EMNIST accuracy.

For the architectural programme, however, P0–8 does **not** simply discard A₁ and A₂. They remain implemented and validated because they demonstrably encode local i2D properties that ordinary pooled orientation statistics can fail to represent. What P0–8 rejects is the stronger empirical claim that those properties are useful **on EMNIST**.

Thus the full default experimental vector at this stage remains conceptually:

`orientation + low-pass + A₁ + A₂`

with 198 dimensions at a 3 × 3 grid, but the ablation evidence says that the EMNIST performance benefit comes almost entirely from:

`orientation + low-pass`.

This distinction becomes important in later subprojects, where the same operators can be tested on tasks whose labels depend more directly on contour geometry rather than on handwritten-character identity.

---

## 20. What P0–8 says about hand-designed front ends versus CNNs

P0–8 does not establish that a fixed front end is universally better than a CNN.

Its strongest comparison is more specific.

A raw-pixel CNN with very little data performs poorly because it must learn both the task and the relevant geometric invariances. The hand-designed front end begins with orientation selectivity, polarity invariance, coarse positional tolerance, and some scale/rotation tolerance already present.

When augmentation supplies the CNN with transformed examples, much of the gap disappears.

So the correct conclusion is:

**the front end can substitute prior geometric structure for training data.**

That is a genuine advantage when data are scarce or interpretability matters. But it is not evidence that learning is unnecessary. Rather, P0–8 isolates which properties can be cheaply and transparently imposed in advance.

This also explains why the negative results about A₁/A₂ are scientifically useful. A hand-designed feature earns its place only if it contributes information relevant to the task that the rest of the representation does not already make accessible. P0–8 repeatedly tests that criterion instead of keeping a biologically attractive operator merely because it “sounds right.”

---

## 21. Bottom line

P0–8 begins with a plausible biological intuition — local junctions and line endings should deserve explicit representation — and ends with a more disciplined front end than it started with.

What unquestionably survives is the **rationalised log-Gabor oriented-energy bank**, whose scales are tied to the measured image spectrum, whose filters are correctly padded and normalised, and whose pooled orientation representation is both compact and highly effective.

The conjunction layer survives in a more qualified sense. A₁ and A₂ are real, validated operators. They do detect local structure that a pool-first orientation summary can erase. But on EMNIST, those signals happen to track information already available from the ordinary orientation representation, so they do not improve classification.

The cleanest evidence for the broader front-end programme comes from augmentation: a CNN must be shown many transformed examples to acquire invariances that the fixed representation already possesses.

P0–8 therefore establishes a useful division of labour:

- **measure image statistics to choose the front end’s scale and geometry;**
- **build in cheap, well-understood invariances where possible;**
- **validate every handcrafted operator on synthetic ground truth;**
- **do not assume that a theoretically distinct feature will help a real task;**
- **use ablations and matched controls to decide what actually earns its place.**

That methodological lesson is at least as important as the particular 93.7% EMNIST number.

---

## Repository figures

- [Phase 5a — learning curves for the old front end, new orientation baseline, and conjunction additions](https://github.com/johnkevinoregan/mother-embedding/blob/main/P0-8_RationalGaborFeatures/figures/phase5a_curves.png)
- [Phase 5b — effect of finer pooling grids](https://github.com/johnkevinoregan/mother-embedding/blob/main/P0-8_RationalGaborFeatures/figures/phase5b_curves.png)
- [Phase 5c — few-shot comparison of orientation baseline and conjunction features](https://github.com/johnkevinoregan/mother-embedding/blob/main/P0-8_RationalGaborFeatures/figures/phase5c_fewshot.png)
- [Phase 6 — effect of data augmentation on the designed features versus a small CNN](https://github.com/johnkevinoregan/mother-embedding/blob/main/P0-8_RationalGaborFeatures/figures/phase6_augmentation.png)
- [Phase 8 — synthetic touching/gap stimuli](https://github.com/johnkevinoregan/mother-embedding/blob/main/P0-8_RationalGaborFeatures/figures/phase8_stimuli.png)
- [Phase 8 — classification across the gap sweep](https://github.com/johnkevinoregan/mother-embedding/blob/main/P0-8_RationalGaborFeatures/figures/phase8_gapsweep.png)

## Main repository sources

- [P0–8 README](https://github.com/johnkevinoregan/mother-embedding/blob/main/P0-8_RationalGaborFeatures/README.md)
- [Expanded standalone results report](https://github.com/johnkevinoregan/mother-embedding/blob/main/P0-8_RationalGaborFeatures/RESULTSexpanded.md)
- [Current results and corrections](https://github.com/johnkevinoregan/mother-embedding/blob/main/P0-8_RationalGaborFeatures/RESULTS.md)
- [GaborStack.module.jl](https://github.com/johnkevinoregan/mother-embedding/blob/main/P0-8_RationalGaborFeatures/GaborStack.module.jl)
- [AndLayer.module.jl](https://github.com/johnkevinoregan/mother-embedding/blob/main/P0-8_RationalGaborFeatures/AndLayer.module.jl)
- [Pooling.module.jl](https://github.com/johnkevinoregan/mother-embedding/blob/main/P0-8_RationalGaborFeatures/Pooling.module.jl)
- [RayHarmonics.module.jl](https://github.com/johnkevinoregan/mother-embedding/blob/main/P0-8_RationalGaborFeatures/RayHarmonics.module.jl)
- [Phase5a_EMNIST.jl](https://github.com/johnkevinoregan/mother-embedding/blob/main/P0-8_RationalGaborFeatures/Phase5a_EMNIST.jl)
- [Phase5b_FinerGrid.jl](https://github.com/johnkevinoregan/mother-embedding/blob/main/P0-8_RationalGaborFeatures/Phase5b_FinerGrid.jl)
- [Phase5c_FewShot.jl](https://github.com/johnkevinoregan/mother-embedding/blob/main/P0-8_RationalGaborFeatures/Phase5c_FewShot.jl)
- [Phase6_Augmentation.jl](https://github.com/johnkevinoregan/mother-embedding/blob/main/P0-8_RationalGaborFeatures/Phase6_Augmentation.jl)
- [Phase7_FfProbe.jl](https://github.com/johnkevinoregan/mother-embedding/blob/main/P0-8_RationalGaborFeatures/Phase7_FfProbe.jl)
- [Phase8_JunctionBenchmark.jl](https://github.com/johnkevinoregan/mother-embedding/blob/main/P0-8_RationalGaborFeatures/Phase8_JunctionBenchmark.jl)
