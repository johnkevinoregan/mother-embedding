# A designed visual front end, and what happened when it met data

*A self-contained report. It assumes no knowledge of this project, of computer vision, or
of the terminology. Every term is defined where it first appears, and every experiment is
described in enough detail to be repeated. It is long by intention: the short version lives
in [`RESULTS.md`](RESULTS.md).*

---

# Part 0 · The one-paragraph version

We built a hand-designed image front end — a fixed, non-learned stage that converts a
picture into a few hundred numbers — with the specific aim of representing something
ordinary systems get wrong: whether two edges *meet at a point* or merely pass near each
other. On handwritten characters it works, and beats the previous design by 1.4 percentage
points. **But the part that represents "meeting" contributes nothing**, and we spent the
larger part of the work establishing why, since the obvious explanations turned out to be
wrong. The answer is that the new measurement is **93 % linearly predictable** from
measurements the system already had — not because the two are the same operation, which we
proved they are not, but because handwritten letters happen not to contain the
configurations that tell them apart. Separately, a control we had been postponing produced
the project's cleanest positive result: **data augmentation improves a conventional neural
network by 9–13 points and improves these features by nothing**, which is direct evidence
that the designed features already contain the invariances augmentation exists to teach.

---

# Part 1 · Background

## 1.1 The task

**EMNIST-Balanced** is a public dataset of handwritten characters: 131,600 images of
28×28 pixels, greyscale, each showing one character. There are 47 classes — the ten digits
and 37 letter classes, where visually identical upper- and lower-case forms have already
been merged by the dataset's authors. It ships with an official split of **112,800 training
images and 18,800 test images**, 2,400 and 400 per class.

Every number in this report uses that official split, so results are comparable with
published work.

**Homoglyph merging.** Some remaining classes are the *same drawn shape*: a handwritten
zero and a capital O, or 1 / I / L. No system can separate them from the ink alone, and
earlier work in this project measured that this accounts for a constant ~6 points of error
across every model tried. We therefore merge five groups — `0/O`, `1/I/L`, `2/Z`, `5/S`,
`9/g/q` — leaving **40 classes**, with a random-guess baseline of 2.50 %. This is not
cheating; it is removing a known-undecidable component so the remaining differences are
about the *representation* rather than about the dataset's labelling.

**Standard error.** With 18,800 test images, an accuracy near 93 % has a standard error of
about **0.19 percentage points**. Differences smaller than roughly 0.4 points are not
meaningful from a single run. This number is used throughout to decide what counts as a
result.

## 1.2 What a "front end" is, and why build one by hand

A modern image classifier learns everything from data: it is handed raw pixels and adjusts
millions of parameters until it classifies well. A **front end** is a fixed stage placed
*before* any learning, which converts the image into a smaller set of measurements. It has
no parameters to train — it computes the same function on every dataset, forever.

Two reasons to want one:

**Sample efficiency.** A learned system must discover from examples that a letter is the
same letter when shifted two pixels left, rotated slightly, or drawn with a thicker pen. A
front end can be *built* with those insensitivities, so nothing has to be learned about
them. With few training examples that should matter a great deal.

**Biological plausibility.** The early visual system appears to compute something like a
fixed set of oriented filters before any task-specific learning. If we want a model of
that, it must be built rather than trained.

The project's stated goal is a front end that is **general** — testable on photographs, not
just characters — so anything specific to handwriting is a defect, not a convenience. This
becomes important in §4.10.

## 1.3 Gabor filters and oriented energy

The building block is a **Gabor filter**: a wave of a particular wavelength and direction,
multiplied by a bell-shaped (Gaussian) window that confines it to a small patch. It
responds strongly where the image contains a stripe or edge of matching orientation and
thickness, and weakly elsewhere. It is the standard model of a simple cell in primary
visual cortex.

Two of them are used together, one a sine and one a cosine of the same wave — a
**quadrature pair**. Squaring and adding their responses,

```
E  =  (even response)²  +  (odd response)²
```

gives the **oriented energy**: how much stripe-like structure of that orientation and
thickness is present at that point, *regardless of whether it is a light stripe on dark or
a dark stripe on light*, and regardless of exactly where within the patch it falls. That
property is called **polarity invariance**, and it is exact here rather than approximate —
negating the image leaves `E` numerically unchanged, which the tests verify to the last
bit.

Applying a whole **bank** of such filters — several orientations at several wavelengths —
at every pixel gives a stack of energy maps `E(x, y, θ, λ)`. That stack is the raw material
for everything below.

## 1.4 Intrinsic dimension: i0D, i1D, i2D

A standard classification of local image structure:

- **i0D** — nothing happening. A flat region.
- **i1D** — variation in one direction only: a straight edge, a straight stripe, a grating.
- **i2D** — genuine two-dimensional structure: a **corner**, a **junction** where strokes
  meet, a **line end**, a point of high curvature.

i2D structure is where the information about *shape* lives. A capital F and a lower-case f
are drawn from the same strokes at the same orientations; what differs is how those strokes
*meet*.

**The classical difficulty** (Zetzsche & Barth): a *linear* filter cannot selectively
signal i2D structure. Any linear filter that responds to a corner also responds to an
ordinary edge. Isolating i2D requires an **AND-like** nonlinearity — a mechanism that fires
only when two conditions hold *at the same place*. In cortex this is attributed to
end-stopped (hypercomplex) cells.

## 1.5 The specific problem: co-location

Here is the difficulty this project is built around, in its simplest form.

Consider two pictures. In the first, a horizontal stroke and a vertical stroke **meet** at a
point — a corner. In the second, the same two strokes are present but **pulled apart** so
they do not touch.

Both pictures contain exactly the same amount of horizontal structure and exactly the same
amount of vertical structure. A summary that reports *how much of each orientation is
present in this region* returns nearly the same answer for both. The difference is entirely
whether the two orientations occupy the **same point**.

This is not a hypothetical. Measured on our own system, the pooled orientation summary
gives a cosine similarity of **0.97** between a corner and two separated strokes — they are
nearly the same vector.

---

# Part 2 · The design and its central claim

## 2.1 The pipeline

```
image (28×28)
  → upsample to 112×112                        (interpolation; adds no information, but
                                                allows filters to be placed sub-pixel)
  → pad into a 224×224 field                   (prevents wraparound in the FFT)
  → bank of log-Gabor filters                  (3 wavelengths × 8/12/16 orientations)
  → oriented energy  E = even² + odd²          (polarity-invariant)
  → CONJUNCTION LAYER: A₁, A₂                  (the contribution of this work)
  → pool over a 3×3 grid                       (positional tolerance)
  → 198 numbers
  → one hidden layer of 256 units → 40 classes
```

## 2.2 The central claim: multiplication and pooling do not commute

**Pooling** means averaging a quantity over a spatial region — here, over each cell of a
3×3 grid laid across the character. It is what gives tolerance to small shifts: a junction
that moves three pixels between two handwritten instances should not produce a different
answer.

The claim is about *order*. Consider two energy maps `e₁` and `e₂` (say, horizontal and
vertical structure), and a pooling window `w`:

```
pool, then multiply :   ( Σₓ w(x)·e₁(x) ) · ( Σₓ w(x)·e₂(x) )
multiply, then pool :     Σₓ w(x)·e₁(x)·e₂(x)
```

These are **not equal**. Their difference is exactly the spatial covariance of `e₁` and `e₂`
inside the window — *do the two quantities peak in the same place?* — and that covariance
**is** the co-location signal.

The consequence is severe and is an information argument, not a difficulty of training:
**any statistic computed from already-pooled orientation energy has had the co-location
signal averaged away, and no amount of downstream capacity can recover it.** That applies to
the previous version of this project's features, and it applies to the "squeeze-and-
excitation" channel-gating popular in the literature, which multiplies whole feature maps by
scalars derived from *globally* pooled statistics and therefore has no spatial resolution at
all.

So the design rule is:

```
select (oriented filter)  →  multiply (conjunction)  →  pool (tolerance)
```

which is also simple cell → complex cell, and convolution → nonlinearity → pooling in a
convolutional network. **Detect, then tolerate.** Pooling is not the enemy; pooling *first*
is.

## 2.3 The two conjunction operators

**A₁ — same-location orientation conjunction.**

At each pixel, take the profile of energy across orientations, normalise it, and correlate
it with itself shifted by 90°:

```
A₁(x)  =  C₀(x) · Σₖ pₖ(x) · pₖ₊ₙ⁄₂(x)
```

where `pₖ` is the normalised energy at orientation `k`, `n` is the number of orientations,
and the `n/2` shift is exactly a right angle. `C₀` is the total energy, which reweights the
result so it is an energy rather than a shape measure, keeping it well-behaved where there
is no ink.

It is **zero** where a single orientation dominates, and **maximal** where two perpendicular
orientations coexist *at one point*. Corners, crossings, junctions.

**A₂ — end-stopping.**

At each pixel, take the orientation that dominates there, and compare the energy of that
same channel at two points offset **along** the stroke:

```
A₂(x)  =  E(x) · |E₊ − E₋| / (E₊ + E₋ + κ·E(x)),      E±  =  E(x ± d·u)
```

The fraction is an asymmetry measure. It is ~0 in the middle of a line (the stroke continues
both ways), ~0 on an isolated dot (it continues neither way), and large at a **termination**
(it continues one way only).

Both are built from energies alone, so both inherit exact polarity invariance. Neither uses
phase, a bispectrum, or complex values.

---

# Part 3 · Why these tests were necessary

## 3.1 The project had already been badly burned once

Earlier work in this repository evaluated feature sets with a deliberately simple
classifier — nearest-class-mean, which represents each class by the average of its examples
and assigns each new item to the closest one. It is cheap and requires no training.

It was **wrong by 24 percentage points**, and worse, it *inverted a qualitative conclusion*.
Under that classifier, combining two feature families appeared to be worth +6.4 points, and
this was written up as strong evidence they were complementary, with a control to back it.
Under a properly trained network the same combination was worth **+0.17** — the apparent
complementarity was the weak classifier's inability to exploit either family on its own.

The lesson, recorded at the time: *a weak classifier gives a floor, not a measurement, and
comparisons between representations demand a classifier strong enough to exploit each one
alone.* Every experiment below inherits that.

## 3.2 A front end validated only by accuracy can be silently broken

This is the second discipline. If the only check is downstream classification accuracy,
then a front end that is subtly wrong may still score respectably, because a trained network
compensates. This project had already lost time to exactly that: a keypoint detector that
turned out to be miscalibrated **on clean synthetic input**, discovered long after it had
been built on.

So every stage here is gated on **synthetic stimuli with known ground truth**, with pass
criteria written down *before* the numbers are produced, and nothing proceeds until the
gates pass. That discipline caught two real bugs (§4.2) that no accuracy number would have
revealed.

## 3.3 The specific doubts

Each experiment addresses a stated doubt:

| # | doubt | test |
|--:|:--|:--|
| 1 | Does the new front end even reproduce what the old one achieved? | §4.5 |
| 2 | Does the conjunction layer help? | §4.5 |
| 3 | If not, is the 3×3 pooling destroying it? | §4.6 |
| 4 | Or does it help only when data is scarce? | §4.7 |
| 5 | Do the features really *carry* invariances, or just outperform? | §4.8 |
| 6 | Is it the difference between sharp synthetic edges and blurry real ones? | §4.9 |
| 7 | Is the conjunction genuinely different information, or the same measured twice? | §4.10 |
| 8 | Are the operator's constants general, or secretly fitted to this dataset? | §4.11 |

---

# Part 4 · The experiments

## 4.1 Phase 0 — fixing every parameter by measurement

**Why.** Every later choice depends on the scale of the structures in the image. Guessing
them is how a filter bank ends up analysing noise.

**What was done.** Two measurements on 20,000 training images.

*Stroke width.* For a long thin ribbon, width ≈ 2 × area / perimeter. Applied per image
after thresholding at 0.5, taking the median.

*Radial power spectrum.* The 2-D Fourier transform of each image, squared, averaged over
images, with the constant (DC) term removed, then averaged into rings of equal radial
frequency ρ measured in cycles per image width.

**Results.**

Stroke width: **12.67 px** on the 112-grid (3.17 px natively).

| ρ | wavelength @112 | share of energy | cumulative above |
|--:|--:|--:|--:|
| 1 | 112 px | 33.9 % | 100 % |
| 2 | 56 | 19.4 % | 66.1 % |
| 3 | 37 | 15.9 % | 46.7 % |
| 4 | 28 | 14.5 % | 30.8 % |
| 5 | 22 | 6.0 % | 16.4 % |
| 7 | 16 | 2.2 % | **6.1 %** |
| 9–14 | 12–8 | 2.6 % total | 2.6 % |

**What it settled.** The usable band is ρ ∈ [2, 7]; above ρ = 7 only 6 % of the energy
remains, and most of that is interpolation artefact. The scale ladder was placed there:
**ρ = 2.00 / 3.74 / 7.00**, i.e. wavelengths 56 / 30 / 16 px.

**What it exposed.** The project's older filter bank used wavelengths [3, 6, 12, 24] on a
56-px grid. Converting: one of those channels sits **beyond the Nyquist limit of the source
data entirely** — it was measuring interpolation — and another sits where ~1 % of the energy
lives. Half the bank had been analysing nothing, which is a better explanation of that
line's disappointing results than any of the ones recorded at the time.

## 4.2 Phases 1–2 — building the bank, and the gates that caught two bugs

**What was done.** Filters are constructed **in the frequency domain** as one-sided
(analytic) Gaussian bumps, rather than as spatial kernels. Three consequences: there is no
kernel to truncate, the bank works at any image size, and the even/odd quadrature pair is
exact by construction. The family is **log-Gabor** — a Gaussian on a *logarithmic* frequency
axis — which is exactly zero at DC (`log 0 = −∞`) and, more importantly, decouples the
spatial extent of the filter from its wavelength, so a filter tuned to a coarse wavelength
can still localise finely.

**Six gates**, criteria fixed in advance:

1. **polarity invariance** — must be *exactly* zero difference between `I` and `−I`
2. **DC rejection** — a constant image must give zero energy
3. **orientation readout** — a bar at angle θ must excite the correct filter, within half
   the orientation spacing
4. **padding invariance** — doubling the border must not change the answer (wraparound is an
   artefact, so the test is an invariance, not a magnitude)
5. **scale tuning** — each filter must peak at the wavelength it is tuned to
6. **localisation** — energy must sit on the stimulus, judged per scale against that
   scale's own extent

**Two real bugs were caught, neither of which any accuracy number would have shown.**

*Units.* The frequency ρ was being computed in cycles per **padded field** width while the
ladder specified cycles per **image** width. Every filter was mistuned by 2.86×. The
diagnostic tell was peculiar and instructive: **adding padding made results worse**, which
is the opposite of what a genuine wraparound problem does.

*Normalisation.* Filters were normalised by the sum of squares over the discrete frequency
grid, which grows with the grid. Every response therefore scaled as 1/√(HW), so merely
adding padding shifted every feature by **63 %**. Predicted from the ratio: 1/1.64 in
amplitude, 0.37 in energy, 0.63 change. Measured: 0.628. Fixed by normalising to RMS, which
approximates the continuous integral and is field-size independent.

**Three of my own tests were also wrong** and had to be repaired: a fixed localisation mask
applied to scales whose extents differ by 4×; a "which scale wins" criterion applied to a
deliberately overlapping bank, where the question is not well posed; and a monotonicity
check that **passed vacuously** on a constant vector.

## 4.3 Phase 3 — the conjunction layer and its gates

**What was done.** A₁ and A₂ as defined in §2.3, validated on synthetic figures where the
answer is known.

The key stimulus pair: **two bars at 90° whose centres are separated by a controllable gap.**
At gap 0 they cross at the centre — maximal co-location. At gap 70 their nearest points are
~30 px apart. **The orientation content is identical at every gap**; only co-location
changes.

**Results.**

| | |
|:--|--:|
| A₁ at the junction, crossing vs separated, per scale | 10.3× / 7.1× / **17.6×** |
| A₁ across the gap sweep | 1.00 → **0.06** |
| …while pooled orientation energy over the same sweep | 1.00 → **0.97** |
| A₂ end/interior, finest scale | **10.4×** |
| A₂ end/flank · end/isolated blob | 4.4× · 4.8× |

The second and third rows are the point: over the same stimuli, the conjunction collapses to
6 % while the conventional statistic does not move. **This is a proof, on ground truth, that
the two measure different things.** It matters later, when the same two turn out to be
near-interchangeable on real characters.

**Four bugs found here**, all invisible downstream:

- `σ_φ` was stored in two places, so A₂'s probe offsets silently kept stale values when the
  bank's tuning changed. It now lives in the bank's metadata and cannot drift.
- A₂'s asymmetry used an absolute epsilon, so a channel whose own flanks were both near-zero
  scored ≈1 from noise and A₂ degenerated into plain energy. Normalising against the centre
  response fixed it.
- A₂ took a maximum over all orientations; taking the **locally dominant** orientation
  instead changed end/interior from **2.5× to 10.4×**. It is also the faithful model of an
  end-stopped cell, which inherits the local orientation.
- The angular tuning parameter had been inherited from a standard reference at 1.5, giving a
  filter **34 px long** on a 112-px image — longer than any stroke in the dataset, so no
  stroke ever looked uniform and end-stopping could not work. Halving it fixed that and
  shrank the padded field from 320 to 224 as a bonus.

The probe offset `d` was chosen by **sweeping**, after two failed guesses (1.5 too large,
0.75 too small). That sweep is revisited critically in §4.11.

## 4.4 Phase 4 — pooling, and the control built in from the start

**What was done.** Gaussian-weighted, overlapping pooling windows on an n×n grid specified
*relative to the image*, never in pixels, so the same specification transfers to another
dataset at another size. Feature blocks are independently switchable, so any of them can be
ablated by a one-line change.

The default vector is **198 numbers**: 135 orientation statistics + 9 low-pass + 27 A₁ + 27
A₂.

**The `orient` block is deliberately the *deficient* baseline.** It computes exactly the
statistics that pool first and combine after, which we have argued cannot represent
co-location. Keeping it as a block rather than a footnote makes the whole experiment a
one-line ablation.

**The dimensionality control was built in at this point rather than added later.**
`shuffle_block!` permutes a block's rows *across samples*: the columns keep their marginal
distribution and their count, but each sample's correspondence to its own values is
destroyed. This is necessary because earlier work established that *a fixed projection into
a few hundred dimensions plus a trained classifier is a strong baseline whatever the
projection is* — scrambled features still reached 75 %. Without the twin, "adding A₁ helped"
and "adding 27 more columns helped" are the same measurement.

**The gate**, and the property pooling exists for: a 4 px shift (about a third of a stroke
width) leaves the 3×3 vector at **cosine 0.998**, against 0.995 at 6×6 and 0.994 at 11×11.
Finer grids keep more detail and tolerate less — the trade the grid parameter prices.

**A performance note that mattered.** Extraction started at 287 ms per image and ended at
**20.9 ms**, a 13.7× speedup, from two causes: the conjunction loops iterated pixel-outer
with the channel as the *last* array index, so every channel access strode 12,544 floats and
missed cache; and the bank's metadata was an abstractly-typed container, making every lookup
type-unstable. Every gate number was verified bit-identical after each rewrite — an
optimisation that changes results is not an optimisation. This took full-dataset extraction
from 78 minutes to about 6, which is what made the later ablation grids practical.

## 4.5 Phase 5a — does it reproduce, and does the conjunction help?

**Why the reference arm comes first.** A new pipeline that cannot reproduce a known result
has a bug, and discovering that *after* running an ablation wastes the ablation.

**What was done.** The previous feature set (88 numbers, a different construction entirely —
windowed Fourier patches rather than log-Gabor energy) was **re-extracted and re-trained
inside the new harness**, with the same training loop, the same optimiser settings, the same
seed. Its published figure is 92.30 %.

Then three arms of the new pipeline, each adding one block.

Training throughout: one hidden layer of 256 units, Adam at 1e-3, batch 128, 15 epochs,
seed 1, inputs standardised using training statistics only and clipped at 3 standard
deviations.

**Results.**

| arm | numbers | final | best |
|:--|--:|--:|--:|
| **reference — previous features** | 88 | **92.31 %** | 92.42 % |
| new: orientation + low-pass | 144 | **93.71 %** | 93.78 % |
| + A₁ | 171 | 93.78 % | 93.78 % |
| + A₁ + A₂ | 198 | 93.71 % | 93.71 % |

The reference lands on **92.30 %** against the recorded 92.30 %. The harness is sound.

**The new front end is worth +1.40 points**, about 8 standard errors — a real gain, and it
comes from the bank itself: the ladder placed on the measured spectrum, more orientations,
correct padding.

**The conjunction layer adds +0.07, then −0.12.** Nothing.

**But it is not that it computes nothing.** Training on each block alone:

| block alone | numbers | accuracy |
|:--|--:|--:|
| orientation statistics | 135 | 93.59 % |
| **A₁ + A₂** | **54** | **88.45 %** |
| A₂ | 27 | 78.27 % |
| A₁ | 27 | 75.63 % |
| low-pass | 9 | 62.15 % |

54 conjunction numbers alone reach 88.45 %.

**A targeted check.** A confusion analysis of the previous system had shown its errors were
concentrated in pairs distinguishable by junction structure — `F`/`f` alone was **17.3 % of
all remaining errors**. Counting errors on those pairs:

| pair | previous | new | new + conjunctions |
|:--|--:|--:|--:|
| **F / f** | 251 | 265 | **257** |
| 0 / D | 37 | 32 | 28 |
| T / t | 34 | 27 | 33 |
| 4 / Y | 30 | 27 | 26 |
| U / V | 26 | 26 | 27 |
| C / e | 20 | 14 | 19 |
| X / Y | 10 | 6 | 7 |
| K / h | 9 | 4 | 6 |
| **total** | **417** | **401** | **403** |

Total errors fall from 1,448 to 1,176 while junction-pair errors stay flat — so they *rise*
as a share, from 28.8 % to 34.1 %. The new bank fixes non-junction errors and leaves the
junction errors exactly where they were.

**A ceiling worth stating.** `F`/`f` is 17.3 % of *errors* but only **1.3 % of test items**.
Solving it perfectly caps any gain at **+1.34 points**, and a partial improvement on a 1.3 %
subset against a 0.19 % standard error is not measurable in aggregate. Absence of evidence
is weak evidence here — which is why §4.10 tests the pair directly.

## 4.6 Phase 5b — is the pooling destroying it?

**The hypothesis.** A₁ is a *point* property, and it is being read out by averaging over
cells 37 px across. A sharp peak averaged over a region that size may simply not survive.

**What was done.** Pool the conjunction blocks at 3×3, 6×6 and 11×11 — the last being the
finest grid the sampling theory licenses — while holding the orientation baseline at 3×3.

Two controls, both necessary:

- **the baseline at the same grids.** If finer pooling helps everything, the finding is about
  the grid rather than about conjunction.
- **a shuffled twin for every arm.** 3×3 → 11×11 takes the conjunction blocks from 54 to 726
  columns, a 13× increase.

**Results.**

| conjunction grid | A alone | baseline + A | **Δ vs baseline** | shuffled twin |
|:--|--:|--:|--:|--:|
| 3×3 | 88.40 % | 93.71 % | **+0.01** | −0.81 |
| 6×6 | **91.49 %** | 93.16 % | **−0.55** | −1.88 |
| 11×11 | 90.94 % | 92.52 % | **−1.19** | −3.32 |

**The pooling *was* discarding signal** — the conjunction blocks alone gain **+3.1 points**
at 6×6. **And recovering it changes nothing downstream.**

**The grid control is what makes this conclusive.** The baseline at the same three grids
gives 93.71 → 93.73 → 93.10. Finer pooling does not help it either. **The task is saturated
at 3×3 spatial resolution** — a fact about the dataset, not about conjunctions.

**And the columns are demonstrably informative.** At 11×11, adding them costs −1.19 while
adding *the same number of shuffled columns* costs −3.32. They hurt far less than noise.

## 4.7 Phase 5c — does it help when data is scarce?

**Where the hypothesis came from.** Looking at the learning curves from 5a, the conjunction
arm sits slightly above the others for the first four epochs (+0.16 average) and the gap
closes by epoch 12 (+0.01). The suggested mechanism: the conjunctions supply something the
network can otherwise *learn* from the orientation statistics, given enough data. If so, the
advantage should grow as data shrinks — which is the project's central thesis about
sample efficiency.

The +0.16 is below the 0.186 % standard error, so it is not evidence. But it is a testable
mechanism.

**What was done.** Training sets of k images per class for k ∈ {5, 10, 20, 50, 100, 400},
drawn once per seed and reused, with **every arm training on the identical images**. A fixed
budget of 4,000 gradient steps regardless of k, so a larger k does not also buy more
updates. Five seeds. A shuffled twin at every k.

**Results.**

| k | images | baseline | + conjunctions | **Δ** | Δ from shuffled |
|--:|--:|--:|--:|--:|--:|
| 5 | 200 | 64.23 | 63.78 | **−0.44 ± 0.49** | −10.10 |
| 10 | 400 | 74.88 | 75.11 | **+0.22 ± 0.46** | −7.38 |
| 20 | 800 | 81.44 | 81.64 | **+0.20 ± 0.34** | −3.97 |
| 50 | 2 000 | 86.39 | 86.61 | **+0.22 ± 0.24** | −1.91 |
| 100 | 4 000 | 88.56 | 88.53 | **−0.03 ± 0.44** | −1.34 |
| 400 | 16 000 | 90.83 | 90.71 | **−0.12 ± 0.16** | −1.18 |
| all | 112 800 | | | **+0.01** | |

**Flat and indistinguishable from zero at every sample size**, every value inside its own
standard deviation. The mechanism predicted several points at small k; the early-epoch
pattern was noise. The correlation is **unconditional** — the orientation statistics already
contain the relevant information at 200 training images.

The shuffled column earns its keep here: at k = 5, adding 54 *shuffled* columns costs
**−10.10 points**. A data-starved model is badly hurt by extra columns, so the conjunctions
costing only −0.44 is itself evidence that they are informative.

## 4.8 Phase 6 — augmentation, and the cleanest positive result

**Why this control matters more than the others.** The project's headline claim had been
that designed features beat a small convolutional network by +7.7 points when given only 10
examples per class. But that network had to *discover* its invariances from raw pixels with
nothing to help it. **Data augmentation — training on randomly rotated, shifted and rescaled
copies — is precisely how you hand those invariances to a network.** So the recorded claim
was "designed invariances versus invariances learned from raw data alone", which is weaker
and less interesting than the claim we want.

The interesting form is a **differential**:

> if the features really encode these invariances, augmentation should help the network a
> lot and help the features barely

**What was done.** Augmentation is ±10° rotation, ±10 % scale, ±2 px translation, applied to
the 28×28 image by inverse mapping with bilinear sampling. Ten augmented copies per training
image.

Two design points that matter:

- **augmenting the feature pipeline means augmenting the images and re-extracting**, not
  perturbing feature vectors, which would test something else entirely
- **both arms see the identical augmented images** — generated once per (k, seed); the
  feature arm re-extracts from them, the network takes their raw pixels. Otherwise the two
  differ in their data as well as their treatment.

Fixed 4,000-step budget, so augmentation buys *variety* and not extra updates. Five seeds.

**Results.**

| k | images | features | **Δ aug** | small CNN | **Δ aug** |
|--:|--:|--:|--:|--:|--:|
| 5 | 200 | 63.78 ± 1.32 | **+0.39** | 48.48 ± 0.89 | **+11.54** |
| 10 | 400 | 75.11 ± 1.37 | **−0.13** | 61.19 ± 2.50 | **+12.77** |
| 20 | 800 | 81.64 ± 0.79 | **−0.18** | 70.60 ± 0.61 | **+9.10** |
| 50 | 2 000 | 86.60 ± 0.37 | **−1.21** | 81.42 ± 0.84 | **+3.87** |

**Augmentation buys the network 3.9 to 12.8 points and buys the features nothing.** The
differential is 5–13 points and the direction is unambiguous. **This is direct evidence that
the designed features already carry the invariances augmentation exists to teach** — a much
stronger statement than "the features win", and the thing the front end was built to
demonstrate.

Two honest riders.

**Augmentation largely closes the gap.** The features' advantage over the network, before
and after:

| k | before | after |
|--:|--:|--:|
| 5 | +15.30 | **+4.15** |
| 10 | +13.92 | **+1.02** |
| 20 | +11.04 | **+1.76** |
| 50 | +5.18 | **+0.10** |

By k = 50 they are level. The *mechanism* claim is confirmed; the *practical* claim weakens.
You can buy most of the same invariance with augmentation — at the cost of ten times the
training data and no interpretability.

**Augmentation makes the features slightly worse at larger k** (−1.21). Expected: it adds no
information they lack, but under a fixed step budget it dilutes the effective sample.

## 4.9 Was it the blurred edges?

**The hypothesis.** Our synthetic stimuli are hard binary — pixels are exactly 0 or 1. Real
characters are anti-aliased at source and then interpolated. **A blurred corner is not two
orientations superposed at a point**; it is a smoothly *rotating* single orientation along a
rounded contour — which is exactly the structure A₁ detects, dissolved.

**What was done.** First measure the blur. Two independent statistics on 5,000 images: the
fraction of ink pixels at intermediate intensity, and the edge transition width inferred from
the median gradient along edges. Then blur the synthetic stimuli with a Gaussian across a
range of widths and re-run the Phase 3 contrasts — **both on the dense maps and on the
pooled features the classifier actually receives**, because those are not the same thing.

**The blur measurement.**

| | EMNIST | synthetic |
|:--|--:|--:|
| ink pixels at intermediate intensity | **49.1 %** | 0.0 % |
| edge transition width | 2.05 px natively = **8.2 px @112** | 0 |

Nearly half of every character is edge ramp rather than solid ink, and the ramp is
comparable to the 12.7 px stroke width.

**The sweep.**

| blur σ | A₁ dense | **A₁ pooled** | A₂ dense | **A₂ pooled** |
|--:|--:|--:|--:|--:|
| 0 (sharp) | 17.6× | **4.9×** | 10.4× | **2.6×** |
| 2.0 — matches the 49 % figure | 16.3× | **5.0×** | 8.1× | 2.4× |
| 3.3 — matches the 8.2 px figure | 15.5× | **5.0×** | 6.3× | 2.3× |
| 8.0 | 3.7× | 1.8× | 3.2× | 1.9× |

**The hypothesis is excluded, and the reason is instructive.** **Pooling costs far more than
blur.** At zero blur, 3×3 pooling already takes A₁ from 17.6× to 4.9×. EMNIST-level blur then
costs *nothing on top* — 4.9 → 5.0. It takes double the real blur before the contrast
collapses.

One real finding inside the negative: **A₂ is about three times more blur-sensitive than A₁**
(−40 % versus −12 % at matched blur). A termination is a sharper event than a crossing.

**This also exposed a flaw in how Phase 3 was gated** — see §5.2.

## 4.10 Phase 7 — the direct test, and a number for the correlation

**Two doubts remained**, both stemming from the same problem: aggregate accuracy over 40
classes cannot say *why* the conjunctions add nothing.

### 4.10.1 A binary F-versus-f classifier

**Why.** `F`/`f` is 17.3 % of errors but 1.3 % of items, so a partial fix is unmeasurable in
aggregate. Training on that pair alone removes the dilution entirely and asks the question
directly: *for the distinction these operators were designed for, is the conjunction better
than the conventional statistic?*

**Ray harmonics were included, and this needs explaining.** A₁ is built on the orientation
profile, which is **π-periodic** — it knows orientation but not direction. A T-junction (3
rays) and an X-crossing (4 rays) have *identical* orientation content, {0°, 90°}. So **A₁
provably cannot count rays**, and `F` (3-ray T) versus `f` (4-ray X) is exactly a ray-count
distinction.

The operator that can is the **ray transform**, from an earlier line of this project:

```
R(p, φ) = E( p + d·u(φ),  stroke orientation = φ mod π )
c_n = Fourier coefficients of R over φ
```

For each direction φ around a ring of radius d, sample the energy channel *aligned with that
direction*. The spatial offset is what converts a mod-π quantity into a mod-2π one — east and
west read different pixels. `c₀` counts rays; `|c₁|/c₀` measures asymmetry. Note it is
**linear in the energy field** where A₁ is bilinear: the work is done by the geometry of the
sampling, not by a product. This was implemented for the test.

**What was done.** 4,800 training and 800 test images, perfectly balanced. A small network
(64 hidden units, 40 epochs), three seeds, best epoch reported.

**Results.**

| features | numbers | accuracy |
|:--|--:|--:|
| **orientation + low-pass** | 144 | **69.88 % ± 0.88** |
| A₁ + A₂ | 54 | 67.54 % ± 0.19 |
| ray harmonics | 81 | 67.50 % ± 0.87 |
| orientation + A₁ + A₂ | 198 | 69.62 % ± 0.45 |
| orientation + rays | 225 | 68.88 % ± 0.78 |
| everything | 279 | 69.29 % ± 0.75 |

**No operator beats the conventional statistic, and none adds to it** — on the very pair they
were designed for, with the dilution removed. The "different information, drowned by
averaging" hypothesis does not survive.

**And note the absolute level.** 69.9 % on a **balanced binary** task, against 50 % chance.
The whole front end is poor at `F`/`f`. It is not that the conjunctions fail where the
conventional statistic succeeds — **all three are near-equally mediocre**, which is a sharper
open question than the one this test was meant to close.

### 4.10.2 How correlated are they, exactly?

**Why.** "Redundant" and "correlated" are different claims. Redundant means one is a function
of the other — nothing there to use. Correlated means they are different measurements whose
values happen to co-vary on this data. Accuracy cannot distinguish them; a regression can.

**What was done.** Least-squares fit of each conjunction column on the 144
orientation+low-pass columns, fitted on training data and scored on held-out test data,
reporting R² per column. R² is the fraction of a column's variance predictable from the
others: 1.0 means perfectly recoverable, 0 means not at all.

**Results.**

| | median R² | mean R² |
|:--|--:|--:|
| **all 40 classes, 112,800 images** | **0.933** | 0.924 |
| F/f images only | 0.956 | 0.943 |
| ray harmonics ← orientation, F/f | 0.910 | 0.897 |

**42 of the 54 conjunction columns have R² above 0.9; all 54 exceed 0.75.**

**This settles the question.** The conjunctions are **93 % linearly recoverable** from the
orientation statistics *on this dataset*. So "strongly correlated in this image set" is the
correct description and "redundant" was wrong — because Phase 3 showed the same two
operators dissociating by 4.9× (pooled) on stimuli built to require it.

**They are different operators that become near-collinear on handwritten characters.** Real
letters do not produce the configurations that separate them.

## 4.11 Are the operator's constants general, or fitted to this dataset?

**The doubt.** Two constants sit inside a module that is meant to be general: the probe
offset `d_factor = 1.0` and the angular tuning `dtheta = 0.75`. Both were chosen against
EMNIST-shaped evidence. `d_factor` was swept on a synthetic bar 13 px wide — a number chosen
to match EMNIST's stroke. `dtheta` was justified explicitly by "the alternative gives a
filter longer than any EMNIST stroke". **That is an EMNIST argument for a constant in a
general module.**

The distinction worth preserving: the **scale ladder** *should* adapt to the data, and does.
The **operator's internal geometry** should be a constant of the operator, fixed relative to
its own filters — as an end-stopped cell's inhibitory zone is fixed relative to its own
receptive field, not to the stimulus.

**What was done.** A dimensionless sweep with no EMNIST in it. Note the algebra: writing
`σ_φ = (π/n)/dtheta`, the filter's along-contour extent satisfies

```
σ_along / λ  =  n · dtheta / (2π²)
```

which is **independent of the wavelength**. The operator's shape is fixed by the orientation
count and `dtheta` alone, so the only free ratio left is the **stimulus scale relative to the
filter**, `w/λ`. Sweeping that with the stimulus *fixed* and the *filter scale* varied keeps
every index range constant.

**Results.**

```
best d_factor across w/λ = 0.30 / 0.50 / 0.80 / 1.20 :  [3.0, 0.5, 1.5, 2.0]   DRIFTS
best dtheta   across the same                        :  [0.5, 0.5, 0.75, 1.0]  DRIFTS
```

Converting the optima into physical offsets (excluding w/λ = 0.30, where every score is
1–2.8 and the operator is failing regardless):

| w/λ | λ | σ_along | best d | **d / stroke width** |
|--:|--:|--:|--:|--:|
| 0.50 | 25.8 | 15.7 | 7.8 | 0.60 |
| 0.80 | 16.3 | 9.9 | 14.9 | 1.15 |
| 1.20 | 10.8 | 6.6 | 13.2 | 1.01 |

**The optimal offset is approximately one stroke width, not one filter envelope.** `d` is
anchored to the wrong quantity — to the filter's own extent rather than to the structure it
measures.

1.0 looked correct on EMNIST only because σ_along = 9.7 against a 12.7 px stroke gives
d = 0.76 stroke widths **by coincidence of that operating point**. A constant fitted at one
scale, which would fail silently rather than loudly at another.

**The fix fits the existing architecture.** The pipeline already derives its ladder from the
data; it should equally derive a **structure scale** — Phase 0 already measures stroke width
— and anchor `d` to that.

*Caveats: one stimulus family, a coarse four-point sweep, and the smallest ratio is degenerate
rather than informative.*

---

# Part 5 · Corrections to earlier claims

Recorded because the repository contains both the original claim and its correction, and
because the pattern is more informative than any single number.

## 5.1 "A₁ orders junctions by ray count" — an ink artefact

Phase 3 reported: straight 6.3e4 < L-corner 9.5e4 < T-junction 1.15e5 < X-crossing 1.58e5,
and this was written up as an unprompted result — the conjunction spontaneously grading with
junction complexity.

**The four stimuli contain 980 / 1052 / 1359 / 1708 inked pixels**, and the totals track that
almost exactly. Normalised by total energy the ordering breaks: L-corner **0.0415** outranks
T-junction **0.0391**. With ink held constant, a T and an X give 1.15e5 against 1.16e5 —
indistinguishable.

**The theory says it must be so**, which is why the claim should have been suspect
immediately. A₁ is π-periodic; ray count is 2π; a T and an X have identical orientation
content. What A₁ genuinely separates is one orientation (0.029) from orientations *meeting*
(0.039–0.044) — i2D-ness, not junction order.

**Consequence:** A₁ was structurally the wrong operator for `F`/`f` from the beginning.

## 5.2 The gates measured dense maps, not the features the classifier sees

Phase 3's headline contrasts — 10.3× / 7.1× / 17.6× — were measured on the **dense maps**.
The classifier consumes **pooled** features, and pooling costs about 4×: A₁ 17.6× → **4.9×**,
A₂ 10.4× → **2.6×**.

The gates were therefore validating the *operator* while the pipeline consumes something four
times weaker. The contrasts that survive are still real, but the figure quoted as evidence
overstated what reaches the classifier.

## 5.3 The operator's constants are not scale-free

Covered in §4.11. Both drift with the stimulus scale, and the correct anchor is a measured
structure scale rather than the filter's own envelope.

---

# Part 6 · Predictions made, and which missed

| prediction | outcome |
|:--|:--|
| the conjunction layer adds +0 to +0.5 (original estimate) | **right** — measured +0.01 |
| revised to +0.5 to +1.5 after the confusion analysis | **wrong** — measured −0.12 |
| A₁ orders junctions by ray count | **wrong** — an ink artefact |
| blur explains the synthetic/real gap | **wrong** — pooling costs 4×, blur ~0 |
| the conjunction helps more when data is scarce | **wrong** — flat at every k |
| the conjunction beats the baseline on `F`/`f` alone | **wrong** — 67.5 % vs 69.9 % |
| `d_factor` and `dtheta` are scale-free constants | **wrong** — both drift |
| augmentation helps pixels far more than features | **right** — 5–13 point differential |

**The instructive failure is the second.** The original estimate was correct and reasoned
from the task; it was then revised *upward* on the strength of a confusion analysis showing
that the residual errors were junction-distinguishable. That felt like hard evidence and made
the prediction worse. **"Distinguishable in principle" is not "this feature adds
something"**, because the information can already be present in another form — which is
exactly what the R² measurement later showed.

---

# Part 7 · What is and is not established

## Established

- The front end is **validated on synthetic ground truth** against six criteria fixed in
  advance, with polarity invariance exact and orientation readout accurate to 0°.
- It **reproduces a known result exactly** — 92.30 % against 92.30 % — so its harness is
  trustworthy.
- It is **worth +1.40 points** over the previous design, about 8 standard errors.
- It runs at **20.9 ms per image**, so the full 131,600-image dataset takes about 6 minutes.
- **It carries the invariances it was designed to carry.** Augmentation buys a conventional
  network 9–13 points and buys these features nothing. This is the strongest result here and
  it is independent of everything below.

## Not established

- **That the conjunction layer helps anything.** Three independent lines converge on zero:
  full data (+0.01), finer pooling (recovers 3.1 points of signal, adds nothing, while the
  grid control shows the baseline does not improve either), and sample size (flat from 200
  images to 112,800). A fourth, the direct binary probe, finds it *worse* than the baseline
  on the pair it was designed for.
- **Why it does not help** is now quantified rather than guessed: the conjunctions are 93 %
  linearly recoverable from the orientation statistics on this data, while being provably
  different operators on stimuli designed to separate them.

## Open

1. **Why is `F`/`f` at ~70 % for *every* representation?** The newest and sharpest question.
   All three of orientation statistics, conjunctions and ray harmonics sit at 67–70 % on a
   balanced binary task. Either the information survives in the dense maps and dies in the
   pooling, or the front end lacks it entirely. A dense-readout probe would separate those.
2. **Re-anchor the probe offset to a measured structure scale**, and re-derive the angular
   tuning the same way, so the module stops carrying constants fitted at one operating point.
3. **A task that requires i2D structure.** 54 conjunction numbers alone reach 88.45 %, so the
   signal is real. The negative result is about handwritten characters, which apparently do
   not contain the configurations where orientation statistics fail.

## The honest summary

The front end is good and does what it claims. The specific mechanism it was built
around — representing whether structures meet — works exactly as designed on stimuli that
require it, and is **superfluous on handwritten characters**, because two operations that are
mathematically distinct turn out to be 93 % interchangeable on this particular kind of image.

That is a real result rather than a failure, but it is a result about the dataset as much as
about the design, and it argues for testing the front end on images where i2D structure is
not optional.
