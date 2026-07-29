# Phase 9, explained from scratch

This document assumes no knowledge of the project, of computer vision, or of machine
learning. Every term is defined where it first appears. `RESULTS.md` is the same material
compressed for someone already inside the project.

---

## 1. What the project is trying to build

A **front end** is the first stage of a vision system: the part that turns raw pixels into a
description that later stages can work with. Biological vision has one — the retina and
primary visual cortex do a lot of processing before anything resembling recognition
happens — and the aim here is to build something in that spirit: general-purpose, not
trained for any particular task, and useful for images in general rather than for one
dataset.

The front end in this project takes an image and returns **279 numbers**. Those numbers are
computed by a fixed recipe. Nothing about them is learned, and the recipe was written before
this dataset existed.

The question is: **are those 279 numbers a good description of what is in the picture?**

---

## 2. Why the earlier tests could not answer that

The project spent a long time on EMNIST, a dataset of handwritten letters. The tests took
the form: *can a classifier tell an `A` from a `B` using these numbers?* The trouble is that
the answer is nearly always yes, for almost any numbers you care to compute, provided you
train a big enough classifier on enough examples.

An earlier phase (Phase 8) built a synthetic test where two lines either touched or had a
small gap between them, expecting that telling those apart would require special machinery.
Every method scored 100 %, including the crudest baseline. The test was uninformative — not
because everything is equally good, but because **"can a network eventually learn this?" is
the wrong question**.

## 3. The question this phase asks instead

Not *can something learn it*, but **is it already there?**

The technical version: fit a **linear readout** and see how well it does. A linear readout
is the simplest possible use of a set of numbers — multiply each by a weight, add them up,
that's your answer. No cleverness, no hidden layers. If a single weighted sum of the 279
numbers tells you the angle of a corner in the picture, then the angle is **explicit** in
the representation: available, not merely present.

That distinction is the whole point of a front end. A JPEG file *contains* the corner angle
too, in the sense that the information is not lost — but you can't get at it with a weighted
sum. Making things reachable is the job.

---

## 4. The dataset

Every image is **one stroke on a flat grey background**, 112 × 112 pixels. See
`contactsheet.png` for fifty examples.

Each image is described by **eight numbers** — not a category label, but eight graded
measurements:

| | property | what it means | range |
|--:|:--|:--|:--|
| 1 | `curvedness` | how sharply the stroke curves | 0 (dead straight) → 0.8 |
| 2 | `brokenness` | how big the gap is, relative to how thick the stroke is | 0 (unbroken) → 0.8 |
| 3 | `closedness` | is it a closed loop, like an O | 0 or 1 |
| 4 | `vangle` | the angle at the corner | 32° (sharp V) → 180° (no corner) |
| 5 | `arms` | how many arms meet at a point | 2 (plain line), 3 (a T), 4 (an X) |
| 6 | `thickness` | how thick the stroke is | 3 → 12 pixels |
| 7 | `fuzziness` | how blurry its edges are | 0.8 (crisp) → 20 pixels (very soft) |
| 8 | `polarity` | is the stroke lighter or darker than the background | +1 or −1 |

**Rows 1–5 are geometry.** Rows 6–8 are something else — they are *controls*, and section 9
explains why they are the most interesting rows in the table.

### Why graded numbers rather than categories

If the answer is a category — "this is a right angle" — then an 82° corner guessed as 78° is
scored as a total failure, the same as guessing 20°. With a graded number you can measure
*how close* the answer was, and you can detect systematic biases, such as a method that
squashes every angle toward 90°.

### Everything else is randomised

Position, rotation, contrast, background brightness and image noise all vary from image to
image and are **not** among the eight things to be predicted. They are **nuisances**:
things that change the picture without changing the answer. A good description should be
unaffected by them.

### The dataset is generated, not collected

There is no fixed pool of images. A program draws a random set of parameters and renders the
corresponding picture. This matters for a reason explained in section 7.

---

## 5. What is being compared

Five methods, called **arms**.

First, a distinction. A **representation** is how the picture is encoded; a **readout** is
what is fitted on top of it to produce an answer. The two can be varied independently.

| arm | representation | readout |
|--:|:--|:--|
| 1 | the raw pixels, all 12,544 of them | linear |
| 2 | the raw pixels | a neural network with two hidden layers |
| 3 | a **CNN** — the representation is *learned* | trained end to end |
| 4 | **our 279 features** | **linear** |
| 5 | our 279 features | a neural network with two hidden layers |

A **CNN** (convolutional neural network) is the standard modern approach: rather than being
handed a description, it invents its own by looking at many labelled examples. It is the
strong baseline here.

**Arm 4 is the measurement.** Arm 4 against arm 1 says what our front end made explicit that
wasn't explicit in the pixels. Arm 4 against arm 3 asks whether a fixed, hand-designed
description with the simplest possible readout can match a network that learned its own
description specifically for these eight questions.

### The comparison is deliberately unfair to us

- The CNN **sees the answers** while building its representation. Ours was fixed in advance.
- The pixel readout has **12,544 free parameters**; ours has **279**. 45× fewer.
- Our features use a fixed 3 × 3 spatial grid, so they are **not translation invariant** —
  the same shape in a different place gives different numbers. A CNN gets translation
  tolerance for free from its architecture, and position is randomised in every image.

Every one of those runs against us. That is intentional: a win under these conditions means
something, and a loss is unsurprising.

---

## 6. How the score is defined

The score is **R²** ("R squared"), the fraction of the variation in the true answer that the
prediction accounts for.

- **R² = 1** — perfect.
- **R² = 0** — no better than ignoring the picture and always guessing the average.
- **R² < 0** — *worse* than always guessing the average. This is possible, and it happens in
  the results below. It means the method has learned something confidently wrong.

### Scored against a baseline, not against zero

Some of these properties can be partly guessed from almost nothing at all. `closedness` is
one: a closed loop has more ink than an open arc, so simply measuring total ink gets you
about 46 % of the way. If we reported raw R², every method would be credited for that.

So each table includes a **trivial baseline**: the R² of a linear fit on just three numbers
— total ink, average brightness, and contrast. **A method has demonstrated something only
insofar as it beats that line.** This exists because Phase 8's failure was precisely a cue
nobody had measured turning out to be sufficient.

---

## 7. How training and test sets were built, and why the comparisons are fair

This section is the guarantee that the numbers mean what they appear to mean.

### No overlap is possible between training and test

Test images are generated from a **different random seed**, not carved out of a fixed pool.
Since the generator draws from continuous ranges, no test image has ever existed before.
Duplicate or near-duplicate images across the split aren't merely unlikely — they are
structurally impossible. With a fixed corpus like EMNIST this has to be checked; here it is
guaranteed by construction.

### Every arm sees exactly the same images

Train and test sets are generated **once** and handed to all five arms. Nobody gets an easier
draw. Differences between arms are therefore differences between methods, not between
datasets. The features are also computed once and reused, so the sample-efficiency curve
compares five training-set sizes on identical data.

### Settings are chosen without touching the test set

Every method has settings — how strongly to penalise large weights, how long to train. These
are chosen on a **validation set**: a slice held out of the *training* data, never the test
data. If they were chosen by looking at test performance, the test score would be optimistic
by an unknown amount.

For the neural network arms, the reported prediction is from **whichever training epoch was
best on validation** — so a method that trains well and then degrades is scored where it was
good, rather than being penalised for stopping late.

### Two kinds of split

**i.i.d.** stands for *independent and identically distributed*, and just means training and
test come from the same distribution — the same ranges of thickness, blur, contrast,
polarity and everything else. This is the ordinary way of splitting data, and it mostly
measures **capacity**: how much a method can absorb from 12,000 examples.

**Extrapolation splits** test something else. Here the test set is restricted to nuisance
values the training set never contained:

| split | trained on | tested on |
|:--|:--|:--|
| polarity | light strokes only | **dark strokes only** |
| fuzziness | crisp edges (≤ 3 px) | **blurred edges (≥ 8 px)** |
| thickness | thin strokes (≤ 6 px) | **thick strokes (≥ 8 px)** |

The *geometry* is sampled identically on both sides. Only the way the stroke is drawn
changes. A method that has genuinely extracted shape should be unaffected; a method that
learned "this pattern of pixels means a corner" should collapse.

**One trap, handled.** In the polarity split, every training image is a light stroke — so
`polarity` doesn't vary in training and no method can be scored on predicting it. That row
is dropped from that split, and shown as `—`. Reporting a number there would be meaningless.
(This is subtle enough that it initially caused a bug — see section 11.)

---

## 8. The main result

**i.i.d., 12,000 training images, 3,000 test.** Higher is better; the shaded reading is
"beats the trivial baseline".

| arm | curvedness | brokenness | closedness | vangle | arms |
|:--|--:|--:|--:|--:|--:|
| *trivial baseline* | 0.009 | 0.008 | 0.457 | 0.066 | 0.029 |
| pixels · linear | −0.000 | −0.001 | −0.000 | −0.000 | 0.000 |
| pixels · MLP | −0.183 | −0.404 | 0.721 | −0.137 | 0.108 |
| CNN | 0.449 | 0.254 | 0.947 | 0.291 | **0.874** |
| **ours · linear** | **0.682** | **0.340** | **0.985** | **0.567** | 0.859 |
| ours · MLP | **0.829** | **0.615** | **0.992** | **0.831** | **0.916** |

Read the `vangle` column — the angle of a corner. Raw pixels: nothing. A large neural network
on raw pixels: worse than nothing. A CNN that learned its own features for this exact task:
0.291. **Our fixed features with a plain weighted sum: 0.567.**

The CNN wins one of the five, `arms` — how many branches meet at a point — by 0.874 to 0.859.
With a neural network reading our features instead of a weighted sum we lead there too
(0.916). An earlier version of this experiment used a weaker CNN and we beat it on all five;
that was not a fair fight, and the numbers above are from a proper one.

**Raw pixels scoring exactly zero is the expected result, not a bug.** A linear readout on
pixels is a fixed template — it asks "is the image bright *here* and dark *there*?" Since
every shape appears at a random position and angle, no fixed template can work. The pixels
contain the shape; they just don't make it available.

And the check that the machinery works: those same pixels score **0.709 on polarity**,
because average brightness *is* a weighted sum of pixels. The method isn't broken; the
information simply isn't reachable that way.

---

## 9. Why `polarity` is the most interesting row

`polarity` is whether the stroke is lighter or darker than its background. Our front end
scores **−0.000** on it — it cannot tell at all.

**This is the correct answer, and it was predicted in advance.**

The front end measures *energy* — how much edge-like structure is present at each place,
orientation and scale — and energy is unchanged if you invert the picture. A white line on
black and a black line on white produce identical numbers. That is **polarity invariance**,
and it is a design requirement of the project, because a shape is the same shape whichever
way its contrast runs.

So the interesting comparison is not who scores highest but **who scores what they should**:

| | should it know? | pixels | CNN | ours |
|:--|:--|--:|--:|--:|
| `polarity` | no — invariance is the goal | 0.709 | 0.949 | **−0.000** |

And the payoff shows up in the extrapolation split. **Trained on light strokes, tested on
dark:**

| arm | curvedness | closedness | vangle | arms |
|:--|--:|--:|--:|--:|
| pixels · linear | −0.130 | −2.352 | −0.532 | −1.773 |
| pixels · MLP | −1.806 | **−6.042** | −5.117 | −3.973 |
| **ours · linear** | **0.682** | **0.985** | **0.570** | **0.857** |

Our numbers are **identical to the i.i.d. row to three decimal places**. The methods that
knew about polarity are destroyed by inverting it — scoring far below simply guessing the
average — while the method that was blind to it never noticed the change.

That is the argument for invariance stated as a measurement: **information you refuse to
represent cannot mislead you later.**

---

## 10. Does the conjunction layer earn its place?

Our 279 numbers are not homogeneous. They come in blocks:

- **`orient`** (135) — conventional oriented-energy statistics. How much edge energy at each
  place, orientation and scale. This is roughly what standard approaches compute.
- **`A1`, `A2`** (54) — the **conjunction layer**. Designed to detect where two orientations
  *meet at a point* rather than merely both being present nearby.
- **`rays`** (81) — counts how many contour branches radiate from a point.
- **`lowpass`** (9) — coarse average brightness.

The conjunction layer has been the project's open question for months. On EMNIST it was
worth **+0.01** — nothing.

Here, adding it to `orient`:

| | `orient` alone | everything | apparent gain |
|:--|--:|--:|--:|
| `brokenness` | 0.159 | 0.340 | +0.181 |
| `vangle` | 0.394 | 0.567 | +0.173 |
| `arms` | 0.667 | 0.859 | +0.192 |

### But that comparison is not yet honest

`orient` alone is 135 numbers; everything is 279. A method with twice as many numbers can
look better purely from having more knobs, regardless of whether the extra numbers mean
anything. **This exact mistake has produced a false positive three times in this project.**

The control is a **shuffle test**. Take the full 279 numbers, but scramble the conjunction
and ray columns *across images* — so image 1 gets image 800's conjunction numbers. The
column count is identical, the range of values is identical, and only the correspondence
between those numbers and the picture is destroyed. Anything the shuffled version scores
above `orient` is what extra knobs buy on their own.

| | `orient` | **shuffled** | everything | **real gain** |
|:--|--:|--:|--:|--:|
| `brokenness` | 0.159 | 0.174 | 0.340 | **+0.166** |
| `vangle` | 0.394 | 0.403 | 0.567 | **+0.164** |
| `arms` | 0.667 | 0.727 | 0.859 | **+0.132** |
| `closedness` | 0.949 | 0.953 | 0.985 | +0.032 |

**The shuffled version lands on top of `orient`.** The extra knobs buy almost nothing, so
the gain is real information.

**This is the first time the conjunction layer has paid for itself anywhere in the project.**

And it pays where the theory said it would. `vangle` (corner angle) and `arms` (how many
branches meet) are exactly the properties that *cannot* be computed from a list of how much
energy there is at each orientation — you need to know whether the orientations meet at a
point. Those two rows carry the biggest gains. `closedness` — whether a curve eventually
returns to where it started, a question about the whole figure rather than any point —
gains almost nothing, which is also what you would expect.

---

## 10a. One column that does not mean what it says

`closedness` — is the stroke a closed loop like an O — scores very high for everything:
0.985 for our features, 0.929 for the CNN, and even the trivial three-number baseline gets
0.457, the highest baseline of the eight properties. That is suspicious, and on
investigation it is not measuring closure at all.

Three simpler things happen to go along with being a closed loop in this dataset:

| what it could be reading instead | open strokes | closed loops | how much this alone explains |
|:--|--:|--:|--:|
| **how long the contour is** | 77 px | 245 px | **0.898** |
| **how many different directions it contains** | anisotropic | isotropic | 0.490 |
| how tightly it curves | gentle to sharp | always sharp | some |

The lengths do not even overlap: open strokes run 68–90 px, closed loops 225–283 px. Simply
asking "is this contour longer than 150 pixels?" would label every image correctly.

And crucially, **neither cue requires tracing the contour**. Total length is just the total
amount of ink, adjusted for how thick and dark the stroke is — a sum, not a journey. And
"how many directions does it contain" is a histogram of local edge orientations: a closed
loop must turn all the way round, so it contains every direction, while an open arc here
turns at most 120° and so leaves a third of the directions empty.

**Why this happens is geometry, not a bug.** In a box of a given size, a closed contour can
be about π times the box width long, while an open arc that doesn't curl up much is limited
to roughly 1.2 times the box width. Closing a curve *buys* you length. You cannot put a long
open contour in a small frame without making it wiggle or spiral.

**The honest reading:** the `closedness` column should be treated as "can you tell a long,
direction-rich contour from a short, direction-poor one" — which everything can, including
the trivial baseline. It is not evidence that anything detects closure. **The other seven
columns are unaffected.**

Fixing it properly would mean adding spirals and serpentines — contours that are long and
full of directions but still have two loose ends — so that only genuine closure separates
the two cases. That was considered and rejected: it would turn clean single strokes into
scribbles, and the question is not central enough to be worth changing the dataset for.

---

## 11. How many training images does it take?

![sample efficiency](phase9_samples.png)

`ours · linear`, as the training set grows:

| training images | curvedness | brokenness | closedness | vangle | arms |
|--:|--:|--:|--:|--:|--:|
| 500 | 0.617 | 0.122 | 0.971 | 0.360 | 0.782 |
| 2,000 | 0.648 | 0.283 | 0.982 | 0.486 | 0.833 |
| 6,000 | 0.674 | 0.325 | 0.985 | 0.548 | 0.855 |
| 12,000 | 0.682 | 0.340 | 0.985 | 0.567 | 0.859 |
| **fraction of final score reached at 500** | **90 %** | 36 % | **99 %** | 64 % | **91 %** |

With **500 images** — a few minutes of drawing — the front end is at 90 % of what it reaches
with 12,000, for three of the five geometric properties.

The CNN — which builds its own description from the examples — over the same subsets:

| training images | curvedness | brokenness | closedness | vangle | arms |
|--:|--:|--:|--:|--:|--:|
| 500 | 0.030 | −0.003 | 0.664 | 0.012 | 0.089 |
| 2,000 | 0.038 | −0.031 | 0.807 | 0.049 | 0.250 |
| 6,000 | 0.116 | −0.022 | 0.899 | 0.119 | 0.504 |
| 12,000 | 0.235 | −0.002 | 0.929 | 0.155 | 0.650 |

**This is the clearest way to see what a front end buys you.** With 500 images, our fixed
description already answers "how many arms meet here" at 0.782, while the CNN manages 0.089
— roughly nine times worse. By 12,000 images the CNN has reached 0.650 and is still
improving. It is *learning* what we *supplied*.

Being fair to the CNN: two of its curves are still rising steeply at the right-hand edge, so
with much more data it would probably catch up on some properties. But note `brokenness` —
whether the stroke has a gap in it. The CNN sits at zero for every training-set size tried.
A 3-pixel gap in a 112-pixel image is apparently too fine for it to pick up at all, while our
description reaches 0.340.

Raw pixels sit at 0.000 for every property at every size. **They do not improve with more
data**, which is the sharpest statement of what a front end is for: extra examples cannot
recover information the representation never made available.

The two rows still climbing at 12,000 — `brokenness` and `vangle` — are the finest-grained
of the eight, and are the ones most likely limited by the coarseness of the spatial grid
(see caveats).

---

## 12. A bug this dataset found in the front end itself

This is arguably the most valuable outcome, and it is a correction to a claim the project
has been making for months.

The very first run showed the `orient` block predicting **polarity at R² 0.65** — when it
should have scored zero, as explained in section 9. Rendering the same shape twice, once
light and once dark, showed the numbers differing by **29 %**. **The front end was not
polarity invariant.**

The cause: the front end pads the image with zeros before filtering (a routine step, to
avoid edge effects). On EMNIST the background *is* zero — black — so the padding blended in
seamlessly. Here the background is mid-grey, so padding with zeros surrounds the picture with
a hard black border, and that artificial border interacts with the stroke differently
depending on whether the stroke is lighter or darker than the background.

The fix is one line: subtract the background level before filtering, so the padding matches.
Invariance is now exact to the precision of the arithmetic — **0.00000024** instead of 29 %.

Removing that spurious border also improved everything else, because it had been injecting
noise into every number: on 1,000 training images, `closedness` went 0.63 → 0.93 and
`vangle` 0.04 → 0.38.

**This could not have been found on EMNIST**, whose black background hides it exactly. It is
the clearest possible argument for testing a general-purpose front end on something that is
not handwritten characters.

### A prediction that was wrong

Before the run, on record: the `orient` block should score ~0 on polarity, and the `lowpass`
block should score ~1, since lowpass carries average brightness. The first half was right.
The second was **wrong**: `lowpass` is also a magnitude, and magnitudes discard sign, so it
is equally blind. The representation throws contrast sign away *completely* — cleaner than
claimed, but it does mean nothing downstream can ever recover it.

### And a bug in the experiment, not the front end

In the polarity split, both neural-network arms scored a flat 0.000 on everything, which
looked like three independent collapses. It was one bug: when a held-out property has no
variation, its R² is undefined (`NaN`, "not a number"), and the rule for choosing the best
training epoch averaged over all properties — so the average became undefined, the comparison
"is this epoch better?" was always false, and no prediction was ever recorded. Fixed; the
corrected numbers appear in section 9, and they are considerably worse for the pixel methods
than the buggy zeros were.

---

## 13. What these results do and do not show

**They show:**

- The 279 numbers make five geometric properties linearly available where raw pixels make
  none of them available at any training-set size.
- That advantage survives being tested on stroke appearances never seen in training, where
  pixel-based methods fall below chance.
- The conjunction layer contributes real information — not extra parameters — on precisely
  the properties it was designed for.
- The front end genuinely discards contrast polarity, now that a padding bug that broke this
  has been fixed.

**They do not show:**

- That our front end beats a *well-trained* CNN. Ours was trained for 12 epochs on a CPU and
  its score was still improving when it stopped. It is a floor, not a ceiling.
- That the advantage carries to natural images. These are synthetic single strokes on flat
  backgrounds.
- That the gaps are statistically solid at the fine end. One random seed per arm, no error
  bars: read a difference of 0.4 as real, and a difference of 0.02 as noise.
- Anything about closure. The `closedness` column is confounded (section 10a) and every
  number in it should be discounted.
- That 3 × 3 pooling is the right choice. `brokenness` in particular may be limited by it —
  a 3-pixel gap inside a 37-pixel cell — rather than by the operators.

---

## 14. Where to look next

1. Repeat the shuffle control over several random permutations, rather than one.
2. Try a finer 5 × 5 spatial grid, to separate "the operators can't do it" from "the pooling
   is too coarse".
3. Train the CNN to convergence, ideally on a GPU, so the comparison is against a real
   ceiling rather than a compute limit.
4. Re-check the EMNIST results with the polarity fix in place. The bug should be absent
   there — but "should be" is not "was measured".
