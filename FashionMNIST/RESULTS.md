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

**And it points at a specific test.** Phase 7 explained the EMNIST null with
`R²(A ← orient) = 0.933` — A and orientation energy near-collinear *on handwriting*. If that is
the explanation, then the same regression on Fashion-MNIST should give a substantially lower
R², and the two numbers together would account for the whole difference between +0.01 and
+1.62. That is a cheap run on features already cached, and it is the first item under *Open*.

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

**In `SimpleStrokeTests` position is randomised**, so a fixed spatial grid is pure liability —
the same shape at a different location produces different numbers and the readout must learn to
undo it. **Here garments are centred with their parts in consistent places** — sleeves up, soles
down, straps at the top of a bag — so the grid carries real information.

So the Phase 9 grid result was about **position randomisation in that dataset**, not about
pooling in general, and the caution attached to it in `SimpleStrokeTests/RESULTS.md` was
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

1. **`R²(A ← orient)` on Fashion-MNIST**, against Phase 7's 0.933 on EMNIST. If it is markedly
   lower, the collinearity explanation accounts for the whole +0.01 → +1.62 difference. The
   features are cached, so this is minutes.
2. **An i1D response gate on A₁.** Zetzsche & Barth (Vision Research, 1990) prove no linear
   filter can be i2D-selective, and give the design criterion: the quadratic kernel must vanish
   on collinear frequency pairs. A₁ should therefore read ≈ 0 on a straight line at *every*
   orientation, and it will not be exactly 0 because orientation channels have bandwidth. If it
   leaks, then A₁ is partly a function of orientation energy by construction, and Phase 7's
   0.933 is a fact about our implementation rather than about handwriting. This bears directly
   on item 1.
3. **The near-duplicate correction.** About 6 % of the test set are near-duplicates of training
   images (arXiv:1906.08255), so every number on this page, ours included, is slightly
   optimistic. It costs ~0.4 points where it has been measured, and it applies to all arms, so
   no comparison here is affected.
