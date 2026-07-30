# Phase 10 — Fashion-MNIST

The first dataset in this project that is not a line drawing. Filled silhouettes with texture
rather than strokes on an empty field, and **published baselines** to calibrate against instead
of only our own arms. See `README.md` here for the benchmark literature and its caveats.

60,000 train / 10,000 test, 10 classes, chance 10.0 %. Features extracted at 112×112 after
bilinear upsampling from 28×28, exactly as the EMNIST phases do. One hidden layer of 256, best
epoch chosen on a validation slice carved out of training.

---

## Results

| arm | features | accuracy |
|:--|--:|--:|
| **pixels + MLP** *(calibration)* | 12,544 | **87.70 %** |
| | | |
| **grid 3** — orient+lowpass | 144 | 89.03 % |
| grid 3 — + A₁+A₂ | 198 | **89.70 %** |
| grid 3 — + rays | 225 | 89.57 % |
| grid 3 — everything | 279 | 89.66 % |
| grid 3 — A₁+A₂ alone | 54 | 86.51 % |
| grid 3 — rays alone | 81 | 86.26 % |
| | | |
| **grid 1** — orient+lowpass | 16 | 78.31 % |
| grid 1 — + A₁+A₂ | 22 | 80.69 % |
| grid 1 — everything | 31 | 81.63 % |
| grid 1 — A₁+A₂ alone | 6 | 55.39 % |
| grid 1 — rays alone | 9 | 65.20 % |

Published for comparison: linear on pixels 85.2 %, MLP ≈ 88 %, the CNN cluster that replicates
93.7–94.6 %, hybrids 95–96.6 %.

**The calibration arm validates the harness.** Our own pixels+MLP gives 87.70 % against a
published ≈ 88 %, so the pipeline is not quietly losing or gaining points somewhere.

---

## The three predictions

Recorded in `README.md` before the run.

### 1. Features between the published MLP and CNN numbers, ≈ 88–91 % — **correct**

**89.66 %** at grid 3, from 279 numbers with no learned parameters in the representation. Above
the published MLP on 12,544 pixels, below a good CNN by about four points.

Worth noting what is doing the work: **A₁+A₂ alone reach 86.51 % from 54 numbers**, and the ray
block alone 86.26 % from 81. Neither was designed for texture.

### 2. The AND layer adds ≈ 0, as on EMNIST — **wrong, though weakly**

Adding A₁+A₂ to `orient+lowpass` is worth **+0.67** at grid 3 and **+2.38** at grid 1. On
EMNIST the same addition was worth +0.06.

**This does not yet count as a result.** There is no shuffle control here — `+A₁+A₂` is also
54 more columns, and Phase 5b showed on EMNIST that adding 54 *shuffled* columns costs −0.75,
so extra columns are not free but the sign of their effect depends on the model's capacity
relative to the data. With 60,000 training images the model is not data-starved, so a capacity
gain is plausible. **The shuffled twin has to be run before +0.67 is attributed to conjunction.**

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

## What this does and does not establish

**Establishes:** the front end works off line drawings. Multi-scale oriented energy is a usable
texture descriptor — 89.7 % on a texture-and-silhouette task, beating an MLP on raw pixels from
45× fewer numbers, with nothing in the representation fitted to the data.

**Does not establish** anything about the design questions left open by Phase 9. The background
is black and uniform, there is one centred object, contrast barely varies within a frame, and
the resolution is 28×28 upsampled. **Divisive normalisation, whose whole purpose is handling
spatially varying local contrast, still has no test here.** That needs natural greyscale images;
BSDS boundary detection remains the intended target.

## Open

1. **The shuffle control**, without which the +0.67 from A₁+A₂ is not attributable.
2. **A CNN arm** trained here, rather than comparing against published numbers gathered under
   unknown conditions — the literature review's own figures span 90.1–99.18 % and several are
   not credible.
3. **The near-duplicate correction.** About 6 % of the test set are near-duplicates of training
   images (arXiv:1906.08255), so every number on this page, ours included, is slightly
   optimistic.
