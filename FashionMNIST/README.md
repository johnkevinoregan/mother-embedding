# FashionMNIST — the front end on something that is not a line drawing

Every dataset this project has used is a line drawing on an empty background: EMNIST letters,
synthetic single strokes. Fashion-MNIST is the cheapest step away — same 28×28 IDX format, so
the reader is reused unchanged, but the content is **filled silhouettes with texture** (weave,
ribbing, sole tread). Multi-scale oriented energy is a texture descriptor and that has never
been tested here.

It also brings something the synthetic data cannot: **published baselines**. Roughly 84 % for
a linear model on pixels, 88 % for an MLP, 93 % for a good CNN, 96 % SOTA. So the result is
calibrated against an external scale rather than only against our own arms.

## Files

| file | kind | opens in Pluto? |
|:--|:--|:--|
| `Explore_FashionMNIST.jl` | **Pluto notebook** — every stage as dense heatmaps | **yes** |
| `Phase10_FashionMNIST.jl` | plain script — the classification experiment | no |

## The notebook

```bash
cd ~/claude-code/mother-embedding
julia --project=. -e 'using Pluto; Pluto.run()'
```

then open `FashionMNIST/Explore_FashionMNIST.jl`. Sliders choose the garment class and the
instance; the panels show, for that image, the oriented energy at every orientation of a
chosen scale, then `A₁`, `A₂`, `c₀` and `|c₁|/c₀` at all three scales, then the pooled feature
vector with the grid as a slider so you can watch spatial detail traded for a shorter
description.

Things worth looking for: whether the oriented energy picks up fabric texture as well as the
silhouette; whether `A₂` fires at hems, cuffs and strap ends; and whether `c₀` rises where
straps meet bags.

## The experiment

```bash
cd ~/claude-code/mother-embedding/FashionMNIST
julia --project=.. -t 16 Phase10_FashionMNIST.jl 2>&1 | tee phase10.log
```

| variable | default | |
|:--|:--|:--|
| `F_NTRAIN` / `F_NTEST` | 0 | 0 = all 60,000 / 10,000 |
| `F_EPOCHS` | 25 | |
| `F_GRIDS` | `1,3` | pooling grids to compare |

## Two traps in reusing the EMNIST reader

Both were found by checking rather than assuming, and both are handled in `load_split`.

**The reader un-transposes.** EMNIST stores its images transposed relative to MNIST, so
`read_emnist_images` corrects for it — which *introduces* a transpose here. Measured before
fixing: trousers came out 11.8 px tall and 27.9 px wide, lying on their side. Classification
would largely have survived that; the prediction below about the pooling grid would not have.

**It already returns 1-based labels.** Adding one more put them out of range.

## Published benchmarks — what to score against

From the literature review by Bbouzidi, Hcini, Jdey and Drira, *Convolutional Neural Networks
and Vision Transformers for Fashion MNIST Classification: A Literature Review*
([arXiv:2406.03478](https://arxiv.org/abs/2406.03478)), which tabulates results across a large
number of papers:

| family | reported range | the cluster that replicates |
|:--|:--|:--|
| classical / linear | ~85 % | linear classifier **85.17 %** |
| CNNs | 90.1 – 99.18 % | **93.7 / 94.04 / 94.11 / 94.62** |
| Vision Transformers | 87.3 – 95.25 % | **92.6 / 92.6 / 92.71 / 93.57** |
| hybrids | 95.0 – 96.56 % | 95 – 96.6 |

**Score against the middle column, not the maxima.** Several of the headline figures are not
credible for this dataset: Zalando's own benchmark tops out near 96.7 % with heavy
augmentation, so `CNN-dropout-3` at 99.1 % and `CNNTuner` at 99.18 % sit ~2.5 points above the
best independently reproduced result. `LeNet` at 98.4 % is almost certainly wrong — LeNet-5 on
Fashion-MNIST is normally 89–91 %, and the same table has a modern CNN at 93.7 %, so a 1998
architecture is shown beating it by 4.7 points without comment. The review aggregates claims
rather than auditing them.

**Part of the inflation is measurable.** *Training on test data: Removing near duplicates in
Fashion-MNIST* ([arXiv:1906.08255](https://arxiv.org/abs/1906.08255)) found that **≈ 5.98 % of
the 10,000 test images are near-duplicates of training images** — matched by CNN feature
distance, then verified by a human against explicit criteria (outlines 90 % similar, differing
by at most one feature such as buttons or print). Removing them costs about 0.4 points, e.g.
Random Forest 84.4 % → 84.0 %. Small, but it means every number here is slightly optimistic,
and comparisons *between* published numbers inherit it unevenly.

**There is no human baseline.** No study appears to have measured human accuracy on
Fashion-MNIST, which is a pity: the shirt / T-shirt / pullover / coat cluster is where models
lose most of their remaining accuracy, and whether people find those hard too would say
whether the ceiling is perceptual or an artefact of 28×28.

**One observation from the tables that bears on this project.** Vision Transformers come in
*below* CNNs — 92.6–93.6 against 93.7–94.6 — on 60,000 small images. That is the
inductive-bias argument one rung down from the ConvNeXt and NFNet results: a weaker prior
costs accuracy when data is limited, and 60k 28×28 images is a small-data regime by modern
standards. Consistent with the claim that a designed front end should help most at small `n`.

> **Run complete — see `RESULTS.md`.** Features reach **89.70 %** at grid 3 from 198 numbers;
> the calibration arm lands at 87.70 % against a published ≈ 88 % and a CNN trained here at
> 93.10 %. **Grid 3 beats grid 1 by 8 points**, the reverse of Phase 9, which settles that
> earlier result as being about position randomisation. Prediction 2 was wrong: against a
> 5-permutation shuffle control the AND layer is worth **+1.62 at grid 3 and +2.85 at grid 1**
> (σ ≤ 0.27), where EMNIST gave +0.01 — the first time the conjunction layer has paid on a
> dataset this project did not construct.

## Predictions, recorded before the first full run

1. The features land between the published MLP (~88 %) and the replicating CNN cluster
   (~94 %), so **≈ 88–91 %**, because texture suits multi-scale oriented energy.
2. **The AND layer adds ≈ 0**, as on EMNIST — silhouettes contain few junctions.
3. **Grid 3 beats grid 1**, the reverse of `SimpleStrokeTests`. Grid 1 won there only because
   position was randomised, making a fixed grid pure liability; here garments are centred with
   parts in consistent places. If grid 1 wins anyway, that earlier result was about pooling in
   general rather than about position randomisation, which would change how it should be read.

## What this still does not test

Black uniform background, one centred object, contrast barely varying within a frame, 28×28
upsampled to 112. Those are exactly the conditions under which **divisive normalisation**
would matter, so that design question — see `SimpleStrokeTests/RESULTS.md` — remains open
after this. Natural greyscale images are needed for it; BSDS boundary detection is the
intended target.
