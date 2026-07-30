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

## Predictions, recorded before the first full run

1. The features land between the published MLP and CNN numbers, **≈ 88–91 %**, because texture
   suits multi-scale oriented energy.
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
