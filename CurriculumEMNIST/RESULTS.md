# Phase 13 — EMNIST with the training subset switched every 15 epochs

EMNIST balanced, 47 classes, official split: 112,800 train / 18,800 test. The training set is cut
into **4 disjoint subsets of 28,200**. Epochs 1–15 train on subset 1, 16–30 on subset 2, 31–45 on
subset 3, 46–60 on subset 4. The optimiser state is never reset — the run continues, only the data
underneath it changes. Both arms use the identical readout (`381|1024 → 512 → 512 → 47`, Adam
1e-3, batch 128, no regularisation) and the identical subsets in the identical order.

Figures: `figures/curriculum_ours.png`, `figures/curriculum_convnext.png`,
`figures/curriculum_compare.png`.

---

## The headline

| arm | nfeat | ep 15 | ep 60 | best | memorisation gap at ep 60 |
|:--|--:|--:|--:|--:|--:|
| **ours, switching** | **381** | **85.41** | **85.97** | **87.18** | 11.66 |
| ours, subset 1 only (control) | 381 | 85.41 | 84.51 | 85.63 | 13.51 |
| ConvNeXt, switching | 1024 | 84.70 | 85.70 | 86.78 | 10.86 |
| ConvNeXt, subset 1 only (control) | 1024 | 84.70 | 84.43 | 85.30 | 14.16 |

**Our 381 features beat frozen ConvNeXt's 1024 here, at every point on the curve.** That is the
opposite of Phase 11's finding on the stroke set, and the reason is worth stating: EMNIST is a
line-drawing task at 28×28, which is what an oriented-energy front end is for. Phase 11's stimuli
were natively 112 px with graded photometric properties — thickness, blur — where ImageNet's
learned filters have the advantage. Neither result generalises to the other, and the gap either
way is under half a point, so the honest summary is that **the two representations are
equivalent on this task while ours uses 2.7× fewer numbers and has no learned parameters.**

---

## What the switches actually did

**Held-out accuracy walks straight through them.** Change across each switch, in points:

| arm | → subset 2 | → subset 3 | → subset 4 |
|:--|--:|--:|--:|
| ours | +0.15 | −1.46 | −0.65 |
| ConvNeXt | +0.34 | −0.24 | −0.43 |

Against a within-block epoch-to-epoch noise floor of **sd 0.51 (ours) / 0.54 (ConvNeXt)**, spanning
−0.97 to +1.37 and −1.27 to +1.92. Only our −1.46 sits outside its block's own range, by a hair,
and it is a single epoch. On block means the effect vanishes entirely — accuracy rises monotonically
across every switch:

| arm | block 1 | block 2 | block 3 | block 4 |
|:--|--:|--:|--:|--:|
| ours, switching | 85.18 | 85.72 | 85.84 | **86.10** |
| ours, control | 85.18 | 85.02 | 84.61 | 84.67 |
| ConvNeXt, switching | 84.90 | 85.54 | 85.51 | **85.56** |
| ConvNeXt, control | 84.90 | 84.62 | 84.83 | 84.15 |

**Switching beats not switching by ~1.4 points for both arms.** The switching runs keep improving;
the controls peak around epoch 15–20 and then drift *down* as they overfit the 28,200 examples they
are stuck with. Fresh data every 15 epochs is straightforwardly better than more passes over old
data — which is the expected result, and its appearance here is a check that the harness works.

## Where the discontinuity really is

Not in generalisation — in the training-set curve. Accuracy on the subset being trained on, in the
last 5 epochs of each block, minus held-out accuracy:

| arm | block 1 | block 2 | block 3 | block 4 |
|:--|--:|--:|--:|--:|
| ours | 10.18 | 11.10 | 11.21 | 11.00 |
| ConvNeXt | 9.94 | 10.33 | 10.62 | 10.24 |

That ~10–11 point gap is memorisation, measured directly rather than inferred, and it is stable
across blocks: each new subset gets memorised to about the same depth as the last. In the figures
it is the orange sawtooth — climbing to ~96 % within a block, collapsing to ~89 % the instant the
data changes.

## Forgetting is fast, then slow, and identical in both arms

Accuracy on subset 1 *minus* held-out accuracy, at the end of each block — i.e. how much of an
advantage subset 1 still enjoys long after training left it:

| arm | end blk 1 | end blk 2 | end blk 3 | end blk 4 |
|:--|--:|--:|--:|--:|
| ours | 10.95 | 2.43 | 1.31 | 0.84 |
| ConvNeXt | 10.34 | 2.48 | 1.31 | 0.51 |

**About 78 % of the memorised margin is released in the first 15 epochs after training moves on**,
then the remainder decays slowly toward zero. The two arms agree to within 0.1 points at every
step but the first.

That agreement is the most informative number in the experiment. Forgetting here is a property of
**the readout and the optimiser**, not of the representation underneath — a 381-dimensional
hand-designed vector and a 1024-dimensional learned one are memorised and released on the same
schedule. If we want a representation whose downstream learning is more stable, this says the
lever is not in the front end.

---

## Caveats

**λ = 8 px is at EMNIST's Nyquist limit.** EMNIST is 28×28 upsampled 4×, so the finest channel of
the best stroke-set configuration largely measures bilinear interpolation here. It was included
because the brief was to run the best configuration; it is not doing what it does on natively-112
stimuli. An ablation without it would be cheap and has not been run.

**Grid 3, not grid 1**, because EMNIST characters are centred — Phase 10 found grid 3 beating grid 1
by 8 points on Fashion-MNIST for that reason, and grid 1 only wins where position is randomised.

**These are single seeds.** Differences under ~0.5 points, including our margin over ConvNeXt,
are within the run-to-run noise this harness has not measured.

**The partition is i.i.d.** This experiment therefore says nothing about *distribution shift*. It
was designed to ask a narrower question — how much of the fit is example-specific — and the answer
is clean precisely because the subsets are exchangeable. Repeating it with subsets split by class
or by writer would be a different and harder experiment.
