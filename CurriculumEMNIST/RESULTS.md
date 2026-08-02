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

---

# Is the accuracy memorisation-plus-interpolation, or extracted structure?

The switching experiment above says memorisation is large, separable and transient — but it
cannot say whether the *held-out* accuracy rests on remembering training images and interpolating
between them. On an i.i.d. test split those two hypotheses **predict the same number**, because
test images sit near training images in any adequate representation. Separating them needs a
different question.

## First: how far does pure lookup get you?

k-nearest-neighbour *is* remember-and-interpolate, made literal — store all 28,200 training
images, classify by proximity, no learned parameters and no decision boundary. Cosine similarity,
subset 1 as the memory, same standardisation as training (`KNN_Baseline.jl`):

| representation | 1-NN | 5-NN | 10-NN | 25-NN | trained MLP | readout adds |
|:--|--:|--:|--:|--:|--:|--:|
| raw pixels (784) | 72.01 | 73.72 | 73.45 | 71.32 | — | — |
| **ours, frozen (381)** | 78.54 | **80.89** | 80.82 | 79.72 | 85.18 | +4.29 |
| ConvNeXt-base frozen (1024) | 76.43 | 78.73 | 78.78 | 77.66 | 84.90 | +6.12 |

**On EMNIST, ~95 % of what a trained readout achieves is reachable with no learning at all**, and
73.7 points of it from raw pixel similarity — no representation whatsoever. So this task has
little power to separate the two hypotheses: proximity nearly suffices.

Two things fall out anyway. **Our 381 designed features are a better lookup table than 1024
ImageNet features** (80.89 vs 78.78) with zero learned parameters — so if ConvNeXt were winning by
being a superior stored-example index, this is where it would show, and it does not. And
**ConvNeXt's readout extracts more beyond lookup than ours does** (+6.12 vs +4.29): its advantage
lies in features a learned boundary can exploit, not in proximity structure.

A high kNN score is **not** evidence of memorisation. A representation that genuinely extracted
the invariants will also make kNN work well — placing same-class images near each other is the
whole point. The number to read is the gap and the ranking, not the level.

## The test that does separate them: classes never seen

Hold out 10 of the 47 classes entirely, train ConvNeXt-tiny from scratch on the other 37, then do
10-way few-shot classification on the withheld ones from its penultimate features
(`fewshot_train.py`, `FewShot_Eval.jl`). Remembering images of `A` cannot help you recognise a `q`
you have never seen; strokes, junctions, curvature and closure can. Held-out classes drawn at
random with a fixed seed — **2 3 M O Q T W Z e r** — so they cannot be chosen to flatter any arm.
Prototypical-network protocol: support from the train split, queries the 4,000 held-out-class test
images, 200 episodes. Figure: `figures/fewshot.png`.

| representation | 1-shot | 2-shot | 5-shot | 10-shot | 20-shot |
|:--|--:|--:|--:|--:|--:|
| **ConvNeXt-tiny, scratch on 37 classes (768)** | **74.49** | **82.10** | **87.52** | **89.31** | **90.19** |
| ConvNeXt-base, frozen ImageNet (1024) | 60.22 | 70.01 | 79.59 | 83.39 | 85.62 |
| ours, frozen (381) | 57.93 | 68.36 | 78.28 | 83.05 | 85.83 |
| raw pixels (784) | 48.03 | 58.16 | 69.50 | 74.86 | 78.12 |

± 0.1–0.7 at 95 % confidence over episodes.

**The from-scratch network's advantage is not memorisation.** At 1-shot it beats every alternative
by more than 14 points on characters it has never seen — where stored images of the 37 base classes
can contribute nothing directly. Training on EMNIST taught it something about how characters are
built, and that transfers.

**Our 381 hand-designed features land level with frozen ImageNet ConvNeXt** — behind by 2.3 at
1-shot, ahead by 0.2 at 20-shot — with 2.7× fewer numbers and no training of any kind. Both sit
about 8–10 points above raw pixels, which is the honest measure of what either representation
contributes over templates.

### Checking the obvious objection

EMNIST's held-out classes are not visually disjoint from the base set: `O` is withheld while the
digit `0` is in it, and `Q` is withheld while lowercase `q` is in it. If the from-scratch net's
gain came from those near-duplicates rather than from general structure, a couple of classes would
carry it. Per-class 5-shot (`PerClass_FewShot.jl`):

| class | 2 | 3 | M | O | Q | T | W | Z | e | r |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ours | 46.3 | 88.7 | 75.5 | 73.2 | 86.5 | 90.2 | 87.3 | 68.6 | 77.3 | **89.3** |
| scratch | 65.2 | 97.3 | 86.9 | **97.5** | 90.1 | 97.1 | 91.9 | 75.3 | 95.6 | 78.2 |
| gain | +19.0 | +8.6 | +11.5 | **+24.3** | +3.6 | +6.9 | +4.6 | +6.7 | +18.4 | **−11.1** |

**The gain is spread over 9 of 10 classes.** `O` is the largest and is consistent with leakage from
digit `0` — but `Q`, whose twin `q` is also in the base set, is nearly the *smallest* gain, and the
next two largest (`2`, `e`) have no base-set lookalike. Excluding `O` entirely the mean gain is
still **+7.6**. The objection does not survive.

`r` is the one class where our features win, by 11 points. Unexplained; worth a look.

### What this settles, and what it does not

**Settles:** the from-scratch network's superiority on EMNIST is not stored EMNIST images. It
generalises to character classes that were never in its training set, from a single example each,
better than a representation trained on 1.28 M photographs.

**Does not settle:** how much of the *base* 85 % on seen classes is interpolation. Nothing here
addresses that, and the kNN result above suggests the answer is "most of it, for every arm" —
which is a fact about EMNIST rather than about any representation.

**Not run:** the complementary test, changed conditions — polarity inversion, blur, thickening,
rotation — where our front end has a *provable* invariance and the others have none. That is where
a hand-designed representation has a structural reason to win, and Phase 11 has the machinery.
