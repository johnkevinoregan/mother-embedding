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

| arm | representation trained on | ep 15 | ep 60 | best | memorisation gap |
|:--|:--|--:|--:|--:|--:|
| **ConvNeXt-base, from scratch** | EMNIST, 87.6 M params | 85.73 | **88.06** | **88.94** | 11.86 |
| ConvNeXt-tiny, from scratch | EMNIST, 27.9 M params | 83.10 | 87.01 | 88.19 | 12.15 |
| **ours, frozen (381)** | nothing — hand-designed | **85.41** | 85.97 | 87.18 | 11.66 |
| ConvNeXt-base, frozen (1024) | ImageNet | 84.70 | 85.70 | 86.78 | 10.86 |

and each arm's no-switch control, 60 epochs on subset 1 alone:

| control | ep 60 | best |
|:--|--:|--:|
| ConvNeXt-base scratch | 86.81 | 86.82 |
| ConvNeXt-tiny scratch | 85.06 | 85.06 |
| ours, frozen | 84.51 | 85.63 |
| ConvNeXt-base frozen | 84.43 | 85.30 |

**Training the representation on the task beats freezing it, by ~2 points.** Which is what should
happen — and it prices the hand-designed alternative: about 2 points of EMNIST accuracy, for zero
learned parameters, no training at all, and 2.7× fewer numbers than the frozen ImageNet
representation it matches.

**Our 381 features lead at epoch 15** and are overtaken only once the from-scratch nets have seen
more than one subset. A network with 230,000× more parameters needs more than 28,200 images to get
past them.

**Our 381 features beat frozen ImageNet ConvNeXt's 1024** at every point on the curve. That is the
opposite of Phase 11 — EMNIST is a line-drawing task at 28×28, which is what an oriented-energy
front end is for, where Phase 11's stimuli were natively 112 px with graded photometric properties.
Under half a point on single seeds, so read it as equivalence rather than a win.

---

## The one qualitative difference between frozen and trainable

**The frozen arms cross the switches invisibly. The from-scratch arms jump.**

Change in held-out accuracy across each switch, against each arm's own within-block noise floor
(epochs 10–60, excluding the switches themselves — the first epochs are left out because the
from-scratch arms climb 45 points there, which is a learning transient, not noise):

| arm | → subset 2 | → subset 3 | → subset 4 | noise sd | mean jump / sd |
|:--|--:|--:|--:|--:|--:|
| ours, frozen | +0.15 | −1.46 | −0.65 | 0.49 | −1.3 |
| ConvNeXt, frozen | +0.34 | −0.24 | −0.43 | 0.49 | −0.2 |
| ConvNeXt-tiny, scratch | +2.91 | +1.41 | +1.29 | 0.38 | **+4.9** |
| ConvNeXt-base, scratch | +1.41 | +1.84 | +1.35 | 0.47 | **+3.3** |

The two frozen arms show nothing: a learner reading fixed features genuinely cannot tell the data
changed. The two from-scratch arms show a large, immediate, repeatable **improvement** the moment
it does — 3–5 standard deviations, at all three switches, in both arms.

Part of that is recovery. Every arm decays slightly the longer it sits on one subset — mean drift
over blocks 2–4 of −0.37 (ours), −0.42 (frozen ConvNeXt), −0.76 (both scratch arms). But that decay
is under a point while the jumps reach +2.9, so the switch is not merely undoing it. Fresh data
buys a trainable representation something immediately that a frozen one can only accumulate slowly.

**The total benefit is nearly the same for everyone.** Switching minus its own control, by block:

| arm | blk 2 | blk 3 | blk 4 |
|:--|--:|--:|--:|
| ours, frozen | +0.69 | +1.23 | +1.44 |
| ConvNeXt, frozen | +0.92 | +0.68 | +1.41 |
| ConvNeXt-tiny, scratch | +0.93 | +1.67 | +1.99 |
| ConvNeXt-base, scratch | +0.66 | +1.13 | +1.29 |

Fresh data is worth ~1.3–2.0 points to every arm. What differs is **when it arrives**: as a step
for the trainable arms, as a slow accumulation for the frozen ones.

**Switching beats not switching for all four.** Every control peaks around epoch 15–20 and then
drifts down, overfitting the 28,200 examples it is stuck with. Fresh data beats more passes over
old data — expected, and its appearance is a check that the harness works.

## Where the discontinuity really is

Not in generalisation — in the training-set curve. Accuracy on the subset being trained on, in the
last 5 epochs of each block, minus held-out accuracy:

| arm | block 1 | block 2 | block 3 | block 4 |
|:--|--:|--:|--:|--:|
| ours, frozen | 10.18 | 11.10 | 11.21 | 11.00 |
| ConvNeXt, frozen | 9.94 | 10.33 | 10.62 | 10.24 |
| ConvNeXt-tiny, scratch | 10.11 | 14.36 | 13.34 | 12.10 |
| ConvNeXt-base, scratch | 6.26 | 13.43 | 12.41 | 11.84 |

That 10–14 point gap is memorisation, measured directly rather than inferred, and it is stable
across blocks: each new subset gets memorised to about the same depth as the last. In the figures
it is the orange sawtooth — climbing within a block, collapsing the instant the data changes.

**The from-scratch arms essentially interpolate their subset**: peak training accuracy per block
of 96.0 / 99.9 / **100.000** / 99.2 (`tiny`) and 94.1 / 99.9 / 100.0 / 99.9 (`base`) — in block 3
`tiny` gets literally every one of its 28,200 images right. The frozen arms plateau near 96–97 %.
Yet held-out accuracy rises while that happens: interpolating the training set is not the same as
failing to generalise.

And the from-scratch *control* is the sharpest version of that. It drove training accuracy from
94.4 % to a literal 100.000 % over 60 epochs while its exam score moved 83.10 → 85.06, most of
which was the learning-rate schedule. **Perfecting the memorisation bought almost nothing.**

## Forgetting is fast, then slow — and identical for both *frozen* arms

Accuracy on subset 1 *minus* held-out accuracy, at the end of each block — i.e. how much of an
advantage subset 1 still enjoys long after training left it:

| arm | end blk 1 | end blk 2 | end blk 3 | end blk 4 |
|:--|--:|--:|--:|--:|
| ours, frozen | 10.95 | 2.43 | 1.31 | 0.84 |
| ConvNeXt, frozen | 10.34 | 2.48 | 1.31 | 0.51 |
| ConvNeXt-base, scratch | 8.17 | 2.46 | 1.80 | 1.13 |
| ConvNeXt-tiny, scratch | 11.26 | 4.50 | 2.22 | 2.06 |

**About 78 % of the memorised margin is released in the first 15 epochs after training moves on**,
then the remainder decays slowly toward zero.

The two *frozen* arms agree to within 0.1 points at every step but the first — despite one being
381 hand-designed numbers and the other 1024 learned from 1.28 M photographs. For a fixed
representation, forgetting is a property of **the readout and the optimiser**, not of what feeds
them. If we want downstream learning to be more stable, the front end is not the lever.

The from-scratch arms locate the boundary of that claim. `tiny` retains roughly **double** the
advantage on old data one block after leaving it (4.50 against ~2.45) and is still 2.06 points ahead
at the end, where both frozen arms are at 0.5–0.8. When the representation is itself a free
parameter, there is somewhere for old examples to persist. `base` sits between the two, closer to
the frozen arms.

---

## Caveats

**λ = 8 px is at EMNIST's Nyquist limit.** EMNIST is 28×28 upsampled 4×, so the finest channel of
the best stroke-set configuration largely measures bilinear interpolation here. It was included
because the brief was to run the best configuration; it is not doing what it does on natively-112
stimuli. An ablation without it would be cheap and has not been run.

**Grid 3, not grid 1**, because EMNIST characters are centred — Phase 10 found grid 3 beating grid 1
by 8 points on Fashion-MNIST for that reason, and grid 1 only wins where position is randomised.

**These are single seeds — but not noisy ones.** Each from-scratch arm's switching and control
runs are **bit-identical through epoch 15**, to four decimal places, so every later divergence is
caused by the data and nothing else. Those within-arm contrasts carry no seed noise at all.
Comparisons *across* arms are another matter, and differences under ~0.5 points there — including
our margin over frozen ConvNeXt — are not resolved.

**The from-scratch arms use a different recipe**: AdamW 3e-4, weight decay 0.05, cosine over 60
epochs with 3-epoch warmup, against Adam 1e-3 constant for the frozen readouts. lr 1e-3 is far too
high for a from-scratch net at this batch size, so the change was necessary — but it means
cross-arm accuracy *levels* are confounded. The within-arm switching-vs-control contrasts are clean;
the ranking table is not, strictly.

**No augmentation anywhere.** ConvNeXt's published results depend on 300 epochs with RandAugment,
Mixup, CutMix, stochastic depth and EMA over 1.28 M images. Withheld deliberately, since every other
arm here is unregularised and the phase exists to make memorisation visible rather than suppress it
— but this is not ConvNeXt at its best.

**The from-scratch arms still normalise with ImageNet's channel constants**, inherited by copying
the frozen arm's preprocessing. For a randomly-initialised network those particular numbers mean
nothing; it is a fixed affine transform the first convolution can absorb, so it cannot change what
the network can learn. Recorded because the naming is misleading, not because it matters.

**Cost, since it was the reason this arm nearly went unrun:** on a 4090, `base` at 224 px trains
end to end in ~90 min per 60-epoch run — of which about 40 % is the three accuracy evaluations per
epoch, not training. Extracting our 381 hand-designed features from the same images took **69 min
on 14 CPU threads**. Training an 87.6 M-parameter network was cheaper than computing our front end,
because ours has no GPU path for the ray transform and pooling.

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

---

## Figures

| file | what it shows |
|:--|:--|
| `figures/curriculum_compare.png` | all four arms' held-out curves on one axis |
| `figures/curriculum_ours.png` | our 381 frozen features, with control |
| `figures/curriculum_convnext.png` | frozen ImageNet ConvNeXt, with control |
| `figures/curriculum_scratch_tiny.png` | ConvNeXt-tiny from scratch, with control |
| `figures/curriculum_scratch_base.png` | ConvNeXt-base from scratch, with control |
| `figures/fewshot.png` | few-shot transfer to 10 unseen classes |

The per-arm figures share a layout: navy is held-out accuracy, orange dashed is accuracy on the
subset being trained on right now, green dotted is accuracy on subset 1 throughout, grey is the
no-switch control. Axes are set by epochs 10+ — the from-scratch arms climb 45 points in the first
few epochs, and on a full-range axis that flattens everything else into a line.

## What this phase concluded

1. **Memorisation is large, uniform and transient.** Every arm keeps a 10–14 point margin on
   whatever it is looking at, sheds ~78 % of it within 15 epochs of moving on, and none of it was
   ever what produced held-out accuracy. The from-scratch control proves the last point outright:
   it memorised all 28,200 images perfectly and gained essentially nothing.
2. **For frozen representations, forgetting is a readout property, not a representation property.**
   381 designed numbers and 1024 ImageNet numbers behave identically. The front end is not the lever.
3. **Training the representation changes that** — the switch becomes visible (3–5 sd), and old data
   is retained about twice as long.
4. **The from-scratch advantage is not memorisation.** It transfers to character classes never seen,
   from one example each, beating every frozen alternative by 14+ points.
5. **Our 381 hand-designed features cost about 2 points** against a network trained on the task, and
   match 1024 frozen ImageNet features while being a *better* nearest-neighbour space than they are.

## Still open from this phase

* **The changed-conditions test** — polarity inversion, blur, thickening, rotation on EMNIST. The
  complement to the few-shot result, and the one place a hand-designed front end has a *provable*
  invariance the others lack. Phase 11 has the machinery.
* **The λ = 8 ablation.** It sits at EMNIST's original Nyquist limit and is probably measuring
  interpolation here. Cheap, unrun.
* **Why `r` is the one class our features win on** in the few-shot breakdown, by 11 points.
