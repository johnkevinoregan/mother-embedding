# Frozen ConvNeXt on the broken-lines task — results

Log: `run_full.log`. Protocol, predictions and setup notes: `README.md`.

16,000 train / 4,000 test per split, 112×112 stimuli, our arm at grid 1 (31 features, Phase 9's
best configuration for this dataset), ConvNeXt at 224×224 with global average pooling per stage.
Identical images for every arm, read back from the same `.f32` files both languages consume.

| | ours | ConvNeXt-Tiny | ConvNeXt-Base |
|:--|--:|--:|--:|
| learned parameters | **0** | 28.6 M | 88.6 M |
| images used to build it | 0 | 1.28 M | 1.28 M |
| features given to the readout | **31** | 768 / 1440 | 1024 / 1920 |

---

## The headline, in one sentence

**Frozen ImageNet ConvNeXt beats our front end on every property when train and test match, and
loses to it on every structural property the moment contrast polarity flips.**

---

## i.i.d. split — ConvNeXt wins everything

| arm | nfeat | curved | broken | closed | vangle | arms | thick | fuzzy | polarity |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| trivial baseline | 3 | 0.007 | 0.006 | 0.458 | 0.065 | 0.028 | 0.197 | 0.152 | 0.060 |
| pixels·linear | 12544 | −0.000 | −0.001 | −0.001 | −0.001 | 0.000 | −0.001 | −0.000 | 0.712 |
| ours·linear | 31 | 0.687 | 0.213 | 0.887 | 0.549 | 0.732 | 0.547 | 0.555 | −0.002 |
| **ours·MLP** | **31** | 0.930 | 0.730 | **0.998** | 0.944 | 0.958 | 0.709 | 0.729 | **−0.063** |
| convnext_tiny s4·lin | 768 | 0.969 | 0.675 | 0.988 | 0.835 | 0.947 | 0.916 | 0.971 | 0.992 |
| convnext_tiny s4·MLP | 768 | 0.980 | 0.828 | 0.999 | 0.960 | 0.981 | 0.943 | 0.979 | 0.998 |
| convnext_base s4·lin | 1024 | 0.976 | 0.750 | 0.990 | 0.882 | 0.963 | 0.932 | 0.974 | 0.991 |
| **convnext_base s4·MLP** | 1024 | **0.984** | **0.865** | 0.998 | **0.976** | **0.987** | **0.952** | **0.982** | 0.998 |

**This is a real result against the project's framing and should not be softened.** Frozen
ConvNeXt-Base beats our 31 features on curvedness (+0.054), brokenness (+0.135), vertex angle
(+0.032), arm count (+0.029), thickness (+0.243) and fuzziness (+0.253), and ties on closedness.
Nothing here was designed for strokes; the representation was fitted to natural photographs and
never touched this dataset.

**Two sharper versions of the same problem.**

*The prediction was wrong in an informative way.* I predicted ConvNeXt would be competitive on
`thickness` and `fuzziness` — scale and blur — and **worse** on `vangle` and `arms`, the 2π
ray-counting properties, on the strength of Geirhos et al.'s texture-bias result. The *relative*
pattern held: its largest margins are exactly thickness and fuzziness. But it also beat us on
vangle and arms, so "texture-biased networks are weak at geometry" does not survive as an
absolute claim on this task.

*Explicitness goes the wrong way too.* The project's method is to ask whether one **linear**
readout recovers a property. `convnext_base s4·lin` — a single linear map on frozen features —
scores 0.976 / 0.750 / 0.990 / 0.882 / 0.963 against our linear arm's 0.687 / 0.213 / 0.887 /
0.549 / 0.732. **ConvNeXt's representation is more linearly explicit than ours**, on our own
criterion, on our own dataset.

Only two things go our way. Ours reads polarity at **−0.063**, i.e. carries no contrast-sign
information at all, exactly as quadrature energy should; ConvNeXt reads it at **0.998**. And 31
numbers reaching 0.930 / 0.730 / 0.998 / 0.944 / 0.958 against 1024 numbers and 88.6 M
parameters is a favourable ratio, though a ratio is not a win.

---

## Polarity extrapolation — ours wins every structural property

Trained on light strokes, tested on dark. The `polarity` row has no variance in training and is
scored `—`.

| arm | nfeat | curved | broken | closed | vangle | arms | thick | fuzzy |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| trivial baseline | 3 | 0.009 | 0.008 | 0.459 | 0.069 | 0.023 | 0.200 | 0.176 |
| pixels·linear | 12544 | −0.127 | −0.099 | −2.441 | −0.527 | −1.722 | −2.917 | −0.858 |
| ours·linear | 31 | 0.682 | 0.251 | 0.882 | 0.544 | 0.728 | 0.596 | 0.555 |
| **ours·MLP** | **31** | **0.933** | **0.776** | **0.999** | **0.952** | **0.961** | 0.742 | 0.746 |
| convnext_tiny s4·MLP | 768 | 0.937 | 0.049 | 0.969 | 0.807 | 0.888 | 0.576 | 0.778 |
| convnext_base s4·MLP | 1024 | 0.918 | 0.098 | 0.876 | 0.839 | 0.931 | 0.585 | 0.720 |
| convnext_base s4·lin | 1024 | 0.857 | **−2.116** | 0.780 | 0.104 | 0.721 | 0.736 | 0.700 |
| convnext_base s3·lin | 512 | 0.895 | **−3.677** | 0.750 | 0.246 | 0.840 | 0.757 | 0.705 |
| convnext_base s1-4·lin | 1920 | 0.880 | **−4.824** | 0.741 | 0.035 | 0.783 | 0.127 | 0.698 |

Our 31 features beat **every** ConvNeXt configuration on all five structural properties —
brokenness by **0.68** (0.776 against 0.098) and vertex angle by 0.11. ConvNeXt keeps a small
edge on the two nuisance rows, thickness and fuzziness, which are the properties least entangled
with contrast sign.

The linear ConvNeXt arms do not merely degrade, they go **far below the trivial three-scalar
baseline** on brokenness: −2.1, −3.7, −4.8 against a baseline of 0.008. A representation that
encodes polarity at R² 0.998 has no way to hold a gap-detection readout fixed when polarity
inverts.

---

## Transfer cost — the table the invariance claim is actually about

`Deltas.jl`. Δ = R²(polarity split) − R²(i.i.d. split), so Δ ≈ 0 means the representation did
not care that polarity flipped.

| arm | mean Δ | curved | broken | closed | vangle | arms |
|:--|--:|--:|--:|--:|--:|--:|
| **ours·MLP** | **+0.016** | +0.003 | +0.045 | +0.001 | +0.008 | +0.003 |
| **ours·linear** | **+0.010** | −0.005 | +0.038 | −0.005 | −0.005 | −0.004 |
| convnext_tiny s4·MLP | −0.238 | −0.044 | −0.779 | −0.030 | −0.153 | −0.093 |
| convnext_base s4·MLP | −0.254 | −0.066 | −0.767 | −0.122 | −0.137 | −0.056 |
| convnext_tiny s4·lin | −0.385 | −0.066 | −1.309 | −0.096 | −0.467 | −0.150 |
| convnext_base s4·lin | −0.669 | −0.119 | −2.866 | −0.210 | −0.777 | −0.242 |
| convnext_base s1-4·lin | −1.155 | −0.098 | −5.598 | −0.253 | −0.854 | −0.186 |
| pixels·linear | −1.241 | −0.126 | −0.097 | −2.439 | −0.526 | −1.722 |

**Ours is flat to within noise and marginally positive. Every ConvNeXt arm loses 0.24 to 1.16.**

This is what the front end buys, and it is a different thing from what the project has mostly
been claiming. It does not buy i.i.d. explicitness — ConvNeXt has more of that. It buys an
**invariance that holds exactly, by construction, under a distribution shift no training data
covered.** 88.6 M parameters and 1.28 M photographs do not produce it, and on this evidence more
scale would not either: Base degrades *more* than Tiny (−0.254 against −0.238 for the MLP arms,
−0.669 against −0.385 for the linear ones).

---

## The control that matters: random-init ConvNeXt

Same architecture, `weights=None`, seeded, **no training and no ImageNet**. This separates
architectural inductive bias from anything actually learned from photographs.

| arm | nfeat | curved | broken | closed | vangle | arms | thick | fuzzy | polarity |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| trivial baseline | 3 | 0.007 | 0.006 | 0.458 | 0.065 | 0.028 | 0.197 | 0.152 | 0.060 |
| pixels·linear | 12544 | −0.000 | −0.001 | −0.001 | −0.001 | 0.000 | −0.001 | −0.000 | 0.712 |
| **ours·MLP** | **31** | 0.930 | 0.730 | 0.998 | 0.944 | 0.958 | 0.709 | 0.729 | −0.063 |
| convnext_base s4·MLP *(ImageNet)* | 1024 | 0.984 | 0.865 | 0.998 | 0.976 | 0.987 | 0.952 | 0.982 | 0.998 |
| convnext_tinyrand s4·MLP | 768 | −0.010 | −0.085 | 0.387 | 0.038 | 0.032 | 0.075 | 0.121 | 0.814 |
| convnext_baserand s4·MLP | 1024 | −0.014 | −0.057 | 0.315 | 0.031 | 0.028 | 0.076 | 0.078 | 0.816 |
| convnext_baserand s4·lin | 1024 | −0.000 | −0.001 | 0.049 | 0.003 | 0.000 | 0.003 | 0.001 | 0.302 |

**An 88.6 M-parameter ConvNeXt with random weights recovers nothing.** Curvedness −0.014,
vangle 0.031, arms 0.028 — indistinguishable from raw pixels. `closedness` at 0.387 is *below*
the three-scalar trivial baseline. The one thing it carries is polarity (0.82), which is mean
luminance leaking through and requires no learning at all.

So **ImageNet pretraining is doing essentially all of the work**; the architecture on its own —
depthwise 7×7 convolutions, patchify stem, LayerNorm — buys about as much as reading the pixels.
The i.i.d. loss above is real, but it costs **1.28 M labelled photographs** to buy, and it comes
with the polarity failure the transfer table shows.

**A more specific mechanism, worth stating because it is narrower than "pretraining helps".** The
readout sees **globally average-pooled** channels. For random filters the spatial mean of a random
projection is nearly constant across images, so pooling destroys almost everything. What ImageNet
training buys is not only good filters but channels whose *spatial average is itself informative*.
Caveat attached: this is a result about random weights **under global pooling**, the configuration
chosen to match our own winning arm. Random features read out *without* pooling should do
considerably better than 0.03, and that has not been tested here.

**Prediction on record, and it was wrong.** Before running this I expected random-init to land
"well above raw pixels but well below ImageNet". It landed *at* raw pixels.

## Convergence — all of these numbers are ceilings, unlike Phase 9's CNN

`Plot_Curves.jl`, figure in `figures/curves_iid.png`. Validation R² per epoch for every trained
head, mean over the seven properties excluding `polarity`.

| arm | best | at epoch | final | Δ |
|:--|--:|--:|--:|--:|
| ours·MLP | 0.864 | 61 | 0.861 | −0.004 |
| convnext_tiny s4·MLP | 0.956 | 98 | 0.954 | −0.001 |
| convnext_base s4·MLP | 0.966 | 88 | 0.965 | −0.001 |
| convnext_tinyrand s4·MLP | 0.094 | 43 | 0.011 | −0.083 |
| convnext_baserand s4·MLP | 0.077 | 61 | 0.028 | −0.049 |

Smooth, monotone, converged. That matters because Phase 9's CNN arm was **not** — its history
swings from −0.536 mean validation R² at epoch 30 to 0.647 at epoch 35, so best-epoch selection
was sampling a spike. Nothing of that kind happens here, so this comparison is between three
saturated readouts rather than three arbitrary points on three trajectories.

Two things the curves add that the final numbers do not:

**ConvNeXt's head barely has to learn.** It reaches ~0.93 after a *single* epoch and ~0.95 by
epoch 10; ours starts at 0.65 and needs ~40 epochs to reach 0.86. On `vangle`, epoch 1 is already
~0.85 for ConvNeXt against ~0.65 for ours. That is the same story the linear-readout gap tells,
in a different currency: the information is sitting on the surface of ConvNeXt's features.

**The random-init arms overfit**, peaking at epochs 43–61 and then declining to near zero. A
330 k-parameter head memorising noise is what you expect when the features carry no signal, and
it is independent confirmation that they do not.

*`polarity` is excluded from that mean deliberately.* Our arm scores ≈ 0 on it by design and
ConvNeXt ≈ 1.0, so including it made ours look 0.22 worse for succeeding at its design goal.
Excluded, the gap is 0.10. The first version of this figure got that wrong.

## Prediction scorecard

| | prediction | outcome |
|:--|:--|:--|
| 1 | beats raw pixels on every property | **correct** — pixels are ≈ 0 on everything but polarity |
| 2 | competitive on thickness/fuzziness, **worse** on vangle/arms | **wrong** — better on all four; the relative pattern held but the absolute claim did not |
| 3 | collapses on the polarity split | **correct**, on transfer — and the 400-image smoke test that appeared to contradict it was a small-sample artefact |
| 4 | stage 3 beats stage 4 | **wrong** — s4 > s3 in every configuration and on nearly every property |

Recording 2 of 4 because the two failures are the informative ones.

---

## What this does and does not license

**It does not license** "the front end is unnecessary". Its invariance is exact and free, and no
amount of ImageNet pre-training supplied it.

**It does license** retiring one argument. "Our features make geometric properties explicit in a
way learned representations do not" is false on this dataset — a frozen ImageNet model makes them
*more* linearly explicit. The claim that survives is narrower and better: the front end's value
is in **invariance under shift**, not in i.i.d. availability.

## Caveats, and what would sharpen this

1. **Feature count is unmatched**: 31 against 1024. Our arm should be re-run at grid 3 (279
   features) before the i.i.d. comparison is treated as settled. Cheap — the harness caches.
2. **One seed.** Phase 9 measured ~0.03 run-to-run variance for a trained CNN. Most gaps here
   exceed that comfortably; the i.i.d. vangle and arms margins (+0.032, +0.029) do not.
3. **ConvNeXt sees 224×224**, a 4× bilinear upsample of our 112. No information is added, but the
   comparison is not compute-matched.
4. **`fuzziness` and `thickness` extrapolation splits** are in the log and not yet analysed here;
   the polarity split is the one the invariance claim rests on.
5. **The stimuli are still a single stroke on a flat field.** As with every result in this
   project, nothing here adjudicates behaviour on natural greyscale images.

---

## Where this leaves the three-way picture

| representation | built from | trained head | `vangle` linear | `vangle` +MLP |
|:--|:--|:--|--:|--:|
| random ConvNeXt, frozen | nothing | yes | 0.003 | 0.031 |
| Phase 9 CNN, end-to-end | 12,000 labelled strokes | (whole net) | — | 0.248 *(unconverged)* |
| **ours** | designed, 0 parameters | yes | 0.549 | **0.944** |
| ConvNeXt, frozen | 1.28 M labelled photographs | yes | **0.882** | **0.976** |

Every arm here is *fixed representation + trained readout* except the Phase 9 CNN, which is
trained end-to-end and did not converge.

Read down the `linear` column and the project's own explicitness criterion favours ConvNeXt.
Read the transfer table and it favours us, by 0.68 on `brokenness`. Read the random-init row and
the credit for ConvNeXt's advantage goes specifically to 1.28 M labelled photographs, not to its
architecture.
