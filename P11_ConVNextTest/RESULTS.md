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
loses to it on every structural property the moment contrast polarity flips — but there are three
extrapolation splits, and they do not agree.**

An earlier version of this file stopped at the polarity split and claimed the front end buys "an
invariance that holds exactly under a distribution shift no training data covered". That is
**true for polarity and false as a general statement**; see *All three extrapolation splits*
below. Reading one split and generalising was the mistake.

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

**Ours is flat to within noise and marginally positive.** On *this* split, 88.6 M parameters and
1.28 M photographs do not produce the invariance, and more scale would not either: Base degrades
*more* than Tiny (−0.254 against −0.238 for the MLP arms, −0.669 against −0.385 for the linear
ones).

**How to read Δ, and its trap.** Δ answers *"does this representation care that the distribution
moved?"*, which is a different question from *"how good is it?"* — that one is the absolute R².
Both are needed. Our polarity result only appears in Δ, because ConvNeXt scores higher in
absolute terms on the polarity split while being the arm that degrades. Conversely **Δ flatters a
weak arm, because it has less to lose**: the random-init arms score Δ ≈ +0.004 on the thickness
split, perfectly "robust" at a performance level of zero. Never read one column alone.

---

## All three extrapolation splits — and they disagree

The polarity split above is one of three. Reported together because taking the first as
representative is exactly the error this section corrects. Δ is against each arm's **own** i.i.d.
number for the same property, e.g. ours·MLP `curvedness` = 0.799 on the thickness split against
0.930 i.i.d., so Δ = −0.131.

### Fuzziness (trained sharp ≤ 3 px, tested blurred ≥ 8 px) — ours wins structure, loses one row

| Δ | curved | broken | closed | vangle | arms | **thickness** | mean |
|:--|--:|--:|--:|--:|--:|--:|--:|
| **ours·MLP** | **−0.038** | **−0.330** | **−0.007** | **−0.078** | **−0.036** | −2.703 | −0.452 |
| convnext_base s4·MLP | −0.047 | −0.624 | −0.040 | −0.137 | −0.082 | **−0.561** | **−0.218** |

Ours transfers better on **all five structural rows** and still loses the mean, because the
`thickness` readout falls by 2.7 when edges blur.

### Thickness (trained thin ≤ 6 px, tested thick ≥ 8 px) — ConvNeXt wins structure too

| Δ | curved | broken | closed | vangle | arms | **fuzziness** | mean |
|:--|--:|--:|--:|--:|--:|--:|--:|
| ours·MLP | −0.131 | −0.482 | −0.014 | −0.096 | **−0.032** | −2.983 | −0.537 |
| convnext_base s4·MLP | **−0.032** | **−0.132** | **−0.009** | **−0.054** | −0.047 | **−0.088** | **−0.053** |

Here ConvNeXt is better on three of five structural rows as well as on the mean.

### The finding underneath: thickness and fuzziness are confounded in our representation

Blur the edges and our **thickness** estimate breaks (−2.703). Thicken the stroke and our
**fuzziness** estimate breaks (−2.983). ConvNeXt's equivalents are −0.561 and −0.088.

Both of ours are read from the **scale distribution of oriented energy**, and a wider stroke and a
softer edge push that distribution the same way — so a readout fitted on one range cannot separate
them out of range. This is a specific, diagnosable weakness that no earlier phase surfaced, and it
is in precisely the place a multi-scale representation ought to be strong. It also means two of the
dataset's eight target properties are not independently recoverable by our front end under shift.

### So the invariance claim, stated correctly

| shift | verdict |
|:--|:--|
| **contrast polarity** | **ours wins decisively.** Δ +0.016 against −0.24 to −1.16, and ours wins the absolute structural rows too. Invariant *by construction* — quadrature energy discards contrast sign. |
| **edge blur** | **ours wins all five structural rows**, loses the mean on the `thickness` row alone. |
| **stroke thickness** | **ConvNeXt wins**, on three of five structural rows and on the mean. |

One designed invariance that is exact and that scale cannot buy, plus better structural transfer
under blur — set against a thickness/fuzziness confound ConvNeXt does not share. That is a
narrower and more useful claim than the one this file made first.

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

---

## Where in the network does it live? (stage probe)

`CX_STAGEPROBE=1`, log in `stageprobe.log`. The first version of this file scored only s4, s3 and
s1-4 concatenated, so it could not say whether ConvNeXt's advantage came from **low-level filters**
or from **mid-level combinations**. Each stage is now read out alone.

The stages double in width with depth (Tiny: 96 / 192 / 384 / 768), which confounds *which stage*
with *how many columns the readout gets* — and the confound points the same way as the effect. So
every stage is also reduced to a common 96 dimensions by PCA fitted on the training split.

**Dimension-matched (PCA-96), linear readout:**

| stage | curved | broken | vangle | arms | thick | fuzzy | polarity |
|:--|--:|--:|--:|--:|--:|--:|--:|
| **s1** | 0.002 | 0.003 | 0.126 | 0.231 | 0.567 | 0.706 | **0.762** |
| **s2** | 0.232 | 0.057 | 0.159 | 0.665 | 0.734 | 0.836 | 0.876 |
| **s3** | **0.949** | **0.518** | **0.684** | **0.914** | 0.856 | 0.949 | 0.969 |
| **s4** | 0.951 | 0.430 | 0.690 | 0.897 | 0.844 | 0.943 | 0.978 |
| ours (31, native) | 0.687 | 0.213 | 0.549 | 0.732 | 0.547 | 0.555 | −0.002 |

**The advantage is not in the low-level features.** Stage 1 reads curvedness at **0.002** and
brokenness at **0.003**. Stage 2 barely improves. Everything arrives at **stage 3**, and stage 4
adds nothing beyond it.

**A clean dissociation.** Stage 1 is already strong on the *photometric* rows — polarity 0.762,
fuzziness 0.706, thickness 0.567 — while carrying no geometry whatever. Contrast, blur and scale
are low-level; **shape is a mid-level construction** in this network.

**Which reframes the headline comparison.** With the MLP readout, at native widths:

| | nfeat | curved | broken | vangle |
|:--|--:|--:|--:|--:|
| convnext_tiny s1 | 96 | 0.477 | 0.002 | 0.151 |
| convnext_tiny s1-2 | 288 | 0.743 | 0.364 | 0.412 |
| **ours** | **31** | **0.930** | **0.730** | **0.944** |
| convnext_tiny s3 | 384 | 0.981 | 0.817 | 0.964 |

**Our 31 hand-designed features beat ConvNeXt's first two stages combined — 288 features — on
every geometric property, and are overtaken only at stage 3.** So the earlier claim needs
qualifying: a frozen ImageNet model makes geometry more linearly explicit than our operators do
*at its mid level*. Against its **low-level** representation, which is the level our front end
actually occupies, ours wins decisively.

**And prediction 4 was half-right after all.** Marked "wrong" above because s4 ≥ s3 at native
width — but that is the width confound. Dimension-matched, **s3 beats s4 on brokenness (0.518 vs
0.430)** and ties on vangle. ImageNet's last stage is tuned to object category and does lose
geometric detail; the 384-vs-768 column difference was hiding it.

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

**It does not license** "the front end is unnecessary". Its polarity invariance is exact and free,
88.6 M parameters and 1.28 M photographs did not supply it, and scaling from Tiny to Base made it
worse rather than better.

**It does license** retiring one argument. "Our features make geometric properties explicit in a
way learned representations do not" is false on this dataset — a frozen ImageNet model makes them
*more* linearly explicit, under our own linear criterion.

**And it does not license the replacement I first reached for.** Writing that the value is
"invariance under shift" was wrong twice over. It is loose terminology — in vision *shift* means
translation, which nothing here tests — and it is too general: across the three nuisances actually
tested, ours wins polarity decisively, wins the structural rows under blur, and **loses stroke
thickness outright**. The defensible claim is specific: **one exact, constructed invariance
(contrast polarity) that scale cannot buy, plus better structural transfer under blur, against a
thickness/fuzziness confound ConvNeXt does not have.**

## Caveats, and what would sharpen this

1. **Feature count is unmatched**: 31 against 1024. Our arm should be re-run at grid 3 (279
   features) before the i.i.d. comparison is treated as settled. Cheap — the harness caches.
2. **One seed.** Phase 9 measured ~0.03 run-to-run variance for a trained CNN. Most gaps here
   exceed that comfortably; the i.i.d. vangle and arms margins (+0.032, +0.029) do not.
3. **ConvNeXt sees 224×224**, a 4× bilinear upsample of our 112. No information is added, but the
   comparison is not compute-matched.
4. **A thickness/fuzziness disentangling test.** The confound above is the sharpest open lead
   from this phase: two target properties that our scale-space representation cannot separate
   under shift, where ConvNeXt can. A stimulus set crossing stroke width with edge softness
   independently would say whether it is the ladder's scale resolution (3 scales, ρ = 2 / 3.74 / 7)
   or the pooling that loses the distinction.
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
Read the **polarity** transfer table and it favours us, by 0.68 on `brokenness`. Read the
**thickness** transfer table and it favours ConvNeXt again. Read the random-init row and the
credit for ConvNeXt's advantage goes specifically to 1.28 M labelled photographs, not to its
architecture.

Four questions, four different answers — which is the actual result of this phase, and the reason
no single sentence about "our features versus learned features" was going to survive contact with
it.
