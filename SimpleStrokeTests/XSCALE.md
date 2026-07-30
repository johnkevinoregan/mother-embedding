# Cross-scale features and the thickness/fuzziness confound

`Sweep_Capacity.jl` with `cross_scale=`, log in `xscale.log`. 16,000/4,000, grid 1, 100 epochs.

**The hypothesis.** Thickness and fuzziness are confounded because both are read from how energy
distributes across scale, and nothing in the feature set encodes the *relationship between* scales
at a point — only per-scale amounts, which pooling then averages. A thick sharp stroke has energy
at coarse **and** fine scales; a thin blurred one has coarse only.

**Two forms.** `:product` is the existing `a3_maps`, `C₀(k)·C₀(k+1)` — an energy, so mean pooling
is valid. `:ratio` is `C₀(k+1)/(C₀(k)+C₀(k+1))`, the fraction of energy at the finer scale — a
**bounded, scale-free** quantity, so numerator and denominator are pooled *separately* and divided
afterwards with a relative floor, by the rule established when the ray ratios turned out wrong.
Two features each at grid 1.

**Predictions, recorded before running:** product moves the confound by < 0.05; ratio moves it by
> 0.2 on both splits. Falsification stated in advance: if the ratio failed, stop proposing
operators and go empirical.

## The confound rows

| arm | nfeat | thickness (blur split) | Δ | fuzziness (thickness split) | Δ |
|:--|--:|--:|--:|--:|--:|
| baseline | 31 | −2.060 | — | −2.448 | — |
| xscale product | 33 | −2.059 | +0.001 | −2.406 | +0.042 |
| xscale ratio | 33 | −2.086 | **−0.026** | −2.338 | **+0.110** |
| harmonics+offsets | 54 | −2.051 | +0.009 | −2.460 | −0.012 |
| **adopted+xscale** | 58 | **−1.899** | **+0.161** | **−2.288** | **+0.160** |

## What this actually shows

**The prediction was wrong, and the hypothesis is right anyway — via an interaction.**

*Product:* < 0.05 on both, as predicted. A product is large whenever both scales carry energy and
cannot separate thick-sharp from thin-blurred.

*Ratio alone:* **−0.026 and +0.110**, against a predicted > +0.2. It fails outright on the blur
split and half-works on the thickness split. The effect is **directional** — the fine/coarse
fraction is more nearly a blur measure than a width measure, so it repairs the row it encodes
(fuzziness) and not the other.

*In combination:* **+0.161 and +0.160**, replicated in both directions to within 0.001, and
attributable to the cross-scale features specifically — `harmonics+offsets` without them gives
+0.009 and −0.012. This is the first thing in the project to move the confound at full n.

**Why the interaction is plausible:** the same fine/coarse energy fraction means different things
at different stroke widths, so it is only interpretable once the readout has width information to
condition on — which the crossed offsets supply. That is a hypothesis, not a measurement.

**The cost.** On the thickness split, adding cross-scale to the adopted configuration loses
0.02–0.05 on every geometric row (curvedness 0.905 → 0.885, brokenness 0.542 → 0.496, vangle
0.881 → 0.843). So it trades geometry for the confound rather than adding freely.

## Corrections this forces

**To `SWEEP_FULLN.md` and commit 6298a13:** "Nothing touches the confound" was true of the axes
tested there and is now false in general — `adopted+xscale` moves it ~+0.16 in both directions.

**To my own falsification call.** I declared the hypothesis dead after seeing the isolated ratio
arm fail on one split, and the very next arm contradicted it. The lesson is specific: an operator
that does nothing alone can still matter in combination, so testing it only in isolation is not a
sufficient test of the idea behind it.

## Where this leaves the configuration

| use | configuration | why |
|:--|:--|:--|
| best geometry under shift | `harmonics+offsets` (54) | curvedness +0.080, brokenness +0.075 under thickness shift |
| best on the confound | `adopted+xscale` (58) | ~+0.16 both directions, at 0.02–0.05 geometric cost |
| best i.i.d. | `adopted+xscale` (58) | thickness 0.752, fuzziness 0.771 — best of anything tried |

No single configuration dominates. The choice depends on whether the confound or the geometric
rows matter more, and that is a question about the eventual task, not about the front end.

## Still open

The confound is **reduced, not fixed** — −1.899 and −2.288 are still far below zero, so a readout
trained on one range remains badly wrong on the other. The empirical route stands: find which
stage-3 dimensions carry the distinction under shift, and look at what drives them.
