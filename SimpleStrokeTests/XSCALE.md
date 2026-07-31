# Cross-scale features and the thickness/fuzziness confound

> ## ⚠ RETRACTED — the cross-scale confound result was an implementation artefact
>
> This page reported that cross-scale features move the thickness/fuzziness confound by
> **+0.161 and +0.160**, replicated in both directions. **That is wrong.**
>
> I said I was enabling `A₃`, the cross-scale operator that already existed in `AndLayer`, and
> instead reimplemented the formula inline in `Frontend._feat`. The two are not the same feature:
> `Pooling.assemble` emits **`sqrt(pooled)`** for every A-block, so `A₃` is amplitude-like and
> consistent with A₁ and A₂, while the inline copy was energy-like. After standardisation those
> are different features.
>
> Re-run through the real `a3_maps` path:
>
> | | thickness (blur split) | fuzziness (thickness split) |
> |:--|--:|--:|
> | baseline | −2.060 | −2.448 |
> | adopted + **A₃ (correct)** | −2.041 (+0.019) | **−2.718 (−0.270)** |
> | adopted + inline variant | −1.899 (+0.161) | −2.288 (+0.160) |
>
> **`A₃` does not move the confound.** Alone it makes it *worse* in both directions (−0.159,
> −0.196). Nothing tested so far fixes it.
>
> The inline variant's effect was real and replicated — but it is a **raw pooled cross-scale
> product**, not `A₃`, and dropping the `sqrt` for one block while keeping it for the others is an
> inconsistency rather than a design. It is recoverable from commit `094ed0e` and would need its
> own justification before being taken seriously.


> **Start with [`FINDINGS.md`](FINDINGS.md)** — a plain-language summary of what these
> experiments found. This file is the detailed tables.


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

## The confound rows — corrected

Re-run with `:product` routed through the real `a3_maps`/`assemble` path.

| arm | nfeat | thickness (blur split) | Δ | fuzziness (thickness split) | Δ |
|:--|--:|--:|--:|--:|--:|
| baseline | 31 | −2.060 | — | −2.448 | — |
| A₃ alone | 33 | −2.219 | **−0.159** | −2.644 | **−0.196** |
| xscale ratio alone | 33 | −2.086 | −0.026 | −2.338 | +0.110 |
| harmonics+offsets | 54 | −2.051 | +0.009 | −2.460 | −0.012 |
| **adopted + A₃ + ratio** | 58 | −2.041 | **+0.019** | −2.718 | **−0.270** |

**Nothing moves the confound**, and `A₃` alone makes it worse in both directions.

## What the two forms are

`:product` is `AndLayer.a3_maps` — `C₀(k)·C₀(k+1)`, emitted through `assemble` like A₁ and A₂,
so it arrives as `sqrt(pooled)`. `:ratio` is `C₀(k+1)/(C₀(k)+C₀(k+1))`, the fraction of energy at
the finer scale — **bounded and scale-free**, so numerator and denominator are pooled *separately*
and divided afterwards, which `assemble` cannot express and which is why it stays inline.

## Predictions, scored

Recorded before running: product moves the confound by < 0.05; ratio moves it by > 0.2 on both
splits; and if the ratio failed, stop proposing operators.

| | outcome |
|:--|:--|
| product < 0.05 | **wrong** — −0.159 and −0.196, i.e. it makes things worse |
| ratio > 0.2 both splits | **wrong** — −0.026 and +0.110 |

Both predictions failed. The falsification condition triggered and stands this time.

## The i.i.d. side, which did work

On the ordinary split the cross-scale features are the best thing tried: `adopted + A₃ + ratio`
reaches **thickness 0.762 and fuzziness 0.781**, against 0.714 and 0.734 for the 31-feature
baseline and 0.731 / 0.746 for `harmonics+offsets`. So the information is real and useful — it
simply does not survive a shift in the range, which is the actual problem.

## What was published and withdrawn

The first version of this page reported **+0.161 and +0.160**, replicated in both directions, and
concluded that cross-scale features move the confound in combination. That came from an inline
reimplementation of `a3_maps` in `Frontend._feat` which omitted the `sqrt` that `assemble` applies
to every A-block, making it energy-like where A₁ and A₂ are amplitude-like. Through the real path
the effect is +0.019 and −0.270.

The variant's effect was real and did replicate — but it is a **raw pooled cross-scale product**,
not `A₃`, and dropping the `sqrt` for one block while keeping it elsewhere is an inconsistency
rather than a design. It is recoverable from commit `094ed0e` if anyone wants to justify and test
it on its own terms.

**Two further claims collapse with it.** "An ingredient that does nothing alone can still matter
in company" was inferred from this artefact and has no support here. And the earlier decision to
stop proposing operators, withdrawn on the strength of the combination result, should not have
been withdrawn.
