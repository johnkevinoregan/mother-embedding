# Capacity sweep — scales, orientation harmonics, orientation count, ray offsets

> **Start with [`FINDINGS.md`](FINDINGS.md)** — a plain-language summary of what these
> experiments found. This file is the detailed tables.


> ## ⚠ RETRACTION — read `SWEEP_FULLN.md` first
>
> This page reports a **reduced-n selection sweep** (3,000/1,000). A full-n confirmation
> (16,000/4,000) showed its gains are inflated 2–5× and that its headline claim is **false**:
>
> * *"Orientation count and ray offsets both move the thickness/fuzziness confound by ~+0.25 —
>   the first things in the project to touch it."* At full n, `offsets ×3` moves it by
>   **−0.018**. **Nothing here touches the confound.**
> * `offsets` brokenness gain: **+0.130 → +0.026**. Harmonics curvedness: **+0.058 → +0.026**.
>
> The *rankings* below largely survive; the *magnitudes* do not, and the confound claim is
> withdrawn entirely. Reduced-n selection systematically overstates capacity gains, because
> capacity matters most when the baseline is data-starved.

`Sweep_Capacity.jl`, log in `sweep.log`. 3,000 train / 1,000 test, grid 1, MLP readout, 60 epochs,
one seed. Stimuli are a **prefix of the files in `ConVNextTest/data`**, so every arm sees
byte-identical images.

**Why these axes and not the grid.** The pooling grid was swept in Phase 9 (1–5) and grid 1 won on
every property, because stroke position is randomised and a fixed grid is pure liability. These
three axes carry no such penalty, so they can add capacity without it. They had never been swept.

**Reduced n for selection.** Absolute numbers are well below the published tables — baseline
`brokenness` is 0.463 here against 0.730 at 16,000 images — so these are for *ranking arms*, not
for quoting. With one seed, |Δ| below ~0.05 is at the noise floor.

## Δ from baseline, all three splits

| arm | nfeat | | curved | broken | vangle | arms | thick | fuzzy |
|:--|--:|:--|--:|--:|--:|--:|--:|--:|
| **harmonics +C₆C₈** | 36 | i.i.d. | **+0.058** | +0.043 | +0.031 | +0.006 | −0.021 | −0.028 |
| | | blur | **+0.060** | −0.021 | +0.041 | −0.023 | −0.077 | — |
| | | thick | **+0.045** | +0.029 | **+0.120** | 0.000 | — | −0.013 |
| **orient ×2 +C₆C₈** | 37 | i.i.d. | +0.058 | +0.019 | +0.019 | 0.000 | −0.025 | −0.031 |
| | | blur | **+0.078** | **+0.062** | +0.036 | −0.006 | **+0.237** | — |
| | | thick | +0.054 | **+0.116** | +0.076 | −0.044 | — | **+0.073** |
| **offsets ×3 crossed** | 49 | i.i.d. | +0.002 | **+0.130** | **+0.048** | **+0.030** | **+0.038** | **+0.036** |
| | | blur | 0.000 | +0.020 | +0.021 | +0.032 | **+0.275** | — |
| | | thick | +0.014 | **+0.154** | **+0.137** | −0.007 | — | −0.166 |
| scales 5 | 51 | i.i.d. | +0.002 | +0.054 | +0.008 | +0.016 | −0.028 | −0.022 |
| | | blur | −0.005 | −0.025 | +0.023 | −0.004 | −0.103 | — |
| | | thick | −0.001 | +0.120 | +0.068 | −0.010 | — | −0.547 |

## Verdicts

**Harmonics C₆/C₈ — accept.** Five extra features, and the curvedness gain replicates across all
three conditions (+0.058 / +0.060 / +0.045) with vangle behind it. The most solid result in the
sweep, and the cheapest. Mechanistically expected: curvature is about how the orientation profile
*spreads*, and C₂/C₄ cannot describe two-lobe structure with independent amplitudes.

**Ray offsets, crossed d × λ — accept for structure.** Largest gains on brokenness (+0.130 i.i.d.,
+0.154 under thickness shift) and vangle, and best on thickness-under-blur (+0.275). Two caveats:
its headline i.i.d. brokenness gain **shrinks from +0.130 to +0.020 under blur**, so part of it was
fitted to the nuisance distribution; and it makes fuzziness-under-thickness-shift worse (−0.166).

**Orientation count ×2 at constant σφ — accept, but for robustness only.** Buys **nothing i.i.d.**
(identical to harmonics-alone on curvedness, slightly worse elsewhere) for 2× the channels and 2×
the extraction time. Under *both* shifts it is the broadest gainer, and the only arm that improves
the confound pair in both directions (+0.237 thickness-under-blur, +0.073 fuzziness-under-thickness).
Plausibly because σφ is fixed, so the extra channels buy finer angular *sampling* and better
harmonic estimates — in-distribution the readout absorbs sampling error, out of distribution it
cannot.

**Scales 3 → 5 — reject.** Inconsistent brokenness gains, and it makes the thickness/fuzziness pair
worse in **all four** tests, catastrophically so on fuzziness-under-thickness-shift (−0.547). It is
also the least trustworthy arm: its `betas` were interpolated by me rather than derived from the
data's spectrum, against Phase 0's discipline. If revisited, derive them with `scale_ladder`.

## Predictions, scored

| prediction | outcome |
|:--|:--|
| offsets → `arms` and `brokenness` | **correct**, brokenness by a wide margin |
| harmonics → `vangle` | **correct in direction**, though curvedness gained more |
| scales → `thickness`/`fuzziness` | **wrong, and consistently backwards** — worse in 4/4 tests |
| orientations → further `vangle` gain | **wrong i.i.d.**; right that they help, but on robustness and on other rows |

## The one thing that moved the confound

Phase 11 found thickness and fuzziness confounded in our representation and nothing had touched it.
**Orientation count and ray offsets both move it by ~+0.25.** Scales — the axis predicted to fix
it — makes it worse, which argues the confound is **not** about scale-sampling resolution. A
blurred thin stroke and a sharp thick stroke may produce near-identical oriented-energy scale
distributions at any sampling density, in which case separating them needs something sensitive to
the *edge profile* itself: phase congruency, or the energy ratio across the stroke's two edges.

## Next

**Confirm `harmonics +C₆C₈` combined with `offsets ×3` at full n** — they gain on disjoint rows
(curvedness/vangle versus brokenness/arms), ~54 features, and neither needs the extra channel cost.
Then test whether `orient ×2` on top is worth 2× extraction for its robustness gain. Any winner
must hold on the extrapolation splits, not just i.i.d.
