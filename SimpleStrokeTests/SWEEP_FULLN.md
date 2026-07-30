# Capacity sweep — full-n confirmation

> **Start with [`FINDINGS.md`](FINDINGS.md)** — a plain-language summary of what these
> experiments found. This file is the detailed tables.


`Sweep_Capacity.jl` with `SW_NTRAIN=16000 SW_NTEST=4000 SW_EPOCHS=100`, log in `confirm.log`.
Grid 1, MLP readout, four splits. Supersedes the reduced-n numbers in `SWEEP.md`.

**Two cross-checks passed.** The baseline reproduces `ConVNextTest`'s independently-computed
`ours·MLP` to **≤ 0.006** on both the i.i.d. and polarity splits — two harnesses, same images,
same protocol. That run also predates the A₁ analytic-floor default, so it doubles as a
measurement of that change: **the floor subtraction moves the published tables by ≤ 0.006**, far
inside run-to-run noise. The reproduction flag is still right to keep, but the risk was much
smaller than the warnings claimed.

## Δ from baseline, all four splits

| arm | nfeat | split | curved | broken | vangle | arms | thick | fuzzy |
|:--|--:|:--|--:|--:|--:|--:|--:|--:|
| harmonics +C₆C₈ | 36 | i.i.d. | +0.026 | −0.005 | +0.007 | +0.001 | −0.015 | −0.017 |
| | | polarity | +0.031 | +0.002 | +0.002 | 0.000 | −0.019 | −0.019 |
| | | blur | +0.028 | +0.036 | +0.030 | +0.002 | −0.051 | — |
| | | thickness | **+0.075** | +0.010 | +0.011 | +0.003 | — | −0.005 |
| offsets ×3 | 49 | i.i.d. | +0.018 | +0.026 | +0.018 | +0.015 | +0.029 | +0.032 |
| | | polarity | +0.017 | +0.025 | +0.011 | +0.012 | +0.028 | +0.031 |
| | | blur | +0.003 | +0.047 | +0.045 | +0.026 | −0.018 | — |
| | | thickness | +0.065 | +0.044 | +0.024 | 0.000 | — | **−0.151** |
| **harmonics+offsets** | 54 | i.i.d. | +0.027 | +0.022 | +0.016 | +0.011 | +0.017 | +0.012 |
| | | polarity | +0.033 | +0.020 | +0.016 | +0.014 | +0.022 | +0.031 |
| | | blur | +0.037 | +0.060 | +0.049 | +0.018 | +0.009 | — |
| | | thickness | **+0.080** | **+0.075** | +0.030 | −0.002 | — | −0.012 |

## Verdict: adopt `harmonics + offsets` (54 features)

**Its value is robustness, not i.i.d. accuracy.** Gains against baseline grow monotonically with
distribution shift — ~+0.02 i.i.d., +0.04–0.06 under blur, +0.08 under thickness shift. On the
i.i.d. split alone `offsets` by itself is marginally better, and judged there the combination
would have been rejected.

**And the combination is only additive under shift.** On i.i.d. it is slightly *worse* than
`offsets` alone on five of six rows — the two axes are largely the same information by two routes.
Under shift it becomes genuinely additive: it keeps offsets' geometry gains **without offsets'
fuzziness penalty** (−0.012 against −0.151 under thickness shift). That is the single strongest
reason to take the combination over `offsets` alone.

**Polarity invariance is untouched** by either axis: every arm's polarity-split numbers are ≥ its
own i.i.d. numbers.

## What did NOT happen

**The thickness/fuzziness confound is unmoved.** Every arm sits at −2.05 to −2.11 on the fuzziness
split and −2.45 to −2.60 on the thickness split, against baselines of −2.060 and −2.448. The
reduced-n sweep's +0.25 was an artefact.

This strengthens rather than weakens the case for a genuinely different operator. Both properties
are about **how energy is distributed across scale at one location**, and no amount of finer
sampling along the existing axes separates them — 5 scales made it worse, more offsets did nothing.
`AndLayer` already implements `:A3`, cross-scale conjunction, currently **off by default** with the
comment *"not because any argument requires it"*. There is now an argument requiring it.

## Method note worth carrying forward

Reduced-n selection inflated every gain 2–5× and reversed one sign. If a sweep of this kind is run
again, treat reduced n as a filter for *which arms to confirm*, never as a source of effect sizes —
and confirm on the extrapolation splits, since the i.i.d. split alone would have picked the wrong
configuration here.
