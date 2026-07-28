# RationalGaborFeatures

A **bio-inspired early-vision front end**: a dense oriented-energy stack with pointwise
conjunctions, applied before any spatial pooling.

Nothing here imports EMNIST. The scale ladder is *derived from the data* by
`radial_spectrum` + `scale_ladder`, so retargeting to another dataset means re-running two
functions, not editing constants.

## Which files open in Pluto

| file | kind | |
|:--|:--|:--|
| `GaborStack.module.jl` | **plain module** | ⚠ do not open in Pluto |
| `AndLayer.module.jl` | **plain module** | ⚠ do not open in Pluto |
| `Stimuli.module.jl` | **plain module** | ⚠ do not open in Pluto |
| `Pooling.module.jl` | **plain module** | ⚠ do not open in Pluto |
| `Validate_GaborStack.jl` | **Pluto notebook** | ✅ |
| `Validate_AndLayer.jl` | **Pluto notebook** | ✅ |
| `Validate_Pooling.jl` | **Pluto notebook** | ✅ |

The four modules are `include`d and `using`-ed by the notebooks, so they must stay plain
`.jl` — a Pluto notebook cannot supply `module … end`. **Opening a plain module in Pluto
rewrites the file and leaves a `<name> backup 1.jl` beside it.** Each one carries a marker
comment on its first line as a reminder.

Both notebooks also run headless as gates:

```bash
julia --project=. RationalGaborFeatures/Validate_GaborStack.jl
julia --project=. RationalGaborFeatures/Validate_AndLayer.jl
julia --project=. RationalGaborFeatures/Validate_Pooling.jl
```

## Design, and the measurements behind it

**Filters are built in the frequency domain** as one-sided (analytic) bumps. There is no
spatial kernel to truncate, the bank is size-agnostic, and even/odd quadrature is exact —
so `|·|²` is the complex-cell energy and **polarity invariance is exact, not approximate**.

**log-Gabor by default.** Zero DC exactly (`log 0 = −∞`), and bandwidth freedom decouples
the spatial envelope from the wavelength. That decoupling is what lets a filter tuned to a
coarse wavelength still localise finely — `σ_x = 7.6 px` at `λ = 16 px` against a 12.7 px
stroke — and it is what makes end-stopping viable at all on 28×28 images.

**The ladder comes from EMNIST's measured spectrum**, not from convention: 94 % of the
energy sits at `λ ≥ 8 px` (112-grid) and only 3.9 % above `ρ = 8`. Hence
`ρ = 2.00 / 3.74 / 7.00` → `λ@112 = 56 / 30 / 16`. For comparison, the old `Config.SCALES
= [3, 6, 12, 24]` put one channel *past Nyquist entirely* and another where ~1 % of the
energy lives.

**`dtheta_on_sigma = 0.75`, not Kovesi's 1.5.** This parameter — *not* the orientation
count — controls spatial elongation, since `σ_along = W/(2πρ₀σ_φ)` and `σ_φ = (π/n)/dts`,
so `n` cancels out of angular coverage. At 1.5 the coarse channel has `σ_along = 34 px`,
**longer than any EMNIST stroke**, so no stroke ever looks uniform and end-stopping cannot
work. The field also drops 320 → 224.

**The AND layer is the point.** Multiplication and pooling do not commute; their difference
is the within-window covariance `Cov_x(e₁,e₂)`, which *is* the co-location signal. No
statistic of already-pooled orientation energy can represent it — including a
Squeeze-and-Excitation gate, which multiplies whole channels by scalars derived from
*global* average pooling and so has no spatial resolution at all.

## Measured

| | |
|:--|--:|
| A₁ at a junction, crossing vs separated bars | 10.3× / 7.1× / **17.6×** |
| A₁ across the gap sweep | 1.00 → **0.06** |
| …while pooled orientation energy | 1.00 → **0.97** |
| A₂ end/interior, finest scale | **10.4×** |
| A₂ end/flank · end/blob | 4.4× · 4.8× |
| A₁ by ray count (straight/L/T/X) | 6.3e4 / 9.5e4 / 1.15e5 / **1.58e5** |

The last row was not designed for: A₁ orders junctions by ray count unprompted, which is
the `F`/`f` distinction directly (3-ray T versus 4-ray X) — and `F`/`f` is **17.3 % of all
remaining error** on merged-class EMNIST.

## Pooling comes last

`select → multiply → pool` — simple cell → complex cell, and conv → ReLU → pool. **Detect,
then tolerate.** Pooling is not the enemy; pooling *first* is. The old Fourier grid pooled
via its Gaussian window and only then combined orientations into `|E₄|`, and no downstream
capacity can undo an averaging that has already happened.

Measured: a 4 px shift (about a third of a stroke width) leaves the 3×3 vector at
**cos = 0.998**, against 0.995 at 6×6 and 0.994 at 11×11 — finer grids keep more detail and
tolerate less, which is the trade the grid parameter prices.

Default feature vector, `grid = 3`, blocks `(:orient, :lowpass, :A1, :A2)`:

| block | columns | |
|:--|--:|:--|
| `:orient` | 135 | orientation harmonics of the **pooled** energy — the deficient baseline |
| `:lowpass` | 9 | |
| `:A1` | 27 | conjunction computed pointwise, **then** pooled |
| `:A2` | 27 | end-stopping, likewise |
| **total** | **198** | |

Every ablation arm has a `shuffle_block!` twin, which permutes a block's rows across
samples — same marginals, same column count, correspondence destroyed. §7.8 showed a fixed
projection into a few hundred dimensions plus a trained head reaches 75 % *whatever the
projection is*, so without that control "A₁ helped" and "27 more columns helped" are
indistinguishable.

## Extraction cost

**20.9 ms per image**, so the full 131,600-image split takes about **6 minutes** on 8
threads. `energy_stack` is 16.4 ms of that and sits at the FFT floor; the conjunctions are
1.9 + 2.2 ms.

Getting there took two fixes worth remembering, both worth 10× or more and neither visible
in any output:

* the AND loops iterated **pixel-outer with the channel as the last array index**, so every
  channel access strided 12,544 floats and missed cache — 190 ms for A₁ against 8 ms for
  the same arithmetic channel-outer;
* `bank.meta` is a `Vector{NamedTuple}` (abstract), so `m.theta` returned `Any` and made
  every `@view E[:,:,that]` type-unstable — another 8× on top. `scale_channels` now
  annotates its return type.

## Phase 5a — EMNIST

`Phase5a_EMNIST.jl`, full official split, 40 homoglyph-merged classes, the section 7.10
classifier unchanged (`Dense(n=>256,relu) → Dense(256=>40)`, Adam 1e-3, batch 128, 15
epochs, seed 1).

**The reference arm came first**, and it landed exactly: the old F3×3+2 no-DC features,
re-extracted and re-trained in this harness, give **92.30 %** against the 92.30 % recorded
in section 7.10. The harness is sound, so the rest of the table can be read.

| arm | n | final | best |
|:--|--:|--:|--:|
| reference — old F3×3+2 no-DC | 88 | 92.31 % | 92.42 % |
| new: `orient` + `lowpass` | 144 | **93.71 %** | 93.78 % |
| + `A1` | 171 | 93.78 % | 93.78 % |
| + `A1` + `A2` | 198 | 93.71 % | 93.71 % |

![Phase 5a curves](figures/phase5a_curves.png)

**The new front end is worth +1.40 points** over the old features — about 8 standard errors,
so real. That gain comes from the bank itself: a scale ladder placed on the measured
spectrum, more orientations, correct padding.

**The AND layer adds nothing: +0.07 for A₁, and −0.12 once A₂ joins it.** This is the
layer the whole design argument was built around, and on this task it does not pay.

### …but not because it fails to compute anything

Each block trained alone:

| block alone | n | accuracy |
|:--|--:|--:|
| `orient` | 135 | 93.59 % |
| **`A1` + `A2`** | **54** | **88.45 %** |
| `A2` | 27 | 78.27 % |
| `A1` | 27 | 75.63 % |
| `lowpass` | 9 | 62.15 % |

**54 conjunction features alone reach 88.45 %.** They are strongly informative — they are
simply *redundant* with what the pooled orientation statistics already carry. Not absent:
redundant.

That is the outcome the theory should have predicted, and did, before a confusion analysis
talked us out of it. At a fixed spatial resolution the orientation profile is fully
described by its harmonics, and products of energies are functions of that same profile;
the non-commutation of multiply and pool only buys something when there is sub-cell
structure the covariance can see *and* the classes actually depend on it. On EMNIST the
second condition fails.

### The targeted check agrees

Errors on the pairs the confusion analysis said A₁ should fix:

| pair | old | new | new + AND |
|:--|--:|--:|--:|
| **F / f** | 251 | 265 | 257 |
| 0 / D | 37 | 32 | 28 |
| T / t | 34 | 27 | 33 |
| 4 / Y | 30 | 27 | 26 |
| U / V | 26 | 26 | 27 |
| C / e | 20 | 14 | 19 |
| X / Y | 10 | 6 | 7 |
| K / h | 9 | 4 | 6 |
| **total** | **417** | **401** | **403** |

`F`/`f` — 17 % of all remaining error, and the motivating case — is **not** improved. Total
errors fall from 1,448 to 1,176, but junction-pair errors stay flat, so they *rise* as a
share of the total from 28.8 % to 34.1 %. The new bank fixes non-junction errors and leaves
the junction errors exactly where they were.

### A prediction that was wrong

Original estimate: **+0 to +0.5**, on the grounds that the task does not reward i2D
structure. Revised to **+0.5 to +1.5** after the confusion analysis showed the residual
errors were junction-distinguishable. Measured: **−0.12**.

The revision was the mistake. "Distinguishable in principle" is not "this feature adds
something", because the information was already present in another form. This is the same
error made repeatedly in this project — over-predicting that a representational addition
will pay off — and the confusion analysis, which felt like hard evidence, made it worse
rather than better.

## Status

Phases 0–5a complete. The front end is validated, faster than the old one, and **+1.4
points better** — but its distinctive contribution, the conjunction layer, is redundant on
EMNIST rather than useful.

Open, in the order that would settle the most:

1. **Is it the pooling?** A₁ is a point property read out over 37 px cells. A finer grid for
   the A blocks alone would separate "the conjunction is redundant" from "3×3 pooling
   destroys it". One re-extraction, ~9 minutes.
2. **The shuffled-block control** is now moot for A₁/A₂ (there is no gain to attribute) but
   still wanted for the +1.40, which is 56 extra columns as well as a better bank.
3. **A task that requires i2D structure.** The negative result here is about EMNIST, not
   about the layer: `A1+A2` alone at 88.45 % from 54 numbers says the signal is real.
