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

## Status

Phases 0–4 complete and gated (6 + 6 + 5 checks). Next: Phase 5a — reproduce the existing
92.30 % before believing any improvement — then the ablation.
