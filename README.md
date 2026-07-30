# mother-embedding

Building a **bio-inspired early-vision front end** in Julia — dense oriented-energy
(log-Gabor) representations, with operators for the things a contour does at a point:
terminate, turn, branch, cross. The aim is a *general* front end, testable on images
generally rather than tuned to one dataset, so most of the work is about telling a real
result apart from an artefact of the benchmark.

Each pipeline stage is an independent module (`include("X.module.jl"); using .X`) with a
companion Pluto notebook for sanity-checking it in isolation.

**Start with [`PHASES.md`](PHASES.md).** It maps the current line of work — what each phase
asked, what it found, and what was later corrected. [`PROGRESS_2026-07-30.md`](PROGRESS_2026-07-30.md)
is the latest narrative report; the earlier `PROGRESS_*.md` files carry the history.

---

## Where everything is

**Current work.** Three directories, numbered as phases, each with its own `README.md` for
how to run it and `RESULTS.md` for the tables.

| directory | phases | what it is |
|:--|:--|:--|
| **`RationalGaborFeatures/`** | 0–8 | **The front end itself**, and its tests on EMNIST. Log-Gabor oriented energy, exactly polarity-invariant, with an ablatable layer of pointwise conjunctions applied *before* spatial pooling. Scale ladder derived from measured spectra rather than hard-coded. |
| **`SimpleStrokeTests/`** | 9 | A synthetic dataset labelled with **graded properties** rather than classes, which turns the question from "can a classifier separate these" into "how much of each property is linearly available". Where the conjunction layer first paid for itself. |
| **`FashionMNIST/`** | 10 | The first dataset here that is **not a line drawing** — filled silhouettes with texture, and published baselines to calibrate against. |

**Earlier investigations**, each self-contained, in rough order of age. They are worth reading
because several were superseded for *diagnosable reasons*, and those diagnoses shaped what came
after.

| directory | what it tried | outcome |
|:--|:--|:--|
| `ExptsWithGlobalFourier/` | Describe a character with low-order 2D **Fourier** coefficients, globally and on a 3×3 grid | The 3×3 "tic-tac-toe" signature became the reference feature set that `RationalGaborFeatures/` had to beat |
| `ExptsWithZernike/` | Describe it with **Zernike moments** on a disc | Better features, *worse* classification — and the reason why is instructive |
| `TestFeaturesWithMLP/` | Score those features with a **real** classifier on the official EMNIST split instead of a deliberately weak one | Overturned earlier conclusions; see the methodological warning below. `README_MLP_FPE_Experiment.md` is self-contained |
| `New_Gabor_FPE/` | Junction type by **linear projection only** — ray profiles → circular harmonics | The ray transform survives into the current front end. Diagnosed *why* orientation energy alone cannot count rays |
| `Dense_Gabors/` | Dense per-pixel Gabor sampling with peak-counting and ring analysis | **Superseded.** Kept as a baseline and a cautionary tale: its thresholds were patching a hole in the representation |

| `EarlyGaborLifting/` | The **first attempt** — a Gabor lifting and a hand-built T-junction detector | Superseded, and the diagnosis of *why* is what produced the ray transform. See the last section |

---

## Two results you should know before trusting any older number

**The evaluation protocol was wrong for a long time.** Leave-one-out nearest-class-mean, used
throughout the early work, **understates features by ~24 points** and in one recorded case
manufactured a qualitative conclusion that a stronger classifier does not reproduce. Every
accuracy comparison predating `TestFeaturesWithMLP/` should be read with that in mind —
`PROGRESS_2026-07-26.md` §9 has the details.

**Orientation energy cannot count rays, and this is provable rather than empirical.** The
orientation fibre `E(x, y, θ)` is **π-periodic**, while junction type is a **2π** property —
it is about which rays *leave* a point, and mod-π orientation cannot distinguish east from
west. Measured cosine similarity of `E(θ)` at the centre of canonical figures:

```
L-corner   vs T-junction : 0.9031
L-corner   vs X-crossing : 0.8868
T-junction vs X-crossing : 0.9234
```

L, T and X are effectively the same vector. No amount of downstream learning recovers what the
representation never encoded — which is why `Dense_Gabors/` needed its ring probe, and why the
**ray transform** replaced it:

```
R(p, φ) = E( p + d·u(φ),  θ = φ mod π ),      u(φ) = (cos φ, sin φ)
```

*"Is there a contour at distance `d` in direction `φ`, oriented along `φ`?"* The offset turns a
mod-π quantity back into a mod-2π one, because east and west read different pixels. `R` has one
lobe per branch, and its circular harmonics are the type signature:

| configuration | c₀ | \|c₁\|/c₀ | \|c₂\|/c₀ |
|:--|--:|--:|--:|
| endpoint (1 ray) | 1 | 1.000 | 1.000 |
| straight (2 opposite) | 2 | 0.000 | 1.000 |
| L-corner (2 at 90°) | 2 | 0.707 | 0.000 |
| **T-junction (3 rays)** | **3** | **0.333** | **0.333** |
| X-crossing (4 rays) | 4 | 0.000 | 0.000 |

`c₀` ≈ ray count, `|c₁|/c₀` ≈ asymmetry. This is `RationalGaborFeatures/RayHarmonics.module.jl`
today, and its gates are in `Validate_RayHarmonics.jl`.

---

## Conventions

**`*.module.jl` is a plain Julia module, not a notebook.** Everything else ending `.jl` is
either a Pluto notebook or a plain script, and each says which on its first line.

The distinction matters: **opening a module in Pluto rewrites the file** and leaves a
`<name> backup 1.jl` beside it, which is easy to do by accident.

Notebooks come in two flavours. `Test_*.jl` sanity-check a component interactively;
`Validate_*.jl` in `RationalGaborFeatures/` do the same but also run **headless as gates**,
printing `ALL GATES PASSED` or naming what failed. Run those after touching the front end.

Two shared names to avoid: `Plots` exports `bar` and `with`, and when two modules export the
same name Julia binds **neither** — which is why `Stimuli` has `barstim` and `Contours` has
`respec`.

## Requirements and setup

Julia 1.11 (developed against 1.11.2). Optionally CUDA — the Phase 9 harness uses a GPU when
`CUDA.functional()` and falls back to CPU silently otherwise.

```bash
git clone https://github.com/johnkevinoregan/mother-embedding.git
cd mother-embedding
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Datasets go under `~/Julia/DATABASES/`:

- **EMNIST** (balanced, IDX format) in `EMNIST/` — from
  <https://www.nist.gov/itl/products-and-services/emnist-dataset>. The default path is
  `DEFAULT_DATA_DIR` in `LoadEMNIST.module.jl`.
- **Fashion-MNIST** in `FashionMNIST/` — see `FashionMNIST/README.md` for the download.

## Running things

A notebook:

```bash
julia --project=. -e 'using Pluto; Pluto.run()'
```

then open whichever `.jl` you want. `run_pluto.sh` shows the pattern for running headless on a
remote server and viewing through an SSH tunnel.

A plain script — note `--project=..` from inside a subdirectory, and `-t` for threads, since
feature extraction is parallel across images:

```bash
cd SimpleStrokeTests && julia --project=.. -t 16 Phase9_Readouts.jl
```

`SimpleStrokeTests/run.sh` wraps the common cases. Experiment scripts take their settings from
environment variables (`P9_*`, `F_*`) so a run can be resized without editing code.

---

## The original work — `EarlyGaborLifting/`

The first attempt, and the ancestor of everything above: a **Gabor lifting** and a hand-built
T-junction detector reading stem/crossbar pairs scored by phase compatibility. It has its own
`README.md` explaining why it was superseded — briefly, extending a template per junction type
does not scale, and the π-periodicity result above says no scoring rule on the orientation
fibre alone can separate L from T from X.

**Only `LoadEMNIST.module.jl` remains in this root directory**, because every phase uses it.
Everything else from that period moved into `EarlyGaborLifting/` on 2026-07-30; the move was
safe because nothing outside those notebooks ever included them — earlier greps suggesting
otherwise were matching the module names in prose.
