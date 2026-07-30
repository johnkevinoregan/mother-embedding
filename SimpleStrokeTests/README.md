# SimpleStrokeTests — how to reproduce everything here

One stroke on a uniform grey field, described by eight graded properties. `RESULTS.md` has
the tables; `RESULTSexpanded.md` explains the whole thing from scratch. This file is how to
re-run it.

## What each file is

| file | kind | opens in Pluto? |
|:--|:--|:--|
| `Contours.module.jl` | module — the stimulus generator | **no** — a module, `include`d |
| `Frontend.module.jl` | module — wraps the Gabor front end | **no** — a module, `include`d |
| `Explore_Contours.jl` | **Pluto notebook** — interactive stimulus explorer | **yes** |
| `Preview_Contours.jl` | plain script — contact sheet, sweeps, dataset audit | no |
| `Phase9_Readouts.jl` | plain script — the experiment | no |
| `Plot_Phase9.jl` | plain script — figures from the saved results | no |

Opening a `.module.jl` file in Pluto rewrites it and leaves a `<name> backup 1.jl` beside
it — that is why they carry the extension.

## Interactive: the stimulus explorer

```bash
cd ~/claude-code/mother-embedding
julia --project=. -e 'using Pluto; Pluto.run()'
```

then open `SimpleStrokeTests/Explore_Contours.jl`. Sliders for every generative parameter,
with the target vector recomputed live, so you can see what any setting actually produces.

## If you have only ever used Pluto

A plain script is just a file of Julia that runs top to bottom. Two ways to run one:

**From the Julia prompt**, which is closest to Pluto and keeps the session warm so a second
run skips the ~30 s of startup and compilation:

```
julia --project=/home/kevin/claude-code/mother-embedding
julia> cd("/home/kevin/claude-code/mother-embedding/SimpleStrokeTests")
julia> include("Preview_Contours.jl")
```

**From the shell**, one command that exits when finished:

```bash
cd ~/claude-code/mother-embedding/SimpleStrokeTests
julia --project=.. Preview_Contours.jl
```

`--project=..` points Julia at the `Project.toml` one directory up — the package environment
Pluto normally activates for you. Without it, `Plots` and `Flux` will not be found. `-t 12`
gives Julia 12 threads, which matters because feature extraction runs in parallel across
images. `NAME=value` written *before* the command sets an option for that run only, which is
how the settings below are passed without editing any code.

`Ctrl-C` stops a script running in the foreground; `pkill -f Phase9_Readouts` stops a
backgrounded one.

**Or just use the runner**, which has all of this built in:

```bash
./run.sh              # list what it can do
./run.sh preview      # figures and dataset audit
./run.sh fast         # every arm except the CNN
./run.sh full         # all five arms, GPU, full-resolution CNN
./run.sh full bg      # the same, in the background, logging to a file
./run.sh repl         # an interactive Julia session in this project
```

It echoes each command before running it, so you can copy one out and vary it by hand.

## Reproducing the figures and the dataset audit

Fast — about 4 minutes, mostly the 3,000-image audit.

```bash
cd ~/claude-code/mother-embedding/SimpleStrokeTests
julia --project=.. Preview_Contours.jl
```

Writes `contactsheet.png`, `sweeps.png`, `edge_profiles.png`, and prints the target ranges,
the target correlation matrix and the trivial-cue table.

## Reading and modifying the code

These are plain text files — any editor works. VS Code with the Julia extension gives
jump-to-definition and can run a file or a single line in an attached REPL, but nothing here
depends on it.

Every function carries a docstring saying **why** it is the way it is, usually naming the
bug that forced it. From the Julia prompt:

```julia
julia> include("Contours.module.jl"); using .Contours
julia> ?excess_turn          # press ? then the name
```

### Where to change what

| to change | file | function |
|:--|:--|:--|
| stimulus ranges — thickness, blur, curvature, gap size | `Contours.module.jl` | `sample_params` |
| what the eight target numbers are | `Contours.module.jl` | `PROPS`, `targets_of` |
| how a shape becomes pixels | `Contours.module.jl` | `render_geom` |
| pooling grid (3×3 → 5×5) | `Frontend.module.jl` | `build_frontend(N; grid)` |
| the Gabor bank — scales, orientations | `Frontend.module.jl` | `LADDER`, `BETAS`, `NORI` |
| the CNN architecture | `Phase9_Readouts.jl` | `cnn_model` |
| the MLP — width, depth, learning rate | `Phase9_Readouts.jl` | `mlp` |
| how train/test splits are built | `Phase9_Readouts.jl` | `make_split` |
| which arms exist | `Phase9_Readouts.jl` | `ARMS`, `evaluate` |

### Changing something safely

**Try it small first.** Every script takes its sizes from the environment, so a change can be
exercised end to end in under a minute before committing an hour to it:

```bash
P9_NTRAIN=800 P9_NTEST=300 P9_STAGES=iid P9_ARMS=4 P9_OUT=results_scratch \
  julia --project=.. -t 8 Phase9_Readouts.jl
```

**If you touch the generator, re-run the audit.** `julia --project=.. Preview_Contours.jl`
prints the target ranges, the correlation matrix between targets, and the R² of each target
from three trivial image statistics. Every serious bug in this dataset was caught by that
table rather than by a crash — a target with no support at one end, two targets that turned
out to be the same thing, a cue that made a property guessable without looking at structure.

**If you touch the front end, run the validators** in `../RationalGaborFeatures/`:
`Validate_GaborStack.jl`, `Validate_AndLayer.jl`, `Validate_Pooling.jl`,
`Validate_RayHarmonics.jl`, `Validate_Convention.jl`. They are Pluto notebooks and each
prints `ALL GATES PASSED` or names what failed.

### Traps specific to this code

**`Plots` exports common short names.** It exports `bar` and `with`; when two modules export
the same name Julia binds *neither*, and the failure appears far from its cause. That is why
`Stimuli` has `barstim` and `Contours` has `respec`. Avoid short generic names for anything
exported.

**Never open a `.module.jl` file in Pluto.** Pluto rewrites any file it opens and leaves a
`<name> backup 1.jl` beside it. The extension exists as a warning.

**Targets are measured from the drawn geometry, not asserted from the parameters.** If you
change how a shape is built, check that its measurement still describes it — the corner angle
was wrong for a third of samples precisely because it was asserted rather than measured.

**Everything is seeded.** Any change to the generator changes every published number, even if
the change looks cosmetic. Re-run and re-check rather than assuming a table still holds.

**Julia 1.11.2 GC segfault** — see the section below.

## Reproducing the experiment

Everything is seeded, so these commands give the numbers in `RESULTS.md` exactly.

**The linear and MLP arms** — about 20 minutes. This includes the whole measurement the
experiment is built around, and the block attribution with its shuffle control.

```bash
cd ~/claude-code/mother-embedding/SimpleStrokeTests
P9_NTRAIN=12000 P9_NTEST=3000 P9_KS=500,2000,6000,12000 \
P9_EPOCHS=50 P9_ARMS=1,2,4,5 P9_OUT=results_nocnn \
  julia --project=.. -t 16 Phase9_Readouts.jl 2>&1 | tee nocnn.log
```

**All five arms including the CNN.** On CPU with the small net, 2–3 hours, ~95 % of it the
CNN. On a GPU with the full-resolution net (`P9_CNN=big`), well under an hour and the CNN is
no longer the bottleneck — feature extraction is.

```bash
P9_NTRAIN=12000 P9_NTEST=3000 P9_KS=500,2000,6000,12000 \
P9_EPOCHS=60 P9_CEPOCHS=60 P9_CNN=big P9_OUT=results_canon3 \
  julia --project=.. -t 16 Phase9_Readouts.jl 2>&1 | tee results_phase9.log
```

**The figures**, once the runs above have written their `results*/` directories:

```bash
julia --project=.. Plot_Phase9.jl
```

### Settings

| variable | default | what it does |
|:--|:--|:--|
| `P9_NTRAIN` | 16000 | training-pool size |
| `P9_NTEST` | 4000 | test-set size, generated from a separate seed |
| `P9_KS` | `500,2000,6000,16000` | training sizes for the sample-efficiency curve (nested) |
| `P9_EPOCHS` | 60 | MLP epochs |
| `P9_CEPOCHS` | 18 | CNN epochs |
| `P9_ARMS` | `1,2,3,4,5` | which arms: 1 pixels·linear, 2 pixels·MLP, 3 CNN, 4 ours·linear, 5 ours·MLP |
| `P9_CURVE_ARMS` | `1,4` | which arms appear on the sample-efficiency curve |
| `P9_STAGES` | `iid,blocks,curve,extrap` | which stages to run |
| `P9_OUT` | `results` | output directory, so parallel runs cannot clobber each other |

### Reading the `.jls` files

```bash
julia --project=.. read_results.jl                        # every file in results_canon3
julia --project=.. read_results.jl results_canon1         # another directory
julia --project=.. read_results.jl results_canon3/iid.jls # one file
```

`.jls` is Julia's own `Serialization` format — compact and exact, and unreadable without
Julia. It exists so figures can be redrawn without re-running an experiment. It is **not** an
archival format: it is tied to the Julia version and to the types in scope when it was
written, so an old file may simply refuse to load. **The `.log` files are the durable record;
treat the `.jls` as a cache.**

From the REPL directly:

```julia
using Serialization
r = deserialize("results_canon3/iid.jls")
r.R        # 5x8, arms x properties
r.arms     # arm names
r.props    # property names
```

| file | structure |
|:--|:--|
| `iid.jls`, `extrap_*.jls` | `(R, base, props, arms)` — R is arms × properties |
| `curve.jls` | `Dict{Int, Matrix}` — training size → arms × properties |
| `blocks.jls` | `(battr, shuf)` — block name → 8 values, plus the raw permutations |
| `history.jls` | `Dict{String, (val, loss, props, best)}` — `val` is epochs × properties |

**Which results directories are kept.** Only `results_canon3` (grid 3, all arms, all stages),
`results_canon1` (grid 1) and `results_canon_g2/g4/g5` (the pooling sweep) — every number and
figure in `RESULTS.md` derives from those, and `Plot_Phase9.jl` reads `results_canon3`. The
exploratory runs from development were deleted; their printed tables survive in the `.log`
files beside them, which is the readable record. Do not plot from a directory produced by an
older code state.
| `P9_GPU` | `1` | use the GPU when `CUDA.functional()`; `0` forces CPU. Falls back to CPU silently on a machine without one |
| `P9_CNN` | `small` | `small` = the two strided convolutions the CPU-era results used; `big` = full resolution, four conv stages with pooling and batch norm |

Both the CNN and the MLP arms honour `P9_GPU`. The ridge arms are closed-form and have no
epochs or device to choose. **Feature extraction is still CPU-only** — FFT-based, ~19 ms per
image, so ~5 minutes per 15,000-image split, and now the slowest part of a GPU run.

A single stage, for example just the block attribution:

```bash
P9_STAGES=blocks P9_ARMS=4 P9_OUT=results_blocks julia --project=.. -t 16 Phase9_Readouts.jl
```

## What guarantees the comparisons are fair

Set out in full in `RESULTSexpanded.md` §7. In short: the generator is parametric and
unbounded, so a test image is a fresh draw and overlap with training is structurally
impossible rather than merely unlikely; train and test are generated once and handed to
every arm, and the features are computed once and reused across all training sizes; the
ridge penalty and the training epoch are chosen on a validation slice carved out of
*training*, never from test; and in each extrapolation split the held-out nuisance's own
target row is dropped, since nothing can be scored on predicting a constant.

## A Julia 1.11.2 quirk to know about

Running the generator repeatedly in one process segfaults inside the garbage collector
roughly one time in three (`gc_mark_obj8`). Forcing bounds checking on
(`julia --check-bounds=yes`) makes it disappear over repeated trials, and every isolated
case runs cleanly, so it looks like a GC bug in this Julia version rather than an
out-of-bounds write in this code.

It does not affect the results: a segfault kills the process rather than corrupting output,
and the numbers in `RESULTS.md` were reproduced identically to three decimals by **three
independent runs** (`nocnn.log`, `nocnn2.log`, `results_phase9.log`). If a long run dies
unexpectedly, re-run it, or add `--check-bounds=yes` at some cost in speed.

## A rule this codebase learned twice

**No absolute epsilon anywhere in the front end.** Any conditioning constant added to a
denominator must be *relative* to a local energy.

It has now been the cause of two bugs. `A₂` originally used an absolute ε and collapsed to
plain energy (`RationalGaborFeatures/RESULTS.md`), fixed with a relative `κ·E(x)`. The ray
transform kept the absolute form — `c₀ > 1e-12`, writing 0 where it failed — until the same
problem was found again. Writing 0 there is not a neutral fallback: in a normalised quantity
it asserts a specific value at locations with no evidence, and it makes the operator behave
differently on line drawings than on photographs.

Related: **form ratios after pooling, not before.** Pooling a per-pixel ratio weights noisy
low-energy locations equally with strong ones; pooling numerator and denominator separately
and dividing afterwards is defined everywhere and energy-weights itself.

## Known limitations

Read `RESULTS.md` §"`closedness` is confounded" before quoting that column — it is not a
measure of closure. The CNN is undertrained at 12 CPU epochs and its numbers are a floor.
One seed per arm, so read a difference of 0.4 as real and 0.02 as noise.
