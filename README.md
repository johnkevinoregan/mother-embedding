# mother-embedding

A modular Julia rewrite of a Vector-Symbolic-Architecture / Fractional-Power-Encoding
letter classifier built on a Gabor-filter front end. Each pipeline stage is an
independent module (`include("X.jl"); using .X`) with its own companion Pluto
notebook for visually sanity-checking that component in isolation.

Two self-contained investigations live in subfolders, both concerned with reading
**junction structure** (endpoints, corners, T-junctions, X-crossings) off a Gabor
representation. They are best read in order, because the second diagnoses why the
first cannot work:

- **`Dense_Gabors/`** — dense per-pixel Gabor sampling with a ring/peak-counting
  keypoint detector. **Superseded**; kept as a baseline and a cautionary tale.
- **`New_Gabor_FPE/`** — junction type by *linear projection only* (ray profiles →
  circular harmonics = FPE with integer frequencies). **This is the current line
  of work.**

See `PROGRESS_2026-07-17.md` for the latest writeup (and the earlier
`PROGRESS_*.md` files for the history) — what's implemented, what's still to come,
and the reasoning behind the changes.

## Requirements

- Julia 1.11 (developed against 1.11.2)
- The EMNIST dataset (IDX format), balanced split. Download from
  <https://www.nist.gov/itl/products-and-services/emnist-dataset> and place
  `emnist-balanced-train-images-idx3-ubyte` and
  `emnist-balanced-train-labels-idx1-ubyte` in `~/Julia/DATABASES/EMNIST/`
  (this default path is set in `LoadEMNIST.jl`'s `DEFAULT_DATA_DIR`; pass a
  different `data_dir` keyword to `load_emnist` to use another location).

## Setup

```bash
git clone https://github.com/johnkevinoregan/mother-embedding.git
cd mother-embedding
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

This installs this project's own package environment (Pluto, PlutoUI, Plots,
Colors, ImageFiltering, FFTW), independent of any other Julia environment on your
machine.

## Running a test notebook

Each component has a `Test_<Component>.jl` Pluto notebook. Launch Pluto
pointed at one directly:

```bash
julia --project=. -e 'using Pluto; Pluto.run(notebook="Test_LoadEMNIST.jl")'
```

`run_pluto.sh` in this repo shows the pattern used for running headless on a
remote server (`host="0.0.0.0"`, fixed `port`, `launch_browser=false`) and
viewing the notebook through an SSH tunnel from another machine — edit its
`notebook=` argument to point at whichever component you're testing.

## Project layout

```
Config.jl                     Single source of truth for shared constants
LoadEMNIST.jl                 EMNIST IDX loading, class bucketing, display-orientation fix
Test_LoadEMNIST.jl            Sanity-check notebook for LoadEMNIST
CreateGaborLifting.jl         Complex Gabor filter bank -> raw (modulus, phase) tokens
Test_CreateGaborLifting.jl    Sanity-check notebook for CreateGaborLifting
CreateTJunctionLifting.jl     T-junction detector over the Gabor grid: stem/crossbar
                              pairs scored by phase-compatibility x weaker modulus
Test_CreateTJunctionLifting.jl            Sanity-check notebook for CreateTJunctionLifting (EMNIST)
Test_CreateTJunctionLifting_SyntheticT.jl Same, on a controlled synthetic T stimulus
Test_TJunction_CornerDemo.jl  Synthetic-stimulus notebook comparing the old vs new
                              phase-compatibility term: T's and all 4 corner types
```

To change a shared constant (image size, filter scales/orientations, etc.),
edit `Config.jl` and restart the Pluto server — these are `const` bindings,
so a browser refresh alone won't pick up the change.

## Dense_Gabors/ — dense-sampling keypoint extraction (superseded)

> **Superseded by `New_Gabor_FPE/`.** The approach here works well enough to be a
> useful behavioural baseline, but it is GOFAI — thresholds, peak counting, ring
> run-length analysis — and `New_Gabor_FPE/` shows those hacks were patching a
> hole in the *representation*, not a lack of tuning. Read that section before
> building on anything here.

A self-contained side investigation, independent of the `Config.jl` /
`CreateGaborLifting.jl` pipeline above (different Gabor convention — see the
notebooks for the details): instead of a sparse grid of Gabor samples, convolve
a character with a *dense*, per-pixel bank of oriented Gabor filters and read
discrete, typed keypoints — **endpoint / corner / T-junction / X-crossing** —
directly off the resulting oriented-energy field.

```
Dense_Gabors/
  gabor_orientation_demo.py    Python source: fixed-scale argmax-orientation analysis
  Gabor_Orientation_Demo.jl    Julia/Pluto port of the above
  Gabor_Feature_Layer.jl       Julia/Pluto port of the feature-type layer below
  Gabor_Feature_Layer_MultiScale.jl  Round-2 extension: same operations at 3 Gabor
                                     scales with cross-scale voting. Documents what
                                     the multi-scale idea fixes (spurious ring-based
                                     T/X junctions) and what it doesn't (separating
                                     a curve's false corners from real ones).
  Gabor_feature_layer_python/
    feature_layer.py                        Python source: end-stopping, orientation-
                                             profile bimodality, and ring spoke-count
                                             turned into typed keypoints
    Gabor_feature_layer_design_notes.md      Design rationale for every threshold/choice
```

The approach: three cheap per-pixel operations read off the oriented-energy
stack — end-stopping (segment termination -> endpoint), orientation-profile
peak structure (bimodal -> corner-family), and a multi-radius ring "spoke
count" (junction order). The key architectural idea, laid out in the design
notes, is *propose ≠ classify*: the two dense operations propose keypoints and
decide corner-family vs. endpoint; the sparse geometric ring count only
refines an already-proposed point (corner -> T -> X), never proposes one.

## New_Gabor_FPE/ — junction type by linear projection

The current line of work. Same goal — endpoints, corners, T-junctions,
X-crossings on EMNIST letters from a Gabor lift — but using **linear/filtering
operations only**: no peak counting, no ring run-length analysis, no
`if npk >= 2`, no hand-tuned thresholds. The output is a **continuous
descriptor** suitable for FPE/VSA binding, not a symbolic label.

Verified in Julia on real EMNIST. `New_Gabor_FPE_handoff_for_claude-code.md` is the primary
document — read its §1–§3 before touching the code.

### Why the previous approaches failed

The orientation fibre `E(x, y, θ)` is **π-periodic**. Junction type is a **2π
(directional)** property — it is about *which rays leave a point*. Orientation
mod π cannot tell "ray going east" from "ray going west", so it cannot count
rays. Measured at the centre of canonical figures (cosine similarity of `E(θ)`):

```
L-corner   vs T-junction : 0.9031
L-corner   vs X-crossing : 0.8868
T-junction vs X-crossing : 0.9234
```

L, T and X are effectively **the same vector** in the orientation fibre. No
downstream FPE / attention / learning can recover what the representation never
encoded. This is also why `Dense_Gabors/` needed the ring probe: the ring was
sampling mod-2π structure *because the fibre couldn't* — a patch over a
representational hole, not a stylistic lapse.

### The fix: ray profile → circular harmonics

At each point `p`, for `φ ∈ [0, 2π)`:

```
R(p, φ) = E( p + d·u(φ),  θ = φ mod π ),      u(φ) = (cos φ, sin φ)
```

"Is there a contour at distance `d` in direction `φ`, oriented *along* `φ`?"
The `d`-offset is what turns a mod-π quantity back into a mod-2π one — east and
west read different pixels. `R` is 2π-periodic with **one lobe per branch**:
endpoint → 1, straight → 2 opposite, L → 2 adjacent, T → 3, X → 4. Its Fourier
coefficients `cₙ` are the type signature:

| config | c₀ | \|c₁\|/c₀ | \|c₂\|/c₀ |
|---|---|---|---|
| endpoint (1 ray) | 1 | 1.000 | 1.000 |
| straight (2 opposite) | 2 | 0.000 | 1.000 |
| L-corner (2 @ 90°) | 2 | 0.707 | 0.000 |
| **T-junction (3 rays)** | **3** | **0.333** | **0.333** |
| X-crossing (4 rays) | 4 | 0.000 | 0.000 |

`c₀` ≈ ray count ("how many branches leave here"); `|c₁|/c₀` ≈ asymmetry
(1 = endpoint, 0 = centrally symmetric). `(c₀, |c₁|/c₀)` alone separates all
five. Junctions are simply the **brightest points of the `c₀` map** — that is
the detector, no ring probe.

**Why this isn't GOFAI:** for a fixed `φ`, the term is one orientation channel
*rigidly shifted*. The whole expression is "K rigid shifts, weighted by
`e^{-inφ}`, summed" — a linear filter over the lifted `(x, y, θ)` field, the same
operation class as the Gabor convolution that produced `E`. `|cₙ|` is
rotation-invariant by construction, and rotation acts as binding (`V → z^α ⊙ V`).

**FPE *is* the harmonic expansion** (`V = Σ_φ R(φ)·z^φ`), so this is not a new
module bolted on — it is the FPE layer with the base fixed to integer
frequencies.

### Architecture

```
image
  └─ Gabor bank (multi-θ, multi-scale)      ─┐
  └─ modulus  → E[θ, y, x]                   │  DENSE, convolutional,
  └─ ray shift+combine → cₙ[y, x]            │  NO bundling
  └─ dense maps: c₀, |c₁|/c₀, …             ─┘
        │
        │  sparse keypoint selection (top-N of c₀)
        ▼
  └─ FPE bundle over keypoints, bound to
     object-relative polar position (r, α)   ─┐  SPARSE, bundled
  └─ object descriptor ∈ ℂ^d                 ─┘
```

The split is forced, not an optimisation: bundling `N` items into `d` dimensions
has SNR ≈ `√(d/N)`, so bundling ~600k dense tokens into `d = 1024` is pure noise,
while a few dozen keypoints is fine. (It also mirrors dense retinotopic maps in
V1/V2/V4 vs sparse object codes in IT; `(r, α)` is the Pasupathy–Connor
angular-position code.)

### Files

```
New_Gabor_FPE/
  New_Gabor_FPE_handoff_for_claude-code.md   The handoff doc -- read §1-§3 first
  ray_fpe_junctions.jl                       Pluto notebook: the core implementation
  EMNIST_Junction_Keypoints.jl               Real EMNIST + ray-harmonic keypoints
  EMNIST_Junction_GlobalShape.jl             Above + a global-shape descriptor (same
                                             construction one level up: harmonics of
                                             the mass distribution about the centroid)
  TASK_add_global_shape.md                   Spec for the global-shape extension
  rayharm.py                                 Python reference (estack, ray_profile,
                                             ray_harmonics)
  pluto2md.py                                Renders a Pluto notebook's markdown
                                             without Pluto
  figA_profiles.png                          KEY FIGURE: col 2 E(θ) -- L/T/X identical;
                                             col 3 R(φ) -- 1/2/2/3/4 lobes
  figB_signature.png                         c₀ and normalised harmonics per type
  figC_letters.png                           Dense c₀ and |c₁|/c₀ maps + polar probes
                                             on T, K
  KeyPointDiagnosticity.md                   Findings: how diagnostic are keypoints +
                                             shape harmonics of letter identity?
  KeyPointDiagnosticity.jl                   Interactive notebook reproducing every
                                             detector / encoding / scale variant
```

### Design rules (from `New_Gabor_FPE_handoff_for_claude-code.md` §8)

- **No thresholds, no counting, no `if/elif` on feature type.**
  `(c₀, |c₁|/c₀, |c₂|/c₀, …)` **is** the descriptor — bind the continuous vector
  and let similarity do the work.
- **Do not classify into symbolic labels.** A 30° corner has signature
  ≈(0.97, 0.87), close to an *endpoint* — and that is correct: a sharp corner
  really is nearly a doubled-back single ray. Corner-ness is graded, exactly as
  V4 curvature tuning is graded. The discrete labels were the GOFAI residue.
- **Never bundle globally before a locality-dependent readout.**
- **Any circular FPE variable needs integer base frequencies** (otherwise `z^θ`
  is not π-periodic — periodicity is a property of the base frequencies, not the
  exponent).
- **Orientation mod π ⇒ encode 2θ. Ray direction mod 2π ⇒ encode φ.**

### Diagnosticity experiments (`KeyPointDiagnosticity.md` / `.jl`)

A round of experiments asking whether the local keypoints and the global shape
harmonics are actually **diagnostic of letter identity** (360 EMNIST instances,
12 classes; η² per feature + leave-one-out nearest-class-mean accuracy):

- **The global shape harmonics carry the identity** — ~57 % accuracy alone
  (≈ 7× chance), rising to ~61 % once extended from `|M1..4|` to `|M1..6|` plus a
  radial (filled-vs-hollow) profile. `|M2|,|M3|,|M4|` are the strongest single
  features (η² ≈ 0.6).
- **Local keypoint *counts* are weak** (16–19 %) — a census of types discards
  *where* each keypoint is, and configuration is what separates the letters. This
  motivates the next step: bind each keypoint to its centroid-relative position
  `(r, α)` and encode the configuration, not the count.
- Detectors compared (greedy ridge-tiling vs. clear local maxima vs. a
  two-channel junction+endpoint detector) and encodings (mean-pool vs. typed
  counts). A synthetic-figure check shows the ray *signature* is correct but the
  *detector* is miscalibrated even on clean input (junction boundaries; an
  endpoint channel confounded by background asymmetry). The ray-probe scale
  `D_RAY` is the endpoint lever — larger reaches past the stroke.
- Methodological caution: "full < shape-only" in the accuracy table is a
  nearest-mean *artifact* (equal-weighted noisy features dilute good ones);
  η²-weighting flips it back (53.6 % → 61.4 %). Trust the per-feature η² and the
  shape-only accuracy, not the raw ranking.
