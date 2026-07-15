# mother-embedding

A modular Julia rewrite of a Vector-Symbolic-Architecture / Fractional-Power-Encoding
letter classifier built on a Gabor-filter front end. Each pipeline stage is an
independent module (`include("X.jl"); using .X`) with its own companion Pluto
notebook for visually sanity-checking that component in isolation.

See `PROGRESS_2026-07-08.md` for a detailed writeup of what's implemented so far
and what's still to come.

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
Colors, ImageFiltering), independent of any other Julia environment on your
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

## Dense_Gabors/ — dense-sampling keypoint extraction

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
                                     a curve's false corners from real ones) -- an
                                     open problem, not swept under the rug.
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
