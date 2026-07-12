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
Gabor_Orientation_Demo.jl     Standalone notebook: fixed-scale argmax-orientation
                              analysis (flow field, mask, modulus(theta) profiles,
                              even/odd phase) -- Julia port of gabor_orientation_demo.py
```

To change a shared constant (image size, filter scales/orientations, etc.),
edit `Config.jl` and restart the Pluto server — these are `const` bindings,
so a browser refresh alone won't pick up the change.
