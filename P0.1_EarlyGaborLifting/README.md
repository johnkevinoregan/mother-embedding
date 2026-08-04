# P0.1_EarlyGaborLifting — the first attempt

The original work, and the ancestor of everything else in this repository: a **Gabor lifting**
and a T-junction detector built directly on it. Kept because the reason it was superseded is
more useful than the code.

```
Config.module.jl                  Single source of truth for the constants this line of work uses
CreateGaborLifting.module.jl      Complex Gabor bank → raw (modulus, phase) tokens
CreateTJunctionLifting.module.jl  T-junction detector over the Gabor grid: stem/crossbar pairs
                                  scored by phase-compatibility × weaker modulus
Gabor_filter_mover.jl             Interactive filter explorer
View_GaborKernels.jl              Kernel visualisation
Test_CreateGaborLifting.jl        Companion notebook
Test_CreateTJunctionLifting.jl              on EMNIST
Test_CreateTJunctionLifting_SyntheticT.jl   on a controlled synthetic T
Test_TJunction_CornerDemo.jl                T's and all four corner types, comparing the old
                                            and new phase-compatibility terms
```

`Config.module.jl` holds the constants for *this* directory only — the current front end in
`P0-8_RationalGaborFeatures/` derives its scale ladder from measured spectra instead. These are
`const` bindings, so after editing it you must **restart the Pluto server**; a browser refresh
will not pick up the change.

`LoadEMNIST.module.jl` stays in the repository root, because everything in the project uses it.

## Why it was superseded

The detector reads a T-junction as a **pair** of Gabor responses scored by phase compatibility.
That is a hand-built template for one junction type, and extending it to corners and crossings
meant a new template each time.

The deeper problem is representational, and it is provable rather than a matter of tuning. The
orientation fibre `E(x, y, θ)` is **π-periodic**, while junction type is a **2π** property — it
is about which rays *leave* a point, and mod-π orientation cannot tell east from west. Measured
cosine similarity of `E(θ)` at the centre of canonical figures: L vs T 0.903, L vs X 0.887,
T vs X 0.923. Those three are effectively the same vector, so no scoring rule reads them apart.

The fix, developed in `P0.3_New_Gabor_FPE/` and now in
`P0-8_RationalGaborFeatures/RayHarmonics.module.jl`, is to sample the energy at an **offset** —
`R(p, φ) = E(p + d·u(φ), θ = φ mod π)` — which restores the 2π structure because east and west
read different pixels.

## What survived

The framing: that a front end should report **what a contour does at a point** — terminate,
turn, branch, cross — rather than which letter it belongs to. And `LoadEMNIST.module.jl`, still
used by every phase.
