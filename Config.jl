module Config

# Single source of truth for project-wide constants. Include this before any
# other component module, and have that module read its defaults from here
# (e.g. `scales::Vector{Float32}=Config.SCALES`) rather than defining its own
# local constants. To change a value: edit it here, then restart the Pluto
# server (these are `const` bindings — Julia won't let them be mutated live,
# only replaced by reloading the module).

export IMG_SIZE, NATIVE_SIZE, SCALES, ORIENTATIONS, OVERLAP_FRAC

# --- Gabor lifting (CreateGaborLifting.jl) ---
const IMG_SIZE = 56
const NATIVE_SIZE = 28
const SCALES = Float32[6, 12, 24]
const ORIENTATIONS = Float32[0, π/4, π/2, 3π/4]
const OVERLAP_FRAC = 0.25f0

end # module
