module CreateTJunctionLifting

include(joinpath(@__DIR__, "Config.jl"))
using .Config

export TJunctionSample, t_junction_lift

# A T-junction candidate: a stem/crossbar pair found at grid point P, pointing
# toward one of P's 8 grid neighbors. Unlike Gabor orientation (π-periodic),
# α is 2π-periodic: it distinguishes e.g. a stem pointing left from one
# pointing right, even though both share the same underlying Gabor
# orientation (α mod π).
struct TJunctionSample
    x::Float32
    y::Float32
    α::Float32        # 2π-periodic direction from the stem point toward the crossbar point
    s::Float32         # scale index
    strength::Float32
    phase::Float32    # phase of the stem Gabor sample
end

# The 8 grid-neighbor directions, as (dx, dy) in grid-step units, and the
# angle each corresponds to.
const NEIGHBOR_DIRECTIONS = [
    ( 1,  0, 0f0),
    ( 1,  1, Float32(π/4)),
    ( 0,  1, Float32(π/2)),
    (-1,  1, Float32(3π/4)),
    (-1,  0, Float32(π)),
    (-1, -1, Float32(5π/4)),
    ( 0, -1, Float32(3π/2)),
    ( 1, -1, Float32(7π/4)),
]

# Closest entry of `orientations` to `target`, both interpreted mod π (Gabor
# orientation is π-periodic) — so this works for any orientation count or
# spacing, not just a set closed under 90° rotation.
function closest_orientation(target::Float32, orientations::Vector{Float32})
    best = orientations[1]
    best_dist = abs(rem(target - best, Float32(π), RoundNearest))
    for o in orientations[2:end]
        d = abs(rem(target - o, Float32(π), RoundNearest))
        if d < best_dist
            best_dist = d
            best = o
        end
    end
    return best
end

"""
    t_junction_lift(gabor_samples; scales=SCALES, orientations=ORIENTATIONS,
                     overlap_frac=OVERLAP_FRAC, img_size=IMG_SIZE)

Given the (unthresholded) output of `gabor_lift` for one image, detect
T-junction candidates directly from the Gabor grid, without going through FPE
encoding. For every sampled grid point P and each of its 8 grid neighbors N,
treat P as a candidate stem (Gabor orientation = direction P->N, mod π) and N
as a candidate crossbar (Gabor orientation orthogonal to that, mod π). Each
lookup uses whichever available `orientations` entry is closest to the target
angle, so this works with any orientation count/spacing.

Combines stem and crossbar into a strength via a phase-compatibility measure
(large when the two "look the same" — same line/edge polarity, e.g. two
orthogonal dark-on-light strokes) and the weaker of the two moduli (so both
must actually be present, not just one).

`gabor_samples` must have been produced with the same `scales`/`orientations`/
`overlap_frac`/`img_size` passed here (defaults match `gabor_lift`'s own
defaults).

Returns one `TJunctionSample` per (grid point, neighbor direction) pair that
falls within the image — up to 8 per grid point, fewer at the border. No
thresholding is applied, matching `gabor_lift`'s own philosophy.
"""
function t_junction_lift(gabor_samples::AbstractVector;
                          scales::Vector{Float32}=SCALES,
                          orientations::Vector{Float32}=ORIENTATIONS,
                          overlap_frac::Float32=OVERLAP_FRAC,
                          img_size::Int=IMG_SIZE)
    # Index Gabor samples by (grid x, grid y, scale index, orientation) for O(1) lookup.
    # Elements just need .x .y .θ .s .modulus .phase fields (as GaborSample has) —
    # deliberately untyped so this doesn't depend on which module's copy of
    # GaborSample the caller happens to be using.
    lookup = Dict{Tuple{Int,Int,Int,Float32}, eltype(gabor_samples)}()
    for samp in gabor_samples
        cx = round(Int, samp.x * (img_size - 1)) + 1
        cy = round(Int, samp.y * (img_size - 1)) + 1
        s_idx = round(Int, samp.s)
        lookup[(cx, cy, s_idx, samp.θ)] = samp
    end

    results = TJunctionSample[]
    for (s_idx, λ) in enumerate(scales)
        step = max(1, round(Int, (1 - overlap_frac) * λ))
        grid_coords = collect(step ÷ 2 + 1 : step : img_size)
        lo, hi = first(grid_coords), last(grid_coords)

        for cx in grid_coords, cy in grid_coords
            for (dx, dy, α) in NEIGHBOR_DIRECTIONS
                nx = cx + dx * step
                ny = cy + dy * step
                (lo <= nx <= hi && lo <= ny <= hi) || continue

                stem_θ = closest_orientation(mod(α, Float32(π)), orientations)
                cross_θ = closest_orientation(mod(α + Float32(π / 2), Float32(π)), orientations)

                stem = lookup[(cx, cy, s_idx, stem_θ)]
                cross = lookup[(nx, ny, s_idx, cross_θ)]

                phase_compat = (1 + cos(cross.phase - stem.phase)) / 2
                strength = min(stem.modulus, cross.modulus) * phase_compat

                push!(results, TJunctionSample(
                    (cx - 1) / (img_size - 1),
                    (cy - 1) / (img_size - 1),
                    α,
                    Float32(s_idx),
                    strength,
                    stem.phase
                ))
            end
        end
    end
    return results
end

end # module
