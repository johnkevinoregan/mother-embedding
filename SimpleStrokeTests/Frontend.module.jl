# ── PLAIN MODULE — .module.jl, not a Pluto notebook ─────────────────────────
# Included by other files. Opening it in Pluto rewrites it and leaves a
# "<name> backup 1.jl" beside it. Notebooks are the plain .jl files.

"""
    Frontend

The `RationalGaborFeatures` front end, wrapped as a single `featurize(imgs)` call, plus the
block masks the experiment needs to attribute a result to a part of the representation.

The bank is built once at module load for a fixed image size. Every experiment arm sees
exactly the same feature matrix, so differences between arms are differences between
readouts, not between preprocessing.

## The blocks

| block | columns at grid 3 | what it is |
|:--|--:|:--|
| `orient` | 135 | pooled quadrature energy, 3 scales × 8/12/16 orientations |
| `lowpass` | 9 | pooled DC — the only block that knows absolute level |
| `A1` | 27 | orientation-profile autocorrelation at 90° lag |
| `A2` | 27 | end-stopping along the locally dominant orientation |
| `rays` | 27 | ray-transform harmonics `c₀`, `|c₁|/c₀`, `|c₂|/c₀` |

The split between `orient` and `lowpass` is what makes the polarity control sharp. Quadrature
energy discards the sign of contrast by construction, so `orient` *should* be unable to
predict polarity — being unable to is the correct result. `lowpass` carries mean level and
should predict it perfectly. If `orient` predicts polarity, the invariance claim is wrong,
and that is worth finding out.
"""
module Frontend

using Statistics, LinearAlgebra, FFTW

include(joinpath(@__DIR__, "..", "RationalGaborFeatures", "GaborStack.module.jl"))
include(joinpath(@__DIR__, "..", "RationalGaborFeatures", "AndLayer.module.jl"))
include(joinpath(@__DIR__, "..", "RationalGaborFeatures", "RayHarmonics.module.jl"))
include(joinpath(@__DIR__, "..", "RationalGaborFeatures", "Pooling.module.jl"))
using .GaborStack, .AndLayer, .RayHarmonics, .Pooling

export build_frontend, featurize, block_cols, FrontendSpec

"The rational scale ladder and its per-scale angular resolution, as validated in RESULTS.md."
const LADDER = [2.0, 3.742, 7.0]
const BETAS  = [2.0, 1.6, 1.2]
const NORI   = [8, 12, 16]

struct FrontendSpec
    bank
    wts
    labels::Vector{String}
    n::Int
    grid::Int
end

"""
    build_frontend(N; grid=3)

Bank, pooling weights and column labels for `N×N` input. Padding is sized from both the
across- and along-contour extents, and rounded to a length FFTW factors well.
"""
function build_frontend(N::Int; grid::Int=3)
    HF, WF, _ = field_for((N, N), LADDER; n_orient=NORI, beta=BETAS)
    bank = make_bank((HF, WF), LADDER; imwidth=N, n_orient=NORI, beta=BETAS)
    wts  = grid_weights(N, N, grid)
    f, lab = _feat(zeros(Float32, N, N), bank, wts, grid)
    FrontendSpec(bank, wts, lab, length(f), grid)
end

function _feat(img, bank, wts, grid)
    # Padding mode is `:replicate` by default in `energy_stack` — see the note on `embed`
    # in GaborStack. This wrapper previously subtracted the image median to work around
    # zero-padding breaking polarity invariance; the fix now lives in the front end itself,
    # where every caller gets it rather than only this one.
    Es = energy_stack(img, bank)
    A, al = and_maps(Es, bank.meta; forms=(:A1, :A2))
    Rm, rl = ray_maps(Es, bank.meta)
    f1, l1 = assemble(Es, bank.meta, A, al,
                      PoolSpec(grid=grid, blocks=(:orient, :lowpass, :A1, :A2)); Wts=wts)
    # `ray_maps` returns c₀, |c₁| and |c₂| unnormalised; the ratios are formed **here**, from
    # the pooled energies, rather than per pixel. Dividing per pixel and then averaging gave
    # every low-energy location the same weight as a strong contour, and — because the old
    # divide-by-zero guard wrote zeros — made the pooled ratio scale with how much of the
    # window contained ink. Dividing after pooling is defined everywhere and needs no guard.
    #
    # The denominator carries a **relative** floor, a thousandth of the image's own mean c₀,
    # so an empty window tends to zero smoothly instead of through a branch. An absolute
    # epsilon here is what went wrong in A₂ (see RESULTS.md) and in ray_maps before this.
    PR = pool_maps(Rm, wts)
    nc = grid*grid
    fr = Float32[]; lr = String[]
    ρs = unique(l.rho0 for l in rl)
    for ρ in ρs
        k0 = findfirst(l -> l.rho0 == ρ && l.form === :R0, rl)
        k1 = findfirst(l -> l.rho0 == ρ && l.form === :R1, rl)
        k2 = findfirst(l -> l.rho0 == ρ && l.form === :R2, rl)
        floor_ρ = 1f-3 * max(mean(@view PR[:, k0]), 1f-12)
        for c in 1:nc
            den = PR[c, k0] + floor_ρ
            push!(fr, PR[c, k0]);        push!(lr, "rays.R0.ρ$(round(ρ,digits=2)).cell$(c)")
            push!(fr, PR[c, k1] / den);  push!(lr, "rays.R1.ρ$(round(ρ,digits=2)).cell$(c)")
            push!(fr, PR[c, k2] / den);  push!(lr, "rays.R2.ρ$(round(ρ,digits=2)).cell$(c)")
        end
    end
    vcat(f1, fr), vcat(l1, lr)
end

"""
    featurize(imgs, spec) -> n × d matrix

Threaded over images. Each image is independent, and the bank is read-only, so this is a
plain parallel map — the only shared state is FFTW's plan cache, which is why `FFTW`
threads are left at 1 and the parallelism is taken at the image level instead.
"""
function featurize(imgs::Vector{Matrix{Float32}}, spec::FrontendSpec)
    F = zeros(Float32, length(imgs), spec.n)
    Threads.@threads for i in eachindex(imgs)
        F[i, :] = _feat(imgs[i], spec.bank, spec.wts, spec.grid)[1]
    end
    F
end

"""
    block_cols(spec, names...) -> Vector{Int}

Column indices for one or more feature blocks. `block_cols(spec, "orient", "lowpass")` is
the conventional-statistics arm; `block_cols(spec, "A1", "A2")` is the conjunction layer
alone; `block_cols(spec, "lowpass")` is the block that should carry polarity.
"""
block_cols(spec::FrontendSpec, names::AbstractString...) =
    findall(l -> any(startswith(l, n * ".") || startswith(l, n) for n in names), spec.labels)

end # module
