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
include(joinpath(@__DIR__, "..", "RationalGaborFeatures", "GaborStackGPU.module.jl"))
using .GaborStack, .AndLayer, .RayHarmonics, .Pooling, .GaborStackGPU
using CUDA

export build_frontend, featurize, featurize_gpu, block_cols, FrontendSpec

"The rational scale ladder and its per-scale angular resolution, as validated in RESULTS.md."
const LADDER = [2.0, 3.742, 7.0]
const BETAS  = [2.0, 1.6, 1.2]
const NORI   = [8, 12, 16]

"""
i1D floor correction for A₁ — see `AndLayer.a1_maps`.

`:analytic` (the default) subtracts the closed-form response A₁ gives on a straight line, so the
operator is exactly zero on i1D input. `:none` is the original.

**⚠ Every published table in this project was computed with `:none` and has not been re-run.**
Set `FRONTEND_A1_FLOOR=none` to reproduce them, or pass `a1_floor=:none` to `build_frontend`.
"""
const A1_FLOOR = Symbol(get(ENV, "FRONTEND_A1_FLOOR", "analytic"))

struct FrontendSpec
    bank
    wts
    labels::Vector{String}
    n::Int
    grid::Int
    scale_mode::Symbol
    normalize::Symbol
    kappa::Float32
    a1_floor::Symbol
    d_factors
    harmonics::Tuple
end

"""
    build_frontend(N; grid=3)

Bank, pooling weights and column labels for `N×N` input. Padding is sized from both the
across- and along-contour extents, and rounded to a length FFTW factors well.
"""
function build_frontend(N::Int; grid::Int=3, scale_mode::Symbol=:per_scale,
                        normalize::Symbol=:none, kappa::Float32=0.10f0,
                        a1_floor::Symbol=A1_FLOOR,
                        ladder=LADDER, nori=NORI, betas=BETAS, dts::Real=0.75,
                        d_factors=(1.0,), harmonics::Tuple=(2, 4))
    # `ladder`/`nori`/`betas`/`dts`/`d_factors` exist so the three capacity axes — scales,
    # orientations, ray offsets — can be swept. Defaults reproduce every published table.
    #
    # `dts` (dtheta_on_sigma) matters when raising `nori`: σφ = (π/n)/dts, so leaving dts at 0.75
    # while doubling n HALVES σφ and DOUBLES σ_along (17 → 34 px at ρ=2), which is the "filters
    # longer than any stroke" bug. Raise dts in step with n to add orientation channels at
    # constant σφ — finer angular sampling, no spatial cost. Those are two different experiments.
    #
    # `d_factors` gives the ray transform more than one offset per scale. With three factors and
    # three scales it is the crossed d × λ design: every offset against every wavelength, rather
    # than only the matched diagonal.
    HF, WF, _ = field_for((N, N), ladder; n_orient=nori, beta=betas, dtheta_on_sigma=dts)
    bank = make_bank((HF, WF), ladder; imwidth=N, n_orient=nori, beta=betas, dtheta_on_sigma=dts)
    wts  = grid_weights(N, N, grid)
    f, lab = _feat(zeros(Float32, N, N), bank, wts, grid, scale_mode, normalize, kappa, a1_floor, d_factors, harmonics)
    FrontendSpec(bank, wts, lab, length(f), grid, scale_mode, normalize, kappa, a1_floor, d_factors, harmonics)
end

function _feat(img, bank, wts, grid, scale_mode=:per_scale, normalize=:none, kappa=0.10f0,
               a1_floor=A1_FLOOR, d_factors=(1.0,), harmonics::Tuple=(2, 4))
    # Padding mode is `:replicate` by default in `energy_stack` — see the note on `embed`
    # in GaborStack. This wrapper previously subtracted the image median to work around
    # zero-padding breaking polarity invariance; the fix now lives in the front end itself,
    # where every caller gets it rather than only this one.
    Es = energy_stack(img, bank)
    A, al = and_maps(Es, bank.meta; forms=(:A1, :A2), a1_floor=a1_floor)
    # One ray_maps call per offset factor. With length(d_factors) > 1 this is the crossed
    # d × λ design: each offset is applied at every wavelength, so the labels carry the factor
    # to keep the columns distinguishable.
    Rm, rl = ray_maps(Es, bank.meta; d_factor=d_factors[1], scale_mode=scale_mode,
                      normalize=normalize, kappa=kappa)
    extraR = []
    for df in d_factors[2:end]
        m, l = ray_maps(Es, bank.meta; d_factor=df, scale_mode=scale_mode,
                        normalize=normalize, kappa=kappa)
        push!(extraR, (m, l, df))
    end
    f1, l1 = assemble(Es, bank.meta, A, al,
                      PoolSpec(grid=grid, blocks=(:orient, :lowpass, :A1, :A2), harmonics=harmonics); Wts=wts)
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
    # `:max` emits one scale-invariant profile plus `Rs`, the winning scale — a local
    # stroke-width estimate obtained with no preliminary pass over the image.
    if scale_mode === :max
        k0 = findfirst(l -> l.form === :R0, rl); k1 = findfirst(l -> l.form === :R1, rl)
        k2 = findfirst(l -> l.form === :R2, rl); ks = findfirst(l -> l.form === :Rs, rl)
        fl = 1f-3 * max(mean(@view PR[:, k0]), 1f-12)
        for c in 1:nc
            den = PR[c, k0] + fl
            push!(fr, PR[c,k0]);       push!(lr, "rays.R0.max.cell$(c)")
            push!(fr, PR[c,k1]/den);   push!(lr, "rays.R1.max.cell$(c)")
            push!(fr, PR[c,k2]/den);   push!(lr, "rays.R2.max.cell$(c)")
            push!(fr, PR[c,ks]);       push!(lr, "rays.Rs.max.cell$(c)")
        end
        return vcat(f1, fr), vcat(l1, lr)
    end
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
    # extra offsets: identical assembly, labels tagged with the factor so the crossed d × λ
    # columns stay distinguishable from the matched diagonal.
    for (Rm2, rl2, df) in extraR
        PR2 = pool_maps(Rm2, wts)
        for ρ in unique(l.rho0 for l in rl2)
            k0 = findfirst(l -> l.rho0 == ρ && l.form === :R0, rl2)
            k1 = findfirst(l -> l.rho0 == ρ && l.form === :R1, rl2)
            k2 = findfirst(l -> l.rho0 == ρ && l.form === :R2, rl2)
            fl2 = 1f-3 * max(mean(@view PR2[:, k0]), 1f-12)
            tag = "d$(round(df, digits=2))"
            for c in 1:nc
                den = PR2[c, k0] + fl2
                push!(fr, PR2[c, k0]);       push!(lr, "rays.R0.ρ$(round(ρ,digits=2)).$tag.cell$(c)")
                push!(fr, PR2[c, k1] / den); push!(lr, "rays.R1.ρ$(round(ρ,digits=2)).$tag.cell$(c)")
                push!(fr, PR2[c, k2] / den); push!(lr, "rays.R2.ρ$(round(ρ,digits=2)).$tag.cell$(c)")
            end
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
        F[i, :] = _feat(imgs[i], spec.bank, spec.wts, spec.grid, spec.scale_mode,
                        spec.normalize, spec.kappa, spec.a1_floor, spec.d_factors,
                        spec.harmonics)[1]
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


"""
    featurize_gpu(imgs, spec; batch=32, blocks=(:orient,:lowpass,:A1,:A2))

CUDA-accelerated `featurize` for the blocks `GaborStackGPU` covers.

The FFTs and the `A₁`/`A₂` operators run batched on the device; the maps come back to the host
and **`Pooling.assemble` runs unchanged**, so the feature vector is produced by the same code as
`featurize` and cannot drift from it. `RationalGaborFeatures/Validate_GPU.jl` gates the maps
against the CPU reference.

**The ray block is not available here** — it is not ported. Requesting it is an error rather than
a silent fallback, because a silent CPU fallback would move the whole energy stack back across
PCIe and quietly undo the speedup. Use `featurize` for ray configurations.

**Not bit-identical to `featurize`.** CUFFT's rounding is not FFTW's; measured agreement is
~1.4e-5 relative on oriented energy and ~2.5e-3 on `A₂`, whose conditioned denominator amplifies
it. Say which backend produced a table.
"""
function featurize_gpu(imgs::Vector{Matrix{Float32}}, spec::FrontendSpec;
                       batch::Int=32, blocks=(:orient, :lowpass, :A1, :A2))
    :rays in blocks && error("featurize_gpu: the ray block is not ported to GPU — use featurize")
    CUDA.functional() || error("featurize_gpu: CUDA is not functional")
    N = size(first(imgs), 1)
    gb = upload_bank(spec.bank)
    ps = PoolSpec(grid=spec.grid, blocks=blocks)
    F = nothing
    for lo in 1:batch:length(imgs)
        hi = min(lo + batch - 1, length(imgs))
        chunk = reshape(reduce(hcat, [vec(im) for im in imgs[lo:hi]]), N, N, hi - lo + 1)
        E = energy_batch(CuArray(chunk), gb; crop_to=N)
        A, alab = and_batch(E, spec.bank.meta; a1_floor=spec.a1_floor,
                            a1_floor_fn=AndLayer.a1_i1d_floor)
        Eh = Array(E); Ah = Array(A)
        E = nothing; A = nothing; CUDA.reclaim()
        for b in 1:(hi - lo + 1)
            f, _ = assemble(Eh[:, :, :, b], spec.bank.meta, Ah[:, :, :, b], alab, ps; Wts=spec.wts)
            F === nothing && (F = zeros(Float32, length(imgs), length(f)))
            F[lo + b - 1, :] = f
        end
    end
    F
end

end # module
