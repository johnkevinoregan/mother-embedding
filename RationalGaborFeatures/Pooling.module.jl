# ── PLAIN MODULE — .module.jl, not a Pluto notebook ─────────────────────────
# Included by other files. Opening it in Pluto rewrites it and leaves a
# "<name> backup 1.jl" beside it. Notebooks are the plain .jl files.

"""
    Pooling

Spatial pooling and feature assembly — the step that turns dense maps into a fixed vector.

**Pooling comes last, deliberately.** An i2D conjunction is spatially sharp and therefore
fragile: a junction moves several pixels between handwritten instances, so a pixel-exact
detector would give a different vector every time. Pooling is what buys the positional
tolerance. The whole design is `select → multiply → pool`, which is simple cell → complex
cell, and conv → ReLU → pool. Getting that order right is the point; pooling itself is not
the enemy.

Blocks are independently switchable so Phase 5 can ablate them:

| block | what it is | per cell |
|:--|:--|--:|
| `:orient` | orientation harmonics of the **pooled** energy | 5 per scale |
| `:lowpass` | the low-pass channel | 1 |
| `:A1` `:A2` `:A3` | conjunctions computed pointwise, **then** pooled | 1 per scale |

`:orient` is deliberately the *deficient* baseline — it pools first and combines after, so
it cannot represent co-location. The A-blocks are the same information computed in the
other order. That contrast is the experiment, so it is structural here rather than a
comment.

Grid geometry is specified **relative to the image**, never in absolute pixels, so the same
spec transfers to another dataset at another size.
"""
module Pooling

using Statistics, LinearAlgebra, Random

export PoolSpec, grid_weights, pool_maps, assemble, shuffle_block!, nyquist_grid

# ---------------------------------------------------------------- geometry

"""
    grid_weights(H, W, n; overlap=1.0)

`H·W × n²` matrix of Gaussian pooling weights: column `c` is one cell's window, centred on
an `n × n` lattice with `σ = overlap × spacing/2`. Columns are L1-normalised so a pooled
value is a weighted *mean*, which keeps cells comparable when one sits partly off-image.

Overlapping (rather than tiling) windows matter: a stroke falling on a hard cell boundary
would otherwise be assigned arbitrarily to one side.
"""
function grid_weights(H::Int, W::Int, n::Int; overlap::Real=1.0)
    Wts = zeros(Float32, H * W, n * n)
    sy, sx = H / n, W / n
    σy, σx = overlap * sy / 2, overlap * sx / 2
    c = 0
    for i in 1:n, j in 1:n
        c += 1
        cy = (i - 0.5) * sy; cx = (j - 0.5) * sx
        s = 0.0
        @inbounds for x in 1:W, y in 1:H
            v = exp(-((y - cy)^2 / (2σy^2) + (x - cx)^2 / (2σx^2)))
            Wts[(x - 1) * H + y, c] = v; s += v
        end
        s > 0 && (@view(Wts[:, c]) ./= Float32(s))
    end
    Wts
end

"""
    nyquist_grid(imsize, sigma_along)

The **finest** grid the demodulation argument licenses: after `|·|²` the envelope is
baseband with bandwidth set by σ, so samples closer than about σ are redundant. Returns
`n` such that the spacing is ≈ σ.

This is an upper bound on useful density, not a requirement — a coarser grid is a
compactness choice, and at σ = 9.7 px on a 112-px image this returns 11, i.e. 121 cells,
which is far more than a 3×3 layout offers.
"""
nyquist_grid(imsize::Int, sigma_along::Real) = max(1, floor(Int, imsize / sigma_along))

# ---------------------------------------------------------------- pooling

"""
    pool_maps(maps, Wts)

`n² × nmaps` pooled values. Done as one `gemm`, so pooling 40-odd maps costs less than the
convolutions that produced them.
"""
function pool_maps(maps::Array{Float32,3}, Wts::Matrix{Float32})
    H, W, K = size(maps)
    size(Wts, 1) == H * W || error("weights are $(size(Wts,1)) rows, maps are $(H*W) pixels")
    transpose(Wts) * reshape(maps, H * W, K)
end

# ---------------------------------------------------------------- the spec

"""
    PoolSpec(; grid=3, blocks=(:orient, :lowpass), overlap=1.0)

`blocks` selects which feature families are emitted. `()` is legal and yields an empty
vector, which is how a block is ablated.
"""
Base.@kwdef struct PoolSpec
    grid::Int = 3
    blocks::Tuple = (:orient, :lowpass)
    overlap::Float64 = 1.0
end

const BLOCKS = (:orient, :lowpass, :A1, :A2, :A3)

"""
    assemble(E, meta, A, alab, spec; Wts=nothing)

Build the feature vector from a dense energy stack `E`, its bank `meta`, and the AND maps
`A` with labels `alab`. Returns `(features, labels)`; `labels` names every column so an
ablation can be read back afterwards.

`:orient` emits, per cell and scale, `√C₀` and the orientation harmonics `Re C₂, Im C₂,
|C₂|, |C₄|` **computed from the pooled energy** — pool-then-combine, the arrangement that
provably cannot represent co-location. The A-blocks are the same information computed the
other way round.
"""
function assemble(E::Array{Float32,3}, meta, A::Array{Float32,3}, alab, spec::PoolSpec;
                  Wts::Union{Nothing,Matrix{Float32}}=nothing)
    all(b in BLOCKS for b in spec.blocks) ||
        error("unknown block in $(spec.blocks); valid: $BLOCKS")
    H, W, _ = size(E)
    Wt = Wts === nothing ? grid_weights(H, W, spec.grid; overlap=spec.overlap) : Wts
    nc = spec.grid^2
    feats = Float32[]; labels = String[]

    ρs = unique(m.rho0 for m in meta if m.kind === :oriented)
    if :orient in spec.blocks
        PE = pool_maps(E, Wt)                       # nc × nchannels
        for ρ in ρs
            ch = [(i, m.theta) for (i, m) in enumerate(meta)
                  if m.kind === :oriented && m.rho0 ≈ ρ]
            for c in 1:nc
                c0 = sum(PE[c, i] for (i, _) in ch)
                s2 = sum(PE[c, i] * cis(2θ) for (i, θ) in ch)
                s4 = sum(PE[c, i] * cis(4θ) for (i, θ) in ch)
                n2 = c0 > 0 ? s2 / c0 : zero(s2)
                n4 = c0 > 0 ? s4 / c0 : zero(s4)
                append!(feats, Float32[sqrt(c0), real(n2), imag(n2), abs(n2), abs(n4)])
                for nm in ("c0", "ReC2", "ImC2", "absC2", "absC4")
                    push!(labels, "orient.ρ$(round(ρ,digits=2)).cell$(c).$(nm)")
                end
            end
        end
    end
    if :lowpass in spec.blocks
        lp = [i for (i, m) in enumerate(meta) if m.kind === :lowpass]
        if !isempty(lp)
            P = pool_maps(E[:, :, lp], Wt)
            for c in 1:nc
                push!(feats, sqrt(max(P[c, 1], 0f0))); push!(labels, "lowpass.cell$(c)")
            end
        end
    end
    for blk in (:A1, :A2, :A3)
        blk in spec.blocks || continue
        sel = [k for (k, l) in enumerate(alab) if l.form === blk]
        isempty(sel) && continue
        P = pool_maps(A[:, :, sel], Wt)
        for (j, k) in enumerate(sel), c in 1:nc
            push!(feats, sqrt(max(P[c, j], 0f0)))
            push!(labels, "$(blk).ρ$(round(alab[k].rho0,digits=2)).cell$(c)")
        end
    end
    feats, labels
end

# ---------------------------------------------------------------- the control

"""
    shuffle_block!(F, cols; seed=1)

Permute the rows of columns `cols` **across samples**, destroying each sample's
correspondence to its own values while leaving the marginal distribution and the feature
count untouched.

This is the control that separates *"this block carries information"* from *"more columns
help"*. It is not optional bookkeeping: §7.8 established that a fixed projection into a few
hundred dimensions plus a trained head is a strong baseline **whatever the projection is** —
scrambled features still reached 75 % — so an unshuffled comparison cannot distinguish the
two, and a gain of a point or two is exactly the size that dimensionality alone can buy.
"""
function shuffle_block!(F::AbstractMatrix, cols; seed::Int=1)
    rng = MersenneTwister(seed); n = size(F, 1)
    p = randperm(rng, n)
    @views F[:, cols] = F[p, cols]
    F
end

end # module
