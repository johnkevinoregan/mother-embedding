# ── PLAIN MODULE — .module.jl, not a Pluto notebook ─────────────────────────
# Included by other files. Opening it in Pluto rewrites it and leaves a
# "<name> backup 1.jl" beside it. Notebooks are the plain .jl files.

"""
    AndLayer

Pointwise conjunctions of oriented energy — the operation that has to happen **before**
spatial pooling.

The whole content of this layer is that multiplication and pooling do not commute:

    pool-then-multiply :  (Σₓ w(x)·e₁(x)) · (Σₓ w(x)·e₂(x))
    multiply-then-pool :   Σₓ w(x)·e₁(x)·e₂(x)

and the difference between them is the within-window spatial covariance `Cov_x(e₁,e₂)`.
That covariance *is* the co-location signal. A statistic computed from already-pooled
orientation energy — our `|E₂|`, `|E₄|`, or a Squeeze-and-Excitation gate — cannot
represent it at any capacity, because the pooling has already averaged it away. This is
an information argument, not an optimisation one: no downstream network can recover it.

Three forms, independently switchable:

  * `:A1` **same-location orientation conjunction.** The orientation profile's
    autocorrelation at 90° lag, computed per pixel. Zero for a single orientation,
    maximal where two orthogonal orientations coexist *at a point* — which is what
    separates a corner from two strokes that merely share a pooling window.
  * `:A2` **end-stopping.** Its flank offset can be anchored two ways — see `d_anchor` in
    `a2_maps`. The default `:envelope` reproduces everything in `RESULTS.md`; `:structure`
    is the scale-free choice for new datasets. Excitation at the centre, differenced against the two flanks
    *along* the stroke. Zero in a line's interior (both flanks present), zero for an
    isolated blob (neither present), maximal at a termination (one present, one absent).
  * `:A3` **cross-scale conjunction.** Orientation-pooled energy multiplied across
    adjacent scales. Off by default; included because it is nearly free once the stack
    exists, not because any argument requires it.

Note none of this needs phase, a bispectrum, or complex values. Energies are already
polarity-invariant, and products of energies remain so.
"""
module AndLayer

using Statistics

export and_maps, AND_FORMS

const AND_FORMS = (:A1, :A2, :A3)

# ---------------------------------------------------------------- helpers

"Bilinear sample of `M` at fractional `(y,x)`; zero outside."
@inline function bilin(M::AbstractMatrix{Float32}, y::Float64, x::Float64)
    H, W = size(M)
    (y < 1 || x < 1 || y > H || x > W) && return 0f0
    y0 = floor(Int, y); x0 = floor(Int, x)
    y1 = min(y0 + 1, H); x1 = min(x0 + 1, W)
    fy = Float32(y - y0); fx = Float32(x - x0)
    (1 - fy) * (1 - fx) * M[y0, x0] + fy * (1 - fx) * M[y1, x0] +
    (1 - fy) * fx * M[y0, x1] + fy * fx * M[y1, x1]
end

"""
Indices of the oriented channels of one scale, with their carrier angles, θ-sorted.

The return type is annotated deliberately. `bank.meta` is a `Vector{NamedTuple}` — an
abstract element type — so an unannotated comprehension yields `Vector{Tuple{Int,Any}}`,
`ch[k][1]` is `Any`, and every `@view E[:,:,that]` in the hot loops is type-unstable.
Measured: 52 ms against 6 ms for the identical arithmetic once this is concrete.
"""
function scale_channels(meta, rho)::Vector{Tuple{Int,Float64}}
    ch = Tuple{Int,Float64}[(Int(i), Float64(m.theta)) for (i, m) in enumerate(meta)
                            if m.kind === :oriented && m.rho0 ≈ rho]
    sort!(ch; by = last)
    ch
end

scales_of(meta)::Vector{Float64} = unique(Float64(m.rho0) for m in meta if m.kind === :oriented)

# ---------------------------------------------------------------- A1

"""
Same-location orientation conjunction, per scale.

`A₁(x) = C₀(x) · Σ_k p_k(x)·p_{k+n/2}(x)` where `p` is the orientation profile
normalised at that pixel and the `n/2` shift is exactly 90°.

Multiplying by `C₀` turns a shape measure into an energy, which keeps the map
well-behaved where there is no ink (the normalised profile is meaningless there).
"""
# NOTE on the `eps` guards in this file, checked when the ray transform's equivalent guard
# turned out to be a bug (see RayHarmonics and SimpleStrokeTests/RESULTS.md).
#
# An epsilon guard that writes a fill value is safe **only when the fill value is the true
# limit of the quantity as the energy goes to zero.** That is the case here and was not the
# case there:
#
#   A₁ = Σₖ EₖEₖ₊ₙ⁄₂ / C₀ .... numerator is O(E²), denominator O(E), so A₁ → 0. Writing 0 is
#                              the limit.
#   A₂ = E·|E₊−E₋|/(E₊+E₋+κE) . the leading E factor forces A₂ → 0. Writing 0 is the limit.
#   |c₁|/c₀ ................... a *bounded ratio* with no limit as c₀ → 0. Writing 0 asserted
#                              "perfectly symmetric" where there was no evidence, and pooling
#                              those zeros multiplied the result by ink coverage.
#
# Both maps here are energy-weighted, so a spatial mean of them is meaningful and no
# ratio-after-pooling treatment is needed. The distinction is energy-like vs scale-free, not
# absolute vs relative epsilon.
function a1_maps(E::Array{Float32,3}, meta; eps::Float32=1f-12)
    H, W, _ = size(E)
    out = Array{Float32,3}(undef, H, W, 0); labels = NamedTuple[]
    maps = Matrix{Float32}[]
    for ρ in scales_of(meta)
        ch = scale_channels(meta, ρ); n = length(ch)
        iseven(n) || error("A1 needs an even orientation count at ρ=$ρ (got $n)")
        half = n ÷ 2
        # Channel-OUTER, pixel-inner. The channel is the last array index, so a
        # pixel-outer loop strides H*W floats per channel access and misses cache on
        # every one: measured 190 ms against 8 ms for the same arithmetic this way.
        C0 = zeros(Float32, H, W); S = zeros(Float32, H, W)
        for k in 1:n
            Ea = @view E[:, :, ch[k][1]]
            Eb = @view E[:, :, ch[mod1(k + half, n)][1]]
            @inbounds @simd for p in eachindex(C0)
                C0[p] += Ea[p]; S[p] += Ea[p] * Eb[p]
            end
        end
        A = Matrix{Float32}(undef, H, W)
        @inbounds @simd for p in eachindex(A)
            A[p] = C0[p] > eps ? S[p] / C0[p] : 0f0
        end
        push!(maps, A); push!(labels, (form=:A1, rho0=ρ))
    end
    maps, labels
end

# ---------------------------------------------------------------- A2

"""
End-stopping, per scale and orientation.

For a channel whose *carrier* angle is `θ_c`, the stroke runs along `θ_s = θ_c + π/2`.
Sampling the same channel at `x ± d·u(θ_s)`:

`A₂ = E(x) · |E₊ − E₋| / (E₊ + E₋ + κ·E(x))`

The asymmetry factor is ~0 inside a line (both flanks present), ~0 on an isolated blob
(neither present), and ~⅔ at a termination (one present, one absent).

The `κ·E(x)` term is load-bearing, not a numerical guard. With an absolute `ε` instead,
a channel whose *own* flanks are both ≈0 — any channel not aligned with the local stroke
— yields `|E₊−E₋|/(E₊+E₋+ε) ≈ 1` from noise alone, so A₂ collapses to plain energy and
fires everywhere. Normalising against the centre response instead makes "both flanks
empty" score ~0, as it must.
"""
function a2_maps(E::Array{Float32,3}, meta; d=:auto, d_anchor::Symbol=:envelope,
                 structure_scale::Union{Nothing,Real}=nothing, kappa::Float32=0.5f0,
                 eps::Float32=1f-12, d_factor::Real=1.0)
    d_anchor in (:envelope, :structure) ||
        error("d_anchor must be :envelope (default, reproduces RESULTS.md) or :structure")
    d_anchor === :structure && structure_scale === nothing &&
        error("d_anchor = :structure needs structure_scale (e.g. the measured stroke width)")
    H, W, _ = size(E)
    maps = Matrix{Float32}[]; labels = NamedTuple[]
    for ρ in scales_of(meta)
        ch = scale_channels(meta, ρ)
        # `:auto` sets the flank offset from the channel's own along-contour envelope,
        # so the probe sits just outside the excitatory region at every scale. A fixed
        # pixel offset is degenerate at coarse scales, where both flanks fall inside one
        # envelope and the asymmetry vanishes by construction.
        #
        # d_factor = 1.0 puts the flank one σ_along out, ~10 px at the finest scale,
        # which is about one stroke width. Chosen by sweeping rather than by argument:
        # end/interior peaks there (10.4×) and falls off either side, while end/blob
        # rises monotonically with the offset — 1.0 is where all three contrasts clear.
        # σ_φ and the image width are read from the bank's own metadata rather than
        # recomputed here — holding the same parameter in two places is exactly how the
        # offsets silently kept their old values when the bank's angular tuning changed.
        # ── WHERE THE FLANK OFFSET IS ANCHORED ──────────────────────────────────────
        # Two options, and the default is deliberately the weaker one.
        #
        # :envelope (DEFAULT) — d = d_factor · σ_along, i.e. one along-contour envelope of
        #   the filter itself. Every result in RESULTS.md was produced this way and the
        #   default is kept so they reproduce exactly. It is also the biologically natural
        #   form: an end-stopped cell's inhibitory zone is fixed relative to its own
        #   receptive field, not to the stimulus.
        #
        #   BUT IT IS NOT SCALE-FREE, and this was measured rather than argued. Sweeping in
        #   dimensionless units (Validate_ScaleFree.jl) — stimulus fixed, filter scale
        #   varied — the best d_factor DRIFTS with the stimulus scale: [3.0, 0.5, 1.5, 2.0]
        #   at w/λ = 0.30 / 0.50 / 0.80 / 1.20. Converting the optima to physical offsets
        #   gives d/stroke_width = 0.60 / 1.15 / 1.01 — i.e. the optimum tracks the STROKE
        #   WIDTH, not the envelope. d_factor = 1.0 looked right on EMNIST only because
        #   σ_along = 9.7 px against a 12.67 px stroke happens to give 0.76 stroke widths at
        #   that one operating point. A constant fitted at one scale fails silently, not
        #   loudly, at another.
        #
        # :structure — d = d_factor · structure_scale, one measured structure scale, the
        #   same at every filter scale. This is the general choice, and it matches how the
        #   rest of the pipeline already works: `scale_ladder` derives the bank from the
        #   data's own spectrum, so the operator should likewise derive its geometry from
        #   the data's own structure scale (stroke width, measured in Phase 0 at 12.67 px
        #   for EMNIST). Use this when pointing the front end at a new dataset.
        dρ = if d !== :auto
            Float64(d)
        elseif d_anchor === :structure
            d_factor * Float64(structure_scale)
        else
            m = first(m for m in meta if m.kind === :oriented && m.rho0 ≈ ρ)
            d_factor * m.imwidth / (2π * ρ * m.sigma_phi)
        end
        # Winner-take-all over orientation: the end-stop inherits the LOCALLY DOMINANT
        # orientation, as a real end-stopped cell does. Taking a max over all orientations
        # instead lets weakly-responding off-orientation channels contribute spurious
        # asymmetry (their own flanks are near-empty, so the ratio is ill-conditioned),
        # which flattens the contrast: measured end/interior 2.5× taking the max against
        # 10.4× taking the dominant orientation.
        #
        # Two passes, both channel-outer for cache locality (see the note in A1).
        best = zeros(Float32, H, W); bidx = zeros(Int32, H, W)
        for (ci, (i, _)) in enumerate(ch)
            Ei = @view E[:, :, i]
            @inbounds @simd for p in eachindex(best)
                v = Ei[p]
                newbest = v > best[p]
                best[p] = ifelse(newbest, v, best[p])
                bidx[p] = ifelse(newbest, Int32(ci), bidx[p])
            end
        end
        A = zeros(Float32, H, W)
        for (ci, (i, θc)) in enumerate(ch)
            θs = θc + π/2                          # along the stroke
            dy = dρ * sin(θs); dx = dρ * cos(θs)
            Ei = @view E[:, :, i]
            @inbounds for x in 1:W, y in 1:H
                bidx[y, x] == ci || continue
                e = best[y, x]; e <= eps && continue
                ep = bilin(Ei, y + dy, x + dx)
                em = bilin(Ei, y - dy, x - dx)
                A[y, x] = e * abs(ep - em) / (ep + em + kappa * e)
            end
        end
        push!(maps, A); push!(labels, (form=:A2, rho0=ρ, d=dρ))
    end
    maps, labels
end

# ---------------------------------------------------------------- A3

"Cross-scale conjunction of orientation-pooled energy, adjacent scales only."
function a3_maps(E::Array{Float32,3}, meta)
    H, W, _ = size(E)
    ρs = sort(scales_of(meta))
    C0 = [begin
              M = zeros(Float32, H, W)
              for (i, _) in scale_channels(meta, ρ); M .+= @view E[:, :, i]; end
              M
          end for ρ in ρs]
    maps = Matrix{Float32}[]; labels = NamedTuple[]
    for k in 1:length(ρs)-1
        push!(maps, C0[k] .* C0[k+1])
        push!(labels, (form=:A3, rho0=ρs[k], rho1=ρs[k+1]))
    end
    maps, labels
end

# ---------------------------------------------------------------- entry point

"""
    and_maps(E, meta; forms=(:A1,), d=:auto, d_factor=1.0)

Pointwise conjunction maps from a dense energy stack. Returns `(maps, labels)` where
`maps` is `H × W × k` and `labels` describes each channel. `forms=()` returns nothing,
which is how the layer is ablated.
"""
function and_maps(E::Array{Float32,3}, meta; forms=(:A1,), d=:auto, d_factor::Real=1.0,
                  d_anchor::Symbol=:envelope, structure_scale::Union{Nothing,Real}=nothing)
    all(f in AND_FORMS for f in forms) ||
        error("unknown form in $forms; valid: $AND_FORMS")
    H, W, _ = size(E)
    allmaps = Matrix{Float32}[]; alllab = NamedTuple[]
    if :A1 in forms
        m, l = a1_maps(E, meta); append!(allmaps, m); append!(alllab, l)
    end
    if :A2 in forms
        m, l = a2_maps(E, meta; d=d, d_factor=d_factor, d_anchor=d_anchor,
                       structure_scale=structure_scale)
        append!(allmaps, m); append!(alllab, l)
    end
    if :A3 in forms
        m, l = a3_maps(E, meta); append!(allmaps, m); append!(alllab, l)
    end
    isempty(allmaps) && return Array{Float32,3}(undef, H, W, 0), alllab
    out = Array{Float32,3}(undef, H, W, length(allmaps))
    for (k, M) in enumerate(allmaps); out[:, :, k] = M; end
    out, alllab
end

end # module
