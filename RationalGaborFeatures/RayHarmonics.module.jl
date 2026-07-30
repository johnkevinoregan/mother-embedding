# ── PLAIN MODULE — .module.jl, not a Pluto notebook ─────────────────────────
# Included by other files. Opening it in Pluto rewrites it and leaves a
# "<name> backup 1.jl" beside it. Notebooks are the plain .jl files.

"""
    RayHarmonics

The **ray transform** and its circular harmonics, computed on a `GaborStack` energy field.

This exists because A₁ provably cannot do what it was built for. A₁ is the orientation
profile's autocorrelation at 90° lag, and the orientation profile is **π-periodic** — so a
T-junction (3 rays) and an X-crossing (4 rays) have *identical* orientation content,
{0°, 90°}, and A₁ cannot separate them. Ray count is a **2π** property.

The ray transform gets 2π structure from a **spatial offset** rather than from any
nonlinearity:

    R(p, φ) = E( p + d·u(φ),  θ_stroke = φ mod π ),      u(φ) = (cos φ, sin φ)

*"Is there a contour at distance `d` in direction `φ`, oriented along `φ`?"* The offset is
what converts a mod-π quantity to a mod-2π one — east and west read different pixels. `R`
is 2π-periodic with **one lobe per branch**: endpoint 1, straight 2 opposite, L-corner 2
adjacent, T 3, X 4. Its Fourier coefficients are the type signature:

Signature table, in terms of the ratios formed after pooling:

| configuration | c₀ | \\|c₁\\|/c₀ | \\|c₂\\|/c₀ |
|:--|--:|--:|--:|
| endpoint (1 ray) | 1 | 1.000 | 1.000 |
| straight (2 opposite) | 2 | 0.000 | 1.000 |
| L-corner (2 at 90°) | 2 | 0.707 | 0.000 |
| **T-junction (3)** | **3** | **0.333** | **0.333** |
| X-crossing (4) | 4 | 0.000 | 0.000 |

`c₀` ≈ ray count; `|c₁|/c₀` ≈ asymmetry (1 = endpoint, 0 = centrally symmetric).

Note it is **linear in the energy field** — a weighted sum of rigidly shifted orientation
channels — where A₁ is bilinear. The simpler operator captures what the more complicated
one cannot, because the work is done by the geometry of the sampling rather than by a
product.

Our channels are indexed by **carrier** angle `θ_c` (the wavevector), and a stroke runs
perpendicular to its carrier, so the channel wanted for ray direction `φ` is the one with
`θ_c ≈ (φ + π/2) mod π`.
"""
module RayHarmonics

using Statistics

export ray_maps

@inline function bilin(M::AbstractMatrix{Float32}, y::Float64, x::Float64)
    H, W = size(M)
    (y < 1 || x < 1 || y > H || x > W) && return 0f0
    y0 = floor(Int, y); x0 = floor(Int, x)
    y1 = min(y0 + 1, H); x1 = min(x0 + 1, W)
    fy = Float32(y - y0); fx = Float32(x - x0)
    (1-fy)*(1-fx)*M[y0,x0] + fy*(1-fx)*M[y1,x0] + (1-fy)*fx*M[y0,x1] + fy*fx*M[y1,x1]
end

scales_of(meta)::Vector{Float64} =
    unique(Float64(m.rho0) for m in meta if m.kind === :oriented)

function chan_of(meta, rho)::Vector{Tuple{Int,Float64}}
    ch = Tuple{Int,Float64}[(Int(i), Float64(m.theta)) for (i,m) in enumerate(meta)
                            if m.kind === :oriented && m.rho0 ≈ rho]
    sort!(ch; by = last); ch
end

"""
    ray_maps(E, meta; nphi=nothing, d=:auto, d_factor=1.0)

Returns `(maps, labels)` with three channels per scale: **`c₀`, `|c₁|` and `|c₂|`, all
unnormalised**. The scale-free ratios `|c₁|/c₀` and `|c₂|/c₀` are to be formed by the caller
**after pooling**, not here.

## Why the ratio is not formed per pixel

It used to be, guarded by `c₀ > 1e-12` with `r = 0` written where that failed. Two things
were wrong with it, and the second matters on any image.

**The guard smuggled in a line-drawing assumption.** Writing `0` where there is no energy is
not neutral — `0` means *perfectly symmetric*, a positive claim about a location with no
evidence. On a stroke drawing the branch fires at most pixels; on a photograph `c₀ > 0`
everywhere and it never fires. An operator that behaves differently in kind between those two
cases is not a general front end.

**Pooling a ratio is wrong regardless of background.** A per-pixel ratio where `c₀` is small
is numerically fine and statistically meaningless, and a spatial mean gives it exactly the
same weight as a ratio measured on a strong contour. Worse, since the guard wrote zeros, the
pooled value came out as *the true ratio multiplied by the fraction of the window containing
contour* — confounding a shape descriptor with ink coverage, which depends on stroke
thickness, length and blur.

Pooling the numerator and denominator separately and dividing afterwards is defined
everywhere, needs no guard and no fill value, weights each location by its own energy
automatically, and behaves identically on a line drawing and a photograph.

This is the same error `A₂` had: `RESULTS.md` records that its original **absolute**-ε
conditioning collapsed it to plain energy, and the fix was a **relative** term `κ·E(x)`. The
ray transform kept the absolute form until it was found again here.

`nphi` ray directions over [0, 2π); defaults to twice the orientation count of the scale,
so each orientation channel is used once in each of its two directions.

`d_anchor` selects how the probe radius is set, exactly as in `AndLayer.a2_maps`:
`:envelope` (default) from the channel's own along-contour envelope, which reproduces every
published number; `:structure` from a caller-supplied scale measured off the image, which is
the right anchoring when the image content does not scale with the frame.
"""
function ray_maps(E::Array{Float32,3}, meta; nphi=nothing, d=:auto, d_factor::Real=1.0,
                  d_anchor::Symbol=:envelope, structure_scale::Union{Nothing,Real}=nothing,
                  scale_mode::Symbol=:per_scale, normalize::Symbol=:none,
                  kappa::Float32=0.25f0)
    H, W, _ = size(E)
    d_anchor in (:envelope, :structure) ||
        error("d_anchor must be :envelope (default, reproduces published numbers) or :structure")
    d_anchor === :structure && structure_scale === nothing &&
        error("d_anchor = :structure needs structure_scale")
    scale_mode in (:per_scale, :max, :cross) ||
        error("scale_mode must be :per_scale, :max or :cross")
    normalize in (:none, :divisive) ||
        error("normalize must be :none or :divisive")

    """
    Divisive normalisation of a ray profile, in place.

        R'(φ) = R(φ) / (R(φ) + κ·maxφ R)

    Needed because the harmonics compare lobes on **raw magnitude**, so a junction whose arms
    differ in thickness reads as fewer rays: a 4 px bar crossing a 15 px bar gave
    |c₂|/c₀ = 0.638 (two rays) where an equal-thickness X gives 0.047. The weak arm's
    response was 22 % of the strong one's and the profile was dominated by one axis.

    A plain rescaling cannot fix this — dividing every lobe by the same number leaves their
    ratio unchanged. The nonlinearity has to **saturate**, so that a present-but-weak ray
    counts nearly as much as a strong one. That is divisive normalisation, which is also what
    V1 is generally modelled as doing.
    """
    function normprofile!(R::AbstractVector{Float32}, κ)
        M = maximum(R); M <= 0 && return R
        @inbounds for j in eachindex(R); R[j] = R[j] / (R[j] + κ*M); end
        R
    end
    maps = Matrix{Float32}[]; labels = NamedTuple[]

    # ── :max — one scale-invariant ray profile instead of one per scale ─────
    #
    # For each direction φ, take the largest response over scales, each probed at its own
    # offset dₛ. Because dₛ ∝ λₛ, **the winning scale brings its own offset with it**: a
    # thick stroke wins at a coarse scale with a large d, a thin one at a fine scale with a
    # small d. So the offset tracks the local structure without any preliminary measurement
    # of the image — which is the objection to anchoring d to a measured structure scale,
    # since nothing in biology performs a global pass before setting a receptive field.
    #
    # It also explains the drift recorded in RESULTS.md §3, where the best d_factor moved 4×
    # across w/λ. That sweep held the stimulus fixed and varied the filter scale, forcing the
    # operator *off* the diagonal. A max over scales keeps it on the diagonal by
    # construction, so one constant suffices.
    #
    # The max is meaningful only because the bank is RMS-normalised and responses are
    # comparable across scales; without that it would simply select the loudest filter.
    #
    # `argmax` is emitted alongside as `Rs`: which scale won is a **local stroke-width
    # estimate**, the structure scale recovered as an output rather than assumed as an input.
    if scale_mode === :max
        ρs = scales_of(meta)
        # Angular resolution set by the FINEST scale, not the first one. The scales carry 8,
        # 12 and 16 orientations and share one profile, so sampling at twice the coarsest
        # count would probe the finest scale in half the directions it can resolve — and the
        # fine scale is where junction detail lives.
        K = nphi === nothing ? 2 * maximum(length(chan_of(meta, ρ)) for ρ in ρs) : Int(nphi)
        Rmax = zeros(Float32, H, W, K); Sbest = zeros(Float32, H, W)
        best = fill(-1f0, H, W)
        for ρ in ρs
            ch = chan_of(meta, ρ); m1 = first(m for m in meta if m.kind === :oriented && m.rho0 ≈ ρ)
            dρ = d !== :auto ? Float64(d) :
                 d_anchor === :structure ? d_factor * Float64(structure_scale) :
                 d_factor * m1.imwidth / (2π * ρ * m1.sigma_phi)
            for j in 1:K
                φ = 2π*(j-1)/K; want = mod(φ + π/2, π)
                bi = 1; bd = Inf
                for (k,(_,θ)) in enumerate(ch)
                    δ = abs(atan(sin(θ-want), cos(θ-want))); δ < bd && (bd = δ; bi = k)
                end
                Ej = @view E[:, :, ch[bi][1]]
                dy = dρ*sin(φ); dx = dρ*cos(φ)
                @inbounds for x in 1:W, y in 1:H
                    v = bilin(Ej, y + dy, x + dx)
                    v > Rmax[y,x,j] && (Rmax[y,x,j] = v)
                end
            end
            # track which scale gives the strongest response anywhere in the profile
            @inbounds for x in 1:W, y in 1:H
                s = 0f0; for j in 1:K; s += Rmax[y,x,j]; end
                if s > best[y,x]; best[y,x] = s; Sbest[y,x] = Float32(ρ); end
            end
        end
        C0 = zeros(Float32,H,W); C1r = zeros(Float32,H,W); C1i = zeros(Float32,H,W)
        C2r = zeros(Float32,H,W); C2i = zeros(Float32,H,W)
        prof = Vector{Float32}(undef, K)
        @inbounds for x in 1:W, y in 1:H
            for j in 1:K; prof[j] = Rmax[y,x,j]; end
            normalize === :divisive && normprofile!(prof, kappa)
            for j in 1:K
                φ = 2π*(j-1)/K; v = prof[j]
                C0[y,x]  += v
                C1r[y,x] += v*Float32(cos(φ));  C1i[y,x] -= v*Float32(sin(φ))
                C2r[y,x] += v*Float32(cos(2φ)); C2i[y,x] -= v*Float32(sin(2φ))
            end
        end
        m1m = Matrix{Float32}(undef,H,W); m2m = Matrix{Float32}(undef,H,W)
        @inbounds for p in eachindex(C0)
            m1m[p] = sqrt(C1r[p]^2 + C1i[p]^2)/K; m2m[p] = sqrt(C2r[p]^2 + C2i[p]^2)/K
            C0[p] /= K
        end
        push!(maps, C0);   push!(labels, (form=:R0, rho0=0.0, d=0.0))
        push!(maps, m1m);  push!(labels, (form=:R1, rho0=0.0, d=0.0))
        push!(maps, m2m);  push!(labels, (form=:R2, rho0=0.0, d=0.0))
        push!(maps, Sbest);push!(labels, (form=:Rs, rho0=0.0, d=0.0))
        out = Array{Float32,3}(undef, H, W, length(maps))
        for (k,M) in enumerate(maps); out[:,:,k] = M; end
        return out, labels
    end

    for ρ in scales_of(meta)
        ch = chan_of(meta, ρ); n = length(ch)
        K = nphi === nothing ? 2n : Int(nphi)
        m1 = first(m for m in meta if m.kind === :oriented && m.rho0 ≈ ρ)
        # Probe radius. `:envelope` (default) ties d to the filter's own along-contour extent,
        # d ∝ λ = imwidth/ρ — correct when image content scales with the frame, wrong when
        # content keeps its pixel size and the frame grows. `:structure` ties it to a scale
        # measured from the image instead. Same switch, same defaults and same reasoning as
        # `AndLayer.a2_maps`; it was added there first and this function was missed, while
        # RESULTS.md claimed both had it.
        dρ = if d !== :auto
            Float64(d)
        elseif d_anchor === :structure
            d_factor * Float64(structure_scale)
        else
            d_factor * m1.imwidth / (2π * ρ * m1.sigma_phi)
        end

        # for ray direction φ, the stroke runs along φ, so the carrier is φ+90° (mod π)
        pick = Vector{Int}(undef, K); φs = Vector{Float64}(undef, K)
        for j in 1:K
            φ = 2π * (j-1) / K; φs[j] = φ
            want = mod(φ + π/2, π)
            best = 1; bd = Inf
            for (k,(_,θ)) in enumerate(ch)
                δ = abs(atan(sin(θ-want), cos(θ-want)))
                δ < bd && (bd = δ; best = k)
            end
            pick[j] = ch[best][1]
        end

        C0 = zeros(Float32,H,W); C1r = zeros(Float32,H,W); C1i = zeros(Float32,H,W)
        C2r = zeros(Float32,H,W); C2i = zeros(Float32,H,W)
        # Gather the profile per pixel first so `normalize` can act on it before the
        # harmonics are taken — the same treatment the `:max` path gets, so the two designs
        # are comparable with normalisation held constant. Without this the per-scale path
        # silently ignored `normalize`, and a "per_scale + divisive" run came out
        # bit-identical to "per_scale + none" — a fabricated cell in a 2x2 comparison.
        Rp = Array{Float32,3}(undef, H, W, K)
        for j in 1:K
            φ = φs[j]; dy = dρ*sin(φ); dx = dρ*cos(φ)
            Ej = @view E[:, :, pick[j]]
            @inbounds for x in 1:W, y in 1:H
                Rp[y,x,j] = bilin(Ej, y + dy, x + dx)
            end
        end
        prof = Vector{Float32}(undef, K)
        @inbounds for x in 1:W, y in 1:H
            for j in 1:K; prof[j] = Rp[y,x,j]; end
            normalize === :divisive && normprofile!(prof, kappa)
            for j in 1:K
                φ = φs[j]; v = prof[j]
                C0[y,x]  += v
                C1r[y,x] += v*Float32(cos(φ));  C1i[y,x] -= v*Float32(sin(φ))
                C2r[y,x] += v*Float32(cos(2φ)); C2i[y,x] -= v*Float32(sin(2φ))
            end
        end
        # Returned **unnormalised**: c₀, |c₁| and |c₂| are all energies, and the ratios are
        # formed *after* pooling by the caller. See the note on normalisation above.
        m1 = Matrix{Float32}(undef,H,W); m2 = Matrix{Float32}(undef,H,W)
        @inbounds for p in eachindex(C0)
            m1[p] = sqrt(C1r[p]^2 + C1i[p]^2) / K
            m2[p] = sqrt(C2r[p]^2 + C2i[p]^2) / K
            C0[p] = C0[p] / K                    # mean, so all three are comparable across K
        end
        push!(maps, C0); push!(labels, (form=:R0, rho0=ρ, d=dρ))
        push!(maps, m1); push!(labels, (form=:R1, rho0=ρ, d=dρ))
        push!(maps, m2); push!(labels, (form=:R2, rho0=ρ, d=dρ))
    end
    out = Array{Float32,3}(undef, H, W, length(maps))
    for (k,M) in enumerate(maps); out[:,:,k] = M; end
    out, labels
end

end # module
