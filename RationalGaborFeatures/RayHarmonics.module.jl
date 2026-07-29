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

Returns `(maps, labels)` with three channels per scale: `c0`, `|c1|/c0`, `|c2|/c0`.

`nphi` ray directions over [0, 2π); defaults to twice the orientation count of the scale,
so each orientation channel is used once in each of its two directions. `d = :auto` sets
the probe radius from the channel's own along-contour envelope, as `AndLayer` does — see
the caveat in `RESULTS.md` about that anchoring being wrong in general.
"""
function ray_maps(E::Array{Float32,3}, meta; nphi=nothing, d=:auto,
                  d_factor::Real=1.0, eps::Float32=1f-12)
    H, W, _ = size(E)
    maps = Matrix{Float32}[]; labels = NamedTuple[]
    for ρ in scales_of(meta)
        ch = chan_of(meta, ρ); n = length(ch)
        K = nphi === nothing ? 2n : Int(nphi)
        m1 = first(m for m in meta if m.kind === :oriented && m.rho0 ≈ ρ)
        dρ = d === :auto ? d_factor * m1.imwidth / (2π * ρ * m1.sigma_phi) : Float64(d)

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
        for j in 1:K
            φ = φs[j]; dy = dρ*sin(φ); dx = dρ*cos(φ)
            Ej = @view E[:, :, pick[j]]
            c1c, c1s = Float32(cos(φ)),   Float32(sin(φ))
            c2c, c2s = Float32(cos(2φ)),  Float32(sin(2φ))
            @inbounds for x in 1:W, y in 1:H
                v = bilin(Ej, y + dy, x + dx)
                C0[y,x]  += v
                C1r[y,x] += v*c1c; C1i[y,x] -= v*c1s
                C2r[y,x] += v*c2c; C2i[y,x] -= v*c2s
            end
        end
        r1 = Matrix{Float32}(undef,H,W); r2 = Matrix{Float32}(undef,H,W)
        @inbounds for p in eachindex(C0)
            c0 = C0[p]
            if c0 > eps
                r1[p] = sqrt(C1r[p]^2 + C1i[p]^2)/c0
                r2[p] = sqrt(C2r[p]^2 + C2i[p]^2)/c0
            else
                r1[p] = 0f0; r2[p] = 0f0
            end
            C0[p] = c0 / K                       # mean, so c0 is comparable across K
        end
        push!(maps, C0); push!(labels, (form=:R0, rho0=ρ, d=dρ))
        push!(maps, r1); push!(labels, (form=:R1, rho0=ρ, d=dρ))
        push!(maps, r2); push!(labels, (form=:R2, rho0=ρ, d=dρ))
    end
    out = Array{Float32,3}(undef, H, W, length(maps))
    for (k,M) in enumerate(maps); out[:,:,k] = M; end
    out, labels
end

end # module
