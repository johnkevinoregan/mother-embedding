# ── PLAIN MODULE — .module.jl, not a Pluto notebook ─────────────────────────
# Included by other files. Opening it in Pluto rewrites it and leaves a
# "<name> backup 1.jl" beside it. Notebooks are the plain .jl files.

"""
    Stimuli

Synthetic figures with known ground truth, shared by the validation batteries.

These exist so the front end can be checked against what it is *supposed to compute*
rather than only against downstream accuracy. The negative controls matter as much as
the positive ones: a corner and two disjoint strokes have the same orientation content
and differ only in whether the orientations meet, so any detector that cannot separate
them is not measuring i2D structure however good its classification numbers look.

Size-agnostic — nothing here assumes 112, or characters, or EMNIST.
"""
module Stimuli

export barstim, corner, two_bars, gabor_patch, blob, tee, cross_bars

"""
Bar of width `w`, length `len`, orientation `θ`, centred at `(cy,cx)`.

Named `barstim` rather than `bar` because `Plots` also exports `bar`; when two modules
export the same name Julia binds neither, and the failure surfaces far from its cause.
"""
function barstim(N::Int, θ::Real; w::Real=13.0, len::Real=70.0,
             cy::Real=(N+1)/2, cx::Real=(N+1)/2)
    I = zeros(Float32, N, N); s, c = sin(θ), cos(θ)
    @inbounds for y in 1:N, x in 1:N
        dy = y - cy; dx = x - cx
        t =  dx*c + dy*s                       # along the bar
        n = -dx*s + dy*c                       # across it
        (abs(t) <= len/2 && abs(n) <= w/2) && (I[y, x] = 1f0)
    end
    I
end

"Isolated round blob of radius `r` — a termination with *no* attached stroke."
function blob(N::Int; r::Real=6.5, cy::Real=(N+1)/2, cx::Real=(N+1)/2)
    I = zeros(Float32, N, N)
    @inbounds for y in 1:N, x in 1:N
        hypot(y - cy, x - cx) <= r && (I[y, x] = 1f0)
    end
    I
end

"""
Two half-bars meeting at the centre with angle `α` between them: an L-corner at α=π/2.
Both arms start at the centre, so the orientations genuinely **meet at a point**.
"""
function corner(N::Int, α::Real; w::Real=13.0, len::Real=40.0)
    c = (N+1)/2
    I = zeros(Float32, N, N)
    for θ in (0.0, α)
        I .= max.(I, barstim(N, θ; w=w, len=len,
                         cy=c + (len/2)*sin(θ), cx=c + (len/2)*cos(θ)))
    end
    I
end

"""
    two_bars(N, gap; w, len)

Two bars at 90° whose **centres** are `gap` apart along the diagonal — the negative
control for `corner`. Identical orientation content at every `gap`; only co-location
changes. At `gap = 0` they cross at the centre (maximal co-location); by `gap ≈ 70` their
nearest points are ~30 px apart, comfortably beyond every σ_x in the bank.

Separating perpendicular bars requires moving them along the diagonal. Offsetting each
perpendicular to *itself* barely separates them at all — a 45 px "gap" built that way
leaves the bars 3.5 px apart, which is inside every envelope in the bank and therefore
still co-located.
"""
function two_bars(N::Int, gap::Real; w::Real=13.0, len::Real=40.0)
    c = (N+1)/2; o = gap / (2 * sqrt(2))
    max.(barstim(N, 0.0;  w=w, len=len, cy=c-o, cx=c-o),
         barstim(N, π/2;  w=w, len=len, cy=c+o, cx=c+o))
end

"T-junction: a full bar with a stem meeting its middle (3 rays)."
function tee(N::Int; w::Real=13.0, len::Real=70.0)
    c = (N+1)/2
    max.(barstim(N, 0.0; w=w, len=len, cy=c, cx=c),
         barstim(N, π/2;  w=w, len=len/2, cy=c + len/4, cx=c))
end

"X-crossing: two full bars crossing at the centre (4 rays)."
cross_bars(N::Int; w::Real=13.0, len::Real=70.0, α::Real=π/2) =
    max.(barstim(N, 0.0; w=w, len=len), barstim(N, α; w=w, len=len))

"Gabor patch (Gaussian-windowed grating) — clean spectral content at one `λ`."
gabor_patch(N::Int, λ::Real, θ::Real; σ::Real=22.0) = (c = (N+1)/2;
    Float32[exp(-((y-c)^2 + (x-c)^2)/(2σ^2)) *
            cos(2π*((x-c)*cos(θ) + (y-c)*sin(θ))/λ) for y in 1:N, x in 1:N])

end # module
