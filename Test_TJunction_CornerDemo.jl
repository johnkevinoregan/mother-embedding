### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ c0000000-0000-0000-0000-000000000001
begin
    import Pkg
    Pkg.activate(@__DIR__)
end

# ╔═╡ c0000000-0000-0000-0000-000000000002
begin
    using PlutoUI
    using Plots
    using Colors
end

# ╔═╡ c0000000-0000-0000-0000-000000000003
begin
    include(joinpath(@__DIR__, "Config.jl"))
    using .Config
    include(joinpath(@__DIR__, "CreateGaborLifting.jl"))
    using .CreateGaborLifting
    include(joinpath(@__DIR__, "CreateTJunctionLifting.jl"))
    using .CreateTJunctionLifting
end

# ╔═╡ c0000000-0000-0000-0000-000000000004
md"""
# T-junction detector: T's **and** all four corners

`CreateTJunctionLifting` combines a stem and a crossbar Gabor into a strength
via a **phase-compatibility** term. This notebook compares the old term against
the current one on controlled synthetic stimuli — a T and the four corner
orientations — so you can see exactly where they differ.

- **old**: `phase_compat = (1 + cos(cross.phase − stem.phase)) / 2`
- **new**: `phase_compat = clamp(cos φ_s·cos φ_c + |sin φ_s·sin φ_c|, 0, 1)`

The old term matches phases directly. That works for **line⟂line T's** (even
phases, 0/π) but mishandles **corners**: edge⟂edge phases are ±90°, and that
±90 sign is an artifact of canonicalizing orientation to [0, π). So the old
term fires on corners whose region straddles one diagonal (phases ≈ 90/90) and
**cancels** the mirror-image corners on the other diagonal (≈ 90/270 → Δ ≈ π).

The new term treats the even (line) and odd (edge) parts separately: it matches
the *sign* of the even part (0 vs π is a real bright/dark polarity) but is
*sign-blind* on the odd part. Result: full strength for same-polarity T's **and
all four corners**, still zero for opposite-polarity crossings, and invariant to
both contrast inversion and the per-detector normal-sign artifact.

**How to read the effect.** The two-panel overlay below is *max-pooled* per grid
point, which partly hides the difference (the old rule can find some other
weakly-compatible pair nearby). The crisp comparison is the **winning-pairing
readout** under it, which probes the actual junction pairing, and the summary
table at the bottom.
"""

# ╔═╡ c0000000-0000-0000-0000-000000000005
begin
    const N_DEMO = 48
    to_image(m) = reverse(m, dims=1)                       # row 1 at top for heatmap
    old_compat(s, c) = (1 + cos(c.phase - s.phase)) / 2
    new_compat(s, c) = clamp(cos(s.phase)*cos(c.phase) +
                             abs(sin(s.phase)*sin(c.phase)), 0f0, 1f0)
    degmod(x) = round(Int, mod(rad2deg(x), 360))
    # NEIGHBOR_DIRECTIONS / closest_orientation are internal to the module.
    ND = CreateTJunctionLifting.NEIGHBOR_DIRECTIONS
    closest = CreateTJunctionLifting.closest_orientation
end

# ╔═╡ c0000000-0000-0000-0000-000000000006
# For one image and one scale index: the best (old, new) strength over the 8
# directions at every grid point, plus the single pairing with the strongest
# NEW strength (the junction the detector "means"). Mirrors t_junction_lift's
# pairing but scores both compat formulas.
function pair_scores(img, si)
    N = N_DEMO
    gs = gabor_lift(img; img_size=N)
    lk = Dict{Tuple{Int,Int,Int,Float32}, eltype(gs)}()
    for s in gs
        lk[(round(Int, s.x*(N-1))+1, round(Int, s.y*(N-1))+1, round(Int, s.s), s.θ)] = s
    end
    λ = SCALES[si]
    step = max(1, round(Int, (1 - OVERLAP_FRAC) * λ))
    gc = collect(step÷2+1 : step : N); lo, hi = first(gc), last(gc)
    best = Dict{Tuple{Int,Int}, Tuple{Float32,Float32}}()
    winner = nothing
    for cx in gc, cy in gc
        bo = bn = 0f0
        for (dx, dy, α) in ND
            nx = cx + dx*step; ny = cy + dy*step
            (lo <= nx <= hi && lo <= ny <= hi) || continue
            s = lk[(cx, cy, si, closest(mod(α, Float32(π)), ORIENTATIONS))]
            c = lk[(nx, ny, si, closest(mod(α + Float32(π/2), Float32(π)), ORIENTATIONS))]
            mm = min(s.modulus, c.modulus)
            oc = old_compat(s, c); nc = new_compat(s, c)
            bo = max(bo, mm*oc); bn = max(bn, mm*nc)
            if winner === nothing || mm*nc > winner.ns
                winner = (ns = mm*nc, cx = cx, cy = cy, sp = s.phase, cp = c.phase,
                          oc = oc, nc = nc, mm = mm)
            end
        end
        best[(cx, cy)] = (bo, bn)
    end
    return (best, winner, step)
end

# ╔═╡ c0000000-0000-0000-0000-000000000007
# Controlled stimuli: the four corner orientations (a bright quadrant meeting at
# a corner), a T of bright bars, an opposite-polarity crossing, and a straight
# edge as a control.
stimuli = let N = N_DEMO
    mk() = zeros(Float32, N, N)
    c_ne = mk(); c_ne[1:24, 24:end] .= 1f0
    c_nw = mk(); c_nw[1:24, 1:24]   .= 1f0
    c_sw = mk(); c_sw[24:end, 1:24]  .= 1f0
    c_se = mk(); c_se[24:end, 24:end].= 1f0
    tj   = mk(); tj[10:13, 8:40] .= 1f0; tj[10:40, 22:25] .= 1f0
    opp  = fill(0.5f0, N, N); opp[22:25, :] .= 1f0; opp[:, 22:25] .= 0f0
    edge = mk(); edge[:, 24:end] .= 1f0
    ["corner ↗ (bright NE)" => c_ne, "corner ↖ (bright NW)" => c_nw,
     "corner ↙ (bright SW)" => c_sw, "corner ↘ (bright SE)" => c_se,
     "T-junction" => tj, "opposite-polarity ✚" => opp,
     "straight edge (control)" => edge]
end

# ╔═╡ c0000000-0000-0000-0000-000000000008
md"""
## Explore one stimulus

**Stimulus**: $(@bind stim_name Select(first.(stimuli)))

**Scale λ**: $(@bind lam_sel Select(SCALES, default=6f0))
"""

# ╔═╡ c0000000-0000-0000-0000-000000000009
selected_stim = Dict(stimuli)[stim_name]

# ╔═╡ c0000000-0000-0000-0000-00000000000a
scan_result = pair_scores(selected_stim, findfirst(==(lam_sel), SCALES))

# ╔═╡ c0000000-0000-0000-0000-00000000000b
let
    best, _, _ = scan_result
    gmax = maximum(max(o, n) for (o, n) in values(best)); gmax = gmax == 0 ? 1f0 : gmax
    function panel(idx, ttl)
        p = heatmap(to_image(selected_stim), color=:grays, aspect_ratio=:equal,
                    axis=false, colorbar=false, title=ttl, titlefontsize=9,
                    xlim=(0.5, N_DEMO+0.5), ylim=(0.5, N_DEMO+0.5))
        for ((cx, cy), v) in best
            a = v[idx] / gmax
            a < 0.08 && continue
            scatter!(p, [cx], [N_DEMO - cy + 1], markersize=7, markerstrokewidth=0,
                     color=RGBA(1, 0.15, 0.15, clamp(a, 0, 1)), label=false)
        end
        p
    end
    plot(panel(1, "OLD  (1+cos Δφ)/2"), panel(2, "NEW  cos·cos+|sin·sin|"),
         layout=(1, 2), size=(760, 410))
end

# ╔═╡ c0000000-0000-0000-0000-00000000000c
let
    w = scan_result[2]
    verdict = w.oc < 0.3 && w.nc > 0.5 ? "**old misses this, new keeps it.**" :
              w.oc > 0.5 && w.nc > 0.5 ? "both keep it." :
              w.oc < 0.3 && w.nc < 0.3 ? "both reject it." : "(mixed)"
    md"""
    **Winning pairing** for *$stim_name* at λ=$(Int(lam_sel)) (probed directly):
    stem φ = **$(degmod(w.sp))°**, crossbar φ = **$(degmod(w.cp))°** →
    old compat **$(round(w.oc, digits=2))**, new compat **$(round(w.nc, digits=2))** — $verdict
    """
end

# ╔═╡ c0000000-0000-0000-0000-00000000000d
md"""
## Summary: every stimulus at once

Winning-pairing phases and old vs new compat, taking the strongest NEW pairing
over scales λ=6 and λ=12. Corners whose winner is odd/odd-opposite (≈ 90/270)
are the ones the old term cancels and the new term rescues.
"""

# ╔═╡ c0000000-0000-0000-0000-00000000000e
let
    rows = ["| stimulus | stem/cross φ | old | new |", "|---|---|---|---|"]
    for (nm, img) in stimuli
        w = nothing
        for si in findall(x -> x in (6f0, 12f0), SCALES)
            _, wi, _ = pair_scores(img, si)
            (w === nothing || wi.ns > w.ns) && (w = wi)
        end
        push!(rows, "| $nm | ($(degmod(w.sp))°, $(degmod(w.cp))°) | " *
                    "$(round(w.oc, digits=2)) | $(round(w.nc, digits=2)) |")
    end
    Markdown.parse(join(rows, "\n"))
end

# ╔═╡ Cell order:
# ╠═c0000000-0000-0000-0000-000000000001
# ╠═c0000000-0000-0000-0000-000000000002
# ╠═c0000000-0000-0000-0000-000000000003
# ╟─c0000000-0000-0000-0000-000000000004
# ╠═c0000000-0000-0000-0000-000000000005
# ╠═c0000000-0000-0000-0000-000000000006
# ╠═c0000000-0000-0000-0000-000000000007
# ╟─c0000000-0000-0000-0000-000000000008
# ╠═c0000000-0000-0000-0000-000000000009
# ╠═c0000000-0000-0000-0000-00000000000a
# ╠═c0000000-0000-0000-0000-00000000000b
# ╟─c0000000-0000-0000-0000-00000000000c
# ╟─c0000000-0000-0000-0000-00000000000d
# ╠═c0000000-0000-0000-0000-00000000000e
