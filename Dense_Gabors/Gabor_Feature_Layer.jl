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

# ╔═╡ 20000000-0000-0000-0000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 20000000-0000-0000-0000-000000000002
begin
    using PlutoUI
    using Plots
    using Colors
    using ImageFiltering
    using Random
end

# ╔═╡ 20000000-0000-0000-0000-000000000003
begin
    include(joinpath(@__DIR__, "..", "LoadEMNIST.jl"))
    using .LoadEMNIST
end

# ╔═╡ 20000000-0000-0000-0000-000000000004
md"""
# Gabor feature layer (Julia port)

A faithful Julia/Pluto port of `Gabor_feature_layer_python/feature_layer.py` —
the mid-level feature-type layer on the Gabor oriented-energy field. It turns
the fixed-scale complex-Gabor stack (from the orientation demo) into discrete,
typed keypoints — **endpoint / corner / T-junction / X-crossing** — using only
operations that read off the oriented-energy field:

1. **end-stopping** along the winning orientation (fires where an oriented
   segment terminates → endpoints);
2. **orientation-profile peak structure** (1 genuine peak = simple edge,
   2 = corner/junction — with a bimodality gate so curves don't fake corners);
3. a **multi-radius ring "spoke count"** (# contour branches → junction order).

The architecture is *propose ≠ classify*: the two dense per-pixel operations
propose points and decide corner-family vs endpoint; the sparse geometric ring
count only refines an already-proposed point (corner → T → X). The rationale
for every threshold is in `Gabor_feature_layer_python/Gabor_feature_layer_design_notes.md`.

**Port notes.** `scipy.fftconvolve(…, "same")` → `imfilter(…, Fill(0))`;
`scipy.ndimage.shift` / `map_coordinates` (both `order=1`, constant 0 outside)
→ one shared `bilinear` sampler; `scipy.gaussian_filter1d(σ=1, mode="wrap")`
along θ → an explicit circular 9-tap Gaussian; cubic `zoom` upsample →
bilinear (as in the orientation-demo port); EMNIST via `LoadEMNIST` instead of
the Python `emnist` package. All coordinates are `(y, x)` row/col, 1-based.
"""

# ╔═╡ 20000000-0000-0000-0000-000000000005
# The Python Config, as used by feature_layer.py's __main__: defaults except
# n_orient=60. lam is the carrier wavelength (THE "scale"), sigma_n/sigma_t the
# across-/along-contour envelope widths, on the 4x-upsampled 112x112 image.
cfg = (
    lam      = 12.0,
    sigma_n  = 6.0,
    sigma_t  = 11.0,
    ksize    = 41,
    n_orient = 60,
    upsample = 4,
    noise_sigma = 0.03,
    seed     = 0,
    chars    = ("A", "K", "X", "T"),
)

# ╔═╡ 20000000-0000-0000-0000-000000000006
md"""
## Gabor bank and oriented-energy stack

Same complex Gabor as the orientation demo (θ = contour tangent, carrier along
the normal, zero DC). `energy_stack` is the modulus of the full θ-bank:
`S[t, y, x] = |Gabor_θt * img|`.
"""

# ╔═╡ 20000000-0000-0000-0000-000000000007
# Complex Gabor whose preferred (stripe) orientation is θ.
function gabor(θ, cfg)
    h = cfg.ksize ÷ 2
    g = zeros(ComplexF64, cfg.ksize, cfg.ksize)
    for (ii, y) in enumerate(-h:h), (jj, x) in enumerate(-h:h)
        xt =  x*cos(θ) + y*sin(θ)          # along tangent
        xn = -x*sin(θ) + y*cos(θ)          # along normal
        env = exp(-(xt^2/(2cfg.sigma_t^2) + xn^2/(2cfg.sigma_n^2)))
        car = cis(2π * xn / cfg.lam)       # modulate across the contour
        g[ii, jj] = env * car
    end
    return g .- sum(g)/length(g)           # zero DC → uniform regions give ~0
end

# ╔═╡ 20000000-0000-0000-0000-000000000008
begin
    # Bilinear sample of A at fractional (y, x); 0 outside — the stand-in for
    # scipy's order=1 shift / map_coordinates with mode="constant".
    function bilinear(A, y, x)
        h, w = size(A)
        y0 = floor(Int, y); x0 = floor(Int, x)
        fy = y - y0; fx = x - x0
        v = 0.0
        for (yy, wy) in ((y0, 1 - fy), (y0 + 1, fy)),
            (xx, wx) in ((x0, 1 - fx), (x0 + 1, fx))
            if wy*wx > 0 && 1 <= yy <= h && 1 <= xx <= w
                v += wy*wx * A[yy, xx]
            end
        end
        return v
    end

    # Bilinear upscale (Python used cubic scipy.ndimage.zoom).
    function upsample_bilinear(img, factor)
        h, w = size(img)
        nh, nw = h*factor, w*factor
        return [bilinear(img, 1 + (i-1)*(h-1)/(nh-1), 1 + (j-1)*(w-1)/(nw-1))
                for i in 1:nh, j in 1:nw]
    end

    # prep(a, cfg): upsample 28 → 112 + clamp + Gaussian noise floor.
    function prep(a, cfg)
        b = clamp.(upsample_bilinear(a, cfg.upsample), 0, 1)
        if cfg.noise_sigma > 0
            b = clamp.(b .+ cfg.noise_sigma .* randn(MersenneTwister(cfg.seed), size(b)...), 0, 1)
        end
        return b
    end
end

# ╔═╡ 20000000-0000-0000-0000-000000000009
function energy_stack(img, thetas, cfg)
    h, w = size(img)
    S = Array{Float64,3}(undef, length(thetas), h, w)
    for (t, θ) in enumerate(thetas)
        S[t, :, :] .= abs.(imfilter(img, centered(gabor(θ, cfg)), Fill(zero(ComplexF64))))
    end
    return S
end

# ╔═╡ 20000000-0000-0000-0000-00000000000a
md"""
## Operation 1 — end-stopping along the winning orientation

At each pixel, compare the energy to the mean of two samples a distance `d`
away *along the contour tangent*: `max(S − ½(S₊ + S₋), 0)`. Big where an
oriented segment terminates (the classic hypercomplex-cell nonlinearity).
"""

# ╔═╡ 20000000-0000-0000-0000-00000000000b
function end_stopping(S, thetas; d=15.0)
    ES = zero(S)
    N, h, w = size(S)
    for (t, th) in enumerate(thetas)
        sx, sy = cos(th)*d, sin(th)*d
        At = view(S, t, :, :)
        for j in 1:w, i in 1:h
            plus  = bilinear(At, i + sy, j + sx)   # sample p + d*tangent
            minus = bilinear(At, i - sy, j - sx)   # sample p - d*tangent
            ES[t, i, j] = max(S[t, i, j] - 0.5*(plus + minus), 0.0)
        end
    end
    return ES
end

# ╔═╡ 20000000-0000-0000-0000-00000000000c
md"""
## Operation 2 — orientation-profile peak structure

Per pixel: smooth the modulus(θ) profile circularly, count local peaks above
`relthr`·max, and return a **bimodality-gated** second-peak height. The second
peak counts only if it is ≥ `sep_bins` away in orientation AND there is a real
valley between the two peaks (below `0.62 ×` the smaller peak) — so a smoothly
curving contour (one broad, rotating peak) does not masquerade as a corner.
"""

# ╔═╡ 20000000-0000-0000-0000-00000000000d
# Return (n_peaks, first-peak height, gated second-peak height) per pixel.
function peak_maps(S; relthr=0.40, sep_bins=10)
    N, h, w = size(S)
    M = dropdims(maximum(S, dims=1), dims=1)
    # circular Gaussian smoothing along θ (scipy gaussian_filter1d σ=1, wrap: 9 taps)
    r = 4
    wts = exp.(-(collect(-r:r)).^2 ./ 2); wts ./= sum(wts)
    Ss = similar(S)
    for j in 1:w, i in 1:h, t in 1:N
        acc = 0.0
        for (ki, k) in enumerate(-r:r)
            acc += wts[ki] * S[mod1(t + k, N), i, j]
        end
        Ss[t, i, j] = acc
    end
    npk = zeros(Int, h, w); first_pk = zeros(h, w); second_pk = zeros(h, w)
    hpk = zeros(N)
    for j in 1:w, i in 1:h
        thr = relthr * max(M[i, j], 1e-9)
        c = 0
        for t in 1:N
            v = Ss[t, i, j]
            ispk = v >= Ss[mod1(t + 1, N), i, j] && v > Ss[mod1(t - 1, N), i, j] && v > thr
            hpk[t] = ispk ? v : 0.0
            c += ispk
        end
        npk[i, j] = c
        i1 = argmax(hpk); first_pk[i, j] = hpk[i1]
        for s in -sep_bins:sep_bins            # blank window around 1st peak
            hpk[mod1(i1 + s, N)] = 0.0
        end
        i2 = argmax(hpk); sec = hpk[i2]
        # valley test: minimum of the profile on the arc i1 → i2 must sit well
        # below the smaller peak (a genuine bimodal dip, not a shoulder).
        valley = Inf
        for s in 0:mod(i2 - i1, N)
            valley = min(valley, Ss[mod1(i1 + s, N), i, j])
        end
        second_pk[i, j] = sec * (valley < 0.62 * min(max(sec, 1e-9), first_pk[i, j]))
    end
    return npk, first_pk, second_pk
end

# ╔═╡ 20000000-0000-0000-0000-00000000000e
md"""
## Operation 3 — ring spoke count

Sample the max-energy field `M` on a circle of radius `r` around a point and
count runs of "on" samples (contour branches crossing the ring). Degenerate
rings — empty (0) or all-on (the `99` sentinel: the ring sits inside the
junction blob) — are rejected. `classify_point` samples several radii and
accepts a junction order of 3 (T) or 4 (X) **only if confirmed at ≥ 2 radii**;
two arcs ≈ 180° apart = a straight stroke passing through.
"""

# ╔═╡ 20000000-0000-0000-0000-00000000000f
# Count contour branches crossing a ring of radius r; also their arc centers.
function arc_count(M, y, x, r; K=72, relthr=0.4, min_run=3)
    ph = [2π*(k - 1)/K for k in 1:K]
    g = [bilinear(M, y + r*sin(p), x + r*cos(p)) for p in ph]
    gmax = maximum(g)
    gmax <= 0 && return 0, Float64[]
    on = g .> relthr*gmax
    all(on) && return 99, Float64[]
    n = 0; centers = Float64[]
    for st in 1:K
        (on[st] && !on[mod1(st - 1, K)]) || continue
        run = Int[]; j = st
        while on[mod1(j, K)] && (j - st) <= K
            push!(run, mod1(j, K)); j += 1
        end
        if length(run) >= min_run
            n += 1
            push!(centers, angle(sum(cis(ph[t]) for t in run) / length(run)))
        end
    end
    return n, centers
end

# ╔═╡ 20000000-0000-0000-0000-000000000010
function classify_point(M, y, x; radii=(15, 18, 21, 24))
    res = [arc_count(M, y, x, r) for r in radii]
    valid = [n for (n, _) in res if 0 < n < 99]
    isempty(valid) && return nothing
    cnt(k) = count(==(k), valid)
    confirmed = [k for k in (4, 3) if cnt(k) >= 2]   # junction order confirmed at >=2 radii
    if !isempty(confirmed)
        return maximum(confirmed) == 4 ? "X" : "T"
    end
    collinear = any(n == 2 && length(c) == 2 &&
                    abs(abs(angle(cis(c[1] - c[2]))) - π) < deg2rad(30)
                    for (n, c) in res)
    ones = count(==(1), valid)
    ones >= length(valid)/2 && return "endpoint"
    2 in valid && return (collinear ? "straight" : "corner")
    return "endpoint"
end

# ╔═╡ 20000000-0000-0000-0000-000000000011
md"""
## Detection: propose (dense) → refine (ring)

The coarse class comes from *which operation fires*: two genuine orientation
peaks → **corner-family** (`cornerness`); a single orientation whose energy
stops → **endpoint** (`endpointness`, gated to sit on a real ridge). Each map
gets non-maximum suppression; proposals within 11 px of an already-kept point
are dropped. The ring count then only refines: corner → T → X (and rejects
mid-stroke "straight" false alarms from the endpoint stream).
"""

# ╔═╡ 20000000-0000-0000-0000-000000000012
# Non-maximum suppression: local maxima of an md×md window above frac·max.
function nms(score, md, frac)
    h, w = size(score); r = md ÷ 2
    smax = maximum(score)
    pts = Tuple{Int,Int}[]
    for j in 1:w, i in 1:h
        s = score[i, j]
        (s > frac*smax && s > 0) || continue
        ismax = true
        for jj in max(1, j-r):min(w, j+r), ii in max(1, i-r):min(h, i+r)
            if score[ii, jj] > s
                ismax = false; break
            end
        end
        ismax && push!(pts, (i, j))
    end
    return pts
end

# ╔═╡ 20000000-0000-0000-0000-000000000013
function detect(img, cfg; end_stop_d=12.0)
    thetas = collect(range(0, π; length=cfg.n_orient + 1))[1:end-1]   # endpoint=false
    S = energy_stack(img, thetas, cfg)
    N, h, w = size(S)
    M = dropdims(maximum(S, dims=1), dims=1); Mmax = maximum(M)
    ES = end_stopping(S, thetas; d=end_stop_d)
    ori = zeros(h, w); ESw = zeros(h, w)
    for j in 1:w, i in 1:h
        k = argmax(view(S, :, i, j))
        ori[i, j] = thetas[k]
        ESw[i, j] = ES[k, i, j]                # end-stopping at the winning θ
    end
    npk, first_pk, second_pk = peak_maps(S)

    # coarse class comes from WHICH operation fires:
    #   orientation change (two genuine orientation peaks) -> corner-family
    #   segment termination (end-stopping, single orientation) -> endpoint
    cornerness   = second_pk .* (npk .>= 2) .* (second_pk .> 0.5 .* first_pk) .* (M .> 0.20*Mmax)
    endpointness = ESw .* (npk .<= 1) .* (M .> 0.35*Mmax)   # must sit on a real ridge

    proposals = vcat([("corner", y, x) for (y, x) in nms(cornerness, 11, 0.35)],
                     [("endpoint", y, x) for (y, x) in nms(endpointness, 11, 0.38)])
    kept = Tuple{String,Int,Int}[]; taken = Tuple{Int,Int}[]
    for (src, y, x) in proposals
        any((y - yy)^2 + (x - xx)^2 < 11^2 for (yy, xx) in taken) && continue
        ring = classify_point(M, y, x)         # "corner"/"T"/"X"/"endpoint"/"straight"/nothing
        if src == "corner"
            typ = (ring == "T" || ring == "X") ? ring : "corner"
        else                                   # endpoint proposal
            if ring == "T" || ring == "X"
                typ = ring                     # ring overrides if clearly a junction
            elseif ring == "straight"
                continue                       # mid-stroke false alarm
            else
                typ = "endpoint"
            end
        end
        push!(kept, (typ, y, x)); push!(taken, (y, x))
    end
    return (; img, M, ori, ESw, cornerness, keypoints=kept)
end

# ╔═╡ 20000000-0000-0000-0000-000000000014
md"""
## Characters and detection
"""

# ╔═╡ 20000000-0000-0000-0000-000000000015
# One real EMNIST sample per requested letter (class 11 = 'A' in EMNIST balanced).
emnist = load_emnist(n_images_to_load=5000, n_classes=length(cfg.chars),
                     class_order=[11 + Int(c[1]) - Int('A') for c in cfg.chars])

# ╔═╡ 20000000-0000-0000-0000-000000000016
imgs = Dict(c => prep(Float64.(emnist.class_images[i][1]), cfg)
            for (i, c) in enumerate(cfg.chars))

# ╔═╡ 20000000-0000-0000-0000-000000000017
res = Dict(c => detect(imgs[c], cfg) for c in cfg.chars)

# ╔═╡ 20000000-0000-0000-0000-000000000018
md"""
### Fig 5 — labelled keypoints

**red square = corner  ·  cyan circle = endpoint  ·  yellow triangle = T  ·
magenta star = X**
"""

# ╔═╡ 20000000-0000-0000-0000-000000000019
begin
    STYLE = Dict("corner"   => (:rect,      "#ff3b30"),
                 "endpoint" => (:circle,    "#00c2d1"),
                 "T"        => (:utriangle, "#ffcc00"),
                 "X"        => (:star5,     "#ff2fd0"))

    # Open (unfilled) marker outlines as line paths — GR renders "transparent"
    # marker fills as black, so we draw the outlines ourselves. Image y grows
    # downward, hence sin(-π/2) = up for the triangle apex / star tip.
    function marker_path(mk, r)
        if mk == :circle
            θ = range(0, 2π; length=33)
        elseif mk == :rect
            return r .* [-1, 1, 1, -1, -1], r .* [-1, -1, 1, 1, -1]
        elseif mk == :utriangle
            θ = -π/2 .+ 2π/3 .* (0:3)
        else                                       # :star5
            θ = -π/2 .+ π/5 .* (0:10)
            rr = [iseven(k) ? r : 0.45r for k in 0:10]
            return rr .* cos.(θ), rr .* sin.(θ)
        end
        return r .* cos.(θ), r .* sin.(θ)
    end
end

# ╔═╡ 20000000-0000-0000-0000-00000000001a
fig5 = let
    panels = map(collect(cfg.chars)) do c
        r = res[c]
        p = plot(Gray.(r.img), axis=false, title=c, titlefontsize=12)
        for (typ, y, x) in r.keypoints
            mk, col = STYLE[typ]
            dx, dy = marker_path(mk, 5.0)
            plot!(p, x .+ dx, y .+ dy, color=col, lw=2.4, label="")
        end
        p
    end
    plot(panels..., layout=(2, 2), size=(680, 720), legend=false)
end

# ╔═╡ 20000000-0000-0000-0000-00000000001b
Markdown.parse(join(["**$c** → " * (isempty(res[c].keypoints) ? "*none*" :
                     join(["`$typ` ($y, $x)" for (typ, y, x) in res[c].keypoints], ", "))
                     for c in cfg.chars], "\n\n"))

# ╔═╡ 20000000-0000-0000-0000-00000000001c
md"""
### Fig 6 — the operations feeding the classifier

$(@bind op_char Select(collect(String, cfg.chars), default="A"))
"""

# ╔═╡ 20000000-0000-0000-0000-00000000001d
fig6 = let
    r = res[op_char]
    style = (yflip=true, aspect_ratio=:equal, axis=false, colorbar=false, titlefontsize=10)
    p1 = heatmap(r.M;   c=:magma,   title="oriented energy (modulus)", style...)
    p2 = heatmap(r.ESw; c=:viridis, title="end-stopping → endpoints", style...)
    p3 = heatmap(r.cornerness; c=:viridis, title="bimodality → corners/junctions", style...)
    plot(p1, p2, p3, layout=(1, 3), size=(950, 320))
end

# ╔═╡ Cell order:
# ╠═20000000-0000-0000-0000-000000000001
# ╠═20000000-0000-0000-0000-000000000002
# ╠═20000000-0000-0000-0000-000000000003
# ╟─20000000-0000-0000-0000-000000000004
# ╠═20000000-0000-0000-0000-000000000005
# ╟─20000000-0000-0000-0000-000000000006
# ╠═20000000-0000-0000-0000-000000000007
# ╠═20000000-0000-0000-0000-000000000008
# ╠═20000000-0000-0000-0000-000000000009
# ╟─20000000-0000-0000-0000-00000000000a
# ╠═20000000-0000-0000-0000-00000000000b
# ╟─20000000-0000-0000-0000-00000000000c
# ╠═20000000-0000-0000-0000-00000000000d
# ╟─20000000-0000-0000-0000-00000000000e
# ╠═20000000-0000-0000-0000-00000000000f
# ╠═20000000-0000-0000-0000-000000000010
# ╟─20000000-0000-0000-0000-000000000011
# ╠═20000000-0000-0000-0000-000000000012
# ╠═20000000-0000-0000-0000-000000000013
# ╟─20000000-0000-0000-0000-000000000014
# ╠═20000000-0000-0000-0000-000000000015
# ╠═20000000-0000-0000-0000-000000000016
# ╠═20000000-0000-0000-0000-000000000017
# ╟─20000000-0000-0000-0000-000000000018
# ╠═20000000-0000-0000-0000-000000000019
# ╠═20000000-0000-0000-0000-00000000001a
# ╠═20000000-0000-0000-0000-00000000001b
# ╟─20000000-0000-0000-0000-00000000001c
# ╠═20000000-0000-0000-0000-00000000001d
