### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 30000000-0000-0000-0000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 30000000-0000-0000-0000-000000000002
begin
    using PlutoUI
    using Plots
    using Colors
    using ImageFiltering
    using Random
end

# ╔═╡ 30000000-0000-0000-0000-000000000003
begin
    include(joinpath(@__DIR__, "..", "LoadEMNIST.jl"))
    using .LoadEMNIST
end

# ╔═╡ 30000000-0000-0000-0000-000000000004
md"""
# Gabor feature layer — multi-scale (round 2)

`Gabor_Feature_Layer.jl` runs everything at **one** Gabor scale (`λ ≈ 12` on the
upsampled glyph), and its design notes call out the consequence directly:
**corner-vs-curve is scale-relative**. A curve whose radius of curvature is
comparable to the filter size spans two appreciably different tangent
orientations within one receptive field, so the bimodality gate — genuinely
meant to catch corners — cannot tell a tight curve from a polygon. A smooth
`S`, at a single scale, reads as a string of false corners.

**The round-2 fix from the design notes:** run the layer at **several scales**
and require a corner-family point to **survive as bimodal across scales**. A
true corner stays bimodal at the finer scale (the angle doesn't change with
receptive-field size); a curve's rotating peak resolves into a single
orientation once the receptive field shrinks below its radius of curvature,
and it drops out. Junction order (the ring count) and endpoint confirmation
get the same multi-scale vote.

This notebook keeps every operation from `Gabor_Feature_Layer.jl` unchanged —
end-stopping, orientation-profile bimodality, ring spoke count — and adds one
thing: each operation now runs at **3 scales** (`λ = 8, 12, 18`, the same
`λ=12` bank as the single-scale notebook is the middle one) and a keypoint
must be **confirmed at ≥ 2 of the 3 scales** to survive. The ring/spoke count
gets a two-level vote of its own: the single-scale rule (order confirmed at
≥ 2 of 4 radii) runs independently at each scale, then the order itself must
agree across ≥ 2 of the 3 scales — this part **does work**: it kills spurious
T/X detections that a naive flat pool across 12 (scale, radius) samples would
have produced (see Fig 7's caption for the bug this caught).

**Open problem, found while validating this notebook: the corner-vs-curve
half of the round-2 idea does not hold cleanly on real handwriting.** The
premise is that a curve's bimodal-fraction should shrink at fine scale and
vanish while a true corner's stays roughly flat. Measured directly (fraction
of on-ridge pixels gated bimodal, at increasing λ):

    A (real corners): 0.29 0.12 0.11 0.10 0.08 0.06 0.03   (λ=6…22, shrinks steadily)
    K (real corners):  0.20 0.11 0.10 0.10 0.12 0.14 0.16
    T (real corners):  0.22 0.13 0.12 0.11 0.10 0.10 0.11
    X (real junction):  0.29 0.20 0.21 0.19 0.20 0.22 0.29
    S (curve, no corner): 0.27 0.21 0.18 0.18 0.19 0.24 0.33

At the scales usable before filter noise dominates (λ ≳ 6, once the kernel
approaches the stroke width), `S`'s fraction sits in `X`'s range, not `A`'s —
the mechanism doesn't separate them on this sample. Fig 6/7 below show this
directly rather than papering over it; it's left as an open problem (a
peak-*separation-angle* discriminator — constant across scale for a true
corner, growing for a curve — is one candidate for round 3, untested here).

**Second open problem: junction-order confirmation is itself scale-fragile.**
The `need = min(2, nS)` cross-scale vote on the ring count was tuned to kill
a real bug (see Fig 7) — pooling all 12 (scale, radius) samples into one flat
"≥ 2 of N" test let false T/X junctions fire all over `A`/`X`/`S`/`T`. But at
`need=2`, `K`'s two genuine T-junctions (correctly found by the single-scale
notebook) *also* drop to "corner": at this real sample's stroke geometry, the
crossbar/leg meet reads as order-3 at λ=12 but not at λ=8 or λ=18 — the ring
radii scaled to those λ apparently over/under-shoot the branch structure.
Loosening to `need=1` restores `K`'s T's but reopens the exact false-positive
class `need=2` was added to close (verified directly: at `need=1`, `X`/`S`/`A`/`T`
all sprout spurious T's again, plus a false `X` on `K` itself). This isn't a
threshold to retune away — it's evidence that a single fixed set of ring
radii-per-scale doesn't resolve junction structure robustly across scales on
real handwriting. `need=2` ships as the safer default (no spurious multi-way
junctions) at the cost of some missed real ones.
"""

# ╔═╡ 30000000-0000-0000-0000-000000000005
# Three scales spanning the single-scale notebook's λ=12: (fine, mid, coarse).
# sigma_n/sigma_t/ksize are all derived from λ at a fixed ratio (see scale_cfg),
# the same ratios the single-scale notebook used at λ=12 — so the λ=12 bank
# here is byte-for-byte the single-scale notebook's bank.
cfg = (
    lams        = (8.0, 12.0, 18.0),
    sigma_n_ratio = 0.5,        # sigma_n = 0.5λ   (= 6 at λ=12, matching Gabor_Feature_Layer.jl)
    sigma_t_ratio = 11.0/12.0,  # sigma_t = (11/12)λ (= 11 at λ=12)
    n_orient    = 60,
    upsample    = 4,
    noise_sigma = 0.03,
    seed        = 0,
    chars       = ("A", "K", "X", "T", "S"),
)

# ╔═╡ 30000000-0000-0000-0000-000000000006
md"""
## Scale-dependent Gabor bank

`scale_cfg(cfg, λ)` derives a single-scale config (λ, σₙ, σₜ, ksize) from the
shared ratios above; `gabor` is unchanged from the single-scale notebook, just
parameterized by whichever scale's config is passed in. Kernel support grows
with λ (`radius = ceil(5λ/3)`, chosen so λ=12 reproduces the single-scale
notebook's `ksize=41` exactly) so the longer envelope at coarse scales isn't
truncated.
"""

# ╔═╡ 30000000-0000-0000-0000-000000000007
begin
    function scale_cfg(cfg, lam)
        sigma_n = cfg.sigma_n_ratio * lam
        sigma_t = cfg.sigma_t_ratio * lam
        radius = ceil(Int, 5lam/3)
        (; lam, sigma_n, sigma_t, ksize = 2radius + 1)
    end

    # Complex Gabor whose preferred (stripe) orientation is θ (unchanged).
    function gabor(θ, scfg)
        h = scfg.ksize ÷ 2
        g = zeros(ComplexF64, scfg.ksize, scfg.ksize)
        for (ii, y) in enumerate(-h:h), (jj, x) in enumerate(-h:h)
            xt =  x*cos(θ) + y*sin(θ)          # along tangent
            xn = -x*sin(θ) + y*cos(θ)          # along normal
            env = exp(-(xt^2/(2scfg.sigma_t^2) + xn^2/(2scfg.sigma_n^2)))
            car = cis(2π * xn / scfg.lam)      # modulate across the contour
            g[ii, jj] = env * car
        end
        return g .- sum(g)/length(g)           # zero DC → uniform regions give ~0
    end
end

# ╔═╡ 30000000-0000-0000-0000-000000000008
begin
    # Bilinear sample of A at fractional (y, x); 0 outside.
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

# ╔═╡ 30000000-0000-0000-0000-000000000009
function energy_stack(img, thetas, scfg)
    h, w = size(img)
    S = Array{Float64,3}(undef, length(thetas), h, w)
    for (t, θ) in enumerate(thetas)
        S[t, :, :] .= abs.(imfilter(img, centered(gabor(θ, scfg)), Fill(zero(ComplexF64))))
    end
    return S
end

# ╔═╡ 3000000a-0000-0000-0000-00000000000a
md"""
## Operation 1 — end-stopping along the winning orientation (unchanged)

`end_stopping` is exactly the single-scale notebook's function; only the
sampling distance `d` changes with scale (`d = λ`, so it is 8/12/18 at the
three scales — the same numeric coincidence `d ≈ λ` the single-scale notebook
used).
"""

# ╔═╡ 3000000b-0000-0000-0000-00000000000b
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

# ╔═╡ 3000000c-0000-0000-0000-00000000000c
md"""
## Operation 2 — orientation-profile peak structure (unchanged)

`peak_maps` is exactly the single-scale notebook's function, applied
independently at each scale's energy stack. The **multi-scale vote** — a
corner-family point must be bimodal at ≥ 2 of the 3 scales, with the finest
scale required — is applied afterwards in `detect_multiscale`, not here.
"""

# ╔═╡ 3000000d-0000-0000-0000-00000000000d
function peak_maps(S; relthr=0.40, sep_bins=10)
    N, h, w = size(S)
    M = dropdims(maximum(S, dims=1), dims=1)
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
        for s in -sep_bins:sep_bins
            hpk[mod1(i1 + s, N)] = 0.0
        end
        i2 = argmax(hpk); sec = hpk[i2]
        valley = Inf
        for s in 0:mod(i2 - i1, N)
            valley = min(valley, Ss[mod1(i1 + s, N), i, j])
        end
        second_pk[i, j] = sec * (valley < 0.62 * min(max(sec, 1e-9), first_pk[i, j]))
    end
    return npk, first_pk, second_pk
end

# ╔═╡ 3000000e-0000-0000-0000-00000000000e
md"""
## Operation 3 — ring spoke count, pooled across scales

`arc_count` is unchanged. The single-scale notebook pools **4 radii at one
scale** and asks for junction order confirmed at ≥ 2 of those 4 samples.
`classify_point_multiscale` runs that *exact* per-scale rule independently at
each of the 3 scales (ring radii scale with λ, `r = λ · (1.25, 1.5, 1.75,
2.0)`, reproducing `(15, 18, 21, 24)` exactly at λ=12), then takes a **second
vote across scales**: the order must be confirmed at ≥ 2 of the 3 scales.
Flattening all 12 (scale, radius) samples into one pool and reusing the raw
"≥ 2 of N" rule does *not* work — at nS=3 that's only 2-of-12 agreement,
much weaker than the single-scale 2-of-4, and it overfires on corners and
curves alike. The two-level vote keeps the same agreement strength
regardless of scale count.
"""

# ╔═╡ 3000000f-0000-0000-0000-00000000000f
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

# ╔═╡ 30000010-0000-0000-0000-000000000010
# Ms: one energy-modulus field per scale. radii_per_scale[s]: the radii to
# sample at scale s. Junction order (T/X) needs a TWO-LEVEL vote: within each
# scale, the single-scale rule (order confirmed at >=2 of that scale's 4
# radii); then across scales, the order must be confirmed at >= min(2, nS)
# scales. (Flattening all scales' radii into one pool and reusing the
# raw ">=2 of N" rule does NOT work — with nS=3 that is only 2-of-12
# agreement, far weaker than the single-scale 2-of-4, and it overfires on
# corners/curves. Voting scale-by-scale keeps the same agreement strength at
# every scale count, and reduces exactly to classify_point at nS=1.)
function classify_point_multiscale(Ms, y, x, radii_per_scale)
    nS = length(Ms)
    orders = Union{Int,Nothing}[]
    pooled = Tuple{Int,Vector{Float64}}[]
    for s in 1:nS
        res_s = [arc_count(Ms[s], y, x, r) for r in radii_per_scale[s]]
        append!(pooled, res_s)
        valid_s = [n for (n, _) in res_s if 0 < n < 99]
        cnt_s(k) = count(==(k), valid_s)
        confirmed_s = [k for k in (4, 3) if cnt_s(k) >= 2]   # per-scale: order confirmed at >=2 of 4 radii
        push!(orders, isempty(confirmed_s) ? nothing : maximum(confirmed_s))
    end
    need = min(2, nS)
    for k in (4, 3)
        count(==(k), orders) >= need && return k == 4 ? "X" : "T"
    end
    valid = [n for (n, _) in pooled if 0 < n < 99]
    isempty(valid) && return nothing
    collinear = any(n == 2 && length(c) == 2 &&
                    abs(abs(angle(cis(c[1] - c[2]))) - π) < deg2rad(30)
                    for (n, c) in pooled)
    ones = count(==(1), valid)
    ones >= length(valid)/2 && return "endpoint"
    2 in valid && return (collinear ? "straight" : "corner")
    return "endpoint"
end

# ╔═╡ 30000011-0000-0000-0000-000000000011
md"""
## Detection: propose (dense, multi-scale) → refine (ring, multi-scale)

Per scale, exactly the single-scale notebook's gates:

    bimodal_s   = (npk_s >= 2) & (second_s > 0.5·first_s) & (M_s > 0.20·Mmax_s)
    endpoint_s  = (npk_s <= 1) & (M_s > 0.35·Mmax_s)

Then the **round-2 vote**: a pixel is corner-family only if it is bimodal at
the **finest** scale (a true corner's two orientations don't depend on
receptive-field size; a curve's rotating peak is where the fine scale kills
the false bimodality) **and** confirmed at ≥ 2 of the 3 scales overall — the
score itself (for NMS) comes from the finest scale, since it localizes best.
Endpoints get the same cross-scale confirmation. With a single scale
(`lams` of length 1) this is exactly the single-scale notebook's rule.
"""

# ╔═╡ 30000012-0000-0000-0000-000000000012
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

# ╔═╡ 30000013-0000-0000-0000-000000000013
function detect_multiscale(img, cfg; lams=cfg.lams, end_stop_d_ratio=1.0,
                            radii_ratios=(1.25, 1.5, 1.75, 2.0))
    thetas = collect(range(0, π; length=cfg.n_orient + 1))[1:end-1]
    nS = length(lams)

    Ms = Vector{Matrix{Float64}}(undef, nS)
    ESws = Vector{Matrix{Float64}}(undef, nS)
    npks = Vector{Matrix{Int}}(undef, nS)
    firsts = Vector{Matrix{Float64}}(undef, nS)
    seconds = Vector{Matrix{Float64}}(undef, nS)
    Mmaxs = zeros(nS)
    oris = Vector{Matrix{Float64}}(undef, nS)
    radii_per_scale = [lam .* collect(radii_ratios) for lam in lams]

    for (s, lam) in enumerate(lams)
        scfg = scale_cfg(cfg, lam)
        S = energy_stack(img, thetas, scfg)
        M = dropdims(maximum(S, dims=1), dims=1)
        Mmaxs[s] = maximum(M)
        ES = end_stopping(S, thetas; d=end_stop_d_ratio*lam)
        hh, ww = size(M)
        ESw = zeros(hh, ww); ori = zeros(hh, ww)
        for j in 1:ww, i in 1:hh
            k = argmax(view(S, :, i, j))
            ori[i, j] = thetas[k]
            ESw[i, j] = ES[k, i, j]
        end
        npk, first_pk, second_pk = peak_maps(S)
        Ms[s] = M; ESws[s] = ESw; npks[s] = npk; firsts[s] = first_pk; seconds[s] = second_pk
        oris[s] = ori
    end

    bimodal_s  = [(npks[s] .>= 2) .& (seconds[s] .> 0.5 .* firsts[s]) .& (Ms[s] .> 0.20*Mmaxs[s])
                  for s in 1:nS]
    endpoint_s = [(npks[s] .<= 1) .& (Ms[s] .> 0.35*Mmaxs[s]) for s in 1:nS]
    n_bimodal  = reduce(+, bimodal_s)
    n_endpoint = reduce(+, endpoint_s)
    need = min(2, nS)

    fine = 1   # lams sorted ascending: index 1 = finest scale
    cornerness   = seconds[fine] .* bimodal_s[fine]  .* (n_bimodal  .>= need)
    endpointness = ESws[fine]    .* endpoint_s[fine] .* (n_endpoint .>= need)

    proposals = vcat([("corner", y, x) for (y, x) in nms(cornerness, 11, 0.35)],
                     [("endpoint", y, x) for (y, x) in nms(endpointness, 11, 0.38)])
    kept = Tuple{String,Int,Int}[]; taken = Tuple{Int,Int}[]
    for (src, y, x) in proposals
        any((y - yy)^2 + (x - xx)^2 < 11^2 for (yy, xx) in taken) && continue
        ring = classify_point_multiscale(Ms, y, x, radii_per_scale)
        if src == "corner"
            typ = (ring == "T" || ring == "X") ? ring : "corner"
        else
            if ring == "T" || ring == "X"
                typ = ring
            elseif ring == "straight"
                continue
            else
                typ = "endpoint"
            end
        end
        push!(kept, (typ, y, x)); push!(taken, (y, x))
    end

    per_scale = (; lams, Ms, ESws, seconds, bimodal_s, endpoint_s, n_bimodal, n_endpoint)
    return (; img, M=Ms[fine], ori=oris[fine], ESw=ESws[fine], cornerness, endpointness,
             keypoints=kept, per_scale)
end

# ╔═╡ 30000014-0000-0000-0000-000000000014
md"""
## Characters and detection
"""

# ╔═╡ 30000015-0000-0000-0000-000000000015
emnist = load_emnist(n_images_to_load=5000, n_classes=length(cfg.chars),
                     class_order=[11 + Int(c[1]) - Int('A') for c in cfg.chars])

# ╔═╡ 30000016-0000-0000-0000-000000000016
imgs = Dict(c => prep(Float64.(emnist.class_images[i][1]), cfg)
            for (i, c) in enumerate(cfg.chars))

# ╔═╡ 30000017-0000-0000-0000-000000000017
res = Dict(c => detect_multiscale(imgs[c], cfg) for c in cfg.chars)

# ╔═╡ 30000018-0000-0000-0000-000000000018
md"""
### Fig 5 — labelled keypoints, multi-scale

**red square = corner  ·  cyan circle = endpoint  ·  yellow triangle = T  ·
magenta star = X**
"""

# ╔═╡ 30000019-0000-0000-0000-000000000019
begin
    STYLE = Dict("corner"   => (:rect,      "#ff3b30"),
                 "endpoint" => (:circle,    "#00c2d1"),
                 "T"        => (:utriangle, "#ffcc00"),
                 "X"        => (:star5,     "#ff2fd0"))

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

    function keypoint_panel(r, title)
        p = plot(Gray.(r.img), axis=false, title=title, titlefontsize=12)
        for (typ, y, x) in r.keypoints
            mk, col = STYLE[typ]
            dx, dy = marker_path(mk, 5.0)
            plot!(p, x .+ dx, y .+ dy, color=col, lw=2.4, label="")
        end
        p
    end
end

# ╔═╡ 3000001a-0000-0000-0000-00000000001a
fig5 = let
    panels = [keypoint_panel(res[c], c) for c in cfg.chars]
    plot(panels..., layout=(3, 2), size=(680, 1080), legend=false)
end

# ╔═╡ 3000001b-0000-0000-0000-00000000001b
Markdown.parse(join(["**$c** → " * (isempty(res[c].keypoints) ? "*none*" :
                     join(["`$typ` ($y, $x)" for (typ, y, x) in res[c].keypoints], ", "))
                     for c in cfg.chars], "\n\n"))

# ╔═╡ 3000001c-0000-0000-0000-00000000001c
md"""
### Fig 6 — per-scale cornerness on `S` (the mechanism doesn't clean up)

Per-scale `cornerness` (bimodality-gated second-peak, **before** the
cross-scale vote) on `S`, at each of the 3 scales. The hoped-for pattern was
the fine scale visibly thinning out to near-nothing while corner-family
points elsewhere stay solid. What actually shows up: cornerness along the
curve doesn't meaningfully shrink from λ=18 to λ=8 — consistent with the
bimodal-fraction table above. This is the honest result, not the fixed one.
"""

# ╔═╡ 3000001d-0000-0000-0000-00000000001d
fig6 = let
    ps = res["S"].per_scale
    style = (yflip=true, aspect_ratio=:equal, axis=false, colorbar=false, titlefontsize=10)
    panels = [heatmap(ps.seconds[s] .* ps.bimodal_s[s]; c=:viridis,
                       title="λ=$(ps.lams[s]) cornerness", style...) for s in eachindex(ps.lams)]
    plot(panels..., layout=(1, length(panels)), size=(320*length(panels), 340))
end

# ╔═╡ 3000001e-0000-0000-0000-00000000001e
md"""
### Fig 7 — single-scale vs. multi-scale keypoints on `S`

Left: the single-scale notebook's rule (only `λ=12`, `nS=1` so the "≥ 2 of 3"
vote degrades to "the one scale we have" — identical to `Gabor_Feature_Layer.jl`).
Right: the full 3-scale vote. Given the Fig 6 finding, don't expect the
false corners along the curve to disappear here — they largely don't. What
the 3-scale vote *does* fix (caught while building this notebook): pooling
all 12 (scale, radius) ring samples into one flat "≥ 2 of N" test made
spurious T/X junctions fire all over `X`, `K`, and even mid-stroke on `S` —
confirming junction order scale-by-scale first, then voting across scales,
removed that false-positive class. Compare keypoint *types*, not just count.
"""

# ╔═╡ 3000001f-0000-0000-0000-00000000001f
fig7 = let
    single = detect_multiscale(imgs["S"], cfg; lams=[cfg.lams[2]])   # λ=12 only
    multi  = res["S"]
    p1 = keypoint_panel(single, "single-scale (λ=12): $(length(single.keypoints)) keypoints")
    p2 = keypoint_panel(multi,  "multi-scale (3 λ, ≥2 vote): $(length(multi.keypoints)) keypoints")
    plot(p1, p2, layout=(1, 2), size=(680, 380))
end

# ╔═╡ Cell order:
# ╠═30000000-0000-0000-0000-000000000001
# ╠═30000000-0000-0000-0000-000000000002
# ╠═30000000-0000-0000-0000-000000000003
# ╟─30000000-0000-0000-0000-000000000004
# ╠═30000000-0000-0000-0000-000000000005
# ╟─30000000-0000-0000-0000-000000000006
# ╠═30000000-0000-0000-0000-000000000007
# ╠═30000000-0000-0000-0000-000000000008
# ╠═30000000-0000-0000-0000-000000000009
# ╟─3000000a-0000-0000-0000-00000000000a
# ╠═3000000b-0000-0000-0000-00000000000b
# ╟─3000000c-0000-0000-0000-00000000000c
# ╠═3000000d-0000-0000-0000-00000000000d
# ╟─3000000e-0000-0000-0000-00000000000e
# ╠═3000000f-0000-0000-0000-00000000000f
# ╠═30000010-0000-0000-0000-000000000010
# ╟─30000011-0000-0000-0000-000000000011
# ╠═30000012-0000-0000-0000-000000000012
# ╠═30000013-0000-0000-0000-000000000013
# ╟─30000014-0000-0000-0000-000000000014
# ╠═30000015-0000-0000-0000-000000000015
# ╠═30000016-0000-0000-0000-000000000016
# ╠═30000017-0000-0000-0000-000000000017
# ╟─30000018-0000-0000-0000-000000000018
# ╠═30000019-0000-0000-0000-000000000019
# ╠═3000001a-0000-0000-0000-00000000001a
# ╠═3000001b-0000-0000-0000-00000000001b
# ╟─3000001c-0000-0000-0000-00000000001c
# ╠═3000001d-0000-0000-0000-00000000001d
# ╟─3000001e-0000-0000-0000-00000000001e
# ╠═3000001f-0000-0000-0000-00000000001f
