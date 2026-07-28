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

# ╔═╡ c1000000-0000-0000-0000-000000000001
md"""
# Pooling and feature assembly

**Pooling comes last, and that is the whole point.** An i2D conjunction is spatially sharp
and therefore fragile — a junction moves several pixels between handwritten instances, so a
pixel-exact detector would produce a different vector every time. Pooling is what buys the
positional tolerance.

```
select (oriented filter) → multiply (conjunction) → pool (tolerance)
```

which is simple cell → complex cell, and conv → ReLU → pool. **Detect, then tolerate.** The
old Fourier grid did it backwards — pool first via the Gaussian window, then combine
orientations into `|E₄|` — and no downstream capacity can undo an averaging that has
already happened.

So the gate here is the property pooling is *for*: translation tolerance, measured rather
than assumed.
"""

# ╔═╡ c1000000-0000-0000-0000-000000000002
begin
    using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
    using Printf, Statistics, Random, Plots
    include(joinpath(@__DIR__, "GaborStack.module.jl"))
    include(joinpath(@__DIR__, "AndLayer.module.jl"))
    include(joinpath(@__DIR__, "Stimuli.module.jl"))
    include(joinpath(@__DIR__, "Pooling.module.jl"))
    using .GaborStack, .AndLayer, .Stimuli, .Pooling
    val(x, d) = ismissing(x) ? d : x
    gr()
    md"*setup loaded*"
end

# ╔═╡ c1000000-0000-0000-0000-000000000003
begin
    N      = 112
    LADDER = [2.0, 3.742, 7.0]
    BETAS  = [2.0, 1.6, 1.2]
    NORI   = [8, 12, 16]
    Hf, Wf, _ = field_for((N, N), LADDER; n_orient=NORI, beta=BETAS)
    bank = make_bank((Hf, Wf), LADDER; imwidth=N, n_orient=NORI, beta=BETAS)
    σ_along_fine = let m = first(m for m in bank.meta
                                 if m.kind === :oriented && m.rho0 ≈ LADDER[end])
        m.imwidth / (2π * m.rho0 * m.sigma_phi)
    end
    BLOCKS = (:orient, :lowpass, :A1, :A2)

    "Full pipeline: image → dense energy → pointwise conjunction → pooled vector."
    function features(img, grid; blocks=BLOCKS, Wts=nothing)
        Es = energy_stack(img, bank)
        A, al = and_maps(Es, bank.meta; forms=filter(b -> b in (:A1, :A2, :A3), blocks))
        assemble(Es, bank.meta, A, al, PoolSpec(grid=grid, blocks=blocks); Wts=Wts)
    end
    results = Tuple{String,Bool,String}[]
    gate!(n, ok, d) = push!(results, (n, ok, d))
    md"""bank: **$(length(bank))** filters on a $(Hf)×$(Wf) field ·
    finest `σ_along` = **$(round(σ_along_fine, digits=1)) px**"""
end

# ╔═╡ c1000000-0000-0000-0000-000000000004
# Markdown.parse rather than md"..." on purpose: Julia interpolates the string first, so
# the markdown parser never sees identifiers like `nyquist_grid` or `σ_along_fine`. Inside
# md"...", their underscores are read as emphasis markers and run straight into the
# $(...) interpolation, giving a runtime ParseError that points at "none:1" and names no
# file. Two notebooks hit this before it was understood.
Markdown.parse("""
## The grid

Windows are Gaussian and **overlapping**, not tiling: a stroke landing on a hard cell
boundary would otherwise be assigned arbitrarily to one side. Columns are L1-normalised so
a pooled value is a weighted *mean*, which keeps cells comparable when one sits partly off
the image.

The demodulation argument sets the *finest useful* grid: after `|·|²` the envelope is
baseband with bandwidth set by σ, so samples closer than about σ are redundant. At
`σ_along` = $(round(σ_along_fine, digits=1)) px on a 112-px image that is
**$(nyquist_grid(112, σ_along_fine))×$(nyquist_grid(112, σ_along_fine))**. Using 3×3 is
therefore a *compactness* choice, not a resolution limit — which is exactly the trade the
tolerance measurement below prices.
""")

# ╔═╡ c1000000-0000-0000-0000-000000000005
md"""grid = $(@bind grid_ui html"<input type=range min=2 max=8 step=1 value=3>")"""

# ╔═╡ c1000000-0000-0000-0000-000000000006
let
    g = val(grid_ui, 3)
    Wt = grid_weights(N, N, g)
    p1 = heatmap(reshape(sum(Wt; dims=2), N, N); c=:viridis, yflip=true, aspect_ratio=1,
                 axis=nothing, colorbar=false, framestyle=:none, titlefontsize=9,
                 title="Σ of all $(g)² windows — coverage")
    p2 = heatmap(reshape(Wt[:, 1], N, N); c=:viridis, yflip=true, aspect_ratio=1,
                 axis=nothing, colorbar=false, framestyle=:none, titlefontsize=9,
                 title="one cell's window")
    p3 = plot(1:N, [reshape(Wt[:, c], N, N)[round(Int, (N+1)/2), :] for c in 1:g];
              lw=2, legend=false, xlabel="x (px)", titlefontsize=9,
              title="windows across the middle row — they overlap")
    plot(p1, p2, p3; layout=(1, 3), size=(920, 250))
end

# ╔═╡ c1000000-0000-0000-0000-000000000007
md"""
## What a feature vector looks like

Four blocks, independently switchable. `:orient` is deliberately the **deficient** baseline
— it pools first and combines after, so it provably cannot represent co-location. The
A-blocks are the same information computed in the other order. That contrast is the
experiment, so it is structural in the code rather than a comment.
"""

# ╔═╡ c1000000-0000-0000-0000-000000000008
let
    f, l = features(tee(N), 3)
    blk(name) = findall(x -> startswith(x, name), l)
    cols = [(:orient, :steelblue), (:lowpass, :grey), (:A1, :firebrick), (:A2, :seagreen)]
    p = plot(xlabel="feature index", ylabel="value", size=(920, 270), legend=:topright,
             title="feature vector for a T-junction — $(length(f)) numbers",
             titlefontsize=9)
    for (nm, c) in cols
        idx = blk(String(nm)); isempty(idx) && continue
        scatter!(p, idx, f[idx]; ms=2.5, msw=0, c=c,
                 label="$(nm) ($(length(idx)))")
    end
    p
end

# ╔═╡ c1000000-0000-0000-0000-000000000009
md"""
## Gate 1 — translation tolerance, and what the grid costs

Shift the stimulus and measure how much the feature vector moves. A finer grid keeps more
spatial detail and is correspondingly *less* tolerant; that trade is the whole reason the
grid is a parameter.

The criterion is deliberately modest — a 4 px shift, roughly a third of a stroke width,
must leave the 3×3 vector above 0.99 cosine similarity — because a front end that cannot
survive that is of no use on handwriting.
"""

# ╔═╡ c1000000-0000-0000-0000-00000000000a
begin
    cosim(a, b) = sum(a .* b) / (sqrt(sum(abs2, a)) * sqrt(sum(abs2, b)))
    shifts = 0:2:10
    grids = [3, 6, 11]
    base = tee(N)
    tol = Dict(g => Float64[] for g in grids)
    for g in grids
        Wt = grid_weights(N, N, g)
        f0 = features(base, g; Wts=Wt)[1]
        for s in shifts
            fs = features(circshift(base, (0, s)), g; Wts=Wt)[1]
            push!(tol[g], cosim(f0, fs))
        end
    end
    i4 = findfirst(==(4), collect(shifts))
    gate!("translation tolerance: 3×3 holds cos > 0.99 at 4 px",
          tol[3][i4] > 0.99,
          @sprintf("3×3 %.4f | 6×6 %.4f | 11×11 %.4f at a 4 px shift",
                   tol[3][i4], tol[6][i4], tol[11][i4]))
    plot(collect(shifts), [tol[g] for g in grids]; lw=2.5, marker=:circle, ms=5,
         label=reshape(["$(g)×$(g) grid" for g in grids], 1, :),
         xlabel="translation (px)", ylabel="cosine similarity to unshifted",
         title="finer grids keep more detail and tolerate less — the trade pooling prices",
         titlefontsize=9, legend=:bottomleft, size=(780, 300))
end

# ╔═╡ c1000000-0000-0000-0000-00000000000b
md"""
## Gate 2 — the invariances survive assembly

Polarity invariance was exact through the bank and exact through the conjunction. Pooling
is a non-negative weighted sum, so it must stay exact — anything else means a bug crept in
between.
"""

# ╔═╡ c1000000-0000-0000-0000-00000000000c
begin
    Ic = corner(N, π/2)
    fp = features(Ic, 3)[1]; fm = features(-Ic, 3)[1]
    dpol = maximum(abs.(fp .- fm))
    gate!("polarity invariance through the full pipeline (exactly 0)", dpol == 0,
          @sprintf("max|Δf| = %.3e over %d features", dpol, length(fp)))
    md"*gate 2 run*"
end

# ╔═╡ c1000000-0000-0000-0000-00000000000d
md"""
## Gate 3 — blocks really are ablatable

Phase 5 turns blocks off to attribute credit, so `()` must genuinely yield nothing and each
block must contribute exactly its own columns.
"""

# ╔═╡ c1000000-0000-0000-0000-00000000000e
begin
    counts = Dict{Any,Int}()
    for b in [(), (:orient,), (:lowpass,), (:A1,), (:A2,), (:orient, :lowpass),
              (:orient, :lowpass, :A1, :A2)]
        counts[b] = length(features(tee(N), 3; blocks=b)[1])
    end
    additive = counts[(:orient, :lowpass, :A1, :A2)] ==
               counts[(:orient,)] + counts[(:lowpass,)] + counts[(:A1,)] + counts[(:A2,)]
    gate!("blocks ablate and compose additively", counts[()] == 0 && additive,
          "orient $(counts[(:orient,)]) + lowpass $(counts[(:lowpass,)]) + " *
          "A1 $(counts[(:A1,)]) + A2 $(counts[(:A2,)]) = " *
          "$(counts[(:orient,:lowpass,:A1,:A2)]); none = $(counts[()])")
    md"*gate 3 run*"
end

# ╔═╡ c1000000-0000-0000-0000-00000000000f
md"""
## Gate 4 — the dimensionality control

`shuffle_block!` permutes a block's rows **across samples**: the marginal distribution and
the column count are untouched, but each sample's correspondence to its own values is
destroyed.

This is not bookkeeping. §7.8 established that *a fixed projection into a few hundred
dimensions plus a trained head is a strong baseline whatever the projection is* — scrambled
features still reached 75 %. So "adding A₁ helped" and "adding 27 more columns helped" are
not distinguishable without this control, and the gain we are hunting is exactly the size
that dimensionality alone can buy.
"""

# ╔═╡ c1000000-0000-0000-0000-000000000010
begin
    f0, lab = features(tee(N), 3)
    F = randn(Float32, 200, length(f0))
    a1cols = findall(x -> startswith(x, "A1"), lab)
    G = copy(F); shuffle_block!(G, a1cols; seed=1)
    others = setdiff(1:size(F, 2), a1cols)
    ok = G[:, a1cols] != F[:, a1cols] &&
         G[:, others] == F[:, others] &&
         maximum(abs.(sort(vec(G[:, a1cols])) .- sort(vec(F[:, a1cols])))) < 1f-6
    gate!("shuffle_block! destroys correspondence, preserves marginals", ok,
          "$(length(a1cols)) A₁ columns permuted; $(length(others)) others untouched; " *
          "value multiset identical")
    md"*gate 4 run*"
end

# ╔═╡ c1000000-0000-0000-0000-000000000011
md"""
## Gate 5 — no degenerate features

A column that is constant across stimuli carries nothing and will be divided by a zero
standard deviation downstream. Checked over a spread of synthetic figures.
"""

# ╔═╡ c1000000-0000-0000-0000-000000000012
begin
    stim = [barstim(N, 0.0), barstim(N, π/4), corner(N, π/2), tee(N),
            cross_bars(N), blob(N; r=8.0), two_bars(N, 60.0)]
    FM = reduce(vcat, [features(s, 3)[1]' for s in stim])
    sds = vec(std(FM; dims=1))
    nconst = count(<(1f-12), sds)
    gate!("no constant or non-finite features", all(isfinite, FM) && nconst == 0,
          @sprintf("%d features, %d constant across %d stimuli, all finite: %s",
                   size(FM, 2), nconst, length(stim), all(isfinite, FM)))
    md"*gate 5 run*"
end

# ╔═╡ c1000000-0000-0000-0000-000000000013
begin
    allpass = all(r[2] for r in results)
    for r in results
        @printf("  [%s] %-52s %s\n", r[2] ? "PASS" : "FAIL", r[1], r[3])
    end
    println(allpass ? "ALL GATES PASSED" : "GATE FAILURE")
    rows = join(["| $(r[2] ? "✅" : "❌") | $(r[1]) | `$(r[3])` |" for r in results], "\n")
    Markdown.parse("""
## Verdict

| | gate | measured |
|:--|:--|:--|
$rows

$(allpass ? "**ALL GATES PASSED** — cleared for Phase 5 (EMNIST replication, then ablation)." :
            "**GATE FAILURE** — fix before wiring into the classifier.")
""")
end

# ╔═╡ Cell order:
# ╟─c1000000-0000-0000-0000-000000000001
# ╠═c1000000-0000-0000-0000-000000000002
# ╠═c1000000-0000-0000-0000-000000000003
# ╠═c1000000-0000-0000-0000-000000000004
# ╟─c1000000-0000-0000-0000-000000000005
# ╠═c1000000-0000-0000-0000-000000000006
# ╟─c1000000-0000-0000-0000-000000000007
# ╠═c1000000-0000-0000-0000-000000000008
# ╟─c1000000-0000-0000-0000-000000000009
# ╠═c1000000-0000-0000-0000-00000000000a
# ╟─c1000000-0000-0000-0000-00000000000b
# ╠═c1000000-0000-0000-0000-00000000000c
# ╟─c1000000-0000-0000-0000-00000000000d
# ╠═c1000000-0000-0000-0000-00000000000e
# ╟─c1000000-0000-0000-0000-00000000000f
# ╠═c1000000-0000-0000-0000-000000000010
# ╟─c1000000-0000-0000-0000-000000000011
# ╠═c1000000-0000-0000-0000-000000000012
# ╟─c1000000-0000-0000-0000-000000000013
