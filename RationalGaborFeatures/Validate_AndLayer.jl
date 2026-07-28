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

# ╔═╡ b1000000-0000-0000-0000-000000000002
begin
    using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
    using Printf, Statistics, Plots
    include(joinpath(@__DIR__, "GaborStack.module.jl"))
    include(joinpath(@__DIR__, "AndLayer.module.jl"))
    include(joinpath(@__DIR__, "Stimuli.module.jl"))
    using .GaborStack, .AndLayer, .Stimuli
    val(x, d) = ismissing(x) ? d : x
    gr()
    md"*setup loaded*"
end

# ╔═╡ b1000000-0000-0000-0000-000000000001
md"""
# The AND layer — conjunction *before* pooling

The claim is narrow and testable. Multiplication and spatial pooling **do not commute**:

```
pool-then-multiply :  (Σₓ w(x)·e₁(x)) · (Σₓ w(x)·e₂(x))
multiply-then-pool :   Σₓ w(x)·e₁(x)·e₂(x)
```

and the difference between them is the within-window covariance `Cov_x(e₁,e₂)`. **That
covariance is the co-location signal.** A statistic computed from already-pooled
orientation energy — our `|E₂|`, `|E₄|`, or a Squeeze-and-Excitation gate — cannot
represent it at any capacity, because the pooling has already averaged it away. This is an
information argument, not an optimisation one: no downstream network can recover it.

So every test below has the same shape: *does this separate stimuli that pooled energy
provably cannot separate?*

The baseline was recorded in Phase 2 and is **not adjustable here**.
"""

# ╔═╡ b1000000-0000-0000-0000-000000000003
begin
    N      = 112
    LADDER = [2.0, 3.742, 7.0]
    BETAS  = [2.0, 1.6, 1.2]
    NORI   = [8, 12, 16]
    H, W, _ = field_for((N, N), LADDER; n_orient=NORI, beta=BETAS)
    bank = make_bank((H, W), LADDER; imwidth=N, n_orient=NORI, beta=BETAS)
    ORI  = [i for (i, m) in enumerate(bank.meta) if m.kind === :oriented]
    E(img) = energy_stack(img, bank)
    pooled(Es) = [sum(@view Es[:, :, i]) for i in ORI]
    cosim(a, b) = sum(a .* b) / (sqrt(sum(abs2, a)) * sqrt(sum(abs2, b)))

    results = Tuple{String,Bool,String}[]
    gate!(name, ok, detail) = push!(results, (name, ok, detail))
    md"""field **$(H)×$(W)** · **$(length(bank))** filters · ladder ρ = $(join(LADDER, ", "))"""
end

# ╔═╡ b1000000-0000-0000-0000-000000000004
md"""
## What A₁ looks like

`A₁(x) = C₀(x) · Σₖ pₖ(x)·pₖ₊ₙ⁄₂(x)` — the orientation profile's autocorrelation at 90°
lag, computed **per pixel**. Zero for a single orientation; maximal where two orthogonal
orientations coexist *at a point*.

Note it fires at line **ends** as well as junctions — an end is i2D too. That is correct
behaviour, and it is why the gate below measures A₁ *at the junction* rather than summed
over the image: `two_bars` has four free ends against a corner's two, which cancels the
junction out of any whole-image total.
"""

# ╔═╡ b1000000-0000-0000-0000-000000000005
let
    ims = [("L-corner", corner(N, π/2; len=40.0)), ("two bars, apart", two_bars(N, 70.0)),
           ("T-junction", tee(N)), ("X-crossing", cross_bars(N))]
    ps = []
    for (nm, im) in ims
        A, _ = and_maps(E(im), bank.meta; forms=(:A1,))
        push!(ps, heatmap(im; c=:grays, yflip=true, title=nm, titlefontsize=8,
                          aspect_ratio=1, axis=nothing, colorbar=false, framestyle=:none))
        push!(ps, heatmap(dropdims(sum(A; dims=3); dims=3); c=:inferno, yflip=true,
                          title="A₁", titlefontsize=8, aspect_ratio=1, axis=nothing,
                          colorbar=false, framestyle=:none))
    end
    plot(ps...; layout=(2, 4), size=(880, 450))
end

# ╔═╡ b1000000-0000-0000-0000-000000000006
md"""
## Gate 1 — A₁ at the junction

Measured in a small window at the centre. Co-location can only be resolved to within the
envelope, so the coarse scale cannot separate a junction from bars 30 px apart *by
construction*; the finest scale is where the claim lives.
"""

# ╔═╡ b1000000-0000-0000-0000-000000000007
begin
    cw = round(Int, (N+1)/2); win = 6
    cen(M, k) = maximum(@view M[cw-win:cw+win, cw-win:cw+win, k])
    Ex0 = E(two_bars(N, 0.0))      # bars cross at the centre → junction present
    Ex1 = E(two_bars(N, 70.0))     # same bars, pulled apart  → no junction
    A0, _  = and_maps(Ex0, bank.meta; forms=(:A1,))
    A1s, _ = and_maps(Ex1, bank.meta; forms=(:A1,))
    rj = [cen(A0, k) / max(cen(A1s, k), eps()) for k in 1:size(A0, 3)]
    gate!("A₁ at the junction, finest scale (> 10)", rj[end] > 10,
          "per-scale ρ=2/3.74/7: " * join([@sprintf("%.1f×", r) for r in rj], "  "))
    md"*gate 1 run*"
end

# ╔═╡ b1000000-0000-0000-0000-000000000008
md"""
## Gate 2 — and pooled energy *cannot* do this

The control that makes the claim non-trivial. Over the same sweep the orientation
histogram is unchanged, so the pooled statistic barely moves — while A₁ at the junction
collapses.
"""

# ╔═╡ b1000000-0000-0000-0000-000000000009
begin
    gaps = [0.0, 20.0, 40.0, 55.0, 70.0]
    a1c = Float64[]; poolc = Float64[]; ref = pooled(Ex0)
    for g in gaps
        Eg = E(two_bars(N, g))
        Ag, _ = and_maps(Eg, bank.meta; forms=(:A1,))
        push!(a1c, cen(Ag, size(Ag, 3))); push!(poolc, cosim(pooled(Eg), ref))
    end
    a1n = a1c ./ a1c[1]
    gate!("A₁ falls with the gap while pooled energy does not",
          a1n[end] < 0.15 && poolc[end] > 0.9,
          @sprintf("A₁ → %.2f ; pooled cos → %.2f", a1n[end], poolc[end]))
    plot(gaps, [a1n poolc]; lw=2.5, marker=:circle, ms=5,
         label=["A₁ at the junction" "pooled orientation energy (cos)"],
         xlabel="separation between the two bars (px)", ylabel="relative to gap = 0",
         title="same orientation content throughout — only co-location changes",
         titlefontsize=9, legend=:left, size=(760, 300), ylims=(0, 1.15))
end

# ╔═╡ b1000000-0000-0000-0000-00000000000a
md"""
## Gate 3 — A₂ end-stopping

`A₂ = E(x) · |E₊ − E₋| / (E₊ + E₋ + κ·E(x))`, sampled along the stroke at the **locally
dominant** orientation.

Two details are load-bearing, both found by measurement rather than argument:

* the `κ·E(x)` term is not a numerical guard. With an absolute `ε`, any channel whose own
  flanks are both near-empty scores `≈1` from noise, so A₂ collapses to plain energy.
* taking the **dominant** orientation rather than a max over all orientations changes
  end/interior from **2.5× to 10.4×** — off-orientation channels otherwise inject spurious
  asymmetry. It is also the faithful model of an end-stopped cell, which inherits the
  local orientation.

`d_factor` was chosen by sweeping: end/interior peaks at 1.0 and falls off either side,
while end/blob rises monotonically with the offset.
"""

# ╔═╡ b1000000-0000-0000-0000-00000000000b
md"""
`d_factor` — flank offset in units of `σ_along` = $(@bind df_ui html"<input type=range min=0.5 max=3.0 step=0.25 value=1.0>")
"""

# ╔═╡ b1000000-0000-0000-0000-00000000000c
begin
    DF = val(df_ui, 1.0)
    Ebar = E(barstim(N, 0.0; w=13.0, len=90.0))
    A2, lab2 = and_maps(Ebar, bank.meta; forms=(:A2,), d_factor=DF)
    c = round(Int, (N+1)/2)
    ratios = Tuple{Float64,Float64}[]
    for k in 1:size(A2, 3)
        M = @view A2[:, :, k]
        e  = maximum(M[c-6:c+6, [c-46:c-36; c+36:c+46]])
        it = maximum(M[c-6:c+6, c-15:c+15])              # the true interior
        fl = maximum(M[[c-26:c-14; c+14:c+26], c-25:c+25])
        push!(ratios, (e / max(it, eps()), e / max(fl, eps())))
    end
    kf = argmax([l.rho0 for l in lab2])
    gate!("A₂ finest scale: end/interior and end/flank > 3",
          ratios[kf][1] > 3 && ratios[kf][2] > 3,
          @sprintf("end/interior %.1f×, end/flank %.1f×  (d = %.1f px)",
                   ratios[kf]..., lab2[kf].d))
    p1 = heatmap(A2[:, :, kf]; c=:inferno, yflip=true, aspect_ratio=1, axis=nothing,
                 colorbar=false, framestyle=:none, titlefontsize=9,
                 title="A₂, finest scale — should light up only at the two ends")
    p2 = plot(1:N, A2[c, :, kf]; lw=2, label="A₂ along the bar's centre line",
              xlabel="x (px)", titlefontsize=9, legend=:top,
              title="profile: peaks at the ends, dips in the middle")
    vline!(p2, [c-45, c+45]; ls=:dash, lc=:grey, label="bar ends")
    plot(p1, p2; layout=(1, 2), size=(880, 290))
end

# ╔═╡ b1000000-0000-0000-0000-00000000000d
Markdown.parse("""
| scale | d (px) | end / interior | end / flank |
|:--|--:|--:|--:|
""" * join(["| ρ = $(round(lab2[k].rho0, digits=2)) | $(round(lab2[k].d, digits=1)) | " *
            "$(round(ratios[k][1], digits=1))× | $(round(ratios[k][2], digits=1))× |"
            for k in 1:length(lab2)], "\n"))

# ╔═╡ b1000000-0000-0000-0000-00000000000e
md"""
## Gate 4 — A₂ must ignore an isolated blob

A blob is a "termination" with **no stroke attached**: both flanks are empty, so a genuine
end-stop stays quiet. This is what separates end-stopping from mere blob saliency.
"""

# ╔═╡ b1000000-0000-0000-0000-00000000000f
begin
    A2b, _ = and_maps(E(blob(N; r=6.5)), bank.meta; forms=(:A2,), d_factor=DF)
    end_fine  = maximum(A2[c-6:c+6, [c-46:c-36; c+36:c+46], kf])
    blob_fine = maximum(@view A2b[:, :, kf])
    gate!("A₂: line-end ≫ isolated blob (> 3)", end_fine / blob_fine > 3,
          @sprintf("end %.3g vs blob %.3g → %.1f×", end_fine, blob_fine,
                   end_fine / blob_fine))
    md"*gate 4 run*"
end

# ╔═╡ b1000000-0000-0000-0000-000000000010
md"""
## Gates 5–6 — invariance and ablatability

Products of energies are still polarity-invariant, so the conjunction must not spoil what
Phase 2 established. And `forms=()` must genuinely switch the layer off — that is how the
ablation in Phase 5 will work.
"""

# ╔═╡ b1000000-0000-0000-0000-000000000011
begin
    Ic = corner(N, π/2)
    Ap, _ = and_maps(E(Ic),  bank.meta; forms=(:A1, :A2, :A3))
    Am, _ = and_maps(E(-Ic), bank.meta; forms=(:A1, :A2, :A3))
    dpol = maximum(abs.(Ap .- Am))
    gate!("polarity invariance preserved (exactly 0)", dpol == 0,
          @sprintf("max|ΔA| = %.3e", dpol))
    n0 = size(and_maps(E(Ic), bank.meta; forms=())[1], 3)
    n1 = size(and_maps(E(Ic), bank.meta; forms=(:A1,))[1], 3)
    gate!("forms are switchable", n0 == 0 && n1 == 3 && size(Ap, 3) == 8,
          "channels: none=$n0, A₁=$n1, A₁+A₂+A₃=$(size(Ap, 3))")
    md"*gates 5–6 run*"
end

# ╔═╡ b1000000-0000-0000-0000-000000000012
md"""
## Junction order — a claim that did NOT survive checking

This section first reported that A₁ "orders junctions by ray count": straight 6.3e4 <
L-corner 9.5e4 < T 1.15e5 < X 1.58e5. **That was an artefact of total ink.** The four
stimuli contain 980 / 1052 / 1359 / 1708 inked pixels, and ΣA₁ tracks that almost exactly.

Normalised by total energy the ordering breaks: **L-corner (0.0415) outranks T-junction
(0.0391)**. And with ink held constant, a T and an X give 1.15e5 against 1.16e5 —
indistinguishable.

**A₁ cannot count rays, and the theory says it must not.** It is built on the orientation
profile, which is **π-periodic**, whereas ray count is a **2π** property: a T (stem +
crossbar) and an X (+ crossing) have the *same* orientation content, {0°, 90°}. What A₁
genuinely separates is one orientation (0.029) from orientations *meeting* (0.039–0.044) —
i2D-ness, not junction order.

This matters beyond bookkeeping. `F` has a 3-ray T where `f` has a 4-ray X, so **A₁ was
structurally the wrong operator for the case that motivated it**, which is consistent with
its moving only 8 of 251 `F`/`f` errors on EMNIST. The operator for ray *count* is `c₀`
from the ray transform in `New_Gabor_FPE/`, which is 2π by construction because its
`d`-offset makes east and west read different pixels.
"""

# ╔═╡ b1000000-0000-0000-0000-000000000013
let
    cases = [("straight (2)", barstim(N, 0.0)), ("L-corner (2, meeting)", corner(N, π/2)),
             ("T-junction (3)", tee(N)), ("X-crossing (4)", cross_bars(N))]
    raw = Float64[]; norm = Float64[]; ink = Int[]
    for (_, im) in cases
        Es = E(im); A, _ = and_maps(Es, bank.meta; forms=(:A1,))
        push!(raw, sum(A)); push!(norm, sum(A)/sum(Es)); push!(ink, count(>(0.5f0), im))
    end
    p1 = Plots.bar(raw; xticks=(1:4, [nm for (nm,_) in cases]), legend=false, ylabel="Σ A₁",
                   title="raw ΣA₁ — but ink is $(join(ink, " / "))", titlefontsize=8,
                   c=:grey, bottom_margin=6Plots.mm)
    p2 = Plots.bar(norm; xticks=(1:4, [nm for (nm,_) in cases]), legend=false,
                   ylabel="Σ A₁ / Σ E", c=:steelblue, bottom_margin=6Plots.mm,
                   title="normalised — the ray-count ordering disappears", titlefontsize=8)
    plot(p1, p2; layout=(1,2), size=(950, 300))
end

# ╔═╡ b1000000-0000-0000-0000-000000000014
begin
    allpass = all(r[2] for r in results)
    # also echo to stdout, so `julia --project=. <this file>` works as a headless gate
    for r in results
        @printf("  [%s] %-46s %s\n", r[2] ? "PASS" : "FAIL", r[1], r[3])
    end
    println(allpass ? "ALL GATES PASSED" : "GATE FAILURE")
    rows = join(["| $(r[2] ? "✅" : "❌") | $(r[1]) | `$(r[3])` |" for r in results], "\n")
    Markdown.parse("""
## Verdict

| | gate | measured |
|:--|:--|:--|
$rows

$(allpass ? "**ALL GATES PASSED** — the AND layer does what it claims." :
            "**GATE FAILURE** — fix before wiring into features.")
""")
end

# ╔═╡ Cell order:
# ╟─b1000000-0000-0000-0000-000000000001
# ╠═b1000000-0000-0000-0000-000000000002
# ╠═b1000000-0000-0000-0000-000000000003
# ╟─b1000000-0000-0000-0000-000000000004
# ╠═b1000000-0000-0000-0000-000000000005
# ╟─b1000000-0000-0000-0000-000000000006
# ╠═b1000000-0000-0000-0000-000000000007
# ╟─b1000000-0000-0000-0000-000000000008
# ╠═b1000000-0000-0000-0000-000000000009
# ╟─b1000000-0000-0000-0000-00000000000a
# ╟─b1000000-0000-0000-0000-00000000000b
# ╠═b1000000-0000-0000-0000-00000000000c
# ╟─b1000000-0000-0000-0000-00000000000d
# ╟─b1000000-0000-0000-0000-00000000000e
# ╠═b1000000-0000-0000-0000-00000000000f
# ╟─b1000000-0000-0000-0000-000000000010
# ╠═b1000000-0000-0000-0000-000000000011
# ╟─b1000000-0000-0000-0000-000000000012
# ╠═b1000000-0000-0000-0000-000000000013
# ╟─b1000000-0000-0000-0000-000000000014
