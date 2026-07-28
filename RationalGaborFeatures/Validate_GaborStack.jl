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

# ╔═╡ a1000000-0000-0000-0000-000000000001
md"""
# Validating the log-Gabor front end

The gate that must pass **before any accuracy number is worth reading**.

Every test uses synthetic stimuli with known ground truth, and every pass criterion is
stated before its number appears. This exists because a front end validated only by
downstream accuracy can be broken in ways a classifier quietly compensates for —
`Dense_Gabors`' keypoint detector turned out to be miscalibrated on *clean* input, and
that was found far too late.

It caught two real bugs that no accuracy number would have surfaced:

* `ρ` was computed in cycles per **field** width while the ladder is in cycles per
  **image** width, mistuning every filter by 2.86×. The tell was that *adding padding
  made results worse*.
* Filters normalised by `sum(G²)` over the discrete grid scale as `1/√(HW)`, so merely
  adding padding shifted every feature by 63 %.

Nothing here imports EMNIST.
"""

# ╔═╡ a1000000-0000-0000-0000-000000000002
begin
    using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
    using Printf, Statistics, Plots, FFTW
    include(joinpath(@__DIR__, "GaborStack.module.jl"))
    include(joinpath(@__DIR__, "Stimuli.module.jl"))
    using .GaborStack, .Stimuli
    "Outside Pluto a bound variable is `missing`; fall back to the default."
    val(x, d) = ismissing(x) ? d : x
    gr()
    md"*setup loaded*"
end

# ╔═╡ a1000000-0000-0000-0000-000000000003
md"""
## The bank

Three scales, not four. The usable band is `ρ ∈ [2, 7]` — only **1.81 octaves** — so four
scales sat 0.6 octaves apart while carrying 2-octave bandwidth, making adjacent channels
nearly the same filter. Three gives 0.90-octave spacing: still deliberately overcomplete,
but no longer paying for channels that duplicate their neighbours.

**`dtheta_on_sigma` is the one worth playing with.** It, *not* the orientation count,
controls spatial elongation: `σ_along = W/(2πρ₀σ_φ)` with `σ_φ = (π/n)/dtheta_on_sigma`,
so `n` cancels out of angular *coverage* and only sets resolution. Kovesi's conventional
1.5 gives `σ_along = 34 px` at `ρ₀ = 2` on a 112-px image — **longer than any EMNIST
stroke**, so no stroke ever looks uniform and end-stopping cannot work. Slide it up and
watch `σ_along` and the field size grow.
"""

# ╔═╡ a1000000-0000-0000-0000-000000000004
md"""
`dtheta_on_sigma` = $(@bind dts_ui html"<input type=range min=0.5 max=2.0 step=0.05 value=0.75>")
"""

# ╔═╡ a1000000-0000-0000-0000-000000000005
begin
    N      = 112
    LADDER = [2.0, 3.742, 7.0]
    LAMS   = [N/r for r in LADDER]
    BETAS  = [2.0, 1.6, 1.2]
    NORI   = [8, 12, 16]
    DTS    = val(dts_ui, 0.75)

    H, W, border = field_for((N, N), LADDER; n_orient=NORI, beta=BETAS, dtheta_on_sigma=DTS)
    bank = make_bank((H, W), LADDER; imwidth=N, n_orient=NORI, beta=BETAS,
                     dtheta_on_sigma=DTS)
    ORI = [(i, m) for (i, m) in enumerate(bank.meta) if m.kind === :oriented]
    idx_of(ρ) = [i for (i, m) in enumerate(bank.meta) if m.kind === :oriented && m.rho0 ≈ ρ]

    # σ_φ is read from the bank rather than recomputed — duplicating it is exactly how the
    # AND layer's probe offsets silently kept stale values when the tuning changed.
    function extents(i)
        m = first(m for m in bank.meta if m.kind === :oriented && m.rho0 ≈ LADDER[i])
        (sigma_x(LAMS[i], BETAS[i]), sigma_along(LADDER[i], m.sigma_phi, N))
    end
    Markdown.parse("""
| | ρ = 2.00 | ρ = 3.74 | ρ = 7.00 |
|:--|--:|--:|--:|
| λ @112 (px) | $(join([@sprintf("%.0f", l) for l in LAMS], " | ")) |
| σ_x, across contour | $(join([@sprintf("%.1f", extents(i)[1]) for i in 1:3], " | ")) |
| **σ_along**, along contour | $(join([@sprintf("%.1f", extents(i)[2]) for i in 1:3], " | ")) |
| orientations | $(join(NORI, " | ")) |

field **$(H)×$(W)**, border **$(border)** · **$(length(bank))** filters
(**$(length(ORI))** oriented + lowpass) · stroke ≈ 12.7 px
""")
end

# ╔═╡ a1000000-0000-0000-0000-000000000006
md"""
### The bank in the frequency plane

Each filter is a one-sided (analytic) bump, so the response is analytic and `|·|²` is the
complex-cell energy. `log 0 = −∞` puts an exact zero at DC — no mean-subtraction hack and
no constraint keeping the bump clear of the origin.
"""

# ╔═╡ a1000000-0000-0000-0000-000000000007
let
    tot = fftshift(sum(bank.filters))
    p1 = heatmap(tot; c=:viridis, title="whole bank (Σ filters)", titlefontsize=9,
                 aspect_ratio=1, axis=nothing, colorbar=false, framestyle=:none)
    ps = [heatmap(fftshift(bank.filters[idx_of(ρ)[1]]); c=:viridis, aspect_ratio=1,
                  title=@sprintf("ρ=%.2f, θ=0", ρ), titlefontsize=8,
                  axis=nothing, colorbar=false, framestyle=:none) for ρ in LADDER]
    plot(p1, ps...; layout=(1, 4), size=(880, 230))
end

# ╔═╡ a1000000-0000-0000-0000-000000000008
md"""
## The stimuli

The negative controls matter as much as the positive ones. A corner and two disjoint
strokes have the **same orientation content** and differ only in whether the orientations
meet, so a detector that cannot separate them is not measuring i2D structure however good
its classification numbers look.
"""

# ╔═╡ a1000000-0000-0000-0000-000000000009
let
    ims = [("bar 30°", barstim(N, π/6)), ("L-corner", corner(N, π/2; len=40.0)),
           ("two bars, gap 70", two_bars(N, 70.0)), ("T-junction", tee(N)),
           ("X-crossing", cross_bars(N)), ("blob", blob(N; r=6.5))]
    plot([heatmap(im; c=:grays, yflip=true, title=nm, titlefontsize=8, aspect_ratio=1,
                  axis=nothing, colorbar=false, framestyle=:none) for (nm, im) in ims]...;
         layout=(1, 6), size=(950, 175))
end

# ╔═╡ a1000000-0000-0000-0000-00000000000a
md"""
## Gates 1–2 — polarity invariance and DC

Polarity invariance is a **requirement, not an optimisation**, and it must be *exact*:
`|even|² + |odd|²` is algebraically invariant under negation, so anything other than zero
means the quadrature pair is not exact.
"""

# ╔═╡ a1000000-0000-0000-0000-00000000000b
begin
    results = Tuple{String,Bool,String}[]
    gate!(name, ok, detail) = push!(results, (name, ok, detail))

    let I = barstim(N, π/6)
        d = maximum(abs.(energy_stack(I, bank) .- energy_stack(-I, bank)))
        gate!("polarity invariance — exactly 0", d == 0, @sprintf("max|ΔE| = %.3e", d))
    end
    let Ec = energy_stack(fill(0.5f0, N, N), bank; mode=:replicate)
        gate!("constant image → zero energy (< 1e-10)", maximum(Ec) < 1e-10,
              @sprintf("max E = %.3e", maximum(Ec)))
    end
    md"*gates 1–2 run*"
end

# ╔═╡ a1000000-0000-0000-0000-00000000000c
md"""
## Gate 3 — orientation readout

A bar at `θ` has its wavevector at `θ+90°`, so the winning carrier must be `θ+90` (mod π),
within half the orientation spacing.
"""

# ╔═╡ a1000000-0000-0000-0000-00000000000d
begin
    finest = [(i, m) for (i, m) in ORI if m.rho0 ≈ LADDER[3]]
    spacing = π / NORI[3]
    oerrs = Float64[]; tuning = Vector{Float64}[]
    for θdeg in 0:15:165
        θ = deg2rad(θdeg)
        Eo = energy_stack(barstim(N, θ), bank)
        tot = [sum(@view Eo[:, :, i]) for (i, _) in finest]
        push!(tuning, tot ./ maximum(tot))
        got = finest[argmax(tot)][2].theta
        want = mod(θ + π/2, π)
        push!(oerrs, rad2deg(abs(atan(sin(got - want), cos(got - want)))))
    end
    gate!("orientation readout (≤ ½ spacing = $(round(rad2deg(spacing)/2, digits=1))°)",
          maximum(oerrs) <= rad2deg(spacing)/2 + 1e-9,
          @sprintf("max error %.1f° over 12 angles", maximum(oerrs)))
    plot(rad2deg.([m.theta for (_, m) in finest]), reduce(hcat, tuning);
         xlabel="filter carrier angle (°)", ylabel="normalised response", legend=false,
         title="orientation tuning — one curve per stimulus angle", titlefontsize=9,
         lw=1.5, marker=:circle, ms=2, size=(760, 260))
end

# ╔═╡ a1000000-0000-0000-0000-00000000000e
md"""
## Gate 4 — padding invariance

Wraparound is an **artefact**, so the test is an invariance: doubling the border must not
change the answer. Measuring "energy far from a blob" instead measures the filters'
genuine spatial tails, which have nothing to do with the FFT.

The residual is frequency-grid discretisation, confirmed by two signatures: it
concentrates in the coarsest, worst-sampled channel (ρ=2 gives 3.1e-4 against ρ=7's
6.6e-7) and it shrinks as the grid refines (320↔525: 3.1e-4; 525↔729: 5.1e-6). The lowpass
at ρ₀=1 is coarser still, hence reported separately.
"""

# ╔═╡ a1000000-0000-0000-0000-00000000000f
begin
    Iedge = zeros(Float32, N, N); Iedge[50:62, 3:15] .= 1f0
    H2, W2, _ = field_for((N, N), LADDER; n_orient=NORI, beta=BETAS,
                          dtheta_on_sigma=DTS, k=6)
    bank2 = make_bank((H2, W2), LADDER; imwidth=N, n_orient=NORI, beta=BETAS,
                      dtheta_on_sigma=DTS)
    Ea = energy_stack(Iedge, bank); Eb2 = energy_stack(Iedge, bank2)
    relof(sel) = maximum(abs.(@view(Ea[:, :, sel]) .- @view(Eb2[:, :, sel]))) /
                 maximum(@view Ea[:, :, sel])
    rel_ori = maximum(relof(idx_of(ρ)) for ρ in LADDER)
    rel_lp  = relof([i for (i, m) in enumerate(bank.meta) if m.kind === :lowpass])
    gate!("padding invariance, oriented channels (< 1e-3)", rel_ori < 1e-3,
          @sprintf("field %d vs %d → %.2e   (lowpass ρ₀=1: %.2e)", H, H2, rel_ori, rel_lp))
    md"*gate 4 run*"
end

# ╔═╡ a1000000-0000-0000-0000-000000000010
md"""
## Gate 5 — scale tuning

The criterion is each filter's **own tuning curve**, not argmax across filters: the bank is
deliberately overcomplete, so "which scale wins" is not well posed. What must hold is that
each filter peaks at the wavelength it is tuned to.

The stimulus is a *Gabor patch*, not a bare grating — a square-windowed grating's edges
dump broadband energy at low frequency, which hands every trial to the coarsest filter
regardless of λ. Measured: bare gratings put λ=16 at the ρ=2 channel.
"""

# ╔═╡ a1000000-0000-0000-0000-000000000011
begin
    SWEEP = [72.0, 56.0, 45.0, 37.0, 30.0, 24.0, 20.0, 16.0, 13.0]
    resp = zeros(length(LADDER), length(SWEEP))
    for (li, λ) in enumerate(SWEEP)
        Eg = energy_stack(gabor_patch(N, λ, 0.0), bank)
        for (si, ρ) in enumerate(LADDER)
            resp[si, li] = sum(sum(@view Eg[:, :, i]) for i in idx_of(ρ))
        end
    end
    peakλ = [SWEEP[argmax(@view resp[si, :])] for si in 1:length(LADDER)]
    gate!("each filter peaks at its own λ (within ⅓ octave)",
          all(abs(log2(peakλ[i] / LAMS[i])) <= 0.35 for i in 1:length(LADDER)),
          "peak λ = " * join([@sprintf("%.0f", p) for p in peakλ], ", ") *
          " vs tuned " * join([@sprintf("%.0f", l) for l in LAMS], ", "))
    plot(SWEEP, (resp ./ maximum(resp; dims=2))'; xscale=:log10, xflip=true, lw=2,
         marker=:circle, xlabel="stimulus λ (px)", ylabel="normalised response",
         label=reshape(["ρ=$(round(r, digits=2))  (λ=$(round(Int, l)))"
                        for (r, l) in zip(LADDER, LAMS)], 1, :),
         title="scale tuning — each curve should peak at its own λ", titlefontsize=9,
         legend=:topleft, size=(760, 280))
end

# ╔═╡ a1000000-0000-0000-0000-000000000012
md"""
## Gate 6 — localisation

Judged **per scale**, against each scale's own extent. A single fixed mask fails trivially
because the two extents differ per scale and neither is small.
"""

# ╔═╡ a1000000-0000-0000-0000-000000000013
begin
    Ebar = energy_stack(barstim(N, 0.0; w=13.0, len=70.0), bank)
    fracs = Float64[]
    for (si, ρ) in enumerate(LADDER)
        sx, sa = extents(si)
        tot = dropdims(sum(@view Ebar[:, :, idx_of(ρ)]; dims=3); dims=3)
        mask = barstim(N, 0.0; w=13 + 4sx, len=70 + 4sa) .> 0
        push!(fracs, sum(tot .* mask) / sum(tot))
    end
    gate!("energy localised on the bar, per scale (> 0.75)", minimum(fracs) > 0.75,
          "fracs = " * join([@sprintf("%.3f", f) for f in fracs], ", "))
    plot([heatmap(dropdims(sum(@view Ebar[:, :, idx_of(ρ)]; dims=3); dims=3);
                  c=:inferno, yflip=true, aspect_ratio=1, axis=nothing, colorbar=false,
                  framestyle=:none, titlefontsize=8,
                  title=@sprintf("ρ=%.2f — %.3f on the bar", ρ, fracs[si]))
          for (si, ρ) in enumerate(LADDER)]...; layout=(1, 3), size=(760, 260))
end

# ╔═╡ a1000000-0000-0000-0000-000000000014
md"""
## The Phase-3 baseline

Pooled orientation energy **cannot** separate a corner from two disjoint strokes. That is
not a bug — it is the measurement that motivates the AND layer, recorded here so the
layer's benefit is judged against a number rather than an impression.
"""

# ╔═╡ a1000000-0000-0000-0000-000000000015
begin
    pooled(Es) = [sum(@view Es[:, :, i]) for (i, _) in ORI]
    cosim(a, b) = sum(a .* b) / (sqrt(sum(abs2, a)) * sqrt(sum(abs2, b)))
    baseline = cosim(pooled(energy_stack(corner(N, π/2; len=40.0), bank)),
                     pooled(energy_stack(two_bars(N, 70.0), bank)))
    Markdown.parse("""
!!! warning "Baseline for the AND layer"
    Corner vs two disjoint strokes, under **pooled** orientation energy:
    **cos = $(round(baseline, digits=4))**.

    `Validate_AndLayer.jl` must lower this materially.
""")
end

# ╔═╡ a1000000-0000-0000-0000-000000000016
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

$(allpass ? "**ALL GATES PASSED** — cleared for Phase 3." :
            "**GATE FAILURE** — fix before proceeding.")
""")
end

# ╔═╡ Cell order:
# ╟─a1000000-0000-0000-0000-000000000001
# ╠═a1000000-0000-0000-0000-000000000002
# ╟─a1000000-0000-0000-0000-000000000003
# ╟─a1000000-0000-0000-0000-000000000004
# ╠═a1000000-0000-0000-0000-000000000005
# ╟─a1000000-0000-0000-0000-000000000006
# ╠═a1000000-0000-0000-0000-000000000007
# ╟─a1000000-0000-0000-0000-000000000008
# ╠═a1000000-0000-0000-0000-000000000009
# ╟─a1000000-0000-0000-0000-00000000000a
# ╠═a1000000-0000-0000-0000-00000000000b
# ╟─a1000000-0000-0000-0000-00000000000c
# ╠═a1000000-0000-0000-0000-00000000000d
# ╟─a1000000-0000-0000-0000-00000000000e
# ╠═a1000000-0000-0000-0000-00000000000f
# ╟─a1000000-0000-0000-0000-000000000010
# ╠═a1000000-0000-0000-0000-000000000011
# ╟─a1000000-0000-0000-0000-000000000012
# ╠═a1000000-0000-0000-0000-000000000013
# ╟─a1000000-0000-0000-0000-000000000014
# ╠═a1000000-0000-0000-0000-000000000015
# ╟─a1000000-0000-0000-0000-000000000016
