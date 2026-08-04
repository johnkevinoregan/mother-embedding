### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ f1000000-0000-0000-0000-000000000001
md"""
# Is the orientation convention consistent everywhere?

A Gabor has two angles that differ by 90°: the **carrier** (wavevector) direction, and the
**ridge** direction along which the stripes run. Earlier work outside this project
(unconventionally) took the angle to mean the *ridge* direction. This bank stores the
**carrier**. A mix-up between the two would be an easy thing to leak into the operators
and hard to see, so it is tested here rather than argued about.

Where such a confusion could and could not do damage:

* **A₁ is immune.** It uses a 90° *shift* in the orientation index, and a shift is the same
  quantity in either convention.
* **A₂ and the ray transform would fail loudly, not quietly.** Flipped, A₂ would probe
  *across* the stroke — both samples off the bar, so ~0 everywhere — and the ray transform
  would read channels orthogonal to each ray, so `c₀` would be near zero for a straight
  line. Neither could pass its own gates.
* **The `orient` block is a consistent relabelling.** `E₂` and `E₄` are computed in carrier
  coordinates, so every orientation reads 90° from the ridge direction — the same
  convention the project's older Fourier work documented. A global relabelling of the same
  measurements is invisible to a classifier.

So the only real exposure is a claim about *absolute* orientation values. These three tests
pin it.
"""

# ╔═╡ f1000000-0000-0000-0000-000000000002
begin
    using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
    using Printf, Statistics
    include(joinpath(@__DIR__, "GaborStack.module.jl"))
    include(joinpath(@__DIR__, "AndLayer.module.jl"))
    include(joinpath(@__DIR__, "Stimuli.module.jl"))
    using .GaborStack, .AndLayer, .Stimuli
    N = 160
    Hf, Wf, _ = field_for((N,N), [7.0]; n_orient=16, beta=1.2)
    bank = make_bank((Hf,Wf), [7.0]; imwidth=N, n_orient=16, beta=1.2, lowpass=false)
    CH = [(i, m.theta) for (i,m) in enumerate(bank.meta)]
    CC = round(Int, (N+1)/2)
    results = Tuple{String,Bool,String}[]
    gate!(n,ok,d) = push!(results,(n,ok,d))
    md"*single-scale bank, 16 orientations*"
end

# ╔═╡ f1000000-0000-0000-0000-000000000003
md"""
## Test 1 — what does the stored `theta` mean?

A bar drawn at β has its stripes along β, hence its wavevector at β+90°. If `theta` is the
carrier the winner is **β+90°**; if it were the ridge direction the winner would be **β**.
"""

# ╔═╡ f1000000-0000-0000-0000-000000000004
begin
    errs1 = Float64[]
    @printf("%10s %14s %14s\n", "bar β", "winning theta", "β+90 (mod π)")
    for βd in (0.0, 30.0, 60.0, 105.0, 150.0)
        β = deg2rad(βd)
        E = energy_stack(barstim(N, β; w=13.0, len=90.0), bank)
        tot = [sum(@view E[:,:,i]) for (i,_) in CH]
        got = CH[argmax(tot)][2]; want = mod(β + π/2, π)
        push!(errs1, rad2deg(abs(atan(sin(got-want), cos(got-want)))))
        @printf("%9.0f° %13.1f° %13.1f°\n", βd, rad2deg(got), rad2deg(want))
    end
    gate!("`theta` is the CARRIER: winner = β+90° (≤ ½ spacing)",
          maximum(errs1) <= 90/16 + 1e-6,
          @sprintf("max error %.1f° over 5 angles", maximum(errs1)))
    md"*test 1 run*"
end

# ╔═╡ f1000000-0000-0000-0000-000000000005
md"""
## Test 2 — which axis does A₂ probe?

If it probes *along* the stroke it fires at the two ends. If the axis were flipped it would
probe across, both samples would sit off the bar, and it would fire nowhere.
"""

# ╔═╡ f1000000-0000-0000-0000-000000000006
begin
    β2 = deg2rad(30.0)
    A2, _ = and_maps(energy_stack(barstim(N, β2; w=13.0, len=100.0), bank),
                     bank.meta; forms=(:A2,))
    M2 = @view A2[:,:,1]
    alongv = [M2[clamp(round(Int, CC + t*sin(β2)),1,N), clamp(round(Int, CC + t*cos(β2)),1,N)]
              for t in -55:5:55]
    ends2 = maximum(alongv[[1,2,end-1,end]]); mid2 = maximum(alongv[10:13])
    gate!("A₂ probes ALONG the stroke (ends ≫ middle)", ends2 > 3*mid2,
          @sprintf("ends %.3g vs middle %.3g → %.1f×", ends2, mid2, ends2/mid2))
    md"*test 2 run*"
end

# ╔═╡ f1000000-0000-0000-0000-000000000007
md"""
## Test 3 — the decisive one

For a **single ray** at direction φ₀ the theory gives `c₁ = e^{−iφ₀}`, so `−arg(c₁)` must
recover φ₀ over the full 2π range. `c₁` is recomputed explicitly here rather than taken
from `ray_maps`, so the test does not depend on that function's own bookkeeping.
"""

# ╔═╡ f1000000-0000-0000-0000-000000000008
begin
    errs3 = Float64[]
    m1 = first(bank.meta); dR = m1.imwidth / (2π * m1.rho0 * m1.sigma_phi)
    @printf("%10s %14s %10s\n", "ray φ₀", "−arg(c₁)", "error")
    for φd in (0.0, 45.0, 120.0, 200.0, 300.0)
        φ = deg2rad(φd)
        E = energy_stack(rays(N, [φ]; w=13.0, r=45.0), bank)
        C1 = ComplexF64(0)
        for j in 1:32
            ψ = 2π*(j-1)/32; want = mod(ψ + π/2, π)
            k = argmin([abs(atan(sin(θ-want), cos(θ-want))) for (_,θ) in CH])
            y = CC + dR*sin(ψ); x = CC + dR*cos(ψ)
            C1 += E[clamp(round(Int,y),1,N), clamp(round(Int,x),1,N), CH[k][1]] * cis(-ψ)
        end
        got = mod(-angle(C1), 2π)
        push!(errs3, rad2deg(abs(atan(sin(got-φ), cos(got-φ)))))
        @printf("%9.0f° %13.1f° %9.1f°\n", φd, rad2deg(got), errs3[end])
    end
    gate!("−arg(c₁) recovers the ray direction over 2π (< 5°)", maximum(errs3) < 5,
          @sprintf("max error %.1f° over 5 directions", maximum(errs3)))
    md"*test 3 run*"
end

# ╔═╡ f1000000-0000-0000-0000-000000000009
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

$(allpass ? "**`theta` is the carrier throughout** — bank, A₂ and the ray transform agree." :
            "**GATE FAILURE** — a convention is inconsistent.")
""")
end

# ╔═╡ Cell order:
# ╟─f1000000-0000-0000-0000-000000000001
# ╠═f1000000-0000-0000-0000-000000000002
# ╟─f1000000-0000-0000-0000-000000000003
# ╠═f1000000-0000-0000-0000-000000000004
# ╟─f1000000-0000-0000-0000-000000000005
# ╠═f1000000-0000-0000-0000-000000000006
# ╟─f1000000-0000-0000-0000-000000000007
# ╠═f1000000-0000-0000-0000-000000000008
# ╠═f1000000-0000-0000-0000-000000000009
