### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ d1000000-0000-0000-0000-000000000001
md"""
# Are the operator's constants scale-free, or fitted to EMNIST?

`d_factor = 1.0` and `dtheta_on_sigma = 0.75` sit inside a module that is supposed to be
general. Both were chosen against **EMNIST-shaped evidence**:

* `d_factor` was swept on `barstim(w=13, len=90)` and `blob(r=6.5)` — dimensions picked to
  match EMNIST's 12.7 px stroke. So what was established is *"1.0 is right when
  σ_along ≈ 0.76 × stroke width"*, not that 1.0 is right.
* `dtheta_on_sigma` was justified explicitly by "σ_along = 34 px at 1.5 is longer than any
  EMNIST stroke" — an EMNIST argument for a constant in a general module.

The **ladder** is supposed to adapt to the data; the **operator's internal geometry** is
supposed to be a constant of the operator, fixed relative to its own filters — as an
end-stopped cell's inhibitory zone is fixed relative to its own receptive field, not to the
stimulus. If those constants drift with the data, EMNIST has been smuggled into the module.

**The dimensionless statement.** With `σ_φ = (π/n)/dtheta`,

```
σ_along / λ  =  n · dtheta / (2π²)
```

which is **independent of ρ**. The operator's shape is fixed by `n` and `dtheta` alone, so
the only free ratio left is the stimulus scale, `w / λ`. That is what to sweep.

No EMNIST anywhere in this notebook.
"""

# ╔═╡ d1000000-0000-0000-0000-000000000002
begin
    using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
    using Printf, Statistics, Plots
    include(joinpath(@__DIR__, "GaborStack.module.jl"))
    include(joinpath(@__DIR__, "AndLayer.module.jl"))
    include(joinpath(@__DIR__, "Stimuli.module.jl"))
    using .GaborStack, .AndLayer, .Stimuli
    gr()
    md"*setup loaded*"
end

# ╔═╡ d1000000-0000-0000-0000-000000000003
begin
    const N = 160
    const NORI1 = 16
    const WBAR, LBAR, RBLOB = 13.0, 90.0, 6.5     # stimulus FIXED

    # Sweep the FILTER scale, not the stimulus. Same dimensionless ratio w/λ = w·ρ/N, but
    # the stimulus and therefore every index range stays constant — scaling the stimulus
    # instead runs it off the field at large w/λ.
    const BAR  = barstim(N, 0.0; w=WBAR, len=LBAR)
    const BLOB = blob(N; r=RBLOB)
    wlam(ρ) = WBAR * ρ / N

    function onebank(ρ, dtheta)
        H, W, _ = field_for((N,N), [ρ]; n_orient=NORI1, beta=1.2, dtheta_on_sigma=dtheta)
        make_bank((H,W), [ρ]; imwidth=N, n_orient=NORI1, beta=1.2,
                  lowpass=false, dtheta_on_sigma=dtheta)
    end
    σalong(bk) = (m = first(bk.meta); m.imwidth / (2π * m.rho0 * m.sigma_phi))

    const C = round(Int, (N+1)/2)
    const EO = round(Int, LBAR/2) - 5      # end band
    const IO = round(Int, LBAR/6)          # interior band
    "min(end/interior, end/blob) — both must hold, so the weaker one is what matters."
    function score(bk, dfac)
        A, lab = and_maps(energy_stack(BAR, bk), bk.meta; forms=(:A2,), d_factor=dfac)
        M = @view A[:, :, 1]
        ends = maximum(M[C-6:C+6, [C-EO:C-EO+10; C+EO-10:C+EO]])
        intr = maximum(M[C-6:C+6, C-IO:C+IO])
        Ab, _ = and_maps(energy_stack(BLOB, bk), bk.meta; forms=(:A2,), d_factor=dfac)
        min(ends/max(intr,eps()), ends/max(maximum(Ab),eps())), lab[1].d
    end
    md"*helpers defined — stimulus fixed, filter scale swept*"
end

# ╔═╡ d1000000-0000-0000-0000-000000000004
md"""
## Is the best `d_factor` the same at every stimulus scale?

If it is, 1.0 is a property of the operator. If the optimum drifts with `w/λ`, then `d` is
anchored to the wrong thing and should follow a measured structure scale instead.
"""

# ╔═╡ d1000000-0000-0000-0000-000000000005
begin
    const DFACS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    const RHOS  = [3.7, 6.2, 9.8, 14.8]           # → w/λ ≈ 0.30 / 0.50 / 0.80 / 1.20
    bk = Dict(ρ => onebank(ρ, 0.75) for ρ in RHOS)
    @printf("σ_along/λ = %.3f at every ρ  (predicted n·dtheta/2π² = %.3f)\n\n",
            σalong(bk[RHOS[1]])/(N/RHOS[1]), NORI1*0.75/(2π^2))
    @printf("%-9s", "w/λ");  for d in DFACS; @printf("%9s", "d=$(d)"); end; println("     best")
    bestd = Float64[]
    for ρ in RHOS
        vs = [score(bk[ρ], d)[1] for d in DFACS]
        @printf("%-9.2f", wlam(ρ)); for v in vs; @printf("%9.2f", v); end
        push!(bestd, DFACS[argmax(vs)]); @printf("   %5.2f\n", bestd[end])
    end
    println("\nbest d_factor across scales: ", bestd,
            all(==(bestd[1]), bestd) ? "  → CONSTANT, the operator is scale-free" :
                                       "  → DRIFTS, d is anchored to the wrong quantity")
end

# ╔═╡ d1000000-0000-0000-0000-000000000006
md"""
## Is the best `dtheta_on_sigma` the same at every stimulus scale?

`dtheta` sets `σ_along/λ` and therefore how long the filter is relative to its own
wavelength. The claim behind 0.75 was "σ_along must be shorter than the structures you
want to resolve" — general in form, but the value was picked for EMNIST.
"""

# ╔═╡ d1000000-0000-0000-0000-000000000007
begin
    const DTS = [0.5, 0.75, 1.0, 1.5, 2.0]
    @printf("%-9s", "w/λ"); for dt in DTS; @printf("%11s", "dθ=$(dt)"); end; println("     best")
    bestdt = Float64[]
    for ρ in RHOS
        vs = [score(onebank(ρ, dt), 1.0)[1] for dt in DTS]
        @printf("%-9.2f", wlam(ρ)); for v in vs; @printf("%11.2f", v); end
        push!(bestdt, DTS[argmax(vs)]); @printf("   %5.2f\n", bestdt[end])
    end
    println("\nbest dtheta across scales: ", bestdt,
            all(==(bestdt[1]), bestdt) ? "  → CONSTANT" : "  → DRIFTS")
    println("σ_along/λ at each dθ: " *
            join([@sprintf("%.2f", NORI1*dt/(2π^2)) for dt in DTS], "  "))
end

# ╔═╡ Cell order:
# ╟─d1000000-0000-0000-0000-000000000001
# ╠═d1000000-0000-0000-0000-000000000002
# ╠═d1000000-0000-0000-0000-000000000003
# ╟─d1000000-0000-0000-0000-000000000004
# ╠═d1000000-0000-0000-0000-000000000005
# ╟─d1000000-0000-0000-0000-000000000006
# ╠═d1000000-0000-0000-0000-000000000007
