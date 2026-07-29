### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ e1000000-0000-0000-0000-000000000001
md"""
# Validating the ray transform

`RayHarmonics.module.jl` went into the repository and was used in the `F`/`f` probe
**without a validation battery**, which every other module here has. That is exactly the
gap this project has been burned by twice — a detector built and relied on before anyone
checked it against ground truth. This notebook closes it.

The operator claims a precise signature. For `n` rays of equal length meeting at a point:

| configuration | c₀ | \\|c₁\\|/c₀ | \\|c₂\\|/c₀ |
|:--|--:|--:|--:|
| endpoint (1 ray) | 1 | 1.000 | 1.000 |
| straight (2 opposite) | 2 | 0.000 | 1.000 |
| L-corner (2 at 90°) | 2 | 0.707 | 0.000 |
| **T-junction (3)** | **3** | **0.333** | **0.333** |
| X-crossing (4) | 4 | 0.000 | 0.000 |

These follow exactly from `cₙ = Σₖ e^{−inφₖ}` over the ray directions, so they are
predictions rather than fits.

**The gate that matters is T versus X**, and it has to be set up carefully.

A₁ is built on the π-periodic orientation profile, so it should be blind to the difference.
But *how* the T and X are built decides whether that is testable. Three equal rays versus
four equal rays also changes the **orientation balance** (horizontal:vertical is 2:1 against
2:2), and A₁ reads balance perfectly well — measured, it separates that pair by 45 %, which
is a proxy for ray count rather than a count.

The pair that isolates ray count uses the **same two bars repositioned**: a stem meeting the
edge of a crossbar versus the same stem crossing its centre. Identical orientations in
identical amounts; only the meeting differs. On that pair A₁ falls to **0.2 %** while c₀
holds **9.1 %**. That is the comparison gated below.

**Readout is at the junction point, not summed over the image.** Summing is precisely how a
total-ink confound was once mistaken for a ray-count result (`RESULTS.md` §5.1).
"""

# ╔═╡ e1000000-0000-0000-0000-000000000002
begin
    using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
    using Printf, Statistics, Plots
    include(joinpath(@__DIR__, "GaborStack.module.jl"))
    include(joinpath(@__DIR__, "AndLayer.module.jl"))
    include(joinpath(@__DIR__, "RayHarmonics.module.jl"))
    include(joinpath(@__DIR__, "Stimuli.module.jl"))
    using .GaborStack, .AndLayer, .RayHarmonics, .Stimuli
    gr()
    md"*setup loaded*"
end

# ╔═╡ e1000000-0000-0000-0000-000000000003
begin
    N = 160
    LADDER = [2.0, 3.742, 7.0]; BETAS = [2.0, 1.6, 1.2]; NORI = [8, 12, 16]
    Hf, Wf, _ = field_for((N,N), LADDER; n_orient=NORI, beta=BETAS)
    bank = make_bank((Hf,Wf), LADDER; imwidth=N, n_orient=NORI, beta=BETAS)
    CW = round(Int, (N+1)/2)
    CASES = [("endpoint (1)",  [0.0],                    1),
             ("straight (2)",  [0.0, π],                 2),
             ("L-corner (2)",  [0.0, π/2],               2),
             ("T-junction (3)",[0.0, π/2, π],            3),
             ("X-crossing (4)",[0.0, π/2, π, 3π/2],      4)]
    "Read the harmonics AT the junction, averaged over a small window."
    function sig(angles; win=4)
        E = energy_stack(rays(N, angles), bank)
        M, lab = ray_maps(E, bank.meta)
        k = findall(l -> l.rho0 ≈ LADDER[end], lab)          # finest scale
        [mean(@view M[CW-win:CW+win, CW-win:CW+win, i]) for i in k]   # c0, |c1|/c0, |c2|/c0
    end
    results = Tuple{String,Bool,String}[]
    gate!(n,ok,d) = push!(results,(n,ok,d))
    md"*bank built*"
end

# ╔═╡ e1000000-0000-0000-0000-000000000004
md"## The stimuli — equal-length rays, so only the count varies"

# ╔═╡ e1000000-0000-0000-0000-000000000005
plot([heatmap(rays(N, a); c=:grays, yflip=true, title=nm, titlefontsize=8, aspect_ratio=1,
              axis=nothing, colorbar=false, framestyle=:none) for (nm,a,_) in CASES]...;
     layout=(1,5), size=(950,200))

# ╔═╡ e1000000-0000-0000-0000-000000000006
md"## The measured signature"

# ╔═╡ e1000000-0000-0000-0000-000000000007
begin
    SIG = [sig(a) for (_,a,_) in CASES]
    c0 = [s[1] for s in SIG]; r1 = [s[2] for s in SIG]; r2 = [s[3] for s in SIG]
    nrays = [n for (_,_,n) in CASES]
    c0n = c0 ./ c0[1]
    @printf("%-16s %6s %10s %12s %10s %10s\n",
            "configuration","rays","c₀","c₀ (÷ 1-ray)","|c₁|/c₀","|c₂|/c₀")
    for (i,(nm,_,n)) in enumerate(CASES)
        @printf("%-16s %6d %10.4g %12.2f %10.3f %10.3f\n", nm, n, c0[i], c0n[i], r1[i], r2[i])
    end
    println("\npredicted c₀ ÷ 1-ray: 1 / 2 / 2 / 3 / 4")
    println("predicted |c₁|/c₀   : 1.000 / 0.000 / 0.707 / 0.333 / 0.000")
    md"*signature measured*"
end

# ╔═╡ e1000000-0000-0000-0000-000000000008
md"""
## Gate 1 — does c₀ track ray count?

The claim A₁ failed. Ray count is a 2π property; the offset sampling is what makes it
available.
"""

# ╔═╡ e1000000-0000-0000-0000-000000000009
begin
    ordered = c0n[1] < c0n[4] < c0n[5] && c0n[3] < c0n[4]
    gate!("c₀ increases with ray count (1 < 3 < 4, and L < T)", ordered,
          "c₀ ÷ 1-ray = " * join([@sprintf("%.2f",v) for v in c0n], " / "))
    p1 = scatter(nrays, c0n; ms=7, c=:steelblue, legend=false, xlabel="rays at the junction",
                 ylabel="c₀ (relative to 1 ray)", title="c₀ versus ray count",
                 titlefontsize=9, xticks=1:4)
    plot!(p1, 1:4, 1:4; ls=:dash, lc=:grey)
    p2 = scatter(1:5, r1; ms=7, c=:firebrick, legend=false,
                 xticks=(1:5,[nm for (nm,_,_) in CASES]), ylabel="|c₁|/c₀",
                 title="asymmetry — 1.0 at an endpoint, 0 when centrally symmetric",
                 titlefontsize=9, xrotation=20, bottom_margin=8Plots.mm)
    plot(p1,p2; layout=(1,2), size=(950,320))
end

# ╔═╡ e1000000-0000-0000-0000-00000000000a
md"""
## Gate 2 — T versus X, the thing A₁ cannot do

A T and an X have **identical orientation content**, so any statistic of the orientation
profile is blind to the difference. This is the operator's reason to exist.
"""

# ╔═╡ e1000000-0000-0000-0000-00000000000b
begin
    # ENERGY-MATCHED T and X: the same two bars, the stem repositioned. Using 3-vs-4 equal
    # rays instead would confound ray count with orientation balance, which A₁ can read.
    cbar = (N+1)/2
    Tm = max.(barstim(N, 0.0; w=13.0, len=70.0),
              barstim(N, π/2;  w=13.0, len=35.0, cy=cbar+17.5))
    Xm = max.(barstim(N, 0.0; w=13.0, len=70.0),
              barstim(N, π/2;  w=13.0, len=35.0))
    cen(M,k) = mean(@view M[CW-4:CW+4, CW-4:CW+4, k])
    MT, lt = ray_maps(energy_stack(Tm, bank), bank.meta)
    MX, _  = ray_maps(energy_stack(Xm, bank), bank.meta)
    k0 = findall(l -> l.rho0 ≈ LADDER[end] && l.form === :R0, lt)[1]
    ray_sep = abs(cen(MX,k0) - cen(MT,k0)) / max(cen(MT,k0), cen(MX,k0))
    AT,_ = and_maps(energy_stack(Tm,bank), bank.meta; forms=(:A1,))
    AX,_ = and_maps(energy_stack(Xm,bank), bank.meta; forms=(:A1,))
    a1_sep = abs(cen(AX,size(AX,3)) - cen(AT,size(AT,3))) /
             max(cen(AT,size(AT,3)), cen(AX,size(AX,3)))
    gate!("energy-matched T vs X: c₀ separates, A₁ does not",
          ray_sep > 5*a1_sep && ray_sep > 0.05,
          @sprintf("c₀ %.1f %%  vs  A₁ %.1f %%  (%.0f× better)",
                   100ray_sep, 100a1_sep, ray_sep/max(a1_sep,1e-9)))
    md"*gate 2 run*"
end

# ╔═╡ e1000000-0000-0000-0000-00000000000c
md"## Gate 3 — polarity invariance, as everywhere else in this front end"

# ╔═╡ e1000000-0000-0000-0000-00000000000d
begin
    I = rays(N, CASES[4][2])
    Mp,_ = ray_maps(energy_stack(I, bank), bank.meta)
    Mm,_ = ray_maps(energy_stack(-I, bank), bank.meta)
    dpol = maximum(abs.(Mp .- Mm))
    gate!("polarity invariance (exactly 0)", dpol == 0, @sprintf("max|ΔM| = %.3e", dpol))
    md"*gate 3 run*"
end

# ╔═╡ e1000000-0000-0000-0000-00000000000e
md"""
## Gate 4 — the endpoint signature

`|c₁|/c₀` should be near 1 for a single ray and near 0 for a centrally symmetric figure.
This is the quantity that distinguishes a termination from a through-line, and it is what
A₂ measures by a different route.
"""

# ╔═╡ e1000000-0000-0000-0000-00000000000f
begin
    gate!("|c₁|/c₀ : endpoint high, straight and X low",
          r1[1] > 0.5 && r1[2] < 0.35 && r1[5] < 0.35,
          @sprintf("endpoint %.3f · straight %.3f · L %.3f · T %.3f · X %.3f",
                   r1[1], r1[2], r1[3], r1[4], r1[5]))
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

$(allpass ? "**ALL GATES PASSED**" : "**GATE FAILURE**")
""")
end

# ╔═╡ Cell order:
# ╟─e1000000-0000-0000-0000-000000000001
# ╠═e1000000-0000-0000-0000-000000000002
# ╠═e1000000-0000-0000-0000-000000000003
# ╟─e1000000-0000-0000-0000-000000000004
# ╠═e1000000-0000-0000-0000-000000000005
# ╟─e1000000-0000-0000-0000-000000000006
# ╠═e1000000-0000-0000-0000-000000000007
# ╟─e1000000-0000-0000-0000-000000000008
# ╠═e1000000-0000-0000-0000-000000000009
# ╟─e1000000-0000-0000-0000-00000000000a
# ╠═e1000000-0000-0000-0000-00000000000b
# ╟─e1000000-0000-0000-0000-00000000000c
# ╠═e1000000-0000-0000-0000-00000000000d
# ╟─e1000000-0000-0000-0000-00000000000e
# ╠═e1000000-0000-0000-0000-00000000000f
