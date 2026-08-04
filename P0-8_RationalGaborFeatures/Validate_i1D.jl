# ── PLAIN SCRIPT, not a Pluto notebook — runs headless as a gate ────────────
# `julia --project=.. Validate_i1D.jl`   (from the P0-8_RationalGaborFeatures directory)
#
# Does A₁ actually satisfy the i2D-selectivity criterion it is built on?
#
# Zetzsche & Barth (*Fundamental limits of linear filters in the visual processing of
# two-dimensional signals*, Vision Research 30:1111–1117, 1990) prove that **no linear filter
# can be selective for i2D structure**. An *intrinsically 1D* signal — a straight edge or bar,
# any cross-section — has its spectrum confined to a line through the origin. Any linear
# filter's passband has area, so it intersects some such line, so it responds to some straight
# edge. Selectivity therefore requires a nonlinearity, and specifically requires the quadratic
# Volterra kernel `H₂(f₁,f₂)` to **vanish on collinear frequency pairs**.
#
# That result also states, in one line, what Phases 5–8 discovered empirically. Quadrature
# energy `E = (f∗g_even)² + (f∗g_odd)²` *is* a quadratic Volterra operator: the term
# oscillating at `f₁+f₂` cancels between the even and odd halves — that cancellation is exactly
# what makes it phase-invariant — leaving support concentrated on `f₂ = −f₁`, which is
# collinear. **Oriented energy sits precisely on the forbidden diagonal.** It cannot be
# i2D-selective, by construction, and no amount of downstream learning changes that.
#
# `A₁ = Σₖ Eₖ·Eₖ₊ₙ⁄₂ / C₀` is our answer: an AND of two such kernels at 90°, so 4th order in
# the image and off-diagonal. Zetzsche's minimal operator multiplies two *bandpass* outputs
# instead, which is 2nd order — polarity-invariant, since it is even in `f`, but **not**
# phase-invariant, so it oscillates as you slide along a contour and needs rectifying to give a
# stable reading at a corner. We pay the extra order up front and get positional stability.
#
# WHAT THIS SCRIPT MEASURES. The criterion is falsifiable and we had never tested it: `A₁`
# should read ≈ 0 on an i1D input at **every** orientation. It will not be exactly 0, because
# orientation channels have angular bandwidth — with `dtheta_on_sigma = 0.75` the angular σ is
# `(π/n)/0.75`, so at n = 8 the 90° partner sits 3σ away and receives `exp(−4.5) ≈ 1.1 %` of
# the amplitude. The question is how much leaks in practice.
#
# WHY IT MATTERS BEYOND TIDINESS. Phase 7 measured `R²(A ← orient) = 0.933` on EMNIST and read
# it as "different operators that become near-collinear *on handwriting*". But if A₁ leaks i1D
# energy, then it is **partly a function of orientation energy by construction**, and that 0.933
# is a fact about our implementation rather than about handwriting. The two readings have very
# different consequences, and this separates them.
#
# ── the confound this script had to be rewritten to remove ──────────────────
#
# First version generated stimuli at 112×112 and let `energy_stack` replicate-pad them. It
# reported ~5 % leakage at ρ = 2 and passed the other two scales by four orders of magnitude —
# and the worst orientation was the **diagonal** in both stimulus families. That is the tell:
# replicate padding extends border pixels outward, which continues an axis-aligned bar
# correctly and a diagonal bar into a fan of wrong geometry. At ρ = 2 the filter's 3σ reach is
# ~52 px and the image centre is 56 px from the border, so the coarsest scale sees it.
#
# So each measurement is made twice:
#
#   * **as deployed** — stimulus at 112, replicate-padded, exactly what production does. This
#     is the honest number for what the front end computes on a real image.
#   * **field** — stimulus generated across the whole 224×224 padded field, so the i1D signal
#     is exact over the entire filter support and no padding is synthesised at all. This
#     isolates the *operator's* selectivity from the test's own border artefact.
#
# The gate is on `field`, because that is the property of A₁. The gap between the two columns
# is the cost of padding a finite image, which is a separate (and unavoidable) matter.
#
# Two stimulus families, because they probe different things:
#
#   * a **grating** at the scale's own wavelength — exactly i1D, spectral support at one
#     collinear pair ±f, no endpoints. The literal test of the criterion.
#   * a **full-field bar** of the measured EMNIST stroke width — what the front end actually
#     sees in a line drawing. An infinite bar is also exactly i1D (its transform is
#     `G(f_n)·δ(f_t)`, a line through the origin), but it spans many scales at once.
#
# ONE EXPECTED DISCREPANCY. The analytic prediction below matches measurement to within 2 % in
# five of six cells. The exception is the **bar at ρ = 7** (λ = 16 px), where measurement is
# ~170× the prediction — still 1.7e−5 of the i2D response, so three orders inside the gate.
# The cause is the stimulus, not the operator: a bar rasterised at a non-axis angle has a
# pixel staircase along its edges, and that staircase is real i2D structure. At ρ = 7 the
# filter is only σ_x ≈ 7.6 px and resolves it; at the coarser scales it does not. The analytic
# model assumes a single exact orientation and so cannot contain this term.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Printf, Statistics, LinearAlgebra
include(joinpath(@__DIR__, "GaborStack.module.jl"))
include(joinpath(@__DIR__, "AndLayer.module.jl"))
using .GaborStack, .AndLayer

const N      = 112
const LADDER = [2.0, 3.742, 7.0]
const BETAS  = [2.0, 1.6, 1.2]
const NORI   = [8, 12, 16]
const BG     = 0.5f0        # grey, so the test also covers the non-zero background case
const AMP    = 0.4f0
const WSTROKE = 13.0        # measured EMNIST stroke width
const TOL_LEAK = 0.02       # gate: i1D leakage must be under 2 % of the crossing response
# Which A1 to test. `:analytic` is the production default and subtracts the closed-form i1D
# floor; `:none` is the original operator and is expected to FAIL this gate at rho=2 — that
# failure is what motivated the default change, so it is kept runnable as evidence.
const FLOOR = Symbol(get(ENV, "I1D_FLOOR", "analytic"))

# ---------------------------------------------------------------- stimuli
# All take the canvas size and a wavelength **in pixels**, so the same stimulus can be built at
# 112 or across the full padded field without retuning it.

"Sinusoidal grating of wavelength `lam` px whose ridges run at θ."
function grating(M, θ; lam=29.9, amp=AMP, bg=BG)
    I = Matrix{Float32}(undef, M, M); k = 2π / lam
    c, s = cos(θ), sin(θ)
    @inbounds for y in 1:M, x in 1:M
        I[y, x] = bg + amp * sin(k * (-(x - (M+1)/2) * s + (y - (M+1)/2) * c))
    end
    I
end

"Sum of two orthogonal gratings — the i2D reference."
plaid(M, θ; kw...) = grating(M, θ; kw...) .+ grating(M, θ + π/2; kw...) .- BG

"Bar of width `w` px whose ridge runs at θ, spanning the canvas, with smoothstep edges."
function fullbar(M, θ; w=WSTROKE, amp=AMP, bg=BG, ramp=1.5, kw...)
    I = Matrix{Float32}(undef, M, M); c, s = cos(θ), sin(θ)
    @inbounds for y in 1:M, x in 1:M
        dx = x - (M+1)/2; dy = y - (M+1)/2
        n = abs(-dx * s + dy * c)
        t = clamp((w/2 + ramp - n) / (2ramp), 0, 1)
        I[y, x] = bg + amp * (t * t * (3 - 2t))
    end
    I
end

"Two spanning bars at 90° — an X whose arms are unbounded, so the crossing is the only i2D structure."
crossbar(M, θ; kw...) = max.(fullbar(M, θ; kw...), fullbar(M, θ + π/2; kw...))

# ---------------------------------------------------------------- measurement

"""
`A₁ / C₀` at the exact canvas centre, per scale.

Normalising by `C₀` makes it dimensionless and comparable across stimuli of different contrast.
It is 0 for a perfect AND on i1D input. The i2D reference is **not** 0.5, even though a
crossing puts equal energy in two channels: `C₀` sums all `n` channels, and with σφ = 30° at
n = 8 the neighbours of an active channel pick up `exp(−(22.5/30)²/2) = 0.75` of the amplitude,
so `C₀` is several times the peak channel and the ratio lands near 0.15. That is why the gate is
on the *ratio* of i1D leakage to the measured i2D response rather than on an absolute value.
"""
function a1_over_c0(img, bank)
    E = energy_stack(img, bank; mode=:replicate, crop=true)
    maps, labels = AndLayer.a1_maps(E, bank.meta; floor=FLOOR)
    cy, cx = (size(img, 1) + 1) ÷ 2, (size(img, 2) + 1) ÷ 2
    out = Float64[]
    for (m, lab) in zip(maps, labels)
        ch = AndLayer.scale_channels(bank.meta, lab.rho0)
        c0 = sum(E[cy, cx, i] for (i, _) in ch)
        push!(out, c0 > 0 ? m[cy, cx] / c0 : 0.0)
    end
    out
end

"""
Analytic `A₁/C₀` for a perfectly i1D input at angle `θ0`, from the angular tuning alone.

The bank is polar-separable, so for a single-orientation input the radial factor is common to
every channel and cancels in the ratio. Channel `j` at angle `θⱼ` then has energy
`exp(−(Δⱼ/σφ)²)` with `Δⱼ` the circular distance on [0,π) — energy, so amplitude
`exp(−Δ²/2σφ²)` squared — and `A₁/C₀ = S/C₀²` follows directly.

This exists to identify *which* channel pair leaks. The obvious suspect is the 90° partner of
the active channel, but at n = 8 that sits 3σφ away and contributes only `exp(−9) = 1.2e−4`.
The real culprit is the pair at **±45°**, which straddles the line: each is 1.5σφ away and
retains `exp(−2.25) = 0.105` of the energy, and the two are exactly 90° apart, so A₁ multiplies
them together. That term is `0.105² = 1.1e−2` — two orders of magnitude above the ⊥ pair, and
the whole of the measured leakage.
"""
function a1_predicted(n, σφ; θ0=0.0)
    θj = [(j - 1) * π / n for j in 1:n]
    Δ = [min(abs(t - θ0), π - abs(t - θ0)) for t in θj]
    E = exp.(-(Δ ./ σφ) .^ 2)
    C0 = sum(E)
    S = sum(E[k] * E[mod1(k + n ÷ 2, n)] for k in 1:n)
    S / C0^2
end

function main()
    @printf("The i2D-selectivity criterion applied to A₁ (floor=%s) — grey background %.2f\n\n",
            FLOOR, BG)
    HF, WF, border = field_for((N, N), LADDER; n_orient=NORI, beta=BETAS)
    bank = make_bank((HF, WF), LADDER; imwidth=N, n_orient=NORI, beta=BETAS)
    ρs = sort(AndLayer.scales_of(bank.meta))
    λs = N ./ ρs
    @printf("ρ = %s   λ = %s px   orientations %s\n",
            join(round.(ρs, digits=2), ", "), join(round.(λs, digits=1), ", "),
            join(NORI, ", "))
    @printf("padded field %d×%d (border %d px), image %d×%d\n\n", HF, WF, border, N, N)

    θs = range(0, π - π/60, length=60)      # 3° steps — channel centres and the gaps between
    fails = String[]

    for (fam, i1d, i2d) in (("grating — exactly i1D", grating, plaid),
                            ("bar w=13 px — i1D, broadband", fullbar, crossbar))
        @printf("%s\n%s\n%s\n", "="^88, fam, "="^88)
        @printf("%6s %6s | %11s %11s %9s | %11s %11s %9s %8s\n",
                "ρ", "n", "i1D max", "i2D mean", "ratio", "i1D max", "i2D mean", "ratio",
                "worst θ")
        @printf("%6s %6s | %33s | %42s\n", "", "", "--- as deployed (112, padded) ---",
                "----------- field (224, exact i1D) -----------")
        for (k, ρ) in enumerate(ρs)
            lam = λs[k]
            # `a1_over_c0` returns all three scales; keep row k of each measurement
            dep_l = [a1_over_c0(i1d(N,  θ; lam=lam), bank)[k] for θ in θs]
            dep_x = [a1_over_c0(i2d(N,  θ; lam=lam), bank)[k] for θ in θs]
            fld_l = [a1_over_c0(i1d(HF, θ; lam=lam), bank)[k] for θ in θs]
            fld_x = [a1_over_c0(i2d(HF, θ; lam=lam), bank)[k] for θ in θs]
            rd = maximum(dep_l) / max(mean(dep_x), eps())
            rf = maximum(fld_l) / max(mean(fld_x), eps())
            ok = rf < TOL_LEAK
            σφ = (π / NORI[k]) / 0.75
            pred = maximum(a1_predicted(NORI[k], σφ; θ0=θ) for θ in θs)
            @printf("%6.2f %6d | %11.2e %11.4f %9.1e | %11.2e %11.4f %9.1e %7.1f°%s\n",
                    ρ, NORI[k], maximum(dep_l), mean(dep_x), rd,
                    maximum(fld_l), mean(fld_x), rf,
                    rad2deg(θs[argmax(fld_l)]), ok ? "" : "  LEAKS")
            @printf("%13s | angular tuning alone predicts i1D max %.2e  (measured %.2e, %.0f %% of it)\n",
                    "", pred, maximum(fld_l), 100 * maximum(fld_l) / max(pred, eps()))
            ok || push!(fails, @sprintf("%s ρ=%.2f: field leakage %.3f of i2D response",
                                        fam, ρ, rf))
            flush(stdout)
        end
        println()
    end

    println("="^88)
    if isempty(fails)
        @printf("ALL GATES PASSED — A₁ is i2D-selective to better than %.0f %% at every scale,\n",
                100TOL_LEAK)
        println("orientation and stimulus family tested.\n")
        println("""
So A₁ is a genuine AND in Zetzsche & Barth's sense, and Phase 7's R²(A ← orient) = 0.933
on EMNIST is NOT an artefact of i1D leakage. That near-collinearity is a fact about
handwriting: it is the within-cell covariance Cov_x(Eₖ, Eₖ₊ₙ⁄₂) being small or predictable
on those images, not A₁ secretly reporting orientation energy.

Note the `as deployed` column separately. Where it exceeds `field`, the excess is the cost
of replicate-padding a finite image: padding continues an axis-aligned bar correctly and a
diagonal one into wrong geometry, and the coarsest scale's filter reaches the border. That
is a property of finite images, not of A₁.""")
    else
        println("GATES FAILED:"); for f in fails; println("  - ", f); end
        println("""
A₁ leaks on exactly-i1D input at the coarsest scale, so at ρ = 2 it is partly a function
of orientation energy by construction. The analytic column identifies the mechanism
exactly: not the 90° partner of the active channel — that is 3σφ away and worth
exp(−9) = 1.2e−4 — but the **pair straddling the line at ±45°**, each 1.5σφ away and
retaining exp(−2.25) = 0.105 of the energy, and exactly 90° apart, so A₁ multiplies them.

TWO FIXES, both with costs, so this is a design decision and not a bug fix:

  * more orientations at ρ = 2 (8 → 12 would buy ~280×), but σφ = (π/n)/dtheta_on_sigma
    shrinks with n and σ_along = W/(2πρσφ) grows with it, so the coarsest filter lengthens
    from 17 to 26 px — and "filters longer than any stroke" is a bug this project already
    fixed once.
  * subtract the analytic floor. The leakage is `c(n,σφ)·C₀` with `c` known in closed form
    and image-independent, so `A₁' = max(0, A₁ − c·C₀)` is exactly zero on i1D input and
    costs the i2D response only 4.6 % at ρ = 2, less than 0.02 % elsewhere.

WHAT THIS DOES *NOT* SHOW. It does not rescue the reading that Phase 7's
R²(A ← orient) = 0.933 is a leakage artefact. Phase 10b measured R² per scale: A₁ at ρ = 7,
whose leakage is 6.1e−8 — effectively zero — is still 0.856 (EMNIST) and 0.883 (Fashion)
predictable from orient. Predictability at that level survives the removal of leakage
entirely, so leakage is not what produces it.""")
    end
    isempty(fails)
end

main() || exit(1)
