# Validation battery for `GaborStack` — the gate that must pass before any accuracy
# number is worth reading.
#
# Every test is on synthetic stimuli with known ground truth, and every pass criterion
# is stated before the number is printed. This exists because a front end validated only
# by downstream accuracy can be quietly broken in ways the classifier compensates for;
# `Dense_Gabors`' keypoint detector turned out to be miscalibrated on *clean* input, and
# that was found late.
#
# Nothing here imports EMNIST. Run:
#
#     julia --project=. RationalGaborFeatures/Validate_GaborStack.jl

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "GaborStack.jl"))
include(joinpath(@__DIR__, "Stimuli.jl"))
using .GaborStack, .Stimuli, Printf, Statistics

const N = 112

# ---------------------------------------------------------------- bank
# Three scales, not four. The usable band is ρ ∈ [2, 7] — only 1.81 octaves — so four
# scales sat 0.6 octaves apart while carrying 2-octave bandwidth, making adjacent
# channels nearly the same filter. Three gives 0.90-octave spacing against the same
# bandwidth: still an overcomplete frame (deliberately), but no longer paying for
# channels that duplicate their neighbours. Bandwidth is kept broad rather than narrowed
# to separate the scales, because σ_x ∝ (2^β+1)/(2^β−1) nearly doubles from β=2 to β=1,
# and localisation is exactly what the AND layer needs.
const LADDER = [2.0, 3.742, 7.0]
const LAMS   = [N/r for r in LADDER]
const BETAS  = [2.0, 1.6, 1.2]
const NORI   = [8, 12, 16]

H, W, border = field_for((N, N), LADDER; n_orient=NORI, beta=BETAS)
bank = make_bank((H, W), LADDER; imwidth=N, n_orient=NORI, beta=BETAS)
ORI  = [(i, m) for (i, m) in enumerate(bank.meta) if m.kind === :oriented]

"Both spatial extents per scale — the larger sets the padding and the pooling scale.
σ_φ is read from the bank rather than recomputed; duplicating it is how the AND layer's
probe offsets silently kept stale values when the bank's angular tuning changed."
function extents(i)
    m = first(m for m in bank.meta if m.kind === :oriented && m.rho0 ≈ LADDER[i])
    (sigma_x(LAMS[i], BETAS[i]), sigma_along(LADDER[i], m.sigma_phi, N))
end
idx_of(ρ) = [i for (i, m) in enumerate(bank.meta) if m.kind === :oriented && m.rho0 ≈ ρ]

@printf("field %d×%d (border %d, FFT-friendly)   bank: %d filters = %d oriented + lowpass\n",
        H, W, border, length(bank), length(ORI))
@printf("ladder ρ = %s   λ@%d = %s\n", join(LADDER, ", "), N,
        join([@sprintf("%.0f", l) for l in LAMS], ", "))
@printf("σ_x    = %s px  (localisation; stroke ≈ 12.7 px)\n\n",
        join([@sprintf("%.1f", sigma_x(l, b)) for (l, b) in zip(LAMS, BETAS)], ", "))

pass = Ref(true)
function check(name, ok, detail)
    pass[] &= ok
    @printf("  [%s] %-46s %s\n", ok ? "PASS" : "FAIL", name, detail)
end

println("="^78); println("VALIDATION"); println("="^78)

# ---- 1. polarity invariance: must be EXACT, not approximate -------------------
I = bar(N, π/6)
E⁺ = energy_stack(I, bank); E⁻ = energy_stack(-I, bank)
d = maximum(abs.(E⁺ .- E⁻))
check("polarity invariance (criterion: exactly 0)", d == 0, @sprintf("max|ΔE| = %.3e", d))

# ---- 2. DC rejection ---------------------------------------------------------
Ec = energy_stack(fill(0.5f0, N, N), bank; mode=:replicate)
check("constant image gives zero energy (< 1e-10)", maximum(Ec) < 1e-10,
      @sprintf("max E = %.3e", maximum(Ec)))

# ---- 3. orientation readout --------------------------------------------------
# A bar at θ has its wavevector at θ+90°, so the winning carrier should be θ+90 (mod π).
finest = [(i, m) for (i, m) in ORI if m.rho0 ≈ LADDER[3]]
spacing = π / NORI[3]
errs = Float64[]
for θdeg in 0:15:165
    θ = deg2rad(θdeg)
    E = energy_stack(bar(N, θ), bank)
    tot = [sum(@view E[:, :, i]) for (i, _) in finest]
    got = finest[argmax(tot)][2].theta
    want = mod(θ + π/2, π)
    e = abs(atan(sin(got - want), cos(got - want)))
    push!(errs, rad2deg(e))
end
check("orientation readout (criterion: ≤ ½ spacing = $(round(rad2deg(spacing)/2,digits=1))°)",
      maximum(errs) <= rad2deg(spacing)/2 + 1e-9,
      @sprintf("max error %.1f° over 12 angles", maximum(errs)))

# ---- 4. no wraparound contamination -----------------------------------------
# Wraparound is an ARTEFACT, so the test is an invariance: doubling the border must not
# change the answer. (Measuring "energy far from a blob" instead measures the filters'
# genuine spatial tails — at σ_along = 34 px a response 2.5σ away is legitimately ~2e-3
# of peak, which has nothing to do with the FFT.)
#
# The residual after that is *frequency-grid discretisation*, not wraparound, and it is
# confirmed by two signatures: it concentrates in the coarsest, worst-sampled channel
# (ρ=2 gives 3.1e-4 against ρ=7's 6.6e-7 — a factor of 470) and it shrinks as the grid
# refines (320↔525: 3.1e-4; 525↔729: 5.1e-6). At field 320 the ρ₀=2 filter is spanned by
# only ~6 frequency samples, so this is expected, and it is irrelevant operationally
# since a given dataset uses one fixed field throughout.
Iedge = zeros(Float32, N, N); Iedge[50:62, 3:15] .= 1f0
H2, W2, _ = field_for((N, N), LADDER; n_orient=NORI, beta=BETAS, k=6)
bank2 = make_bank((H2, W2), LADDER; imwidth=N, n_orient=NORI, beta=BETAS)
Ea = energy_stack(Iedge, bank); Eb = energy_stack(Iedge, bank2)
relof(sel) = maximum(abs.(@view(Ea[:, :, sel]) .- @view(Eb[:, :, sel]))) /
             maximum(@view Ea[:, :, sel])
rel_ori = maximum(relof(idx_of(ρ)) for ρ in LADDER)
rel_lp  = relof([i for (i, m) in enumerate(bank.meta) if m.kind === :lowpass])
check("padding invariance, oriented channels (< 1e-3)", rel_ori < 1e-3,
      @sprintf("field %d vs %d → max rel. change %.2e  (lowpass, ρ₀=1: %.2e)",
               H, H2, rel_ori, rel_lp))

# ---- 5. scale selectivity ----------------------------------------------------
# Each scale must win for the grating it is tuned to. The stimulus is a *Gabor patch*
# (Gaussian-windowed grating), not a bare grating: a square-windowed grating's edges
# dump broadband energy at low frequency, which hands every trial to the coarsest
# filter regardless of λ. (Measured: bare gratings put λ=16 at the ρ=2 channel.)
# Criterion is each filter's own TUNING CURVE, not argmax across filters: the bank is
# deliberately overcomplete (4 scales over 1.81 octaves, i.e. 0.6 octaves apart, with
# 2-octave bandwidth), so adjacent channels overlap heavily by design and "which scale
# wins" is not a well-posed question. What must hold is that each filter peaks at the
# wavelength it is tuned to.
SWEEP = [72.0, 56.0, 45.0, 37.0, 30.0, 24.0, 20.0, 16.0, 13.0]
resp = zeros(length(LADDER), length(SWEEP))
for (li, λ) in enumerate(SWEEP)
    E = energy_stack(gabor_patch(N, λ, 0.0), bank)
    for (si, ρ) in enumerate(LADDER)
        resp[si, li] = sum(sum(@view E[:, :, i]) for i in idx_of(ρ))
    end
end
peakλ = [SWEEP[argmax(@view resp[si, :])] for si in 1:length(LADDER)]
tuned = [abs(log2(peakλ[si] / LAMS[si])) <= 0.35 for si in 1:length(LADDER)]  # ⅓ octave
check("each filter peaks at its own λ (criterion: within ⅓ octave)", all(tuned),
      "peak λ = " * join([@sprintf("%.0f", p) for p in peakλ], ", ") *
      "  vs tuned " * join([@sprintf("%.0f", l) for l in LAMS], ", "))

# ---- 6. localisation, PER SCALE ---------------------------------------------
# Must be judged against each scale's own extent. A single fixed mask fails trivially,
# since the two extents differ per scale and neither is small.
fracs = Float64[]
for (si, ρ) in enumerate(LADDER)
    sx, sa = extents(si)
    E = energy_stack(bar(N, 0.0; w=13.0, len=70.0), bank)
    tot = dropdims(sum(@view E[:, :, idx_of(ρ)]; dims=3); dims=3)
    mask = bar(N, 0.0; w=13 + 4sx, len=70 + 4sa) .> 0    # dilated by 2σ in each axis
    push!(fracs, sum(tot .* mask) / sum(tot))
end
check("energy localised on the bar, per scale (> 0.75)", minimum(fracs) > 0.75,
      "fracs = " * join([@sprintf("%.3f", f) for f in fracs], ", "))

# ---- 7. the co-location negative control (for the AND layer) -----------------
# |E₄|-style pooled statistics CANNOT tell a corner from two disjoint strokes.
# This is not a bug — it is the measurement that motivates the AND layer, recorded
# here so the AND's benefit is measured against a known baseline.
Ecor = energy_stack(corner(N, π/2; len=40.0), bank)
Esep = energy_stack(two_bars(N, 70.0), bank)
poolstat(E) = [sum(@view E[:, :, i]) for (i, _) in ORI]
a, b = poolstat(Ecor), poolstat(Esep)
cosim = sum(a .* b) / (sqrt(sum(abs2, a)) * sqrt(sum(abs2, b)))
@printf("  [ -- ] %-46s cos = %.4f\n",
        "corner vs two disjoint strokes, POOLED", cosim)
println("         ^ baseline for the AND layer: pooled orientation energy cannot")
println("           separate these. Phase 3 must lower this materially.")

println("="^78)
println(pass[] ? "ALL GATES PASSED — cleared for Phase 3 (AND layer)" :
                 "GATE FAILURE — fix before proceeding")
println("="^78)
