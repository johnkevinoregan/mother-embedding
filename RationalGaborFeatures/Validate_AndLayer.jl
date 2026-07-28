# Validation battery for `AndLayer` — Phase 3 gate.
#
# The layer's claim is narrow and testable: pointwise conjunction, applied BEFORE spatial
# pooling, represents co-location, which no statistic of already-pooled orientation
# energy can. So the tests are all of the form "does this separate stimuli that pooled
# energy provably cannot separate".
#
# The baseline was recorded in Phase 2 and is not adjustable here: a corner and two
# disjoint strokes sit at cos = 0.858 under pooled orientation energy.
#
#     julia --project=. RationalGaborFeatures/Validate_AndLayer.jl

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "GaborStack.jl"))
include(joinpath(@__DIR__, "AndLayer.jl"))
include(joinpath(@__DIR__, "Stimuli.jl"))
using .GaborStack, .AndLayer, .Stimuli, Printf, Statistics

const N      = 112
const LADDER = [2.0, 3.742, 7.0]
const LAMS   = [N/r for r in LADDER]
const BETAS  = [2.0, 1.6, 1.2]
const NORI   = [8, 12, 16]

H, W, border = field_for((N, N), LADDER; n_orient=NORI, beta=BETAS)
bank = make_bank((H, W), LADDER; imwidth=N, n_orient=NORI, beta=BETAS)
ORI  = [i for (i, m) in enumerate(bank.meta) if m.kind === :oriented]

@printf("field %d×%d  bank %d filters  ladder ρ = %s\n\n",
        H, W, length(bank), join(LADDER, ", "))

pass = Ref(true)
function check(name, ok, detail)
    pass[] &= ok
    @printf("  [%s] %-44s %s\n", ok ? "PASS" : "FAIL", name, detail)
end
E(img) = energy_stack(img, bank)
pooled(Es) = [sum(@view Es[:, :, i]) for i in ORI]          # the Phase-2 baseline stat
cosim(a, b) = sum(a .* b) / (sqrt(sum(abs2, a)) * sqrt(sum(abs2, b)))

println("="^78); println("PHASE 3 — AND LAYER"); println("="^78)

# ---- 1. A1 at the JUNCTION -------------------------------------------------
# Measured in a window at the centre, not summed over the image. A1 correctly fires at
# line ENDS as well as junctions (an end is i2D too), so a whole-image sum confounds the
# two: `two_bars` has four free ends against a corner's two, which cancels the junction
# out. Localising isolates co-location, which is what the layer claims.
cw = round(Int, (N+1)/2); win = 6
centre(M) = maximum(@view M[cw-win:cw+win, cw-win:cw+win, :])

Ex0 = E(two_bars(N, 0.0))        # bars cross at the centre  -> junction present
Ex1 = E(two_bars(N, 70.0))       # same bars, pulled apart   -> no junction
A0, _ = and_maps(Ex0, bank.meta; forms=(:A1,))
A1_, _ = and_maps(Ex1, bank.meta; forms=(:A1,))
# Per scale: co-location can only be resolved to within the envelope, so the coarse
# scale (σ_along = 34 px) cannot separate a junction from bars 30 px apart, by
# construction. The finest scale is where the claim lives.
cen(M, k) = maximum(@view M[cw-win:cw+win, cw-win:cw+win, k])
rj = [cen(A0, k) / max(cen(A1_, k), eps()) for k in 1:size(A0, 3)]
check("A1 at the junction, finest scale (> 10)", rj[end] > 10,
      "per-scale ratio ρ=2/3.74/7: " * join([@sprintf("%.1f×", r) for r in rj], "  "))

# ---- 2. ... and pooled energy cannot do this -------------------------------
# The control that makes the claim non-trivial: over the same sweep, the pooled
# orientation statistic barely moves, because the orientation histogram is unchanged.
gaps = [0.0, 20.0, 40.0, 55.0, 70.0]
a1c = Float64[]; poolc = Float64[]; ref = pooled(Ex0)
for g in gaps
    Eg = E(two_bars(N, g))
    Ag, _ = and_maps(Eg, bank.meta; forms=(:A1,))
    push!(a1c, cen(Ag, size(Ag,3))); push!(poolc, cosim(pooled(Eg), ref))
end
a1n = a1c ./ a1c[1]
check("A1 falls with gap while pooled energy does not",
      a1n[end] < 0.15 && poolc[end] > 0.9,
      @sprintf("A1(centre) %s ; pooled cos %s",
               join([@sprintf("%.2f", v) for v in a1n], " "),
               join([@sprintf("%.2f", v) for v in poolc], " ")))

# ---- 3. A2 fires at terminations, not interiors or flanks -------------------
# Judged PER SCALE and gated on the finest. End-stopping needs the probe offset to be
# small against the stroke's length; at ρ=2 the offset is ~26 px, so on a 112-px letter
# that scale is inherently degenerate — which is a fact about the image size, not a bug.
Eb = E(bar(N, 0.0; w=13.0, len=90.0))
A2, lab2 = and_maps(Eb, bank.meta; forms=(:A2,))
c = round(Int, (N+1)/2)
ratios = Tuple{Float64,Float64}[]
for k in 1:size(A2, 3)
    M = @view A2[:, :, k]
    e  = maximum(M[c-6:c+6, [c-46:c-36; c+36:c+46]])
    it = maximum(M[c-6:c+6, c-15:c+15])   # true interior: end effects reach ~2σ_along inward
    fl = maximum(M[[c-26:c-14; c+14:c+26], c-25:c+25])
    push!(ratios, (e/max(it,eps()), e/max(fl,eps())))
end
fine = ratios[argmax([l.rho0 for l in lab2])]
check("A2 at the finest scale: end/interior and end/flank > 3",
      fine[1] > 3 && fine[2] > 3,
      @sprintf("finest: end/interior %.1f×, end/flank %.1f×", fine...))
for (k, l) in enumerate(lab2)
    @printf("         ρ=%-6.2f d=%5.1f px   end/interior %6.1f×  end/flank %6.1f×\n",
            l.rho0, l.d, ratios[k]...)
end

# ---- 4. A2 does not fire on an isolated blob --------------------------------
# A blob is a "termination" with no stroke attached: both flanks empty, so a genuine
# end-stop must stay quiet. This separates end-stopping from mere blob saliency.
kf = argmax([l.rho0 for l in lab2])
end_fine = maximum(A2[c-6:c+6, [c-46:c-36; c+36:c+46], kf])
A2b, _ = and_maps(E(blob(N; r=6.5)), bank.meta; forms=(:A2,))
blob_fine = maximum(@view A2b[:, :, kf])
check("A2: line-end ≫ isolated blob (ratio > 3)", end_fine / blob_fine > 3,
      @sprintf("end %.3g vs blob %.3g  → %.1f×", end_fine, blob_fine, end_fine/blob_fine))

# ---- 5. polarity invariance survives the conjunction ------------------------
I = corner(N, π/2)
Ap, _ = and_maps(E(I),  bank.meta; forms=(:A1, :A2, :A3))
Am, _ = and_maps(E(-I), bank.meta; forms=(:A1, :A2, :A3))
d = maximum(abs.(Ap .- Am))
check("polarity invariance preserved (exactly 0)", d == 0, @sprintf("max|ΔA| = %.3e", d))

# ---- 6. ablation actually ablates -------------------------------------------
n0 = size(and_maps(E(I), bank.meta; forms=())[1], 3)
n1 = size(and_maps(E(I), bank.meta; forms=(:A1,))[1], 3)
n3 = size(and_maps(E(I), bank.meta; forms=(:A1, :A2, :A3))[1], 3)
check("forms are switchable", n0 == 0 && n1 == 3 && n3 == 3 + 3 + 2,
      "channels: none=$n0, A1=$n1, A1+A2+A3=$n3")

# ---- 7. informational: junction order ---------------------------------------
# Not a gate. Records what A1 does across ray counts, for the F/f question (F has a
# 3-ray T where f has a 4-ray X).
println()
for (nm, im) in (("straight (2 rays)", bar(N, 0.0)), ("L-corner (2, meeting)", corner(N, π/2)),
                 ("T-junction (3)", tee(N)), ("X-crossing (4)", cross_bars(N)))
    Ai, _ = and_maps(E(im), bank.meta; forms=(:A1,))
    @printf("  [ -- ] %-30s ΣA1 = %.4g\n", nm, sum(Ai))
end

println("="^78)
println(pass[] ? "ALL GATES PASSED — AND layer does what it claims" :
                 "GATE FAILURE — fix before wiring into features")
println("="^78)
