# ── PLAIN SCRIPT, not a Pluto notebook ──────────────────────────────────────
# `julia --project=.. Plot_Phase10.jl`   (from the P10_FashionMNIST directory, after the main run)
#
# Accuracy-per-epoch for every Phase 10 arm, from `curves.jls`.
#
# WHY THIS EXISTS. Phase 10 originally reported final numbers and kept no curves at all — the only
# per-epoch record anywhere was four sampled lines for the CNN arm in `phase10_full.log`, and
# nothing for the feature arms. A final number cannot tell a converged run from a sample of a
# trajectory, which this project has been caught by before: a CNN reported as "properly trained"
# turned out to swing from −0.54 to +0.65 mean validation R² between epochs 30 and 35, with
# best-epoch-on-validation selecting a spike out of noise.
#
# Solid is held-out test, dotted is the validation slice the reported number is selected on. The
# marker shows which epoch that selection landed on — if it sits on a spike rather than a plateau,
# the number is a sample of noise and should not be quoted without saying so.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Statistics, Printf, Serialization, Plots
gr()

d = deserialize(joinpath(@__DIR__, "curves.jls"))
curves = d.curves
FIG = joinpath(@__DIR__, "figures"); mkpath(FIG)

"One panel per arm: test solid, validation dotted, selected epoch marked."
function panel(name, r)
    ep = 1:length(r.test)
    sel = argmax(r.val)
    lo = min(minimum(r.test), minimum(r.val)); hi = max(maximum(r.test), maximum(r.val))
    pad = max(0.4, 0.10*(hi-lo))
    pl = plot(; title=@sprintf("%s\nbest %.2f %% @ epoch %d", name, r.acc, sel), titlefontsize=7,
              xlabel="epoch", ylabel="accuracy (%)", legend=:bottomright, legendfontsize=5,
              tickfontsize=5, guidefontsize=6, grid=true, gridalpha=0.2,
              ylims=(lo-pad, hi+pad), xlims=(0, length(ep)+1))
    plot!(pl, ep, r.val;  c=:grey45, lw=1.1, ls=:dot,   label="validation (selected on)")
    plot!(pl, ep, r.test; c=:navy,   lw=1.9,            label="held-out test")
    scatter!(pl, [sel], [r.test[sel]]; c=:firebrick, ms=4, msw=0, label="reported epoch")
    pl
end

n = length(curves); cols = 4; rows = cld(n, cols)
fig = plot([panel(k, v) for (k, v) in curves]...; layout=(rows, cols),
           size=(300*cols, 235*rows), left_margin=3Plots.mm, bottom_margin=3Plots.mm,
           plot_title="Phase 10 — Fashion-MNIST: accuracy per epoch, every arm",
           plot_titlefontsize=11)
savefig(fig, joinpath(FIG, "phase10_curves.png"))
println("wrote figures/phase10_curves.png")

# the test curves of the feature arms on one axis, which is what the phase's claim rests on
main = [(k, v) for (k, v) in curves if !occursin("SHUFFLED", k) && !occursin("CNN", k)]
pl = plot(; xlabel="epoch", ylabel="held-out test accuracy (%)", legend=:outerright,
          legendfontsize=5, grid=true, gridalpha=0.2, size=(880, 470), titlefontsize=10,
          title="Phase 10 — held-out accuracy per epoch (feature arms and the pixel calibration)")
for (k, v) in main
    # grid 1 dashed, grid 3 solid: with 11 arms on one axis, colour alone does not separate them
    plot!(pl, 1:length(v.test), v.test; lw=1.7, label=k,
          ls = startswith(k, "g1") ? :dash : :solid)
end
savefig(pl, joinpath(FIG, "phase10_curves_compare.png"))
println("wrote figures/phase10_curves_compare.png")

# ── is the reported number a plateau value or an artefact of where selection landed?
#
# `dev` is the reported accuracy minus the mean of the last five epochs. A large POSITIVE dev is
# the dangerous case: best-epoch-on-validation caught an upward excursion, and the headline
# overstates what the arm actually settles at. A negative dev is harmless — the arm ends better
# than its reported number, so the figure is conservative.
@printf("\n%-34s %8s %8s %7s %9s %8s %s\n",
        "arm", "reported", "plateau", "epoch", "last-5 sd", "dev", "verdict")
for (k, v) in curves
    sel = argmax(v.val); tail = v.test[max(1, end-4):end]
    dev = v.acc - mean(tail); sd = max(std(tail), 0.05)
    verdict = dev > 2sd ? "OVERSTATES" : dev < -2sd ? "conservative" : "converged"
    @printf("%-34s %8.2f %8.2f %7d %9.2f %+8.2f %s\n",
            k, v.acc, mean(tail), sel, std(tail), dev, verdict)
end
