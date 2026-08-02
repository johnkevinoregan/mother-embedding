# ── PLAIN SCRIPT ────────────────────────────────────────────────────────────
# `julia --project=.. Plot_Curriculum.jl`
#
# Draws the curves from `curriculum.jls`. One figure per arm, plus a direct comparison.
#
# Each arm's panel carries three curves and the control:
#   * test (solid, dark)          held-out accuracy — the thing we care about
#   * current subset (dashed)     the gap above `test` is memorisation, measured
#   * subset 1 (dotted)           decays back toward `test` once training leaves it
#   * control (grey)              60 epochs on subset 1 alone, same steps, no switches
#
# Vertical lines mark the switches. With an i.i.d. partition a learner that generalised rather
# than memorised should cross them without noticing.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Statistics, Printf, Serialization, Plots
gr()

const FIG = joinpath(@__DIR__, "figures"); mkpath(FIG)
const SRC = get(ENV, "CU_SRC", "curriculum_ours_convnext.jls")
d = deserialize(joinpath(@__DIR__, SRC))
res, PER, NSET = d.results, d.per, d.nset
byname(n) = (i = findfirst(r -> r.name == n, res); i === nothing ? nothing : res[i])

function panel(sw, ctl, title)
    ep = 1:length(sw.test)
    # zoom to the data: on a 0-100 axis every curve here is a flat line near the top and the
    # whole point of the figure -- the step at each switch -- is invisible
    all_ = vcat(sw.test, sw.current, sw.set1, ctl.test)
    lo, hi = minimum(all_), maximum(all_); pad = 0.08*(hi-lo)
    yl = (100*(lo-pad), 100*(hi+pad))
    pl = plot(; xlabel="epoch", ylabel="accuracy (%)", title=title, titlefontsize=9,
              legend=:bottomright, legendfontsize=6, grid=true, gridalpha=0.2, size=(660, 450),
              ylims=yl, xlims=(0, length(ep)+1))
    for s in sw.switches
        vline!(pl, [s - 0.5]; c=:grey60, ls=:dash, lw=1.0, label="")
    end
    plot!(pl, ep, 100 .* ctl.test; c=:grey55, lw=1.6, label="control: subset 1 only, no switch")
    plot!(pl, ep, 100 .* sw.current; c=:darkorange, lw=1.5, ls=:dash,
          label="accuracy on the subset being trained on")
    plot!(pl, ep, 100 .* sw.set1; c=:seagreen, lw=1.5, ls=:dot, label="accuracy on subset 1")
    plot!(pl, ep, 100 .* sw.test; c=:navy, lw=2.4, label="held-out test accuracy")
    for (i, s) in enumerate(sw.switches)
        annotate!(pl, s + 0.4, yl[2] - 0.04*(yl[2]-yl[1]),
                  text("subset $(i+1)", 6, :grey35, :left))
    end
    pl
end

for (tag, a, b) in (("ours", "ours (switching)", "ours (set 1 only)"),
                    ("convnext", "ConvNeXt (switching)", "ConvNeXt (set 1 only)"))
    sw, ctl = byname(a), byname(b)
    (sw === nothing || ctl === nothing) && continue
    lbl = tag == "ours" ? "our front end" : "frozen ConvNeXt stage 4"
    fig = panel(sw, ctl, @sprintf("EMNIST balanced — %s (%d features)\ntraining subset switched every %d epochs",
                                  lbl, sw.nfeat, PER))
    savefig(fig, joinpath(FIG, "curriculum_$(tag).png"))
    @printf("wrote curriculum_%s.png\n", tag)
end

# direct comparison of the two test curves
sw1, sw2 = byname("ours (switching)"), byname("ConvNeXt (switching)")
c1, c2   = byname("ours (set 1 only)"), byname("ConvNeXt (set 1 only)")
if sw1 !== nothing && sw2 !== nothing
ep = 1:length(sw1.test)
allc = vcat(sw1.test, sw2.test, c1.test, c2.test)
lo, hi = minimum(allc), maximum(allc); pad = 0.08*(hi-lo)
pl = plot(; xlabel="epoch", ylabel="held-out test accuracy (%)", legend=:bottomright,
          legendfontsize=6, grid=true, gridalpha=0.2, size=(700, 470),
          ylims=(100*(lo-pad), 100*(hi+pad)), xlims=(0, length(sw1.test)+1),
          title="EMNIST balanced — held-out accuracy across training-subset switches",
          titlefontsize=9)
for s in sw1.switches; vline!(pl, [s - 0.5]; c=:grey60, ls=:dash, lw=1.0, label=""); end
plot!(pl, ep, 100 .* c1.test;  c=:steelblue, lw=1.3, ls=:dot, label="ours, subset 1 only")
plot!(pl, ep, 100 .* c2.test;  c=:indianred, lw=1.3, ls=:dot, label="ConvNeXt, subset 1 only")
plot!(pl, ep, 100 .* sw1.test; c=:navy,      lw=2.4, label="ours ($(sw1.nfeat) features), switching")
plot!(pl, ep, 100 .* sw2.test; c=:firebrick, lw=2.4, label="ConvNeXt (1024), switching")
savefig(pl, joinpath(FIG, "curriculum_compare.png"))
println("wrote curriculum_compare.png")
end

# ── the numbers behind the figures
@printf("\n%-24s %6s %8s %8s %8s %8s\n", "arm", "nfeat", "ep15", "ep60", "best", "mem gap")
for r in res
    @printf("%-24s %6d %8.2f %8.2f %8.2f %8.2f\n", r.name, r.nfeat, 100r.test[15],
            100r.test[end], 100maximum(r.test), 100*(r.current[end] - r.test[end]))
end
println("\nchange in held-out accuracy across each switch (epoch before → epoch after):")
for r in res
    r.switching || continue
    @printf("  %-24s", r.name)
    for s in r.switches
        @printf("  %+.2f", 100*(r.test[s] - r.test[s-1]))
    end
    println()
end
println("\nsubset-1 accuracy at the end of each block (memorised → released):")
for r in res
    r.switching || continue
    @printf("  %-24s", r.name)
    for k in 1:NSET
        @printf("  %6.2f", 100*r.set1[k*PER])
    end
    println()
end
println("\nfigures in $FIG")
