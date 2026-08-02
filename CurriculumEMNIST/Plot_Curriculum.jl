# ── PLAIN SCRIPT ────────────────────────────────────────────────────────────
# `julia --project=.. Plot_Curriculum.jl`
#
# Draws the curves from `curriculum_ours_convnext.jls` (the two FROZEN arms, trained in Julia)
# and from `scratch_runs/*.tsv` (ConvNeXt trained END TO END from random init, in Python).
#
# One figure per arm, each carrying four curves:
#   * test (solid, dark)          held-out accuracy — the thing we care about
#   * current subset (dashed)     the gap above `test` is memorisation, measured
#   * subset 1 (dotted)           decays back toward `test` once training leaves it
#   * control (grey)              60 epochs on subset 1 alone, same steps, no switches
#
# Plus one comparison figure of every switching arm's held-out curve. Controls are left out of
# that one deliberately — eight lines is unreadable, and each control is already drawn beside its
# own arm.
#
# Vertical lines mark the switches. With an i.i.d. partition a learner that generalised rather
# than memorised should cross them without noticing.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Statistics, Printf, Serialization, DelimitedFiles, Plots
gr()

const FIG = joinpath(@__DIR__, "figures"); mkpath(FIG)
const SCR = joinpath(@__DIR__, "scratch_runs")
const PER, NSET = 15, 4

d = deserialize(joinpath(@__DIR__, get(ENV, "CU_SRC", "curriculum_ours_convnext.jls")))
res = Any[r for r in d.results]

"Read one Python run: epoch, set, test, current, set1."
function read_tsv(tag, name)
    p = joinpath(SCR, "$(tag).tsv")
    isfile(p) || return nothing
    M, _ = readdlm(p, '\t'; header=true)
    sw = [e for e in 2:size(M,1) if M[e,2] != M[e-1,2]]
    (name=name, nfeat=0, test=Float64.(M[:,3]), current=Float64.(M[:,4]),
     set1=Float64.(M[:,5]), switches=sw, switching=!isempty(sw))
end

for (tag, name) in (("tiny112_switch", "ConvNeXt-tiny scratch (switching)"),
                    ("tiny112_set1",   "ConvNeXt-tiny scratch (set 1 only)"),
                    ("base224_switch", "ConvNeXt-base scratch (switching)"),
                    ("base224_set1",   "ConvNeXt-base scratch (set 1 only)"))
    r = read_tsv(tag, name); r === nothing || push!(res, r)
end
byname(n) = (i = findfirst(r -> r.name == n, res); i === nothing ? nothing : res[i])

function panel(sw, ctl, title)
    ep = 1:length(sw.test)
    # zoom to the data: on a 0–100 axis every curve is a flat line near the top and the whole
    # point of the figure — the step at each switch — is invisible
    # from epoch 10 on: the from-scratch arms start near chance and climb 45 points in the first
    # few epochs, which on a full-range axis flattens every curve and hides the switches entirely
    w = 10:length(sw.test)
    all_ = vcat(sw.test[w], sw.current[w], sw.set1[w], ctl.test[w])
    lo, hi = minimum(all_), maximum(all_); pad = 0.08*(hi-lo)
    yl = (100*(lo-pad), 100*(hi+pad))
    pl = plot(; xlabel="epoch", ylabel="accuracy (%)", title=title, titlefontsize=9,
              legend=:bottomright, legendfontsize=6, grid=true, gridalpha=0.2, size=(660, 450),
              ylims=yl, xlims=(0, length(ep)+1))
    for s in sw.switches; vline!(pl, [s-0.5]; c=:grey60, ls=:dash, lw=1.0, label=""); end
    plot!(pl, ep, 100 .* ctl.test; c=:grey55, lw=1.6, label="control: subset 1 only, no switch")
    plot!(pl, ep, 100 .* sw.current; c=:darkorange, lw=1.5, ls=:dash,
          label="accuracy on the subset being trained on")
    plot!(pl, ep, 100 .* sw.set1; c=:seagreen, lw=1.5, ls=:dot, label="accuracy on subset 1")
    plot!(pl, ep, 100 .* sw.test; c=:navy, lw=2.4, label="held-out test accuracy")
    for (i, s) in enumerate(sw.switches)
        annotate!(pl, s+0.4, yl[2]-0.04*(yl[2]-yl[1]), text("subset $(i+1)", 6, :grey35, :left))
    end
    pl
end

PANELS = [("ours",         "ours (switching)",  "ours (set 1 only)",
           "our front end, frozen (381 features)"),
          ("convnext",     "ConvNeXt (switching)", "ConvNeXt (set 1 only)",
           "frozen ImageNet ConvNeXt-base, stage 4 (1024 features)"),
          ("scratch_tiny", "ConvNeXt-tiny scratch (switching)", "ConvNeXt-tiny scratch (set 1 only)",
           "ConvNeXt-tiny from random init, end to end (27.9 M params, 112 px)"),
          ("scratch_base", "ConvNeXt-base scratch (switching)", "ConvNeXt-base scratch (set 1 only)",
           "ConvNeXt-base from random init, end to end (87.6 M params, 224 px)")]

for (tag, a, b, lbl) in PANELS
    sw, ctl = byname(a), byname(b)
    (sw === nothing || ctl === nothing) && continue
    savefig(panel(sw, ctl, @sprintf("EMNIST balanced — %s\ntraining subset switched every %d epochs",
                                    lbl, PER)), joinpath(FIG, "curriculum_$(tag).png"))
    @printf("wrote curriculum_%s.png\n", tag)
end

# ── every switching arm's held-out curve on one axis
COMPARE = [("ours (switching)", "ours, frozen (381)", :navy, :solid, 2.6),
           ("ConvNeXt (switching)", "ConvNeXt-base, frozen ImageNet (1024)", :firebrick, :solid, 2.2),
           ("ConvNeXt-tiny scratch (switching)", "ConvNeXt-tiny, trained from scratch", :seagreen, :dash, 1.9),
           ("ConvNeXt-base scratch (switching)", "ConvNeXt-base, trained from scratch", :darkorange, :dash, 1.9)]
have = [(byname(n), l, c, s, w) for (n, l, c, s, w) in COMPARE if byname(n) !== nothing]
allc = vcat([r.test[10:end] for (r, _, _, _, _) in have]...)   # see the note in `panel`
lo, hi = minimum(allc), maximum(allc); pad = 0.08*(hi-lo)
n_ep = length(have[1][1].test)
pl = plot(; xlabel="epoch", ylabel="held-out test accuracy (%)", legend=:bottomright,
          legendfontsize=6, grid=true, gridalpha=0.2, size=(720, 480),
          ylims=(100*(lo-pad), 100*(hi+pad)), xlims=(0, n_ep+1), titlefontsize=9,
          title="EMNIST balanced — held-out accuracy across training-subset switches\n(axis set by epochs 10+; no-switch controls are in the per-arm figures)")
for s in have[1][1].switches; vline!(pl, [s-0.5]; c=:grey60, ls=:dash, lw=1.0, label=""); end
for (r, l, c, s, w) in have
    plot!(pl, 1:length(r.test), 100 .* r.test; c=c, ls=s, lw=w, label=l)
end
savefig(pl, joinpath(FIG, "curriculum_compare.png"))
println("wrote curriculum_compare.png")

# ── the numbers behind the figures
@printf("\n%-36s %6s %8s %8s %8s %8s\n", "arm", "nfeat", "ep15", "ep60", "best", "mem gap")
for r in res
    @printf("%-36s %6s %8.2f %8.2f %8.2f %8.2f\n", r.name, r.nfeat == 0 ? "—" : string(r.nfeat),
            100r.test[15], 100r.test[end], 100maximum(r.test), 100*(r.current[end]-r.test[end]))
end
println("\nchange in held-out accuracy across each switch (epoch before → epoch after):")
for r in res
    r.switching || continue
    @printf("  %-36s", r.name)
    for s in r.switches; @printf("  %+.2f", 100*(r.test[s]-r.test[s-1])); end
    println()
end
println("\nwithin-block epoch-to-epoch noise floor (sd of held-out change):")
for r in res
    r.switching || continue
    dif = [100*(r.test[e]-r.test[e-1]) for e in 2:length(r.test) if !(e in r.switches)]
    @printf("  %-36s  sd %.2f   range %+.2f to %+.2f\n", r.name, std(dif), minimum(dif), maximum(dif))
end
println("\nsubset-1 advantage over held-out at the end of each block:")
for r in res
    r.switching || continue
    @printf("  %-36s", r.name)
    for k in 1:NSET; @printf("  %6.2f", 100*(r.set1[k*PER]-r.test[k*PER])); end
    println()
end
println("\nfigures in $FIG")
