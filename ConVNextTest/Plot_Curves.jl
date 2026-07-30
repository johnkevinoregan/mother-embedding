# ── PLAIN SCRIPT, not a Pluto notebook ──────────────────────────────────────
# `julia --project=.. Plot_Curves.jl`   (after ConvNextReadout.jl has written results/)
#
# Validation R² against epoch for every trained readout head.
#
# The representations here are frozen; the only thing trained is the MLP head, so these curves
# say whether each arm's reported number is a **ceiling** or just where an unconverged run
# happened to be sampled. That distinction is not cosmetic: Phase 9's CNN was described as
# "properly trained" for a whole phase, and its saved history turned out to swing from −0.54 to
# +0.65 mean validation R² between epochs 30 and 35, with best-epoch selection picking a spike
# out of the noise.
#
# Two panels. Left is the mean over properties that have a defined R²; right is `vangle`
# alone, the property the whole comparison turns on.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Serialization, Statistics, Printf, Plots
gr()

const OUT = joinpath(@__DIR__, "results")
const FIG = joinpath(@__DIR__, "figures")
mkpath(FIG)

res = deserialize(joinpath(OUT, "iid.jls"))
haskey(res, :hists) || error("no histories in results — re-run ConvNextReadout.jl after the patch that keeps them")
props = res.props
hists = res.hists
isempty(hists) && error("histories are empty")

nanmean(v) = (u = filter(!isnan, v); isempty(u) ? NaN : mean(u))
vidx = findfirst(==("vangle"), props)

# `polarity` is EXCLUDED from the mean. Our front end is designed to carry no contrast-sign
# information and scores ≈ 0 on it; ConvNeXt scores ≈ 1.0. Averaging that row in makes our arm
# look 0.2 worse overall for succeeding at its design goal, which is the opposite of
# informative. The first version of this figure did exactly that.
keep = [j for j in 1:length(props) if props[j] != "polarity"]

# fixed colours so the same arm is the same colour in both panels
order = ["ours·MLP", "convnext_tiny s4 ·MLP", "convnext_base s4 ·MLP",
         "convnext_tinyrand s4 ·MLP", "convnext_baserand s4 ·MLP"]
cols  = [:black, :dodgerblue, :navy, :orange, :firebrick]
styles = [:solid, :solid, :solid, :dash, :dash]

p1 = plot(xlabel="epoch", ylabel="validation R² (mean, polarity excluded)",
          title="mean over 7 properties (polarity excluded)", legend=:bottomright,
          ylims=(-0.1, 1.0))
p2 = plot(xlabel="epoch", ylabel="validation R² (vangle)",
          title="vangle only", legend=:bottomright, ylims=(-0.1, 1.0))

println("\nfinal-epoch vs best-epoch — a gap means the run was still moving")
@printf("%-28s %8s %8s %8s %8s\n", "arm", "best", "at e", "final", "Δ")
for (k, nm) in enumerate(order)
    haskey(hists, nm) || continue
    V = hists[nm].val; ne = size(V, 1)
    m = [nanmean(V[e, keep]) for e in 1:ne]
    plot!(p1, 1:ne, m, label=nm, c=cols[k], ls=styles[k], lw=2)
    plot!(p2, 1:ne, V[:, vidx], label=nm, c=cols[k], ls=styles[k], lw=2)
    @printf("%-28s %8.3f %8d %8.3f %+8.3f\n", nm, maximum(m), argmax(m), m[end],
            m[end] - maximum(m))
end

plt = plot(p1, p2, layout=(1, 2), size=(1100, 430),
           plot_title="frozen representation + trained MLP head — i.i.d. split")
savefig(plt, joinpath(FIG, "curves_iid.png"))
println("\nwrote ", joinpath(FIG, "curves_iid.png"))
