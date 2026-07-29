# ── PLAIN SCRIPT ────────────────────────────────────────────────────────────
# `julia --project=.. Plot_Phase9.jl` — figures for RESULTS.md, from the serialised runs.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Serialization, Printf, Plots, Statistics

const P = ["curvedness","brokenness","closedness","vangle","arms","thickness","fuzziness","polarity"]
const STRUCT = 1:5          # the geometry rows; 6-8 are the photometric controls

iid   = deserialize(joinpath(@__DIR__, "results", "iid.jls"))
curve = deserialize(joinpath(@__DIR__, "results", "curve.jls"))
blk   = deserialize(joinpath(@__DIR__, "results_nocnn2", "blocks.jls"))

# ── 1. the arms, i.i.d. ─────────────────────────────────────────────────────
arms = ["pixels·linear","pixels·MLP","CNN","ours·linear","ours·MLP"]
cols = [:grey70, :grey45, :steelblue, :firebrick, :darkred]
p1 = plot(size=(1000,420), ylabel="test R²", legend=:topleft, xrotation=20,
          title="Phase 9 — how much of each property is recoverable, i.i.d.",
          titlefontsize=11, ylims=(-0.45, 1.05), bottom_margin=8Plots.mm,
          left_margin=6Plots.mm, grid=true, gridalpha=0.25)
w = 0.15
for (k, a) in enumerate(arms)
    v = [max(iid.R[k, j], -0.45) for j in 1:8]
    bar!(p1, (1:8) .+ (k-3)*w, v; bar_width=w, label=a, c=cols[k], lw=0)
end
plot!(p1, 0.5:8.5, [iid.base[min(round(Int, x+0.5), 8)] for x in 0.5:8.5];
      lt=:steppost, lc=:black, ls=:dash, lw=1.5, label="trivial baseline")
hline!(p1, [0]; lc=:black, lw=1, label="")
xticks!(p1, 1:8, P)
vline!(p1, [5.5]; lc=:grey30, lw=2, ls=:dot, label="")
annotate!(p1, 3.0, 1.0, text("structure", 9, :grey30))
annotate!(p1, 7.0, 1.0, text("photometric controls", 9, :grey30))
savefig(p1, joinpath(@__DIR__, "phase9_arms.png"))

# ── 2. block attribution against the shuffle control ────────────────────────
p2 = plot(size=(1000,400), ylabel="test R²  (linear readout)", legend=:topleft,
          xrotation=20, ylims=(-0.05, 1.05), grid=true, gridalpha=0.25,
          title="Which part of the representation carries each property?",
          titlefontsize=11, bottom_margin=8Plots.mm, left_margin=6Plots.mm)
order = ["orient","lowpass","A1+A2","rays","all·SHUFFLED","all"]
cols2 = [:steelblue, :grey60, :orange, :seagreen, :grey35, :firebrick]
w2 = 0.14
for (k, b) in enumerate(order)
    bar!(p2, (1:8) .+ (k-3.5)*w2, [max(blk[b][j], -0.05) for j in 1:8];
         bar_width=w2, label=b, c=cols2[k], lw=0)
end
xticks!(p2, 1:8, P); hline!(p2, [0]; lc=:black, lw=1, label="")
savefig(p2, joinpath(@__DIR__, "phase9_blocks.png"))

# ── 3. how many training images does it take? ───────────────────────────────
ks = sort(collect(keys(curve)))
panels = Any[]
for j in STRUCT
    pj = plot(xscale=:log10, ylims=(-0.15, 1.03), xlabel="training images",
              ylabel = j == 1 ? "test R²" : "", title=P[j], titlefontsize=9,
              legend = j == 1 ? :bottomright : false, grid=true, gridalpha=0.25,
              xticks=(ks, string.(ks)), xrotation=35)
    plot!(pj, ks, [curve[k][4, j] for k in ks]; lw=2.5, marker=:circle, ms=5,
          c=:firebrick, label="ours·linear")
    plot!(pj, ks, [curve[k][1, j] for k in ks]; lw=2.5, marker=:square, ms=4,
          c=:grey55, label="pixels·linear")
    hline!(pj, [0]; lc=:black, lw=1, label="")
    push!(panels, pj)
end
p3 = plot(panels...; layout=(1, length(STRUCT)), size=(1350, 330),
          plot_title="Sample efficiency — most properties reach 90 % of their ceiling by 500 images",
          plot_titlefontsize=11, bottom_margin=10Plots.mm, left_margin=7Plots.mm)
savefig(p3, joinpath(@__DIR__, "phase9_samples.png"))

println("wrote phase9_arms.png, phase9_blocks.png, phase9_samples.png")
for k in ks
    @printf("k=%-6d ours %s\n", k, join([@sprintf("%.3f", curve[k][4,j]) for j in 1:8], " "))
end
