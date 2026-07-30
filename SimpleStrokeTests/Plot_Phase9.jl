# ── PLAIN SCRIPT ────────────────────────────────────────────────────────────
# `julia --project=.. Plot_Phase9.jl` — figures for RESULTS.md, from the serialised runs.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Serialization, Printf, Plots, Statistics

const P = ["curvedness","brokenness","closedness","vangle","arms","thickness","fuzziness","polarity"]
const STRUCT = 1:5          # the geometry rows; 6-8 are the photometric controls

iid   = deserialize(joinpath(@__DIR__, "results_canon3", "iid.jls"))
curve = deserialize(joinpath(@__DIR__, "results_canon3", "curve.jls"))
blk_raw = deserialize(joinpath(@__DIR__, "results_canon3", "blocks.jls"))
# blocks.jls became a NamedTuple (battr, shuf) when the shuffle spread started being tracked
blk = blk_raw isa NamedTuple ? blk_raw.battr : blk_raw
hist  = deserialize(joinpath(@__DIR__, "results_canon3", "history.jls"))

# ── 1. the arms, i.i.d. ─────────────────────────────────────────────────────
arms = ["pixels·linear","pixels·MLP","CNN","ours·linear","ours·MLP"]
cols = [:grey70, :grey45, :steelblue, :firebrick, :darkred]
p1 = plot(size=(1000,420), ylabel="test R²", legend=:topleft, xrotation=20,
          title="Phase 9 — how much of each property is recoverable, i.i.d. (CNN: full-resolution, 60 epochs, GPU)",
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
p2 = plot(size=(1150,430), ylabel="test R²  (linear readout)", legend=:topleft,
          xrotation=20, ylims=(-0.05, 1.05), grid=true, gridalpha=0.25,
          title="Which part of the representation carries each property?  (gap from grey to red = the conjunction layer)",
          titlefontsize=10, bottom_margin=8Plots.mm, left_margin=6Plots.mm)
order = ["orient","A1+A2","rays","all·SHUFFLED","all·noRAYS","all·noA","all"]
# the control keeps orient+lowpass intact and shuffles only A1/A2/rays, so its bar should
# land on `orient` — the gap from it to `all` is the conjunction layer's real contribution
labels = ["orient (135)","A1+A2 (54)","rays (81)",
          "CONTROL: A+rays shuffled","all, rays shuffled","all, A shuffled","all 279"]
cols2 = [:steelblue, :orange, :seagreen, :grey35, :grey55, :grey75, :firebrick]
w2 = 0.12
for (k, b) in enumerate(order)
    bar!(p2, (1:8) .+ (k-4)*w2, [max(blk[b][j], -0.05) for j in 1:8];
         bar_width=w2, label=labels[k], c=cols2[k], lw=0)
end
xticks!(p2, 1:8, P); hline!(p2, [0]; lc=:black, lw=1, label="")
savefig(p2, joinpath(@__DIR__, "phase9_blocks.png"))

# ── 3. how many training images does it take? ───────────────────────────────
ks = sort(collect(keys(curve)))
panels = Any[]
for j in STRUCT
    pj = plot(xscale=:log10, ylims=(-0.15, 1.03), xlabel="training images",
              ylabel = j == 1 ? "test R²" : "", title=P[j], titlefontsize=9,
              legend = j == 2 ? :topleft : false, grid=true, gridalpha=0.25,
              xticks=(ks, string.(ks)), xrotation=35)
    plot!(pj, ks, [curve[k][4, j] for k in ks]; lw=2.5, marker=:circle, ms=5,
          c=:firebrick, label="ours·linear")
    plot!(pj, ks, [curve[k][3, j] for k in ks]; lw=2.5, marker=:diamond, ms=5,
          c=:steelblue, label="CNN (learned)")
    plot!(pj, ks, [curve[k][1, j] for k in ks]; lw=2.5, marker=:square, ms=4,
          c=:grey55, label="pixels·linear")
    hline!(pj, [0]; lc=:black, lw=1, label="")
    push!(panels, pj)
end
p3 = plot(panels...; layout=(1, length(STRUCT)), size=(1350, 330),
          plot_title="Sample efficiency — a fixed representation with a linear readout, against a CNN learning its own",
          plot_titlefontsize=11, bottom_margin=10Plots.mm, left_margin=7Plots.mm)
savefig(p3, joinpath(@__DIR__, "phase9_samples.png"))

println("wrote phase9_arms.png, phase9_blocks.png, phase9_samples.png")
for k in ks
    @printf("k=%-6d ours %s\n", k, join([@sprintf("%.3f", curve[k][4,j]) for j in 1:8], " "))
end

# ── 4. learning curves — why a single final number is not enough ────────────
# The CNN's validation score swings by more than 3 R² between adjacent epochs while its
# training loss falls smoothly. Best-epoch selection protects the reported number, but a
# score picked off a curve like this is weaker evidence than one picked off a plateau.
nanm(v) = (u = filter(!isnan, v); isempty(u) ? NaN : mean(u))
p4 = plot(size=(1000, 380), xlabel="epoch", ylabel="validation R², mean over properties",
          legend=:bottomright, ylims=(-1.2, 1.0), grid=true, gridalpha=0.25,
          title="Training stability, i.i.d. split", titlefontsize=11,
          bottom_margin=6Plots.mm, left_margin=6Plots.mm)
for (k, c) in (("iid/CNN", :steelblue), ("iid/ours·MLP", :firebrick), ("iid/pixels·MLP", :grey55))
    haskey(hist, k) || continue
    h = hist[k]; v = [nanm(h.val[e, :]) for e in 1:size(h.val, 1)]
    plot!(p4, 1:length(v), max.(v, -1.2); lw=2, c=c, label=split(k, "/")[2])
end
hline!(p4, [0]; lc=:black, lw=1, label="")
savefig(p4, joinpath(@__DIR__, "phase9_learning.png"))

# ── 5. per-epoch, per-property, per-arm ─────────────────────────────────────
# The summary above averages over eight properties, which can hide a model that is improving
# on one row while collapsing on another. One panel per trained arm, one line per property,
# plus the training loss on its own axis so smooth-loss-with-erratic-validation is visible
# as the single picture it is.
trained = [k for k in ("iid/pixels·MLP", "iid/CNN", "iid/ours·MLP") if haskey(hist, k)]
pans = Any[]
for k in trained
    h = hist[k]; ne = size(h.val, 1)
    pj = plot(ylims=(-1.0, 1.05), xlabel="epoch", title=split(k, "/")[2], titlefontsize=10,
              ylabel = k == trained[1] ? "validation R²" : "", grid=true, gridalpha=0.25,
              legend = k == trained[end] ? :bottomright : false, legendfontsize=6)
    for (j, nm) in enumerate(P)
        plot!(pj, 1:ne, max.(h.val[:, j], -1.0); lw=1.6, label=nm)
    end
    hline!(pj, [0]; lc=:black, lw=1, label="")
    # training loss, rescaled onto the same axis purely so the shapes can be compared
    L = h.loss ./ maximum(h.loss)
    plot!(pj, 1:ne, L .- 1.0; lw=2.5, ls=:dash, lc=:black, label="train loss (rescaled)")
    push!(pans, pj)
end
p5 = plot(pans...; layout=(1, length(pans)), size=(460*length(pans), 400),
          plot_title="Per-epoch validation R² by property — i.i.d. split",
          plot_titlefontsize=11, bottom_margin=9Plots.mm, left_margin=7Plots.mm)
savefig(p5, joinpath(@__DIR__, "phase9_learning_detail.png"))

# ── 6. the same, per split, so transfer failures are visible during training ─
splits = [s for s in ("iid", "extrap_polarity", "extrap_fuzziness", "extrap_thickness")
          if haskey(hist, "$s/CNN")]
p6 = plot(size=(1000, 380), xlabel="epoch", ylabel="validation R², mean over properties",
          legend=:bottomright, ylims=(-1.2, 1.0), grid=true, gridalpha=0.25,
          title="CNN validation by split — the instability is not specific to one dataset",
          titlefontsize=11, bottom_margin=6Plots.mm, left_margin=6Plots.mm)
for (s, c) in zip(splits, [:steelblue, :firebrick, :seagreen, :orange])
    h = hist["$s/CNN"]; v = [nanm(h.val[e, :]) for e in 1:size(h.val, 1)]
    plot!(p6, 1:length(v), max.(v, -1.2); lw=2, c=c, label=s)
end
hline!(p6, [0]; lc=:black, lw=1, label="")
savefig(p6, joinpath(@__DIR__, "phase9_learning_splits.png"))
println("wrote phase9_learning_detail.png, phase9_learning_splits.png")
println("wrote phase9_learning.png")

# ── 7. pooling grid sweep ───────────────────────────────────────────────────
# The two readouts want opposite things. A linear map needs the grid to express
# position-dependent combinations, so it peaks at 3x3. An MLP can build those itself, and
# would rather have the translation invariance that global pooling gives for free — so it
# improves monotonically as the grid coarsens, and 31 globally-pooled numbers beat 775.
gr_cols = [31, 124, 279, 496, 775]
# canonical run, grids 1..5, current front end
gr_lin = Dict("curvedness"=>[0.685,0.677,0.694,0.674,0.664], "brokenness"=>[0.224,0.260,0.318,0.304,0.307],
              "vangle"=>[0.543,0.562,0.580,0.556,0.544], "arms"=>[0.729,0.775,0.848,0.840,0.836])
gr_mlp = Dict("curvedness"=>[0.925,0.872,0.835,0.795,0.764], "brokenness"=>[0.737,0.637,0.592,0.526,0.560],
              "vangle"=>[0.938,0.883,0.835,0.800,0.772], "arms"=>[0.954,0.923,0.905,0.890,0.895])
pans7 = Any[]
for nm in ("curvedness","brokenness","vangle","arms")
    pj = plot(xticks=(1:5, ["1×1\n31","2×2\n124","3×3\n279","4×4\n496","5×5\n775"]),
              title=nm, titlefontsize=10, ylims=(0.15, 1.0), grid=true, gridalpha=0.25,
              xlabel="pooling grid / columns", ylabel = nm=="curvedness" ? "test R²" : "",
              legend = nm=="brokenness" ? :bottomright : false)
    plot!(pj, 1:5, gr_mlp[nm]; lw=2.5, marker=:circle, ms=5, c=:firebrick, label="ours·MLP")
    plot!(pj, 1:5, gr_lin[nm]; lw=2.5, marker=:square, ms=4, c=:steelblue, label="ours·linear")
    push!(pans7, pj)
end
p7 = plot(pans7...; layout=(1,4), size=(1250, 340), bottom_margin=11Plots.mm, left_margin=7Plots.mm,
          plot_title="Pooling grid: a linear readout wants the grid, a nonlinear one wants the invariance",
          plot_titlefontsize=11)
savefig(p7, joinpath(@__DIR__, "phase9_grid.png"))
println("wrote phase9_grid.png")
