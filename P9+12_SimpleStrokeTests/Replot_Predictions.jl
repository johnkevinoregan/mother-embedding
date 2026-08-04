# ── PLAIN SCRIPT, not a Pluto notebook ──────────────────────────────────────
# `julia --project=.. Replot_Predictions.jl`   (from the P9+12_SimpleStrokeTests directory)
#
# Redraws the predicted-vs-true scatters from `figures_predictions/predictions.jls` with a
# **binned median and interquartile range** overlaid, and a line joining the medians.
#
# Why the overlay earns its place. A cloud of 4,000 translucent points shows the spread but not
# the central tendency — and the two failures this project cares about look alike in a raw
# scatter. A readout that is *unbiased but imprecise* gives a wide cloud whose binned medians sit
# on the diagonal; one that is *biased or saturating* gives medians that bend away from it. The
# IQR bars then say how much of the width is genuine spread rather than a few outliers.
#
# Six bins of equal width across the observed range of the true value, except `closedness`, which
# takes only two values and gets one bin each. Bins with fewer than 20 points are dropped rather
# than drawn, because a median over a handful of points is noise wearing a marker.
#
# NOTE on `vangle`: its true values carry a large point mass at 180° (the no-kink case), so with
# equal-width bins the top bin holds far more images than the rest. That is a property of the
# stimulus set, not of the binning, and it is visible in the figure as the dense vertical stripe.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Statistics, Printf, Serialization, Plots
gr()

const FIG   = joinpath(@__DIR__, "figures_predictions")
const MINN  = 20        # a bin needs this many points before its median is drawn

d = deserialize(joinpath(FIG, "predictions.jls"))
preds, Yte, PROPS, SNAPS = d.preds, d.Yte, d.props, d.snaps
const ARMS = ["our CNN (end to end)", "our features (31)", "ours + fine λ=8 (41)",
              "ours + spatial max (40)", "ours + λ=8 + smax (53)", "frozen ConvNeXt (1024)"]

"""
Binned median and quartiles of `ŷ` against `y`.

Returns bin centres, medians, and the distances down to Q1 and up to Q3 — the form Plots wants
for asymmetric error bars, since the quartiles are not symmetric about the median in general.
"""
function binned(y, ŷ; nbins=6)
    u = sort(unique(y))
    edges = length(u) <= 2 ? nothing : range(minimum(y), maximum(y); length=nbins+1)
    ctr = Float64[]; med = Float64[]; lo = Float64[]; hi = Float64[]
    groups = if edges === nothing
        [(v, findall(==(v), y)) for v in u]
    else
        [(0.5*(edges[b]+edges[b+1]),
          findall(i -> (edges[b] <= y[i] < edges[b+1]) ||
                       (b == nbins && y[i] == edges[end]), eachindex(y)))
         for b in 1:nbins]
    end
    for (c, idx) in groups
        length(idx) < MINN && continue
        q = quantile(ŷ[idx], [0.25, 0.5, 0.75])
        push!(ctr, c); push!(med, q[2]); push!(lo, q[2]-q[1]); push!(hi, q[3]-q[2])
    end
    ctr, med, lo, hi
end

for p in [q for q in PROPS if q != "polarity"]
    j = findfirst(==(p), PROPS)
    y = Float64.(Yte[:, j])
    lo_, hi_ = extrema(y); pad = 0.05*(hi_-lo_); ax = (lo_-pad, hi_+pad)
    nb = length(unique(y)) <= 2 ? 2 : 6
    panels = []
    for a in ARMS, e in SNAPS
        ŷ = Float64.(preds[a][e][:, j])
        r = 1 - sum(abs2, ŷ .- y)/sum(abs2, y .- mean(y))
        pl = scatter(y, ŷ; ms=1.1, msw=0, alpha=0.07, c=:steelblue, legend=false,
                     xlims=ax, ylims=ax, aspect_ratio=:equal,
                     title=@sprintf("%s\ne%d   R²=%.3f", a, e, r), titlefontsize=6,
                     tickfontsize=5, grid=false)
        plot!(pl, [ax[1], ax[2]], [ax[1], ax[2]]; c=:black, lw=0.8, ls=:dash)
        c, m, dl, dh = binned(y, ŷ; nbins=nb)
        if !isempty(c)
            plot!(pl, c, m; yerror=(dl, dh), c=:firebrick, lw=1.6, marker=:circle, ms=3.2,
                  msc=:firebrick, msw=0.8, markerstrokecolor=:firebrick)
        end
        push!(panels, pl)
    end
    fig = plot(panels...; layout=(length(ARMS), length(SNAPS)),
               size=(230*length(SNAPS), 250*length(ARMS)),
               plot_title="$p — predicted (y) vs true (x);  red = binned median, bars = IQR",
               plot_titlefontsize=10, left_margin=3Plots.mm, bottom_margin=3Plots.mm)
    savefig(fig, joinpath(FIG, "pred_$(p).png"))
    @printf("wrote pred_%s.png  (%d bins)\n", p, nb)
end
println("\nredrawn in $FIG")
