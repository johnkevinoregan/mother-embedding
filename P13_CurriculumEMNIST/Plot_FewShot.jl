# ── PLAIN SCRIPT ────────────────────────────────────────────────────────────
# `julia --project=.. Plot_FewShot.jl`
#
# Ten-way few-shot accuracy on EMNIST classes the from-scratch network never saw, against the
# number of support examples per class. Shaded bands are 95% confidence intervals over episodes.
#
# The dashed line at 10% is chance for a 10-way problem.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Statistics, Printf, Serialization, Plots
gr()

const FIG = joinpath(@__DIR__, "figures"); mkpath(FIG)
d = deserialize(joinpath(@__DIR__, "fewshot", "fewshot_results.jls"))
R, SHOTS, held, NEPI = d.results, d.shots, d.held, d.nepi

# EMNIST balanced label order, 1-based: 1-10 digits 0-9, 11-36 A-Z, 37-47 a b d e f g h n q r t
const GLYPH = vcat(string.(0:9), string.('A':'Z'), ["a","b","d","e","f","g","h","n","q","r","t"])
const HSTR = join([GLYPH[c] for c in held], " ")

ORDER = [("ours, frozen (381)", :navy, :solid),
         ("ConvNeXt-tiny scratch on 37 classes (768)", :darkorange, :dash),
         ("ConvNeXt-base frozen ImageNet (1024)", :firebrick, :dash),
         ("raw pixels (784)", :grey55, :dot)]

pl = plot(; xlabel="support examples per class (k)", ylabel="10-way accuracy (%)",
          xscale=:log10, xticks=(collect(SHOTS), string.(SHOTS)), legend=:bottomright,
          legendfontsize=6, grid=true, gridalpha=0.2, size=(680, 460), titlefontsize=9,
          title="Few-shot transfer to 10 EMNIST classes the trained network never saw\n" *
                "held out:  $HSTR    ($NEPI episodes, band = 95% CI)")
hline!(pl, [10]; c=:black, ls=:dashdot, lw=0.8, label="chance (10-way)")
for (name, c, s) in ORDER
    haskey(R, name) || continue
    m  = [100*mean(R[name][k]) for k in SHOTS]
    ci = [100*1.96*std(R[name][k])/sqrt(length(R[name][k])) for k in SHOTS]
    plot!(pl, collect(SHOTS), m; ribbon=ci, fillalpha=0.15, c=c, ls=s, lw=2.2,
          marker=:circle, ms=3.5, msw=0, label=name)
end
savefig(pl, joinpath(FIG, "fewshot.png"))
println("wrote fewshot.png")

@printf("\n%-44s", "representation"); for k in SHOTS; @printf("%11s", "$(k)-shot"); end; println()
for (name, _, _) in ORDER
    haskey(R, name) || continue
    @printf("%-44s", name)
    for k in SHOTS; @printf("  %6.2f±%.2f", 100*mean(R[name][k]),
                            100*1.96*std(R[name][k])/sqrt(length(R[name][k]))); end
    println()
end
