# ── PLAIN SCRIPT ────────────────────────────────────────────────────────────
# `julia --project=.. Preview_Contours.jl`
#
# Renders the contact sheet and audits the dataset before any model is trained. Phase 8
# failed because a cue nobody had measured turned out to be diagnostic, so the cues that
# could substitute for structure are measured here first.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Statistics, Printf, Random, LinearAlgebra, Plots
include(joinpath(@__DIR__, "Contours.module.jl"))
using .Contours

const N = 112

# ── contact sheet, organised by event so the sheet is readable ──────────────
println("="^76); println("Contact sheet"); println("="^76)
rng = MersenneTwister(20260729)
NCOL = 10          # random samples per event row
NSW  = 6           # steps per sweep row: six well-separated values read better than ten
                   # near-identical ones, which is why the fuzziness range looked flat
tile(img, ttl) = heatmap(img; c=:grays, clims=(0,1), axis=false, ticks=false,
                         colorbar=false, yflip=true, aspect_ratio=1,
                         title=ttl, titlefontsize=7, titlelocation=:left)

# ── random samples, one row per event ───────────────────────────────────────
rows = [(:none, "none:"), (:gap, "gap:"), (:kink, "kink:"), (:tee, "tee:"), (:cross, "cross:")]
panels = Any[]
for (ev, name) in rows, k in 1:NCOL
    p = sample_params(rng; event=ev); img, v, _ = stimulus(p, rng; N=N)
    # every tile carries its own event name. Putting it only on the first tile of the row
    # made "no event" read as a property of that one image rather than a row heading.
    cl = v[3] > 0.5 ? " ○closed" : ""
    ttl = if ev === :kink;  @sprintf("%s %.0f° %s%s", name, v[4], band_of(v[4]), cl)
          elseif ev === :gap; @sprintf("%s brk %.2f%s", name, v[2], cl)
          else @sprintf("%s crv %.2f%s", name, v[1], cl) end
    push!(panels, tile(img, ttl))
end
plot(panels...; layout=(length(rows), NCOL), size=(146NCOL, 150length(rows)),
     plot_title="SimpleStrokeTests — 10 random samples per event type",
     plot_titlefontsize=12)
savefig(joinpath(@__DIR__, "contactsheet.png")); println("wrote contactsheet.png")

# ── sweeps: one parameter moving, everything else pinned ────────────────────
# Random samples cannot show what a single parameter does, because every other parameter
# moves at the same time. These rows fix the geometry, the rotation and the placement, and
# vary one thing across the row.
base = respec(sample_params(MersenneTwister(11); event=:kink), kappa=0.0, turn=0.0,
            arclen=78.0, vturn=deg2rad(90), w=7.0, ramp=1.0, amp=0.85, pol=1, bg=0.5)
GEOM(seed) = MersenneTwister(seed)
sw = Any[]
for r in range(0.8, 22.0; length=NSW)          # edge fuzziness: step edge → heavy blur
    push!(sw, tile(render_params(respec(base; ramp=r), GEOM(3); N=N, rot=0.6, at=(56.0,56.0)),
                   @sprintf("fuzziness %.1f px", r)))
end
for w in range(3.0, 25.0; length=NSW)          # stroke thickness
    push!(sw, tile(render_params(respec(base; w=w), GEOM(3); N=N, rot=0.6, at=(56.0,56.0)),
                   @sprintf("thickness %.1f px", w)))
end
for g in range(1.0, 30.0; length=NSW)          # gap, from a nick to a clear break
    q = respec(base; event=:gap, gap=g)
    push!(sw, tile(render_params(q, GEOM(3); N=N, rot=0.6, at=(56.0,56.0)),
                   @sprintf("gap %.0f px  brk %.2f", g, targets_of(q)[1][2])))
end
for a in range(20, 180; length=NSW)            # vertex angle: 180 = straight through
    q = respec(base; vturn=deg2rad(180 - a))
    push!(sw, tile(render_params(q, GEOM(3); N=N, rot=0.6, at=(56.0,56.0)),
                   @sprintf("vertex %.0f°  %s", a, band_of(a))))
end
for (i, pol) in enumerate(vcat(fill(1, NSW÷2), fill(-1, NSW÷2)))   # polarity × contrast
    amp = 0.18 + 0.82*((i-1) % (NSW÷2))/(NSW÷2 - 1)
    push!(sw, tile(render_params(respec(base; pol=pol, amp=amp), GEOM(3); N=N, rot=0.6,
                                 at=(56.0,56.0)),
                   @sprintf("%s  amp %.2f", pol == 1 ? "light" : "dark", amp)))
end
plot(sw...; layout=(5, NSW), size=(168NSW, 172*5),
     plot_title="one parameter swept, everything else pinned",
     plot_titlefontsize=12)
savefig(joinpath(@__DIR__, "sweeps.png")); println("wrote sweeps.png")

# ── the edge profile itself ─────────────────────────────────────────────────
# The sweep row shows fuzziness qualitatively; this shows what the ramp actually does to
# the intensity, which is what the Gabor bank sees.
pr = plot(xlabel="pixels from the stroke centre", ylabel="intensity", legend=:topright,
          size=(760, 330), title="edge profile at four fuzziness values (stroke width 7 px, peak normalised)",
          titlefontsize=10, left_margin=5Plots.mm, bottom_margin=5Plots.mm)
for r in (0.8, 5.0, 12.0, 22.0)
    # a *straight* stroke, laid vertically, so one image row is a clean cross-section.
    # Cutting a kinked stimulus (the first version) missed the stroke entirely at the sharp
    # settings and only the widest ramp reached the sampled row.
    img = render_params(respec(base; event=:none, ramp=r, noise=0.0), GEOM(3);
                        N=N, rot=π/2, at=(56.0, 56.0))
    plot!(pr, -26:26, img[56, 30:82]; lw=2.2, marker=:circle, ms=3,
          label=@sprintf("fuzziness %.1f px", r))
end
vline!(pr, [-3.5, 3.5]; ls=:dash, lc=:black, lw=1, label="nominal stroke edge")
savefig(pr, joinpath(@__DIR__, "edge_profiles.png"))
println("wrote edge_profiles.png")

# ── audit ───────────────────────────────────────────────────────────────────
imgs, Y, M, ps = contour_batch(3000, 1; N=N)

println("\n" * "="^76)
println("Target distributions"); println("="^76)
@printf("\n%-12s %8s %8s %8s %8s %8s\n", "property", "min", "mean", "max", "sd", "defined")
for (j, nm) in enumerate(PROPS)
    v = Y[M[:,j], j]
    @printf("%-12s %8.3f %8.3f %8.3f %8.3f %7.0f%%\n", String(nm),
            minimum(v), mean(v), maximum(v), std(v), 100mean(M[:,j]))
end

println("\n" * "="^76)
println("Target correlations — the couplings we could not remove by construction")
println("="^76)
@printf("\n%-12s", "")
for nm in PROPS; @printf("%7s", String(nm)[1:min(6,end)]); end; println()
for (i, ni) in enumerate(PROPS)
    @printf("%-12s", String(ni))
    for j in 1:N_PROPS
        m = M[:,i] .& M[:,j]
        @printf("%7.2f", cor(Y[m,i], Y[m,j]))
    end
    println()
end

println("\n" * "="^76)
println("Are the cheap shortcuts informative?  R² of each target on 3 image summaries")
println("="^76)
ink = [sum(abs.(im .- median(im))) for im in imgs]
mn  = [mean(im) for im in imgs]
sd  = [std(im)  for im in imgs]
X   = hcat(ones(length(imgs)), ink, mn, sd)
@printf("\n%-12s %10s\n", "property", "R2")
for (j, nm) in enumerate(PROPS)
    m = M[:,j]; y = Y[m,j]; A = X[m,:]
    r = y .- A*(A\y)
    @printf("%-12s %10.3f\n", String(nm), 1 - sum(abs2, r)/sum(abs2, y .- mean(y)))
end

# curvedness/closedness correlate because a straight stroke is necessarily open, so the
# 20 % straight atom puts a spike at (0,0). Among curved samples the coupling should be
# near zero — that is the check that radius and turn really were sampled independently.
cv = Y[:,1] .> 0.01
@printf("\ncor(curvedness, closedness) = %.2f overall, %.2f among the %.0f%% that curve\n",
        cor(Y[:,1], Y[:,3]), cor(Y[cv,1], Y[cv,3]), 100mean(cv))

# The angle band mix must be flat: a three-way readout on unbalanced bands would report a
# prior, not a discrimination.
kn = Y[:,4] .< 180 - 1e-9
@printf("\nvertex-angle band mix over %d kinked samples: ", sum(kn))
for b in (:acute, :right, :obtuse)
    @printf("%s %.2f  ", b, mean(band_of.(Y[kn,4]) .=== b))
end
println()

println("""

The R² column is the Phase 8 post-mortem made routine. Any property a linear function of
three scalar image summaries can already predict is not testing structure, and a model
scoring well on it would be telling us nothing. `thickness`, `fuzziness` and `polarity` are
*expected* to be predictable here — they are photometric controls, not structure.""")
