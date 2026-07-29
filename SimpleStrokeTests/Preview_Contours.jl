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
rows = [(:none, "no event"), (:gap, "gap"), (:kink, "kink"), (:tee, "tee"), (:cross, "crossing")]
NCOL = 8
panels = Any[]
for (ev, name) in rows, k in 1:NCOL
    p = sample_params(rng; event=ev)
    v, _ = targets_of(p)
    ttl = if ev === :kink;  @sprintf("%.0f°  %s", v[5], band_of(v[5]))
          elseif ev === :gap; @sprintf("brk %.2f", v[2])
          else @sprintf("crv %.2f cls %.2f", v[1], v[3]) end
    push!(panels, heatmap(render_params(p, rng; N=N); c=:grays, clims=(0,1), axis=false,
                          ticks=false, colorbar=false, yflip=true, aspect_ratio=1,
                          title=(k == 1 ? name * " — " : "") * ttl, titlefontsize=7,
                          titlelocation=:left))
end
plot(panels...; layout=(length(rows), NCOL), size=(150NCOL, 152length(rows)),
     plot_title="SimpleStrokeTests — graded properties, everything photometric randomised",
     plot_titlefontsize=11)
savefig(joinpath(@__DIR__, "contactsheet.png")); println("wrote contactsheet.png")

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
kn = M[:,5]
@printf("\nangle band mix over %d kinked samples: ", sum(kn))
for b in (:acute, :right, :obtuse)
    @printf("%s %.2f  ", b, mean(band_of.(Y[kn,5]) .=== b))
end
println()

println("""

The R² column is the Phase 8 post-mortem made routine. Any property a linear function of
three scalar image summaries can already predict is not testing structure, and a model
scoring well on it would be telling us nothing. `thickness`, `softness` and `polarity` are
*expected* to be predictable here — they are photometric controls, not structure.""")
