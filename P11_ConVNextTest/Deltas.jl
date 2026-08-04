# ── PLAIN SCRIPT, not a Pluto notebook ──────────────────────────────────────
# `julia --project=.. Deltas.jl`   (after ConvNextReadout.jl has written results/)
#
# Transfer cost, which is what the invariance claim is actually about.
#
# `ConvNextReadout.jl` prints absolute R² per split, and reading prediction 3 off those tables
# is a mistake I nearly made: an arm can score *higher* on the polarity split in absolute terms
# while still being the one that degrades. "Our features are unchanged when polarity flips" is
# a statement about the **change** from the i.i.d. split, not about the level.
#
# So for every arm and property this prints
#
#     Δ = R²(extrapolation split) − R²(i.i.d. split)
#
# Δ ≈ 0 means the representation genuinely did not care about the held-out nuisance. A large
# negative Δ means it did, whatever its absolute score. The held-out property itself has no
# defined R² in its own split and shows as "—".

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Serialization, Printf, Statistics

const OUT = joinpath(@__DIR__, "results")

# Read the PER-SPLIT files, not `all.jls`. A partial re-run (`CX_SPLITS=iid`, used to regenerate
# the training curves) rewrote `all.jls` with only the split it ran, silently leaving nothing to
# compare against — while `polarity.jls`, `fuzziness.jls` and `thickness.jls` were untouched and
# still correct. Loading the individual files makes this script independent of which splits the
# last run happened to cover.
res = Dict{String,Any}()
for s in ["iid", "polarity", "fuzziness", "thickness"]
    p = joinpath(OUT, "$s.jls")
    isfile(p) && (res[s] = deserialize(p))
end
haskey(res, "iid") || error("no results/iid.jls — nothing to compare against")

props = res["iid"].props
iid = Dict(nm => v for (nm, _, v) in res["iid"].rows)

nanfmt(x) = isnan(x) ? @sprintf("%11s", "—") : @sprintf("%11.3f", x)

for split in ["polarity", "fuzziness", "thickness"]
    haskey(res, split) || continue
    println("\n" * "="^119)
    println("TRANSFER COST — Δ R² from i.i.d. to the \"$split\" held-out split")
    println("="^119)
    @printf("\n%-24s%7s", "arm", "meanΔ")
    for p in props; @printf("%11s", p[1:min(10, end)]); end; println()
    println("-"^119)
    rows = res[split].rows
    # ranked by mean Δ over the properties that have one, best transfer first
    scored = map(rows) do (nm, nf, v)
        d = [(haskey(iid, nm) && !isnan(v[j]) && !isnan(iid[nm][j])) ? v[j] - iid[nm][j] : NaN
             for j in 1:length(props)]
        u = filter(!isnan, d)
        (nm, isempty(u) ? -Inf : mean(u), d)
    end
    for (nm, m, d) in sort(scored; by = x -> -x[2])
        @printf("%-24s%7s", nm[1:min(24, end)], isfinite(m) ? @sprintf("%+.3f", m) : "—")
        for x in d; print(nanfmt(x)); end
        println()
    end
end

println("\n" * "="^119)
println("""
READ: Δ ≈ 0 means the representation is genuinely invariant to the held-out nuisance.
Large negative Δ means it is not, regardless of how high its absolute R² was. Compare arms
by Δ, and only then look at the level.""")
