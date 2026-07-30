# ── PLAIN SCRIPT, not a Pluto notebook ──────────────────────────────────────
# `julia --project=.. PartialOut.jl`   (after ConvNextStimuli/extract/readout have run)
#
# What does each representation know that the other does not?
#
# The stage probe showed frozen ConvNeXt's geometry is a **stage-3** construction: s1 reads
# curvedness at 0.002 and carries only photometric properties, s3 reads it at 0.949. So the
# question is whether s3 contains geometric information our 31 features cannot express — in which
# case there is a missing operator to find — or whether the gap is readout capacity and template
# count, in which case designing another operator is the wrong move.
#
# METHOD. Incremental R², not a two-stage residual fit. For each property, three ridge fits under
# the Phase 9 protocol:
#
#     a = R²(y ← ours)          b = R²(y ← s3)          c = R²(y ← ours ⊕ s3)
#
#     c − a   what s3 ADDS on top of ours     ← is there a missing function?
#     c − b   what ours ADDS on top of s3     ← is our contribution unique?
#
# Increments are cleaner than partialling out by hand: no in-sample/out-of-sample bookkeeping, and
# because the readout is linear the two increments are directly interpretable as "variance this
# block explains that the other cannot".
#
# Asking `R²(s3 ← ours)` instead would be a rank artefact — 31 columns cannot span 384 — and would
# look damning for us for a reason that has nothing to do with the operators.
#
# BOTH SPLITS, because they answer different halves. On i.i.d. `c − b` should be ≈ 0: ConvNeXt
# already has everything. On the **polarity** split it should be large, because that is where our
# constructed invariance lives, and it would be the first direct measurement of our unique
# contribution rather than an inference from the Δ tables.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Statistics, Printf, Random, LinearAlgebra, Serialization
include(joinpath(@__DIR__, "Readout.module.jl"))
using .Readout

const DATA  = joinpath(@__DIR__, "data")
const FEAT  = joinpath(@__DIR__, "features")
const CACHE = joinpath(@__DIR__, "cache")
const GRID  = parse(Int, get(ENV, "CX_GRID", "1"))
const MODEL = get(ENV, "CX_MODEL", "tiny")
const STAGE = parse(Int, get(ENV, "CX_STAGE", "3"))

const PROPS = [strip(l) for l in readlines(joinpath(DATA, "props.txt")) if !isempty(strip(l))]
read_manifest() = [(m[1], parse(Int, m[2]), parse(Int, m[3]))
                   for m in split.(filter(l -> !startswith(l, "#") && !isempty(strip(l)),
                                          readlines(joinpath(DATA, "manifest.txt"))))]
function read_matrix(path, n, p)
    a = read!(path, Vector{Float32}(undef, n * p))
    permutedims(reshape(a, p, n))
end
stage_dims(m) = parse.(Int, split(strip(read(joinpath(FEAT, "$(m)_dims.txt"), String))))

"Ridge R² per property under the Phase 9 protocol — same split, same λ selection."
function score(X, Ytr, Xte, Yte; drop=nothing)
    ntr = size(X, 1)
    nva = clamp(ntr ÷ 6, 40, ntr - 40); va = ntr-nva+1:ntr; tr = 1:ntr-nva
    μy, σy = zfit(Ytr[tr, :])
    Zt = zapply(Ytr[tr, :], μy, σy); Zv = zapply(Ytr[va, :], μy, σy)
    μ, σ = zfit(X[tr, :]); A = zapply(X, μ, σ); T = zapply(Xte, μ, σ)
    P = ridge(A[tr, :], Zt, A[va, :], Zv, T)[1] .* σy' .+ μy'
    [(drop !== nothing && PROPS[j] == String(drop)) ? NaN : r2(P[:, j], Yte[:, j])
     for j in 1:size(Yte, 2)]
end

fmt(x) = isnan(x) ? @sprintf("%9s", "—") : @sprintf("%9.3f", x)

function main()
    dims = stage_dims(MODEL)
    @printf("Incremental R²: ours (grid %d) against convnext_%s stage %d (%d dims)\n\n",
            GRID, MODEL, STAGE, dims[STAGE])
    for (split, ntr, nte) in read_manifest()
        split in ("iid", "polarity") || continue
        drop = split == "iid" ? nothing : Symbol(split)
        ck = joinpath(CACHE, "ours_g$(GRID)_$(split)_$(ntr)_$(nte).jls")
        isfile(ck) || (println("no cached ours features for $split — skipping"); continue)
        Otr, Ote = deserialize(ck)
        Str = read_matrix(joinpath(FEAT, "$(MODEL)_$(split)_train_s$(STAGE).f32"), ntr, dims[STAGE])
        Ste = read_matrix(joinpath(FEAT, "$(MODEL)_$(split)_test_s$(STAGE).f32"),  nte, dims[STAGE])
        Ytr = read_matrix(joinpath(DATA, "$(split)_train_y.f32"), ntr, length(PROPS))
        Yte = read_matrix(joinpath(DATA, "$(split)_test_y.f32"),  nte, length(PROPS))

        a = score(Otr, Ytr, Ote, Yte; drop=drop)
        b = score(Str, Ytr, Ste, Yte; drop=drop)
        c = score(hcat(Otr, Str), Ytr, hcat(Ote, Ste), Yte; drop=drop)

        println("="^104)
        @printf("%s split\n", uppercase(split)); println("="^104)
        @printf("%-26s"," "); for p in PROPS; @printf("%9s", p[1:min(8,end)]); end; println()
        for (nm, v) in (("a  ours ($(size(Otr,2)) cols)", a),
                        ("b  s$STAGE ($(size(Str,2)) cols)", b),
                        ("c  both", c))
            @printf("%-26s", nm); for x in v; print(fmt(x)); end; println()
        end
        println("-"^104)
        @printf("%-26s", "c−a  s$STAGE adds over ours"); for j in eachindex(c); print(fmt(c[j]-a[j])); end; println()
        @printf("%-26s", "c−b  ours adds over s$STAGE"); for j in eachindex(c); print(fmt(c[j]-b[j])); end; println()
        println()
        flush(stdout)
    end
    println("""
READ:
  c−a large on vangle/brokenness  → s$STAGE holds geometry our operators cannot express;
                                    the missing-function hunt is justified and quantified.
  c−a ≈ 0                         → no geometry we lack; the gap is capacity, and designing
                                    another operator is the wrong move.
  c−b large on the polarity split → our constructed invariance is a genuinely unique
                                    contribution, measured directly rather than inferred.""")
end

main()
