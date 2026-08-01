# ── PLAIN SCRIPT, not a Pluto notebook ──────────────────────────────────────
# `julia --project=.. -t 14 Add_DeepHead.jl`   (from the SimpleStrokeTests directory)
#
# Adds a fourth arm: our 31 features with a **deeper, tapering head** — 31 → 512 → 128 → 64 → 8 —
# against the 31 → 256 → 256 → 8 used everywhere else.
#
# The point is to separate two things the tables cannot currently distinguish. Where our features
# score below frozen ConvNeXt, is that because the 31 numbers do not carry the information, or
# because a two-layer 256-wide head cannot extract it? A wider first layer and a taper give the
# head more room to fold the input before compressing, at a similar parameter count (~91 k against
# ~76 k), so a large gain would point at the readout and a null result at the representation.
#
# Everything else is held fixed: same cached features, same images, same targets, same standard-
# isation, same Adam 1e-3, same batch 128, same snapshot epochs. Only the head's shape changes.
#
# Merges into `figures_predictions/predictions.jls` rather than rewriting it, so the three
# existing arms are untouched and the figures redraw with four rows.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Statistics, Printf, Random, LinearAlgebra, Serialization, FFTW
using Flux, CUDA
include(joinpath(@__DIR__, "..", "ConVNextTest", "Readout.module.jl"))
using .Readout
BLAS.set_num_threads(min(16, Sys.CPU_THREADS)); FFTW.set_num_threads(1)

const DATA  = joinpath(@__DIR__, "..", "ConVNextTest", "data")
const CACHE = joinpath(@__DIR__, "..", "ConVNextTest", "cache")
const FIG   = joinpath(@__DIR__, "figures_predictions")
const NTR, NTE = 16000, 4000
const NAME  = "our features deep (31)"

d = deserialize(joinpath(FIG, "predictions.jls"))
PROPS, SNAPS, Yte = d.props, d.snaps, d.Yte
np = length(PROPS)

read_mat(p, n, k) = permutedims(reshape(read!(open(p), Vector{Float32}(undef, n*k)), k, n))
Ytr = read_mat(joinpath(DATA, "iid_train_y.f32"), NTR, np)
nva = NTR ÷ 6; tr = 1:NTR-nva; va = NTR-nva+1:NTR
μy, σy = zfit(Ytr[tr, :]); Zt = zapply(Ytr[tr, :], μy, σy)

Ftr, Fte = deserialize(joinpath(CACHE, "ours_g1_iid_$(NTR)_$(NTE).jls"))
μ, σ = zfit(Ftr[tr, :]); A = zapply(Ftr, μ, σ); T = zapply(Fte, μ, σ)

Random.seed!(1)
dev = CUDA.functional() ? gpu : cpu
Xg = dev(permutedims(A[tr, :])); Tg = dev(permutedims(T)); Yg = dev(permutedims(Zt))
m = Chain(Dense(size(A,2) => 512, relu), Dense(512 => 128, relu),
          Dense(128 => 64, relu), Dense(64 => np)) |> dev
@printf("head: %d → 512 → 128 → 64 → %d   (%d parameters)\n",
        size(A,2), np, sum(length, Flux.trainables(m)))
opt = Flux.setup(Flux.Adam(1f-3), m); n = size(Xg, 2)
out = Dict{Int,Matrix{Float32}}()
t = @elapsed for e in 1:maximum(SNAPS)
    for i in Iterators.partition(randperm(n), 128)
        _, gs = Flux.withgradient(mm -> Flux.mse(mm(Xg[:,i]), Yg[:,i]), m)
        Flux.update!(opt, m, gs[1])
    end
    e in SNAPS && (out[e] = permutedims(Array(m(Tg))) .* σy' .+ μy')
end
@printf("trained in %.0f s\n\n", t)

d.preds[NAME] = out
serialize(joinpath(FIG, "predictions.jls"), d)

# ── the table, every arm at every snapshot epoch, from one run on identical data
arms = ["our CNN (end to end)", "our features (31)", NAME, "frozen ConvNeXt (1024)"]
r2of(ŷ, y) = 1 - sum(abs2, ŷ .- y)/sum(abs2, y .- mean(y))
for e in SNAPS
    @printf("\nepoch %d\n%-26s", e, "arm")
    for p in PROPS; @printf("%11s", p[1:min(10,end)]); end; println()
    println("-"^(26 + 11*np))
    for a in arms
        haskey(d.preds, a) || continue
        @printf("%-26s", a)
        for j in 1:np
            @printf("%11.3f", r2of(Float64.(d.preds[a][e][:, j]), Float64.(Yte[:, j])))
        end
        println()
    end
end
println("\nmerged into $(joinpath(FIG, "predictions.jls"))")
