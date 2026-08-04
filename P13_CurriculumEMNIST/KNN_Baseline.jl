# ── PLAIN SCRIPT ────────────────────────────────────────────────────────────
# `julia --project=.. -t 14 KNN_Baseline.jl`
#
# How much of each representation's accuracy is reachable by pure remember-and-interpolate?
#
# k-nearest-neighbour IS that strategy, made literal: store all 28,200 training images, classify a
# test image by which stored ones it lands nearest. It has no learned parameters and no decision
# boundary -- only a metric and a memory. So kNN accuracy on a representation is a direct estimate
# of how far "the test image resembles remembered training images" gets you, and the gap up to the
# trained MLP is what the readout adds beyond lookup.
#
# WHAT THIS DOES NOT SAY. A representation that has genuinely extracted invariants will ALSO have
# good kNN accuracy -- it places same-class images near each other, which is the point. So a high
# kNN score is not evidence of memorisation; it is evidence that the metric is informative. The
# number to read is the GAP, and the comparison across representations, not any single value.
#
# Subset 1 only (28,200) as the memory, matching what every arm in this phase trains on, with the
# same standardisation: subset-1 statistics, clamped to +/-3.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Statistics, Printf, Serialization, LinearAlgebra, DelimitedFiles
BLAS.set_num_threads(min(16, Sys.CPU_THREADS))

const CACHE = joinpath(@__DIR__, "cache")
const KS = (1, 5, 10, 25)

read_f32(p, n, d) = permutedims(reshape(read!(p, Vector{Float32}(undef, n*d)), d, n))
read_i64(p, n)    = read!(p, Vector{Int64}(undef, n))

"""
Top-`k` vote over cosine similarity, chunked over test rows so the 18,800 x 28,200 similarity
matrix is never held whole. Rows are unit-normalised, so the inner product IS the cosine and the
largest inner product is the nearest neighbour.
"""
function knn_acc(Xtr, ytr, Xte, yte; ks=KS, chunk=2000)
    A = Xtr ./ max.(sqrt.(sum(abs2, Xtr, dims=2)), 1f-8)
    B = Xte ./ max.(sqrt.(sum(abs2, Xte, dims=2)), 1f-8)
    nc = maximum(ytr); kmax = maximum(ks)
    hits = Dict(k => 0 for k in ks)
    for lo in 1:chunk:size(B, 1)
        hi = min(lo+chunk-1, size(B, 1))
        S = B[lo:hi, :] * A'                        # (chunk, ntrain) cosine similarities
        for i in 1:(hi-lo+1)
            ord = partialsortperm(view(S, i, :), 1:kmax; rev=true)
            votes = zeros(Int, nc)
            for (r, j) in enumerate(ord)
                votes[ytr[j]] += 1
                if r in ks
                    argmax(votes) == yte[lo+i-1] && (hits[r] += 1)
                end
            end
        end
    end
    Dict(k => hits[k]/size(B,1) for k in ks)
end

perm = Int.(vec(readdlm(joinpath(CACHE, "partition.txt"))))
s1 = perm[1:28200]

reps = Any[]
let (F, y) = deserialize(joinpath(CACHE, "ours_train.jls")),
    (G, z) = deserialize(joinpath(CACHE, "ours_test.jls"))
    push!(reps, ("ours, frozen (381)", F, y, G, z))
end
let d = parse.(Int, split(strip(read(joinpath(CACHE, "cx_dims.txt"), String))))
    push!(reps, ("ConvNeXt-base frozen, stage 4 (1024)",
                 read_f32(joinpath(CACHE, "cx_train_s4.f32"), 112800, d[4]),
                 read_i64(joinpath(CACHE, "cx_train_y.i64"), 112800),
                 read_f32(joinpath(CACHE, "cx_test_s4.f32"), 18800, d[4]),
                 read_i64(joinpath(CACHE, "cx_test_y.i64"), 18800)))
end
# raw pixels: the floor. Any representation not beating this has bought nothing.
include(joinpath(@__DIR__, "..", "LoadEMNIST.module.jl")); using .LoadEMNIST
let S = joinpath(homedir(), "Julia", "DATABASES", "EMNIST", "emnist_source_files")
    A = read_emnist_images(joinpath(S, "emnist-balanced-train-images-idx3-ubyte"))
    B = read_emnist_images(joinpath(S, "emnist-balanced-test-images-idx3-ubyte"))
    push!(reps, ("raw pixels (784)",
                 permutedims(reshape(A, 784, :)), read_i64(joinpath(CACHE,"cx_train_y.i64"),112800),
                 permutedims(reshape(B, 784, :)), read_i64(joinpath(CACHE,"cx_test_y.i64"),18800)))
end

@printf("kNN over subset 1 (%d stored images), cosine similarity\n\n", length(s1))
@printf("%-38s %7s %7s %7s %7s\n", "representation", "1-NN", "5-NN", "10-NN", "25-NN")
for (name, Ftr, ytr, Fte, yte) in reps
    X = Ftr[s1, :]
    μ = vec(mean(X, dims=1)); σ = vec(std(X, dims=1)); σ[σ .<= 1f-8] .= 1f0
    z(M) = clamp.((M .- μ') ./ σ', -3f0, 3f0)
    a = knn_acc(z(X), ytr[s1], z(Fte), yte)
    @printf("%-38s", name); for k in KS; @printf(" %6.2f%%", 100a[k]); end; println(); flush(stdout)
end
