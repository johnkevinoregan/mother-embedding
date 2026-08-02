# ── PLAIN SCRIPT ────────────────────────────────────────────────────────────
# `julia --project=.. -t 14 FewShot_Eval.jl`     (after fewshot_train.py)
#
# Step 2 of 2. Ten-way few-shot classification on the 10 EMNIST classes the from-scratch network
# was never shown, comparing four representations.
#
# THE POINT. On an i.i.d. split, "memorised the training images and interpolated between them" and
# "extracted the invariant structure" predict the SAME accuracy, because test images sit near
# training images in any adequate representation. Holding out whole classes breaks that tie:
# remembering images of 'A' cannot help you recognise a 'q' you have never seen, but strokes,
# junctions, curvature and closure transfer. So this measures the part of a representation that
# is not stored examples.
#
# ARMS, and what a win means for each:
#   ours (381)              never trained on anything, so these classes are not "unseen" for it
#                           in any meaningful sense -- this is simply its ordinary performance,
#                           and that is exactly the asymmetry being tested
#   ConvNeXt-base frozen    trained on ImageNet, never on EMNIST -- transfer from photographs
#   ConvNeXt-tiny scratch   trained on 37 EMNIST classes -- the arm that could have memorised
#   raw pixels              the floor
#
# PROTOCOL. Prototypical-network style: k support images per class drawn from the EMNIST *train*
# split, class prototypes are support means, queries are ALL held-out-class images from the
# *test* split (image-disjoint from support), assigned by cosine similarity to the nearest
# prototype. 200 episodes per (arm, k); episodes differ in which support images are drawn.
#
# STANDARDISATION uses the held-out-class train pool -- no labels, identical for every arm.
# Without it, cosine distance on our features would be dominated by whichever channels happen to
# have the largest scale, which would be an artefact of units rather than of information.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Statistics, Printf, Random, Serialization, LinearAlgebra, DelimitedFiles
BLAS.set_num_threads(min(16, Sys.CPU_THREADS))

const CACHE = joinpath(@__DIR__, "cache")
const FS    = joinpath(@__DIR__, "fewshot")
const SHOTS = (1, 2, 5, 10, 20)
const NEPI  = 200

read_f32(p, n, d) = permutedims(reshape(read!(p, Vector{Float32}(undef, n*d)), d, n))
read_i64(p, n)    = read!(p, Vector{Int64}(undef, n))

held    = Int.(vec(readdlm(joinpath(FS, "heldout.txt"))))
itr     = read_i64(joinpath(FS, "heldidx_train.i64"), filesize(joinpath(FS,"heldidx_train.i64"))÷8) .+ 1
ite     = read_i64(joinpath(FS, "heldidx_test.i64"),  filesize(joinpath(FS,"heldidx_test.i64"))÷8) .+ 1
@printf("held-out classes: %s\n%d support-pool images, %d query images\n\n",
        string(held), length(itr), length(ite))

ytr_all = read_i64(joinpath(CACHE, "cx_train_y.i64"), 112800)
yte_all = read_i64(joinpath(CACHE, "cx_test_y.i64"),  18800)
ys, yq  = ytr_all[itr], yte_all[ite]

reps = Any[]
let (F, _) = deserialize(joinpath(CACHE, "ours_train.jls")),
    (G, _) = deserialize(joinpath(CACHE, "ours_test.jls"))
    push!(reps, ("ours, frozen (381)", F[itr, :], G[ite, :]))
end
let d = parse.(Int, split(strip(read(joinpath(CACHE, "cx_dims.txt"), String))))[4]
    push!(reps, ("ConvNeXt-base frozen ImageNet (1024)",
                 read_f32(joinpath(CACHE, "cx_train_s4.f32"), 112800, d)[itr, :],
                 read_f32(joinpath(CACHE, "cx_test_s4.f32"),  18800,  d)[ite, :]))
end
let d = parse(Int, strip(read(joinpath(FS, "scratch_dim.txt"), String)))
    push!(reps, ("ConvNeXt-tiny scratch on 37 classes (768)",
                 read_f32(joinpath(FS, "scratch_train.f32"), length(itr), d),
                 read_f32(joinpath(FS, "scratch_test.f32"),  length(ite), d)))
end
include(joinpath(@__DIR__, "..", "LoadEMNIST.module.jl")); using .LoadEMNIST
let S = joinpath(homedir(), "Julia", "DATABASES", "EMNIST", "emnist_source_files")
    A = permutedims(reshape(read_emnist_images(joinpath(S,"emnist-balanced-train-images-idx3-ubyte")), 784, :))
    B = permutedims(reshape(read_emnist_images(joinpath(S,"emnist-balanced-test-images-idx3-ubyte")), 784, :))
    push!(reps, ("raw pixels (784)", A[itr, :], B[ite, :]))
end

"Episode: k support per class, prototypes are support means, queries by cosine to nearest."
function episode(S, ys, Q, yq, held, k, rng)
    P = zeros(Float32, length(held), size(S, 2))
    for (c, cl) in enumerate(held)
        pool = findall(==(cl), ys)
        P[c, :] = mean(view(S, pool[randperm(rng, length(pool))[1:k]], :), dims=1)
    end
    P ./= max.(sqrt.(sum(abs2, P, dims=2)), 1f-8)
    pred = held[vec(map(i -> i[2], argmax(Q * P', dims=2)))]
    mean(pred .== yq)
end

results = Dict{String, Dict{Int, Vector{Float64}}}()
@printf("%-42s", "representation"); for k in SHOTS; @printf("%12s", "$(k)-shot"); end; println()
for (name, Str, Qte) in reps
    μ = vec(mean(Str, dims=1)); σ = vec(std(Str, dims=1)); σ[σ .<= 1f-8] .= 1f0
    z(M) = clamp.((M .- μ') ./ σ', -3f0, 3f0)
    S = z(Str); Q = z(Qte); Q ./= max.(sqrt.(sum(abs2, Q, dims=2)), 1f-8)
    results[name] = Dict{Int, Vector{Float64}}()
    @printf("%-42s", name)
    for k in SHOTS
        rng = MersenneTwister(1)
        a = [episode(S, ys, Q, yq, held, k, rng) for _ in 1:NEPI]
        results[name][k] = a
        @printf("  %5.2f±%.2f", 100mean(a), 100*1.96*std(a)/sqrt(NEPI))
    end
    println(); flush(stdout)
end
serialize(joinpath(FS, "fewshot_results.jls"), (results=results, shots=SHOTS, held=held, nepi=NEPI))
println("\nwrote fewshot/fewshot_results.jls   (mean ± 95% CI over $NEPI episodes)")
