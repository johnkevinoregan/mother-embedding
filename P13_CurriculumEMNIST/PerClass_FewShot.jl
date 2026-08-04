# ── PLAIN SCRIPT ────────────────────────────────────────────────────────────
# `julia --project=.. -t 14 PerClass_FewShot.jl`
#
# Per-class 5-shot accuracy, as a check on the headline few-shot result.
#
# EMNIST's held-out classes are NOT visually disjoint from the base classes the network trained
# on: 'O' is held out while the digit '0' is in the base set, and 'Q' is held out while lowercase
# 'q' is in it. Those are near-duplicate glyphs. If the from-scratch network's large advantage
# came from such leakage rather than from general character structure, it would show up as a
# couple of classes carrying the whole gain. If the advantage is spread across all ten, it is
# structure.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Statistics, Printf, Random, Serialization, LinearAlgebra, DelimitedFiles
BLAS.set_num_threads(min(16, Sys.CPU_THREADS))

const CACHE = joinpath(@__DIR__, "cache"); const FS = joinpath(@__DIR__, "fewshot")
const K, NEPI = 5, 200
# EMNIST balanced label order, 1-based: 1-10 digits 0-9, 11-36 A-Z, 37-47 a b d e f g h n q r t
const GLYPH = vcat(string.(0:9), string.('A':'Z'), ["a","b","d","e","f","g","h","n","q","r","t"])

read_f32(p, n, d) = permutedims(reshape(read!(p, Vector{Float32}(undef, n*d)), d, n))
read_i64(p, n)    = read!(p, Vector{Int64}(undef, n))

held = Int.(vec(readdlm(joinpath(FS, "heldout.txt"))))
itr  = read_i64(joinpath(FS,"heldidx_train.i64"), filesize(joinpath(FS,"heldidx_train.i64"))÷8) .+ 1
ite  = read_i64(joinpath(FS,"heldidx_test.i64"),  filesize(joinpath(FS,"heldidx_test.i64"))÷8) .+ 1
ys = read_i64(joinpath(CACHE,"cx_train_y.i64"),112800)[itr]
yq = read_i64(joinpath(CACHE,"cx_test_y.i64"), 18800)[ite]

reps = Any[]
let (F,_) = deserialize(joinpath(CACHE,"ours_train.jls")), (G,_) = deserialize(joinpath(CACHE,"ours_test.jls"))
    push!(reps, ("ours (381)", F[itr,:], G[ite,:])) end
let d = parse(Int, strip(read(joinpath(FS,"scratch_dim.txt"), String)))
    push!(reps, ("scratch (768)", read_f32(joinpath(FS,"scratch_train.f32"), length(itr), d),
                                  read_f32(joinpath(FS,"scratch_test.f32"),  length(ite), d))) end

acc = Dict{String, Vector{Float64}}()
for (name, Str, Qte) in reps
    μ = vec(mean(Str,dims=1)); σ = vec(std(Str,dims=1)); σ[σ .<= 1f-8] .= 1f0
    z(M) = clamp.((M .- μ')./σ', -3f0, 3f0)
    S = z(Str); Q = z(Qte); Q ./= max.(sqrt.(sum(abs2,Q,dims=2)), 1f-8)
    hit = zeros(Int, length(held)); tot = zeros(Int, length(held))
    rng = MersenneTwister(1)
    for _ in 1:NEPI
        P = zeros(Float32, length(held), size(S,2))
        for (c,cl) in enumerate(held)
            pool = findall(==(cl), ys); P[c,:] = mean(view(S, pool[randperm(rng,length(pool))[1:K]], :), dims=1)
        end
        P ./= max.(sqrt.(sum(abs2,P,dims=2)), 1f-8)
        pred = held[vec(map(i->i[2], argmax(Q*P', dims=2)))]
        for (c,cl) in enumerate(held)
            m = yq .== cl; hit[c] += sum(pred[m] .== cl); tot[c] += sum(m)
        end
    end
    acc[name] = hit ./ tot
end

@printf("%d-shot per-class accuracy (%d episodes)\n\n", K, NEPI)
@printf("%-8s %8s %10s %10s %9s\n", "class", "glyph", "ours", "scratch", "gain")
g = Float64[]
for (c, cl) in enumerate(held)
    d = 100*(acc["scratch (768)"][c] - acc["ours (381)"][c]); push!(g, d)
    @printf("%-8d %8s %9.2f%% %9.2f%% %+8.2f\n", cl, GLYPH[cl],
            100acc["ours (381)"][c], 100acc["scratch (768)"][c], d)
end
@printf("\nmean gain %+.2f, median %+.2f, min %+.2f (%s), max %+.2f (%s)\n",
        mean(g), median(g), minimum(g), GLYPH[held[argmin(g)]], maximum(g), GLYPH[held[argmax(g)]])
@printf("classes with a positive gain: %d of %d\n", count(>(0), g), length(g))
