# ── PLAIN SCRIPT ────────────────────────────────────────────────────────────
# `julia --project=.. Curriculum.jl`     (after both extraction steps)
#
# EMNIST balanced, 47 classes, official split: 112,800 train / 18,800 test.
#
# THE EXPERIMENT. The training set is cut into 4 disjoint subsets of 28,200. Epochs 1–15 train on
# subset 1, 16–30 on subset 2, 31–45 on subset 3, 46–60 on subset 4. The optimiser state is NOT
# reset at a switch — the run continues, only the data underneath it changes.
#
# WHAT IT ASKS. The four subsets are a random partition, so they are drawn from the *same*
# distribution: a learner that had extracted the structure of the task rather than memorised
# examples should not be able to tell that the data changed. Every discontinuity at epoch 15, 30
# or 45 is therefore a direct measurement of how much of the fit was specific to the examples in
# front of it.
#
# THREE CURVES per run, because test accuracy alone cannot separate the two things going on:
#   * test      — held-out accuracy on all 18,800 test images. The thing we care about.
#   * current   — accuracy on the subset being trained on right now. Its gap above `test` is
#                 memorisation, measured directly.
#   * set 1     — accuracy on subset 1 throughout, including long after training has left it.
#                 Once training moves on, this decays from "memorised" back toward "test", and
#                 how fast is what forgetting means here.
#
# CONTROL. Each arm is also run for 60 epochs on subset 1 alone. Same number of gradient steps,
# same amount of data per epoch, no switches. Anything the switching run does that the control
# does not is caused by the switch and not by the epoch count or the reduced training set.
#
# STANDARDISATION uses subset 1's statistics only, held fixed for the whole run. Using the full
# training set's statistics would let subsets 2–4 influence the model before they arrive.
#
# NO REGULARISATION — no dropout, no weight decay, no early stopping. Memorisation is the thing
# being measured, so it must not be suppressed.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Statistics, Printf, Random, Serialization, LinearAlgebra
using Flux, CUDA
using Flux: onehotbatch, onecold

const CACHE  = joinpath(@__DIR__, "cache")
const NSET   = 4
const PER    = 15                     # epochs per subset
const EPOCHS = NSET * PER
const HIDDEN = 512
const NCLASS = 47
const BATCH  = 128
const SEED   = 1
const ARMS   = split(get(ENV, "CU_ARMS", "ours,convnext"), ",")

dev = CUDA.functional() ? gpu : cpu
@printf("device: %s\n", CUDA.functional() ? CUDA.name(CUDA.device()) : "CPU")

read_f32(p, n, d) = permutedims(reshape(read!(p, Vector{Float32}(undef, n*d)), d, n))
read_i64(p, n)    = read!(p, Vector{Int64}(undef, n))

function load_ours()
    Ftr, ytr = deserialize(joinpath(CACHE, "ours_train.jls"))
    Fte, yte = deserialize(joinpath(CACHE, "ours_test.jls"))
    Ftr, ytr, Fte, yte
end

function load_cx(stage)
    d = parse.(Int, split(strip(read(joinpath(CACHE, "cx_dims.txt"), String))))[stage]
    Ftr = read_f32(joinpath(CACHE, "cx_train_s$(stage).f32"), 112800, d)
    Fte = read_f32(joinpath(CACHE, "cx_test_s$(stage).f32"),  18800,  d)
    Ftr, read_i64(joinpath(CACHE, "cx_train_y.i64"), 112800),
    Fte, read_i64(joinpath(CACHE, "cx_test_y.i64"),  18800)
end

"""
    run_arm(name, Ftr, ytr, Fte, yte, sets; switching)

60 epochs of an MLP on `Ftr`, following `sets` (a vector of index vectors) if `switching`,
otherwise staying on `sets[1]` throughout. Returns the three accuracy curves.
"""
function run_arm(name, Ftr, ytr, Fte, yte, sets; switching=true)
    Random.seed!(SEED)
    s1 = sets[1]
    μ = vec(mean(Ftr[s1, :], dims=1)); σ = vec(std(Ftr[s1, :], dims=1)); σ[σ .<= 1f-8] .= 1f0
    z(M) = clamp.((M .- μ') ./ σ', -3f0, 3f0)
    nf = size(Ftr, 2)

    Xall = dev(permutedims(z(Ftr)))
    Xte  = dev(permutedims(z(Fte)))
    Yall = dev(onehotbatch(ytr, 1:NCLASS))

    m   = dev(Chain(Dense(nf => HIDDEN, relu), Dense(HIDDEN => HIDDEN, relu),
                    Dense(HIDDEN => NCLASS)))
    opt = Flux.setup(Flux.Adam(1f-3), m)

    acc(X, y) = begin
        p = Int[]
        for i in Iterators.partition(1:length(y), 8192)
            append!(p, Array(onecold(m(X[:, i]), 1:NCLASS)))
        end
        mean(p .== y)
    end

    te = Float64[]; cu = Float64[]; s1a = Float64[]; sw = Int[]
    t0 = time()
    for ep in 1:EPOCHS
        k   = switching ? min(NSET, (ep - 1) ÷ PER + 1) : 1
        idx = sets[k]
        ep > 1 && k != (switching ? min(NSET, (ep - 2) ÷ PER + 1) : 1) && push!(sw, ep)
        for i in Iterators.partition(shuffle(idx), BATCH)
            _, gs = Flux.withgradient(mm -> Flux.logitcrossentropy(mm(Xall[:, i]), Yall[:, i]), m)
            Flux.update!(opt, m, gs[1])
        end
        push!(te,  acc(Xte, yte))
        push!(cu,  acc(Xall[:, idx], ytr[idx]))
        push!(s1a, acc(Xall[:, s1],  ytr[s1]))
        @printf("  %-28s ep %2d  set %d   test %.4f  current %.4f  set1 %.4f  (%.0f s)\n",
                name, ep, k, te[end], cu[end], s1a[end], time()-t0); flush(stdout)
    end
    (name=name, nfeat=nf, test=te, current=cu, set1=s1a, switches=sw, switching=switching)
end

# ── the subset partition: identical for both arms, so they see the same data in the same order
Random.seed!(SEED)
perm = randperm(112800)
sets = [perm[(k-1)*28200+1 : k*28200] for k in 1:NSET]
@printf("4 subsets of %d, switching every %d epochs, %d epochs total\n\n", length(sets[1]),
        PER, EPOCHS)

results = Any[]
"ours" in ARMS && let (Ftr, ytr, Fte, yte) = load_ours()
    @printf("── ours: %d features ──\n", size(Ftr, 2))
    push!(results, run_arm("ours (switching)", Ftr, ytr, Fte, yte, sets; switching=true))
    push!(results, run_arm("ours (set 1 only)", Ftr, ytr, Fte, yte, sets; switching=false))
end
"convnext" in ARMS && let (Ftr, ytr, Fte, yte) = load_cx(4)
    @printf("\n── frozen ConvNeXt stage 4: %d features ──\n", size(Ftr, 2))
    push!(results, run_arm("ConvNeXt (switching)", Ftr, ytr, Fte, yte, sets; switching=true))
    push!(results, run_arm("ConvNeXt (set 1 only)", Ftr, ytr, Fte, yte, sets; switching=false))
end

serialize(joinpath(@__DIR__, "curriculum_$(join(ARMS,'_')).jls"), (results=results, per=PER, nset=NSET))
println("\nwrote curriculum.jls")
