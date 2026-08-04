# ── PLAIN SCRIPT, not a Pluto notebook ──────────────────────────────────────
# `julia --project=.. -t 14 Plot_Predictions.jl`   (from the P9_P12_SimpleStrokeTests directory)
#
# Predicted value against true value, for every property except polarity, for three arms, at
# several points during training.
#
# WHY THIS EXISTS. Every table in this project reports R², a single number per property per arm.
# R² says how much of the variance is explained but not *how* a readout is wrong — whether it is
# noisy everywhere, biased at one end, saturating, or ignoring the target and predicting the mean.
# A scatter of prediction against truth shows all of that at a glance.
#
# THREE ARMS, and they are not the same kind of thing:
#   * `our CNN`      — trained end to end on the images. Representation and readout both fitted.
#   * `our features` — 31 hand-designed numbers, fixed, with a trained MLP head.
#   * `frozen ConvNeXt` — 1024 ImageNet numbers, fixed, with the same trained MLP head.
# The last two differ *only* in the representation. The first differs in both.
#
# EPOCHS 5 / 15 / 25 / 35 / 60. The first four were requested on the assumption that everything
# had plateaued by 35. Two of the three had: frozen ConvNeXt is at ~0.93 mean validation R² after
# a SINGLE epoch and flat from ~10, and our features climb to ~0.86 by 40 and flatten. **The CNN
# had not** — its saved history swings from −0.54 at epoch 30 to +0.65 at epoch 35, with its best
# at 50. Epoch 60 is included so its panels show an endpoint rather than only a model caught
# between swings, and so the instability is visible directly in the scatter rather than inferred
# from a curve.
#
# Identical data for all three arms: the 16,000 / 4,000 i.i.d. stimuli in `P11_ConVNextTest/data`,
# read from the same files, so nothing here depends on a regeneration matching.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Statistics, Printf, Random, LinearAlgebra, Serialization, FFTW
using Flux, CUDA, Plots
include(joinpath(@__DIR__, "..", "P11_ConVNextTest", "Readout.module.jl"))
using .Readout
gr()
BLAS.set_num_threads(min(16, Sys.CPU_THREADS)); FFTW.set_num_threads(1)

const DATA  = joinpath(@__DIR__, "..", "P11_ConVNextTest", "data")
const FEAT  = joinpath(@__DIR__, "..", "P11_ConVNextTest", "features")
const CACHE = joinpath(@__DIR__, "..", "P11_ConVNextTest", "cache")
const FIG   = joinpath(@__DIR__, "figures_predictions")
const N     = 112
const NTR   = 16000
const NTE   = 4000
const SNAPS = [5, 15, 25, 35, 60]
const PROPS = [strip(l) for l in readlines(joinpath(DATA, "props.txt")) if !isempty(strip(l))]
const SHOW  = [p for p in PROPS if p != "polarity"]

read_mat(path, n, p) = permutedims(reshape(read!(open(path), Vector{Float32}(undef, n*p)), p, n))
function read_imgs(path, n)
    a = read!(open(path), Vector{Float32}(undef, n*N*N))
    [permutedims(reshape(@view(a[(i-1)*N*N+1 : i*N*N]), N, N)) for i in 1:n]
end
stage_dims(m) = parse.(Int, split(strip(read(joinpath(FEAT, "$(m)_dims.txt"), String))))

"""
MLP head identical to `Readout.mlp` — two hidden layers of 256, Adam 1e-3, batch 128 — but
returning **test predictions at each requested epoch** instead of only the best one. Predictions
are returned in the target's own units, not standardised.
"""
function mlp_snapshots(Xtr, Ytr, Xva, Yva, Xte, μy, σy; hidden=256, seed=1, bs=128)
    Random.seed!(seed)
    dev = CUDA.functional() ? gpu : cpu
    A = dev(permutedims(Xtr)); T = dev(permutedims(Xte))
    Yt = dev(permutedims(Ytr))
    m = Chain(Dense(size(Xtr,2) => hidden, relu), Dense(hidden => hidden, relu),
              Dense(hidden => size(Ytr,2))) |> dev
    opt = Flux.setup(Flux.Adam(1f-3), m); n = size(A,2)
    out = Dict{Int,Matrix{Float32}}()
    for e in 1:maximum(SNAPS)
        for i in Iterators.partition(randperm(n), bs)
            _, gs = Flux.withgradient(mm -> Flux.mse(mm(A[:,i]), Yt[:,i]), m)
            Flux.update!(opt, m, gs[1])
        end
        e in SNAPS && (out[e] = permutedims(Array(m(T))) .* σy' .+ μy')
    end
    out
end

"The Phase 9 `:big` CNN, copied so the arm here is the same network the tables report."
cnn_model(nout) =
    Chain(Conv((5,5), 1=>32; pad=2),  BatchNorm(32, relu),  MaxPool((2,2)),
          Conv((3,3), 32=>64; pad=1), BatchNorm(64, relu),  MaxPool((2,2)),
          Conv((3,3), 64=>128; pad=1),BatchNorm(128, relu), MaxPool((2,2)),
          Conv((3,3), 128=>128; pad=1),BatchNorm(128, relu),MaxPool((2,2)),
          Flux.flatten, Dense(128*7*7 => 256, relu), Dense(256 => nout))

function cnn_snapshots(Itr, Ytr, Ite, μy, σy; seed=1, bs=64)
    Random.seed!(seed)
    dev = CUDA.functional() ? gpu : cpu
    m = cnn_model(size(Ytr,2)) |> dev
    opt = Flux.setup(Flux.Adam(1f-3), m)
    Xtr = dev(Itr); Yt = dev(permutedims(Ytr)); n = size(Itr,4)
    out = Dict{Int,Matrix{Float32}}()
    # batch norm behaves differently in training and inference, so predictions are taken in
    # test mode and the model put back afterwards
    function predict(M)
        Flux.testmode!(m)
        P = reduce(hcat, [Array(m(dev(M[:,:,:,j]))) for j in Iterators.partition(1:size(M,4), 500)])
        Flux.trainmode!(m); permutedims(P) .* σy' .+ μy'
    end
    for e in 1:maximum(SNAPS)
        for i in Iterators.partition(randperm(n), bs)
            _, gs = Flux.withgradient(mm -> Flux.mse(mm(Xtr[:,:,:,i]), Yt[:,i]), m)
            Flux.update!(opt, m, gs[1])
        end
        e in SNAPS && (out[e] = predict(Ite))
        e % 10 == 0 && (@printf("      cnn epoch %d\n", e); flush(stdout))
    end
    out
end

function main()
    mkpath(FIG)
    @printf("Predicted vs true, %d test images, epochs %s\n\n", NTE, join(SNAPS, "/"))
    np = length(PROPS)
    Ytr = read_mat(joinpath(DATA, "iid_train_y.f32"), NTR, np)
    Yte = read_mat(joinpath(DATA, "iid_test_y.f32"),  NTE, np)
    nva = NTR ÷ 6; tr = 1:NTR-nva; va = NTR-nva+1:NTR
    μy, σy = zfit(Ytr[tr, :])
    Zt = zapply(Ytr[tr, :], μy, σy); Zv = zapply(Ytr[va, :], μy, σy)

    preds = Dict{String,Dict{Int,Matrix{Float32}}}()

    # ── our features
    Ftr, Fte = deserialize(joinpath(CACHE, "ours_g1_iid_$(NTR)_$(NTE).jls"))
    μ, σ = zfit(Ftr[tr, :]); A = zapply(Ftr, μ, σ); T = zapply(Fte, μ, σ)
    t = @elapsed (preds["our features (31)"] =
        mlp_snapshots(A[tr,:], Zt, A[va,:], Zv, T, μy, σy))
    @printf("  our features   %5.0f s\n", t); flush(stdout)

    # ── frozen ConvNeXt, stage 4, the arm the tables report
    d = stage_dims("base")
    Ctr = read_mat(joinpath(FEAT, "base_iid_train_s4.f32"), NTR, d[4])
    Cte = read_mat(joinpath(FEAT, "base_iid_test_s4.f32"),  NTE, d[4])
    μ, σ = zfit(Ctr[tr, :]); A = zapply(Ctr, μ, σ); T = zapply(Cte, μ, σ)
    t = @elapsed (preds["frozen ConvNeXt (1024)"] =
        mlp_snapshots(A[tr,:], Zt, A[va,:], Zv, T, μy, σy))
    @printf("  frozen ConvNeXt %4.0f s\n", t); flush(stdout)
    Ctr = nothing; Cte = nothing; GC.gc()

    # ── our CNN, end to end
    itr = read_imgs(joinpath(DATA, "iid_train_img.f32"), NTR)
    ite = read_imgs(joinpath(DATA, "iid_test_img.f32"),  NTE)
    to4(v) = reshape(reduce(hcat, [vec(x) for x in v]), N, N, 1, length(v))
    Itr = to4(itr[tr]); Ite = to4(ite)
    itr = nothing; ite = nothing; GC.gc()
    t = @elapsed (preds["our CNN (end to end)"] = cnn_snapshots(Itr, Zt, Ite, μy, σy))
    @printf("  our CNN        %5.0f s\n", t); flush(stdout)
    Itr = nothing; Ite = nothing; GC.gc()

    # ── one figure per property: rows = arms, columns = epochs
    arms = ["our CNN (end to end)", "our features (31)", "frozen ConvNeXt (1024)"]
    for p in SHOW
        j = findfirst(==(p), PROPS)
        y = Float64.(Yte[:, j])
        lo, hi = extrema(y); pad = 0.05*(hi-lo); ax = (lo-pad, hi+pad)
        panels = []
        for a in arms, e in SNAPS
            ŷ = Float64.(preds[a][e][:, j])
            r = 1 - sum(abs2, ŷ .- y)/sum(abs2, y .- mean(y))
            pl = scatter(y, ŷ; ms=1.1, msw=0, alpha=0.10, c=:steelblue, legend=false,
                         xlims=ax, ylims=ax, aspect_ratio=:equal,
                         title=@sprintf("%s\ne%d   R²=%.3f", a, e, r), titlefontsize=6,
                         tickfontsize=5, grid=false)
            plot!(pl, [ax[1], ax[2]], [ax[1], ax[2]]; c=:black, lw=0.8, ls=:dash)
            push!(panels, pl)
        end
        fig = plot(panels...; layout=(length(arms), length(SNAPS)),
                   size=(230*length(SNAPS), 250*length(arms)),
                   plot_title="$p — predicted (y) against true (x)", plot_titlefontsize=11,
                   left_margin=3Plots.mm, bottom_margin=3Plots.mm)
        savefig(fig, joinpath(FIG, "pred_$(p).png"))
        @printf("  wrote pred_%s.png\n", p); flush(stdout)
    end
    serialize(joinpath(FIG, "predictions.jls"), (preds=preds, Yte=Yte, props=PROPS, snaps=SNAPS))
    println("\nwrote $FIG")
end

main()
