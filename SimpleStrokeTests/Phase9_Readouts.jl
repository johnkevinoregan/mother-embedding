# ── PLAIN SCRIPT, not a Pluto notebook ──────────────────────────────────────
# Run with `julia --project=.. -t 16 Phase9_Readouts.jl`
#
# Phase 9 — how much of each contour property is *explicit* in each representation?
#
# Phase 8 asked whether a network could separate some classes and the answer was always
# yes, which is why it produced no information. The question here is sharper: fit a
# **linear** readout and see how much of each property it recovers. A property is explicit
# if one weighted sum gets it; if it takes a hidden layer to dig out, the representation
# contains it but has not made it available. That distinction is what the front end is for.
#
# FIVE ARMS. Rows 1-2 are the floor, row 3 the strong baseline, rows 4-5 the front end.
#
#   1  pixels        linear     what a fixed template can read off the image
#   2  pixels        MLP        pixels plus a learned nonlinearity
#   3  CNN           learned    features learned *for these targets*
#   4  our features  linear     <- the measurement
#   5  our features  MLP
#
# Row 4 minus row 1 is what the front end made explicit that was not already. Row 4 against
# row 3 is the harder question, and note the asymmetry: the CNN builds a representation
# optimised for these eight targets, ours was designed without seeing them, and the pixel
# probe has 12,544 free parameters against our 279. Every asymmetry runs against us, which
# is what would make a win worth something.
#
# SCORED AGAINST A TRIVIAL BASELINE, NOT AGAINST ZERO. `closedness` is ~30 % predictable
# from three scalar image summaries alone, for the geometric reason that turn = curvature ×
# length so total ink carries turn. Reporting raw R² would credit every arm for that.
#
# THREE EXTRAPOLATION SPLITS, which are the point. i.i.d. accuracy mostly measures
# capacity; invariance only shows when the test set holds nuisance values training never
# contained. In each split the held-out nuisance's own target row is dropped — a model
# cannot be scored on predicting something it never saw vary.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Statistics, Printf, Random, LinearAlgebra, Serialization, FFTW
using Flux, Plots, CUDA
include(joinpath(@__DIR__, "Contours.module.jl"))
include(joinpath(@__DIR__, "Frontend.module.jl"))
using .Contours, .Frontend

const N      = 112
const NTRAIN = parse(Int, get(ENV, "P9_NTRAIN", "16000"))
const NTEST  = parse(Int, get(ENV, "P9_NTEST",  "4000"))
const KS     = [parse(Int, s) for s in split(get(ENV, "P9_KS", "500,2000,6000,16000"), ",")]
const EPOCHS = parse(Int, get(ENV, "P9_EPOCHS", "60"))
const CEPOCH = parse(Int, get(ENV, "P9_CEPOCHS", "18"))
const OUT    = joinpath(@__DIR__, get(ENV, "P9_OUT", "results"))
# Which arms to run. The CNN is ~95 % of the wall time, so `P9_ARMS=1,2,4,5` gives every
# other arm — including the whole measurement the experiment is built around — in minutes
# rather than hours.
const ARMSEL = [parse(Int, s) for s in split(get(ENV, "P9_ARMS", "1,2,3,4,5"), ",")]
# Which stages to run, and which arms appear on the sample-efficiency curve. The curve
# defaults to the linear arms because a CNN at every k costs about as much as the rest of
# the experiment put together; `P9_CURVE_ARMS=1,3,4` adds it when that is what is wanted.
const STAGES = split(get(ENV, "P9_STAGES", "iid,blocks,curve,extrap"), ",")
const CURVEARMS = [parse(Int, s) for s in split(get(ENV, "P9_CURVE_ARMS", "1,4"), ",")]
# `small` is the CPU-era baseline: two strided convolutions, kept so the published numbers
# stay reproducible. `big` is what the GPU makes affordable — full resolution at the first
# layer, four convolutional stages, batch norm. Striding at the very first layer discards
# fine detail immediately, which is the most likely reason the small net never learned
# `brokenness` at any sample size.
const CNNARCH = Symbol(get(ENV, "P9_CNN", "small"))
const USEGPU  = get(ENV, "P9_GPU", "1") == "1"
# Pooling grid. 3×3 gives 37 px cells on a 112 px image, which is coarse next to a 3 px gap;
# `P9_GRID=5` asks whether `brokenness` and `vangle` are limited by the operators or by the
# pooling. 5×5 takes the feature count from 279 to 775.
const GRID    = parse(Int, get(ENV, "P9_GRID", "3"))
const SCALEMODE = Symbol(get(ENV, "P9_SCALEMODE", "per_scale"))
# How many independent permutations to average the shuffle control over. One permutation is
# a sample of size one, and the conjunction-layer claim rests on this control.
const NSHUF   = parse(Int, get(ENV, "P9_NSHUF", "5"))

# Per-epoch training history, keyed by "<split>/<arm>". Kept for every trained arm, not just
# the CNN: a final number cannot tell you whether a run had converged, was still climbing,
# or had peaked and started overfitting, and all three have appeared in this project.
const HIST = Dict{String,Any}()
const TAG  = Ref("iid")
BLAS.set_num_threads(min(16, Sys.CPU_THREADS)); FFTW.set_num_threads(1)
mkpath(OUT)

# ── metrics ─────────────────────────────────────────────────────────────────

"""
Fraction of variance explained. Negative means worse than predicting the mean.

`NaN` when the target has no variance to explain: in an extrapolation split the held-out
nuisance is constant across the test set, and R² is then a division by zero that prints as
`-Inf` and reads like a catastrophic score rather than an undefined one.
"""
function r2(ŷ, y)
    sst = sum(abs2, y .- mean(y))
    sst <= 0 ? NaN : 1 - sum(abs2, ŷ .- y) / sst
end

"""
Per-property R² of a linear fit on three scalar image summaries — total contrast mass, mean
level, sd. This is the floor every arm has to clear to have shown anything, and it exists
because Phase 8 mistook a cue nobody had measured for a result.
"""
function trivial_baseline(imgs, Ytr, Yte, ntr)
    summ(im) = (sum(abs.(im .- median(im))), mean(im), std(im))
    S = reduce(vcat, [collect(summ(im))' for im in imgs])
    A = hcat(ones(size(S,1)), S)
    Atr, Ate = A[1:ntr, :], A[ntr+1:end, :]
    [r2(Ate * (Atr \ Ytr[:, j]), Yte[:, j]) for j in 1:size(Ytr, 2)]
end

"""
Mean over the properties that have a defined R².

A held-out nuisance is constant in training, so its own row has zero variance and scores
`NaN`. Averaging that in makes the selection metric `NaN`, `NaN > best` is false at every
epoch, and the model records no prediction at all — which showed up as both MLP arms and
the CNN scoring a flat 0.000 across the polarity split, looking exactly like three
independent collapses rather than one bug in a comparison.
"""
nanmean(v) = (u = filter(!isnan, v); isempty(u) ? NaN : mean(u))

# ── readouts ────────────────────────────────────────────────────────────────

"""
    zfit(X) -> (μ, σ)

Column standardisation fitted on training data only. Constant columns get σ = 1 so they
map to zero rather than to NaN — a held-out nuisance makes whole columns constant, and
silently producing NaNs there would poison every property, not just the held-out one.
"""
function zfit(X)
    μ = vec(mean(X, dims=1)); σ = vec(std(X, dims=1)); σ[σ .<= 1e-8] .= 1f0
    Float32.(μ), Float32.(σ)
end
zapply(X, μ, σ) = (X .- μ') ./ σ'

"""
    ridge(Xtr, Ytr, Xva, Yva, Xte) -> (predictions, λ per property)

Closed-form ridge with the penalty chosen per property on a validation split. Per-property
rather than one shared λ because the properties differ by orders of magnitude in how much
signal they carry, and a single λ tuned on their average would under-regularise the easy
rows and over-regularise the hard ones.

Solved through the normal equations. With 12,544 pixel columns `X'X` is 630 MB in Float32,
which is affordable here and lets every λ reuse the same Gram matrix.
"""
function ridge(Xtr, Ytr, Xva, Yva, Xte; cs=Float32[1f-3, 1f-2, 1f-1, 1f0, 1f1, 1f2, 1f3])
    G  = Xtr' * Xtr
    B  = Xtr' * Ytr
    P  = size(Ytr, 2)
    best = fill(-Inf, P); bestλ = zeros(Float32, P)
    Pte = zeros(Float32, size(Xte, 1), P)
    # The penalty is scaled by the mean diagonal of the Gram matrix rather than given in
    # absolute units. Columns are standardised, so that diagonal is ≈ n and the same grid
    # then means the same thing whatever the sample size — and with 12,544 pixel columns
    # against a few hundred rows, an absolute λ of 0.01 leaves the matrix numerically
    # singular in Float32 and the factorisation simply fails.
    scale = Float32(mean(diag(G)))
    for c in cs
        F = try cholesky(Symmetric(G + (c*scale)*I)) catch; continue end
        W = F \ B
        Va = Xva * W; Te = Xte * W
        for j in 1:P
            s = r2(Va[:, j], Yva[:, j])
            if s > best[j]; best[j] = s; bestλ[j] = c*scale; Pte[:, j] = Te[:, j]; end
        end
    end
    Pte, bestλ
end

"""
    mlp(Xtr, Ytr, Xva, Yva, Xte; hidden) -> predictions

One hidden layer, Adam, with the epoch chosen by validation R². The reported prediction is
from the best epoch, not the last, so a diverging run is scored where it was actually good
rather than where it stopped.
"""
function mlp(Xtr, Ytr, Xva, Yva, Xte; hidden=256, epochs=EPOCHS, seed=1, bs=128, tag="MLP")
    Random.seed!(seed)
    # The MLP arms run on the GPU too, so that the epoch budget can be raised to match the
    # CNN's. Giving one arm 100 epochs and another 50 would make the comparison a statement
    # about training budget rather than about representation.
    dev = (USEGPU && CUDA.functional()) ? gpu : cpu
    A = dev(permutedims(Xtr)); V = dev(permutedims(Xva)); T = dev(permutedims(Xte))
    Yt = dev(permutedims(Ytr)); Yv = permutedims(Yva)
    m = Chain(Dense(size(Xtr,2) => hidden, relu), Dense(hidden => hidden, relu),
              Dense(hidden => size(Ytr,2))) |> dev
    opt = Flux.setup(Flux.Adam(1f-3), m)
    n = size(A, 2); best = -Inf; bestP = zeros(Float32, size(T,2), size(Yt,1))
    hist = fill(NaN, epochs, size(Yv, 1)); lossh = fill(NaN, epochs)
    for e in 1:epochs
        tot = 0.0; nb = 0
        for i in Iterators.partition(randperm(n), bs)
            l, gs = Flux.withgradient(mm -> Flux.mse(mm(A[:, i]), Yt[:, i]), m)
            Flux.update!(opt, m, gs[1]); tot += l; nb += 1
        end
        Pv = cpu(m(V))
        hist[e, :] = [r2(vec(Pv[j, :]), vec(Yv[j, :])) for j in 1:size(Yv,1)]
        lossh[e] = tot / nb
        s = nanmean(hist[e, :])
        if s > best; best = s; bestP = permutedims(cpu(m(T))); end
    end
    HIST["$(TAG[])/$(tag)"] = (val=hist, loss=lossh, props=PROPS, best=best)
    bestP
end

"""
    cnn_model(nout, arch)

The two CNN architectures.

`:small` is the CPU-era baseline — two **strided** convolutions and a head, with 97 % of its
817 k parameters in the first dense layer. Striding at the very first layer throws fine
spatial detail away immediately, which is the most likely reason it never learned
`brokenness` at any sample size: a 3 px gap is barely more than one stride. It is kept so
the published numbers stay reproducible, not because it is a good design.

`:big` is a conventional convolutional network: **full resolution at the first layer**,
pooling after each of four convolutional stages, batch normalisation throughout, 3×3 kernels
after the first. Roughly 15× the arithmetic per image, which is why it needed a GPU.

    112² × 1  → conv 5×5 → 32, BN, pool → 56² × 32
              → conv 3×3 → 64, BN, pool → 28² × 64
              → conv 3×3 → 128, BN, pool → 14² × 128
              → conv 3×3 → 128, BN, pool →  7² × 128
              → dense 256 → dense 8
"""
cnn_model(nout, arch) =
    arch === :small ?
        Chain(Conv((5,5), 1=>16, relu; stride=2, pad=2),
              Conv((5,5), 16=>32, relu; stride=2, pad=2),
              MaxPool((2,2)), Flux.flatten,
              Dense(32*14*14 => 128, relu), Dense(128 => nout)) :
        Chain(Conv((5,5), 1=>32; pad=2),  BatchNorm(32, relu),  MaxPool((2,2)),
              Conv((3,3), 32=>64; pad=1), BatchNorm(64, relu),  MaxPool((2,2)),
              Conv((3,3), 64=>128; pad=1),BatchNorm(128, relu), MaxPool((2,2)),
              Conv((3,3), 128=>128; pad=1),BatchNorm(128, relu),MaxPool((2,2)),
              Flux.flatten, Dense(128*7*7 => 256, relu), Dense(256 => nout))

function cnn(Itr, Ytr, Iva, Yva, Ite; epochs=CEPOCH, seed=1, bs=64, arch=CNNARCH)
    Random.seed!(seed)
    dev = (USEGPU && CUDA.functional()) ? gpu : cpu
    m = cnn_model(size(Ytr, 2), arch) |> dev
    opt = Flux.setup(Flux.Adam(1f-3), m)
    Xtr = dev(Itr); Yt = dev(permutedims(Ytr)); Yv = permutedims(Yva)
    n = size(Itr, 4); best = -Inf; bestP = zeros(Float32, size(Ite,4), size(Yv,1))
    # batch norm behaves differently in training and inference, so predictions are taken in
    # test mode and training mode restored afterwards. Getting this wrong would make the
    # validation score a different quantity from the test score.
    function predict(M)
        Flux.testmode!(m)
        out = reduce(hcat, [cpu(m(dev(M[:,:,:,j]))) for j in Iterators.partition(1:size(M,4), 512)])
        Flux.trainmode!(m); out
    end
    hist = fill(NaN, epochs, size(Yv, 1)); lossh = fill(NaN, epochs)
    for e in 1:epochs
        tot = 0.0; nb = 0
        for i in Iterators.partition(randperm(n), bs)
            l, gs = Flux.withgradient(mm -> Flux.mse(mm(Xtr[:,:,:,i]), Yt[:, i]), m)
            Flux.update!(opt, m, gs[1]); tot += l; nb += 1
        end
        P = predict(Iva)
        hist[e, :] = [r2(vec(P[j, :]), vec(Yv[j, :])) for j in 1:size(Yv,1)]
        lossh[e] = tot / nb
        s = nanmean(hist[e, :])
        e % 5 == 0 && (@printf("      cnn[%s] epoch %3d  loss %.4f  val R² %.3f\n",
                               arch, e, lossh[e], s); flush(stdout))
        if s > best; best = s; bestP = permutedims(predict(Ite)); end
    end
    HIST["$(TAG[])/CNN"] = (val=hist, loss=lossh, props=PROPS, best=best)
    @printf("      cnn[%s] best val R² %.3f after %d epochs\n", arch, best, epochs); flush(stdout)
    bestP
end

# ── data ────────────────────────────────────────────────────────────────────

"""
    make_split(ntr, nte, seed; train_kw, test_kw)

Train and test drawn from separate RNG streams. The generator is parametric and unbounded,
so a test image is a fresh draw rather than a held-out slice — leakage is impossible by
construction, unlike a fixed corpus.

`train_kw` and `test_kw` differ only for the extrapolation splits, where the test set is
restricted to nuisance values the training set never contained.
"""
function make_split(ntr, nte, seed; train_kw=(), test_kw=())
    itr, Ytr, _, _ = contour_batch(ntr, seed;      N=N, train_kw...)
    ite, Yte, _, _ = contour_batch(nte, seed+1000; N=N, test_kw...)
    itr, Float32.(Ytr), ite, Float32.(Yte)
end

to4d(imgs) = reshape(reduce(hcat, [vec(im) for im in imgs]), N, N, 1, length(imgs))
toflat(imgs) = permutedims(reduce(hcat, [vec(im) for im in imgs]))

# ── one evaluation ──────────────────────────────────────────────────────────

const ARMS = ["pixels·linear", "pixels·MLP", "CNN", "ours·linear", "ours·MLP"]

"""
    evaluate(...) -> arm × property matrix of test R²

`drop` names the target row to exclude, used by the extrapolation splits: when training
holds polarity fixed, the polarity row is constant in training and no model can be scored
on it. Its R² is returned as `NaN` rather than as a number that looks like a result.
"""
function evaluate(itr, Ytr, ite, Yte, spec; drop=nothing, arms=ARMSEL, ntrain=nothing,
                  Xflat=nothing, Feat=nothing)
    nfull = length(itr)
    ntr = ntrain === nothing ? nfull : min(ntrain, nfull)
    itr = itr[1:ntr]; Ytr = Ytr[1:ntr, :]
    # Validation is a sixth of whatever training set this call was given, floored at 40.
    # A fixed floor of 200 exceeded the training set at the smallest k on the
    # sample-efficiency curve, leaving an empty training range and an all-NaN row.
    nva = clamp(ntr ÷ 6, 40, ntr - 40); va = ntr-nva+1:ntr; tr = 1:ntr-nva

    μy, σy = zfit(Ytr[tr, :])
    Zt = zapply(Ytr[tr, :], μy, σy); Zv = zapply(Ytr[va, :], μy, σy)
    unz(P) = P .* σy' .+ μy'

    R = fill(NaN, length(ARMS), size(Ytr, 2))
    score!(k, P) = for j in 1:size(Yte, 2)
        R[k, j] = (drop !== nothing && PROPS[j] === drop) ? NaN : r2(unz(P)[:, j], Yte[:, j])
    end

    if 1 in arms || 2 in arms
        # `Xflat` and `Feat` are the whole pool, computed once by the caller: the
        # sample-efficiency curve calls this five times and re-deriving 16,000 feature
        # vectors each time would cost more than every fit in the experiment put together.
        Xa = Xflat === nothing ? toflat(vcat(itr, ite)) : vcat(Xflat[1:ntr, :], Xflat[nfull+1:end, :])
        μ, σ = zfit(Xa[tr, :]); Xa = zapply(Xa, μ, σ)
        Xtr, Xva, Xte = Xa[tr, :], Xa[va, :], Xa[ntr+1:end, :]
        1 in arms && score!(1, ridge(Xtr, Zt, Xva, Zv, Xte)[1])
        2 in arms && score!(2, mlp(Xtr, Zt, Xva, Zv, Xte; tag="pixels·MLP"))
        Xa = nothing; GC.gc()
    end
    if 3 in arms
        Ia = to4d(vcat(itr, ite))
        score!(3, cnn(Ia[:,:,:,tr], Zt, Ia[:,:,:,va], Zv, Ia[:,:,:,ntr+1:end]))
        Ia = nothing; GC.gc()
    end
    if 4 in arms || 5 in arms
        Fa = Feat === nothing ? featurize(vcat(itr, ite), spec) : vcat(Feat[1:ntr, :], Feat[nfull+1:end, :])
        μ, σ = zfit(Fa[tr, :]); Fa = zapply(Fa, μ, σ)
        Ftr, Fva, Fte = Fa[tr, :], Fa[va, :], Fa[ntr+1:end, :]
        4 in arms && score!(4, ridge(Ftr, Zt, Fva, Zv, Fte)[1])
        5 in arms && score!(5, mlp(Ftr, Zt, Fva, Zv, Fte; tag="ours·MLP"))
    end
    R
end

function show_table(title, R, base=nothing)
    println("\n" * "="^92); println(title); println("="^92)
    @printf("\n%-15s", "arm"); for p in PROPS; @printf("%11s", String(p)[1:min(10,end)]); end
    println()
    if base !== nothing
        @printf("%-15s", "trivial"); for b in base; @printf("%11.3f", b); end; println()
        println("-"^92)
    end
    for (k, nm) in enumerate(ARMS)
        @printf("%-15s", nm)
        for j in 1:size(R,2); isnan(R[k,j]) ? @printf("%11s", "—") : @printf("%11.3f", R[k,j]); end
        println()
    end
end

# ── run ─────────────────────────────────────────────────────────────────────

function main()
    @printf("Phase 9 — %d train, %d test, %d threads\n\n", NTRAIN, NTEST, Threads.nthreads())
    spec = build_frontend(N; grid=GRID, scale_mode=SCALEMODE)
    @printf("front end: grid %d, %d columns  (orient %d, lowpass %d, A1 %d, A2 %d, rays %d)\n\n",
            GRID, spec.n, length(block_cols(spec,"orient")), length(block_cols(spec,"lowpass")),
            length(block_cols(spec,"A1")), length(block_cols(spec,"A2")),
            length(block_cols(spec,"rays")))

    t = @elapsed ((itr, Ytr, ite, Yte) = make_split(NTRAIN, NTEST, 1))
    @printf("generated %d images in %.0fs\n", NTRAIN+NTEST, t); flush(stdout)

    base = trivial_baseline(vcat(itr, ite), Ytr, Yte, NTRAIN)
    t = @elapsed (Feat = featurize(vcat(itr, ite), spec))
    @printf("featurised in %.0fs (%.1f ms/img)\n", t, 1000t/(NTRAIN+NTEST)); flush(stdout)
    Xflat = toflat(vcat(itr, ite))
    if "iid" in STAGES
        TAG[] = "iid"
        R = evaluate(itr, Ytr, ite, Yte, spec; Xflat=Xflat, Feat=Feat)
        show_table("i.i.d. split — test R² per property, all $NTRAIN training images", R, base)
        serialize(joinpath(OUT, "iid.jls"), (R=R, base=base, props=PROPS, arms=ARMS))
    end

    # ── block attribution: which part of the representation carries each property?
    # The sharp prediction is on `polarity`. Quadrature energy discards the sign of contrast
    # by construction, so `orient` should fail to predict it — being unable to is the
    # correct result — while `lowpass`, which carries mean level, should get it easily. If
    # `orient` predicts polarity the invariance claim is simply wrong.
    "blocks" in STAGES && println("\n" * "="^92)
    "blocks" in STAGES && println("Block attribution — linear readout on one block at a time (test R²)")
    println("="^92)
    nva = clamp(NTRAIN ÷ 6, 40, NTRAIN - 40)
    tr = 1:NTRAIN-nva; va = NTRAIN-nva+1:NTRAIN
    μy, σy = zfit(Ytr[tr, :]); Zt = zapply(Ytr[tr, :], μy, σy); Zv = zapply(Ytr[va, :], μy, σy)
    @printf("\n%-16s", "block")
    for p in PROPS; @printf("%11s", String(p)[1:min(10,end)]); end; println()
    blocks = [("orient", ("orient",)), ("lowpass", ("lowpass",)), ("A1+A2", ("A1","A2")),
              ("rays", ("rays",)), ("all", ("orient","lowpass","A1","A2","rays")),
              ("all·SHUFFLED", ("orient","lowpass","A1","A2","rays")),
              # Leave-one-out with matched capacity. `all` minus each of these is the unique
              # contribution of the block that was scrambled, with column count and marginals
              # held fixed. Needed because the ray transform is **linear** in the energy field
              # while A1 and A2 are bilinear: if the gain is mostly `rays`, the useful
              # ingredient is the geometry of the sampling rather than the conjunction, and
              # "AND layer" would be the wrong name for the result.
              ("all·noRAYS", ("orient","lowpass","A1","A2","rays")),
              ("all·noA", ("orient","lowpass","A1","A2","rays")),
              # The ray block is 81 of 279 columns — 29 % of the representation — split into
              # three harmonics of the ray transform: c₀ (≈ how many branches), |c₁|/c₀ (how
              # asymmetric) and |c₂|/c₀. Scrambling one at a time says which of them is
              # actually carrying anything, and whether 81 columns are earning their place.
              ("all·noR0", ("orient","lowpass","A1","A2","rays")),
              ("all·noR1", ("orient","lowpass","A1","A2","rays")),
              ("all·noR2", ("orient","lowpass","A1","A2","rays")),
              ("rays.R0", ("rays.R0",)), ("rays.R1", ("rays.R1",)), ("rays.R2", ("rays.R2",))]
    battr = Dict{String,Vector{Float64}}()
    shufruns = Vector{Vector{Float64}}()
    for (nm, bs) in ("blocks" in STAGES ? blocks : [])
      # the control is averaged over NSHUF independent permutations; the others run once
      reps = startswith(nm, "all·") ? (1:NSHUF) : (1:1)
      acc = Vector{Vector{Float64}}()
      for rep in reps
        cols = block_cols(spec, bs...)
        Fw = Feat
        if startswith(nm, "all·")
            # Capacity control. `all` has 279 columns against `orient`'s 135, so part of any
            # gain could be the extra parameters rather than extra information. Here the
            # conjunction and ray columns are permuted *across samples*: identical column
            # count, identical marginals, correspondence with the image destroyed. Whatever
            # this row scores above `orient` is what capacity alone buys, and the real claim
            # is `all` minus this, not `all` minus `orient`.
            Fw = copy(Feat)
            bad = nm == "all·noRAYS" ? block_cols(spec, "rays") :
                  nm == "all·noA"     ? block_cols(spec, "A1", "A2") :
                  nm == "all·noR0"    ? block_cols(spec, "rays.R0") :
                  nm == "all·noR1"    ? block_cols(spec, "rays.R1") :
                  nm == "all·noR2"    ? block_cols(spec, "rays.R2") :
                                        block_cols(spec, "A1", "A2", "rays")
            perm = randperm(MersenneTwister(31 + rep), size(Fw, 1))
            Fw[:, bad] = Fw[perm, bad]
        end
        μ, σ = zfit(Fw[tr, cols]); Z = zapply(Fw[:, cols], μ, σ)
        P, _ = ridge(Z[tr, :], Zt, Z[va, :], Zv, Z[NTRAIN+1:end, :])
        push!(acc, [r2(P[:, j] .* σy[j] .+ μy[j], Yte[:, j]) for j in 1:length(PROPS)])
      end
      v = [mean(a[j] for a in acc) for j in 1:length(PROPS)]
      battr[nm] = v
      nm == "all·SHUFFLED" && (shufruns = acc)
      @printf("%-16s", nm); for x in v; @printf("%11.3f", x); end
      if length(acc) > 1
          sd = [std([a[j] for a in acc]) for j in 1:length(PROPS)]
          @printf("   (mean of %d perms, max sd %.3f)", length(acc), maximum(sd))
      end
      println(); flush(stdout)
    end
    if !isempty(shufruns)
        println("\nshuffle control, per-permutation spread:")
        @printf("%-16s", "sd over perms")
        for j in 1:length(PROPS)
            @printf("%11.3f", std([a[j] for a in shufruns]))
        end
        println()
    end
    serialize(joinpath(OUT, "blocks.jls"), (battr=battr, shuf=shufruns))

    # ── sample-efficiency curve, linear arms only: the readout is the thing being
    # compared, and adding a trained CNN at every k would dominate the runtime.
    println("\n" * "="^92); println("Sample efficiency (linear readouts)"); println("="^92)
    curve = Dict{Int,Matrix{Float64}}()
    for k in ("curve" in STAGES ? KS : Int[])
        TAG[] = "curve_k$(k)"
        Rk = evaluate(itr, Ytr, ite, Yte, spec; arms=CURVEARMS, ntrain=k, Xflat=Xflat, Feat=Feat)
        curve[k] = Rk
        @printf("\n  k = %-6d", k)
        for j in 1:length(PROPS); @printf("%11s", String(PROPS[j])[1:min(10,end)]); end
        for a in CURVEARMS
            @printf("\n    %-11s", ARMS[a])
            for j in 1:length(PROPS); @printf("%11.3f", Rk[a,j]); end
        end
        println(); flush(stdout)
    end
    serialize(joinpath(OUT, "curve.jls"), curve)

    # ── extrapolation splits ────────────────────────────────────────────────
    splits = [
        (:polarity,  "trained on light strokes, tested on dark",
         (pol=1,), (pol=-1,)),
        (:fuzziness, "trained on sharp edges (ramp ≤ 3 px), tested on blurred (≥ 8 px)",
         (ramp=(0.8, 3.0),), (ramp=(8.0, 20.0),)),
        (:thickness, "trained on thin strokes (≤ 6 px), tested on thick (≥ 8 px)",
         (w=(3.0, 6.0),), (w=(8.0, 12.0),)),
    ]
    for (nm, desc, tkw, ekw) in ("extrap" in STAGES ? splits : [])
        TAG[] = "extrap_$(nm)"
        itr2, Ytr2, ite2, Yte2 = make_split(NTRAIN, NTEST, 500; train_kw=tkw, test_kw=ekw)
        R2 = evaluate(itr2, Ytr2, ite2, Yte2, spec; drop=nm)
        b2 = trivial_baseline(vcat(itr2, ite2), Ytr2, Yte2, NTRAIN)
        show_table("extrapolation: $nm — $desc", R2, b2)
        serialize(joinpath(OUT, "extrap_$nm.jls"), (R=R2, base=b2))
        flush(stdout)
    end

    serialize(joinpath(OUT, "history.jls"), HIST)
    println("\nwrote $(OUT)/  (including per-epoch history for every trained arm)")
end

main()
