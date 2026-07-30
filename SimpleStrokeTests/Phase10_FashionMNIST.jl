# ── PLAIN SCRIPT, not a Pluto notebook ──────────────────────────────────────
# `julia --project=.. -t 16 Phase10_FashionMNIST.jl`
#
# Phase 10 — does the front end work on something that is not characters?
#
# Everything so far is EMNIST (handwritten letters) or SimpleStrokeTests (synthetic single
# strokes). Both are line drawings on an empty background. The project's stated target is a
# general front end for greyscale images, and neither dataset tests that.
#
# Fashion-MNIST is the cheapest step away: same 28×28 IDX format so the loader is unchanged,
# but the content is **silhouettes with texture** — weave, ribbing, sole tread — rather than
# strokes. Multi-scale oriented energy is a texture descriptor, and that has never been
# tested here.
#
# It also has something our synthetic data cannot give: **published baselines**. Roughly,
# linear on pixels ~84 %, an MLP on pixels ~88 %, a decent CNN ~93 %, SOTA ~96 %. So the
# result is calibrated against an external scale instead of only against our own arms.
#
# WHAT IT STILL DOES NOT TEST, and why BSDS is the real target: the background is black and
# uniform, there is one centred object per image, contrast barely varies within a frame, and
# the resolution is 28×28 upsampled. Every design decision that depends on image statistics —
# divisive normalisation above all — remains unadjudicated after this.
#
# PREDICTIONS, on record before running:
#
#   1. The features should land between the published MLP and CNN numbers, ~88-91 %, because
#      texture suits multi-scale oriented energy.
#   2. The AND layer should add ~0 again. Silhouettes have few junctions, which is the same
#      reason it added +0.01 on EMNIST.
#   3. **Grid 3 should beat grid 1 here** — the reverse of SimpleStrokeTests. There, grid 1
#      won because position was randomised, so a fixed grid was pure liability. Here objects
#      are centred and their parts are in consistent places (sleeves up, sole down), so
#      spatial pooling should pay. If grid 1 wins anyway, the earlier result was about
#      pooling in general rather than about position randomisation, and that would matter.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Statistics, Printf, Random, LinearAlgebra, Serialization, FFTW
using Flux, OneHotArrays, Plots
include(joinpath(@__DIR__, "..", "LoadEMNIST.module.jl"))     # IDX reader; format is identical
include(joinpath(@__DIR__, "Frontend.module.jl"))
using .LoadEMNIST, .Frontend

const DIR    = joinpath(homedir(), "Julia", "DATABASES", "FashionMNIST")
const IMG    = 112
const NTRAIN = parse(Int, get(ENV, "F_NTRAIN", "0"))     # 0 = all 60,000
const NTEST  = parse(Int, get(ENV, "F_NTEST",  "0"))
const EPOCHS = parse(Int, get(ENV, "F_EPOCHS", "25"))
const GRIDS  = [parse(Int, s) for s in split(get(ENV, "F_GRIDS", "1,3"), ",")]
const CLASSES = ["T-shirt","trouser","pullover","dress","coat",
                 "sandal","shirt","sneaker","bag","ankle boot"]
BLAS.set_num_threads(min(16, Sys.CPU_THREADS)); FFTW.set_num_threads(1)

"Bilinear 28 → 112. The same upsample the EMNIST phases use, so the two are comparable."
function upsample(img, N=IMG)
    H, W = size(img); out = zeros(Float32, N, N)
    @inbounds for i in 1:N, j in 1:N
        y = 1 + (i-1)*(H-1)/(N-1); x = 1 + (j-1)*(W-1)/(N-1)
        y0 = floor(Int, y); x0 = floor(Int, x)
        y1 = min(y0+1, H); x1 = min(x0+1, W); fy = y - y0; fx = x - x0
        out[i,j] = (1-fy)*(1-fx)*img[y0,x0] + fy*(1-fx)*img[y1,x0] +
                   (1-fy)*fx*img[y0,x1] + fy*fx*img[y1,x1]
    end
    out
end

"""
Load a split, transposing each image and leaving the labels alone.

Two traps in reusing the EMNIST reader, both found by checking rather than assuming:

* **The reader un-transposes.** EMNIST stores its images transposed relative to MNIST, so
  `read_emnist_images` corrects for that — which *introduces* a transpose on Fashion-MNIST,
  whose layout is standard. Measured: trousers came out 11.8 px tall and 27.9 px wide,
  lying on their side. Classification would largely have survived it; the prediction that a
  3×3 grid helps because parts sit in consistent places (sleeves up, sole down) would not.
* **The reader already returns 1-based labels.** Adding one more pushed them to 2–10 and
  `onehotbatch` refused them.
"""
function load_split(kind, n)
    ims = read_emnist_images(joinpath(DIR, "$(kind)-images-idx3-ubyte"))
    lbs = read_emnist_labels(joinpath(DIR, "$(kind)-labels-idx1-ubyte"))
    m = n > 0 ? min(n, size(ims, 3)) : size(ims, 3)
    [Float32.(upsample(permutedims(@view ims[:, :, i]))) for i in 1:m], Int.(lbs[1:m])
end

"""
Accuracy of a two-layer MLP on a column subset, standardised, best epoch on a validation
slice carved out of training. Same protocol as Phase 9 so the numbers are comparable.
"""
function arm(Xtr, ytr, Xte, yte, cols; hidden=256, epochs=EPOCHS, seed=1, nclass=10)
    Random.seed!(seed)
    nva = length(ytr) ÷ 6; tr = 1:length(ytr)-nva; va = length(ytr)-nva+1:length(ytr)
    μ = vec(mean(Xtr[tr, cols], dims=1)); σ = vec(std(Xtr[tr, cols], dims=1)); σ[σ .<= 1e-8] .= 1f0
    z(M) = clamp.((M[:, cols] .- μ') ./ σ', -3, 3)
    A = permutedims(z(Xtr[tr, :])); V = permutedims(z(Xtr[va, :])); T = permutedims(z(Xte))
    Yt = onehotbatch(ytr[tr], 1:nclass)
    m = Chain(Dense(length(cols) => hidden, relu), Dense(hidden => nclass))
    opt = Flux.setup(Flux.Adam(1f-3), m); n = size(A, 2); best = 0.0; bestte = 0.0
    for _ in 1:epochs
        for i in Iterators.partition(randperm(n), 128)
            _, gs = Flux.withgradient(mm -> Flux.logitcrossentropy(mm(A[:, i]), Yt[:, i]), m)
            Flux.update!(opt, m, gs[1])
        end
        v = mean(onecold(m(V), 1:nclass) .== ytr[va])
        if v > best; best = v; bestte = mean(onecold(m(T), 1:nclass) .== yte); end
    end
    100bestte
end

function main()
    @printf("Phase 10 — Fashion-MNIST, %d threads\n\n", Threads.nthreads())
    itr, ytr = load_split("train", NTRAIN); ite, yte = load_split("t10k", NTEST)
    @printf("%d train, %d test, 10 classes (chance 10.0 %%)\n", length(itr), length(ite))

    # raw pixels, as the calibration arm against the published ~88 %
    flat(v) = permutedims(reduce(hcat, [vec(x) for x in v]))
    Xp_tr = flat(itr); Xp_te = flat(ite)
    px = arm(Xp_tr, ytr, Xp_te, yte, 1:size(Xp_tr, 2))
    @printf("\npixels + MLP (calibration; published MLP ≈ 88 %%)   %.2f %%\n", px)
    Xp_tr = nothing; Xp_te = nothing; GC.gc()

    for g in GRIDS
        spec = build_frontend(IMG; grid=g)
        t = @elapsed (Ftr = featurize(itr, spec)); Fte = featurize(ite, spec)
        @printf("\n=== grid %d — %d features, extraction %.0f s (%.1f ms/img) ===\n",
                g, spec.n, t, 1000t/length(itr))
        blocks = [("orient+lowpass", ("orient","lowpass")),
                  ("+ A1+A2",        ("orient","lowpass","A1","A2")),
                  ("+ rays",         ("orient","lowpass","rays")),
                  ("everything",     ("orient","lowpass","A1","A2","rays")),
                  ("A1+A2 alone",    ("A1","A2")),
                  ("rays alone",     ("rays",))]
        for (nm, bs) in blocks
            cols = block_cols(spec, bs...)
            @printf("  %-16s %4d cols   %.2f %%\n", nm, length(cols),
                    arm(Ftr, ytr, Fte, yte, cols))
            flush(stdout)
        end
    end
end
main()
