# ── PLAIN SCRIPT, not a Pluto notebook ──────────────────────────────────────
# `julia --project=.. -t 16 Phase10b_Collinearity.jl`   (from the P10_FashionMNIST directory)
#
# Phase 10b — why does the conjunction layer pay on garments and not on handwriting?
#
# Phase 7 explained the EMNIST null with one number: `R²(A ← orient) = 0.933`, the median
# over the 54 A columns of a least-squares fit on the 144 orient+lowpass columns. A and
# orientation energy are *different operators that become near-collinear on handwriting* —
# real letters do not produce the configurations that separate them.
#
# Phase 10 then found A₁+A₂ worth +1.62 (grid 3) and +2.85 (grid 1) over a shuffle control on
# Fashion-MNIST, against +0.01 on EMNIST. If the collinearity account is right, the same
# regression on garments must come out **markedly lower**. If it comes out just as high, then
# collinearity is not what separated the two datasets and Phase 7's explanation is incomplete.
#
# WHAT THE NUMBER MEANS, precisely. Pooled A₁ is `Σₓ w(x)·Eₖ(x)·Eₖ₊ₙ⁄₂(x) / C₀`, and the
# pooled orient block is `Σₓ w(x)·Eₖ(x)` per channel. The first is *not* a function of the
# second: the difference is the within-cell covariance `Cov_x(Eₖ, Eₖ₊ₙ⁄₂)`, which is exactly
# the co-location signal (see `Validate_AndLayer.jl`). So R² here measures **how much of the
# co-location signal is predictable from the marginal orientation profile**. High R² does not
# mean A is redundant in general — Phase 3 dissociates the two by 4.9× on stimuli built to
# require it — it means this image set does not exercise the difference.
#
# Both halves are computed by the same code on the same day, because Phase 7's cache was
# written to `tempdir()` and is long gone, and re-deriving the EMNIST number here removes any
# question of the front-end fixes since (`:replicate` padding, ray-ratio pooling) having moved
# it. Neither should affect A on EMNIST — padding is bit-identical on a zero background and A
# uses no rays — but "provably unaffected" was already falsified once by Phase 5a.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Statistics, Printf, Random, LinearAlgebra, Serialization, FFTW
include(joinpath(@__DIR__, "..", "LoadEMNIST.module.jl"))
include(joinpath(@__DIR__, "..", "P9+12_SimpleStrokeTests", "Frontend.module.jl"))
using .LoadEMNIST, .Frontend

const IMG   = 112
const CACHE = joinpath(@__DIR__, "cache")
const NEM   = parse(Int, get(ENV, "C_NEMNIST", "20000"))   # EMNIST images per split
const DOEM  = get(ENV, "C_EMNIST", "1") == "1"
BLAS.set_num_threads(min(16, Sys.CPU_THREADS)); FFTW.set_num_threads(1)

"""
Least-squares R² per target column, fit on train and scored on test — character for character
the estimator in `Phase7_FfProbe.jl`, so the numbers are comparable.

Float64 rather than Phase 7's Float32: the design matrix is 145 columns of unstandardised
energies spanning several orders of magnitude, and a rank-deficient-looking normal equation in
Float32 can lose digits. The script prints both so the change is visible rather than assumed.
"""
function r2(Xtr, Ytr, Xte, Yte)
    A = hcat(ones(eltype(Xtr), size(Xtr, 1)), Xtr)
    B = hcat(ones(eltype(Xte), size(Xte, 1)), Xte)
    W = A \ Ytr
    P = B * W
    [1 - sum(abs2, Yte[:, j] .- P[:, j]) /
         max(sum(abs2, Yte[:, j] .- mean(Yte[:, j])), eps())
     for j in 1:size(Yte, 2)]
end

function report(tag, Ftr, Fte, spec::FrontendSpec)
    OL = block_cols(spec, "orient", "lowpass")
    AB = block_cols(spec, "A1", "A2")
    RY = block_cols(spec, "rays")
    @printf("\n%s\n%s\n%s\n", "="^78, tag, "="^78)
    @printf("  %d train, %d test   |   orient+lowpass %d cols, A %d cols, rays %d cols\n",
            size(Ftr, 1), size(Fte, 1), length(OL), length(AB), length(RY))

    for (nm, T) in (("A  ← orient", AB), ("rays ← orient", RY))
        r64 = r2(Float64.(Ftr[:, OL]), Float64.(Ftr[:, T]),
                 Float64.(Fte[:, OL]), Float64.(Fte[:, T]))
        r32 = r2(Ftr[:, OL], Ftr[:, T], Fte[:, OL], Fte[:, T])
        @printf("\n  R²(%s)   median %.3f   mean %.3f   [%.3f, %.3f]\n",
                nm, median(r64), mean(r64), minimum(r64), maximum(r64))
        @printf("      columns R² > 0.9: %3d of %3d      > 0.75: %3d\n",
                count(>(0.9), r64), length(r64), count(>(0.75), r64))
        @printf("      (same in Float32: median %.3f — precision is not carrying this)\n",
                median(r32))
        flush(stdout)
    end

    # ── per scale and form ────────────────────────────────────────────────────
    # `Validate_i1D.jl` measures A₁'s response to exactly-i1D input and finds it scales
    # steeply with the orientation count: 4.6e-2 of the crossing response at ρ=2 (n=8),
    # 1.6e-4 at ρ=3.74 (n=12), 6.1e-8 at ρ=7 (n=16). Where a cell contains no junction, a
    # leaking A₁ reads `c·C₀` and is then a *pure function of orientation energy*, so if
    # leakage drives the high R², these columns must be ordered ρ=2 > ρ=3.74 > ρ=7 by four
    # orders of magnitude of leakage. If they are flat, leakage is not what makes A
    # predictable. A₂ is the control: it is an end-stop, not an orientation AND, and its
    # leakage has a different origin.
    println("\n  per scale and form — testing the i1D-leakage hypothesis")
    @printf("  %-6s %8s %8s %8s   %s\n", "form", "ρ=2.0", "ρ=3.74", "ρ=7.0", "i1D leakage of A₁")
    for form in ("A1", "A2")
        vals = Float64[]
        for ρs in ("2.0", "3.74", "7.0")
            cs = findall(l -> startswith(l, "$(form).ρ$(ρs)."), spec.labels)
            r = r2(Float64.(Ftr[:, OL]), Float64.(Ftr[:, cs]),
                   Float64.(Fte[:, OL]), Float64.(Fte[:, cs]))
            push!(vals, median(r))
        end
        @printf("  %-6s %8.3f %8.3f %8.3f   %s\n", form, vals...,
                form == "A1" ? "4.6e-02   1.6e-04   6.1e-08" : "(not an orientation AND)")
    end
    flush(stdout)
    OL, AB
end

function main()
    @printf("Phase 10b — R²(A ← orient), Fashion-MNIST against EMNIST\n")
    spec = build_frontend(IMG; grid=3)

    ck = joinpath(CACHE, "g3_n60000_10000.jls")
    isfile(ck) || error("no cached grid-3 features at $ck — run Phase10_FashionMNIST.jl first")
    Ftr, Fte = deserialize(ck)
    report("FASHION-MNIST, grid 3", Ftr, Fte, spec)
    Ftr = nothing; Fte = nothing; GC.gc()

    if DOEM
        # The test IDX files live in `emnist_source_files/`, not the directory above it —
        # same resolution Phase 5a uses.
        S = joinpath(homedir(), "Julia", "DATABASES", "EMNIST", "emnist_source_files")
        # read each file once; the images are 88 MB and indexing a fresh read per image was a
        # real bug in Phase 10.
        Atr = read_emnist_images(joinpath(S, "emnist-balanced-train-images-idx3-ubyte"))
        Ate = read_emnist_images(joinpath(S, "emnist-balanced-test-images-idx3-ubyte"))
        ntr = min(NEM, size(Atr, 3)); nte = min(NEM ÷ 2, size(Ate, 3))
        # EMNIST is 28×28 like Fashion, so it takes the same upsample the other phases use.
        # No `permutedims` here: the reader's un-transpose is *correct* for EMNIST and is only
        # wrong for Fashion-MNIST.
        ek = joinpath(CACHE, "emnist_g3_n$(ntr)_$(nte).jls")
        Etr, Ete, t = if isfile(ek)
            a, b = deserialize(ek); (a, b, 0.0)
        else
            etr = [_up(Float32.(@view Atr[:, :, i])) for i in 1:ntr]
            ete = [_up(Float32.(@view Ate[:, :, i])) for i in 1:nte]
            Atr = nothing; Ate = nothing; GC.gc()
            tt = @elapsed (a = featurize(etr, spec)); b = featurize(ete, spec)
            mkpath(CACHE); serialize(ek, (a, b)); (a, b, tt)
        end
        @printf("\n  EMNIST extraction %s\n",
                t == 0.0 ? "from cache" : @sprintf("%.0f s (%.1f ms/img)", t, 1000t / ntr))
        report("EMNIST balanced, grid 3", Etr, Ete, spec)
    end

    println("\n" * "="^78)
    println("READ: high R² ⇒ the marginal orientation profile already predicts the")
    println("co-location signal on this image set. Low R² ⇒ it does not, and the")
    println("conjunction layer has independent structure to contribute.")
end

"Bilinear 28 → 112, as every other phase does it."
function _up(img, N=IMG)
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

main()
