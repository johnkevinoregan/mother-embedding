# ── PLAIN SCRIPT, not a Pluto notebook ──────────────────────────────────────
# `julia --project=.. -t 16 Sweep_Capacity.jl`   (from the SimpleStrokeTests directory)
#
# The three capacity axes the project has never swept: scales, orientations, ray offsets.
#
# The pooling grid HAS been swept (1,2,3,4,5) and grid 1 won on every property, because stroke
# position is randomised and a fixed grid is pure liability. **These three axes do not carry that
# penalty**, so they can add capacity without it — which is why they are worth trying and the grid
# was not.
#
# ONE AT A TIME, not factorial. Each axis has its own prediction, so this tests three hypotheses
# rather than searching a space; interactions are not the question and 27 cells would not answer
# it any better than 6.
#
# PREDICTIONS, on record:
#   scales 3 → 5      helps `thickness` and `fuzziness`. Phase 11 found those two confounded in
#                     our representation (blur breaks thickness, −2.70; thickening breaks blur,
#                     −2.98) and both are read from how energy spreads across scales, so three
#                     samples may simply be too coarse to separate them.
#   harmonics C₆/C₈   helps `vangle`. A corner is an orientation profile with two lobes; C₂ and C₄
#                     cannot describe two-lobe structure with independent amplitudes.
#   orientations ×2   helps `vangle` further, and is the only arm that can estimate C₈ at ρ=2 —
#                     Nyquist over 180° is n/2, so m ≤ n−2, and n=8 refuses C₈.
#   offsets ×3        helps `arms` and `brokenness`. This fills in the off-diagonal of the d × λ
#                     matrix; today only the matched diagonal exists, and the measured per-scale
#                     |c₁|/c₀ on a T-junction spans 0.13 / 0.27 / 0.38 against a theoretical
#                     0.333, so two of the three offsets are reading something else.
#
# METHOD. Reduced n for selection — extraction is the whole cost and is linear in images. Read a
# PREFIX of the stimulus files ConVNextTest/data already holds, so nothing is regenerated and every
# arm sees byte-identical images. Confirm any winner at full size separately; a configuration that
# gains i.i.d. and loses under blur has fitted the nuisance distribution, not learned anything.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Statistics, Printf, Random, LinearAlgebra, Serialization, FFTW
include(joinpath(@__DIR__, "Frontend.module.jl"))
include(joinpath(@__DIR__, "..", "ConVNextTest", "Readout.module.jl"))
using .Frontend, .Readout
BLAS.set_num_threads(min(16, Sys.CPU_THREADS)); FFTW.set_num_threads(1)

const DATA  = joinpath(@__DIR__, "..", "ConVNextTest", "data")
const N     = 112
const NTR   = parse(Int, get(ENV, "SW_NTRAIN", "3000"))
const NTE   = parse(Int, get(ENV, "SW_NTEST",  "1000"))
const SPL   = split(get(ENV, "SW_SPLITS", "iid,fuzziness,thickness"), ",")
const ARMS  = split(get(ENV, "SW_ARMS", ""), ",", keepempty=false)
const EPOCHS = parse(Int, get(ENV, "SW_EPOCHS", "60"))
const CACHE = joinpath(@__DIR__, "sweep_cache")
const PROPS = [strip(l) for l in readlines(joinpath(DATA, "props.txt")) if !isempty(strip(l))]

"Read the first `n` images from a stimulus file written row-major by ConvNextStimuli.jl."
function read_images(path, n)
    a = read!(open(path), Vector{Float32}(undef, n * N * N))
    [permutedims(reshape(@view(a[(i-1)*N*N+1 : i*N*N]), N, N)) for i in 1:n]
end
function read_matrix(path, n, p, ntot)
    a = read!(open(path), Vector{Float32}(undef, n * p))
    permutedims(reshape(a, p, n))
end

const CFGS = [
    ("baseline",            (nori=[8,12,16],  dts=0.75,  harmonics=(2,4),
                             ladder=[2.0,3.742,7.0], betas=[2.0,1.6,1.2], d_factors=(1.0,), cross_scale=:none)),
    # ── the cross-scale arms: the operator the thickness/fuzziness confound calls for ──
    ("xscale product",      (nori=[8,12,16], dts=0.75, harmonics=(2,4), ladder=[2.0,3.742,7.0],
                             betas=[2.0,1.6,1.2], d_factors=(1.0,), cross_scale=:product)),
    ("xscale ratio",        (nori=[8,12,16], dts=0.75, harmonics=(2,4), ladder=[2.0,3.742,7.0],
                             betas=[2.0,1.6,1.2], d_factors=(1.0,), cross_scale=:ratio)),
    ("adopted+xscale",      (nori=[8,12,16], dts=0.75, harmonics=(2,4,6,8), ladder=[2.0,3.742,7.0],
                             betas=[2.0,1.6,1.2], d_factors=(0.5,1.0,2.0), cross_scale=:both)),
    # the fourth oriented scale at λ = 8 px — see Add_DeepHead.jl HEAD=fine
    ("fine λ=8",            (nori=[8,12,16,20], dts=0.75, harmonics=(2,4),
                             ladder=[2.0,3.742,7.0,14.0], betas=[2.0,1.6,1.2,1.0],
                             d_factors=(1.0,), cross_scale=:none)),
    ("scales 5",            (nori=[8,10,12,14,16], dts=0.75, harmonics=(2,4),
                             ladder=[2.0,2.86,3.742,5.1,7.0], betas=[2.0,1.85,1.6,1.4,1.2],
                             d_factors=(1.0,), cross_scale=:none)),
    ("harmonics +C6C8",     (nori=[8,12,16],  dts=0.75,  harmonics=(2,4,6,8),
                             ladder=[2.0,3.742,7.0], betas=[2.0,1.6,1.2], d_factors=(1.0,), cross_scale=:none)),
    ("orient x2 +C6C8",     (nori=[16,24,32], dts=0.375, harmonics=(2,4,6,8),
                             ladder=[2.0,3.742,7.0], betas=[2.0,1.6,1.2], d_factors=(1.0,), cross_scale=:none)),
    ("offsets x3 crossed",  (nori=[8,12,16],  dts=0.75,  harmonics=(2,4),
                             ladder=[2.0,3.742,7.0], betas=[2.0,1.6,1.2], d_factors=(0.5,1.0,2.0), cross_scale=:none)),
    # The confirmation arm: the two axes that paid, combined. They gained on disjoint rows in the
    # reduced-n sweep — harmonics on curvedness/vangle, offsets on brokenness/arms — so if those
    # gains are real and independent this should pick up both columns.
    ("harmonics+offsets",   (nori=[8,12,16],  dts=0.75,  harmonics=(2,4,6,8),
                             ladder=[2.0,3.742,7.0], betas=[2.0,1.6,1.2], d_factors=(0.5,1.0,2.0), cross_scale=:none)),
]

function score(X, Ytr, Xte, Yte; drop=nothing, linear=true)
    ntr = size(X, 1); nva = clamp(ntr ÷ 6, 40, ntr - 40)
    va = ntr-nva+1:ntr; tr = 1:ntr-nva
    μy, σy = zfit(Ytr[tr, :])
    Zt = zapply(Ytr[tr, :], μy, σy); Zv = zapply(Ytr[va, :], μy, σy)
    μ, σ = zfit(X[tr, :]); A = zapply(X, μ, σ); T = zapply(Xte, μ, σ)
    P = linear ? ridge(A[tr,:], Zt, A[va,:], Zv, T)[1] :
                 mlp(A[tr,:], Zt, A[va,:], Zv, T; epochs=EPOCHS)[1]
    pred = P .* σy' .+ μy'
    [(drop !== nothing && PROPS[j] == String(drop)) ? NaN : r2(pred[:,j], Yte[:,j])
     for j in 1:size(Yte,2)]
end

fmt(x) = isnan(x) ? @sprintf("%10s","—") : @sprintf("%10.3f", x)

function main()
    @printf("Capacity sweep — %d train / %d test, grid 1, %d epochs, %d threads\n", NTR, NTE, EPOCHS, Threads.nthreads())
    @printf("splits: %s\n", join(SPL, ", "))
    for split in SPL
        drop = split == "iid" ? nothing : Symbol(split)
        itr = read_images(joinpath(DATA, "$(split)_train_img.f32"), NTR)
        ite = read_images(joinpath(DATA, "$(split)_test_img.f32"),  NTE)
        Ytr = read_matrix(joinpath(DATA, "$(split)_train_y.f32"), NTR, length(PROPS), 16000)
        Yte = read_matrix(joinpath(DATA, "$(split)_test_y.f32"),  NTE, length(PROPS), 4000)
        println("\n" * "="^116); println("$split split — test R² (MLP readout)"); println("="^116)
        @printf("%-22s%6s", "config", "nfeat")
        for p in PROPS; @printf("%10s", p[1:min(9,end)]); end; println(); println("-"^116)
        for (nm, c) in (isempty(ARMS) ? CFGS : [x for x in CFGS if x[1] in ARMS])
            sp = build_frontend(N; grid=1, ladder=c.ladder, nori=c.nori, betas=c.betas,
                                dts=c.dts, d_factors=c.d_factors, harmonics=c.harmonics,
                                cross_scale=c.cross_scale)
            mkpath(CACHE)
            ck = joinpath(CACHE, "$(replace(nm," "=>"_"))_$(split)_$(NTR)_$(NTE).jls")
            t = @elapsed ((Ftr, Fte) = isfile(ck) ? deserialize(ck) :
                          (a = featurize(itr, sp); b = featurize(ite, sp);
                           serialize(ck, (a, b)); (a, b)))
            v = score(Ftr, Ytr, Fte, Yte; drop=drop, linear=false)
            @printf("%-22s%6d", nm, sp.n); for x in v; print(fmt(x)); end
            @printf("   %5.0fs\n", t); flush(stdout)
        end
    end
end

main()
