# ── PLAIN SCRIPT, not a Pluto notebook ──────────────────────────────────────
# Run with `julia --project=. -t 8 <this file>`. Opening it in Pluto rewrites it.
#
# Phase 6 — does augmentation help the raw pixels but NOT the designed features?
#
# This is the control that section 7.11 has been missing since it was written. Its
# headline — designed features beat a small CNN by +7.7 points at k=10 — is measured
# against a CNN that had to discover its invariances from raw pixels with nothing to help
# it. Augmentation is precisely how you hand those invariances to a CNN. So what is
# currently recorded is "designed invariances vs invariances learned from raw data alone",
# which is a weaker claim than the one the project wants to make.
#
# The interesting form is a DIFFERENTIAL, not a single number:
#
#     if the features really encode these invariances, augmentation should
#         help the CNN a lot        (it supplies what the model lacks)
#         help the features little  (they already have it)
#
# If instead augmentation helps both equally, the features do not contain what we think
# they do, and the polarity/rotation/translation story needs revisiting. Either outcome is
# informative, which is what makes it worth running.
#
# Two design points that matter:
#
#   * Augmenting the FEATURE pipeline means augmenting the IMAGES and re-extracting — not
#     perturbing feature vectors, which would test something else entirely. Expensive at
#     full data, cheap in the few-shot regime, which is also where augmentation matters
#     most.
#   * Both arms see the SAME augmented images. The augmented set is generated once per
#     (k, seed); the feature arm extracts from it, the CNN arm takes its raw pixels.
#     Otherwise the two are being given different data as well as different treatment.
#
# The augmentation mirrors the features' own invariances — small rotations (Zernike's
# |A_nm|), translations (the pooling), scale (fit_disc) — so it is like-for-like rather
# than generic.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Statistics, Printf, Random, LinearAlgebra, Serialization, FFTW
using Flux, OneHotArrays, Plots
include(joinpath(@__DIR__, "GaborStack.module.jl"))
include(joinpath(@__DIR__, "AndLayer.module.jl"))
include(joinpath(@__DIR__, "Pooling.module.jl"))
include(joinpath(@__DIR__, "..", "LoadEMNIST.module.jl"))
using .GaborStack, .AndLayer, .Pooling, .LoadEMNIST

const IMG   = 112
const CACHE = get(ENV, "P6_CACHE", joinpath(tempdir(), "p5a_features.jls"))
const KS    = parse.(Int, split(get(ENV, "P6_KS", "5,10,20,50"), ","))
const SEEDS = 1:parse(Int, get(ENV, "P6_SEEDS", "5"))
const NAUG  = parse(Int, get(ENV, "P6_NAUG", "10"))
const STEPS = parse(Int, get(ENV, "P6_STEPS", "4000"))
BLAS.set_num_threads(2); FFTW.set_num_threads(1)

# ---------------------------------------------------------------- augmentation
@inline function bil(M, y, x)
    H, W = size(M); (y < 1 || x < 1 || y > H || x > W) && return 0f0
    y0, x0 = floor(Int,y), floor(Int,x); y1, x1 = min(y0+1,H), min(x0+1,W)
    fy, fx = y-y0, x-x0
    (1-fy)*(1-fx)*M[y0,x0] + fy*(1-fx)*M[y1,x0] + (1-fy)*fx*M[y0,x1] + fy*fx*M[y1,x1]
end

"""
Random similarity transform: rotation ±10°, isotropic scale ±10 %, translation ±2 px, on
the native 28×28 grid. Inverse-mapped and bilinearly sampled, so no holes.
"""
function augment(img::AbstractMatrix{Float32}, rng)
    H, W = size(img); cy, cx = (H+1)/2, (W+1)/2
    θ = (rand(rng) - 0.5) * 2 * deg2rad(10)
    s = 1 + (rand(rng) - 0.5) * 0.2
    ty = (rand(rng) - 0.5) * 4; tx = (rand(rng) - 0.5) * 4
    c, sn = cos(θ)/s, sin(θ)/s
    out = zeros(Float32, H, W)
    @inbounds for y in 1:H, x in 1:W
        dy = y - cy - ty; dx = x - cx - tx
        out[y,x] = bil(img, cy + ( c*dy + sn*dx), cx + (-sn*dy + c*dx))
    end
    out
end

function upsample(img, N=IMG)
    H, W = size(img); out = zeros(Float32, N, N)
    @inbounds for i in 1:N, j in 1:N
        out[i,j] = bil(img, Float32(1+(i-1)*(H-1)/(N-1)), Float32(1+(j-1)*(W-1)/(N-1)))
    end
    out
end

# ---------------------------------------------------------------- front end
const LADDER = [2.0, 3.742, 7.0]
const BETAS  = [2.0, 1.6, 1.2]
const NORI   = [8, 12, 16]
const HF, WF, _ = field_for((IMG,IMG), LADDER; n_orient=NORI, beta=BETAS)
const BANK = make_bank((HF,WF), LADDER; imwidth=IMG, n_orient=NORI, beta=BETAS)
const WTS  = grid_weights(IMG, IMG, 3)
const SPEC = PoolSpec(grid=3, blocks=(:orient,:lowpass,:A1,:A2))

function feat1(img28)
    Es = energy_stack(upsample(img28), BANK)
    A, al = and_maps(Es, BANK.meta; forms=(:A1,:A2))
    assemble(Es, BANK.meta, A, al, SPEC; Wts=WTS)[1]
end
function featmany(imgs)                 # imgs :: Vector of 28×28
    n = length(imgs); F = zeros(Float32, n, 198)
    Threads.@threads for i in 1:n; F[i,:] = feat1(imgs[i]); end
    F
end

# ---------------------------------------------------------------- data
isfile(CACHE) || error("need the Phase 5a cache at $CACHE for the test features")
C = deserialize(CACHE); NEWte = C.NEWte
D = joinpath(homedir(),"Julia","DATABASES","EMNIST"); S = joinpath(D,"emnist_source_files")
tr_i = read_emnist_images(joinpath(D,"emnist-balanced-train-images-idx3-ubyte"))
tr_l = read_emnist_labels(joinpath(D,"emnist-balanced-train-labels-idx1-ubyte"))
te_i = read_emnist_images(joinpath(S,"emnist-balanced-test-images-idx3-ubyte"))
te_l = read_emnist_labels(joinpath(S,"emnist-balanced-test-labels-idx1-ubyte"))
NAMES = [LoadEMNIST.emnist_class_name(i) for i in 1:47]
HOMO  = [["0","O"],["1","I","L"],["2","Z"],["5","S"],["9","g","q"]]
canon = Dict(s=>s for s in NAMES); for g in HOMO, s in g; canon[s]=g[1]; end
reps  = sort(unique(values(canon))); const NC = length(reps)
mlab(y) = [findfirst(==(canon[NAMES[c]]), reps) for c in y]
ytr, yte = mlab(tr_l), mlab(te_l)
Pte4 = reshape(Float32.(reshape(permutedims(reshape(te_i,784,:)),:,784))', 28,28,1,:)
@printf("%d merged classes, chance %.2f %% · %d augmented copies per image\n\n",
        NC, 100/NC, NAUG)

function subsample(y, k; seed=1)
    rng = MersenneTwister(seed); idx = Int[]
    for c in 1:NC
        pool = findall(==(c), y)
        append!(idx, pool[randperm(rng,length(pool))[1:min(k,length(pool))]])
    end
    sort(idx)
end
function standardise(a, b; clip=3f0)
    μ = vec(mean(a,dims=1)); σ = vec(std(a,dims=1)); σ[σ .<= 0] .= 1f0
    clamp.((a .- μ')./σ', -clip, clip), clamp.((b .- μ')./σ', -clip, clip)
end

"Fixed step budget for every arm, so augmentation buys variety and not extra updates."
function train(model, getb, n, ysub, evalfn; steps=STEPS, batch=32, lr=1f-3)
    opt = Flux.setup(Flux.Adam(lr), model); Y = onehotbatch(ysub,1:NC)
    b = min(batch,n); order = Int[]; pos = 1
    for _ in 1:steps
        if pos+b-1 > length(order); order = randperm(n); pos = 1; end
        idx = order[pos:pos+b-1]; pos += b
        _,gs = Flux.withgradient(m->Flux.logitcrossentropy(m(getb(idx)),Y[:,idx]), model)
        Flux.update!(opt,model,gs[1])
    end
    evalfn(model)
end
acc2(B) = m -> (p=Int[]; for i in 1:10000:size(B,2)
    j=min(i+9999,size(B,2)); append!(p,onecold(m(view(B,:,i:j)),1:NC)) end; mean(p.==yte))
acc4(X) = m -> (p=Int[]; for i in 1:5000:size(X,4)
    j=min(i+4999,size(X,4)); append!(p,onecold(m(view(X,:,:,:,i:j)),1:NC)) end; mean(p.==yte))

function fit_feat(Ftr, ys; seed=1)
    a, b = standardise(Ftr, NEWte)
    A = permutedims(a); B = permutedims(b)
    Random.seed!(seed)
    train(Chain(Dense(198=>256,relu), Dense(256=>NC)), i->A[:,i], size(A,2), ys, acc2(B))
end
function fit_cnn(imgs, ys; seed=1)
    X = zeros(Float32,28,28,1,length(imgs))
    for (i,im) in enumerate(imgs); X[:,:,1,i] = im; end
    Random.seed!(seed)
    m = Chain(Conv((3,3),1=>32,relu;pad=1),MaxPool((2,2)),
              Conv((3,3),32=>64,relu;pad=1),MaxPool((2,2)),Flux.flatten,
              Dense(7*7*64=>256,relu),Dense(256=>NC))
    train(m, i->view(X,:,:,:,i), length(imgs), ys, acc4(Pte4))
end

println("="^84)
println("PHASE 6 — augmentation should help the pixels and not the features")
println("="^84)
res = Dict{Tuple{String,Int,Int},Float64}()
for k in KS, s in SEEDS
    t0 = time()
    idx = subsample(ytr, k; seed=s); ys0 = ytr[idx]
    base = [Float32.(tr_i[:,:,i]) for i in idx]
    rng = MersenneTwister(1000s)
    aug  = vcat(base, [augment(im, rng) for _ in 1:NAUG for im in base])
    ysA  = vcat(ys0, repeat(ys0, NAUG))
    Fb = featmany(base); Fa = featmany(aug)
    res[("feat",k,s)]     = fit_feat(Fb, ys0; seed=s)
    res[("feat+aug",k,s)] = fit_feat(Fa, ysA; seed=s)
    res[("cnn",k,s)]      = fit_cnn(base, ys0; seed=s)
    res[("cnn+aug",k,s)]  = fit_cnn(aug,  ysA; seed=s)
    @printf("  k=%-3d seed %d  feat %.4f→%.4f   cnn %.4f→%.4f   (%.0f s)\n", k, s,
            res[("feat",k,s)], res[("feat+aug",k,s)],
            res[("cnn",k,s)],  res[("cnn+aug",k,s)], time()-t0)
    flush(stdout)
end

println("\n" * "="^84)
println("Test accuracy (%), mean ± sd over $(length(SEEDS)) seeds. Δ is paired per seed.")
println("="^84)
@printf("\n%-5s %7s %17s %9s %17s %9s\n",
        "k", "images", "features", "Δ aug", "small CNN", "Δ aug")
for k in KS
    df = [100*(res[("feat+aug",k,s)] - res[("feat",k,s)]) for s in SEEDS]
    dc = [100*(res[("cnn+aug",k,s)]  - res[("cnn",k,s)])  for s in SEEDS]
    @printf("%-5d %7d %9.2f ±%4.2f %+7.2f %9.2f ±%4.2f %+7.2f\n", k, k*NC,
            100mean(res[("feat",k,s)] for s in SEEDS),
            100std([res[("feat",k,s)] for s in SEEDS]), mean(df),
            100mean(res[("cnn",k,s)] for s in SEEDS),
            100std([res[("cnn",k,s)] for s in SEEDS]), mean(dc))
end
println("""
The prediction is the DIFFERENCE of the two Δ columns. If the designed features already
carry these invariances, augmentation should move the CNN much more than it moves them.
If both move equally, the features do not contain what the design claims.""")

serialize(joinpath(dirname(CACHE), "p6_results.jls"), res)
p = plot(xlabel="training images per merged class (log scale)",
         ylabel="Δ accuracy from augmentation (points)", xscale=:log10,
         xticks=(KS,string.(KS)), grid=true, gridalpha=0.25, size=(880,430),
         legend=:topright, titlefontsize=10, left_margin=7Plots.mm, bottom_margin=5Plots.mm,
         title="What augmentation buys each representation ($(NC) merged classes)")
for (nm, base, lbl, c) in [("feat+aug","feat","designed features",:steelblue),
                           ("cnn+aug","cnn","small CNN on raw pixels",:firebrick)]
    ys = [mean([100*(res[(nm,k,s)]-res[(base,k,s)]) for s in SEEDS]) for k in KS]
    es = [std( [100*(res[(nm,k,s)]-res[(base,k,s)]) for s in SEEDS]) for k in KS]
    plot!(p, KS, ys; yerror=es, lw=2.5, marker=:circle, ms=5, c=c, label=lbl)
end
hline!(p, [0]; lc=:black, ls=:dash, lw=1, label="no effect")
savefig(p, joinpath(@__DIR__, "figures", "phase6_augmentation.png"))
println("\nwrote figures/phase6_augmentation.png")
