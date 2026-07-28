# ── PLAIN SCRIPT, not a Pluto notebook ──────────────────────────────────────
# Run with `julia --project=. -t 8 <this file>`. Opening it in Pluto rewrites it.
#
# Phase 5a — put the RationalGaborFeatures front end on EMNIST and check it against a
# number we already have.
#
# The discipline here is that the REFERENCE ARM comes first. Section 7.10 of
# TestFeaturesWithMLP/README_MLP_FPE_Experiment.md records 92.30 % for the old F3x3+2
# no-DC features on 40 homoglyph-merged classes. That arm is re-extracted and re-trained
# here, in this harness, with this training loop. If it does not land on 92.30 % then the
# harness is wrong and nothing else in the table means anything — a new pipeline that
# cannot reproduce a known result has a bug, and discovering that after running an
# ablation would waste the ablation.
#
# Everything matches section 7.10: official split (112,800 / 18,800), 40 merged classes,
# Dense(n=>256,relu) -> Dense(256=>40), Adam 1e-3, batch 128, 15 epochs, seed 1,
# standardisation fit on train only and clipped at 3 sd.
#
# ~8 minutes: 6 for the dense extraction, seconds for everything else.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Statistics, Printf, Random, LinearAlgebra, Serialization, FFTW
using Flux, OneHotArrays, Plots
include(joinpath(@__DIR__, "GaborStack.module.jl"))
include(joinpath(@__DIR__, "AndLayer.module.jl"))
include(joinpath(@__DIR__, "Pooling.module.jl"))
include(joinpath(@__DIR__, "..", "LoadEMNIST.module.jl"))
using .GaborStack, .AndLayer, .Pooling, .LoadEMNIST

const IMG   = 112
const CACHE = get(ENV, "P5A_CACHE", joinpath(tempdir(), "p5a_features.jls"))
const NIMG  = parse(Int, get(ENV, "P5A_NIMG", "0"))      # 0 = all; small values for smoke
BLAS.set_num_threads(2)
FFTW.set_num_threads(1)          # we thread over images, so keep FFTW single-threaded

# ---------------------------------------------------------------- upsample
@inline function bilinear(M, y, x)
    H, W = size(M); (y < 1 || x < 1 || y > H || x > W) && return 0f0
    y0, x0 = floor(Int, y), floor(Int, x); y1, x1 = min(y0+1, H), min(x0+1, W)
    fy, fx = y - y0, x - x0
    (1-fy)*(1-fx)*M[y0,x0] + fy*(1-fx)*M[y1,x0] + (1-fy)*fx*M[y0,x1] + fy*fx*M[y1,x1]
end
function upsample(img, N=IMG)
    H, W = size(img); out = zeros(Float32, N, N)
    @inbounds for i in 1:N, j in 1:N
        out[i,j] = bilinear(img, Float32(1+(i-1)*(H-1)/(N-1)), Float32(1+(j-1)*(W-1)/(N-1)))
    end
    out
end

# ---------------------------------------------------------------- the OLD features
# F3x3+2 no-DC, 88 numbers — reproduced verbatim so the reference arm is the same
# computation, not a re-derivation of it.
function patch_at(img, cy, cx, P)
    N = size(img,1); out = zeros(Float32, P, P)
    y0 = round(Int,cy)-(P-1)÷2; x0 = round(Int,cx)-(P-1)÷2
    @inbounds for i in 1:P, j in 1:P
        out[i,j] = img[clamp(y0+i-1,1,N), clamp(x0+j-1,1,N)]
    end
    out
end
gwin(P) = (σ=Float32(P)/4; c=Float32((P+1)/2);
           Float32[exp(-((i-c)^2+(j-c)^2)/(2σ^2)) for i in 1:P, j in 1:P])
dbasis(P,K) = ComplexF32[cis(-2f0π*Float32(k)*Float32(n-1)/Float32(P)) for k in -K:K, n in 1:P]
function fcell_noDC(p, w, K, B)
    F = (B*ComplexF32.(w.*p)*transpose(B)) ./ Float32(sum(w))
    pw=0f0; s2=ComplexF32(0); s4=ComplexF32(0); rings=zeros(Float32,3)
    @inbounds for v in -K:K, u in -K:K
        (v==0 && u==0) && continue
        e = abs2(F[v+K+1,u+K+1]); φ = atan(Float32(v),Float32(u))
        pw += e; s2 += e*cis(2φ); s4 += e*cis(4φ)
        rings[clamp(round(Int,hypot(v,u)),1,3)] += e
    end
    pw <= 0 && return zeros(Float64,8)
    Float64[sqrt(pw), real(s2/pw), imag(s2/pw), abs(s2/pw), abs(s4/pw), (rings./pw)...]
end
const P3 = (p=round(Int,1.3*IMG/3); isodd(p) ? p : p+1)     # 49
const W3,B3 = gwin(P3), dbasis(P3,3)
const WM,BM = gwin(75), dbasis(75,3)
const WA,BA = gwin(111), dbasis(111,3)
function old_features(big)
    cs = IMG/3; g = Float64[]
    for i in 1:3, j in 1:3
        append!(g, fcell_noDC(patch_at(big,(i-0.5)*cs,(j-0.5)*cs,P3), W3, 3, B3))
    end
    vcat(g, fcell_noDC(patch_at(big,IMG/2,IMG/2,75), WM,3,BM),
            fcell_noDC(patch_at(big,IMG/2,IMG/2,111), WA,3,BA))
end

# ---------------------------------------------------------------- the NEW front end
const LADDER = [2.0, 3.742, 7.0]
const BETAS  = [2.0, 1.6, 1.2]
const NORI   = [8, 12, 16]
const HF, WF, _ = field_for((IMG,IMG), LADDER; n_orient=NORI, beta=BETAS)
const BANK = make_bank((HF,WF), LADDER; imwidth=IMG, n_orient=NORI, beta=BETAS)
const WTS  = grid_weights(IMG, IMG, 3)
const SPEC = PoolSpec(grid=3, blocks=(:orient,:lowpass,:A1,:A2))

function new_features(big)
    Es = energy_stack(big, BANK)
    A, al = and_maps(Es, BANK.meta; forms=(:A1,:A2))
    assemble(Es, BANK.meta, A, al, SPEC; Wts=WTS)
end

# ---------------------------------------------------------------- extraction
function extract(imgs)
    n = size(imgs,3)
    f1, lab = new_features(upsample(@view imgs[:,:,1]))
    NEW = zeros(Float32, n, length(f1)); OLD = zeros(Float32, n, 88)
    Threads.@threads for i in 1:n
        big = upsample(@view imgs[:,:,i])
        NEW[i,:] = new_features(big)[1]
        OLD[i,:] = old_features(big)
    end
    NEW, OLD, lab
end

D = joinpath(homedir(),"Julia","DATABASES","EMNIST")
S = joinpath(D,"emnist_source_files")
tr_i = read_emnist_images(joinpath(D,"emnist-balanced-train-images-idx3-ubyte"))
tr_l = read_emnist_labels(joinpath(D,"emnist-balanced-train-labels-idx1-ubyte"))
te_i = read_emnist_images(joinpath(S,"emnist-balanced-test-images-idx3-ubyte"))
te_l = read_emnist_labels(joinpath(S,"emnist-balanced-test-labels-idx1-ubyte"))
if NIMG > 0
    tr_i = tr_i[:,:,1:NIMG]; tr_l = tr_l[1:NIMG]
    te_i = te_i[:,:,1:min(NIMG,size(te_i,3))]; te_l = te_l[1:min(NIMG,length(te_l))]
end

if isfile(CACHE)
    C = deserialize(CACHE)
    NEWtr,OLDtr,NEWte,OLDte,LAB = C.NEWtr,C.OLDtr,C.NEWte,C.OLDte,C.LAB
    @printf("loaded cached features from %s\n", CACHE)
else
    t0 = time(); NEWtr, OLDtr, LAB = extract(tr_i); NEWte, OLDte, _ = extract(te_i)
    @printf("extraction: %.1f min on %d threads  (%.1f ms/image)\n",
            (time()-t0)/60, Threads.nthreads(),
            1000*(time()-t0)/(size(tr_i,3)+size(te_i,3)))
    serialize(CACHE, (; NEWtr, OLDtr, NEWte, OLDte, LAB))
end
@printf("new features: %d   old features: %d\n", size(NEWtr,2), size(OLDtr,2))

# ---------------------------------------------------------------- merged labels
NAMES = [LoadEMNIST.emnist_class_name(i) for i in 1:47]
HOMO  = [["0","O"],["1","I","L"],["2","Z"],["5","S"],["9","g","q"]]
canon = Dict(s=>s for s in NAMES); for g in HOMO, s in g; canon[s]=g[1]; end
reps  = sort(unique(values(canon))); const NC = length(reps)
mlab(y) = [findfirst(==(canon[NAMES[c]]), reps) for c in y]
ytr, yte = mlab(tr_l), mlab(te_l)
@printf("homoglyph-merged: %d classes, chance %.2f %%\n\n", NC, 100/NC)

# ---------------------------------------------------------------- the classifier
# Identical to section 7.10: one hidden layer of 256, Adam 1e-3, batch 128, 15 epochs.
function standardise(Xtr, Xte; clip=3f0)
    μ = vec(mean(Xtr,dims=1)); σ = vec(std(Xtr,dims=1)); σ[σ .<= 0] .= 1f0
    f(X) = clamp.((X .- μ') ./ σ', -clip, clip)
    f(Xtr), f(Xte)
end
function fit(Xtr, Xte; hidden=256, epochs=15, batch=128, lr=1f-3, seed=1)
    a,b = standardise(Xtr,Xte); A = permutedims(a); B = permutedims(b)
    Random.seed!(seed)
    m = Chain(Dense(size(A,1)=>hidden,relu), Dense(hidden=>NC))
    opt = Flux.setup(Flux.Adam(lr), m); Y = onehotbatch(ytr,1:NC); n = size(A,2)
    curve = Float64[]
    for _ in 1:epochs
        p = randperm(n)
        for i in 1:batch:n
            idx = p[i:min(i+batch-1,n)]
            _,gs = Flux.withgradient(mm->Flux.logitcrossentropy(mm(A[:,idx]),Y[:,idx]), m)
            Flux.update!(opt,m,gs[1])
        end
        pr = Int[]
        for i in 1:10000:size(B,2)
            j = min(i+9999,size(B,2)); append!(pr, onecold(m(view(B,:,i:j)),1:NC))
        end
        push!(curve, mean(pr .== yte))
    end
    curve
end

cols(pfx) = findall(x -> startswith(x, pfx), LAB)
orient_lp = vcat(cols("orient"), cols("lowpass"))
a1c, a2c  = cols("A1"), cols("A2")

arms = [
  ("REFERENCE  old F3x3+2 no-DC",      OLDtr,                       OLDte),
  ("new  orient+lowpass",              NEWtr[:,orient_lp],          NEWte[:,orient_lp]),
  ("new  orient+lowpass+A1",           NEWtr[:,vcat(orient_lp,a1c)], NEWte[:,vcat(orient_lp,a1c)]),
  ("new  orient+lowpass+A1+A2 (all)",  NEWtr,                       NEWte),
]

println("="^78)
println("PHASE 5a — does the new front end reach the number we already have?")
println("="^78)
@printf("\n%-34s %6s %9s %9s %7s\n", "arm", "n", "final", "best", "time")
curves = Dict{String,Vector{Float64}}()
for (nm,a,b) in arms
    t = time(); c = fit(a,b); curves[nm] = c
    @printf("%-34s %6d %8.2f %% %8.2f %% %6.0f s\n", nm, size(a,2), 100c[end], 100maximum(c), time()-t)
    flush(stdout)
end

ref = 100*curves["REFERENCE  old F3x3+2 no-DC"][end]
if NIMG > 0
    @printf("\nsmoke run on %d images — the 92.30 %% reference does not apply here.\n", NIMG)
else
    @printf("\nreference arm: %.2f %% vs 92.30 %% recorded in section 7.10 — Δ = %+.2f\n",
            ref, ref - 92.30)
    println(abs(ref - 92.30) < 0.6 ?
            "  harness reproduces the known result; the other rows can be read." :
            "  HARNESS MISMATCH — do not trust the other rows until this is explained.")
end

serialize(joinpath(dirname(CACHE), "p5a_curves.jls"), curves)
p = plot(xlabel="epoch", ylabel="test accuracy (%)", legend=:bottomright, xlims=(1,15),
         grid=true, gridalpha=0.25, size=(900,430), titlefontsize=10,
         left_margin=8Plots.mm, bottom_margin=4Plots.mm,
         title="Phase 5a — RationalGaborFeatures on EMNIST ($(NC) merged classes)")
for (nm,_,_) in arms
    c = curves[nm]
    plot!(p, 1:length(c), 100 .*c; lw=2, marker=:circle, ms=3, label=nm)
end
hline!(p, [92.30]; ls=:dot, lc=:black, lw=1.5, label="section 7.10 reference (92.30 %)")
savefig(p, joinpath(@__DIR__, "figures", "phase5a_curves.png"))
println("\nwrote figures/phase5a_curves.png")
