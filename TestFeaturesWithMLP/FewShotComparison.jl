# FewShotComparison.jl — the experiment behind section 7.11 of
# README_MLP_FPE_Experiment.md.
#
# How many training examples does each approach need?  Compares the hand-designed
# F3x3+2 no-DC Fourier features against a small CNN trained on raw pixels, at
# k = 10, 20, 50, 100 training images per merged class, 5 seeds each.
#
#   julia --project=. -t 8 TestFeaturesWithMLP/FewShotComparison.jl
#
# ~90 minutes on 8 CPU threads, dominated by the 20 CNN runs.  Writes
# figures/fewshot_curves.png and figures/fewshot_sample_efficiency.png.
#
# Budget is a fixed number of Adam STEPS, not epochs, so that k=100 does not get ten
# times the gradient updates of k=10 — that would confound sample size with
# optimisation budget.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Statistics, Printf, Random, LinearAlgebra, Serialization
using Flux, OneHotArrays, Plots
include(joinpath(@__DIR__, "..", "LoadEMNIST.jl")); using .LoadEMNIST
BLAS.set_num_threads(8)

const IMG   = 112
const STEPS = parse(Int, get(ENV,"FS_STEPS","4000"))
const EVERY = parse(Int, get(ENV,"FS_EVERY","250"))
const KS    = parse.(Int, split(get(ENV,"FS_KS","10,20,50,100"), ","))
const SEEDS = 1:parse(Int, get(ENV,"FS_SEEDS","5"))
const FIGS  = joinpath(@__DIR__, "figures")

# ---------------------------------------------------------------- helpers
"Standardise columns using TRAIN statistics only, then clip. Fit on the k-subset."
function standardise(Xtr, Xte; clip=3f0)
    μ=vec(mean(Xtr,dims=1)); σ=vec(std(Xtr,dims=1)); σ[σ .<= 0] .= 1f0
    f(X)=clamp.((X .- μ') ./ σ', -clip, clip)
    f(Xtr), f(Xte)
end
transposeT(X) = permutedims(X)

# ---------------------------------------------------------------- feature pipeline
# Identical to section 7.10: 3x3 grid of 49 px patches + centred 75 px + whole-field
# 111 px, 8 numbers each (the DC term a0 is dropped), on the 112x112 upsample.
@inline function bilinear(M,y,x)
    H,W=size(M); (y<1||x<1||y>H||x>W)&&return 0f0
    y0,x0=floor(Int,y),floor(Int,x); y1,x1=min(y0+1,H),min(x0+1,W); fy,fx=y-y0,x-x0
    (1-fy)*(1-fx)*M[y0,x0]+fy*(1-fx)*M[y1,x0]+(1-fy)*fx*M[y0,x1]+fy*fx*M[y1,x1]
end
function upsample(img,N=IMG)
    H,W=size(img); out=zeros(Float32,N,N)
    @inbounds for i in 1:N,j in 1:N
        out[i,j]=bilinear(img,Float32(1+(i-1)*(H-1)/(N-1)),Float32(1+(j-1)*(W-1)/(N-1)))
    end; out
end
function patch_at(img,cy,cx,P)
    N=size(img,1); out=zeros(Float32,P,P); y0=round(Int,cy)-(P-1)÷2; x0=round(Int,cx)-(P-1)÷2
    @inbounds for i in 1:P,j in 1:P; out[i,j]=img[clamp(y0+i-1,1,N),clamp(x0+j-1,1,N)]; end; out
end
gauss_window(P)=(σ=Float32(P)/4;c=Float32((P+1)/2);Float32[exp(-((i-c)^2+(j-c)^2)/(2σ^2)) for i in 1:P,j in 1:P])
dft_basis(P,K)=ComplexF32[cis(-2f0π*Float32(k)*Float32(n-1)/Float32(P)) for k in -K:K, n in 1:P]

"""
The 8 per-patch features, WITHOUT the DC term a0:
  ac              total AC energy (how much structure is here)
  Re E2, Im E2    orientation tensor (pi-periodic, so 2*phi)
  |E2|            anisotropy
  |E4|            4-fold term
  e1, e2, e3      radial energy profile
"""
function fcell_noDC(p,w,K,B)
    F=(B*ComplexF32.(w.*p)*transpose(B))./Float32(sum(w))
    pw=0f0; s2=ComplexF32(0); s4=ComplexF32(0); rings=zeros(Float32,3)
    @inbounds for v in -K:K, u in -K:K
        (v==0&&u==0)&&continue
        e=abs2(F[v+K+1,u+K+1]); φ=atan(Float32(v),Float32(u))
        pw+=e; s2+=e*cis(2φ); s4+=e*cis(4φ); rings[clamp(round(Int,hypot(v,u)),1,3)]+=e
    end
    pw<=0 && return zeros(Float64,8)
    Float64[sqrt(pw), real(s2/pw), imag(s2/pw), abs(s2/pw), abs(s4/pw), (rings./pw)...]
end

const P3   = (p=round(Int,1.3*IMG/3); isodd(p) ? p : p+1)      # 49
const PMID = 75
const PALL = 111
const W3,B3     = gauss_window(P3),   dft_basis(P3,3)
const WMID,BMID = gauss_window(PMID), dft_basis(PMID,3)
const WALL,BALL = gauss_window(PALL), dft_basis(PALL,3)

function ffeat(big)
    cs=IMG/3; g=Float64[]
    for i in 1:3, j in 1:3
        append!(g, fcell_noDC(patch_at(big,(i-0.5)*cs,(j-0.5)*cs,P3), W3, 3, B3))
    end
    vcat(g, fcell_noDC(patch_at(big,IMG/2,IMG/2,PMID), WMID,3,BMID),
            fcell_noDC(patch_at(big,IMG/2,IMG/2,PALL), WALL,3,BALL))
end
function extract(imgs28)
    n=size(imgs28,3); F=zeros(Float32,n,88)
    Threads.@threads for i in 1:n
        F[i,:]=ffeat(upsample(@view imgs28[:,:,i]))
    end; F
end

# ---------------------------------------------------------------- data
D=joinpath(homedir(),"Julia","DATABASES","EMNIST"); S=joinpath(D,"emnist_source_files")
tr_i=read_emnist_images(joinpath(D,"emnist-balanced-train-images-idx3-ubyte"))
tr_l=read_emnist_labels(joinpath(D,"emnist-balanced-train-labels-idx1-ubyte"))
te_i=read_emnist_images(joinpath(S,"emnist-balanced-test-images-idx3-ubyte"))
te_l=read_emnist_labels(joinpath(S,"emnist-balanced-test-labels-idx1-ubyte"))
t0=time(); Ftr=extract(tr_i); Fte=extract(te_i)
@printf("feature extraction (train+test, 88 feats): %.0f s\n", time()-t0)

NAMES=[LoadEMNIST.emnist_class_name(i) for i in 1:47]
HOMO=[["0","O"],["1","I","L"],["2","Z"],["5","S"],["9","g","q"]]
canon=Dict(s=>s for s in NAMES); for g in HOMO, s in g; canon[s]=g[1]; end
reps=sort(unique(values(canon))); const NC=length(reps)
mlab(y)=[findfirst(==(canon[NAMES[c]]), reps) for c in y]
ytr_all, yte = mlab(tr_l), mlab(te_l)
@printf("homoglyph-merged: %d classes, chance %.2f %%\n", NC, 100/NC)

Ptr = Float32.(reshape(permutedims(reshape(tr_i,784,:)),:,784))
Pte = Float32.(reshape(permutedims(reshape(te_i,784,:)),:,784))
towhcn(P)=reshape(permutedims(P),28,28,1,size(P,1))
Xte4 = towhcn(Pte)

"Draw k training indices per merged class ONCE. The same images are reused every step."
function subsample(y, k; seed=1)
    rng=MersenneTwister(seed); idx=Int[]
    for c in 1:NC
        pool=findall(==(c), y)
        append!(idx, pool[randperm(rng,length(pool))[1:min(k,length(pool))]])
    end
    sort(idx)
end

# ---------------------------------------------------------------- training
function train_steps(model, getb, ntr, ysub, evalfn; steps=STEPS, batch=32, lr=1f-3,
                     evalevery=EVERY)
    opt=Flux.setup(Flux.Adam(lr),model); Y=onehotbatch(ysub,1:NC)
    curve=Float64[]; b=min(batch,ntr); order=Int[]; pos=1
    for s in 1:steps
        if pos+b-1 > length(order); order=randperm(ntr); pos=1; end
        idx=order[pos:pos+b-1]; pos+=b
        _,gs=Flux.withgradient(m->Flux.logitcrossentropy(m(getb(idx)),Y[:,idx]), model)
        Flux.update!(opt,model,gs[1])
        (s % evalevery == 0) && push!(curve, evalfn(model))
    end
    curve
end

acc4(m)=(p=Int[]; for i in 1:5000:size(Xte4,4)
    j=min(i+4999,size(Xte4,4)); append!(p,onecold(m(view(Xte4,:,:,:,i:j)),1:NC)) end; mean(p.==yte))
mkacc2(B)=m->(p=Int[]; for i in 1:10000:size(B,2)
    j=min(i+9999,size(B,2)); append!(p,onecold(m(view(B,:,i:j)),1:NC)) end; mean(p.==yte))

"F3x3+2 no-DC -> 256 -> NC. Standardisation fit on the SUBSET only (no leakage)."
function arm_features(idx; seed=1, kw...)
    a,b = standardise(Ftr[idx,:], Fte); A=transposeT(a); B=transposeT(b); Random.seed!(seed)
    train_steps(Chain(Dense(88=>256,relu), Dense(256=>NC)),
                i->A[:,i], length(idx), ytr_all[idx], mkacc2(B); kw...)
end
"F3x3+2 no-DC -> 256 -> 128 -> NC."
function arm_features_deep(idx; seed=1, kw...)
    a,b = standardise(Ftr[idx,:], Fte); A=transposeT(a); B=transposeT(b); Random.seed!(seed)
    train_steps(Chain(Dense(88=>256,relu), Dense(256=>128,relu), Dense(128=>NC)),
                i->A[:,i], length(idx), ytr_all[idx], mkacc2(B); kw...)
end
"Small CNN on raw 28x28: 2 conv + pooling, the strongest arm at full data."
function arm_cnn(idx; seed=1, kw...)
    A=towhcn(Ptr[idx,:]); Random.seed!(seed)
    train_steps(Chain(Conv((3,3),1=>32,relu;pad=1),MaxPool((2,2)),
                      Conv((3,3),32=>64,relu;pad=1),MaxPool((2,2)),Flux.flatten,
                      Dense(7*7*64=>256,relu),Dense(256=>NC)),
                i->view(A,:,:,:,i), length(idx), ytr_all[idx], acc4; kw...)
end

const RUN = ["F 88->256->40"      => arm_features,
             "F 88->256->128->40" => arm_features_deep,
             "small CNN"          => arm_cnn]

println("\n", "="^80)
@printf("FEW-SHOT SWEEP — %d merged classes, full test set (%d), %d Adam steps, batch 32\n",
        NC, length(yte), STEPS)
println("paired: all arms share the same drawn subset for each (k, seed); no augmentation")
println("="^80, "\n")

results = Dict{Tuple{String,Int,Int},Vector{Float64}}()
for k in KS, seed in SEEDS
    idx = subsample(ytr_all, k; seed=seed); @assert length(idx) == k*NC
    for (nm,f) in RUN
        t=time(); c = f(idx; seed=seed); results[(nm,k,seed)] = c
        @printf("  k=%-4d seed %d  %-20s final %.4f  best %.4f   (%5d imgs, %4.0f s)\n",
                k, seed, nm, c[end], maximum(c), length(idx), time()-t); flush(stdout)
    end
end

println("\n", "="^80); println("SUMMARY — test accuracy (%), mean +/- sd over seeds"); println("="^80)
@printf("\n%-22s %5s %8s %20s %20s\n", "arm", "k", "imgs", "final", "best")
for k in KS, (nm,_) in RUN
    cs=[results[(nm,k,s)] for s in SEEDS]
    fi=[100*c[end] for c in cs]; be=[100*maximum(c) for c in cs]
    @printf("%-22s %5d %8d   %6.2f +/- %4.2f     %6.2f +/- %4.2f\n",
            nm, k, k*NC, mean(fi), std(fi), mean(be), std(be))
end
println("\nPAIRED per-seed gap, features(256) minus small CNN, best-eval (%):")
for k in KS
    d=[100*(maximum(results[("F 88->256->40",k,s)])-maximum(results[("small CNN",k,s)])) for s in SEEDS]
    @printf("  k=%-4d  %s   mean %+6.2f +/- %4.2f\n", k,
            join([@sprintf("%+6.2f",v) for v in d]," "), mean(d), std(d))
end

# ---------------------------------------------------------------- figures
col=Dict("F 88->256->40"=>:steelblue,"F 88->256->128->40"=>:seagreen,"small CNN"=>:firebrick)
LBL=Dict("F 88->256->40"=>"F3×3+2 → 256 → 40  (33 k params)",
         "F 88->256->128->40"=>"F3×3+2 → 256 → 128 → 40  (60 k)",
         "small CNN"=>"small CNN, 2 conv + pool  (834 k)")
mstat(nm,k)=(M=reduce(hcat,[results[(nm,k,s)] for s in SEEDS]); (vec(mean(M,dims=2)),vec(std(M,dims=2))))

panels=Any[]
for k in KS
    pnl=plot(xlabel="Adam step", ylabel="test accuracy (%)",
             legend=(k==KS[1] ? :bottomright : false), ylims=(55,90), yticks=55:5:90,
             title="k = $k   ($(k*NC) training images)", titlefontsize=9,
             grid=true, gridalpha=0.25, legendfontsize=6, tickfontsize=7, guidefontsize=8)
    for (nm,_) in RUN
        μ,σ=mstat(nm,k)
        plot!(pnl, EVERY:EVERY:STEPS, 100 .*μ; ribbon=100 .*σ, fillalpha=0.18, lw=2,
              c=col[nm], label=LBL[nm])
    end
    push!(panels,pnl)
end
savefig(plot(panels...; layout=(2,2), size=(1000,620),
        plot_title="Few-shot learning curves — mean ± 1 sd over $(length(SEEDS)) seeds ($(NC) merged classes)",
        plot_titlefontsize=11, left_margin=6Plots.mm, bottom_margin=5Plots.mm),
        joinpath(FIGS,"fewshot_curves.png"))

fig2=plot(xlabel="training images per merged class (log scale)", ylabel="test accuracy (%)",
          xscale=:log10, xticks=(KS,string.(KS)), legend=:bottomright, grid=true,
          gridalpha=0.25, size=(900,510), ylims=(58,95), yticks=60:5:95,
          left_margin=7Plots.mm, bottom_margin=6Plots.mm, legendfontsize=8,
          title="Sample efficiency: designed features vs. learned convolution ($(NC) merged classes)",
          titlefontsize=10)
for (nm,_) in RUN
    ys=[mean([100*maximum(results[(nm,k,s)]) for s in SEEDS]) for k in KS]
    es=[std( [100*maximum(results[(nm,k,s)]) for s in SEEDS]) for k in KS]
    plot!(fig2, KS, ys; yerror=es, lw=2, marker=:circle, ms=5, c=col[nm], label=LBL[nm])
end
hline!(fig2,[92.30]; ls=:dot, lc=:steelblue, lw=1.5, label="features, all 112,800 (92.30 %)")
hline!(fig2,[92.70]; ls=:dot, lc=:firebrick, lw=1.5, label="small CNN, all 112,800 (92.70 %)")
savefig(fig2, joinpath(FIGS,"fewshot_sample_efficiency.png"))

println("\nhow much data the CNN needs to match the feature MLP (log-linear interpolation):")
fa(nm,k)=mean([100*maximum(results[(nm,k,s)]) for s in SEEDS])
for k in KS[1:end-1]
    target=fa("F 88->256->40",k)
    j=findfirst(t->fa("small CNN",KS[t])>=target, 1:length(KS))
    if j!==nothing && j>1
        k0,k1=KS[j-1],KS[j]; y0,y1=fa("small CNN",k0),fa("small CNN",k1)
        lk=log10(k0)+(target-y0)/(y1-y0)*(log10(k1)-log10(k0))
        @printf("  features at k=%-3d (%.2f %%)  ->  CNN needs k≈%.1f   (%.2fx the data)\n",
                k, target, 10^lk, 10^lk/k)
    end
end
println("\nwrote figures/fewshot_curves.png and figures/fewshot_sample_efficiency.png")
