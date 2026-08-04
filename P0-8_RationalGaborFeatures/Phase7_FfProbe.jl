# ── PLAIN SCRIPT, not a Pluto notebook ──────────────────────────────────────
# Run with `julia --project=. -t 8 <this file>`.
#
# Phase 7 — a binary F-versus-f probe, and R²(A ← orient).
#
# Two open questions from RESULTS.md, both aimed at the same doubt: the AND layer adds
# nothing in aggregate, but aggregate accuracy over 40 classes cannot tell us WHY.
#
# (1) THE F/f PROBE. F/f is 17.3 % of all remaining error but only 1.3 % of test items, so
#     perfectly solving it caps any 40-class gain at +1.34 points — a partial fix on a
#     1.3 % subset is unmeasurable against a 0.19 % standard error. Training a binary
#     classifier on that pair alone removes the dilution and asks the question directly:
#     for the distinction these operators were built for, is A better than orient?
#
#     Ray harmonics are included because A₁ provably CANNOT do this job: it is built on the
#     π-periodic orientation profile, and F (3-ray T) versus f (4-ray X) is a 2π property.
#     c₀ from the ray transform is the operator that can.
#
# (2) R²(A ← orient). How much of the A block is linearly recoverable from orient? High R²
#     means the two are near-substitutable on this data; low R² with no accuracy gain means
#     A carries independent structure the task ignores. Aggregate accuracy cannot separate
#     those, and "redundant" versus "correlated" is exactly the distinction at stake.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Statistics, Printf, Random, LinearAlgebra, Serialization, FFTW
using Flux, OneHotArrays
include(joinpath(@__DIR__, "GaborStack.module.jl"))
include(joinpath(@__DIR__, "AndLayer.module.jl"))
include(joinpath(@__DIR__, "RayHarmonics.module.jl"))
include(joinpath(@__DIR__, "Pooling.module.jl"))
include(joinpath(@__DIR__, "..", "LoadEMNIST.module.jl"))
using .GaborStack, .AndLayer, .RayHarmonics, .Pooling, .LoadEMNIST

const IMG = 112
const P5A = get(ENV, "P7_CACHE", joinpath(tempdir(), "p5a_features.jls"))
BLAS.set_num_threads(4); FFTW.set_num_threads(1)

@inline function bil(M,y,x)
    H,W=size(M); (y<1||x<1||y>H||x>W) && return 0f0
    y0,x0=floor(Int,y),floor(Int,x); y1,x1=min(y0+1,H),min(x0+1,W); fy,fx=y-y0,x-x0
    (1-fy)*(1-fx)*M[y0,x0]+fy*(1-fx)*M[y1,x0]+(1-fy)*fx*M[y0,x1]+fy*fx*M[y1,x1]
end
function upsample(img,N=IMG)
    H,W=size(img); o=zeros(Float32,N,N)
    @inbounds for i in 1:N,j in 1:N
        o[i,j]=bil(img,Float32(1+(i-1)*(H-1)/(N-1)),Float32(1+(j-1)*(W-1)/(N-1)))
    end; o
end

const LADDER=[2.0,3.742,7.0]; const BETAS=[2.0,1.6,1.2]; const NORI=[8,12,16]
const HF,WF,_ = field_for((IMG,IMG),LADDER;n_orient=NORI,beta=BETAS)
const BANK = make_bank((HF,WF),LADDER;imwidth=IMG,n_orient=NORI,beta=BETAS)
const WTS  = grid_weights(IMG,IMG,3)

"orient+lowpass (144), A₁+A₂ (54) and ray harmonics (81) for one image."
function allblocks(img28)
    Es = energy_stack(upsample(img28), BANK)
    A, al = and_maps(Es, BANK.meta; forms=(:A1,:A2))
    Rm, rl = ray_maps(Es, BANK.meta)
    f1, l1 = assemble(Es, BANK.meta, A, al,
                      PoolSpec(grid=3, blocks=(:orient,:lowpass,:A1,:A2)); Wts=WTS)
    # ray_maps returns c₀, |c₁|, |c₂| unnormalised; the ratios are formed after pooling, with
    # a relative floor. Dividing per pixel and averaging weighted every low-energy location
    # equally with a strong contour — see RayHarmonics and P9_P12_SimpleStrokeTests/RESULTS.md.
    PR = pool_maps(Rm, WTS); fr = Float32[]; lr = String[]
    for ρ in unique(l.rho0 for l in rl)
        k0 = findfirst(l -> l.rho0 == ρ && l.form === :R0, rl)
        k1 = findfirst(l -> l.rho0 == ρ && l.form === :R1, rl)
        k2 = findfirst(l -> l.rho0 == ρ && l.form === :R2, rl)
        fl = 1f-3 * max(mean(@view PR[:, k0]), 1f-12)
        for c in 1:9
            push!(fr, PR[c,k0]);               push!(lr, "R0.ρ$(round(ρ,digits=2)).cell$(c)")
            push!(fr, PR[c,k1]/(PR[c,k0]+fl)); push!(lr, "R1.ρ$(round(ρ,digits=2)).cell$(c)")
            push!(fr, PR[c,k2]/(PR[c,k0]+fl)); push!(lr, "R2.ρ$(round(ρ,digits=2)).cell$(c)")
        end
    end
    vcat(f1, fr), vcat(l1, lr)
end

D=joinpath(homedir(),"Julia","DATABASES","EMNIST"); S=joinpath(D,"emnist_source_files")
tr_i=read_emnist_images(joinpath(D,"emnist-balanced-train-images-idx3-ubyte"))
tr_l=read_emnist_labels(joinpath(D,"emnist-balanced-train-labels-idx1-ubyte"))
te_i=read_emnist_images(joinpath(S,"emnist-balanced-test-images-idx3-ubyte"))
te_l=read_emnist_labels(joinpath(S,"emnist-balanced-test-labels-idx1-ubyte"))
NAMES=[LoadEMNIST.emnist_class_name(i) for i in 1:47]
iF=findfirst(==("F"),NAMES); if_=findfirst(==("f"),NAMES)
@printf("F is class %d, f is class %d\n", iF, if_)

trsel=findall(c->c==iF||c==if_, tr_l); tesel=findall(c->c==iF||c==if_, te_l)
ytr=[tr_l[i]==iF ? 1 : 2 for i in trsel]; yte=[te_l[i]==iF ? 1 : 2 for i in tesel]
@printf("F/f subset: %d train, %d test (balanced: %.1f %% / %.1f %%)\n\n",
        length(trsel), length(tesel), 100mean(ytr.==1), 100mean(yte.==1))

function extract(imgs, sel)
    f1,lab = allblocks(@view imgs[:,:,sel[1]])
    F = zeros(Float32, length(sel), length(f1))
    Threads.@threads for i in eachindex(sel)
        F[i,:] = allblocks(@view imgs[:,:,sel[i]])[1]
    end
    F, lab
end
t0=time(); FTR,LAB = extract(tr_i,trsel); FTE,_ = extract(te_i,tesel)
@printf("extracted %d + %d images in %.0f s → %d features\n\n",
        length(trsel), length(tesel), time()-t0, size(FTR,2))

cols(p...) = findall(x->any(startswith(x,q) for q in p), LAB)
const OL=cols("orient","lowpass"); const AB=cols("A1","A2"); const RY=cols("R0","R1","R2")

stdz(a,b)=(μ=vec(mean(a,dims=1));σ=vec(std(a,dims=1));σ[σ.<=0].=1f0;
           (clamp.((a.-μ')./σ',-3,3), clamp.((b.-μ')./σ',-3,3)))
function acc(cs; seed=1, hidden=64, epochs=40)
    a,b = stdz(FTR[:,cs],FTE[:,cs]); A=permutedims(a); B=permutedims(b)
    Random.seed!(seed)
    m=Chain(Dense(length(cs)=>hidden,relu),Dense(hidden=>2))
    opt=Flux.setup(Flux.Adam(1f-3),m); Y=onehotbatch(ytr,1:2); n=size(A,2); best=0.0
    for _ in 1:epochs
        p=randperm(n)
        for i in 1:64:n
            idx=p[i:min(i+63,n)]
            _,gs=Flux.withgradient(mm->Flux.logitcrossentropy(mm(A[:,idx]),Y[:,idx]),m)
            Flux.update!(opt,m,gs[1])
        end
        best=max(best, mean(onecold(m(B),1:2).==yte))
    end
    best
end

println("="^74); println("F vs f — binary, no 40-class dilution"); println("="^74)
@printf("\n%-34s %5s %10s\n","features","n","accuracy")
for (nm,cs) in [("orient+lowpass", OL), ("A₁+A₂", AB), ("ray harmonics c₀,|c₁|,|c₂|", RY),
                ("orient+lowpass + A₁+A₂", vcat(OL,AB)),
                ("orient+lowpass + rays", vcat(OL,RY)),
                ("everything", vcat(OL,AB,RY))]
    vs = [acc(cs; seed=s) for s in 1:3]
    @printf("%-34s %5d %8.2f %% ± %.2f\n", nm, length(cs), 100mean(vs), 100std(vs))
    flush(stdout)
end

# ---------------------------------------------------------------- R²(A ← orient)
println("\n" * "="^74)
println("How much of the A block is linearly recoverable from orient?")
println("="^74)
"Least-squares R² per target column, fit on train, scored on test."
function r2(Xtr,Ytr,Xte,Yte)
    A=hcat(ones(Float32,size(Xtr,1)), Xtr); B=hcat(ones(Float32,size(Xte,1)), Xte)
    W=A\Ytr; P=B*W
    [1 - sum(abs2, Yte[:,j].-P[:,j])/max(sum(abs2, Yte[:,j].-mean(Yte[:,j])),eps())
     for j in 1:size(Yte,2)]
end
rr = r2(FTR[:,OL],FTR[:,AB],FTE[:,OL],FTE[:,AB])
@printf("\nF/f images:  R²(A ← orient)  median %.3f   mean %.3f   [%.3f, %.3f]\n",
        median(rr), mean(rr), minimum(rr), maximum(rr))
rc = r2(FTR[:,OL],FTR[:,RY],FTE[:,OL],FTE[:,RY])
@printf("F/f images:  R²(rays ← orient) median %.3f   mean %.3f   [%.3f, %.3f]\n",
        median(rc), mean(rc), minimum(rc), maximum(rc))

if isfile(P5A)
    C=deserialize(P5A); L5=C.LAB
    o5=findall(x->startswith(x,"orient")||startswith(x,"lowpass"), L5)
    a5=findall(x->startswith(x,"A1")||startswith(x,"A2"), L5)
    r5 = r2(C.NEWtr[:,o5], C.NEWtr[:,a5], C.NEWte[:,o5], C.NEWte[:,a5])
    @printf("\nall 40 classes, 112,800 images:  R²(A ← orient)  median %.3f   mean %.3f\n",
            median(r5), mean(r5))
    @printf("  columns with R² > 0.9: %d of %d   > 0.75: %d\n",
            count(>(0.9), r5), length(r5), count(>(0.75), r5))
end
println("\nHigh R² ⇒ near-substitutable on this data. Low R² with no accuracy gain ⇒ A")
println("carries independent structure the task ignores.")
