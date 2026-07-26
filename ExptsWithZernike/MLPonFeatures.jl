### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 90000000-0000-0000-0000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 90000000-0000-0000-0000-000000000002
begin
    using PlutoUI
    using Plots
    using Statistics
    using Printf
    using Random
    using Flux
    using OneHotArrays
end

# ╔═╡ 90000000-0000-0000-0000-000000000003
begin
    include(joinpath(@__DIR__, "..", "LoadEMNIST.jl"))
    using .LoadEMNIST
end

# ╔═╡ 90000000-0000-0000-0000-000000000004
md"""
# The same features through an MLP — raw scalars vs FPE codes

Everything so far has been scored with **leave-one-out nearest-class-mean**, a
deliberately weak classifier chosen to make the *features* the subject. This notebook
swaps in a **plain fully-connected MLP** (no convolution) trained the way EMNIST is
normally trained — on the **official 112,800-image train split**, evaluated on the
**official 18,800-image test split** — so the numbers are comparable to published work.

The question is how to **code** the 156 features on the way in. Four arms through an
identical net:

| arm | input | what it tests |
|:--|--:|:--|
| **raw** | 156 | the features as scalars, standardised |
| **concat-FPE** | 156·d | each scalar as its own FPE code, concatenated — this *is* random Fourier features, so it asks whether the kernel expansion helps |
| **bundle-FPE** | 2·d | `V = (1/√N) Σ_j R_j ⊙ z_j^{x_j}` — the actual VSA object, fixed width, superposition noise ≈ √(N/d) |
| **pixels** | 784 | the 28×28 image itself, as a reference against published MLP baselines |

plus a diagnostic: **analytically unbind** the bundle and decode each scalar back, to
separate "the information was destroyed by superposition" from "SGD could not find the
unbinding".

### What to expect

Bundling cannot *add* information — it is a lossy function of the 156 scalars — so the
interesting question is not whether it beats raw but **how much it loses as a function
of `d`**, which is a capacity measurement of the VSA code itself. Concatenated FPE can
in principle help, by the usual random-Fourier-features argument.

**EMNIST-Balanced context:** SOTA is ≈ 91 % and convolutional; published *non*-convolutional
MLP baselines land ≈ 84–85 %; the original paper's linear/OPIUM baseline is ≈ 78 %.
Our feature arms see 156 numbers rather than 784 pixels, so they are not competing for
SOTA — the pixel arm is the honest yardstick.
"""

# ╔═╡ 90000000-0000-0000-0000-000000000005
begin
    const IMG = 112
    const NCLASS = 47
    const EMNIST_DIR = joinpath(homedir(), "Julia", "DATABASES", "EMNIST")
    # the official test split ships as idx inside emnist_source_files/ — same format as
    # the train files, so the same parser reads it and there is no CSV-orientation risk
    const SRC_DIR = joinpath(EMNIST_DIR, "emnist_source_files")
    const CLASSNAMES = [LoadEMNIST.emnist_class_name(i) for i in 1:NCLASS]
end

# ╔═╡ 90000000-0000-0000-0000-000000000006
# ---- feature extraction: blocks Z and F, identical to AllClassesDiagnosticity.jl ----
begin
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
    gauss_window(P)=(σ=Float32(P)/4;c=Float32((P+1)/2);
                     Float32[exp(-((i-c)^2+(j-c)^2)/(2σ^2)) for i in 1:P,j in 1:P])
    dft_basis(P,K)=ComplexF32[cis(-2f0π*Float32(k)*Float32(n-1)/Float32(P)) for k in -K:K, n in 1:P]
    function fcell(p,w,K,B)
        F=(B*ComplexF32.(w.*p)*transpose(B))./Float32(sum(w))
        a0=real(F[K+1,K+1]); pw=0f0; s2=ComplexF32(0); s4=ComplexF32(0); rings=zeros(Float32,3)
        @inbounds for v in -K:K, u in -K:K
            (v==0&&u==0)&&continue
            e=abs2(F[v+K+1,u+K+1]); φ=atan(Float32(v),Float32(u))
            pw+=e; s2+=e*cis(2φ); s4+=e*cis(4φ); rings[clamp(round(Int,hypot(v,u)),1,3)]+=e
        end
        pw<=0 && return zeros(Float64,9)
        Float64[a0, sqrt(pw), real(s2/pw), imag(s2/pw), abs(s2/pw), abs(s4/pw), (rings./pw)...]
    end
    function fourier_grid(img,w,B; N=3, K=3)
        cs=IMG/N; P=size(w,1); out=Float64[]
        for i in 1:N, j in 1:N
            append!(out, fcell(patch_at(img,(i-0.5)*cs,(j-0.5)*cs,P),w,K,B))
        end; out
    end
    function zrad(n,m,ρ)
        m=abs(m); ((n-m)%2!=0||m>n)&&return 0.0; s=0.0
        for k in 0:((n-m)÷2)
            s+=(-1)^k*factorial(n-k)/(factorial(k)*factorial((n+m)÷2-k)*factorial((n-m)÷2-k))*ρ^(n-2k)
        end; s
    end
    zorders(nmax)=[(n,m) for n in 0:nmax for m in 0:n if (n-m)%2==0]
    zmask(S)=(c=(S+1)/2;[hypot((i-c)/(S/2),(j-c)/(S/2))<=1 for i in 1:S,j in 1:S])
    function zbasis(S,nmax)
        ords=zorders(nmax);c=(S+1)/2;Rd=S/2;dA=1/Rd^2
        B=[zeros(ComplexF64,S,S) for _ in ords]
        for i in 1:S,j in 1:S
            y=(i-c)/Rd;x=(j-c)/Rd;ρ=hypot(y,x);ρ>1&&continue;θ=atan(y,x)
            for (k,(n,m)) in enumerate(ords); B[k][i,j]=(n+1)/π*zrad(n,m,ρ)*cis(-m*θ)*dA; end
        end; B,ords
    end
    function fit_disc(img; q=0.98)
        N=size(img,1);c=(N+1)/2;w=Float64.(img);tot=sum(w);tot<=0&&return img
        cy=0.0; cx=0.0
        @inbounds for i in 1:N,j in 1:N; cy+=w[i,j]*i; cx+=w[i,j]*j; end
        cy/=tot; cx/=tot
        ds=Float64[]
        @inbounds for i in 1:N,j in 1:N; img[i,j]>0.25f0 && push!(ds,hypot(i-cy,j-cx)); end
        isempty(ds)&&return img
        R=quantile(ds,q); R<=0&&return img
        s=R/(N/2); out=zeros(Float32,N,N)
        @inbounds for i in 1:N,j in 1:N; out[i,j]=bilinear(img,cy+(i-c)*s,cx+(j-c)*s); end; out
    end

    "Blocks Z (75) and F (81) plus the raw 784 pixels, for a stack of 28×28 images."
    function extract_features(imgs28)
        n = size(imgs28,3)
        P = (p=round(Int,1.3*IMG/3); isodd(p) ? p : p+1)
        w = gauss_window(P); Bf = dft_basis(P,3)
        BZ, OZ = zbasis(IMG,8); MZ = zmask(IMG)
        Z = zeros(Float32,n,3*length(OZ)); F = zeros(Float32,n,81); PX = zeros(Float32,n,784)
        for i in 1:n
            raw = @view imgs28[:,:,i]
            PX[i,:] = vec(raw)
            big = upsample(raw)
            F[i,:] = fourier_grid(big, w, Bf)
            A = let f=Float64.(fit_disc(big)); m=mean(f[MZ]); f=f.-m; f[.!MZ].=0.0
                [sum(b.*f) for b in BZ]
            end
            Z[i,:] = vcat(abs.(A), real.(A), imag.(A))
        end
        Z, F, PX
    end
end

# ╔═╡ 90000000-0000-0000-0000-000000000007
# ---- encoders: raw / concatenated FPE / bundled FPE, + the unbinding diagnostic ----
begin
    "Standardise with *train* statistics and clip, so FPE phases stay bounded."
    function standardise(Xtr, Xte; clip=3f0)
        μ=vec(mean(Xtr,dims=1)); σ=vec(std(Xtr,dims=1)); σ[σ .<= 0] .= 1f0
        f(X)=clamp.((X .- μ') ./ σ', -clip, clip)
        f(Xtr), f(Xte)
    end

    # The encoders emit **features × samples** (the orientation Flux wants) and are
    # threaded over images: the cost is elementwise cos/sin, which never reaches BLAS,
    # so without threads a large-d bundle takes minutes. Launch Pluto with
    # `julia -t 16` (or set JULIA_NUM_THREADS) to get the benefit.
    """
    Concatenated FPE — every scalar gets its own code
    `V_j = [cos(θ_jk x_j), sin(θ_jk x_j)]`, codes concatenated ⇒ `N·d` inputs.
    With `θ ~ N(0,σφ²)` the induced kernel is `exp(-σφ²(x-y)²/2)`: this is exactly
    random Fourier features, so `σφ` is an inverse kernel width.
    """
    function fpe_concat(Xtr, Xte, d, σφ; seed=1)
        rng=MersenneTwister(seed); nf=size(Xtr,2); h=d÷2
        Θ=randn(rng,Float32,nf,h).*Float32(σφ)
        function enc(X)
            n=size(X,1); out=zeros(Float32, nf*d, n)
            Threads.@threads for i in 1:n
                @inbounds for j in 1:nf
                    base=(j-1)*d; x=X[i,j]
                    for k in 1:h
                        p=x*Θ[j,k]; out[base+k,i]=cos(p); out[base+h+k,i]=sin(p)
                    end
                end
            end
            out
        end
        enc(Xtr), enc(Xte)
    end

    """
    Bundled FPE (FHRR) — one fixed-width vector however many features there are:
    `V = (1/√N) Σ_j R_j ⊙ z_j^{x_j}` with random role phases `R_j`, fed as
    `[Re V, Im V]` ⇒ `2d` inputs. Unbinding one item leaves crosstalk ≈ √(N/d).
    """
    function fpe_bundle(Xtr, Xte, d, σφ; seed=1)
        rng=MersenneTwister(seed); nf=size(Xtr,2)
        Θ=randn(rng,Float32,nf,d).*Float32(σφ); Φ=rand(rng,Float32,nf,d).*2f0π
        s=1f0/sqrt(Float32(nf))
        function enc(X)
            n=size(X,1); out=zeros(Float32, 2d, n)
            Threads.@threads for i in 1:n
                @inbounds for j in 1:nf
                    x=X[i,j]
                    for k in 1:d
                        p=x*Θ[j,k]+Φ[j,k]; out[k,i]+=cos(p); out[d+k,i]+=sin(p)
                    end
                end
                @inbounds for k in 1:2d; out[k,i]*=s; end
            end
            out
        end
        enc(Xtr), enc(Xte)
    end

    """
    Diagnostic: build the bundle, unbind analytically (`u_j = V ⊙ conj(R_j)`), then
    decode `x̂_j = argmax_x Re⟨u_j, z_j^x⟩` on a grid. Returns per-feature R².
    R² ≈ 1 ⇒ the information survived superposition and any MLP shortfall is an
    optimisation failure; R² ≪ 1 ⇒ superposition genuinely destroyed it.
    """
    function unbind_fidelity(X, d, σφ; seed=1, nsample=400, G=121, clip=3f0)
        rng=MersenneTwister(seed); nf=size(X,2)
        Θ=randn(rng,Float32,nf,d).*Float32(σφ); Φ=rand(rng,Float32,nf,d).*2f0π
        idx=randperm(MersenneTwister(99),size(X,1))[1:min(nsample,size(X,1))]
        Xs=X[idx,:]; n=length(idx)
        Vr=zeros(Float32,n,d); Vi=zeros(Float32,n,d)
        for j in 1:nf
            P=Xs[:,j]*Θ[j,:]' .+ Φ[j,:]'; Vr .+= cos.(P); Vi .+= sin.(P)
        end
        grid=collect(range(-clip,clip,length=G)); r2=zeros(Float64,nf)
        for j in 1:nf
            cφ=cos.(Φ[j,:])'; sφ=sin.(Φ[j,:])'
            Ur=Vr.*cφ .+ Vi.*sφ; Ui=Vi.*cφ .- Vr.*sφ
            Gr=cos.(Float32.(grid)*Θ[j,:]'); Gi=sin.(Float32.(grid)*Θ[j,:]')
            S=Ur*Gr' .+ Ui*Gi'
            x̂=[grid[argmax(view(S,i,:))] for i in 1:n]
            xt=Xs[:,j]; ss=sum(abs2, xt .- mean(xt))
            r2[j] = ss<=0 ? 0.0 : 1 - sum(abs2, x̂ .- xt)/ss
        end
        r2
    end
end

# ╔═╡ 90000000-0000-0000-0000-000000000008
# ---- the MLP: plain fully-connected, trained the usual way ----
begin
    """
    `hidden` is a vector of layer widths. Adam + minibatch softmax cross-entropy,
    official test split scored every epoch. No convolution anywhere.

    `Xtr`/`Xte` are **features × samples** — the orientation the encoders produce, so
    nothing is transposed or copied here.
    """
    function train_mlp(Xtr_t, ytr, Xte_t, yte; hidden=[512], epochs=30, batch=128, lr=1f-3,
                       dropout=0.0, seed=1, verbose=false)
        Random.seed!(seed)
        layers=Any[]; d_in=size(Xtr_t,1)
        for h in hidden
            push!(layers, Dense(d_in=>h, relu))
            dropout>0 && push!(layers, Dropout(Float32(dropout)))
            d_in=h
        end
        push!(layers, Dense(d_in=>NCLASS))
        model=Chain(layers...); opt=Flux.setup(Flux.Adam(lr), model)
        Ytr=onehotbatch(ytr,1:NCLASS)
        n=size(Xtr_t,2); hist=(loss=Float64[], test=Float64[])
        function acc(X,y)
            c=0
            for i in 1:10000:size(X,2)
                j=min(i+9999,size(X,2)); c+=sum(onecold(model(view(X,:,i:j)),1:NCLASS).==view(y,i:j))
            end
            c/length(y)
        end
        for ep in 1:epochs
            Flux.trainmode!(model); perm=randperm(n); tot=0.0; nb=0
            for i in 1:batch:n
                idx=perm[i:min(i+batch-1,n)]
                xb=Xtr_t[:,idx]; yb=Ytr[:,idx]
                l,gs=Flux.withgradient(m->Flux.logitcrossentropy(m(xb),yb), model)
                Flux.update!(opt,model,gs[1]); tot+=l; nb+=1
            end
            Flux.testmode!(model)
            push!(hist.loss,tot/nb); push!(hist.test,acc(Xte_t,yte))
            verbose && @printf("    ep %2d  loss %.4f  test %.4f\n", ep, hist.loss[end], hist.test[end])
        end
        model, hist
    end
end

# ╔═╡ 90000000-0000-0000-0000-000000000009
md"""
### Data

The official split is **112,800 train / 18,800 test**, 2400 and 400 per class. Feature
extraction runs at ≈ 1250 images/s, so the full set takes ≈ 105 s; the sliders below let
you work on a subsample first. **Use the full split for any number you intend to quote.**

train images per class: $(@bind ntr Slider(100:100:2400, default=400, show_value=true))
test images per class: $(@bind nte Slider(50:50:400, default=400, show_value=true))
"""

# ╔═╡ 90000000-0000-0000-0000-00000000000a
begin
    function load_split(imgpath, labpath, per_class)
        I = read_emnist_images(imgpath); L = read_emnist_labels(labpath)
        keep = Int[]
        for c in 1:NCLASS
            idx = findall(==(c), L); append!(keep, idx[1:min(per_class,length(idx))])
        end
        I[:,:,keep], L[keep]
    end
    tr_imgs, ytr = load_split(joinpath(EMNIST_DIR,"emnist-balanced-train-images-idx3-ubyte"),
                              joinpath(EMNIST_DIR,"emnist-balanced-train-labels-idx1-ubyte"), ntr)
    te_imgs, yte = load_split(joinpath(SRC_DIR,"emnist-balanced-test-images-idx3-ubyte"),
                              joinpath(SRC_DIR,"emnist-balanced-test-labels-idx1-ubyte"), nte)
    Ztr,Ftr,Ptr = extract_features(tr_imgs)
    Zte,Fte,Pte = extract_features(te_imgs)
    RAWtr, RAWte = standardise(hcat(Ztr,Ftr), hcat(Zte,Fte))
    Markdown.parse("**$(length(ytr)) train / $(length(yte)) test images** · " *
        "features **$(size(RAWtr,2))** (Z $(size(Ztr,2)) + F $(size(Ftr,2))) · " *
        "pixels **784** · chance **$(round(100/NCLASS,digits=2)) %**")
end

# ╔═╡ 90000000-0000-0000-0000-00000000000b
md"""
### Controls

arm: $(@bind arm Select(["raw"=>"raw 156 features", "concat"=>"concatenated FPE",
                         "bundle"=>"bundled FPE (VSA)", "pixels"=>"raw 784 pixels"]))
depth: $(@bind depth Select([1=>"1 hidden layer", 2=>"2 hidden layers", 3=>"3 hidden layers"], default=2))
width: $(@bind width Select([256,512,1024], default=512))
epochs: $(@bind epochs Slider(5:5:60, default=30, show_value=true))

**FPE code size d**: $(@bind fpe_d Select([16,32,64,256,512,1024,2048,4096], default=32))
**FPE bandwidth σφ**: $(@bind fpe_s Select([0.25,0.5,1.0,2.0,4.0], default=1.0))

*(`d` means dims per scalar for concat — input `156·d` — and total bundle width for
bundle — input `2d`.)*

run: $(@bind go CheckBox(default=false))
"""

# ╔═╡ 90000000-0000-0000-0000-00000000000c
if !go
    md"*(tick **run** to train — a full-split run takes minutes)*"
else
    let
        # every arm ends up features × samples
        Xtr, Xte = arm=="raw"    ? (permutedims(RAWtr), permutedims(RAWte)) :
                   arm=="pixels" ? (permutedims(Ptr), permutedims(Pte)) :
                   arm=="concat" ? fpe_concat(RAWtr, RAWte, fpe_d, fpe_s) :
                                   fpe_bundle(RAWtr, RAWte, fpe_d, fpe_s)
        t0=time()
        _, h = train_mlp(Xtr, ytr, Xte, yte; hidden=fill(width,depth), epochs=epochs)
        Markdown.parse(@sprintf("**%s** · input **%d** · %d×%d hidden · **test %.2f %%** (best %.2f %%) · %.0f s",
                        arm, size(Xtr,1), depth, width, 100h.test[end], 100maximum(h.test), time()-t0)),
        plot(1:epochs, 100 .*h.test; lw=2, marker=:circle, ms=3, label="test",
             xlabel="epoch", ylabel="accuracy (%)", legend=:bottomright,
             title="$(arm), d=$(fpe_d), σφ=$(fpe_s)", titlefontsize=9, size=(760,320))
    end
end

# ╔═╡ 90000000-0000-0000-0000-00000000000d
md"""
### Does the information survive bundling?

`unbind_fidelity` builds the bundle, unbinds each role analytically, and decodes the
scalar back by correlating against the code grid. **R² ≈ 1** means superposition kept
the information and any MLP shortfall is an optimisation failure; **R² ≪ 1** means the
information is genuinely gone and no training fixes it.

run: $(@bind go_fid CheckBox(default=false))
"""

# ╔═╡ 90000000-0000-0000-0000-00000000000e
if !go_fid
    md"*(tick to run the sweep)*"
else
    let
        ds = [128,256,512,1024,2048,4096]
        med = Float64[]; lo = Float64[]
        for d in ds
            r2 = unbind_fidelity(RAWtr, d, fpe_s; nsample=300)
            push!(med, median(r2)); push!(lo, quantile(r2,0.1))
        end
        p = plot(ds, med; lw=2, marker=:circle, label="median R²", xscale=:log2,
                 xlabel="bundle width d", ylabel="decode R²", ylims=(-0.05,1.05),
                 xticks=(ds,string.(ds)), legend=:bottomright,
                 title="recovering the 156 scalars from the bundle (σφ=$(fpe_s))",
                 titlefontsize=9, size=(760,330))
        plot!(p, ds, lo; lw=2, ls=:dash, marker=:square, label="10th percentile")
        hline!(p, [1.0]; lc=:grey, ls=:dot, label="")
        vline!(p, [size(RAWtr,2)]; lc=:red, ls=:dot, label="d = N = $(size(RAWtr,2))")
        p
    end
end

# ╔═╡ 90000000-0000-0000-0000-00000000000f
md"""
### Notes — measured on the full official split

All figures: official 18,800-image test set, 47 classes, chance 2.13 %, 1×256 hidden,
15 epochs unless noted. Format *final / best*. **Standard error is 0.25 %, so
differences below ~0.5 points are not meaningful.**

A full standalone write-up — defining Zernike moments, the Fourier grid, FPE, binding
and bundling from scratch, with the interpretation — is in
**`../README_MLP_FPE_Experiment.md`**.

**1. Depth doesn't help** (raw 156, 30 epochs): 1×256 **85.87** / 1×512 85.56 /
2×512 85.29 / 3×512 85.03, all peaking at 86.6–86.8. The features have already done the
nonlinear work; extra capacity only overfits. 15 epochs beats 30 (86.43 vs 85.87).

**2. References.**

| input | numbers | final / best |
|:--|--:|--:|
| **raw 156 features** | 156 | **86.43 / 86.71** |
| raw 784 pixels | 784 | 83.65 / 83.95 |
| block Z alone | 75 | 83.96 / 84.27 *(30 ep)* |
| block F alone | 81 | 85.86 / 86.32 *(30 ep)* |

The features **beat raw pixels by 2.8 points** at a fifth the size — a genuine
re-description, not a compression. And the same features scored **62.5 %** under
leave-one-out nearest-class-mean: that classifier understated them by 24 points.

**It also overturned one of our earlier conclusions.** Under nearest-class-mean, Z+F
beat either block alone by +6.4 points and we called them complementary. Under the MLP,
F alone is 85.86 and Z+F is 86.43 — 0.6 points, at the edge of significance. The
complementarity was mostly the weak classifier's inability to use F on its own.

**3. Concat-FPE hurts, monotonically** (input `156·d_feat`):

| σφ | d_feat=8 | d_feat=32 |
|--:|--:|--:|
| 0.5 | **85.58** / 85.68 | 84.72 / 84.78 |
| 1.0 | 85.11 / 85.47 | 84.27 / 84.77 |
| 2.0 | 82.94 / 84.22 | 82.30 / 83.73 |

Every cell below raw, falling with both more code and sharper kernel — the best cell is
the one least like an FPE code. Expected in hindsight: this *is* random Fourier
features, which help when you need a fixed nonlinear expansion because your model can't
learn one. An MLP's first layer already is one.

**4. Bundle-FPE is nearly free** (input `2D` regardless of feature count):

| σφ | D=256 | D=512 | D=1024 | D=2048 |
|--:|--:|--:|--:|--:|
| 0.5 | 85.63 / 85.75 | 85.75 / 85.75 | **85.79 / 85.79** | 84.00 / 85.04 |
| 1.0 | 84.98 / 85.15 | 84.98 / 85.14 | 85.00 / 85.37 | 84.21 / 84.63 |
| 2.0 | 81.85 / 82.74 | 82.43 / 83.64 | 82.59 / 84.14 | 82.50 / 83.38 |

**0.6 points** below raw, for 156 features superposed into one 1,024-number vector. Flat
in `D` from 256–1024, then dropping at 2048 — that drop is overfitting (final-vs-best
gap widens) while recoverability is still rising, not information loss.

**5. Unbinding fidelity** — median R² recovering the 156 scalars analytically, no
learning:

| D | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|:--|--:|--:|--:|--:|--:|--:|
| σφ=0.5 | −1.19 | −0.31 | 0.37 | 0.75 | 0.88 | 0.94 |
| σφ=1.0 | −1.36 | −0.17 | 0.56 | 0.90 | 0.96 | 0.98 |
| σφ=2.0 | −1.38 | −0.48 | 0.32 | 0.86 | 0.99 | 1.00 |

Negative below `D ≈ N = 156` exactly as `SNR ≈ √(D/N)` predicts. The bandwidth optimum
*moves with width* — σφ=1 best at 512–1024, σφ=2 best at 2048–4096: bandwidth buys
resolution, width buys capacity, and resolution is only affordable once you have
capacity.

**6. The result worth keeping: recovery and classification come apart.**

| D (σφ=0.5) | 256 | 512 | 1024 | 2048 |
|:--|--:|--:|--:|--:|
| decode R² | **−0.31** | 0.37 | 0.75 | 0.88 |
| accuracy | **85.63** | 85.75 | 85.79 | 84.00 |

At `D=256` the individual values are **unrecoverable** (R² negative — worse than
guessing the mean) yet classification is within a point of clean features. Decode R²
sweeps from negative to 0.75 while accuracy moves 0.16 points. Classification never
inverts the superposition; it only needs directions along which the classes separate,
and those survive crosstalk that destroys per-item readback.

**So `√(D/N)` is the right rule for retrieval and the wrong one for classification.**
Sized for readback you'd want `D ≈ 4096`; sized for a discriminative readout `D = 256`
does the same job — an order of magnitude of over-provisioning if you apply the wrong
rule.

**What this doesn't show:** only one classifier family (a convnet on pixels would beat
everything here); a coarse 3×4 sweep with one seed per cell; and a bundle where all 156
features are always present, which is *not* the variable-length structured case FPE
actually exists for.
"""

# ╔═╡ Cell order:
# ╠═90000000-0000-0000-0000-000000000001
# ╠═90000000-0000-0000-0000-000000000002
# ╠═90000000-0000-0000-0000-000000000003
# ╟─90000000-0000-0000-0000-000000000004
# ╠═90000000-0000-0000-0000-000000000005
# ╠═90000000-0000-0000-0000-000000000006
# ╠═90000000-0000-0000-0000-000000000007
# ╠═90000000-0000-0000-0000-000000000008
# ╟─90000000-0000-0000-0000-000000000009
# ╠═90000000-0000-0000-0000-00000000000a
# ╟─90000000-0000-0000-0000-00000000000b
# ╠═90000000-0000-0000-0000-00000000000c
# ╟─90000000-0000-0000-0000-00000000000d
# ╠═90000000-0000-0000-0000-00000000000e
# ╟─90000000-0000-0000-0000-00000000000f
