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
end

# ╔═╡ 90000000-0000-0000-0000-000000000003
begin
    include(joinpath(@__DIR__, "..", "LoadEMNIST.module.jl"))
    using .LoadEMNIST
end

# ╔═╡ 90000000-0000-0000-0000-000000000004
md"""
# Diagnosticity on **all 47** EMNIST classes

Every accuracy in this project so far has been measured on the same **12 uppercase
letters** (`O C I L T X K A H Y E F`), the set inherited from
`P0.3_New_Gabor_FPE/KeyPointDiagnosticity.md`. That set is convenient and comparable, but it
is also *easy*: no digits, no lowercase, and none of the genuinely ambiguous glyph
pairs. This notebook re-runs the same descriptors on the **full EMNIST-balanced
47-class set** — 10 digits, 26 uppercase, and the 11 lowercase letters whose shapes
differ from their uppercase forms (`a b d e f g h n q r t`).

Chance drops from 8.3 % to **2.13 %**.

The two descriptors, unchanged from `CombinedZernikeFourier.jl`:

- **Z** — global Zernike on the ink-fitted disc, `n ≤ 8`, `|A|` + `Re/Im` ⇒ 75 numbers.
- **F** — the Fourier tic-tac-toe grid, 9 features × 9 cells ⇒ 81 numbers.

**Three things change at 47 classes** (details in the Notes): the ranking of the two
blocks **flips**, their complementarity **survives but shrinks**, and about a quarter
of all remaining errors turn out to be **confusions between glyphs that are genuinely
the same shape** (`0`/`O`, `1`/`I`/`L`, `2`/`Z`, `9`/`q`).
"""

# ╔═╡ 90000000-0000-0000-0000-000000000005
begin
    const IMG = 112
    const SET12 = ["O","C","I","L","T","X","K","A","H","Y","E","F"]
    # EMNIST-balanced canonical order: uppercase, then the lowercase subset, then digits
    const ORDER47 = LoadEMNIST.default_class_order()
    const NAMES47 = [LoadEMNIST.emnist_class_name(i) for i in ORDER47]
    kind(nm) = occursin('/', nm) ? :merged :
               (isdigit(nm[1]) ? :digit : (islowercase(nm[1]) ? :lower : :upper))

    # ---- class groups for merging. DISJOINT BY CONSTRUCTION (asserted below): taking
    # the transitive closure of overlapping pairs is wrong — asserting 6≡G, G≡g and
    # 9≡g chains 6 to 9, which are not the same glyph at all.
    #
    # HOMO: digit↔letter collisions. These really are the same handwritten shape.
    const HOMO = [["0","O"], ["1","I","L"], ["2","Z"], ["5","S"], ["9","g","q"]]
    # CASE: upper/lower pairs, excluding any letter already spoken for by HOMO.
    # NB this merge is *weakly* motivated — EMNIST-balanced already merged the case
    # pairs that look identical (C/c, O/o, …); the 11 lowercase classes it keeps are
    # precisely the ones whose shape differs. Measured below: merging them hurts.
    const CASE = [["A","a"],["B","b"],["D","d"],["E","e"],["F","f"],
                  ["H","h"],["N","n"],["R","r"],["T","t"]]
    let seen=Dict{String,Int}()
        for (k,g) in enumerate(vcat(HOMO,CASE)), s in g
            @assert !haskey(seen,s) "class $s appears in two merge groups — not disjoint"
            seen[s]=k
        end
    end
end

# ╔═╡ 90000000-0000-0000-0000-000000000006
# ---- shared image helpers ----
begin
    @inline function bilinear(M,y,x)
        H,W=size(M); (y<1||x<1||y>H||x>W)&&return 0f0
        y0,x0=floor(Int,y),floor(Int,x); y1,x1=min(y0+1,H),min(x0+1,W); fy,fx=y-y0,x-x0
        (1-fy)*(1-fx)*M[y0,x0]+fy*(1-fx)*M[y1,x0]+(1-fy)*fx*M[y0,x1]+fy*fx*M[y1,x1]
    end
    function upsample(img,N=IMG)
        H,W=size(img); out=zeros(Float32,N,N)
        for i in 1:N,j in 1:N
            out[i,j]=bilinear(img,Float32(1+(i-1)*(H-1)/(N-1)),Float32(1+(j-1)*(W-1)/(N-1)))
        end; out
    end
    function patch_at(img,cy,cx,P)
        N=size(img,1); out=zeros(Float32,P,P); y0=round(Int,cy)-(P-1)÷2; x0=round(Int,cx)-(P-1)÷2
        @inbounds for i in 1:P,j in 1:P; out[i,j]=img[clamp(y0+i-1,1,N),clamp(x0+j-1,1,N)]; end; out
    end
    em = load_emnist(n_images_to_load=60000, n_classes=47)
end;

# ╔═╡ 90000000-0000-0000-0000-000000000007
# ---- BLOCK F: Fourier tic-tac-toe cell features ----
begin
    gauss_window(P)=(σ=Float32(P)/4;c=Float32((P+1)/2);
                     Float32[exp(-((i-c)^2+(j-c)^2)/(2σ^2)) for i in 1:P,j in 1:P])
    const NTH=12; const RADII=(1f0,2f0,3f0)
    polar_kernels(P)=[ComplexF32[cis(-2f0π*ω*((j-1)*cos(θ)+(i-1)*sin(θ))/P) for i in 1:P,j in 1:P]
                      for ω in RADII, θ in range(0,π,length=NTH+1)[1:NTH]]
    dft_basis(P,K)=ComplexF32[cis(-2f0π*Float32(k)*Float32(n-1)/Float32(P)) for k in -K:K, n in 1:P]
    function fcell_lattice(p,w,K)
        P=size(p,1); B=dft_basis(P,K); F=(B*ComplexF32.(w.*p)*transpose(B))./Float32(sum(w))
        a0=real(F[K+1,K+1]); pw=0f0; s2=ComplexF32(0); s4=ComplexF32(0); rings=zeros(Float32,3)
        for v in -K:K, u in -K:K
            (v==0&&u==0)&&continue
            e=abs2(F[v+K+1,u+K+1]); φ=atan(Float32(v),Float32(u))
            pw+=e; s2+=e*cis(2φ); s4+=e*cis(4φ); rings[clamp(round(Int,hypot(v,u)),1,3)]+=e
        end
        pw<=0 && return zeros(Float64,9)
        Float64[a0, sqrt(pw), real(s2/pw), imag(s2/pw), abs(s2/pw), abs(s4/pw), (rings./pw)...]
    end
    function fcell_polar(p,w,Ks)
        A=w.*p; sw=Float32(sum(w)); a0=sum(A)/sw
        E=Float32[abs2(sum(Ks[a,b].*A)/sw) for a in 1:3, b in 1:NTH]
        pw=sum(E); pw<=0 && return zeros(Float64,9)
        s2=ComplexF32(0); s4=ComplexF32(0)
        for a in 1:3,b in 1:NTH; θ=(b-1)*π/NTH; s2+=E[a,b]*cis(2θ); s4+=E[a,b]*cis(4θ); end
        Float64[a0, sqrt(pw), real(s2/pw), imag(s2/pw), abs(s2/pw), abs(s4/pw),
                (vec(sum(E,dims=2))./pw)...]
    end
    function fourier_grid(img; N=3, ov=1.3, sampling="lattice", K=3)
        cs=IMG/N; P=round(Int,ov*cs); isodd(P)||(P+=1); w=gauss_window(P)
        Ks = sampling=="polar" ? polar_kernels(P) : nothing
        out=Float64[]
        for i in 1:N, j in 1:N
            p=patch_at(img,(i-0.5)*cs,(j-0.5)*cs,P)
            append!(out, sampling=="polar" ? fcell_polar(p,w,Ks) : fcell_lattice(p,w,K))
        end
        out
    end
end

# ╔═╡ 90000000-0000-0000-0000-000000000008
# ---- BLOCK Z: global Zernike on the ink-fitted disc ----
begin
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
    function zmoments(img,B,mask; zerodc=true)
        f=Float64.(img); zerodc && (f=f.-mean(f[mask]); f[.!mask].=0.0); [sum(b.*f) for b in B]
    end
    function fit_disc(img; q=0.98)
        N=size(img,1);c=(N+1)/2;w=Float64.(img);tot=sum(w);tot<=0&&return img
        cy=sum(w[i,j]*i for i in 1:N,j in 1:N)/tot; cx=sum(w[i,j]*j for i in 1:N,j in 1:N)/tot
        ds=[hypot(i-cy,j-cx) for i in 1:N,j in 1:N if img[i,j]>0.25f0]; isempty(ds)&&return img
        R=q>=1 ? maximum(ds) : quantile(ds,q); R<=0&&return img
        s=R/(N/2); out=zeros(Float32,N,N)
        for i in 1:N,j in 1:N; out[i,j]=bilinear(img,cy+(i-c)*s,cx+(j-c)*s); end; out
    end
end

# ╔═╡ 90000000-0000-0000-0000-000000000009
# ---- evaluation: η² and a class-sum LOO nearest-class-mean (O(n·C·p)) ----
begin
    function eta2(X,y)
        n,p=size(X); out=zeros(Float64,p)
        for j in 1:p
            col=@view X[:,j]; gm=mean(col); tot=sum((col.-gm).^2); tot<=0 && continue
            bet=0.0
            for c in unique(y); idx=findall(==(c),y); bet+=length(idx)*(mean(col[idx])-gm)^2; end
            out[j]=bet/tot
        end; out
    end

    """
    Leave-one-out nearest-class-mean predictions.

    Columns are standardised once, then (optionally) scaled by `√weights` — applied
    **once**, because re-standardising after weighting would cancel it exactly. Class
    means are kept as running sums with the held-out row subtracted, which makes this
    `O(n·C·p)` instead of the naive `O(n·C·n·p)` — necessary at 47 classes × 30.
    """
    function loo_pred(X, y; weights=nothing)
        n,p = size(X); Z = copy(X)
        for j in 1:p
            s=std(@view Z[:,j]); s<=0 ? (Z[:,j].=0) : (Z[:,j].=(Z[:,j].-mean(Z[:,j]))./s)
        end
        weights !== nothing && (Z .*= sqrt.(max.(weights,0))')
        classes=sort(unique(y)); C=length(classes); cid=Dict(c=>k for (k,c) in enumerate(classes))
        S=zeros(Float64,C,p); cnt=zeros(Int,C)
        for i in 1:n; k=cid[y[i]]; S[k,:] .+= Z[i,:]; cnt[k]+=1; end
        pred=similar(y)
        for i in 1:n
            ki=cid[y[i]]; best=Inf; bk=1
            for k in 1:C
                nk=cnt[k]-(k==ki); nk<=0 && continue
                d=0.0
                @inbounds for j in 1:p
                    mj=(S[k,j]-(k==ki ? Z[i,j] : 0.0))/nk; d+=(Z[i,j]-mj)^2
                end
                d<best && (best=d; bk=k)
            end
            pred[i]=classes[bk]
        end
        pred
    end
    acc(pred,y) = mean(pred .== y)
end

# ╔═╡ 90000000-0000-0000-0000-00000000000a
md"""
### Controls

class set: $(@bind setname Select(["47"=>"all 47 (chance 2.13 %)", "26"=>"26 uppercase (3.85 %)",
                                   "12"=>"the usual 12 (8.33 %)", "10"=>"10 digits (10 %)"]))
**label set**: $(@bind labelset Select(["strict"=>"strict — every class separate",
                                        "homo"=>"merge homoglyphs (0/O, 1/I/L, 2/Z, 5/S, 9/g/q)",
                                        "homocase"=>"merge homoglyphs + case pairs"]))
instances per class: $(@bind n_per Slider(10:5:40, default=30, show_value=true))

**Zernike n_max**: $(@bind znmax Slider(4:1:10, default=8, show_value=true))
**Fourier grid N×N**: $(@bind gridN Slider(2:1:4, default=3, show_value=true))
sampling: $(@bind sampling Select(["lattice"=>"lattice (v,u)", "polar"=>"polar ω×θ"]))

*(47 classes × 30 = 1410 images; features build in a few seconds, the LOO sweep a few more)*
"""

# ╔═╡ 90000000-0000-0000-0000-00000000000b
begin
    base = setname=="47" ? NAMES47 :
           setname=="26" ? [n for n in NAMES47 if kind(n)==:upper] :
           setname=="12" ? SET12 : [n for n in NAMES47 if kind(n)==:digit]
    imgs = Matrix{Float32}[]; blabs = Int[]
    for (ci,c) in enumerate(base)
        v = em.class_images[findfirst(==(c), em.class_names)]
        for k in 1:min(n_per,length(v)); push!(imgs, upsample(v[k])); push!(blabs, ci); end
    end

    # merge labels BEFORE classifying: the classifier is never asked to make the
    # distinction, and the merged class mean pools both members' instances
    groups = labelset=="strict" ? Vector{Vector{String}}() :
             labelset=="homo"   ? HOMO : vcat(HOMO, CASE)
    mapname = Dict(s=>s for s in base)
    for g in groups
        present = [s for s in g if s in base]
        length(present) < 2 && continue
        rep = join(sort(present), "/")
        for s in present; mapname[s] = rep; end
    end
    sel  = sort(unique([mapname[s] for s in base]))
    labs = [findfirst(==(mapname[base[c]]), sel) for c in blabs]

    BZ, OZ = zbasis(IMG, znmax); MZ = zmask(IMG)
    Zblk = reduce(vcat, [ (A=zmoments(fit_disc(im;q=0.98),BZ,MZ;zerodc=true);
                           vcat(abs.(A),real.(A),imag.(A))') for im in imgs ])
    Fblk = reduce(vcat, [ fourier_grid(im; N=gridN, sampling=sampling)' for im in imgs ])
    Cblk = hcat(Zblk, Fblk)

    pZ = loo_pred(Zblk,labs); pF = loo_pred(Fblk,labs); pC = loo_pred(Cblk,labs)
    pW = loo_pred(Cblk,labs; weights=eta2(Cblk,labs))
    hZ = pZ.==labs; hF = pF.==labs; hC = pC.==labs; hW = pW.==labs
    chance = 100/length(sel)

    mg = length(base) - length(sel)
    Markdown.parse("**$(length(imgs)) images · $(length(sel)) classes" *
        (mg > 0 ? " (merged down from $(length(base)))" : "") * " · chance " *
        "$(round(chance,digits=2)) %** — block **Z** $(size(Zblk,2)) numbers (Zernike n≤$(znmax)), " *
        "block **F** $(size(Fblk,2)) numbers ($(gridN)×$(gridN) $(sampling))" *
        (mg > 0 ? "\n\nmerged classes: " * join(["`"*s*"`" for s in sel if occursin('/',s)], ", ") : ""))
end

# ╔═╡ 90000000-0000-0000-0000-00000000000c
let
    eZ, eF = eta2(Zblk,labs), eta2(Fblk,labs)
    rows = [("**Z** — global Zernike",   size(Zblk,2), mean(hZ), acc(loo_pred(Zblk,labs;weights=eZ),labs)),
            ("**F** — Fourier grid",     size(Fblk,2), mean(hF), acc(loo_pred(Fblk,labs;weights=eF),labs)),
            ("**Z + F**",                size(Cblk,2), mean(hC), mean(hW))]
    hdr = "| descriptor | numbers | LOO | η²-weighted |\n|:--|--:|--:|--:|\n"
    body = join([@sprintf("| %s | %d | **%.1f %%** | **%.1f %%** |", t,n,100a,100b) for (t,n,a,b) in rows], "\n")
    g = 100*(mean(hC) - max(mean(hZ), mean(hF)))
    tail = @sprintf("\n\nConcatenation gains **%+.1f points** over the better block. Chance is %.2f %%, so Z+F is **%.0f× chance**.", g, chance, 100*mean(hC)/chance)
    Markdown.parse(hdr*body*tail)
end

# ╔═╡ 90000000-0000-0000-0000-00000000000d
let
    both=mean(hZ.&hF); onlyZ=mean(hZ.&.!hF); onlyF=mean(.!hZ.&hF); neither=mean(.!hZ.&.!hF)
    p1 = bar(["both","only Z","only F","neither"], 100 .*[both,onlyZ,onlyF,neither];
             c=[:seagreen,:steelblue,:goldenrod,:firebrick], label="", ylabel="% of images",
             title="error overlap", titlefontsize=9, guidefontsize=8, tickfontsize=8, grid=false)
    lv=["Z","F","Z+F","Z+F (η²)","best-of-two"]
    vals=100 .*[mean(hZ),mean(hF),mean(hC),mean(hW),mean(hZ.|hF)]
    p2 = bar(lv, vals; c=[:steelblue,:goldenrod,:seagreen,:darkgreen,:grey], label="",
             ylabel="LOO accuracy (%)", title="what the combination recovers  (grey = selection bound, not a ceiling)",
             titlefontsize=9, guidefontsize=8, tickfontsize=8, grid=false, ylims=(0,100))
    for (i,v) in enumerate(vals); annotate!(p2,i,v+3,text(@sprintf("%.1f",v),7)); end
    hline!(p2,[100/length(sel)]; lc=:red, ls=:dash, label="chance")
    plot(p1,p2; layout=(1,2), size=(950,350))
end

# ╔═╡ 90000000-0000-0000-0000-00000000000e
md"""
### Per class

Sorted by accuracy of the combined descriptor. **Blue = uppercase, orange = lowercase,
green = digit, purple = merged class.** The tail on the left is where the interesting
failures live.
"""

# ╔═╡ 90000000-0000-0000-0000-00000000000f
let
    a = [100*mean(hW[findall(==(ci),labs)]) for ci in eachindex(sel)]
    ord = sortperm(a)
    cols = [kind(sel[i])==:upper ? :steelblue : kind(sel[i])==:lower ? :darkorange :
            kind(sel[i])==:digit ? :seagreen : :purple for i in ord]
    p = bar(1:length(sel), a[ord]; c=cols, label="", xticks=(1:length(sel), sel[ord]),
            ylabel="LOO accuracy (%)", ylims=(0,105), tickfontsize=7, guidefontsize=8,
            title="per-class accuracy, Z+F (η²-weighted)", titlefontsize=9, grid=false)
    hline!(p,[100*mean(hW)]; lc=:black, ls=:dash, label="overall")
    plot(p; size=(max(950, 26*length(sel)), 330))
end

# ╔═╡ 90000000-0000-0000-0000-000000000010
let
    conf = Dict{Tuple{String,String},Int}()
    for i in eachindex(labs)
        pC[i]==labs[i] && continue
        k=(sel[labs[i]], sel[pC[i]]); conf[k]=get(conf,k,0)+1
    end
    top = sort(collect(conf), by=x->-x[2])[1:min(15,length(conf))]
    hdr = "**Most frequent confusions** (Z + F, unweighted) — how many instances of the " *
          "true class were called the other.\n\n" *
          "| true → predicted | n | note |\n|:--|--:|:--|\n"
    inhomo(a,b) = any(g -> a in g && b in g, HOMO)
    same(a,b) = inhomo(a,b) ? "homoglyph — mergeable" :
                (!occursin('/',a) && !occursin('/',b) && uppercase(a)==uppercase(b)) ?
                "case pair (EMNIST keeps these apart deliberately)" : ""
    body = join([@sprintf("| `%s` → `%s` | %d | %s |", a, b, n, same(a,b)) for ((a,b),n) in top], "\n")
    Markdown.parse(hdr*body)
end

# ╔═╡ 90000000-0000-0000-0000-000000000011
md"""
### Merging vs. rescoring

Two different things can be done about `0`/`O`:

- **rescore** — keep the 47-way classifier and simply forgive confusions inside a
  group. The classifier still had to split the pair, and class means stay tight.
- **merge** — relabel *before* classifying (what the *label set* control above does).
  The distinction is never demanded, but the class mean now pools both members.

They give slightly different answers. Tick to run the comparison (one extra
classification pass over the 47-class labels).

compare: $(@bind cmp_merge CheckBox(default=false))
"""

# ╔═╡ 90000000-0000-0000-0000-000000000012
if !cmp_merge
    md"*(comparison not run — tick the box above)*"
else
    let
        # always compare on the strict 47-class labels, whatever the control is set to
        b47 = NAMES47
        i47 = Int[]; l47 = Int[]
        for (ci,c) in enumerate(b47)
            v = em.class_images[findfirst(==(c), em.class_names)]
            for k in 1:min(n_per,length(v)); push!(i47, length(i47)+1); push!(l47, ci); end
        end
        im47 = Matrix{Float32}[]
        for c in b47
            v = em.class_images[findfirst(==(c), em.class_names)]
            for k in 1:min(n_per,length(v)); push!(im47, upsample(v[k])); end
        end
        Z4 = reduce(vcat, [ (A=zmoments(fit_disc(im;q=0.98),BZ,MZ;zerodc=true);
                             vcat(abs.(A),real.(A),imag.(A))') for im in im47 ])
        F4 = reduce(vcat, [ fourier_grid(im; N=gridN, sampling=sampling)' for im in im47 ])
        C4 = hcat(Z4,F4)

        mapn = Dict(s=>s for s in b47)
        for g in HOMO
            rep = join(sort(g),"/"); for s in g; mapn[s]=rep; end
        end
        reps = sort(unique(values(mapn)))
        ml = [findfirst(==(mapn[b47[c]]), reps) for c in l47]

        p_strict = loo_pred(C4, l47; weights=eta2(C4,l47))
        resc = mean([mapn[b47[c]] for c in l47] .== [mapn[b47[p]] for p in p_strict])
        p_merge = loo_pred(C4, ml; weights=eta2(C4,ml))
        merg = mean(p_merge .== ml)
        strict = mean(p_strict .== l47)
        hdr = "| approach | classes the model must separate | accuracy |\n|:--|--:|--:|\n"
        rows = [@sprintf("| strict 47-way | 47 | %.1f %% |", 100strict),
                @sprintf("| 47-way, homoglyph errors **forgiven** | 47 | %.1f %% |", 100resc),
                @sprintf("| **merged labels, %d-way** | %d | **%.1f %%** |", length(reps), length(reps), 100merg)]
        tail = @sprintf("\n\nRescoring edges out merging by **%.1f points**: the 47-way model keeps one tight mean per glyph (30 instances each), whereas a merged class pools 60–90 instances into a single mean that has to cover both members. The difference is small because the homoglyph members really do overlap — which is the point.", 100*(resc-merg))
        Markdown.parse(hdr * join(rows,"\n") * tail)
    end
end

# ╔═╡ 90000000-0000-0000-0000-000000000013
md"""
### Notes — what was measured

47 classes × 30 instances = **1410 images**, chance **2.13 %**. Block Z = global Zernike
`n ≤ 8` on the 98th-percentile ink-fitted disc (75 numbers); block F = 3×3 grid, lattice
sampling, 9 features per cell (81 numbers).

**1. Headline numbers, and how they compare to the 12-class set.**

| descriptor | 47 classes | *12 classes* |
|:--|--:|--:|
| Z — global Zernike | 55.5 % (η² 57.4 %) | *76.4 % (77.5 %)* |
| F — Fourier 3×3 | 58.2 % (η² 60.4 %) | *74.2 % (77.2 %)* |
| **Z + F** | **62.5 % (η² 65.3 %)** | ***82.8 % (84.2 %)*** |
| best-of-two selection bound | 68.4 % | *85.3 %* |
| chance | 2.13 % | *8.33 %* |

Z+F at 62.5 % is **29× chance** here versus 10× on the 12-class set, so in
information terms the harder problem is being handled *better*, even though the raw
accuracy is 20 points lower.

**2. The ranking of the two blocks flips.** On 12 uppercase letters Zernike led
(76.4 % vs 74.2 %); on all 47 the Fourier grid leads (58.2 % vs 55.5 %), and it leads
η²-weighted too (60.4 % vs 57.4 %). Adding digits and lowercase adds many classes that
differ by *where* their strokes are rather than by global shape — `b` vs `d`, `q` vs
`p`-like forms, `6` vs `G` — and a spatial grid is built for exactly that, while a
centred global moment description is not. **Spatial layout scales with class count
better than global shape does.**

**3. Complementarity survives but shrinks.** Concatenation gains **+4.3 points** over
the better block (58.2 → 62.5) against **+6.4** at 12 classes. The overlap explains it:
both correct 45.2 %, only Z 10.2 %, only F 13.0 %, **neither 31.6 %**. That
"neither" share more than doubles from the 12-class 14.7 %, so a much larger part of the
problem is simply beyond both descriptors, and the exploitable disagreement (23.2 %,
vs 20.0 % before) is a smaller share of what is left. The best-of-two selection
bound is 68.4 % — and as at 12 classes, concatenation is not held to it: of the **445**
images neither block gets right it recovers **28** (**51** with η² weighting), while
losing 112 that one block had.

**4. η²-weighting matters more here** — +2.8 points (62.5 → 65.3) versus +1.4 on the
12-class set. With 47 class means to estimate from 29 examples each, the equal-weighted
nearest-mean classifier is more easily diluted, so down-weighting the noisy columns pays
off more. *(Implementation note: the weights must be applied* once *after
standardisation — standardising again afterwards cancels them exactly, which silently
turns the weighted run into the unweighted one.)*

**5. A quarter of the remaining error is the label set, not the descriptor.**
The top confusions are `F`→`f` (11), `2`→`Z` (11), `0`→`O` (11), `q`→`9` (9), `O`→`0` (9),
`L`→`1` (8), `1`→`I` (7) — mostly pairs that are the *same handwritten shape*
(`F`/`f` is the exception; see §7).

Re-running the whole diagnostic with the labels **merged before classification**
(the *label set* control), 30 instances per original class throughout:

| label set | classes | chance | Z | F | **Z + F** |
|:--|--:|--:|--:|--:|--:|
| strict | 47 | 2.13 % | 55.5 / 57.4 % | 58.2 / 60.4 % | **62.5 / 65.3 %** |
| **homoglyphs merged** | **40** | **2.50 %** | 59.4 / 62.0 % | 62.7 / 65.2 % | **67.8 / 70.6 %** |
| homoglyphs + case merged | 31 | 3.23 % | 57.4 / 59.4 % | 61.7 / 63.6 % | 66.9 / 67.7 % |

*(each cell is LOO / η²-weighted)*

Merging the five homoglyph groups — `0/O`, `1/I/L`, `2/Z`, `5/S`, `9/g/q` — lifts Z+F
by **+5.3 points** (62.5 → 67.8, and 65.3 → 70.6 weighted). "Neither correct" falls from
31.6 % to 27.5 % and the best-of-two selection bound rises to 72.5 %.

**A caution about building the groups.** The obvious implementation — list the
confusable pairs and take the transitive closure — is wrong. Asserting `6≡G`, `G≡g`
(a case pair) and `9≡g` chains `6` to `9`, and the closure collapses `6/9/G/Q/g/q` into
a single six-member class, which is nonsense. The groups above are **disjoint by
construction** and the notebook asserts it.

**7. Merging the case pairs makes things *worse*, despite fewer classes.**
Going 40 → 31 classes drops Z+F from 67.8 % to 66.9 % and η²-weighted from **70.6 % to
67.7 %** — accuracy falls while chance rises. The reason is that EMNIST-balanced has
*already* merged the case pairs that look alike (`C/c`, `O/o`, `S/s`, …); the 11
lowercase classes it keeps are exactly the ones whose shape **differs** from the
uppercase form. Merging them therefore creates **bimodal classes**, and a
nearest-class-mean classifier represents each class by one point. Measured
within-class scatter, merged ÷ mean of the two parts: `D/d` **1.28**, `T/t` 1.22,
`H/h` 1.21, `R/r` 1.19, `N/n` 1.18, `B/b` 1.17, `A/a` 1.14, `E/e` 1.11, `F/f` 1.04 —
every pair is more scattered merged than apart, and `D/d` duly becomes the worst class
in that condition (35 %). So `F`→`f` in the confusion table is a genuine descriptor
failure, not a label artifact; `0`→`O` is the reverse.

**8. Merging vs merely rescoring.** Forgiving homoglyph errors from the strict 47-way
model gives **71.6 %**, while training on the 40 merged labels gives **70.6 %** — the
rescore wins by 1.0 point, because the 47-way model keeps one tight mean per glyph
(30 instances) whereas a merged class pools 60–90 instances into a single mean that
must cover both members. The gap is small precisely because homoglyph members really
do overlap. Either number is a fair thing to quote; the strict 62.5 % on its own is
not, because a quarter of its errors are the task asking for a distinction the ink
does not contain.

**6. Which classes fail** (Z+F, η²-weighted). Worst: `g` 26.7 %, `F` 30.0 %, `L` 33.3 %,
`q` 33.3 %, `2` 40.0 %, `J` 43.3 %, `0` 46.7 %, `8` 46.7 %. Best: `3`, `W`, `M` all
90.0 %, then `T`, `H`, `B` at 83.3 %. The pattern is clean — glyphs with a distinctive
*stroke layout* (`M`, `W`, `H`) are easy for these descriptors; glyphs that collide with
another class (`g`↔`9`/`q`, `F`↔`f`, `0`↔`O`, `2`↔`Z`) are hard for the reason in §5
rather than for any shape-descriptive reason. `L` stays weak for the same reason it was
weak at 12 classes: a bare corner has little global shape *and* little per-cell
orientation structure to distinguish it.
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
# ╠═90000000-0000-0000-0000-000000000009
# ╟─90000000-0000-0000-0000-00000000000a
# ╠═90000000-0000-0000-0000-00000000000b
# ╟─90000000-0000-0000-0000-00000000000c
# ╠═90000000-0000-0000-0000-00000000000d
# ╟─90000000-0000-0000-0000-00000000000e
# ╠═90000000-0000-0000-0000-00000000000f
# ╟─90000000-0000-0000-0000-000000000010
# ╟─90000000-0000-0000-0000-000000000011
# ╟─90000000-0000-0000-0000-000000000012
# ╟─90000000-0000-0000-0000-000000000013
