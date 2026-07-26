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
    include(joinpath(@__DIR__, "..", "LoadEMNIST.jl"))
    using .LoadEMNIST
end

# ╔═╡ 90000000-0000-0000-0000-000000000004
md"""
# Global Zernike **+** Fourier 3×3 — do they combine?

Two descriptors in this project score almost the same and measure different things:

- **Block Z** — global Zernike moments on the ink-fitted disc
  (`ZernikeCharacterMoments.jl`): `|A_nm|` + `Re/Im A_nm`, `n ≤ 8` ⇒ **75 numbers**.
  Rotation-invariant magnitudes, *centred* on the object, no spatial layout.
- **Block F** — the Fourier tic-tac-toe grid
  (`../ExptsWithGlobalFourier/TicTacToeFourierSignature.jl`): 9 features in each of
  9 cells ⇒ **81 numbers**. Translation-invariant *within* each cell, all spatial
  layout, no global shape.

`TicTacToeZernike.jl` §6 argued they should be complementary — one is
rotation-invariant about a centre, the other translation-invariant per cell. This
notebook tests that by simply **concatenating** them.

**Result: 76.4 % and 74.2 % separately → 82.8 % concatenated (84.2 % η²-weighted).**
The gain survives a shuffled-block control, and the error-overlap analysis shows why:
the two fail on *different letters*.
"""

# ╔═╡ 90000000-0000-0000-0000-000000000005
begin
    const IMG = 112
    const CLASSES = ["O","C","I","L","T","X","K","A","H","Y","E","F"]
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
    em = load_emnist(n_images_to_load=20000, n_classes=47)
end;

# ╔═╡ 90000000-0000-0000-0000-000000000007
# ---- BLOCK F: the Fourier tic-tac-toe cell features (lattice or polar sampling) ----
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
        Float64[a0, sqrt(pw), real(s2/pw), imag(s2/pw), abs(s2/pw), abs(s4/pw),
                (rings./pw)...]
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
    "Block F: N² cells × 9 features."
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
    const FNAMES=["a0","ac","ReE2","ImE2","|E2|","|E4|","e1","e2","e3"]
    "Pick feature indices `fs` (1..9) out of every cell."
    fsub(v, fs) = vcat([v[f:9:end] for f in fs]...)
end

# ╔═╡ 90000000-0000-0000-0000-000000000008
# ---- BLOCK Z: global Zernike moments on the ink-fitted disc ----
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
    "Centre the ink and scale its `q`-quantile radius to the inscribed radius."
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
# ---- evaluation: η² and leave-one-out nearest-class-mean ----
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
    "Per-image correct/incorrect under LOO nearest-class-mean on standardised columns."
    function loo_hits(X,y; weights=nothing)
        n,p=size(X); Z=copy(X)
        for j in 1:p; s=std(@view Z[:,j]); s<=0 ? (Z[:,j].=0) : (Z[:,j].=(Z[:,j].-mean(Z[:,j]))./s); end
        weights !== nothing && (Z .*= sqrt.(max.(weights,0))')
        classes=unique(y); hits=falses(n)
        for i in 1:n
            best=Inf; bestc=classes[1]
            for c in classes
                idx=[k for k in 1:n if y[k]==c && k!=i]
                m=vec(mean(Z[idx,:],dims=1)); dd=sum(abs2,Z[i,:].-m); dd<best && (best=dd;bestc=c)
            end
            hits[i]=bestc==y[i]
        end
        hits
    end
    loo(X,y; kw...) = mean(loo_hits(X,y; kw...))
end

# ╔═╡ 90000000-0000-0000-0000-00000000000a
md"""
### Controls

instances per class: $(@bind n_per Slider(5:5:30, default=30, show_value=true))
**Zernike n_max**: $(@bind znmax Slider(4:1:10, default=8, show_value=true))
disc fit: $(@bind qfit Select([0.98=>"ink, 98th pct radius", 1.0=>"ink, max radius", -1.0=>"inscribed in the frame"]))

**Fourier grid N×N**: $(@bind gridN Slider(2:1:4, default=3, show_value=true))
sampling: $(@bind sampling Select(["lattice"=>"lattice (v,u)", "polar"=>"polar ω×θ"]))
Fourier subset: $(@bind fsubset Select(["all"=>"all 9 per cell", "io"=>"ink + orientation (3 per cell)"]))
Zernike subset: $(@bind zsubset Select(["all"=>"|A| + Re/Im", "ri"=>"Re/Im only", "mag"=>"|A| only"]))

*(the full run is ~360 images × two descriptors — allow 30–90 s after any change)*
"""

# ╔═╡ 90000000-0000-0000-0000-00000000000b
begin
    imgs = Matrix{Float32}[]; labs = Int[]
    for (ci,c) in enumerate(CLASSES)
        v = em.class_images[findfirst(==(c), em.class_names)]
        for k in 1:min(n_per,length(v)); push!(imgs, upsample(v[k])); push!(labs, ci); end
    end

    BZ, OZ = zbasis(IMG, znmax); MZ = zmask(IMG)
    qq = qfit < 0 ? nothing : qfit
    zraw = [ zmoments(qq === nothing ? im : fit_disc(im; q=qq), BZ, MZ; zerodc=true) for im in imgs ]
    zvec(A) = zsubset=="all" ? vcat(abs.(A),real.(A),imag.(A)) :
              zsubset=="ri"  ? vcat(real.(A),imag.(A)) : abs.(A)
    Zblk = reduce(vcat, [zvec(A)' for A in zraw])

    fraw = [ fourier_grid(im; N=gridN, sampling=sampling) for im in imgs ]
    fvec(v) = fsubset=="all" ? v : fsub(v,[1,3,4])
    Fblk = reduce(vcat, [fvec(v)' for v in fraw])

    Cblk = hcat(Zblk, Fblk)
    eZ, eF, eC = eta2(Zblk,labs), eta2(Fblk,labs), eta2(Cblk,labs)
    hZ, hF = loo_hits(Zblk,labs), loo_hits(Fblk,labs)
    hC     = loo_hits(Cblk,labs)
    hCw    = loo_hits(Cblk,labs; weights=eC)
    chance = 100/length(CLASSES)

    Markdown.parse("**$(length(imgs)) images, $(length(CLASSES)) classes** — chance $(round(chance,digits=1)) % — " *
      "block **Z** = $(size(Zblk,2)) numbers (Zernike n≤$(znmax), $(zsubset)), " *
      "block **F** = $(size(Fblk,2)) numbers ($(gridN)×$(gridN) $(sampling), $(fsubset))")
end

# ╔═╡ 90000000-0000-0000-0000-00000000000c
let
    rows = [("**Z** — global Zernike alone",            size(Zblk,2), mean(hZ),  loo(Zblk,labs;weights=eZ)),
            ("**F** — Fourier $(gridN)×$(gridN) alone", size(Fblk,2), mean(hF),  loo(Fblk,labs;weights=eF)),
            ("**Z + F** — concatenated",                size(Cblk,2), mean(hC),  mean(hCw))]
    hdr = "| descriptor | numbers | LOO | η²-weighted |\n|:--|--:|--:|--:|\n"
    body = join([@sprintf("| %s | %d | **%.1f %%** | **%.1f %%** |", t, n, 100a, 100b)
                 for (t,n,a,b) in rows], "\n")
    g1 = 100*(mean(hC) - max(mean(hZ), mean(hF)))
    g2 = 100*(mean(hCw) - max(loo(Zblk,labs;weights=eZ), loo(Fblk,labs;weights=eF)))
    gain = @sprintf("\n\nConcatenation gains **%+.1f points** over the better single block (**%+.1f** η²-weighted).", g1, g2)
    Markdown.parse(hdr * body * gain)
end

# ╔═╡ 90000000-0000-0000-0000-00000000000d
md"""
### Control: is the gain just "more columns"?

Concatenating **shuffled** block-F rows keeps the column count and the marginal
distributions but destroys the per-image correspondence. If the gain were a
dimensionality artifact, this would score the same.
"""

# ╔═╡ 90000000-0000-0000-0000-00000000000e
let
    Random.seed!(7)
    sh = [ mean(loo_hits(hcat(Zblk, Fblk[randperm(size(Fblk,1)),:]), labs)) for _ in 1:3 ]
    hdr = "| variant | LOO |\n|:--|--:|\n"
    rows = [@sprintf("| Z alone | %.1f %% |", 100mean(hZ)),
            @sprintf("| **Z + F (real)** | **%.1f %%** |", 100mean(hC))]
    append!(rows, [@sprintf("| Z + F shuffled, trial %d | %.1f %% |", t, 100s) for (t,s) in enumerate(sh)])
    verdict = mean(sh) < mean(hZ) ?
        "\n\nShuffling drops it **below block Z alone** — the extra columns are pure noise " *
        "without the correspondence, so the real gain is genuine complementarity, not dimensionality." :
        "\n\n⚠️ Shuffling did *not* hurt — treat the gain as a dimensionality artifact."
    Markdown.parse(hdr * join(rows,"\n") * verdict)
end

# ╔═╡ 90000000-0000-0000-0000-00000000000f
md"""
### Why it works: the two descriptors fail on different images

Every image is one of four cases. **"exactly one correct"** is the headroom a
combination can exploit; the oracle bar is what a perfect chooser would reach.
"""

# ╔═╡ 90000000-0000-0000-0000-000000000010
let
    both = mean(hZ .& hF); neither = mean(.!hZ .& .!hF)
    onlyZ = mean(hZ .& .!hF); onlyF = mean(.!hZ .& hF)
    p1 = bar(["both\ncorrect","only Z","only F","neither"], 100 .*[both,onlyZ,onlyF,neither];
             c=[:seagreen,:steelblue,:goldenrod,:firebrick], label="", ylabel="% of images",
             title="error overlap", titlefontsize=9, guidefontsize=8, tickfontsize=7, grid=false)
    lv = ["Z alone","F alone","Z+F","Z+F (η²)","oracle"]
    vals = 100 .*[mean(hZ), mean(hF), mean(hC), mean(hCw), mean(hZ .| hF)]
    p2 = bar(lv, vals; c=[:steelblue,:goldenrod,:seagreen,:darkgreen,:grey], label="",
             ylabel="LOO accuracy (%)", title="what the combination recovers",
             titlefontsize=9, guidefontsize=8, tickfontsize=7, xrotation=20, grid=false,
             ylims=(0,100))
    for (i,v) in enumerate(vals); annotate!(p2, i, v+3, text(@sprintf("%.1f",v), 7)); end
    plot(p1,p2; layout=(1,2), size=(950,360))
end

# ╔═╡ 90000000-0000-0000-0000-000000000011
let
    onlyF = .!hZ .& hF; onlyZ = hZ .& .!hF
    rF = sum(hC .& onlyF); rZ = sum(hC .& onlyZ)
    best = max(mean(hZ), mean(hF)); oracle = mean(hZ .| hF)
    head = 100*(mean(hC) - best)/max(oracle - best, 1e-9)
    a = @sprintf("Of the **%d** images block Z gets wrong but block F gets right, the concatenation recovers **%d (%.0f %%)**; ", sum(onlyF), rF, 100rF/max(sum(onlyF),1))
    b = @sprintf("of the **%d** the other way round it recovers **%d (%.0f %%)**. ", sum(onlyZ), rZ, 100rZ/max(sum(onlyZ),1))
    c = @sprintf("Oracle ceiling **%.1f %%**, concatenation reaches **%.1f %%** — it captures **%.0f %%** of the available headroom.", 100oracle, 100mean(hC), head)
    Markdown.parse(a * b * c)
end

# ╔═╡ 90000000-0000-0000-0000-000000000012
let
    zs = [100mean(hZ[findall(==(ci),labs)]) for ci in 1:length(CLASSES)]
    fs = [100mean(hF[findall(==(ci),labs)]) for ci in 1:length(CLASSES)]
    cs = [100mean(hCw[findall(==(ci),labs)]) for ci in 1:length(CLASSES)]
    xs = collect(1:length(CLASSES)); w = 0.26
    p = bar(xs .- w, zs; bar_width=w, label="Z (Zernike)", c=:steelblue,
            xticks=(xs, CLASSES), ylabel="LOO accuracy (%)", ylims=(0,105),
            title="per class", titlefontsize=9, guidefontsize=8, tickfontsize=8,
            legendfontsize=7, legend=:bottomright, grid=false)
    bar!(p, xs,      fs; bar_width=w, label="F (Fourier 3×3)", c=:goldenrod)
    bar!(p, xs .+ w, cs; bar_width=w, label="Z + F (η²)", c=:seagreen)
    plot(p; size=(950,340))
end

# ╔═╡ 90000000-0000-0000-0000-000000000013
md"""
### Notes — what was measured

Defaults: 360 EMNIST instances, 12 classes, LOO nearest-class-mean, chance 8.3 %.
Block Z = global Zernike `n ≤ 8` on the 98th-percentile ink-fitted disc, `|A|` + `Re/Im`
(75 numbers). Block F = 3×3 grid, lattice sampling, all 9 features per cell (81).

**1. Concatenation works, and the gain is large.**

| descriptor | numbers | LOO | η²-weighted |
|:--|--:|--:|--:|
| Z — global Zernike | 75 | 76.4 % | 77.5 % |
| F — Fourier 3×3 | 81 | 74.2 % | 77.2 % |
| **Z + F** | **156** | **82.8 %** | **84.2 %** |

**+6.4 points** over the better block alone. With polar sampling for F the pair gives
82.5 % / **84.4 %**, so the choice of sampling barely matters. For reference, the
previous best anywhere in this project was ≈ 61 % (global shape harmonics,
`KeyPointDiagnosticity.md`), and each of these blocks alone is ≈ 76 %.

**2. It is not a dimensionality artifact.** Shuffling block F's rows — same columns,
same marginals, correspondence destroyed — gives **69.2 / 71.4 / 71.7 %** across three
seeds, i.e. *below* block Z alone (76.4 %). Extra uninformative columns actively hurt
the equal-weighted nearest-mean classifier, exactly as
`KeyPointDiagnosticity.md` documents. So the real +6.4 is complementarity.

**3. The two descriptors fail on different images** — this is the whole mechanism:

| outcome | share of images |
|:--|--:|
| both correct | 65.3 % |
| only Zernike correct | 11.1 % |
| only Fourier correct | 8.9 % |
| neither correct | 14.7 % |

**20.0 %** of images are got right by exactly one of the two. An oracle that always
picked the right block would score **85.3 %**; the concatenation reaches **82.8 %**,
i.e. it captures **72 %** of the 8.9-point gap between the better block and that
ceiling. Broken down by direction: of the 32 images Zernike misses and Fourier catches
it recovers **29 (91 %)**, and of the 40 the other way round **28 (70 %)**. So a
smarter fusion rule has at most ~2.5 points left to win, and the real ceiling is set by
the **14.7 %** that *both* descriptors miss.

**4. Per class, the two are visibly complementary.** `E` goes 66.7 % (Z) / 76.7 % (F)
→ **93.3 %** combined; `K` 76.7 / 63.3 → **86.7 %**; `X` 73.3 / 66.7 → **80.0 %**.
`O` is the one class where Fourier alone (96.7 %) beats the combination (90.0 %) —
adding 75 Zernike columns dilutes a class the grid already separates perfectly.
`L` stays the weak class (36.7 / 50.0 → 56.7 %): a bare corner has little global shape
*and* little per-cell orientation structure to distinguish it from `I` or `T`.

**5. Weighting matters less than expected.** η²-weighting adds ~1.4 points
(82.8 → 84.2 %). Per-*block* rebalancing (equal weight per block rather than per
column) is worth nothing when the blocks are similar sizes (82.8 → 82.5 %) but does
help when they are lopsided — with the 27-number `ink+orientation` F block,
80.6 → 82.5 %. So if you use a small F subset, rebalance; otherwise don't bother.

**6. Why this was predictable.** `TicTacToeZernike.jl` §5–6 measured that Zernike's
free invariance is **rotation about a centre** (and it is *not* translation-invariant —
an off-centre blob reports its position as an orientation), while the Fourier cell
descriptor is built from `|F|²` and so is **translation-invariant within its cell**
(and not rotation-invariant). Each is blind exactly where the other sees. The 43.3°
median disagreement between their orientation estimates on real cells was the early
symptom; this notebook is the payoff.
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
# ╟─90000000-0000-0000-0000-00000000000d
# ╟─90000000-0000-0000-0000-00000000000e
# ╟─90000000-0000-0000-0000-00000000000f
# ╠═90000000-0000-0000-0000-000000000010
# ╟─90000000-0000-0000-0000-000000000011
# ╠═90000000-0000-0000-0000-000000000012
# ╟─90000000-0000-0000-0000-000000000013
