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

# ╔═╡ a0000000-0000-0000-0000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    using PlutoUI
    using Statistics, LinearAlgebra, Random, FFTW
end

# ╔═╡ a0000000-0000-0000-0000-000000000002
begin
    include(joinpath(@__DIR__, "..", "LoadEMNIST.jl"))   # plain module — safe to include
    using .LoadEMNIST
end

# ╔═╡ a0000000-0000-0000-0000-000000000003
md"""
# Curvature-extrema constellations in FPE — position vs. pure topology

Encodes each EMNIST letter as a set of **Gaussian-curvature extrema across scale**,
lifts that constellation into a single **FPE / HRR vector**, and asks the
diagnosticity question: is a letter's vector closer to *other instances of the same
letter* than to *different letters*?

The point of the notebook is to reproduce and document one specific finding:

> **Pure graph topology of the K-nodes is not diagnostic** — encoding the MST
> connectivity as bound edges scores *below* even a plain typed-keypoint census —
> while **coarse absolute position (which quadrant a node is in)** is the strongest
> single ingredient.

### The encoding

Each extremum is a node with a **K-value bin** (a curvature-type label) and,
optionally, a **quadrant** (relative to the |∇I|-weighted centroid — position *within* the
letter). VSA operations: **bind** `⊙` = circular convolution, **bundle** = sum,
**similarity** = cosine. Node symbol `= C[K-bin]` or `C[K-bin] ⊙ Q[quadrant]`.
A letter's vector is the bundle either of its **nodes** (a *bag*) or of its **MST
edges** (`bind(nodeᵢ, nodⱼ)` — the *topology*). Four combinations, scored by
leave-one-out nearest-neighbour accuracy.

Because `K`, the `|∇I|`-weighted centroid, and hence every quadrant are built only
from squared/abs derivatives, the whole encoding is **exactly polarity-invariant**
(verified: constellations and quadrant assignments are identical under `I → 1−I`).
"""

# ╔═╡ a0000000-0000-0000-0000-00000000000a
begin
    const IMG0=112; const PAD=56; const IMG=IMG0+2PAD   # 224 working field
    const SCALES = Float32[3, 6, 12]                    # 3 scales (the § finding used these)
end

# ╔═╡ a0000000-0000-0000-0000-00000000000b
# Gaussian curvature at scale σ — mirrors GaussianCurvatureEMNIST.jl (copied so this
# notebook stays self-contained; including that *Pluto* notebook would clobber @bind).
begin
    function dog1d(σ)
        r=max(1,ceil(Int,3.5σ)); xs=Float32.(-r:r)
        g=exp.(-xs.^2 ./ (2σ^2)); g ./= sum(g)
        g1=-(xs ./ σ^2).*g;              g1 .-= mean(g1)      # ∂ₓGσ  (DC-free)
        g2=((xs.^2 .- σ^2) ./ σ^4).*g;   g2 .-= mean(g2)      # ∂ₓₓGσ (DC-free)
        g, g1, g2
    end
    function conv_cols(A,k)
        H,W=size(A); r=length(k)÷2; out=similar(A)
        @inbounds for y in 1:H, x in 1:W
            s=0f0; for j in -r:r; s+=A[y,clamp(x+j,1,W)]*k[j+r+1]; end; out[y,x]=s
        end; out
    end
    function conv_rows(A,k)
        H,W=size(A); r=length(k)÷2; out=similar(A)
        @inbounds for y in 1:H, x in 1:W
            s=0f0; for j in -r:r; s+=A[clamp(y+j,1,H),x]*k[j+r+1]; end; out[y,x]=s
        end; out
    end
    function jet(img,σ)
        g,g1,g2=dog1d(σ)
        Cg1=conv_cols(img,g1); Cg=conv_cols(img,g); Cg2=conv_cols(img,g2)
        Ix =conv_rows(Cg1,g);  Ixy=conv_rows(Cg1,g1)
        Iy =conv_rows(Cg ,g1); Iyy=conv_rows(Cg ,g2)
        Ixx=conv_rows(Cg2,g)
        Ix,Iy,Ixx,Iyy,Ixy
    end
    function curvature(img,σ; quantity=:K, snorm=false)
        Ix,Iy,Ixx,Iyy,Ixy=jet(img,σ)
        detH=Ixx.*Iyy .- Ixy.^2
        M = quantity==:detH ? detH : detH ./ (1f0 .+ Ix.^2 .+ Iy.^2).^2
        snorm ? M .* Float32(σ)^4 : M
    end
end

# ╔═╡ a0000000-0000-0000-0000-00000000000c
# EMNIST upsample/embed + per-scale local extrema — also mirrors GaussianCurvatureEMNIST.jl.
begin
    @inline function bilinear(M,y,x)
        H,W2=size(M); (y<1||x<1||y>H||x>W2)&&return 0f0
        y0,x0=floor(Int,y),floor(Int,x); y1,x1=min(y0+1,H),min(x0+1,W2); fy,fx=y-y0,x-x0
        (1-fy)*(1-fx)*M[y0,x0]+fy*(1-fx)*M[y1,x0]+(1-fy)*fx*M[y0,x1]+fy*fx*M[y1,x1]
    end
    function upsample(img)
        H,W2=size(img); out=zeros(Float32,IMG0,IMG0)
        for i in 1:IMG0,j in 1:IMG0
            out[i,j]=bilinear(img,Float32(1+(i-1)*(H-1)/(IMG0-1)),Float32(1+(j-1)*(W2-1)/(IMG0-1)))
        end
        out
    end
    embed(letter)=(out=zeros(Float32,IMG,IMG); out[PAD+1:PAD+IMG0,PAD+1:PAD+IMG0].=letter; out)
    function local_extrema(K; frac=0.30f0, r=8, topN=8, w=2)
        m=maximum(abs,K); m==0f0 && return Tuple{Int,Int,Float32}[]
        thr=frac*m; cand=Tuple{Int,Int,Float32}[]
        for y in (w+1):(IMG-w), x in (w+1):(IMG-w)
            v=K[y,x]; abs(v)<thr && continue
            ismax=true; ismin=true
            for dy in -w:w, dx in -w:w
                (dy==0&&dx==0)&&continue; n=K[y+dy,x+dx]
                n>=v && (ismax=false); n<=v && (ismin=false)
            end
            (ismax||ismin) && push!(cand,(y,x,v))
        end
        sort!(cand, by=p->-abs(p[3])); keep=Tuple{Int,Int,Float32}[]
        for c in cand
            any(hypot(c[1]-k[1],c[2]-k[2])<r for k in keep) && continue
            push!(keep,c); length(keep)>=topN && break
        end
        keep
    end
end

# ╔═╡ a0000000-0000-0000-0000-00000000000d
em = load_emnist(n_images_to_load=8000, n_classes=47);

# ╔═╡ a0000000-0000-0000-0000-000000000004
begin
    # ---- FPE / HRR primitives ----
    bindv(a, b) = real(ifft(fft(a) .* fft(b)))     # binding = circular convolution
    unitv(v)    = (n = norm(v); n > 0 ? v ./ n : v)

    # ---- one image → scale-space extrema (x, y, logσ, K) ----
    function constellation(img)
        pts = NTuple{4,Float32}[]
        for σ in SCALES
            K = curvature(img, σ; quantity=:K, snorm=true)     # σ⁴-normalised K
            for (y, x, v) in local_extrema(K; frac=0.30f0, r=8, topN=8)
                push!(pts, (Float32(x), Float32(y), log(σ), v))
            end
        end
        pts
    end

    # ---- |∇I|-weighted centroid: centre of the stroke *edges*. Polarity-invariant
    # (∇I flips sign under inversion, |∇I| does not), unlike an ink (intensity) centroid.
    function centroid(img)
        Ix, Iy, _, _, _ = jet(img, 3f0)                 # gradient at the finest scale
        sx = 0f0; sy = 0f0; tot = 0f0
        for y in 1:IMG, x in 1:IMG
            w = sqrt(Ix[y, x]^2 + Iy[y, x]^2)           # |∇I|
            sx += w*x; sy += w*y; tot += w
        end
        tot > 0 ? (sx/tot, sy/tot) : (Float32((IMG+1)/2), Float32((IMG+1)/2))
    end

    # ---- MST (Prim) over scale-space coords (x, y, γ·logσ) ----
    function mst_edges(pts, γ)
        n = length(pts); n < 2 && return Tuple{Int,Int}[]
        cd(i) = (pts[i][1], pts[i][2], γ*pts[i][3])
        dist(i, j) = (a = cd(i); b = cd(j); hypot(a[1]-b[1], a[2]-b[2], a[3]-b[3]))
        inM = falses(n); inM[1] = true; E = Tuple{Int,Int}[]
        for _ in 1:n-1
            best = (Inf, 0, 0)
            for i in 1:n, j in 1:n
                (inM[i] && !inM[j]) || continue
                dj = dist(i, j); dj < best[1] && (best = (dj, i, j))
            end
            best[2] == 0 && break
            push!(E, (best[2], best[3])); inM[best[3]] = true
        end
        E
    end

    quadof(x, y, cx, cy) = (x >= cx ? 1 : 0) + (y >= cy ? 2 : 0) + 1
    md"*FPE primitives + constellation / centroid / MST / quadrant defined.*"
end

# ╔═╡ a0000000-0000-0000-0000-000000000005
md"""
### Controls

letters $(@bind chars MultiCheckBox(["T","I","X","O","L","K","A","H","E","F"], default=["T","I","X","O","L"]))
instances / letter $(@bind Ninst Slider(5:1:30, default=15, show_value=true))

VSA dimension $(@bind DIM Slider(256:256:4096, default=1024, show_value=true))
K-value bins $(@bind nbins Slider(2:1:8, default=5, show_value=true))
γ (space↔scale weight) $(@bind gamma Slider(5f0:5f0:60f0, default=20f0, show_value=true))
random seed $(@bind seed Slider(1:1:20, default=1, show_value=true))
"""

# ╔═╡ a0000000-0000-0000-0000-000000000006
# Heavy step: extract every instance's constellation. Depends only on letters /
# instance count, so the cheap encoding sliders below don't retrigger it.
data = let
    d = Tuple{String,Vector{NTuple{4,Float32}},Float32,Float32}[]
    for c in chars, i in 1:Ninst
        col = em.class_images[findfirst(==(c), em.class_names)][i]
        img = embed(upsample(col))
        cx, cy = centroid(img)
        push!(d, (c, constellation(img), cx, cy))
    end
    d
end;

# ╔═╡ a0000000-0000-0000-0000-000000000007
# The 4-way decomposition: {nodes | edges} × {K-bin only | K-bin ⊙ quadrant}.
results = let
    Random.seed!(seed)
    allK = Float32[p[4] for (_, pts, _, _) in data for p in pts]
    qe   = quantile(allK, range(0, 1, length=nbins+1))
    binof(K) = clamp(searchsortedlast(qe, K), 1, nbins)
    C = [unitv(randn(DIM)) for _ in 1:nbins]       # codebook: one vector per K-bin
    Q = [unitv(randn(DIM)) for _ in 1:4]           # codebook: one vector per quadrant

    sym(pts, k, cx, cy, up) = up ?
        bindv(C[binof(pts[k][4])], Q[quadof(pts[k][1], pts[k][2], cx, cy)]) :
        C[binof(pts[k][4])]

    function gvec(pts, cx, cy, useedges, usepos)
        g = zeros(DIM)
        if useedges
            for (i, j) in mst_edges(pts, gamma)
                g .+= bindv(sym(pts, i, cx, cy, usepos), sym(pts, j, cx, cy, usepos))
            end
        else
            for k in eachindex(pts)
                g .+= sym(pts, k, cx, cy, usepos)
            end
        end
        unitv(g)
    end

    function score(G, lab)                          # (within, across, gap, LOO %)
        sw = Float64[]; sa = Float64[]
        for a in 1:length(G), b in a+1:length(G)
            s = dot(G[a], G[b]); lab[a] == lab[b] ? push!(sw, s) : push!(sa, s)
        end
        cor = 0
        for a in eachindex(G)
            best = -Inf; bi = 0
            for b in eachindex(G); a == b && continue
                s = dot(G[a], G[b]); s > best && (best = s; bi = b)
            end
            lab[bi] == lab[a] && (cor += 1)
        end
        (mean(sw), mean(sa), mean(sw)-mean(sa), 100cor/length(G))
    end

    lab = [c for (c, _, _, _) in data]
    variants = (("nodes: K-bin only (census)",       false, false),
                ("nodes: K-bin ⊙ quadrant",           false, true),
                ("edges: K-bin only (pure topology)", true,  false),
                ("edges: K-bin ⊙ quadrant",           true,  true))
    [(nm, score([gvec(pts, cx, cy, ue, up) for (_, pts, cx, cy) in data], lab)...)
     for (nm, ue, up) in variants]
end;

# ╔═╡ a0000000-0000-0000-0000-000000000008
let
    chance = round(100 / max(length(chars),1), digits=1)
    hdr = "**$(length(chars)) letters** ($(join(chars, ", "))), $(Ninst) instances each " *
          "— chance = $(chance)% — DIM=$(DIM), bins=$(nbins), γ=$(gamma), seed=$(seed)\n\n" *
          "| encoding | within | across | gap | **LOO** |\n|:--|--:|--:|--:|--:|\n"
    body = join(["| $(r[1]) | $(round(r[2],digits=3)) | $(round(r[3],digits=3)) | " *
                 "$(round(r[4],digits=3)) | **$(round(r[5],digits=1))%** |" for r in results], "\n")
    Markdown.parse(hdr * body)
end

# ╔═╡ a0000000-0000-0000-0000-000000000009
md"""
### What this shows

At the defaults (T, I, X, O, L · 15 each · DIM 1024 · 5 bins · γ 20 · seed 1) the table
reproduces the documented result:

- **Pure topology is not diagnostic.** `edges: K-bin only` (the bag of bound MST
  edges — connectivity, no position) scores **below** the plain `nodes: K-bin only`
  *census*. Binding two nodes into an edge collapses same-letter similarity (watch the
  `within` column drop): an edge only matches another edge if *both* endpoints agree
  in bin, so it is far too brittle, and the unstable MST adds noise. Adjacency of
  curvature extrema, on its own, does **not** separate letters.
- **Coarse position is the strongest ingredient.** `nodes: K-bin ⊙ quadrant` — a bag
  of *positioned* typed keypoints, no graph — is the best row. Just knowing *which
  quadrant* (relative to the |∇I|-weighted centroid) each extremum sits in lifts the census by
  ~8 points. This is the handoff's original "bind each keypoint to its
  object-relative position", now confirmed to beat the topology.
- **Read the LOO column, not the gap.** The census has the *largest* raw within−across
  gap yet the *weakest* accuracy — when everything sits at cosine ≈ 0.8, the gap is not
  the separation that matters. Nearest-neighbour accuracy is the honest metric.

### Caveats

- Small and directional, not a final number: a handful of letters, `Ninst` instances,
  and the codebooks depend on `seed` — sweep the seed slider to see the spread.
- `quadrant` is the crudest possible position cut; a finer grid or an FPE `(r, α)`
  with a wide (approximate-position) kernel is the natural next lever.
- The MST/edge result condemns *pairwise edge-binding*, not connectivity in general —
  a WL-style neighbourhood aggregation (bundle-of-neighbours, not bound pairs) would
  keep within-similarity robust, and is the only way topology might yet earn its keep.

Recorded in `PROGRESS_2026-07-21.md` §7.
"""

# ╔═╡ Cell order:
# ╠═a0000000-0000-0000-0000-000000000001
# ╠═a0000000-0000-0000-0000-000000000002
# ╟─a0000000-0000-0000-0000-000000000003
# ╠═a0000000-0000-0000-0000-00000000000a
# ╠═a0000000-0000-0000-0000-00000000000b
# ╠═a0000000-0000-0000-0000-00000000000c
# ╠═a0000000-0000-0000-0000-00000000000d
# ╠═a0000000-0000-0000-0000-000000000004
# ╟─a0000000-0000-0000-0000-000000000005
# ╠═a0000000-0000-0000-0000-000000000006
# ╠═a0000000-0000-0000-0000-000000000007
# ╠═a0000000-0000-0000-0000-000000000008
# ╟─a0000000-0000-0000-0000-000000000009
