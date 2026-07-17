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

# ╔═╡ d0000000-0000-0000-0000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ d0000000-0000-0000-0000-000000000002
begin
    using PlutoUI
    using Plots
    using FFTW
    using Statistics
end

# ╔═╡ d0000000-0000-0000-0000-000000000003
begin
    include(joinpath(@__DIR__, "..", "LoadEMNIST.jl"))
    using .LoadEMNIST
end

# ╔═╡ d0000000-0000-0000-0000-000000000004
md"""
# Keypoint & shape-descriptor diagnosticity

Interactive companion to `KeyPointDiagnosticity.md`. Pick a **local encoding**
and see how diagnostic the resulting descriptor is of letter identity — measured
two ways:

- **η² per feature** — fraction of that feature's variance explained by letter
  identity (0 = useless, 1 = perfect);
- **leave-one-out accuracy** — nearest-class-mean on the standardized vector,
  split into *local features only*, *shape harmonics only*, and *both*
  (chance = 1/#classes).

The expensive ray-harmonic pipeline is computed **once** and cached; switching
the encoding only re-runs the cheap detector + counting, so the controls are
responsive.

> The shape harmonics carry most of the identity; the local *counts* add little
> because a census discards keypoint *position*. See the markdown for the full
> story and the synthetic-figure validation.
"""

# ╔═╡ d0000000-0000-0000-0000-000000000005
# ---- constants (same scale as EMNIST_Junction_Keypoints.jl) ----
begin
    const IMG=112; const N_THETA=48; const K_PHI=96; const LAM=12f0
    const SIGMA_N=6f0; const SIGMA_T=10f0; const KSIZE=39; const D_RAY=15f0
    const THETAS=Float32.(range(0,π,length=N_THETA+1)[1:N_THETA])
    const PHIS=Float32.(range(0,2π,length=K_PHI+1)[1:K_PHI])
    const CLASSES=["O","C","I","L","T","X","K","A","H","Y","E","F"]
end

# ╔═╡ d0000000-0000-0000-0000-000000000006
# ---- Gabor bank + precomputed kernel spectra (padded FFT convolution) ----
begin
    function gabor_kernel(θ)
        h=KSIZE÷2; K=zeros(ComplexF32,KSIZE,KSIZE); c,s=cos(θ),sin(θ)
        for i in -h:h, j in -h:h
            y,x=Float32(i),Float32(j); xt=x*c+y*s; xn=-x*s+y*c
            K[i+h+1,j+h+1]=exp(-(xt^2/(2SIGMA_T^2)+xn^2/(2SIGMA_N^2)))*cis(2f0π*xn/LAM)
        end
        K .- mean(K)
    end
    const PH,PW = IMG+KSIZE-1, IMG+KSIZE-1
    const O1,O2 = KSIZE÷2, KSIZE÷2
    const BHAT = [begin B=zeros(ComplexF32,PH,PW); B[1:KSIZE,1:KSIZE].=gabor_kernel(θ); fft(B) end
                  for θ in THETAS]
    function energy_stack(img)
        A=zeros(ComplexF32,PH,PW); A[1:IMG,1:IMG].=img; Ah=fft(A)
        E=Array{Float32,3}(undef,N_THETA,IMG,IMG)
        for t in 1:N_THETA
            f=ifft(Ah.*BHAT[t]); E[t,:,:].=abs.(@view f[O1+1:O1+IMG,O2+1:O2+IMG])
        end
        E
    end
end

# ╔═╡ d0000000-0000-0000-0000-000000000007
# ---- fast ray harmonics (§7.2 whole-array shift form) + helpers ----
begin
    @inline function bilinear(M,y,x)
        H,W=size(M); (y<1||x<1||y>H||x>W)&&return 0f0
        y0,x0=floor(Int,y),floor(Int,x); y1,x1=min(y0+1,H),min(x0+1,W); fy,fx=y-y0,x-x0
        (1-fy)*(1-fx)*M[y0,x0]+fy*(1-fx)*M[y1,x0]+(1-fy)*fx*M[y0,x1]+fy*fx*M[y1,x1]
    end
    tidx(φ)=mod(round(Int,(mod(φ,π)/π)*N_THETA),N_THETA)+1
    function ishift(M,sy,sx)
        H,W=size(M); out=zeros(Float32,H,W)
        ys=max(1,1-sy):min(H,H-sy); xs=max(1,1-sx):min(W,W-sx)
        @views out[ys,xs].=M[ys.+sy,xs.+sx]; out
    end
    # ray harmonics at probe radius `d` (the "scale" of the ray profile).
    # Larger d reaches past the stroke — matters most for ENDPOINTS, whose one
    # uncancelled ray only reads cleanly when the probe clears the stroke width.
    function ray_harmonics(E; d=D_RAY)
        C=zeros(ComplexF32,3,IMG,IMG)
        for φ in PHIS
            Eθ=@view E[tidx(φ),:,:]; dy,dx=d*sin(φ),d*cos(φ)
            sy0,sx0=floor(Int,dy),floor(Int,dx); fy,fx=dy-sy0,dx-sx0
            sh=(1-fy)*(1-fx).*ishift(Eθ,sy0,sx0) .+ fy*(1-fx).*ishift(Eθ,sy0+1,sx0) .+
               (1-fy)*fx.*ishift(Eθ,sy0,sx0+1) .+ fy*fx.*ishift(Eθ,sy0+1,sx0+1)
            for n in 0:2; @views C[n+1,:,:].+=sh.*cis(-Float32(n)*φ); end
        end
        C./K_PHI
    end
    function upsample(img)
        H,W=size(img); out=zeros(Float32,IMG,IMG)
        for i in 1:IMG,j in 1:IMG
            out[i,j]=bilinear(img,Float32(1+(i-1)*(H-1)/(IMG-1)),Float32(1+(j-1)*(W-1)/(IMG-1)))
        end
        out
    end
    # object-centred polar frame from image moments: rn = r/r_rms, α about centroid.
    function object_frame(img)
        H,W=size(img); tot=sum(img)
        cy=sum(i*img[i,j] for i in 1:H,j in 1:W)/tot; cx=sum(j*img[i,j] for i in 1:H,j in 1:W)/tot
        r=[sqrt((i-cy)^2+(j-cx)^2) for i in 1:H,j in 1:W]; α=[atan(i-cy,j-cx) for i in 1:H,j in 1:W]
        rms=sqrt(sum(img.*r.^2)/tot); (Float32.(r./rms), Float32.(α), Float32.(img./tot))
    end
    # angular shape harmonics |Mₙ|/M₀ (rotation-invariant): elongation, n-foldness…
    function angular_spectrum(img; nmax=6)
        _,α,w=object_frame(img); M=[sum(w.*cis.(-Float32(n).*α)) for n in 0:nmax]
        [abs(M[n+1])/abs(M[1]) for n in 1:nmax]
    end
    # radial mass profile (filled vs hollow): mass in each normalised-radius bin.
    function radial_profile(img; nbins=4, rmax=2.5f0)
        rn,_,w=object_frame(img); edges=range(0f0,rmax,length=nbins+1)
        [sum(w[(rn.>=edges[i]).&(rn.<edges[i+1])]) for i in 1:nbins]
    end
end

# ╔═╡ d0000000-0000-0000-0000-000000000008
# ---- the two detectors ----
begin
    "greedy top-n of c₀ with non-max suppression radius rad (tiles the ridges)."
    function kp_greedy(c0; n=12, rad=12)
        A=copy(c0); H,W=size(A); pts=Tuple{Int,Int,Float32}[]
        for _ in 1:n
            v,k=findmax(A); v ≤ 0.1f0*(isempty(pts) ? v : pts[1][3]) && break
            y,x=Tuple(k); push!(pts,(y,x,v)); A[max(1,y-rad):min(H,y+rad),max(1,x-rad):min(W,x+rad)].=0f0
        end
        pts
    end
    "clear local maxima: strict 8-neighbour maxima of c₀ above frac·max, top-n."
    function kp_localmax(c0; n=25, frac=0.25f0)
        H,W=size(c0); mx=maximum(c0); pts=Tuple{Int,Int,Float32}[]
        for y in 2:H-1, x in 2:W-1
            v=c0[y,x]; v<=frac*mx && continue
            ismax=true
            for dy in -1:1, dx in -1:1
                (dy==0&&dx==0)&&continue
                c0[y+dy,x+dx]>v && (ismax=false;break)
            end
            ismax && push!(pts,(y,x,v))
        end
        sort!(pts,by=p->-p[3]); length(pts)>n ? pts[1:n] : pts
    end
end

# ╔═╡ d0000000-0000-0000-0000-000000000009
# ---- typing helpers (return 1=endpoint 2=corner 3=T 4=X 0=straight/none) ----
begin
    # 2-D: nearest canonical signature in (|c1|/c0, |c2|/c0)
    const PROTO2=[(1,1.0f0,1.0f0),(2,0.707f0,0.0f0),(3,0.333f0,0.333f0),(4,0.0f0,0.0f0),(0,0.0f0,1.0f0)]
    function kptype2d(r1,r2)
        best=0; bd=Inf32
        for (t,p1,p2) in PROTO2
            d=(r1-p1)^2+(r2-p2)^2; d<bd && (bd=d;best=t)
        end
        best
    end
    # 3-D: nearest in (n_rays, |c1|/c0, |c2|/c0)
    const PROTO3=[(1,1f0,1.0f0,1.0f0),(2,2f0,0.707f0,0.0f0),(0,2f0,0.0f0,1.0f0),
                  (3,3f0,0.333f0,0.333f0),(4,4f0,0.0f0,0.0f0)]
    function kptype3d(nr,r1,r2)
        best=0; bd=Inf32
        for (t,pn,p1,p2) in PROTO3
            d=(nr-pn)^2+(r1-p1)^2+(r2-p2)^2; d<bd && (bd=d;best=t)
        end
        best
    end
end

# ╔═╡ d0000000-0000-0000-0000-00000000000a
# ---- local feature encoders: (method, maps) -> (names, vector) ----
function local_features(method, c0, c1, c2, img)
    mx=maximum(c0); r1=c1./(c0.+1f-6); r2=c2./(c0.+1f-6)
    onstroke=img.>0.5f0; ref2=any(onstroke) ? median(c0[onstroke]) : mx
    ismax8(A,y,x)=all(A[y+dy,x+dx]<=A[y,x] for dy in -1:1, dx in -1:1 if !(dy==0&&dx==0))
    if method=="greedy + mean-pool" || method=="localmax + mean-pool"
        kp = startswith(method,"greedy") ? kp_greedy(c0) : kp_localmax(c0)
        reach=c0.>0.1f0*mx
        flat=count(i->reach[i]&&c0[i]>0.8f0*mx,CartesianIndices(c0))/max(count(reach),1)
        if isempty(kp); m1=0f0;m2=0f0
        else m1=mean(r1[y,x] for (y,x,_) in kp); m2=mean(r2[y,x] for (y,x,_) in kp) end
        (["n_kp","flatness","mean|c1|/c0","mean|c2|/c0"], Float32[length(kp),flat,m1,m2])
    elseif method=="localmax + typed counts (2-D)"
        kp=kp_localmax(c0); ne=nc=nt=nx=0
        for (y,x,_) in kp
            t=kptype2d(r1[y,x],r2[y,x]); t==1&&(ne+=1);t==2&&(nc+=1);t==3&&(nt+=1);t==4&&(nx+=1)
        end
        (["n_end","n_cor","n_T","n_X"], Float32[ne,nc,nt,nx])
    elseif method=="localmax + typed counts (ray-count)"
        kp=kp_localmax(c0); ne=nc=nt=nx=0
        for (y,x,_) in kp
            nr=2f0*c0[y,x]/(ref2+1f-6); t=kptype3d(nr,r1[y,x],r2[y,x])
            t==1&&(ne+=1);t==2&&(nc+=1);t==3&&(nt+=1);t==4&&(nx+=1)
        end
        (["n_end","n_cor","n_T","n_X"], Float32[ne,nc,nt,nx])
    else # "two-channel typed counts"
        H,W=size(c0); ne=nt=nx=0
        for y in 2:H-1, x in 2:W-1
            v=c0[y,x]
            if v>0.30f0*mx && ismax8(c0,y,x)
                nr=2f0*v/(ref2+1f-6); nr>=2.5f0 && (nr>=3.5f0 ? (nx+=1) : (nt+=1))
            end
            (v>0.15f0*mx && r1[y,x]>0.55f0 && ismax8(r1,y,x)) && (ne+=1)
        end
        (["n_end","n_T","n_X"], Float32[ne,nt,nx])
    end
end

# ╔═╡ d0000000-0000-0000-0000-00000000000b
md"""
## Controls

**instances per class** $(@bind n_per Slider(8:2:30, default=20, show_value=true))

**ray-probe scale D_RAY** $(@bind d_ray Slider(8:1:22, default=15, show_value=true))
— larger reaches past the stroke; watch the endpoint channel sharpen (and, on real
letters, the risk of cross-talk between nearby strokes rise). *Recomputes the cache.*
"""

# ╔═╡ d0000000-0000-0000-0000-00000000000c
emnist = load_emnist(n_images_to_load=8000, n_classes=47)

# ╔═╡ d0000000-0000-0000-0000-00000000000d
# EXPENSIVE, cached: ray-harmonic maps (at scale d_ray) + shape descriptors per
# instance. Depends on n_per and d_ray — NOT on the encoding/shape choice — so
# switching those below re-derives features cheaply without recomputing this.
cache = let
    items = NamedTuple[]
    for c in CLASSES
        ci=findfirst(==(c),emnist.class_names); imgs=emnist.class_images[ci]
        for k in 1:min(n_per,length(imgs))
            img=upsample(Float32.(imgs[k])); C=ray_harmonics(energy_stack(img); d=Float32(d_ray))
            push!(items,(label=c, c0=abs.(C[1,:,:]), c1=abs.(C[2,:,:]), c2=abs.(C[3,:,:]),
                         img=img, ang=angular_spectrum(img; nmax=6), rad=radial_profile(img; nbins=4)))
        end
    end
    items
end

# ╔═╡ d0000000-0000-0000-0000-00000000000e
md"""
**local encoding** $(@bind method Select([
    "greedy + mean-pool",
    "localmax + mean-pool",
    "localmax + typed counts (2-D)",
    "localmax + typed counts (ray-count)",
    "two-channel typed counts"], default="two-channel typed counts"))

**shape descriptor** $(@bind shape_opt Select([
    "none",
    "angular |M1..4|",
    "angular |M1..6|",
    "angular |M1..6| + radial"], default="angular |M1..6| + radial"))
"""

# ╔═╡ d0000000-0000-0000-0000-00000000000f
# shape features sliced from the cache per the dropdown (angular |Mₙ| are stored
# up to n=6, radial as 4 bins). Measured: |M1..4| 57%, |M1..6| 59%, +radial 59.4%.
shape_feats(it) =
    shape_opt=="none"                    ? (String[], Float32[]) :
    shape_opt=="angular |M1..4|"         ? (["|M$n|/M0" for n in 1:4], it.ang[1:4]) :
    shape_opt=="angular |M1..6|"         ? (["|M$n|/M0" for n in 1:6], it.ang[1:6]) :
    (vcat(["|M$n|/M0" for n in 1:6], ["rad$i" for i in 1:4]), vcat(it.ang[1:6], it.rad))

# ╔═╡ d0000000-0000-0000-0000-0000000000f2
# CHEAP: apply the selected local encoding + shape descriptor -> feature matrix.
features = let
    rows=Vector{Float32}[]; labels=String[]; names=String[]; nlocal=0
    for it in cache
        lname, lvec = local_features(method, it.c0, it.c1, it.c2, it.img)
        sname, svec = shape_feats(it)
        push!(rows, vcat(lvec, svec)); push!(labels, it.label)
        names = vcat(lname, sname); nlocal = length(lvec)
    end
    (D=reduce(vcat, (r' for r in rows)), y=labels, names=names, nlocal=nlocal)
end

# ╔═╡ d0000000-0000-0000-0000-000000000010
# η² per feature and leave-one-out nearest-class-mean on a column subset.
begin
    function eta2(col,y)
        gm=mean(col); sst=sum((col.-gm).^2); sst==0 && return 0.0
        ssb=sum(count(y.==c)*(mean(col[y.==c])-gm)^2 for c in unique(y)); ssb/sst
    end
    function loo(D,y,cols)
        Z=D[:,cols]; mu=vec(mean(Z,dims=1)); sd=vec(std(Z,dims=1)); sd[sd.==0].=1f0
        Z=(Z.-mu')./sd'; n=size(Z,1); cls=unique(y); ok=0
        for i in 1:n
            best=Inf; bc=""
            for c in cls
                idx=[j for j in 1:n if y[j]==c && j!=i]; isempty(idx)&&continue
                m=vec(mean(Z[idx,:],dims=1)); d=sum((Z[i,:].-m).^2); d<best&&(best=d;bc=c)
            end
            bc==y[i]&&(ok+=1)
        end
        ok/n
    end
end

# ╔═╡ d0000000-0000-0000-0000-000000000011
md"""### Diagnosticity η² per feature"""

# ╔═╡ d0000000-0000-0000-0000-000000000012
let
    D,y,names=features.D,features.y,features.names
    e=[eta2(D[:,j],y) for j in 1:length(names)]
    ord=sortperm(e)
    bar(1:length(names), e[ord], orientation=:h, yticks=(1:length(names),names[ord]),
        legend=false, color=:teal, xlims=(0,0.75), xlabel="η²  (variance explained by letter identity)",
        title="encoding: $method", titlefontsize=9, size=(640,max(220,40length(names))))
end

# ╔═╡ d0000000-0000-0000-0000-000000000013
md"""### Leave-one-out accuracy"""

# ╔═╡ d0000000-0000-0000-0000-000000000014
let
    D,y=features.D,features.y; nl=features.nlocal; ncol=size(D,2)
    chance=round(100/length(unique(y)),digits=1)
    aloc=round(100loo(D,y,1:nl),digits=1)
    afull=round(100loo(D,y,1:ncol),digits=1)
    shapetxt = ncol>nl ?
        "- shape descriptor only : **$(round(100loo(D,y,nl+1:ncol),digits=1)) %**\n" : ""
    Markdown.parse("""
    chance = **$chance %**  ·  $(length(y)) instances, $(length(unique(y))) classes

    - local features only  : **$aloc %**
    $shapetxt- full descriptor      : **$afull %**
    """)
end

# ╔═╡ d0000000-0000-0000-0000-000000000015
md"""### Per-class means"""

# ╔═╡ d0000000-0000-0000-0000-000000000016
let
    D,y,names=features.D,features.y,features.names
    esc(s)=replace(s, "|"=>"\\|")   # feature names contain | (|M1|/M0) — escape for the table
    hdr="| class | " * join(esc.(names)," | ") * " |"
    sep="|" * repeat("---|", length(names)+1)
    rows=[ "| **$c** | " * join([string(round(mean(D[y.==c,j]),digits=2)) for j in 1:length(names)], " | ") * " |"
           for c in CLASSES ]
    Markdown.parse(join(vcat(hdr,sep,rows...), "\n"))
end

# ╔═╡ d0000000-0000-0000-0000-000000000017
md"""
## Notes

- **Shape descriptor dominates.** Set the shape descriptor to "none" to see the
  local features alone (typically 15–20 % vs. ~57–59 % for shape-only): the census
  of keypoint types carries little identity because it discards *where* they are.
- **Higher moments help.** `angular |M1..6|` beats `|M1..4|` (~59 % vs ~57 %:
  `|M5|`, `|M6|` carry real signal), and adding the **radial** profile
  (filled-vs-hollow — the mass at each normalised radius) lifts it a little more
  (~59.4 %). Past `|M6|` it plateaus.
- **Ray-probe scale is the endpoint lever.** Slide `D_RAY`: on clean synthetic
  figures the endpoint signature is near-broken at 15 (`|c₁|/c₀`≈0.55) and
  near-ideal at 20 (`≈0.81`) — an endpoint's one uncancelled ray only reads
  cleanly once the probe clears the stroke width. Junctions (T/X) are scale-stable.
  On real letters a larger probe also raises cross-talk between nearby strokes.
- **Greedy vs. clear local maxima.** Compare the two mean-pool rows: the greedy
  tiles the `c₀` ridges (~12 points), giving a denser — and, as a *mean*, less
  noisy — statistic than the ~5–15 clear maxima. Cleaner points, weaker pooled
  feature.
- **Typed counts** re-introduce hard type labels (against the handoff doc's
  "descriptor is continuous" rule) and depend on a fragile ray-count calibration;
  see the synthetic-figure section of the markdown for where the typing breaks
  even on clean input.
- Next: bind each keypoint to its centroid-relative position `(r, α)` and encode
  the *configuration*, not the count.
"""

# ╔═╡ Cell order:
# ╠═d0000000-0000-0000-0000-000000000001
# ╠═d0000000-0000-0000-0000-000000000002
# ╠═d0000000-0000-0000-0000-000000000003
# ╟─d0000000-0000-0000-0000-000000000004
# ╠═d0000000-0000-0000-0000-000000000005
# ╠═d0000000-0000-0000-0000-000000000006
# ╠═d0000000-0000-0000-0000-000000000007
# ╠═d0000000-0000-0000-0000-000000000008
# ╠═d0000000-0000-0000-0000-000000000009
# ╠═d0000000-0000-0000-0000-00000000000a
# ╟─d0000000-0000-0000-0000-00000000000b
# ╠═d0000000-0000-0000-0000-00000000000c
# ╠═d0000000-0000-0000-0000-00000000000d
# ╟─d0000000-0000-0000-0000-00000000000e
# ╠═d0000000-0000-0000-0000-00000000000f
# ╠═d0000000-0000-0000-0000-0000000000f2
# ╠═d0000000-0000-0000-0000-000000000010
# ╟─d0000000-0000-0000-0000-000000000011
# ╠═d0000000-0000-0000-0000-000000000012
# ╟─d0000000-0000-0000-0000-000000000013
# ╠═d0000000-0000-0000-0000-000000000014
# ╟─d0000000-0000-0000-0000-000000000015
# ╠═d0000000-0000-0000-0000-000000000016
# ╟─d0000000-0000-0000-0000-000000000017
