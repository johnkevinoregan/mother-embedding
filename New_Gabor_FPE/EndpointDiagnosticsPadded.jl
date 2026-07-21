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

# ╔═╡ f0000000-0000-0000-0000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ f0000000-0000-0000-0000-000000000002
begin
    using PlutoUI
    using Plots
    using FFTW
    using Statistics
end

# ╔═╡ f0000000-0000-0000-0000-000000000003
begin
    include(joinpath(@__DIR__, "..", "LoadEMNIST.jl"))
    using .LoadEMNIST
end

# ╔═╡ f0000000-0000-0000-0000-000000000004
md"""
# Endpoint detector — diagnostics on a **padded** field

Padded variant of `EndpointDiagnostics.jl`. The letter still occupies a
**112×112** patch, but it is now embedded in a **background border wide enough to
swallow the largest kernel**, and *that* padded array is the working image:

```
IMG0 = 112   (letter)      PAD = 56   (border)      IMG = 224  (working field)
```

Two consequences:

1. **Every panel — heatmaps and kernels alike — is 224×224**, so a kernel is drawn
   at its true size relative to the letter. That was the point of the change: a
   61 px kernel now visibly spans ~half the letter's width.
2. **Edge-replicate padding is now exact rather than an approximation.** The
   outermost pixels of the working field are pure background, so replicating them
   reproduces the true uniform surround. Previously a stroke reaching the image
   border had its cap smeared by replicated ink; now every stroke end sits in
   genuine background.

**Cost:** 224² is 4× the pixels of 112², so expect a couple of seconds per slider
move rather than under one. If that proves too heavy, `EndpointDiagnostics.jl`
(the 112 version) is unchanged and still works.

### The detector

```
S(q) = max over φ, p of  min( |Ce_θ(φ)(p)| , cap(φ) )    deposited at q = p + δ·u(φ)
cap(φ) = [ −sign(Ce_θ(φ)(p)) · κ · ds(φ) · Co_{θ+π/2}(q) ]₊
```

`Ce = Re(Gabor∗I)` is the **even/line** channel, `Co = Im(Gabor∗I)` the
**odd/edge** channel. δ is a fraction of the line filter's half-length σ_T.

### What to look for

- **`align`** = `|cos(φ* − θ_dominant)|`, tangent taken at the *source* `p`:
  1 = winning direction runs **along** the stroke (as designed), 0 = **across**
  it, meaning the probe landed on the stroke's own flank.
- **`cap`** bright along whole strokes rather than only at ends is the signature
  of the over-firing that has defeated this detector on EMNIST.
- Tick **invert polarity**: every panel should be unchanged (`s` flips — it is the
  reference that makes `cap` invariant).
"""

# ╔═╡ f0000000-0000-0000-0000-000000000005
begin
    const IMG0=112                 # the letter patch
    const PAD=56                   # border ≥ half the largest kernel (ks=113 at w_L=15)
    const IMG=IMG0+2PAD            # 224 working field — all maps and panels are this size
    const N_THETA=24; const K_PHI=24
    const THETAS=Float32.(range(0,π,length=N_THETA+1)[1:N_THETA])
    const PHIS=Float32.(range(0,2π,length=K_PHI+1)[1:K_PHI])
    const CLASSES=["O","C","I","L","T","X","K","A","H","Y","E","F"]
    tidx(φ)=mod(round(Int,(mod(φ,Float32(π))/Float32(π))*N_THETA),N_THETA)+1
    perp(t)=mod(t-1+N_THETA÷2,N_THETA)+1
end

# ╔═╡ f0000000-0000-0000-0000-000000000006
# ---- Gabor bank; edge-replicate padding (now EXACT: the border is real background)
begin
    struct Bank; ks::Int; o::Int; N::Int; bhat::Vector{Matrix{ComplexF32}}; end
    Lparams(w)=(1.5f0*w,   w/2, 2f0*w)   # (sigT,sigN,lam) line filter, along-extent 3w
    Eparams(w)=(w/2.5f0,   w/2, 2f0*w)   # edge filter: same λ and σ_N, ~4× shorter
    function ksize_for(sigT,sigN)
        k=2*floor(Int,2.5*max(sigT,sigN))+1
        isodd(k) ? k : k+1
    end
    function gabor_kernel(sigT,sigN,lam,θ)
        ks=ksize_for(sigT,sigN); h=ks÷2
        K=zeros(ComplexF32,ks,ks); c,s=cos(θ),sin(θ)
        for a in -h:h, b in -h:h
            y,x=Float32(a),Float32(b); xt=x*c+y*s; xn=-x*s+y*c
            K[a+h+1,b+h+1]=exp(-(xt^2/(2sigT^2)+xn^2/(2sigN^2)))*cis(2f0π*xn/lam)
        end
        K .- mean(K)                     # DC-free: inversion only flips sign
    end
    function make_bank(sigT,sigN,lam)
        ks=ksize_for(sigT,sigN); o=ks÷2
        need=(IMG+2o)+ks-1; N=nextprod([2,3,5],need)
        bh=Vector{Matrix{ComplexF32}}(undef,N_THETA)
        for (i,θ) in enumerate(THETAS)
            K=gabor_kernel(sigT,sigN,lam,θ)
            B=zeros(ComplexF32,N,N); B[1:ks,1:ks].=K; bh[i]=fft(B)
        end
        Bank(ks,o,N,bh)
    end
    function CeCo(img,bk::Bank)
        o=bk.o; M=IMG+2o; A=zeros(ComplexF32,bk.N,bk.N)
        for i in 1:M, j in 1:M
            A[i,j]=img[clamp(i-o,1,IMG), clamp(j-o,1,IMG)]
        end
        Ah=fft(A)
        Ce=Array{Float32,3}(undef,N_THETA,IMG,IMG); Co=similar(Ce)
        for t in 1:N_THETA
            f=ifft(Ah.*bk.bhat[t]); v=@view f[2o+1:2o+IMG, 2o+1:2o+IMG]
            Ce[t,:,:].=real.(v); Co[t,:,:].=imag.(v)
        end
        Ce,Co
    end
    const LCACHE=Dict{Float32,Bank}(); const ECACHE=Dict{Float32,Bank}()
    Lbank(w)=get!(LCACHE,w) do; make_bank(Lparams(w)...) end
    Ebank(w)=get!(ECACHE,w) do; make_bank(Eparams(w)...) end
end

# ╔═╡ f0000000-0000-0000-0000-000000000007
begin
    @inline function bilinear(M,y,x)
        H,W2=size(M); (y<1||x<1||y>H||x>W2)&&return 0f0
        y0,x0=floor(Int,y),floor(Int,x); y1,x1=min(y0+1,H),min(x0+1,W2); fy,fx=y-y0,x-x0
        (1-fy)*(1-fx)*M[y0,x0]+fy*(1-fx)*M[y1,x0]+(1-fy)*fx*M[y0,x1]+fy*fx*M[y1,x1]
    end
    # EMNIST 28x28 -> the 112 letter patch
    function upsample(img)
        H,W2=size(img); out=zeros(Float32,IMG0,IMG0)
        for i in 1:IMG0,j in 1:IMG0
            out[i,j]=bilinear(img,Float32(1+(i-1)*(H-1)/(IMG0-1)),Float32(1+(j-1)*(W2-1)/(IMG0-1)))
        end
        out
    end
    # 112 letter patch -> 224 working field, surrounded by background (0 = ink-free)
    function embed(letter)
        out=zeros(Float32,IMG,IMG)
        out[PAD+1:PAD+IMG0, PAD+1:PAD+IMG0] .= letter
        out
    end
    function kpts(M,thr;r=8)
        cand=Tuple{Int,Int,Float32}[]
        for y in 3:IMG-2,x in 3:IMG-2
            v=M[y,x]; v<=thr&&continue; ok=true
            for dy in -2:2,dx in -2:2
                (dy==0&&dx==0)&&continue; M[y+dy,x+dx]>v&&(ok=false;break)
            end
            ok&&push!(cand,(y,x,v))
        end
        sort!(cand,by=p->-p[3]); keep=Tuple{Int,Int,Float32}[]
        for c in cand; any(hypot(c[1]-k[1],c[2]-k[2])<r for k in keep)||push!(keep,c); end
        keep
    end
    function stroke_width(img;t=0.5f0)
        ink=img.>t; area=count(ink); area==0 && return 0f0
        per=0
        for y in 2:IMG-1,x in 2:IMG-1
            ink[y,x]||continue
            (ink[y-1,x]&&ink[y+1,x]&&ink[y,x-1]&&ink[y,x+1])||(per+=1)
        end
        per==0 ? 0f0 : Float32(2*area/per)
    end
end

# ╔═╡ f0000000-0000-0000-0000-000000000008
# ---- detector: gather evidence at p, deposit the score at q = p + δ·u(φ) ----
function diag_maps(Ce,Co; delta, kappa=1f0)
    Lwin=zeros(Float32,IMG,IMG); capw=zeros(Float32,IMG,IMG)
    Ew  =fill(-1f0,IMG,IMG);     sw  =zeros(Float32,IMG,IMG)
    phiw=zeros(Float32,IMG,IMG); Ldom=zeros(Float32,IMG,IMG); thdom=zeros(Float32,IMG,IMG)
    alignm=zeros(Float32,IMG,IMG)
    for y in 1:IMG, x in 1:IMG
        bd=0f0; bt=1
        for t in 1:N_THETA
            a=abs(Ce[t,y,x]); a>bd && (bd=a; bt=t)
        end
        Ldom[y,x]=bd; thdom[y,x]=THETAS[bt]
    end
    for y in 1:IMG, x in 1:IMG
        for φ in PHIS
            t=tidx(φ); ce=Ce[t,y,x]; a=abs(ce); ψ=perp(t)
            qy=y+delta*sin(φ); qx=x+delta*cos(φ)
            co=bilinear(@view(Co[ψ,:,:]), qy, qx)
            ds = φ < Float32(π) ? 1f0 : -1f0
            cap = max(-sign(ce)*kappa*ds*co, 0f0)
            v=min(a,cap)
            iy=round(Int,qy); ix=round(Int,qx)
            (1<=iy<=IMG && 1<=ix<=IMG) || continue
            if v>Ew[iy,ix]
                Ew[iy,ix]=v; Lwin[iy,ix]=a; capw[iy,ix]=cap
                sw[iy,ix]=sign(ce); phiw[iy,ix]=φ
                alignm[iy,ix]=abs(cos(φ-thdom[y,x]))
            end
        end
    end
    @inbounds for i in eachindex(Ew); Ew[i]<0 && (Ew[i]=0f0); end
    (; Lwin, capw, Ew, sw, phiw, Ldom, align=alignm)
end

# ╔═╡ f0000000-0000-0000-0000-000000000014
# ---- Ronan-style symmetric end-stop (classical hypercomplex cell):
#        Es(p) = [ E_θ*(p) − β·( E_θ*(p+Δu) + E_θ*(p−Δu) ) ]₊
# on the LINE bank's oriented energy E_θ = √(Ce² + Co²) — the phase-invariant
# envelope, so no oscillatory |Ce| side-lobes to fire on, and exactly polarity-
# invariant (both parts flip under inversion). θ* = dominant orientation at p; Δ
# reuses δ; score registered at p (~Δ inside the true end). Ce, Co are BOTH from
# Lbank (a true quadrature pair), unlike the min-gate which pairs Lbank·Ce with Ebank·Co.
function endstop_sym(Ce, Co; delta, beta)
    E  = @. sqrt(Ce^2 + Co^2)                      # oriented energy of the line bank
    Es = zeros(Float32, IMG, IMG)
    for y in 1:IMG, x in 1:IMG
        bd=0f0; bt=1
        for t in 1:N_THETA
            E[t,y,x]>bd && (bd=E[t,y,x]; bt=t)
        end
        θ=THETAS[bt]; dy=delta*sin(θ); dx=delta*cos(θ); Et=@view E[bt,:,:]
        f1=bilinear(Et, y+dy, x+dx); f2=bilinear(Et, y-dy, x-dx)
        Es[y,x]=max(0f0, bd - beta*(f1+f2))
    end
    Es
end

# ╔═╡ f0000000-0000-0000-0000-000000000009
# ---- synthetic figures (drawn on the 112 patch, then embedded) + EMNIST ----
begin
    function bar!(img,y0,x0,y1,x1,r)
        n=round(Int,hypot(y1-y0,x1-x0))*2
        for t in range(0,1,length=n)
            yc=y0+t*(y1-y0); xc=x0+t*(x1-x0)
            for dy in -r:r,dx in -r:r
                hypot(dy,dx)<=r && (img[clamp(round(Int,yc+dy),1,IMG0),clamp(round(Int,xc+dx),1,IMG0)]=1f0)
            end
        end
        img
    end
    function ring!(img,cy,cx,R,r)
        for t in range(0,2π,length=800)
            yc=cy+R*sin(t); xc=cx+R*cos(t)
            for dy in -r:r,dx in -r:r
                hypot(dy,dx)<=r && (img[clamp(round(Int,yc+dy),1,IMG0),clamp(round(Int,xc+dx),1,IMG0)]=1f0)
            end
        end
        img
    end
    z()=zeros(Float32,IMG0,IMG0)
    function synth_img(name,r)
        name=="bar"    && return bar!(z(),56,26,56,86,r)
        name=="plus"   && return (i=z();bar!(i,26,56,86,56,r);bar!(i,56,26,56,86,r))
        name=="T"      && return (i=z();bar!(i,36,26,36,86,r);bar!(i,36,56,86,56,r))
        name=="X"      && return (i=z();bar!(i,26,26,86,86,r);bar!(i,26,86,86,26,r))
        name=="L-shape"&& return (i=z();bar!(i,26,40,80,40,r);bar!(i,80,40,80,86,r))
        return ring!(z(),56,56,28,r)
    end
    const SYNTH=["bar","plus","T","X","L-shape","O (ring)"]
    em = load_emnist(n_images_to_load=8000, n_classes=47)
end;

# ╔═╡ f0000000-0000-0000-0000-000000000010
md"""
### The kernels, at the same scale as the image

Both are drawn on the **full 224×224 working field**, exactly as the heatmaps
below — so their size relative to a letter is now literally what you see. The
`Re` of the L kernel (even/line → `Ce`) and the `Im` of the Ed kernel (odd/edge →
`Co`) are the components actually convolved. Row 3 re-plots the edge kernels at
native size, since at true scale the small ones are nearly invisible — that row
alone is *not* to scale.
"""

# ╔═╡ f0000000-0000-0000-0000-000000000011
begin
    function on_canvas(K, C=IMG)
        out=zeros(Float32,C,C); ks=size(K,1); h=ks÷2; c0=C÷2+1
        for a in -h:h, b in -h:h
            y=c0+a; x=c0+b
            (1<=y<=C && 1<=x<=C) && (out[y,x]=K[a+h+1,b+h+1])
        end
        out
    end
    const LW_SHOW=Float32[4,6,8,10,12,15]
    const EW_SHOW=Float32[3,4,6,8,10,12]
    function kpanel(M,ttl)
        m=maximum(abs,M); m=m==0 ? 1f0 : m
        heatmap(M; c=:RdBu, clims=(-m,m), title=ttl, titlefontsize=8,
                aspect_ratio=:equal, axis=false, ticks=false, cbar=false, yflip=true)
    end
    panels=[]
    for w in LW_SHOW
        sT,sN,lam=Lparams(w); K=gabor_kernel(sT,sN,lam,0f0)
        push!(panels, kpanel(on_canvas(real.(K)), "Re L  w=$(Int(w))  ks=$(size(K,1))"))
    end
    for w in EW_SHOW
        sT,sN,lam=Eparams(w); K=gabor_kernel(sT,sN,lam,0f0)
        push!(panels, kpanel(on_canvas(imag.(K)), "Im Ed  w=$(Int(w))  ks=$(size(K,1))"))
    end
    for w in EW_SHOW
        sT,sN,lam=Eparams(w); K=gabor_kernel(sT,sN,lam,0f0)
        push!(panels, kpanel(imag.(K), "Im Ed  w=$(Int(w))  (native, NOT to scale)"))
    end
    plot(panels...; layout=(3,6), size=(1250,650))
end

# ╔═╡ f0000000-0000-0000-0000-00000000000a
md"""
### Controls

source: $(@bind src Select(vcat(SYNTH, "EMNIST: " .* CLASSES), default="bar"))
EMNIST instance: $(@bind inst Slider(1:30, default=1, show_value=true))
synthetic stroke radius: $(@bind srad Slider(2:1:9, default=6, show_value=true))

**Ce (even / line) scale** `w_L`: $(@bind wL Slider(Float32[4,6,8,10,12,15], default=8f0, show_value=true))
**Co (odd / edge) scale** `w_E`: $(@bind wE Slider(Float32[3,4,6,8,10,12], default=8f0, show_value=true))
**δ / σ_T** (probe offset as a fraction of the line filter's half-length): $(@bind dratio Slider(0.5f0:0.1f0:3.0f0, default=1.0f0, show_value=true))

κ (sign convention): $(@bind kappa Select([1f0=>"+1 (falling edge)", -1f0=>"−1 (rising edge)"]))
keypoint threshold (× max): $(@bind kthr Slider(0.10f0:0.05f0:0.60f0, default=0.30f0, show_value=true))
**β** (symmetric end-stop, Ronan A/B): $(@bind beta Slider(0.0f0:0.05f0:1.2f0, default=0.5f0, show_value=true))
invert polarity: $(@bind invert CheckBox(default=false))
"""

# ╔═╡ f0000000-0000-0000-0000-000000000013
begin
    sigT_L = 1.5f0 * wL
    delta  = dratio * sigT_L
end

# ╔═╡ f0000000-0000-0000-0000-00000000000b
begin
    letter = if startswith(src, "EMNIST: ")
        LN = replace(src, "EMNIST: "=>"")
        upsample(em.class_images[findfirst(==(LN), em.class_names)][inst])
    else
        synth_img(src, srad)
    end
    img_raw = embed(letter)                          # 112 patch -> 224 field
    img = invert ? (1f0 .- img_raw) : img_raw        # inverts the WHOLE field, border too
    Ce, CoL = CeCo(img, Lbank(wL))                   # line bank: quadrature pair (Ce, CoL)
    _,  Co  = CeCo(img, Ebank(wE))                   # edge bank: Co for the min-gate cap
    D = diag_maps(Ce, Co; delta=delta, kappa=kappa)
    kp = kpts(D.Ew, kthr*maximum(D.Ew))
    swid = stroke_width(img_raw)
    swid_s = round(swid, digits=1)
    sig_s  = round(sigT_L, digits=1)
    del_s  = round(delta, digits=1)
    nkp    = length(kp)
    Markdown.parse("**source** `$(src)` — stroke width **$(swid_s) px** — " *
        "field **$(IMG)×$(IMG)** (letter $(IMG0), border $(PAD)) — " *
        "`w_L`=$(wL), `w_E`=$(wE), σ_T=$(sig_s), **δ=$(del_s) px** = $(dratio)·σ_T — " *
        "**$(nkp) endpoints detected** *(scored at p+δ·u(φ))*")
end

# ╔═╡ f0000000-0000-0000-0000-000000000012
let
    sT,sN,lam    = Lparams(wL)
    sT2,sN2,lam2 = Eparams(wE)
    KL0 = gabor_kernel(sT,sN,lam,0f0)
    KE0 = gabor_kernel(sT2,sN2,lam2,0f0)
    ttlL = "Re L  w_L=" * string(wL) * "  ks=" * string(size(KL0,1))
    ttlE = "Im Ed w_E=" * string(wE) * "  ks=" * string(size(KE0,1))
    # combined: stroke vertical, line filter at p, edge filter at p+δ (above)
    KLv = gabor_kernel(sT,sN,lam, Float32(π)/2)
    KEv = gabor_kernel(sT2,sN2,lam2, 0f0)
    c0 = IMG÷2 + 1
    canvas = zeros(Float32, IMG, IMG)
    A = real.(KLv); A ./= max(maximum(abs,A), 1f-9)
    hL = size(A,1)÷2
    for a in -hL:hL, b in -hL:hL
        y=c0+a; x=c0+b
        (1<=y<=IMG && 1<=x<=IMG) && (canvas[y,x] += A[a+hL+1, b+hL+1])
    end
    B = imag.(KEv); B ./= max(maximum(abs,B), 1f-9)
    hE = size(B,1)÷2; oy = c0 - round(Int, delta)
    for a in -hE:hE, b in -hE:hE
        y=oy+a; x=c0+b
        (1<=y<=IMG && 1<=x<=IMG) && (canvas[y,x] += B[a+hE+1, b+hE+1])
    end
    m = max(maximum(abs,canvas), 1f-9)
    p3 = heatmap(canvas; c=:RdBu, clims=(-m,m), titlefontsize=8,
                 title="combined: L at p, Ed at p+δ   (δ=" * string(round(delta,digits=1)) * ")",
                 aspect_ratio=:equal, axis=false, ticks=false, cbar=false, yflip=true)
    scatter!(p3, [c0], [c0]; mc=:black, msw=0, ms=4, label="")
    scatter!(p3, [c0], [oy]; mc=:lime,  msw=0, ms=4, label="")
    plot!(p3, [c0,c0], [c0,oy]; lc=:lime, lw=2, label="")
    plot(kpanel(on_canvas(real.(KL0)), ttlL),
         kpanel(on_canvas(imag.(KE0)), ttlE),
         p3; layout=(1,3), size=(1050,360))
end

# ╔═╡ f0000000-0000-0000-0000-00000000000c
begin
    kw=(yflip=true, aspect_ratio=:equal, axis=false, ticks=false, cbar=false,
        xlims=(1,IMG), ylims=(1,IMG))
    p1=heatmap(img; c=:grays, title="input (224 field) + keypoints", kw...)
    isempty(kp) || scatter!(p1,[p[2] for p in kp],[p[1] for p in kp];
                            mc=:cyan, msw=0, ms=5, label="")
    p2=heatmap(D.Lwin; c=:viridis, title="|Ce| of winning source (line)", kw...)
    p3=heatmap(D.capw; c=:viridis, title="cap at this pixel (edge)", kw...)
    p4=heatmap(D.Ew;   c=:magma,   title="Endpoint = min(|Ce|, cap)  @ p+δu", kw...)
    # Where the score is ~0 nothing genuinely won, and phiw/sw/align hold whichever
    # candidate merely wrote first — arbitrary. Mask those out (NaN renders blank)
    # so the direction panel shows real winners only.
    live = D.Ew .> 1f-2*maximum(D.Ew)
    function masked(M)
        out=fill(NaN32, size(M)); out[live].=M[live]; out
    end
    p5=heatmap(masked(D.sw);   c=:RdBu, clims=(-1,1), title="s = sign(Ce)  (live only)", kw...)
    p6=heatmap(masked(D.align); c=:viridis, clims=(0,1),
               title="align: 1=along stroke, 0=across", kw...)
    p7=heatmap(masked(D.phiw); c=:hsv, clims=(0,2π),
               title="winning direction φ*  (24 steps of 15°)", kw...)
    p8=heatmap(D.Ldom; c=:viridis, title="max_θ |Ce| (dominant line)", kw...)
    plot(p1,p2,p3,p4,p5,p6,p7,p8; layout=(2,4), size=(1300,660))
end

# ╔═╡ f0000000-0000-0000-0000-00000000000d
begin
    onink = img_raw .> 0.5f0
    fire  = D.Ew .> kthr*maximum(D.Ew)
    both  = onink .& fire
    al_fire = any(fire) ? round(mean(D.align[fire]),digits=2) : 0.0
    al_ink  = any(both) ? round(mean(D.align[both]),digits=2) : 0.0
    Markdown.parse("### Hypothesis readout\n\n" *
        "Mean **align** over firing pixels: **$(al_fire)** · " *
        "over firing pixels *on ink*: **$(al_ink)**\n\n" *
        "Near **1** ⇒ the winning direction runs **along** the stroke " *
        "(detector working as designed). Near **0** ⇒ it points **across** the " *
        "stroke, i.e. the probe lands on the stroke's own flank rather than a " *
        "genuine end-cap — the suspected mechanism behind the over-firing.")
end

# ╔═╡ f0000000-0000-0000-0000-000000000016
md"""
### A/B — classical symmetric end-stop (Ronan) vs our min-gate

Ronan's notebook obtains end-stopping the **textbook hypercomplex way**: a line-tuned
simple cell inhibited by the *same* channel displaced **±Δ along its own orientation**,
then rectified —

```
C_θ(p) = [ S_θ(p) − β·( S_θ(p+Δ·u_θ) + S_θ(p−Δ·u_θ) ) ]₊
```

— collapsed at the dominant orientation. A straight stroke has both flanks filled and
cancels; an end has one empty flank and fires. It is **symmetric** (fires at both ends,
corners, crossings) and **subtractive**, where ours is **directional** and a **min-gate**.

Below the same mechanism runs on our line bank's **oriented energy**
`S_θ = E_θ = √(Ce_θ² + Co_θ²)` — the phase-invariant envelope (kept full-wave so it stays
polarity-invariant; Ronan's own `S_θ` is ON-only). Energy rather than `|Ce|` matters here:
`|Ce|` of a Gabor has oscillatory side-lobes that `abs` turns into phantom ridges parallel
to the stroke, which a symmetric end-stop then fires on; the energy envelope has none.
It reuses **Δ = δ** and adds only β. The symmetric response sits at **p** (~Δ inside the
true end); ours at **p+δ·u** (the end-cap itself). Compare where the two fire.

**A/B finding (β is critical).** On a plain bar, Ronan's own default **β=0.7 kills the
true ends** here — `E(end) − β·(behind+ahead)` goes negative because our broad `sigT=12`
line kernel makes the *behind* flank full-strength while the end-response is still on its
gentle down-ramp; only diagonal skirt artifacts survive. **β≈0.5** is the working regime:
the straight middle is suppressed but the ends survive (max `Es` lands on an end). Ronan's
sharply-peaked `hw≈5` cell tolerates β=0.7; our smooth Gabor energy does not. Lower `w_L`
(smaller kernel) widens the usable β range.
"""

# ╔═╡ f0000000-0000-0000-0000-000000000015
begin
    Es = endstop_sym(Ce, CoL; delta=delta, beta=beta)
    kp_sym = kpts(Es, kthr*maximum(Es))
    Markdown.parse("**min-gate:** $(nkp) keypoints (score at p+δu) · " *
        "**symmetric** `[E−β(E₊+E₋)]₊` on line-bank energy √(Ce²+Co²), β=$(beta), " *
        "Δ=δ=$(del_s) px: **$(length(kp_sym))** keypoints (score at p).")
end

# ╔═╡ f0000000-0000-0000-0000-000000000017
let
    kw2=(yflip=true, aspect_ratio=:equal, axis=false, ticks=false, cbar=false,
         xlims=(1,IMG), ylims=(1,IMG))
    pa=heatmap(img; c=:grays, legend=:topright, titlefontsize=8,
               title="min-gate (cyan) vs symmetric (orange)", kw2...)
    isempty(kp)     || scatter!(pa,[p[2] for p in kp],     [p[1] for p in kp];
                                mc=:cyan,   msw=0, ms=5, label="min-gate")
    isempty(kp_sym) || scatter!(pa,[p[2] for p in kp_sym], [p[1] for p in kp_sym];
                                mc=:orange, msw=0, ms=5, label="symmetric")
    pb=heatmap(D.Ew; c=:magma, titlefontsize=8,
               title="min-gate  min(|Ce|,cap) @ p+δu", kw2...)
    pc=heatmap(Es;   c=:magma, titlefontsize=8,
               title="symmetric  [E−β·(±Δ flanks)]₊ @ p", kw2...)
    plot(pa,pb,pc; layout=(1,3), size=(1150,400))
end

# ╔═╡ f0000000-0000-0000-0000-00000000000e
md"""
### Notes

- **Why pad the field.** Two gains over `EndpointDiagnostics.jl`: panels are all
  224×224 so kernels appear at true size against the letter; and edge-replicate
  padding becomes *exact*, because the working field's border is genuine uniform
  background rather than replicated ink. A stroke ending near the old 112 border
  previously had its cap corrupted; now every end sits in real background.
- **Counts are not comparable** with the 112 notebook or with the figures in
  `EvenOddChannels.md` — different field, different `maximum(E)` normalisation.
  Re-run any comparison you care about rather than reading across.
- **Performance.** 224² is 4× the pixels; the FFTs run at 450² for the largest
  kernel. Expect a couple of seconds per slider move. Banks are cached, so
  revisiting a scale is instant. If it's too slow, the 112 notebook is unchanged.
- **Polarity.** The invert checkbox flips the *whole* field including the border,
  so the surround stays uniform and the DC-free argument holds exactly; every
  panel except `s` should be visually identical.
- **Scale conventions.** `Lparams(w) = (1.5w, w/2, 2w)`; `Eparams(w) =
  (w/2.5, w/2, 2w)` — the edge filter shares the line filter's wavelength and
  cross-section and is only ~4× shorter along its own axis.
- **Why the φ\\* panel looks blocky.** Three reasons, all expected: φ\\* is an
  `argmax` over only **24 directions**, so it is quantised to 15° steps and
  rendered as 24 discrete hues; `argmax` is **discontinuous** — near-ties flip the
  winner abruptly, producing watershed boundaries (that itself is useful: it marks
  where the choice was marginal); and since scores are **scattered** to
  `q = p + δ·u(φ)`, neighbouring output pixels may be written by unrelated source
  pixels. Pixels with negligible score are now **masked out** rather than showing
  the arbitrary direction that happened to write first. Raise `K_PHI` in the
  constants cell for finer angular resolution at proportional cost.
"""

# ╔═╡ Cell order:
# ╠═f0000000-0000-0000-0000-000000000001
# ╠═f0000000-0000-0000-0000-000000000002
# ╠═f0000000-0000-0000-0000-000000000003
# ╟─f0000000-0000-0000-0000-000000000004
# ╠═f0000000-0000-0000-0000-000000000005
# ╠═f0000000-0000-0000-0000-000000000006
# ╠═f0000000-0000-0000-0000-000000000007
# ╠═f0000000-0000-0000-0000-000000000008
# ╠═f0000000-0000-0000-0000-000000000014
# ╠═f0000000-0000-0000-0000-000000000009
# ╟─f0000000-0000-0000-0000-000000000010
# ╠═f0000000-0000-0000-0000-000000000011
# ╟─f0000000-0000-0000-0000-00000000000a
# ╠═f0000000-0000-0000-0000-000000000013
# ╠═f0000000-0000-0000-0000-00000000000b
# ╠═f0000000-0000-0000-0000-000000000012
# ╠═f0000000-0000-0000-0000-00000000000c
# ╟─f0000000-0000-0000-0000-00000000000d
# ╟─f0000000-0000-0000-0000-000000000016
# ╟─f0000000-0000-0000-0000-000000000015
# ╠═f0000000-0000-0000-0000-000000000017
# ╟─f0000000-0000-0000-0000-00000000000e
