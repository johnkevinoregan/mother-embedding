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

# ╔═╡ e0000000-0000-0000-0000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ e0000000-0000-0000-0000-000000000002
begin
    using PlutoUI
    using Plots
    using FFTW
    using Statistics
end

# ╔═╡ e0000000-0000-0000-0000-000000000003
begin
    include(joinpath(@__DIR__, "..", "LoadEMNIST.jl"))
    using .LoadEMNIST
end

# ╔═╡ e0000000-0000-0000-0000-000000000004
md"""
# Endpoint detector — diagnostic heatmaps

Interactive companion to `EvenOddChannels.md`. The detector under inspection is

```
S(p) = max_φ  min( |Ce_θ(φ)(p)| ,  cap(φ) )
cap(φ) = [ −sign(Ce_θ(φ)(p)) · κ · ds(φ) · Co_{θ+π/2}(p + δ·u(φ)) ]₊
```

— *"a bar of orientation θ is here **AND** the perpendicular edge δ ahead is a
**falling** edge (relative to this bar's own contrast)."* `Ce = Re(Gabor∗I)` is
the **even/line** channel, `Co = Im(Gabor∗I)` the **odd/edge** channel;
`ds(φ) = ±1` flips the required sign between φ and φ+π, which is what makes the
detector directional.

**Ce and Co have independent scale sliders**, plus δ. Each panel shows the value
**at the winning φ** for that pixel, so you can see *which* term is responsible
for a response.

### What to look for

- **`align`** = `|cos(φ* − θ_dominant)|`: 1 = the winning direction runs **along**
  the stroke (what the detector is *supposed* to do), 0 = it points **across**
  the stroke. My untested hypothesis for the ~12× over-firing on EMNIST is that
  `max_φ` picks *across-stroke* directions, where the probe lands on the stroke's
  own **flank** — if so, `align` will be near 0 wherever the detector fires
  spuriously.
- **`cap`** firing along the whole stroke rather than only at ends is the
  signature of that failure.
- **`s`** (sign of Ce) should be +1 on ink, −1 off it — flip the polarity
  checkbox and confirm every map is unchanged.
"""

# ╔═╡ e0000000-0000-0000-0000-000000000005
begin
    const IMG=112; const N_THETA=24; const K_PHI=24
    const THETAS=Float32.(range(0,π,length=N_THETA+1)[1:N_THETA])
    const PHIS=Float32.(range(0,2π,length=K_PHI+1)[1:K_PHI])
    const CLASSES=["O","C","I","L","T","X","K","A","H","Y","E","F"]
    tidx(φ)=mod(round(Int,(mod(φ,Float32(π))/Float32(π))*N_THETA),N_THETA)+1
    perp(t)=mod(t-1+N_THETA÷2,N_THETA)+1
end

# ╔═╡ e0000000-0000-0000-0000-000000000006
# ---- Gabor bank with EDGE-REPLICATE padding (polarity-safe, no frame artifact) ----
begin
    struct Bank; ks::Int; o::Int; N::Int; bhat::Vector{Matrix{ComplexF32}}; end
    # scale conventions — shared by the banks AND the kernel display below,
    # so the pictures can never drift from what is actually convolved
    Lparams(w)=(1.5f0*w,   w/2, 2f0*w) # (sigT,sigN,lam) line filter: along-extent 3w
    # edge filter: SAME wavelength and cross-section as the line filter, just much
    # SHORTER along its own axis (sigT 1.5w -> 0.4w, i.e. ~4x shorter)
    Eparams(w)=(w/2.5f0,   w/2, 2f0*w)
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
        K .- mean(K)                        # DC-free: inversion only flips sign
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
            A[i,j]=img[clamp(i-o,1,IMG), clamp(j-o,1,IMG)]      # edge replicate
        end
        Ah=fft(A)
        Ce=Array{Float32,3}(undef,N_THETA,IMG,IMG); Co=similar(Ce)
        for t in 1:N_THETA
            f=ifft(Ah.*bk.bhat[t]); v=@view f[2o+1:2o+IMG, 2o+1:2o+IMG]
            Ce[t,:,:].=real.(v); Co[t,:,:].=imag.(v)
        end
        Ce,Co
    end
    # cache banks by scale so slider moves stay responsive
    const LCACHE=Dict{Float32,Bank}(); const ECACHE=Dict{Float32,Bank}()
    Lbank(w)=get!(LCACHE,w) do; make_bank(Lparams(w)...) end
    Ebank(w)=get!(ECACHE,w) do; make_bank(Eparams(w)...) end
end

# ╔═╡ e0000000-0000-0000-0000-000000000010
md"""
### The kernels themselves

Exactly the components that get convolved: **`Re` of the L kernel** (the even /
line filter, giving `Ce`) and **`Im` of the Ed kernel** (the odd / edge filter,
giving `Co`), shown at θ = 0 for every scale on the sliders.

Rows 1–2 put both on a **common 113 px canvas**, so the sizes are directly
comparable. The edge filter now shares the line filter's **wavelength (2w) and
cross-section (σ_N = w/2)** and differs only in being ~4× **shorter along its own
axis** (σ_T = w/2.5 vs 1.5w) — so the two rows differ in *length*, not in stripe
spacing. Row 3 re-plots the edge kernels at their **native size** so their
odd/antisymmetric structure is visible.
"""

# ╔═╡ e0000000-0000-0000-0000-000000000011
begin
    const CANVAS=113                    # ≥ largest kernel (L at w=15 -> ks=113)
    function on_canvas(K, C=CANVAS)
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
    for w in LW_SHOW                     # row 1: L kernels, common canvas
        sT,sN,lam=Lparams(w); K=gabor_kernel(sT,sN,lam,0f0)
        push!(panels, kpanel(on_canvas(real.(K)), "Re L  w=$(Int(w))  ks=$(size(K,1))"))
    end
    for w in EW_SHOW                     # row 2: Ed kernels, SAME canvas
        sT,sN,lam=Eparams(w); K=gabor_kernel(sT,sN,lam,0f0)
        push!(panels, kpanel(on_canvas(imag.(K)), "Im Ed  w=$(Int(w))  ks=$(size(K,1))"))
    end
    for w in EW_SHOW                     # row 3: Ed kernels, native size
        sT,sN,lam=Eparams(w); K=gabor_kernel(sT,sN,lam,0f0)
        push!(panels, kpanel(imag.(K), "Im Ed  w=$(Int(w))  (native)"))
    end
    plot(panels...; layout=(3,6), size=(1250,650))
end

# ╔═╡ e0000000-0000-0000-0000-000000000007
begin
    @inline function bilinear(M,y,x)
        H,W2=size(M); (y<1||x<1||y>H||x>W2)&&return 0f0
        y0,x0=floor(Int,y),floor(Int,x); y1,x1=min(y0+1,H),min(x0+1,W2); fy,fx=y-y0,x-x0
        (1-fy)*(1-fx)*M[y0,x0]+fy*(1-fx)*M[y1,x0]+(1-fy)*fx*M[y0,x1]+fy*fx*M[y1,x1]
    end
    function upsample(img)
        H,W2=size(img); out=zeros(Float32,IMG,IMG)
        for i in 1:IMG,j in 1:IMG
            out[i,j]=bilinear(img,Float32(1+(i-1)*(H-1)/(IMG-1)),Float32(1+(j-1)*(W2-1)/(IMG-1)))
        end
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

# ╔═╡ e0000000-0000-0000-0000-000000000008
# ---- the detector, returning every intermediate at the WINNING phi ----
function diag_maps(Ce,Co; delta, kappa=1f0)
    Lwin=zeros(Float32,IMG,IMG); capw=zeros(Float32,IMG,IMG)
    Ew  =zeros(Float32,IMG,IMG); sw  =zeros(Float32,IMG,IMG)
    phiw=zeros(Float32,IMG,IMG); Ldom=zeros(Float32,IMG,IMG); thdom=zeros(Float32,IMG,IMG)
    for y in 1:IMG, x in 1:IMG
        bd=0f0; bt=1
        for t in 1:N_THETA
            a=abs(Ce[t,y,x]); a>bd && (bd=a; bt=t)
        end
        Ldom[y,x]=bd; thdom[y,x]=THETAS[bt]
        best=-1f0; bl=0f0; bc=0f0; bs=0f0; bphi=0f0
        for φ in PHIS
            t=tidx(φ); ce=Ce[t,y,x]; a=abs(ce); ψ=perp(t)
            co=bilinear(@view(Co[ψ,:,:]), y+delta*sin(φ), x+delta*cos(φ))
            ds = φ < Float32(π) ? 1f0 : -1f0
            cap = max(-sign(ce)*kappa*ds*co, 0f0)
            v=min(a,cap)
            if v>best; best=v; bl=a; bc=cap; bs=sign(ce); bphi=φ; end
        end
        Lwin[y,x]=bl; capw[y,x]=bc; Ew[y,x]=best; sw[y,x]=bs; phiw[y,x]=bphi
    end
    align=abs.(cos.(phiw.-thdom))
    (; Lwin, capw, Ew, sw, phiw, Ldom, align)
end

# ╔═╡ e0000000-0000-0000-0000-000000000009
# ---- synthetic figures + EMNIST ----
begin
    function bar!(img,y0,x0,y1,x1,r)
        n=round(Int,hypot(y1-y0,x1-x0))*2
        for t in range(0,1,length=n)
            yc=y0+t*(y1-y0); xc=x0+t*(x1-x0)
            for dy in -r:r,dx in -r:r
                hypot(dy,dx)<=r && (img[clamp(round(Int,yc+dy),1,IMG),clamp(round(Int,xc+dx),1,IMG)]=1f0)
            end
        end
        img
    end
    function ring!(img,cy,cx,R,r)
        for t in range(0,2π,length=800)
            yc=cy+R*sin(t); xc=cx+R*cos(t)
            for dy in -r:r,dx in -r:r
                hypot(dy,dx)<=r && (img[clamp(round(Int,yc+dy),1,IMG),clamp(round(Int,xc+dx),1,IMG)]=1f0)
            end
        end
        img
    end
    z()=zeros(Float32,IMG,IMG)
    function synth_img(name,r)
        name=="bar"    && return bar!(z(),56,26,56,86,r)
        name=="plus"   && return (i=z();bar!(i,26,56,86,56,r);bar!(i,56,26,56,86,r))
        name=="T"      && return (i=z();bar!(i,36,26,36,86,r);bar!(i,36,56,86,56,r))
        name=="X"      && return (i=z();bar!(i,26,26,86,86,r);bar!(i,26,86,86,26,r))
        name=="L-shape"&& return (i=z();bar!(i,26,40,80,40,r);bar!(i,80,40,80,86,r))
        return ring!(z(),56,56,28,r)          # "O (ring)"
    end
    const SYNTH=["bar","plus","T","X","L-shape","O (ring)"]
    em = load_emnist(n_images_to_load=8000, n_classes=47)
end;

# ╔═╡ e0000000-0000-0000-0000-00000000000a
md"""
### Controls

source: $(@bind src Select(vcat(SYNTH, "EMNIST: " .* CLASSES), default="bar"))
EMNIST instance: $(@bind inst Slider(1:30, default=1, show_value=true))
synthetic stroke radius: $(@bind srad Slider(2:1:9, default=6, show_value=true))

**Ce (even / line) scale** `w_L`: $(@bind wL Slider(Float32[4,6,8,10,12,15], default=8f0, show_value=true))
**Co (odd / edge) scale** `w_E`: $(@bind wE Slider(Float32[3,4,6,8,10,12], default=8f0, show_value=true))
**δ (probe offset)**: $(@bind delta Slider(Float32[1,2,3,4,6,8,10,13], default=4f0, show_value=true))

κ (sign convention): $(@bind kappa Select([1f0=>"+1 (falling edge)", -1f0=>"−1 (rising edge)"]))
keypoint threshold (× max): $(@bind kthr Slider(0.10f0:0.05f0:0.60f0, default=0.30f0, show_value=true))
invert polarity: $(@bind invert CheckBox(default=false))
"""

# ╔═╡ e0000000-0000-0000-0000-00000000000b
begin
    img_raw = if startswith(src, "EMNIST: ")
        L = replace(src, "EMNIST: "=>"")
        upsample(em.class_images[findfirst(==(L), em.class_names)][inst])
    else
        synth_img(src, srad)
    end
    img = invert ? (1f0 .- img_raw) : img_raw
    Ce, _  = CeCo(img, Lbank(wL))
    _,  Co = CeCo(img, Ebank(wE))
    D = diag_maps(Ce, Co; delta=delta, kappa=kappa)
    kp = kpts(D.Ew, kthr*maximum(D.Ew))
    swid = stroke_width(img_raw)
    md"""
    **source** `$(src)` — measured stroke width **$(round(swid,digits=1)) px** —
    `w_L`=$(wL), `w_E`=$(wE), δ=$(delta) — **$(length(kp)) endpoints detected**
    """
end

# ╔═╡ e0000000-0000-0000-0000-000000000012
let
    sT,sN,lam    = Lparams(wL)
    sT2,sN2,lam2 = Eparams(wE)
    # panels 1-2: the two kernels alone, common canvas (as before)
    KL0 = gabor_kernel(sT,sN,lam,0f0)
    KE0 = gabor_kernel(sT2,sN2,lam2,0f0)
    ttlL = "Re L  w_L=" * string(wL) * "  ks=" * string(size(KL0,1))
    ttlE = "Im Ed w_E=" * string(wE) * "  ks=" * string(size(KE0,1))
    # panel 3: COMBINED, drawn with the stroke VERTICAL so the probe offset is
    # vertical on screen. Line filter sits at p; edge filter sits at p + δ (above).
    # Each is normalised to unit peak first, else the big L kernel swamps Ed.
    KLv = gabor_kernel(sT,sN,lam, Float32(π)/2)      # line filter along a vertical stroke
    KEv = gabor_kernel(sT2,sN2,lam2, 0f0)            # edge filter perpendicular to it
    C = 181; c0 = C÷2 + 1
    canvas = zeros(Float32, C, C)
    A = real.(KLv); A ./= max(maximum(abs,A), 1f-9)
    hL = size(A,1)÷2
    for a in -hL:hL, b in -hL:hL
        y=c0+a; x=c0+b
        (1<=y<=C && 1<=x<=C) && (canvas[y,x] += A[a+hL+1, b+hL+1])
    end
    B = imag.(KEv); B ./= max(maximum(abs,B), 1f-9)
    hE = size(B,1)÷2; oy = c0 - round(Int, delta)    # yflip: smaller row = higher
    for a in -hE:hE, b in -hE:hE
        y=oy+a; x=c0+b
        (1<=y<=C && 1<=x<=C) && (canvas[y,x] += B[a+hE+1, b+hE+1])
    end
    m = max(maximum(abs,canvas), 1f-9)
    p3 = heatmap(canvas; c=:RdBu, clims=(-m,m), titlefontsize=8,
                 title="combined: L at p, Ed at p+δ   (δ=" * string(delta) * ")",
                 aspect_ratio=:equal, axis=false, ticks=false, cbar=false, yflip=true)
    scatter!(p3, [c0], [c0]; mc=:black, msw=0, ms=4, label="")   # p
    scatter!(p3, [c0], [oy]; mc=:lime,  msw=0, ms=4, label="")   # p + δ
    plot!(p3, [c0,c0], [c0,oy]; lc=:lime, lw=2, label="")
    plot(kpanel(on_canvas(real.(KL0)), ttlL),
         kpanel(on_canvas(imag.(KE0)), ttlE),
         p3; layout=(1,3), size=(1050,360))
end

# ╔═╡ e0000000-0000-0000-0000-00000000000c
begin
    kw=(yflip=true, aspect_ratio=:equal, axis=false, ticks=false, cbar=false,
        xlims=(1,IMG), ylims=(1,IMG))
    p1=heatmap(img; c=:grays, title="input + keypoints", kw...)
    isempty(kp) || scatter!(p1,[p[2] for p in kp],[p[1] for p in kp];
                            mc=:cyan, msw=0, ms=5, label="")
    p2=heatmap(D.Lwin; c=:viridis, title="|Ce| at winning φ  (line)", kw...)
    p3=heatmap(D.capw; c=:viridis, title="cap at winning φ  (edge)", kw...)
    p4=heatmap(D.Ew;   c=:magma,   title="Endpoint = min(|Ce|, cap)", kw...)
    p5=heatmap(D.sw;   c=:RdBu, clims=(-1,1), title="s = sign(Ce)", kw...)
    p6=heatmap(D.align; c=:viridis, clims=(0,1),
               title="align: 1=along stroke, 0=across", kw...)
    p7=heatmap(D.phiw; c=:hsv, clims=(0,2π), title="winning direction φ*", kw...)
    p8=heatmap(D.Ldom; c=:viridis, title="max_θ |Ce| (dominant line)", kw...)
    plot(p1,p2,p3,p4,p5,p6,p7,p8; layout=(2,4), size=(1300,660))
end

# ╔═╡ e0000000-0000-0000-0000-00000000000d
begin
    onink = img_raw .> 0.5f0
    fire  = D.Ew .> kthr*maximum(D.Ew)
    both  = onink .& fire
    al_fire = any(fire) ? round(mean(D.align[fire]),digits=2) : 0.0
    al_ink  = any(both) ? round(mean(D.align[both]),digits=2) : 0.0
    md"""
    ### Hypothesis readout

    Mean **align** over all firing pixels: **$(al_fire)** ·
    over firing pixels *on ink*: **$(al_ink)**

    Near **1** ⇒ the winning direction runs **along** the stroke (detector working
    as designed). Near **0** ⇒ it points **across** the stroke, meaning the probe
    is landing on the stroke's own flank rather than on a genuine end-cap — the
    suspected mechanism behind the ~12× over-firing on EMNIST.
    """
end

# ╔═╡ e0000000-0000-0000-0000-00000000000e
md"""
### Notes

- **Independent scales.** `w_L` sets the even/line filter (σ_T = 1.5·w_L so its
  along-line extent is 3·w_L, σ_N = w_L/2, λ = 2·w_L); `w_E` sets the odd/edge
  filter (σ_T = w_E/2, λ = w_E/2). They were locked together in the batch
  experiments — this notebook is the first place they can be varied separately,
  which is the point.
- **Measured EMNIST stroke width is ≈ 13.4 px** (median, upsampled 112×112) — so
  the letters are only ~8 stroke-widths across. The batch sweep found the best
  *synthetic* margin at scale 8 (not at the matched width), and that same scale
  was the *worst* on EMNIST (36.6 endpoints/letter vs ~2.8 expected).
- **Polarity.** All filters are DC-free and every displayed quantity is either a
  magnitude or sign-referenced, so ticking *invert polarity* should leave every
  panel identical (verified to 3·10⁻⁵ relative in batch). `s` itself flips —
  that's the reference that makes `cap` invariant.
- Rebuilding a bank on a new scale takes a moment the first time; banks are
  cached, so returning to a previous scale is instant.
"""

# ╔═╡ Cell order:
# ╠═e0000000-0000-0000-0000-000000000001
# ╠═e0000000-0000-0000-0000-000000000002
# ╠═e0000000-0000-0000-0000-000000000003
# ╟─e0000000-0000-0000-0000-000000000004
# ╠═e0000000-0000-0000-0000-000000000005
# ╠═e0000000-0000-0000-0000-000000000006
# ╟─e0000000-0000-0000-0000-000000000010
# ╠═e0000000-0000-0000-0000-000000000011
# ╠═e0000000-0000-0000-0000-000000000007
# ╠═e0000000-0000-0000-0000-000000000008
# ╠═e0000000-0000-0000-0000-000000000009
# ╟─e0000000-0000-0000-0000-00000000000a
# ╟─e0000000-0000-0000-0000-00000000000b
# ╠═e0000000-0000-0000-0000-000000000012
# ╠═e0000000-0000-0000-0000-00000000000c
# ╟─e0000000-0000-0000-0000-00000000000d
# ╟─e0000000-0000-0000-0000-00000000000e
