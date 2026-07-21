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
end

# ╔═╡ 90000000-0000-0000-0000-000000000003
begin
    include(joinpath(@__DIR__, "..", "LoadEMNIST.jl"))
    using .LoadEMNIST
end

# ╔═╡ 90000000-0000-0000-0000-000000000004
md"""
# Gaussian curvature of the intensity surface — synthetic + EMNIST

Treats the image as a height surface **z = I(x, y)** and shows the **Gaussian
curvature** at a chosen set of scales, on the **same 224×224 padded field** as the
other notebooks (letter 112, border 56).

```
K = ( I_xx · I_yy − I_xy² ) / ( 1 + I_x² + I_y² )²
```

- The **numerator** is `det Hess(I) = I_xx·I_yy − I_xy²` — the sign-carrier.
  `> 0` **elliptic** (a local max or min — a dome or pit);
  `< 0` **hyperbolic** (a **saddle**);
  `≈ 0` **parabolic** (a straight ridge or edge — one principal curvature is zero).
- The **denominator** `(1 + |∇I|²)²` only *attenuates* on steep slopes; it never
  changes the sign. Toggle to **det Hessian** to drop it.
- **Measured caveat** (see Notes): on these binary strokes *almost every* salient
  point — ends, corners, **and** junctions/crossings — is a local **maximum** of the
  smoothed surface, so it reads `K > 0`. The sign does **not** separate ends from
  crossings; it is a blob/keypoint *saliency*, and what differs between feature
  types is **magnitude**. Saddles (`K < 0`) are thin and sit on the background side
  of concavities, not on the strokes.

### Polarity invariance — for free

Every derivative flips sign under inversion `I → c − I` (the derivative kernels are
DC-free, so a constant contributes nothing). `K` is built from **products and
squares** of them — `I_xx·I_yy`, `I_xy²`, `I_x²`, `I_y²` — each of which is
unchanged by a simultaneous sign flip. So `K` is **exactly polarity-invariant**,
with no special construction (unlike the odd-Gabor cap). The `invert` box and the
printed check confirm it.

### Scale

`K` is built from second derivatives, so it is only defined *at a smoothing scale
σ*: what you see is the curvature of the image blurred to scale σ. The five σ are
log-spaced between the two sliders. σ ≈ w/2 … w corresponds to the Gabor `w ∈
{4…15}` we have been using (stroke width ≈ 13 px).
"""

# ╔═╡ 90000000-0000-0000-0000-000000000005
begin
    const IMG0=112                 # the letter patch
    const PAD=56                   # background border (matches the other padded notebooks)
    const IMG=IMG0+2PAD            # 224 working field — all panels are this size
    const CLASSES=["O","C","I","L","T","X","K","A","H","Y","E","F"]
end

# ╔═╡ 90000000-0000-0000-0000-000000000006
# ---- Gaussian-derivative jets and Gaussian curvature ----
begin
    "1D unit-area Gaussian and its 1st/2nd derivatives at scale σ, forced DC-free
     so a constant (any polarity pedestal) contributes exactly zero."
    function dog1d(σ)
        r=max(1,ceil(Int,3.5σ)); xs=Float32.(-r:r)
        g=exp.(-xs.^2 ./ (2σ^2)); g ./= sum(g)
        g1=-(xs ./ σ^2).*g;                 g1 .-= mean(g1)   # ∂ₓGσ  (odd → ~0, made exact)
        g2=((xs.^2 .- σ^2) ./ σ^4).*g;      g2 .-= mean(g2)   # ∂ₓₓGσ (∫=0, made exact)
        g, g1, g2
    end
    # separable 1D convolutions with edge-replicate (clamp) padding — linear, so
    # inversion I→c−I maps every output to its exact negation.
    function conv_cols(A,k)                 # along x (columns / 2nd index)
        H,W=size(A); r=length(k)÷2; out=similar(A)
        @inbounds for y in 1:H, x in 1:W
            s=0f0; for j in -r:r; s+=A[y,clamp(x+j,1,W)]*k[j+r+1]; end; out[y,x]=s
        end; out
    end
    function conv_rows(A,k)                 # along y (rows / 1st index)
        H,W=size(A); r=length(k)÷2; out=similar(A)
        @inbounds for y in 1:H, x in 1:W
            s=0f0; for j in -r:r; s+=A[clamp(y+j,1,H),x]*k[j+r+1]; end; out[y,x]=s
        end; out
    end
    "The five derivative maps (I_x, I_y, I_xx, I_yy, I_xy) at scale σ."
    function jet(img,σ)
        g,g1,g2=dog1d(σ)
        Cg1=conv_cols(img,g1); Cg=conv_cols(img,g); Cg2=conv_cols(img,g2)
        Ix =conv_rows(Cg1,g);  Ixy=conv_rows(Cg1,g1)
        Iy =conv_rows(Cg ,g1); Iyy=conv_rows(Cg ,g2)
        Ixx=conv_rows(Cg2,g)
        Ix,Iy,Ixx,Iyy,Ixy
    end
    "Gaussian curvature (:K) or det Hessian (:detH) at scale σ. `snorm` applies the
     Lindeberg γ-normalisation σ⁴ so magnitudes are comparable across scales."
    function curvature(img,σ; quantity=:K, snorm=false)
        Ix,Iy,Ixx,Iyy,Ixy=jet(img,σ)
        detH=Ixx.*Iyy .- Ixy.^2
        M = quantity==:detH ? detH : detH ./ (1f0 .+ Ix.^2 .+ Iy.^2).^2
        snorm ? M .* Float32(σ)^4 : M
    end
end

# ╔═╡ 90000000-0000-0000-0000-000000000007
# ---- synthetic figures (drawn on the 112 patch, then embedded) + EMNIST ----
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
    function embed(letter)
        out=zeros(Float32,IMG,IMG); out[PAD+1:PAD+IMG0, PAD+1:PAD+IMG0] .= letter; out
    end
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
    function stroke_width(img;t=0.5f0)
        ink=img.>t; area=count(ink); area==0 && return 0f0
        per=0
        for y in 2:IMG-1,x in 2:IMG-1
            ink[y,x]||continue
            (ink[y-1,x]&&ink[y+1,x]&&ink[y,x-1]&&ink[y,x+1])||(per+=1)
        end
        per==0 ? 0f0 : Float32(2*area/per)
    end
    const SYNTH=["bar","plus","T","X","L-shape","O (ring)"]
    em = load_emnist(n_images_to_load=8000, n_classes=47)
end;

# ╔═╡ 90000000-0000-0000-0000-00000000000c
# ---- local extrema of the (signed) curvature field, per scale ----
begin
    const CY = Float32((IMG+1)/2); const CX = Float32((IMG+1)/2)   # centre of the 224 field
    "Strict local maxima AND minima of K with |K| > frac·max|K|, deduped by NMS
     radius r, keeping the strongest topN. Returns a Vector of (y, x, value)."
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
    # polar coords of (y,x) about the image centre: (r in px, α in degrees)
    radial(y,x) = (hypot(Float32(y)-CY, Float32(x)-CX), atand(Float32(y)-CY, Float32(x)-CX))
end

# ╔═╡ 90000000-0000-0000-0000-000000000008
md"""
### Controls

source: $(@bind src Select(vcat(SYNTH, "EMNIST: " .* CLASSES), default="T"))
EMNIST instance: $(@bind inst Slider(1:30, default=1, show_value=true))
synthetic stroke radius: $(@bind srad Slider(2:1:9, default=6, show_value=true))

**σ min**: $(@bind smin Slider(1.0f0:0.5f0:6.0f0, default=2.0f0, show_value=true))
**σ max**: $(@bind smax Slider(4.0f0:1.0f0:24.0f0, default=12.0f0, show_value=true))

quantity: $(@bind quantity Select([:K=>"K  (Gaussian curvature)", :detH=>"det Hessian  (numerator)"]))
scaling across panels: $(@bind scaling Select(["per-panel"=>"per-panel (each visible)", "shared"=>"shared, σ⁴-normalised"]))
clip percentile: $(@bind clip Slider(0.90f0:0.01f0:1.0f0, default=0.995f0, show_value=true))

**local extrema** — threshold (× max|K|) $(@bind ex_frac Slider(0.05f0:0.05f0:0.80f0, default=0.30f0, show_value=true)) · top-N / scale $(@bind ex_topN Slider(1:1:15, default=8, show_value=true)) · min separation $(@bind ex_rad Slider(3:1:15, default=8, show_value=true))
invert polarity: $(@bind invert CheckBox(default=false))
"""

# ╔═╡ 90000000-0000-0000-0000-000000000009
begin
    letter = if startswith(src, "EMNIST: ")
        LN = replace(src, "EMNIST: "=>"")
        upsample(em.class_images[findfirst(==(LN), em.class_names)][inst])
    else
        synth_img(src, srad)
    end
    img_raw = embed(letter)
    img     = invert ? (1f0 .- img_raw) : img_raw
    scales  = Float32.(exp.(range(log(smin), log(max(smax,smin+1f0)), length=5)))
    snorm   = scaling == "shared"
    Ks      = [curvature(img, σ; quantity=quantity, snorm=snorm) for σ in scales]

    # local extrema per scale (list of (y,x,value)), + a flat table for later use
    ex_by_scale = [ local_extrema(K; frac=ex_frac, r=ex_rad, topN=ex_topN) for K in Ks ]
    EXTREMA = [ (scale=round(scales[i],digits=1), pol=(v>=0 ? "+" : "−"), value=v,
                 r=round(hypot(Float32(y)-CY,Float32(x)-CX),digits=1),
                 alpha=round(atand(Float32(y)-CY,Float32(x)-CX),digits=1), y=y, x=x)
                for i in 1:5 for (y,x,v) in ex_by_scale[i] ]

    # polarity check (independent of the invert box): un-inverted vs inverted input
    σc  = scales[3]
    Kp  = curvature(img_raw, σc; quantity=quantity, snorm=snorm)
    Kpi = curvature(1f0 .- img_raw, σc; quantity=quantity, snorm=snorm)
    poldiff = maximum(abs, Kp .- Kpi)
    polrel  = poldiff / max(maximum(abs, Kp), 1f-9)

    swid   = round(stroke_width(img_raw), digits=1)
    sstr   = join([string(round(σ,digits=1)) for σ in scales], ", ")
    qname  = quantity==:K ? "K" : "det Hessian"
    Markdown.parse("**source** `$(src)` — stroke width **$(swid) px** — " *
        "field **$(IMG)×$(IMG)** — quantity **$(qname)** — σ = [ $(sstr) ] — " *
        "**polarity check** max|K−K′| = $(round(poldiff,sigdigits=3)) " *
        "(relative $(round(polrel,sigdigits=3)))")
end

# ╔═╡ 90000000-0000-0000-0000-00000000000d
if isempty(EXTREMA)
    md"**Local curvature extrema:** none above threshold."
else
    ttl = "**Local curvature extrema** — the dots on the panels below. `(r, α)` are " *
          "polar coordinates about the image centre ($(CY), $(CX)); `EXTREMA` holds these rows.\n\n"
    hdr = "| σ | sign | value | r (px) | α (°) | (y, x) |\n|--:|:--:|--:|--:|--:|:--|\n"
    body = join(["| $(e.scale) | $(e.pol) | $(round(e.value,sigdigits=3)) | $(e.r) | $(e.alpha) | ($(e.y), $(e.x)) |"
                 for e in EXTREMA], "\n")
    Markdown.parse(ttl * hdr * body)
end

# ╔═╡ 90000000-0000-0000-0000-00000000000a
let
    kw=(yflip=true, aspect_ratio=:equal, axis=false, ticks=false, cbar=false,
        xlims=(1,IMG), ylims=(1,IMG))
    cmap = cgrad(:RdBu, rev=true)                 # red = positive (elliptic), blue = negative (saddle)
    rmax(M)=(v=quantile(abs.(vec(M)), clip); v==0 ? 1f0 : Float32(v))
    shared_m = rmax(reduce(vcat, vec.(Ks)))
    panels=Any[ heatmap(img; c=:grays, title="input — $(src)", titlefontsize=8, kw...) ]
    for (i,(σ,K)) in enumerate(zip(scales, Ks))
        m = snorm ? shared_m : rmax(K)
        p = heatmap(K; c=cmap, clims=(-m,m),
            title="σ = $(round(σ,digits=1))  ($(length(ex_by_scale[i])) extrema)",
            titlefontsize=8, kw...)
        ex = ex_by_scale[i]                       # yellow = positive (max), cyan = negative (saddle)
        px=[x for (y,x,v) in ex if v>=0]; py=[y for (y,x,v) in ex if v>=0]
        nx=[x for (y,x,v) in ex if v<0 ]; ny=[y for (y,x,v) in ex if v<0 ]
        isempty(px) || scatter!(p, px, py; mc=:yellow, msc=:black, msw=1, ms=5, label="")
        isempty(nx) || scatter!(p, nx, ny; mc=:cyan,   msc=:black, msw=1, ms=5, label="")
        push!(panels, p)
    end
    plot(panels...; layout=(2,3), size=(1200,780))
end

# ╔═╡ 90000000-0000-0000-0000-00000000000b
md"""
### Notes

- **Dots & table.** Each scale panel is overlaid with its own local extrema of `K`,
  found *separately per scale* — **yellow = positive** (elliptic max), **cyan =
  negative** (saddle) — deduped by the *min separation* radius and capped at
  *top-N / scale* (sliders), so there is a handful per panel. The table above lists
  every dot with its value and its polar position `(r, α)` about the image centre;
  the `EXTREMA` variable holds the same rows for downstream use. Raise *threshold*
  or lower *top-N* if a panel is too busy.
- **Reading the colours (measured, σ≈6, values ×1e6).** Red = `K > 0` = a
  dome-like local **maximum** of the smoothed intensity. It lights up at *every*
  salient point — a free **end** (≈ +46), a **junction / crossing** centre
  (≈ +14–18), a **corner / armpit** (≈ +24) — and is essentially the
  determinant-of-Hessian blob/keypoint **saliency**. Near-white = `K ≈ 0` =
  straight strokes and edges (parabolic). Blue = `K < 0` = saddles, which for
  binary strokes are thin and sit on the **background side** of concave outline,
  not on the strokes.
- **The sign does *not* distinguish an endpoint from a crossing.** A smoothed
  crossing is the *brightest* point (its arms pile up), hence a local maximum, not
  a saddle — so ends, junctions and crossings all read `K > 0`. What separates
  feature types here is **magnitude and the scale at which each peaks**, not sign:
  an end is curved in two directions and peaks higher (≈ +46) than a same-width
  junction (≈ +15). If you want a *type* label you would threshold/rank on
  magnitude across scale, not read the sign.
- **det Hessian vs K.** `K` divides by `(1+|∇I|²)²`, which suppresses the response
  along the high-gradient stroke *flanks* and keeps it where the surface is
  locally dome/saddle-like. `det Hessian` (the numerator) is the same sign field
  without that attenuation — often higher-contrast, and it is the classic
  determinant-of-Hessian blob/keypoint operator. Both are polarity-invariant.
- **Scaling across panels.** *per-panel* rescales every heatmap to its own robust
  range so all five σ are visible; *shared, σ⁴-normalised* applies Lindeberg
  γ-normalisation and one common colour range, so you can compare magnitudes and
  watch a feature peak at its natural scale.
- **Polarity.** The printed `max|K−K′|` compares the curvature of the letter with
  that of its exact intensity-inverse; it should be at float round-off. Ticking
  *invert* leaves every panel visually identical — that is the whole point, and it
  needs no calibration constant.
- **Scale vs the Gabors.** σ is an *isotropic* smoothing scale; the Gabor `w`
  carried separate along/across extents. As a rough guide σ ≈ w/2 matches the
  Gabor cross-section, σ ≈ w the wavelength; the default 2 … 12 spans the `w ∈
  {4…15}` range against a ≈13 px stroke.
- **No chirality.** Unlike the odd-Gabor cap, `K` is built from the isotropic
  Hessian, so it is exactly reflection- and rotation-covariant — none of the
  perp-Gabor sign bookkeeping that produced the mirror asymmetry in
  `EndpointDiagnosticsPadded.jl`.
"""

# ╔═╡ Cell order:
# ╠═90000000-0000-0000-0000-000000000001
# ╠═90000000-0000-0000-0000-000000000002
# ╠═90000000-0000-0000-0000-000000000003
# ╟─90000000-0000-0000-0000-000000000004
# ╠═90000000-0000-0000-0000-000000000005
# ╠═90000000-0000-0000-0000-000000000006
# ╠═90000000-0000-0000-0000-000000000007
# ╠═90000000-0000-0000-0000-00000000000c
# ╟─90000000-0000-0000-0000-000000000008
# ╠═90000000-0000-0000-0000-000000000009
# ╟─90000000-0000-0000-0000-00000000000d
# ╠═90000000-0000-0000-0000-00000000000a
# ╟─90000000-0000-0000-0000-00000000000b
