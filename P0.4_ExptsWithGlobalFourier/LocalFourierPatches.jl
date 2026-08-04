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
    using FFTW
end

# ╔═╡ 90000000-0000-0000-0000-000000000003
begin
    include(joinpath(@__DIR__, "..", "LoadEMNIST.module.jl"))
    using .LoadEMNIST
end

# ╔═╡ 90000000-0000-0000-0000-000000000004
md"""
# The first few 2D Fourier coefficients of a small local patch

What does a **local** patch of a character look like in the **lowest few 2D Fourier
coefficients** — and how does that change with patch **size** (5–40 px) and patch
**location**?

Same image format as the rest of the project: the EMNIST 28×28 sample is upsampled
to a 112×112 letter and embedded in a **224×224 field** (border 56), or a synthetic
figure is drawn on the same 112 patch. Stroke width ≈ 13 px, so patch size 5 …
40 px runs from *well inside a stroke* to *several strokes across*.

### What is computed

A P×P patch `p` is cut at the chosen centre, multiplied by a window `w`, and its
2D DFT is evaluated at the low orders `(v, u) ∈ [−K, K]²`:

```
F(v, u) = (1/Σw) · Σ_{y,x}  w[y,x] · p[y,x] · exp( −2πi ( v(y−1) + u(x−1) ) / P )
```

Order `u` means **u cycles across the patch**: spatial frequency `u/P` cycles/px,
wavelength `P/u` px. The `1/Σw` normalisation makes `F(0,0)` the windowed **mean
intensity** (∈ [0,1]), so magnitudes are comparable across patch sizes and windows.

Orders are clamped to `K ≤ (P−1)÷2` — above that the DFT of a P-point patch
**aliases** (order `P−1` *is* order `−1`), so there are no further coefficients to
look at; the printed `K_eff` shows what was actually used.

### Why the window matters

With `w = ` **box** this is the plain DFT of a cropped square, and the crop's own
edge discontinuity dominates the low orders — you mostly measure the *cut*, not the
letter. With **Hann** or **Gaussian**, `F(v,u)` is literally the inner product of
the image with a **windowed complex sinusoid** — i.e. a **Gabor filter** at
frequency `(u, v)/P` and phase 0/90°. So *"the first few Fourier coefficients of a
local patch"* and *"a Gabor jet at this point"* are the same object; the order grid
`(v,u)` is the polar-frequency grid `(ω, θ)` in Cartesian form. That is the bridge
between this notebook and the Gabor front end used elsewhere in the project.

`|F(v,u)|` is invariant to a **shift of the patch content** (shift theorem: the
shift only rotates the phase) — but only for shifts *within* an unwindowed patch;
with a window, moving content past the window's falloff does change `|F|`.
"""

# ╔═╡ 90000000-0000-0000-0000-000000000005
begin
    const IMG0 = 112                # the letter patch
    const PAD  = 56                 # background border (matches the other padded notebooks)
    const IMG  = IMG0 + 2PAD        # 224 working field
    const CLASSES = ["O","C","I","L","T","X","K","A","H","Y","E","F"]
end

# ╔═╡ 90000000-0000-0000-0000-000000000006
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
    function drawbar!(img,y0,x0,y1,x1,r)
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
        name=="bar"     && return drawbar!(z(),56,26,56,86,r)
        name=="plus"    && return (i=z();drawbar!(i,26,56,86,56,r);drawbar!(i,56,26,56,86,r))
        name=="T"       && return (i=z();drawbar!(i,36,26,36,86,r);drawbar!(i,36,56,86,56,r))
        name=="X"       && return (i=z();drawbar!(i,26,26,86,86,r);drawbar!(i,26,86,86,26,r))
        name=="L-shape" && return (i=z();drawbar!(i,26,40,80,40,r);drawbar!(i,80,40,80,86,r))
        name=="diagonal"&& return drawbar!(z(),26,26,86,86,r)
        return ring!(z(),56,56,28,r)
    end
    const SYNTH=["bar","diagonal","plus","T","X","L-shape","O (ring)"]
    em = load_emnist(n_images_to_load=8000, n_classes=47)
end;

# ╔═╡ 90000000-0000-0000-0000-000000000007
# ---- patch extraction, windows, low-order 2D DFT ----
begin
    "P×P patch centred on (cy,cx), edge-replicate at the field border."
    function patch_at(img, cy::Int, cx::Int, P::Int)
        out = zeros(Float32, P, P)
        y0 = cy - (P - 1) ÷ 2; x0 = cx - (P - 1) ÷ 2
        @inbounds for i in 1:P, j in 1:P
            out[i,j] = img[clamp(y0+i-1,1,IMG), clamp(x0+j-1,1,IMG)]
        end
        out
    end

    "Separable analysis window: box, Hann, or Gaussian (σ = P/4)."
    function patch_window(P::Int, kind::AbstractString)
        kind == "box" && return ones(Float32, P, P)
        if kind == "hann"
            h = P == 1 ? Float32[1] : Float32[0.5f0*(1-cos(2π*(i-1)/(P-1))) for i in 1:P]
            h = max.(h, 1f-3)                       # keep Σw > 0 for tiny P
            return h * h'
        end
        σ = Float32(P)/4; c = Float32((P+1)/2)
        return Float32[exp(-((i-c)^2+(j-c)^2)/(2σ^2)) for i in 1:P, j in 1:P]
    end

    "Highest non-aliasing order for a P-point DFT."
    max_order(P::Int) = (P - 1) ÷ 2

    "Complex-exponential basis: B[k, n] = exp(-2πi·kₙ·(n−1)/P), rows k ∈ −K:K."
    function dft_basis(P::Int, K::Int)
        ComplexF32[cis(-2f0π*Float32(k)*Float32(n-1)/Float32(P)) for k in -K:K, n in 1:P]
    end

    """
        low_coeffs(p, w, K)

    `F[v, u]` for `v, u ∈ −K:K` (rows = vertical order, cols = horizontal order),
    normalised so `F[0,0]` is the windowed mean intensity.
    """
    function low_coeffs(p::AbstractMatrix, w::AbstractMatrix, K::Int)
        P = size(p, 1); B = dft_basis(P, K)
        (B * ComplexF32.(w .* p) * transpose(B)) ./ Float32(sum(w))
    end

    "Band-limited reconstruction of the *windowed* patch from `F` alone."
    function reconstruct(F::AbstractMatrix, P::Int, K::Int, sumw::Real)
        E = ComplexF32[cis(2f0π*Float32(k)*Float32(n-1)/Float32(P)) for k in -K:K, n in 1:P]
        real.(transpose(E) * F * E) .* Float32(sumw / P^2)
    end

    "Fraction of the windowed patch's AC energy that lives in orders |v|,|u| ≤ K."
    function energy_fraction(p::AbstractMatrix, w::AbstractMatrix, K::Int)
        P = size(p, 1); Ff = fft(ComplexF32.(w .* p))
        tot = sum(abs2, Ff) - abs2(Ff[1,1])
        tot <= 0 && return 0f0
        kept = 0f0
        for v in -K:K, u in -K:K
            (v == 0 && u == 0) && continue
            kept += abs2(Ff[mod(v,P)+1, mod(u,P)+1])
        end
        Float32(kept / tot)
    end

    "Unique orders up to K (|F| is conjugate-symmetric for real input), DC first."
    function unique_orders(K::Int)
        out = [(0,0)]
        for v in 0:K, u in -K:K
            (v == 0 && u <= 0) && continue
            push!(out, (v,u))
        end
        out
    end
end

# ╔═╡ 90000000-0000-0000-0000-000000000008
md"""
### Controls

source: $(@bind src Select(vcat(SYNTH, "EMNIST: " .* CLASSES), default="T"))
EMNIST instance: $(@bind inst Slider(1:30, default=1, show_value=true))
synthetic stroke radius: $(@bind srad Slider(2:1:9, default=6, show_value=true))

**patch size P**: $(@bind P Slider(5:1:40, default=24, show_value=true))
**centre y**: $(@bind cy Slider(1:1:IMG, default=92, show_value=true))
**centre x**: $(@bind cx Slider(1:1:IMG, default=112, show_value=true))

**max order K**: $(@bind Kreq Slider(1:1:6, default=3, show_value=true))
window: $(@bind wkind Select(["hann"=>"Hann (⇒ Gabor jet)", "gauss"=>"Gaussian σ=P/4", "box"=>"box (raw crop)"]))
remove DC before transforming: $(@bind killdc CheckBox(default=false))
invert polarity: $(@bind invert CheckBox(default=false))

**location scan** — stride $(@bind stride Slider(2:1:16, default=6, show_value=true)) · orders shown up to $(@bind Kscan Slider(1:1:3, default=2, show_value=true))
"""

# ╔═╡ 90000000-0000-0000-0000-000000000009
begin
    letter = if startswith(src, "EMNIST: ")
        LN = replace(src, "EMNIST: " => "")
        upsample(em.class_images[findfirst(==(LN), em.class_names)][inst])
    else
        synth_img(src, srad)
    end
    img_raw = embed(letter)
    img     = invert ? (1f0 .- img_raw) : img_raw

    K    = min(Kreq, max_order(P))
    pat  = patch_at(img, cy, cx, P)
    win  = patch_window(P, wkind)
    pat_c = killdc ? pat .- (sum(win .* pat) / sum(win)) : pat
    F    = low_coeffs(pat_c, win, K)
    rec  = reconstruct(F, P, K, sum(win))
    efrac = energy_fraction(pat_c, win, K)

    # polarity check (independent of the invert box): AC magnitudes of the patch vs
    # those of its exact intensity-inverse. Exactly 0 only for the box window — see Notes.
    pat_r  = patch_at(img_raw, cy, cx, P)
    Fa     = low_coeffs(pat_r, win, K); Fb = low_coeffs(1f0 .- pat_r, win, K)
    acmask = trues(2K+1, 2K+1); acmask[K+1, K+1] = false
    poldiff = maximum(abs.(abs.(Fa) .- abs.(Fb))[acmask])
    polrel  = poldiff / max(maximum(abs.(Fa)[acmask]), 1f-9)

    Markdown.parse("**source** `$(src)` — patch **$(P)×$(P)** at **(y=$(cy), x=$(cx))** — " *
        "window **$(wkind)** — **K_eff = $(K)**" * (K < Kreq ? " (clamped from $(Kreq): P allows ≤ $(max_order(P)))" : "") *
        " — $((2K+1)^2) coefficients (**$(length(unique_orders(K)))** unique up to conjugation) — " *
        "they capture **$(round(100*efrac, digits=1)) %** of the windowed patch's AC energy — " *
        "finest wavelength **$(round(P/max(K,1), digits=1)) px** — " *
        "**AC polarity check** max‖F|−|F′‖ = $(round(poldiff, sigdigits=3)) " *
        "(relative $(round(polrel, sigdigits=3)))")
end

# ╔═╡ 90000000-0000-0000-0000-00000000000a
let
    kw = (yflip=true, aspect_ratio=:equal, axis=false, ticks=false, cbar=false)
    half = (P-1)÷2
    pf = heatmap(img; c=:grays, title="field 224 — patch box", titlefontsize=8,
                 xlims=(1,IMG), ylims=(1,IMG), kw...)
    bx = [cx-half, cx+P-1-half, cx+P-1-half, cx-half, cx-half]
    by = [cy-half, cy-half, cy+P-1-half, cy+P-1-half, cy-half]
    plot!(pf, bx, by; lc=:red, lw=2, label="")
    scatter!(pf, [cx], [cy]; mc=:red, ms=3, msw=0, label="")

    pkw = (yflip=true, aspect_ratio=:equal, axis=false, ticks=false, cbar=false,
           xlims=(0.5,P+0.5), ylims=(0.5,P+0.5))
    p1 = heatmap(pat;        c=:grays, title="patch $(P)×$(P)", titlefontsize=8, pkw...)
    p2 = heatmap(win .* pat_c; c=:grays, title="window × patch", titlefontsize=8, pkw...)
    p3 = heatmap(rec;        c=:grays, title="reconstruction from K=$(K)", titlefontsize=8, pkw...)

    ords = collect(-K:K)
    mag  = abs.(F)
    p4 = heatmap(ords, ords, mag; c=:viridis, yflip=false, aspect_ratio=:equal,
                 title="|F(v,u)|", titlefontsize=8, xlabel="u (cycles across)",
                 ylabel="v", xticks=ords, yticks=ords, guidefontsize=7, tickfontsize=6)
    cyc = cgrad([:red,:yellow,:green,:cyan,:blue,:magenta,:red])
    ph  = angle.(F); ph[mag .< 0.02f0*maximum(mag)] .= NaN     # phase is meaningless where |F|≈0
    p5 = heatmap(ords, ords, ph; c=cyc, clims=(-π,π), yflip=false, aspect_ratio=:equal,
                 title="arg F(v,u)  (grey = |F|≈0)", titlefontsize=8,
                 xlabel="u", ylabel="v", xticks=ords, yticks=ords,
                 guidefontsize=7, tickfontsize=6, background_color_inside=:lightgrey)

    plot(pf, p1, p2, p3, p4, p5; layout=(2,3), size=(1200,760))
end

# ╔═╡ 90000000-0000-0000-0000-00000000000b
let
    uo = unique_orders(min(K,3))
    hdr = "**The first few coefficients** (`u` = cycles across the patch horizontally, " *
          "`v` vertically; only orders unique up to `F(−v,−u) = conj F(v,u)` are listed).\n\n" *
          "| (v, u) | wavelength (px) | wave direction k | stroke it prefers (⊥ k) | \\|F\\| | arg F (°) | \\|F\\|/\\|F₀₀\\| |\n" *
          "|:--|--:|--:|--:|--:|--:|--:|\n"
    f00 = max(abs(F[K+1, K+1]), 1f-9)
    rows = String[]
    for (v,u) in uo
        z = F[v+K+1, u+K+1]
        λ = (v==0 && u==0) ? "∞ (DC)" : string(round(P/hypot(v,u), digits=1))
        θ = (v==0 && u==0) ? "—" : string(round(atand(v,u), digits=0), "°")
        θs = (v==0 && u==0) ? "—" : string(round(Int, mod(atand(v,u)+90, 180)), "°")
        push!(rows, "| ($(v), $(u)) | $(λ) | $(θ) | $(θs) | $(round(abs(z), sigdigits=3)) | " *
                    "$(round(rad2deg(angle(z)), digits=0)) | $(round(abs(z)/f00, sigdigits=3)) |")
    end
    Markdown.parse(hdr * join(rows, "\n"))
end

# ╔═╡ 90000000-0000-0000-0000-00000000000c
md"""
### Patch size sweep at the same centre

Same location, patch size 5 → 40, drawn at true relative scale (each patch centred
in a common 40-px box). Top row: the raw patch. Bottom row: what the first `K`
orders retain, with the AC-energy fraction they capture.

Watch the **bottom** row, not the percentage: the fraction stays high at every size
(see Notes), but the *absolute* resolution `λ = P/K` px gets coarser as `P` grows,
so the large patches turn into a smudge that merges neighbouring strokes.
"""

# ╔═╡ 90000000-0000-0000-0000-00000000000d
let
    sizes = [5, 9, 13, 19, 25, 32, 40]
    kw = (yflip=true, aspect_ratio=:equal, axis=false, ticks=false, cbar=false)
    tops = Any[]; bots = Any[]; fr = Float32[]
    for Ps in sizes
        Ks = min(Kreq, max_order(Ps))
        p  = patch_at(img, cy, cx, Ps)
        w  = patch_window(Ps, wkind)
        pc = killdc ? p .- (sum(w .* p)/sum(w)) : p
        Fs = low_coeffs(pc, w, Ks)
        rs = reconstruct(Fs, Ps, Ks, sum(w))
        e  = energy_fraction(pc, w, Ks); push!(fr, e)
        lim = (0.5, 40.5)                             # common axes ⇒ true relative scale
        ax  = (41 - Ps)/2 .+ (1:Ps)                   # centred inside the common 40-px box
        push!(tops, heatmap(ax, ax, p; c=:grays, title="P=$(Ps)", titlefontsize=8,
                            xlims=lim, ylims=lim, kw...))
        push!(bots, heatmap(ax, ax, rs; c=:grays, title="K=$(Ks) · $(round(100e, digits=0))%",
                            titlefontsize=8, xlims=lim, ylims=lim, kw...))
    end
    plot(vcat(tops, bots)...; layout=(2, length(sizes)), size=(1200, 380))
end

# ╔═╡ 90000000-0000-0000-0000-00000000000e
md"""
### The same coefficient, everywhere: sliding-patch maps

The patch is slid over the whole field on a `stride` grid and one map is drawn per
order: `|F(v,u)|` at every location, at the **current patch size P**. This is what
a low-order local Fourier *front end* would hand to the next stage — and with the
Hann/Gaussian window each map is exactly the modulus of a **Gabor channel** at
frequency `(u,v)/P`.
"""

# ╔═╡ 90000000-0000-0000-0000-00000000000f
let
    Ks   = min(Kscan, K)
    ys   = 1:stride:IMG; xs = 1:stride:IMG
    uo   = unique_orders(Ks)
    w    = patch_window(P, wkind)
    maps = [zeros(Float32, length(ys), length(xs)) for _ in uo]
    for (iy, yy) in enumerate(ys), (ix, xx) in enumerate(xs)
        p  = patch_at(img, yy, xx, P)
        pc = killdc ? p .- (sum(w .* p)/sum(w)) : p
        Fl = low_coeffs(pc, w, Ks)
        for (k,(v,u)) in enumerate(uo)
            maps[k][iy,ix] = abs(Fl[v+Ks+1, u+Ks+1])
        end
    end
    kw = (yflip=true, aspect_ratio=:equal, axis=false, ticks=false, cbar=false)
    panels = Any[heatmap(img; c=:grays, title="input", titlefontsize=8,
                         xlims=(1,IMG), ylims=(1,IMG), kw...)]
    for (k,(v,u)) in enumerate(uo)
        ttl = (v==0 && u==0) ? "|F(0,0)| — local mean" :
              "|F($(v),$(u))|  λ=$(round(P/hypot(v,u), digits=1))px  stroke $(round(Int, mod(atand(v,u)+90, 180)))°"
        push!(panels, heatmap(maps[k]; c=:viridis, title=ttl, titlefontsize=7, kw...))
    end
    ncol = 5; nrow = ceil(Int, length(panels)/ncol)
    plot(panels...; layout=(nrow, ncol), size=(1250, 250nrow))
end

# ╔═╡ 90000000-0000-0000-0000-000000000010
md"""
### Notes

- **Read the order grid as orientation × frequency.** The coefficient `(v,u)` is a
  plane wave with wavelength `P/√(u²+v²)` px travelling at `atan(v,u)`. Radius in
  the `(v,u)` grid = spatial frequency; angle = orientation. `|F|` on the grid is
  therefore a polar frequency signature of the patch: a *straight stroke* puts its
  energy on a **line through the origin perpendicular to the stroke**; a *corner or
  crossing* puts energy on **two such lines**; a *blob or stroke end* spreads energy
  isotropically; a patch **inside** a stroke is nearly pure DC.
- **DC is the local ink fraction.** `F(0,0)` is the windowed mean, so it tracks how
  much stroke is in the patch. Tick *remove DC* to see the structure without it —
  the other coefficients are unchanged (DC removal only zeroes `F(0,0)`), but the
  reconstruction panel then shows the zero-mean part alone.
- **Patch size is the whole experiment — but energy fraction is the wrong thing to
  watch.** Measured on an EMNIST `K` (Hann, `K = 3`, averaged over nine nearby
  centres), the AC energy captured by orders `≤ 3` is **78 / 74 / 82 / 92 / 93 / 94 /
  96 %** at `P = 5 / 9 / 13 / 19 / 25 / 32 / 40`. It *rises* with `P`: a bigger patch
  of a blurred letter is *relatively* smoother, so a fixed number of orders keeps up.
  What actually degrades is **absolute resolution** — `K` orders on a `P`-px patch
  resolve at best `λ = P/K` px, so at `P = 40, K = 3` the finest describable detail is
  13 px = one whole stroke width, and the reconstruction merges neighbouring strokes
  into a smudge (bottom row of the sweep). At `P ≈ 5–9`, below the 13 px stroke, the
  patch is a flat interior or a single edge — cheap to describe because there is
  almost nothing there. The useful band for a local descriptor is `P ≈` **1–2 stroke
  widths (13–26 px)**, where a handful of orders still resolves `λ ≈ 4–9 px` and the
  coefficients carry real structure: stroke orientation, an end, a corner angle.
- **Window choice is not cosmetic.** With `box`, the low orders are dominated by the
  crop's own edge — two patches with identical content but different surroundings
  can look alike, and a uniform patch that happens to sit at the border rings. Hann
  and Gaussian kill that, at the cost of only seeing the middle of the patch. Since
  a windowed low-order coefficient **is** a Gabor filter response, this notebook and
  `CreateGaborLifting.module.jl` compute the same thing from opposite ends: a Gabor bank
  fixes `(ω, θ)` and slides; here we fix the patch and read the whole `(v,u)` grid
  at once.
- **What is invariant, and what isn't.** `|F|` is invariant to shifting the content
  inside an *unwindowed* patch (shift theorem; verified to 4e-8). **Polarity is more
  subtle than it looks**: under `I → 1 − I`, `F → W − F` where `W` is the transform
  of the *window itself*. For the **box** window `W` is an exact delta at DC, so every
  AC `|F|` is exactly polarity-invariant (the printed check reads 0). For **Hann** and
  **Gaussian** it is not: measured at `P = 21`, `|W|` relative to its DC value is
  **0.54** (Hann) / **0.34** (Gaussian) at order ±1, but only **0.019 / 0.011** at
  order ±2 and ≤ 0.007 at ±3. So the first-order coefficients of a windowed patch
  are substantially polarity-*dependent* and orders ≥ 2 are polarity-invariant for
  practical purposes — the printed **AC polarity check** shows the actual number for
  the current settings. This is the local-patch counterpart of the polarity problem the
  odd-Gabor cap detector had; `GaussianCurvatureEMNIST.jl` gets exact invariance
  instead by using only products of DC-free derivatives.
  `|F|` is also **not** rotation-invariant: rotating the letter rotates
  the whole `(v,u)` grid. Making that explicit is the point of the polar/harmonic
  encodings in `P0.3_New_Gabor_FPE/` — take `|F|` on rings of constant `√(u²+v²)` and
  Fourier-transform *around* the ring and you get rotation-invariant magnitudes, the
  same construction as the ray harmonics `cₙ` one level down.
- **Aliasing guard.** `K` is clamped to `(P−1)÷2`. At `P = 5` only orders `−2…2`
  exist, so a 5-px patch has exactly **13 unique** complex numbers in it (25 real
  numbers = the 25 pixels, as it must be).
"""

# ╔═╡ Cell order:
# ╠═90000000-0000-0000-0000-000000000001
# ╠═90000000-0000-0000-0000-000000000002
# ╠═90000000-0000-0000-0000-000000000003
# ╟─90000000-0000-0000-0000-000000000004
# ╠═90000000-0000-0000-0000-000000000005
# ╠═90000000-0000-0000-0000-000000000006
# ╠═90000000-0000-0000-0000-000000000007
# ╟─90000000-0000-0000-0000-000000000008
# ╠═90000000-0000-0000-0000-000000000009
# ╠═90000000-0000-0000-0000-00000000000a
# ╟─90000000-0000-0000-0000-00000000000b
# ╟─90000000-0000-0000-0000-00000000000c
# ╠═90000000-0000-0000-0000-00000000000d
# ╟─90000000-0000-0000-0000-00000000000e
# ╠═90000000-0000-0000-0000-00000000000f
# ╟─90000000-0000-0000-0000-000000000010
