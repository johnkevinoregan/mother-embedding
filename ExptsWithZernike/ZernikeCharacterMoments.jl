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
end

# ╔═╡ 90000000-0000-0000-0000-000000000003
begin
    include(joinpath(@__DIR__, "..", "LoadEMNIST.jl"))
    using .LoadEMNIST
end

# ╔═╡ 90000000-0000-0000-0000-000000000004
md"""
# Zernike moments of a zero-DC character

One **global** Zernike description of the whole character — the disc-supported
counterpart of `ExptsWithGlobalFourier/`. Same image format as the rest of the
project: EMNIST 28×28 upsampled to a 112×112 letter (stroke ≈ 13 px), or a synthetic
figure on the same frame.

### The moments

Zernike polynomials are orthogonal on the unit disc:

```
V_nm(ρ, θ) = R_n^{|m|}(ρ) · e^{imθ},     ρ ≤ 1,  n ≥ 0,  |m| ≤ n,  n − |m| even
A_nm = (n+1)/π · ∫∫_{ρ≤1} f(ρ,θ) · R_n^{|m|}(ρ) · e^{−imθ} dA
```

Read the two indices separately: **`m` is angular frequency** — how the shape varies
*around* the disc, the same role `θ` played in the Fourier notebook — and **`n` is
radial order**, how it varies *outward*. Since `f` is real, `A_{n,−m} = conj(A_nm)`,
so only `m ≥ 0` is kept. Order `n ≤ n_max` gives a handful of numbers: `n_max = 8` →
**25 moments**.

### Why Zernike rather than a Fourier grid

A rotation by `α` maps `A_nm → A_nm · e^{−imα}`. So **`|A_nm|` is rotation-invariant
by construction** — the thing `|F(v,u)|` on a square DFT lattice was *not*. Measured
below: `|A|` moves by ≤ 1.8 % under a 15° or 45° rotation (and exactly 0 at 90°, a
pixel-grid symmetry), while the complex moments move by 77–170 %.

That invariance turns out to **cost accuracy** on letters, which is the interesting
result — see the Notes.

### Zero-DC

The in-disc mean is subtracted before the moments are taken (`zero-DC` box, on by
default), so `A₀₀ = 0` and the description is about *structure*, not how much ink
there is. Measured effect: about **1 point** of classification accuracy, i.e. the
overall ink level carries little identity information on its own.
"""

# ╔═╡ 90000000-0000-0000-0000-000000000005
begin
    const IMG = 112
    const CLASSES = ["O","C","I","L","T","X","K","A","H","Y","E","F"]
end

# ╔═╡ 90000000-0000-0000-0000-000000000006
# ---- images: synthetic figures + EMNIST, plus rotation and disc fitting ----
begin
    @inline function bilinear(M,y,x)
        H,W=size(M); (y<1||x<1||y>H||x>W)&&return 0f0
        y0,x0=floor(Int,y),floor(Int,x); y1,x1=min(y0+1,H),min(x0+1,W); fy,fx=y-y0,x-x0
        (1-fy)*(1-fx)*M[y0,x0]+fy*(1-fx)*M[y1,x0]+(1-fy)*fx*M[y0,x1]+fy*fx*M[y1,x1]
    end
    function upsample(img, N=IMG)
        H,W=size(img); out=zeros(Float32,N,N)
        for i in 1:N,j in 1:N
            out[i,j]=bilinear(img,Float32(1+(i-1)*(H-1)/(N-1)),Float32(1+(j-1)*(W-1)/(N-1)))
        end; out
    end
    function drawbar!(img,y0,x0,y1,x1,r)
        N=size(img,1); n=round(Int,hypot(y1-y0,x1-x0))*2
        for t in range(0,1,length=n)
            yc=y0+t*(y1-y0); xc=x0+t*(x1-x0)
            for dy in -r:r,dx in -r:r
                hypot(dy,dx)<=r && (img[clamp(round(Int,yc+dy),1,N),clamp(round(Int,xc+dx),1,N)]=1f0)
            end
        end; img
    end
    function ring!(img,cy,cx,R,r)
        N=size(img,1)
        for t in range(0,2π,length=1200)
            yc=cy+R*sin(t); xc=cx+R*cos(t)
            for dy in -r:r,dx in -r:r
                hypot(dy,dx)<=r && (img[clamp(round(Int,yc+dy),1,N),clamp(round(Int,xc+dx),1,N)]=1f0)
            end
        end; img
    end
    function disc!(img,cy,cx,R)
        N=size(img,1); for y in 1:N, x in 1:N; hypot(y-cy,x-cx)<=R && (img[y,x]=1f0); end; img
    end
    z()=zeros(Float32,IMG,IMG)
    function synth_img(name,r)
        name=="vert bar"  && return drawbar!(z(),20,56,92,56,r)
        name=="horz bar"  && return drawbar!(z(),56,20,56,92,r)
        name=="diag /"    && return drawbar!(z(),92,20,20,92,r)
        name=="plus"      && return (i=z();drawbar!(i,20,56,92,56,r);drawbar!(i,56,20,56,92,r))
        name=="X"         && return (i=z();drawbar!(i,20,20,92,92,r);drawbar!(i,20,92,92,20,r))
        name=="T"         && return (i=z();drawbar!(i,36,20,36,92,r);drawbar!(i,36,56,92,56,r))
        name=="L-shape"   && return (i=z();drawbar!(i,26,40,80,40,r);drawbar!(i,80,40,80,86,r))
        name=="disc"      && return disc!(z(),56,56,30)
        name=="3-fold"    && return (i=z(); for a in (90,210,330); drawbar!(i,56,56,56+36sind(a),56+36cosd(a),r); end; i)
        return ring!(z(),56,56,30,r)      # "ring"
    end
    const SYNTH=["vert bar","horz bar","diag /","plus","X","T","L-shape","disc","ring","3-fold"]

    "Rotate about the frame centre by α degrees (bilinear, background 0)."
    function rotate_img(img, α)
        N=size(img,1); c=(N+1)/2; s,co=sind(α),cosd(α); out=zeros(Float32,N,N)
        for i in 1:N, j in 1:N
            y=i-c; x=j-c
            out[i,j]=bilinear(img, co*y+s*x+c, -s*y+co*x+c)
        end; out
    end

    """
    Resample so the **ink** fills the unit disc: centroid to the frame centre, and the
    `q`-quantile of ink radii scaled to the inscribed radius. `q = 1` uses the farthest
    ink pixel; `q < 1` ignores that tail (a single stray pixel would otherwise shrink
    everything). Pass `nothing` to leave the frame alone.
    """
    function fit_disc(img; q=0.98)
        q === nothing && return img
        N=size(img,1); c=(N+1)/2; w=Float64.(img); tot=sum(w); tot<=0 && return img
        cy=sum(w[i,j]*i for i in 1:N, j in 1:N)/tot
        cx=sum(w[i,j]*j for i in 1:N, j in 1:N)/tot
        ds=[hypot(i-cy,j-cx) for i in 1:N, j in 1:N if img[i,j] > 0.25f0]
        isempty(ds) && return img
        R = q >= 1 ? maximum(ds) : quantile(ds, q); R<=0 && return img
        s = R/(N/2); out=zeros(Float32,N,N)
        for i in 1:N, j in 1:N; out[i,j]=bilinear(img, cy+(i-c)*s, cx+(j-c)*s); end
        out
    end
    em = load_emnist(n_images_to_load=20000, n_classes=47)
end;

# ╔═╡ 90000000-0000-0000-0000-000000000007
# ---- Zernike core ----
begin
    "Radial polynomial R_n^{|m|}(ρ)."
    function zrad(n::Int, m::Int, ρ::Float64)
        m=abs(m); ((n-m) % 2 != 0 || m > n) && return 0.0
        s=0.0
        for k in 0:((n-m)÷2)
            s += (-1)^k * factorial(n-k) /
                 (factorial(k)*factorial((n+m)÷2-k)*factorial((n-m)÷2-k)) * ρ^(n-2k)
        end
        s
    end

    "All (n,m) with n ≤ nmax, m ≥ 0, n−m even. (m<0 is the conjugate.)"
    zorders(nmax) = [(n,m) for n in 0:nmax for m in 0:n if (n-m) % 2 == 0]

    "Disc mask inscribed in an S×S frame."
    zmask(S::Int) = (c=(S+1)/2; [hypot((i-c)/(S/2),(j-c)/(S/2)) <= 1 for i in 1:S, j in 1:S])

    "Analysis stack: `A_nm = sum(B[k] .* f)` with the (n+1)/π and pixel-area factors folded in."
    function zbasis(S::Int, nmax::Int)
        ords=zorders(nmax); c=(S+1)/2; Rd=S/2; dA=1/Rd^2
        B=[zeros(ComplexF64,S,S) for _ in ords]
        for i in 1:S, j in 1:S
            y=(i-c)/Rd; x=(j-c)/Rd; ρ=hypot(y,x); ρ>1 && continue
            θ=atan(y,x)
            for (k,(n,m)) in enumerate(ords)
                B[k][i,j] = (n+1)/π * zrad(n,m,ρ) * cis(-m*θ) * dA
            end
        end
        B, ords
    end

    "Moments of `img`; `zerodc` subtracts the in-disc mean first (⇒ A₀₀ = 0)."
    function zmoments(img, B, mask; zerodc=true)
        f = Float64.(img)
        if zerodc
            f = f .- mean(f[mask]); f[.!mask] .= 0.0
        end
        [sum(b .* f) for b in B]
    end

    "The zero-DC, disc-masked field the moments actually describe."
    function zfield(img, mask; zerodc=true)
        f = Float64.(img)
        zerodc && (f = f .- mean(f[mask]))
        f[.!mask] .= 0.0; f
    end

    "Band-limited reconstruction from `A` (m>0 terms count twice: A_{n,−m} = conj A_nm)."
    function zreconstruct(A, ords, S)
        c=(S+1)/2; Rd=S/2; out=zeros(Float64,S,S)
        for i in 1:S, j in 1:S
            y=(i-c)/Rd; x=(j-c)/Rd; ρ=hypot(y,x); ρ>1 && continue
            θ=atan(y,x); s=0.0
            for (k,(n,m)) in enumerate(ords)
                v=zrad(n,m,ρ)
                s += m==0 ? real(A[k])*v : 2*real(A[k]*cis(m*θ))*v
            end
            out[i,j]=s
        end
        out
    end

    "Image of a single basis function Re V_nm (for display)."
    function zbasis_image(n, m, S)
        c=(S+1)/2; Rd=S/2; out=fill(NaN, S, S)
        for i in 1:S, j in 1:S
            y=(i-c)/Rd; x=(j-c)/Rd; ρ=hypot(y,x); ρ>1 && continue
            out[i,j]=zrad(n,m,ρ)*cos(m*atan(y,x))
        end
        out
    end
end

# ╔═╡ 90000000-0000-0000-0000-000000000008
md"""
### Controls

source: $(@bind src Select(vcat(SYNTH, "EMNIST: " .* CLASSES), default="EMNIST: K"))
EMNIST instance: $(@bind inst Slider(1:30, default=1, show_value=true))
synthetic stroke radius: $(@bind srad Slider(2:1:9, default=6, show_value=true))

**max order n_max**: $(@bind nmax Slider(2:1:12, default=8, show_value=true))
**zero-DC** (subtract the in-disc mean): $(@bind zerodc CheckBox(default=true))

unit disc: $(@bind discfit Select(["q98"=>"fitted to the ink (98th pct radius)", "max"=>"fitted to the ink (max radius)", "frame"=>"inscribed in the 112 frame"]))
rotate the character by: $(@bind rot Slider(0:5:180, default=0, show_value=true))°
"""

# ╔═╡ 90000000-0000-0000-0000-000000000009
begin
    raw = startswith(src, "EMNIST: ") ?
          upsample(em.class_images[findfirst(==(replace(src,"EMNIST: "=>"")), em.class_names)][inst]) :
          synth_img(src, srad)
    rotated = rot == 0 ? raw : rotate_img(raw, rot)
    qfit    = discfit=="q98" ? 0.98 : discfit=="max" ? 1.0 : nothing
    img     = fit_disc(rotated; q=qfit)

    MASK = zmask(IMG)
    (BAS, ORDS) = zbasis(IMG, nmax)
    A    = zmoments(img, BAS, MASK; zerodc=zerodc)
    F0   = zfield(img, MASK; zerodc=zerodc)
    REC  = zreconstruct(A, ORDS, IMG)
    rel  = sqrt(mean((REC[MASK] .- F0[MASK]).^2)) / max(sqrt(mean(F0[MASK].^2)), 1e-12)

    Markdown.parse("**source** `$(src)`" * (rot>0 ? " rotated **$(rot)°**" : "") *
        " — disc **$(discfit)** — **n_max = $(nmax)** ⇒ **$(length(ORDS)) moments** (m ≥ 0) — " *
        "zero-DC **$(zerodc)** (A₀₀ = $(round(abs(A[1]), sigdigits=3))) — " *
        "reconstruction relative RMS error **$(round(rel, digits=3))**")
end

# ╔═╡ 90000000-0000-0000-0000-00000000000a
let
    kw=(yflip=true, aspect_ratio=:equal, axis=false, ticks=false, cbar=false,
        xlims=(1,IMG), ylims=(1,IMG))
    div = cgrad(:RdBu, rev=true)
    m1 = maximum(abs, F0); m1 = m1==0 ? 1.0 : m1
    m2 = max(maximum(abs, REC), 1e-12)
    res = REC .- F0; res[.!MASK] .= 0
    p1 = heatmap(img; c=:grays, title="input (disc-fitted)", titlefontsize=8, kw...)
    p2 = heatmap(F0;  c=div, clims=(-m1,m1), title=(zerodc ? "zero-DC field on the disc" : "field on the disc"),
                 titlefontsize=8, kw...)
    p3 = heatmap(REC; c=div, clims=(-m2,m2), title="reconstruction, n ≤ $(nmax)", titlefontsize=8, kw...)
    p4 = heatmap(res; c=div, clims=(-m1,m1), title="residual (rel. RMS $(round(rel,digits=3)))",
                 titlefontsize=8, kw...)
    plot(p1,p2,p3,p4; layout=(1,4), size=(1200,330))
end

# ╔═╡ 90000000-0000-0000-0000-00000000000b
md"""
### The moments themselves

Left: `|A_nm|` on the Zernike pyramid — **`m` across** (angular frequency: `m = 0` is
rotationally symmetric, `m = 2` two-lobed, `m = 4` four-lobed) and **`n` down**
(radial order: how much fine radial structure). Right: the same numbers as a bar
chart, biggest first.
"""

# ╔═╡ 90000000-0000-0000-0000-00000000000c
let
    mags = abs.(A)
    ms = [m for (n,m) in ORDS]; ns = [n for (n,m) in ORDS]
    mx = maximum(mags); mx = mx==0 ? 1.0 : mx
    p1 = scatter(ms, ns; marker_z=mags, ms=[6+22*(v/mx) for v in mags], c=:viridis,
                 msw=0.5, msc=:black, label="", yflip=true, xlabel="m  (angular)",
                 ylabel="n  (radial)", title="|A_nm|", titlefontsize=9,
                 xticks=0:1:nmax, yticks=0:1:nmax, guidefontsize=8, tickfontsize=7,
                 xlims=(-0.7,nmax+0.7), ylims=(-0.7,nmax+0.7), cbar=true)
    ord = sortperm(mags, rev=true); k = min(12, length(ord))
    labs = ["A$(ORDS[i][1]),$(ORDS[i][2])" for i in ord[1:k]]
    p2 = bar(mags[ord[1:k]]; orientation=:v, label="", c=:steelblue,
             xticks=(1:k, labs), xrotation=45, title="largest |A_nm|",
             titlefontsize=9, tickfontsize=7, grid=false)
    plot(p1, p2; layout=(1,2), size=(1050,400))
end

# ╔═╡ 90000000-0000-0000-0000-00000000000d
let
    ord = sortperm(abs.(A), rev=true)
    hdr = "**The largest moments** — `arg A` is what rotation changes (`A_nm → A_nm·e^{−imα}`); " *
          "`|A_nm|` is what it does not.\n\n" *
          "| (n, m) | \\|A_nm\\| | arg A (°) | angular symmetry |\n|:--|--:|--:|:--|\n"
    rows = [ (nm = ORDS[i];
              @sprintf("| (%d, %d) | %.4f | %.0f | %s |", nm[1], nm[2], abs(A[i]),
                       rad2deg(angle(A[i])),
                       nm[2]==0 ? "rotationally symmetric" : "$(nm[2])-fold (period $(round(360/nm[2],digits=0))°)"))
             for i in ord[1:min(10,length(ord))] ]
    Markdown.parse(hdr * join(rows, "\n"))
end

# ╔═╡ 90000000-0000-0000-0000-00000000000e
md"""
### What each moment looks for

`Re V_nm` for the low orders — the templates the moments correlate the character
against. Across a row `m` grows (more angular lobes); down a column `n` grows (more
radial rings).
"""

# ╔═╡ 90000000-0000-0000-0000-00000000000f
let
    show_orders = [(n,m) for (n,m) in zorders(min(nmax,4))]
    kw=(yflip=true, aspect_ratio=:equal, axis=false, ticks=false, cbar=false,
        xlims=(1,IMG), ylims=(1,IMG))
    panels = [heatmap(zbasis_image(n,m,IMG); c=cgrad(:RdBu, rev=true),
                      title="V$(n),$(m)", titlefontsize=8,
                      background_color_inside=:white, kw...) for (n,m) in show_orders]
    nc = 5; nr = ceil(Int, length(panels)/nc)
    plot(panels...; layout=(nr,nc), size=(200nc, 200nr))
end

# ╔═╡ 90000000-0000-0000-0000-000000000010
md"""
### How many moments does the character need?

Reconstruction as `n_max` grows. 25 moments cannot resolve a 13 px stroke inside a
112 px disc, so the reconstruction stays a blurred sketch — the point of the
descriptor is discrimination, not compression.
"""

# ╔═╡ 90000000-0000-0000-0000-000000000011
let
    steps = [2,4,6,8,10,12]
    kw=(yflip=true, aspect_ratio=:equal, axis=false, ticks=false, cbar=false,
        xlims=(1,IMG), ylims=(1,IMG))
    div = cgrad(:RdBu, rev=true)
    panels = Any[heatmap(F0; c=div, clims=(-maximum(abs,F0),maximum(abs,F0)),
                         title="target", titlefontsize=8, kw...)]
    for nm in steps
        Bn, On = zbasis(IMG, nm)
        An = zmoments(img, Bn, MASK; zerodc=zerodc)
        Rn = zreconstruct(An, On, IMG)
        e  = sqrt(mean((Rn[MASK] .- F0[MASK]).^2))/max(sqrt(mean(F0[MASK].^2)),1e-12)
        m  = max(maximum(abs,Rn),1e-12)
        push!(panels, heatmap(Rn; c=div, clims=(-m,m),
              title="n≤$(nm) · $(length(On)) mom · err $(round(e,digits=2))",
              titlefontsize=7, kw...))
    end
    plot(panels...; layout=(1,length(panels)), size=(210*length(panels), 240))
end

# ╔═╡ 90000000-0000-0000-0000-000000000012
md"""
### Rotation invariance — the property Fourier on a square grid did not have

The character is rotated through 0–180° and the moments recomputed. `|A_nm|` should
be flat; the complex moments should not.
"""

# ╔═╡ 90000000-0000-0000-0000-000000000013
let
    αs = 0:15:180
    A0 = zmoments(fit_disc(raw; q=qfit), BAS, MASK; zerodc=zerodc)
    mx = max(maximum(abs, A0), 1e-12)
    dmag = Float64[]; dcpx = Float64[]
    for α in αs
        Aα = zmoments(fit_disc(rotate_img(raw,α); q=qfit), BAS, MASK; zerodc=zerodc)
        push!(dmag, maximum(abs.(abs.(A0) .- abs.(Aα)))/mx)
        push!(dcpx, maximum(abs.(A0 .- Aα))/mx)
    end
    p = plot(αs, dcpx; lw=2, marker=:square, label="complex A_nm",
             xlabel="rotation (°)", ylabel="max change ÷ max|A|",
             title="what rotation does to the moments", titlefontsize=9,
             guidefontsize=8, tickfontsize=7, legend=:right)
    plot!(p, αs, dmag; lw=2, marker=:circle, label="|A_nm|  (invariant)")
    annotate!(p, 90, maximum(dcpx)*0.35,
              text(@sprintf("max |A| drift = %.3f", maximum(dmag)), 8, :left))
    plot(p; size=(750,360))
end

# ╔═╡ 90000000-0000-0000-0000-000000000014
md"""
### Is it diagnostic of letter identity?

Same protocol as `New_Gabor_FPE/KeyPointDiagnosticity.md` and
`ExptsWithGlobalFourier/TicTacToeSignature.md`: EMNIST, 12 classes
(`O C I L T X K A H Y E F`), leave-one-out **nearest-class-mean** on the standardised
vector. **Chance = 8.3 %.**

run: $(@bind run_diag CheckBox(default=false)) · instances per class: $(@bind n_per Slider(5:5:30, default=15, show_value=true))
"""

# ╔═╡ 90000000-0000-0000-0000-000000000015
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
    function loo_ncm(X,y)
        n,p=size(X); Z=copy(X)
        for j in 1:p
            s=std(@view Z[:,j]); s<=0 ? (Z[:,j].=0) : (Z[:,j].=(Z[:,j].-mean(Z[:,j]))./s)
        end
        classes=unique(y); correct=0
        for i in 1:n
            best=Inf; bestc=classes[1]
            for c in classes
                idx=[k for k in 1:n if y[k]==c && k!=i]
                m=vec(mean(Z[idx,:],dims=1)); dd=sum(abs2, Z[i,:].-m)
                dd<best && (best=dd; bestc=c)
            end
            bestc==y[i] && (correct+=1)
        end
        correct/n
    end
end

# ╔═╡ 90000000-0000-0000-0000-000000000016
if !run_diag
    md"*(diagnosticity not run — tick the box above)*"
else
    let
        imgs = Matrix{Float32}[]; labs = Int[]
        for (ci,c) in enumerate(CLASSES)
            v = em.class_images[findfirst(==(c), em.class_names)]
            for k in 1:min(n_per,length(v)); push!(imgs, upsample(v[k])); push!(labs, ci); end
        end
        Alist = [zmoments(fit_disc(im; q=qfit), BAS, MASK; zerodc=zerodc) for im in imgs]
        # the \\| are markdown escapes — an unescaped | would split the table cell
        variants = [("\\|A_nm\\| only — rotation-invariant", [abs.(a) for a in Alist]),
                    ("Re/Im A_nm — orientation-sensitive", [vcat(real.(a),imag.(a)) for a in Alist]),
                    ("both", [vcat(abs.(a),real.(a),imag.(a)) for a in Alist])]
        hdr = "**$(length(imgs)) images · n_max = $(nmax) ($(length(ORDS)) moments) · disc `$(discfit)` · " *
              "zero-DC $(zerodc).** Chance = $(round(100/length(CLASSES),digits=1)) %.\n\n" *
              "| feature set | numbers | LOO nearest-class-mean |\n|:--|--:|--:|\n"
        rows = String[]; ebest = Float64[]
        for (nm,fs) in variants
            X = reduce(vcat, [f' for f in fs])
            nm == "Re/Im A_nm — orientation-sensitive" && (ebest = eta2(X, labs))
            push!(rows, @sprintf("| %s | %d | **%.1f %%** |", nm, size(X,2), 100*loo_ncm(X,labs)))
        end
        top = sortperm(ebest, rev=true)[1:min(6,length(ebest))]
        nmz = vcat(["Re A$(n),$(m)" for (n,m) in ORDS], ["Im A$(n),$(m)" for (n,m) in ORDS])
        tail = "\n\n**Most diagnostic single components** (η², Re/Im set): " *
               join([@sprintf("`%s` %.2f", nmz[i], ebest[i]) for i in top], " · ")
        Markdown.parse(hdr * join(rows,"\n") * tail)
    end
end

# ╔═╡ 90000000-0000-0000-0000-000000000017
md"""
### Notes — what was measured

**1. Rotation invariance holds, to discretisation.** Rotating through 0–180° and
recomputing, the largest change in any `|A_nm|` relative to `max|A|` is **0.6 % (bar),
1.9 % (diagonal), 0.2 % (ring), 0.7–1.3 % (EMNIST K, T, A, E)** on the *inscribed*
disc, and **exactly 0** at 90° — 90° is a symmetry of the pixel grid, so there is no
resampling error at all. The residual at other angles is bilinear-resampling blur, not
a failure of the algebra. The *complex* moments change by **77 % / 154 % / 170 %** at
15° / 45° / 90°. So `|A|` is a genuine invariant and the phase is a genuine
orientation code, cleanly separated — exactly what the square DFT lattice could not
give (there, `|F(v,u)|` rotates with the letter).

**1b. But fitting the disc (§2) degrades that invariance ~5×.** Same measurement with
the ink-fitted disc: **3.0 % (bar), 1.8 % (diagonal), 3.3 % (plus), 10.2 % (ring),
1.0–4.8 % (EMNIST letters)**. The algebra is still exact; the *fit* is not. The
98th-percentile ink radius is re-estimated from each rotated, resampled copy, so the
rescale factor jitters slightly and every moment moves with it. The ring is the worst
case because all its ink sits at one radius, making that percentile maximally
twitchy. **So §2's accuracy gain and §1's invariance are in tension** — take the
fitted disc when the input is upright and you want discrimination, the inscribed disc
when you actually need the invariant.

**2. Fitting the disc to the ink matters more than anything else.** The Zernike basis
lives on the unit disc; if the letter only fills part of it, most of the basis is
spent describing empty space. Measured (n_max = 8, 360 EMNIST instances):

| unit disc | Re/Im | \\|A\\| only |
|:--|--:|--:|
| inscribed in the 112 frame | 64.7 % | 53.1 % |
| fitted: centroid + max ink radius | 70.0 % | 63.1 % |
| **fitted: centroid + 98th-pct radius** | **73.1 %** | **65.8 %** |

The 98th percentile beats the max because one stray pixel would otherwise shrink the
whole letter. (Contrast `TicTacToeSignature.md` §4, where bounding-box normalising the
letter *didn't* help the Fourier grid — there the grid was tied to the frame either
way, so there was nothing to gain.)

**3. Rotation invariance costs accuracy.** With the fitted disc, discarding the phase
drops the score from **73.1 %** to **65.8 %**; on the unfitted frame the gap is bigger
(64.7 % → 53.1 %). This is the same lesson as the Fourier notebook from the other
side: **orientation is where the identity lives**, so an invariant that throws it away
is throwing away signal. Rotation invariance is the right tool when the input really
can arrive at any angle; for upright letters it is a handicap, not a feature.

Keeping *both* — the 25 invariant magnitudes alongside the 50 real/imaginary parts —
scores **76.4 %** (75 numbers, fitted disc, 360 instances), the best result in this
notebook. The magnitudes are redundant in principle, but they hand the
equally-weighted nearest-mean classifier a pre-computed invariant it would otherwise
have to infer, so the redundancy pays.

**4. Zero-DC costs about 1 point** — 64.7 % with the mean subtracted vs 65.6 % with it
kept (inscribed disc, Re/Im); for `|A|` it is 53.1 % vs 53.3 %. `A₀₀` is one number out
of 25, and the overall ink level barely separates letters, so removing it is nearly
free. It does make every remaining moment a statement about *structure*.

**5. Accuracy peaks at n_max ≈ 8 and then falls.** Measured (Re/Im, zero-DC, inscribed
disc): **35.6 % / 56.9 % / 63.9 / 64.7 / 61.1 / 59.2 %** at n_max = 2 / 4 / 6 / 8 / 10 /
12 (4 / 9 / 16 / 25 / 36 / 49 moments). The decline past 8 is the familiar
nearest-class-mean dilution artifact — high-order moments are noisy on 13 px strokes
and the unweighted classifier gives them equal weight. **25 moments is the sweet
spot.**

**6. Reconstruction stays poor, and that is fine.** Relative in-disc RMS error is
**0.91 / 0.83 / 0.74 / 0.66 / 0.58 / 0.51** at n_max = 2 … 12 (synthetic figures,
inscribed disc). A 13 px stroke inside a 112 px disc needs far more than 25 moments to
be *drawn*. It does not need them to be *told apart*: 25 moments give 73 % on 12
classes at 8.3 % chance.

**7. Against the Fourier work.** Best global Zernike here is **73.1 %** (50 numbers,
Re/Im, fitted disc). `TicTacToeSignature.md` measured **49.7 %** for the same nine
Fourier features taken over the whole image with no spatial grid, and **74.2–76.1 %**
once those features were computed per cell of a 3×3 grid. So Zernike is a far better
*global* descriptor than global Fourier moments — the orthogonal disc basis genuinely
buys something — but it lands level with, not ahead of, simply cutting the image into
nine boxes and measuring stroke orientation in each. **Spatial partitioning is worth
about as much as a better global basis, and the two are complementary rather than
competing.**
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
# ╠═90000000-0000-0000-0000-00000000000c
# ╟─90000000-0000-0000-0000-00000000000d
# ╟─90000000-0000-0000-0000-00000000000e
# ╠═90000000-0000-0000-0000-00000000000f
# ╟─90000000-0000-0000-0000-000000000010
# ╠═90000000-0000-0000-0000-000000000011
# ╟─90000000-0000-0000-0000-000000000012
# ╠═90000000-0000-0000-0000-000000000013
# ╟─90000000-0000-0000-0000-000000000014
# ╠═90000000-0000-0000-0000-000000000015
# ╠═90000000-0000-0000-0000-000000000016
# ╟─90000000-0000-0000-0000-000000000017
