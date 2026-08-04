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
    include(joinpath(@__DIR__, "..", "LoadEMNIST.module.jl"))
    using .LoadEMNIST
end

# ╔═╡ 90000000-0000-0000-0000-000000000004
md"""
# A tic-tac-toe **Zernike** signature

The same construction as `P0.4_ExptsWithGlobalFourier/TicTacToeFourierSignature.jl` — an
N×N grid over the character, one small descriptor per cell — but each cell is
described by **Zernike moments on the cell's own disc** instead of low-order Fourier
coefficients. Same image format (112×112 letter, stroke ≈ 13 px), same diagnosticity
protocol, so the two are directly comparable.

**Headline, measured below:** on clean centred input the per-cell *features* are
**better** than the Fourier ones — Zernike detects **crossings** and **loops**, both of
which the Fourier cell descriptor failed at — but the resulting **classification is
worse** (≈ 66 % vs ≈ 76 %), and worse than plain global Zernike too.

The reason is not the one you would guess. It is **not** the hard disc boundary
(swapping the Fourier window for a hard disc costs nothing). It is that Zernike
moments are taken **about the disc centre**, so *where* a stroke sits in the cell leaks
into the coefficients that carry *what shape* it is — an isotropic blob slid off-centre
reports its position angle as an orientation. Tiling space wants a
translation-invariant cell descriptor; Zernike's free invariance is rotation about a
centre instead. See Notes §5.

### The per-cell descriptor

Each cell takes the disc inscribed in a patch of `ov ×` the cell size, and its Zernike
moments `A_nm` up to `n_max`. Because the angular index `m` **is** the angular
harmonic, the summaries are direct reads rather than tensor constructions:

| symbol | definition | reads as |
|:--|:--|:--|
| `a₀` | `A₀₀` | **ink** in the cell |
| `ac` | `√Σ_{nm≠00} \\|A\\|²` | how much structure |
| `O₂` | pooled `m = 2`: `Σ_n conj(A_n2)`, scaled to the `m=2` power share | **orientation** — `\\|O₂\\|` anisotropy, `arg(O₂)/2` = **stroke angle** |
| `f₄` | share of AC power at `m = 4` | **crossing** (two strokes ≈ 90° apart) |
| `m₀` | share of AC power at `m = 0` | rotationally symmetric content (blob / loop) |
| `loop` | `−Re A₄₀ / (\\|A₂₀\\| + \\|A₄₀\\|)` | **hollow vs filled** — the *sign* of `A₄₀` |

Only `A₀₀` responds to adding a constant to the image (verified to 1e-16 for every
`m ≠ 0`), so **zero-DC changes exactly one number** and every other moment is already
a statement about structure. Because of that the moments are always taken with the DC
left in — `a₀` stays available as the cell's ink level, and the *zero-DC* box is a
view: it shows the field and the reconstruction with `A₀₀` removed, which is the same
descriptor minus that one number.
"""

# ╔═╡ 90000000-0000-0000-0000-000000000005
begin
    const IMG = 112
    const CLASSES = ["O","C","I","L","T","X","K","A","H","Y","E","F"]
end

# ╔═╡ 90000000-0000-0000-0000-000000000006
# ---- images (same set as the Fourier tic-tac-toe notebook) ----
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
        name=="diag \\"   && return drawbar!(z(),20,20,92,92,r)
        name=="plus"      && return (i=z();drawbar!(i,20,56,92,56,r);drawbar!(i,56,20,56,92,r))
        name=="X"         && return (i=z();drawbar!(i,20,20,92,92,r);drawbar!(i,20,92,92,20,r))
        name=="T"         && return (i=z();drawbar!(i,36,20,36,92,r);drawbar!(i,36,56,92,56,r))
        name=="L-shape"   && return (i=z();drawbar!(i,26,40,80,40,r);drawbar!(i,80,40,80,86,r))
        name=="disc"      && return disc!(z(),56,56,15)
        name=="small ring"&& return ring!(z(),56,56,15,r)
        name=="3 strokes" && return (i=z();drawbar!(i,10,10,30,30,r);drawbar!(i,50,90,80,90,r);
                                     drawbar!(i,95,15,95,45,r))
        return ring!(z(),56,56,30,r)      # "big ring"
    end
    const SYNTH=["vert bar","horz bar","diag /","diag \\","plus","X","T","L-shape",
                 "disc","small ring","big ring","3 strokes"]
    em = load_emnist(n_images_to_load=20000, n_classes=47)
end;

# ╔═╡ 90000000-0000-0000-0000-000000000007
# ---- Zernike on a cell disc ----
begin
    function zrad(n::Int, m::Int, ρ::Float64)
        m=abs(m); ((n-m) % 2 != 0 || m > n) && return 0.0
        s=0.0
        for k in 0:((n-m)÷2)
            s += (-1)^k * factorial(n-k) /
                 (factorial(k)*factorial((n+m)÷2-k)*factorial((n-m)÷2-k)) * ρ^(n-2k)
        end
        s
    end
    zorders(nmax) = [(n,m) for n in 0:nmax for m in 0:n if (n-m) % 2 == 0]
    zmask(S::Int) = (c=(S+1)/2; [hypot((i-c)/(S/2),(j-c)/(S/2)) <= 1 for i in 1:S, j in 1:S])
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
    function zmoments(img, B, mask; zerodc=true)
        f = Float64.(img)
        zerodc && (f = f .- mean(f[mask]); f[.!mask] .= 0.0)
        [sum(b .* f) for b in B]
    end
    function zreconstruct(A, ords, S)
        c=(S+1)/2; Rd=S/2; out=fill(NaN,S,S)
        for i in 1:S, j in 1:S
            y=(i-c)/Rd; x=(j-c)/Rd; ρ=hypot(y,x); ρ>1 && continue
            θ=atan(y,x); s=0.0
            for (k,(n,m)) in enumerate(ords)
                v=zrad(n,m,ρ); s += m==0 ? real(A[k])*v : 2*real(A[k]*cis(m*θ))*v
            end
            out[i,j]=s
        end
        out
    end

    function patch_at(img, cy::Real, cx::Real, P::Int)
        N=size(img,1); out=zeros(Float32,P,P)
        y0=round(Int,cy)-(P-1)÷2; x0=round(Int,cx)-(P-1)÷2
        @inbounds for i in 1:P, j in 1:P
            out[i,j]=img[clamp(y0+i-1,1,N), clamp(x0+j-1,1,N)]
        end; out
    end
    cellpatch(N, ov) = (cs=IMG/N; P=round(Int,ov*cs); isodd(P) ? P : P+1)

    """
    Summarise one cell's moments. `A_nm` carries `e^{−imθ}`, so `arg(A_n2) = −2φ` and
    the stroke angle is `−arg/2` — taken here via `conj`, so `arg(O₂)/2` reads directly.
    """
    function cell_summary(A, ords)
        i00=findfirst(==((0,0)),ords)
        ac2 = sum(abs2, A[k] for k in eachindex(A) if k != i00) + 1e-24
        s2 = sum(conj(A[k]) for k in eachindex(A) if ords[k][2]==2; init=ComplexF64(0))
        p2 = sum(abs2(A[k]) for k in eachindex(A) if ords[k][2]==2; init=0.0)
        p4 = sum(abs2(A[k]) for k in eachindex(A) if ords[k][2]==4; init=0.0)
        p0 = sum(abs2(A[k]) for k in eachindex(A) if ords[k][2]==0 && k != i00; init=0.0)
        O2 = s2 == 0 ? ComplexF64(0) : (s2/abs(s2)) * sqrt(p2/ac2)
        i20=findfirst(==((2,0)),ords); i40=findfirst(==((4,0)),ords)
        lp = i40 === nothing ? 0.0 :
             -real(A[i40])/(abs(A[i20]) + abs(A[i40]) + 1e-12)
        (a0=real(A[i00]), ac=sqrt(ac2), O2=O2, f4=p4/ac2, m0=p0/ac2, loop=lp)
    end
    stroke_angle(s) = mod(rad2deg(angle(s.O2))/2, 180)
    cell_vector(s) = Float64[s.a0, s.ac, real(s.O2), imag(s.O2), abs(s.O2), s.f4, s.m0, s.loop]
    const FEATNAMES = ["a0","ac","ReO2","ImO2","|O2|","f4","m0","loop"]

    # Moments are always taken with the DC left in, because a constant moves *only*
    # A₀₀ (Notes §1). So `a₀` stays available as the cell's ink level, and the zero-DC
    # view is just the same vector with A₀₀ set to zero — no need to compute twice.
    "Grid of per-cell moment vectors."
    function grid_cells(img, B, mask, P, N)
        cs=IMG/N
        [zmoments(patch_at(img,(i-0.5)*cs,(j-0.5)*cs,P), B, mask; zerodc=false) for i in 1:N, j in 1:N]
    end
    "Copy of `A` with the DC term zeroed (for the zero-DC display)."
    function drop_dc(A, ords)
        A2 = copy(A); A2[findfirst(==((0,0)),ords)] = 0; A2
    end
end

# ╔═╡ 90000000-0000-0000-0000-000000000008
# ---- DISPLAY AID ONLY: a readable label per cell ----
begin
    """
    Thresholds here exist purely to draw the figure. Nothing downstream uses them and
    no number in the Notes comes from them — the descriptor is the continuous 8-vector.
    """
    function cell_label(s; ink=0.02, aniso=0.30, cross=0.45, loop=0.45)
        abs(s.a0) < ink && s.ac < 0.15 && return ("·", :grey)
        if abs(s.O2) >= aniso
            a = stroke_angle(s)
            # angles are measured with y pointing down: 45° runs top-left→bottom-right
            k = argmin(abs.([0,45,90,135,180] .- a))
            return (["—","\\","|","/","—"][k], :yellow)
        end
        s.f4   >= cross && return ("✳", :magenta)
        s.loop >= loop  && return ("O", :cyan)
        return ("●", :orange)
    end
end

# ╔═╡ 90000000-0000-0000-0000-000000000009
md"""
### Controls

source: $(@bind src Select(vcat(SYNTH, "EMNIST: " .* CLASSES), default="EMNIST: K"))
EMNIST instance: $(@bind inst Slider(1:30, default=1, show_value=true))
synthetic stroke radius: $(@bind srad Slider(2:1:9, default=6, show_value=true))

**grid N×N**: $(@bind gridN Slider(1:1:5, default=3, show_value=true))
**disc overlap** (× cell): $(@bind ov Slider(1.0:0.1:2.0, default=1.3, show_value=true))
**max order n_max**: $(@bind nmax Slider(2:1:8, default=6, show_value=true))
show the zero-DC view (a constant moves only A₀₀, so this drops one number): $(@bind zerodc CheckBox(default=true))

label thresholds *(display only)* — ink $(@bind t_ink Slider(0.0:0.01:0.20, default=0.02, show_value=true)) · anisotropy $(@bind t_aniso Slider(0.05:0.05:0.80, default=0.30, show_value=true)) · crossing $(@bind t_cross Slider(0.10:0.05:0.90, default=0.45, show_value=true)) · loop $(@bind t_loop Slider(0.10:0.05:0.90, default=0.45, show_value=true))
"""

# ╔═╡ 90000000-0000-0000-0000-00000000000a
begin
    img = startswith(src, "EMNIST: ") ?
          upsample(em.class_images[findfirst(==(replace(src,"EMNIST: "=>"")), em.class_names)][inst]) :
          synth_img(src, srad)
    CS   = IMG/gridN
    PATCH = cellpatch(gridN, ov)
    (BAS, ORDS) = zbasis(PATCH, nmax)
    CMASK = zmask(PATCH)
    MOM = grid_cells(img, BAS, CMASK, PATCH, gridN)
    G   = [cell_summary(A, ORDS) for A in MOM]

    Markdown.parse("**source** `$(src)` — grid **$(gridN)×$(gridN)**, cell **$(round(CS,digits=1)) px**, " *
        "cell disc **⌀$(PATCH) px** — **n_max = $(nmax)** ⇒ **$(length(ORDS)) moments/cell** — " *
        "descriptor = **$(8*gridN^2) numbers** ($(gridN^2) cells × 8)")
end

# ╔═╡ 90000000-0000-0000-0000-00000000000b
let
    # clims to 1.8 dims the ink to mid-grey so the overlays read on strokes as well as background
    p = heatmap(img; c=:grays, clims=(0f0,1.8f0), yflip=true, aspect_ratio=:equal,
                axis=false, ticks=false, cbar=false, xlims=(1,IMG), ylims=(1,IMG),
                size=(580,580), title="$(src) — Zernike tic-tac-toe", titlefontsize=10)
    θs = range(0,2π,length=80)
    for k in 1:(gridN-1)
        plot!(p, [1,IMG], [k*CS,k*CS]; lc=:red, lw=1, alpha=0.4, label="")
        plot!(p, [k*CS,k*CS], [1,IMG]; lc=:red, lw=1, alpha=0.4, label="")
    end
    for i in 1:gridN, j in 1:gridN
        s = G[i,j]; cy=(i-0.5)*CS; cx=(j-0.5)*CS
        lab, col = cell_label(s; ink=t_ink, aniso=t_aniso, cross=t_cross, loop=t_loop)
        # the analysis disc itself
        plot!(p, cx .+ (PATCH/2).*cos.(θs), cy .+ (PATCH/2).*sin.(θs);
              lc=:red, lw=1, ls=:dot, alpha=0.35, label="")
        if abs(s.O2) > 0.02
            a = stroke_angle(s); ℓ = 0.45*CS*clamp(abs(s.O2)/0.6, 0.15, 1.0)
            plot!(p, [cx-ℓ*cosd(a), cx+ℓ*cosd(a)], [cy-ℓ*sind(a), cy+ℓ*sind(a)];
                  lc=:yellow, lw=4, alpha=0.85, label="")
        end
        if abs(s.O2) < t_aniso && s.ac > 0.15
            r = 0.30*CS
            c2 = s.f4 >= t_cross ? :magenta : (s.loop >= t_loop ? :cyan : :orange)
            plot!(p, cx .+ r.*cos.(θs), cy .+ r.*sin.(θs); lc=c2, lw=3, alpha=0.9, label="")
        end
        annotate!(p, cx-0.44CS, cy-0.36CS, text(lab, 12, col, :left))
        annotate!(p, cx-0.44CS, cy+0.38CS,
                  text(@sprintf("a₀%.2f |O₂|%.2f", s.a0, abs(s.O2)), 6, :springgreen, :left))
    end
    p
end

# ╔═╡ 90000000-0000-0000-0000-00000000000c
let
    hdr = "**Per-cell descriptor.** `label` is the display aid; the numbers are the descriptor.\n\n" *
          "| cell | label | a₀ | ac | \\|O₂\\| | stroke° | f₄ | m₀ | loop |\n" *
          "|:--|:--:|--:|--:|--:|--:|--:|--:|--:|\n"
    rows = String[]
    for i in 1:gridN, j in 1:gridN
        s = G[i,j]; lab,_ = cell_label(s; ink=t_ink, aniso=t_aniso, cross=t_cross, loop=t_loop)
        push!(rows, @sprintf("| (%d,%d) | %s | %.3f | %.3f | %.3f | %.0f | %.3f | %.3f | %+.3f |",
              i, j, lab, s.a0, s.ac, abs(s.O2), stroke_angle(s), s.f4, s.m0, s.loop))
    end
    Markdown.parse(hdr * join(rows, "\n"))
end

# ╔═╡ 90000000-0000-0000-0000-00000000000d
md"""
### What each cell's moments actually retain

Top: the cell disc as the moments see it. Bottom: the reconstruction from that cell's
`n ≤ n_max` moments. Watch **where in each disc the ink sits** — the moments are taken
about the disc centre, so a stroke fragment lying off-centre is described partly by
its *position*, and that is what Notes §5 shows to be the real problem with gridding
Zernike. (The sharp disc boundary, which one might expect to be the culprit, is
measurably *not*: see §5.)
"""

# ╔═╡ 90000000-0000-0000-0000-00000000000e
let
    kw=(yflip=true, aspect_ratio=:equal, axis=false, ticks=false, cbar=false,
        xlims=(0.5,PATCH+0.5), ylims=(0.5,PATCH+0.5))
    div = cgrad(:RdBu, rev=true)
    tops=Any[]; bots=Any[]
    for i in 1:gridN, j in 1:gridN
        pat = patch_at(img, (i-0.5)*CS, (j-0.5)*CS, PATCH)
        f = Float64.(pat); zerodc && (f = f .- mean(f[CMASK])); fm = copy(f); fm[.!CMASK] .= NaN
        rec = zreconstruct(zerodc ? drop_dc(MOM[i,j], ORDS) : MOM[i,j], ORDS, PATCH)
        m1 = max(maximum(abs, f[CMASK]), 1e-12)
        m2 = max(maximum(abs, filter(!isnan, rec)), 1e-12)
        push!(tops, heatmap(fm;  c=div, clims=(-m1,m1), title="($(i),$(j))", titlefontsize=7, kw...))
        push!(bots, heatmap(rec; c=div, clims=(-m2,m2), title="", titlefontsize=7, kw...))
    end
    plot(vcat(tops,bots)...; layout=(2, gridN^2), size=(130*gridN^2, 280))
end

# ╔═╡ 90000000-0000-0000-0000-00000000000f
md"""
### Calibration: orientation, crossings, loops — and the catch

Left: a bar swept through the centre cell — reported stroke angle vs true. Middle: the
three type features on the canonical figures; **`f₄` and `loop` are where this
descriptor beats its Fourier counterpart.** Right: **the catch** — a perfectly
isotropic blob, which has no orientation whatsoever, placed off-centre in the cell.
The `m = 2` phase reports *where it sits*, not what shape it is. See Notes §5.
"""

# ╔═╡ 90000000-0000-0000-0000-000000000010
let
    P = cellpatch(3, ov); B3,O3 = zbasis(P, nmax); M3 = zmask(P)
    angles = 0:15:165
    errs = Float64[]; read = Float64[]
    for θ in angles
        im = z(); L=40
        drawbar!(im, 56-L*sind(θ), 56-L*cosd(θ), 56+L*sind(θ), 56+L*cosd(θ), srad)
        s = cell_summary(zmoments(patch_at(im,56,56,P), B3, M3; zerodc=false), O3)
        r = stroke_angle(s); push!(read, r)
        push!(errs, abs(mod(r-θ+90,180)-90))
    end
    p1 = plot(angles, errs; lw=2, marker=:circle, label=@sprintf("max %.2f°", maximum(errs)),
              xlabel="true angle (°)", ylabel="|error| (°)", title="orientation from the m=2 moments",
              titlefontsize=9, guidefontsize=8, tickfontsize=7, legend=:top)

    names = ["vert bar","diag /","plus","X","disc","small ring"]
    o2=Float64[]; f4=Float64[]; lp=Float64[]
    for nmz in names
        s = cell_summary(zmoments(patch_at(synth_img(nmz,srad),56,56,P), B3, M3; zerodc=false), O3)
        push!(o2, abs(s.O2)); push!(f4, s.f4); push!(lp, s.loop)
    end
    # grouped bars by hand — groupedbar lives in StatsPlots, which this project doesn't use
    xs = collect(1:length(names)); w = 0.26
    p2 = bar(xs .- w, o2; bar_width=w, label="|O₂| (oriented)", c=:goldenrod,
             xticks=(xs, names), xrotation=25, legend=:topleft,
             title="type features on the canonical figures", titlefontsize=9,
             tickfontsize=7, legendfontsize=7, grid=false, ylims=(-0.6,1.05))
    bar!(p2, xs,      f4; bar_width=w, label="f₄ (crossing)", c=:magenta)
    bar!(p2, xs .+ w, lp; bar_width=w, label="loop (hollow)", c=:teal)
    hline!(p2, [0]; lc=:black, lw=1, label="")

    # the position/orientation confound: an isotropic blob has NO orientation, yet the
    # m=2 phase reports the angle at which it sits in the cell
    φs = 0:15:165; rep = Float64[]
    for φ in φs
        im=z(); rr=7.0; d=12.0; cy=56+d*sind(φ); cx=56+d*cosd(φ)
        for y in 1:IMG, x in 1:IMG; hypot(y-cy,x-cx)<=rr && (im[y,x]=1f0); end
        s = cell_summary(zmoments(patch_at(im,56,56,P), B3, M3; zerodc=false), O3)
        push!(rep, stroke_angle(s))
    end
    p3 = scatter(collect(φs), rep; ms=5, c=:crimson, label="reported angle",
                 xlabel="where the blob sits (°)", ylabel="reported stroke angle (°)",
                 title="an isotropic blob reports its POSITION", titlefontsize=9,
                 guidefontsize=8, tickfontsize=7, legend=:topleft, xlims=(-5,170), ylims=(-5,185))
    plot!(p3, [0,165],[0,165]; lc=:grey, ls=:dash, label="y = x")
    plot(p1, p2, p3; layout=(1,3), size=(1400,380))
end

# ╔═╡ 90000000-0000-0000-0000-000000000011
md"""
### Is it diagnostic of letter identity?

Same protocol as `TicTacToeSignature.md` and `ZernikeCharacterMoments.jl`: EMNIST,
12 classes, leave-one-out **nearest-class-mean**, chance **8.3 %**.

run: $(@bind run_diag CheckBox(default=false)) · instances per class: $(@bind n_per Slider(5:5:30, default=15, show_value=true))
"""

# ╔═╡ 90000000-0000-0000-0000-000000000012
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

# ╔═╡ 90000000-0000-0000-0000-000000000013
if !run_diag
    md"*(diagnosticity not run — tick the box above)*"
else
    let
        imgs = Matrix{Float32}[]; labs = Int[]
        for (ci,c) in enumerate(CLASSES)
            v = em.class_images[findfirst(==(c), em.class_names)]
            for k in 1:min(n_per,length(v)); push!(imgs, upsample(v[k])); push!(labs, ci); end
        end
        X = reduce(vcat, [reduce(vcat, [cell_vector(cell_summary(A,ORDS))
                          for A in vec(grid_cells(im,BAS,CMASK,PATCH,gridN))])'
                          for im in imgs])
        e = eta2(X, labs)
        subsets = [("ink only  a₀",[1]), ("orientation only  (Re O₂, Im O₂)",[3,4]),
                   ("ink + orientation",[1,3,4]), ("ink + orientation + f₄ + loop",[1,3,4,6,8]),
                   ("all 8 per cell", collect(1:8))]
        hdr = "**$(length(imgs)) images, $(gridN)×$(gridN) grid, n_max = $(nmax) " *
              "($(length(ORDS)) moments/cell).** Chance = $(round(100/length(CLASSES),digits=1)) %.\n\n" *
              "| feature subset | numbers/cell | total | LOO nearest-class-mean |\n|:--|--:|--:|--:|\n"
        rows = [ (cols = vcat([fi:8:size(X,2) for fi in fs]...);
                  @sprintf("| %s | %d | %d | **%.1f %%** |", nm, length(fs), length(cols),
                           100*loo_ncm(X[:,cols], labs)))
                 for (nm,fs) in subsets ]
        tail = "\n\n**Per-feature η²** (mean over the $(gridN^2) cells): " *
               join([@sprintf("`%s` %.2f", FEATNAMES[k], mean(e[k:8:end])) for k in 1:8], " · ")
        Markdown.parse(hdr * join(rows,"\n") * tail)
    end
end

# ╔═╡ 90000000-0000-0000-0000-000000000014
md"""
### Notes — what was measured

**1. Only `A₀₀` carries the DC.** Adding a constant `0.37` to a cell moves `A₀₀` by
`0.37` and every `m ≠ 0` moment by **≤ 4e-16** — the continuous orthogonality survives
discretisation for the angular harmonics. Two small leaks do appear: the `m = 0`
radial moments move by ~1e-3 (the disc boundary is pixelated, so the mask is not
exactly a disc), and `A₄₄` by **7e-4** — the pixelated disc has the *square grid's own
4-fold symmetry*, which projects onto `m = 4`. That is the same lattice artifact that
biased `|E₄|` in the Fourier notebook, here four orders of magnitude smaller.

**2. Orientation is exact to a fraction of a degree.** A bar swept through 12 angles:
max error **0.24°** reading `−arg(A₂₂)/2`, **0.64°** using the pooled `m = 2` estimate
(pooling trades a little angular precision for a better-behaved anisotropy). Compare
**0.16°** for polar-sampled Fourier and **3.48°** for the square DFT lattice. Zernike's
angular basis *is* `e^{imθ}`, so there is no lattice to bias it.

**3. `f₄` detects crossings — the Fourier `|E₄|` did not.** Measured on the centre cell:

| figure | `\\|O₂\\|` | `f₄` | `loop` | *(Fourier `\\|E₄\\|` for comparison)* |
|:--|--:|--:|--:|--:|
| vertical bar | 0.612 | 0.187 | −0.45 | *0.179* |
| diagonal | 0.607 | 0.194 | — | *0.233* |
| **plus** | 0.000 | **0.699** | — | *0.085* |
| **X** | 0.000 | **0.689** | — | *0.142* |
| disc | 0.000 | 0.000 | −0.29 | *0.001* |
| ring | 0.000 | 0.000 | **+0.75** | *0.009* |

A crossing rings `f₄` at **0.69–0.70** against **0.19** for a single stroke — a 3.6×
separation. The Fourier fourth harmonic went the *wrong way* (a single bar 0.18 beat a
plus at 0.085) because the crossing's broadband isotropic energy diluted the
normalisation. Sorting by angular index `m` instead of by an angle of a power spectrum
avoids that entirely.

**4. Loops are detected by the *sign* of `A₄₀`.** `loop = −Re A₄₀/(|A₂₀|+|A₄₀|)` reads
**+0.75 (thin ring), +0.81 (thick ring)** against **−0.29 (disc), −0.45 (bar)** — and
crucially it is **robust to ring thickness**, where the Fourier `e₂/e₁` ratio flipped
its verdict between stroke radius 4 (loop) and 6 (blob). The radial polynomials are
orthogonal, so "hollow" is a genuine radial-profile fact rather than a proxy for size.

On real EMNIST the same geometric limit as before still applies — the loop must fit
inside a cell. Best-cell loop score, `O` vs the other 11 classes:

| grid | O-vs-rest AUC | mean O | mean rest |
|:--|--:|--:|--:|
| **1×1 (whole letter)** | **0.839** | +0.415 | −0.056 |
| 2×2 | 0.627 | +0.665 | +0.440 |
| 3×3 | 0.545 | +0.701 | +0.636 |

At 1×1 it works well and in the **right direction**, and it ranks `C` (0.516) *above*
`O` (0.415) — correct, an EMNIST `C` is very nearly a closed ring. Compare the Fourier
notebook, where the whole-image score reached AUC 0.80 but with the **sign inverted**,
because it was reading object scale rather than enclosure. So Zernike genuinely solves
loop-vs-blob; it just still needs a cell the size of the loop.

**5. And yet the grid classifies *worse*.** 360 EMNIST instances, LOO
nearest-class-mean, chance 8.3 %:

| descriptor | numbers | LOO |
|:--|--:|--:|
| Zernike 3×3, all 8 summaries/cell | 72 | 64.4 – 65.6 % |
| Zernike 3×3, raw complex moments (`n_max`=4) | 243 | 64.4 % |
| Zernike 3×3, ink + orientation | 27 | 61.7 % |
| Zernike 4×4, all 8/cell | 128 | **66.9 %** |
| Zernike 2×2, all 8/cell | 32 | 54.2 % |
| *global Zernike (no grid), `\\|A\\|` + Re/Im* | *75* | ***76.4 %*** |
| *Fourier 3×3 tic-tac-toe, ink + orientation* | *27* | ***76.1 %*** |

**Gridding hurts Zernike**, and by a lot: the best grid (66.9 %) is ~10 points below
both plain global Zernike and the Fourier grid.

*The obvious explanation is wrong.* The natural suspect is Zernike's **hard disc
boundary** — the Fourier cells used a smooth Gaussian window, so perhaps truncation
is the problem. **Measured: it is not.** Running the *Fourier* cell descriptor with a
hard disc window instead of the Gaussian changes almost nothing (all 5 features/cell:
74.7 % → 73.9 %; ink+orientation: 74.4 % → 75.3 %). The window is not what separates
the two.

*The actual cause is a position/orientation confound.* Zernike moments are taken
**about the disc centre**, so where a feature sits inside the cell leaks into the same
coefficients that carry its shape. Take a perfectly **isotropic blob** — a disc, which
has no orientation at all — and slide it around the cell: the `m = 2` phase reports
its **position angle**, near-exactly.

| blob placed at | 0° | 30° | 60° | 90° | 120° | 150° |
|:--|--:|--:|--:|--:|--:|--:|
| reported "stroke angle" | 180° | 31.9° | 58.1° | 90.0° | 121.9° | 148.1° |

The Fourier cell descriptor is immune because it is built from the **power spectrum**
`|F|²`, which discards phase and is therefore translation-invariant inside the cell.
The consequence shows up directly on real data: the two orientation estimates agree to
**< 1°** on a clean centred bar, but across 749 inked EMNIST cells their median
disagreement is **43.3°** — essentially uncorrelated (45° would be chance).

This also explains, retroactively, why fitting the disc to the ink was worth 8 points
in `ZernikeCharacterMoments.jl`: fitting *centres the object in its disc*, which is
precisely the condition under which Zernike moments describe shape rather than
placement. A grid cell cannot do that — the fragment is wherever the letter put it.

**6. The conclusion is a split verdict.** Per-cell Zernike gives **better features on
clean, centred input** — orientation to 0.6°, a crossing detector that works, a loop
detector that works — and **worse classification on real letters**, because in a grid
cell the input is neither clean nor centred, and §5's confound then charges the
orientation readout for the fragment's position.

So: if you want interpretable local symbols on well-centred patches, these features
are the better ones. If you want a discriminative vector off a real letter, use global
Zernike (with the disc fitted) or the Fourier grid. The natural combination is
suggested by global Zernike (76.4 %) and the Fourier 3×3 grid (76.1 %) scoring the same
while measuring different things — one centred global invariant description, one
translation-invariant spatial layout — so concatenating them is the obvious next test.

The transferable lesson is about **which invariance a basis gives you for free**. The
Fourier cell descriptor is translation-invariant within the cell and not
rotation-invariant; Zernike is rotation-invariant about its centre and not
translation-invariant. Tiling space asks for the first property, so it suits Fourier;
describing one centred object asks for the second, so it suits Zernike. Neither is
better — they are matched to different jobs, and using one for the other's job costs
about 10 points.
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
# ╠═90000000-0000-0000-0000-00000000000b
# ╟─90000000-0000-0000-0000-00000000000c
# ╟─90000000-0000-0000-0000-00000000000d
# ╠═90000000-0000-0000-0000-00000000000e
# ╟─90000000-0000-0000-0000-00000000000f
# ╠═90000000-0000-0000-0000-000000000010
# ╟─90000000-0000-0000-0000-000000000011
# ╠═90000000-0000-0000-0000-000000000012
# ╠═90000000-0000-0000-0000-000000000013
# ╟─90000000-0000-0000-0000-000000000014
