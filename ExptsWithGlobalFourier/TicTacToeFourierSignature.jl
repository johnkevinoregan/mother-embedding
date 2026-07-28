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
# A tic-tac-toe Fourier signature: what kind of stroke is in each area?

Divide the character into an **N×N grid** (3×3 by default) and describe each cell by
the **first two or three 2D Fourier coefficients** of that cell. The question this
notebook answers: *does that give a usable signature of "vertical / horizontal /
diagonal stroke", "blob", "loop" per area?*

**Short answer, measured below:** **orientation yes — very well**; **crossings no**;
**loop-vs-blob only when the cell is matched to the size of the loop**, which for a
3×3 grid on an EMNIST letter it is not. And the whole 9-cell signature turns out to
be strongly **diagnostic of letter identity** — better than anything else tried in
this project so far.

### The per-cell descriptor

The cell's low-order spectrum `|F(ω, θ)|²` is summarised by moments, not read
coefficient by coefficient:

| symbol | definition | reads as |
|:--|:--|:--|
| `a₀` | `F(0,0)` | **ink fraction** in the cell |
| `ac` | `√Σ_{≠0} \\|F\\|²` | how much *structure* (AC energy) |
| `E₂` | `Σ \\|F\\|² e^{2iθ} / Σ \\|F\\|²` | **orientation**: `\\|E₂\\|` = anisotropy (1 = one clean stroke, 0 = isotropic), `arg(E₂)/2 + 90°` = **stroke angle** |
| `E₄` | `Σ \\|F\\|² e^{4iθ} / Σ \\|F\\|²` | intended as "two strokes ~90° apart"; **it does not work** — see Notes |
| `e₁,e₂,e₃` | share of AC power in frequency rings `ω = 1, 2, 3` | radial profile → **loop score** `e₂/(e₁+e₂)` |

`E₂` is just the orientation tensor of the power spectrum. It is the **π-periodic**
quantity the project's design rules ask for (orientation mod π ⇒ encode 2θ), and
`(Re E₂, Im E₂)` is already the continuous 2-vector to bind — no thresholds, no
labels. The cell labels drawn on the figure are a **display aid only**; every number
quoted below comes from the continuous vector.

### Two ways to sample the same low-order spectrum

- **lattice** — the plain DFT orders `(v,u) ∈ [−K,K]²`, `θ = atan(v,u)`.
- **polar** — evaluate the very same integral at `ω ∈ {1,2,3}` × 12 orientations.
  Same coefficients, sampled on a polar grid instead of a square one.

This matters: the square lattice has its own 4-fold symmetry, which **biases the
orientation readout toward 0/45/90/135°**. Measured on a bar swept through 12
angles — lattice: up to **3.5°** error; polar: **0.16°**. Use *polar* to read
angles, *lattice* if you want literal DFT coefficients (for classification the two
are equivalent, 74.2 % vs 73.3 %).
"""

# ╔═╡ 90000000-0000-0000-0000-000000000005
begin
    const IMG = 112                    # the letter frame — the tic-tac-toe grid tiles this
    const CLASSES = ["O","C","I","L","T","X","K","A","H","Y","E","F"]
end

# ╔═╡ 90000000-0000-0000-0000-000000000006
# ---- synthetic figures + EMNIST (same image format as the rest of the project) ----
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
        N=size(img,1)
        for y in 1:N, x in 1:N; hypot(y-cy,x-cx)<=R && (img[y,x]=1f0); end; img
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
        name=="ring"      && return ring!(z(),56,56,15,r)
        name=="3 strokes" && return (i=z();drawbar!(i,10,10,30,30,r);drawbar!(i,50,90,80,90,r);
                                     drawbar!(i,95,15,95,45,r))
        return ring!(z(),56,56,28,r)   # "big ring"
    end
    const SYNTH=["vert bar","horz bar","diag /","diag \\","plus","X","T","L-shape",
                 "disc","ring","big ring","3 strokes"]
    em = load_emnist(n_images_to_load=20000, n_classes=47)
end;

# ╔═╡ 90000000-0000-0000-0000-000000000007
# ---- the per-cell descriptor: low-order spectrum -> (a0, ac, E2, E4, rings) ----
begin
    function patch_at(img, cy::Real, cx::Real, P::Int)
        N=size(img,1); out=zeros(Float32,P,P)
        y0=round(Int,cy)-(P-1)÷2; x0=round(Int,cx)-(P-1)÷2
        @inbounds for i in 1:P, j in 1:P
            out[i,j]=img[clamp(y0+i-1,1,N), clamp(x0+j-1,1,N)]
        end; out
    end
    gauss_window(P) = (σ=Float32(P)/4; c=Float32((P+1)/2);
                       Float32[exp(-((i-c)^2+(j-c)^2)/(2σ^2)) for i in 1:P, j in 1:P])

    # -- lattice sampling: the literal DFT orders (v,u) ∈ [-K,K]² --
    function low_coeffs(p, w, K)
        P=size(p,1)
        B=ComplexF32[cis(-2f0π*Float32(k)*Float32(n-1)/Float32(P)) for k in -K:K, n in 1:P]
        (B * ComplexF32.(w .* p) * transpose(B)) ./ Float32(sum(w))
    end
    function descr_lattice(p, w, K)
        F=low_coeffs(p,w,K); a0=real(F[K+1,K+1])
        pw=0f0; s2=ComplexF32(0); s4=ComplexF32(0); rings=zeros(Float32,3)
        for v in -K:K, u in -K:K
            (v==0&&u==0) && continue
            e=abs2(F[v+K+1,u+K+1]); φ=atan(Float32(v),Float32(u))
            pw+=e; s2+=e*cis(2φ); s4+=e*cis(4φ); rings[clamp(round(Int,hypot(v,u)),1,3)]+=e
        end
        pw<=0 && return (a0=a0, ac=0f0, E2=ComplexF32(0), E4=ComplexF32(0), rings=zeros(Float32,3), F=F)
        (a0=a0, ac=sqrt(pw), E2=s2/pw, E4=s4/pw, rings=rings./pw, F=F)
    end

    # -- polar sampling: the same integral at ω ∈ {1,2,3} × NTH orientations --
    const NTH = 12
    const RADII = (1f0, 2f0, 3f0)
    polar_kernels(P) = [ComplexF32[cis(-2f0π*ω*((j-1)*cos(θ)+(i-1)*sin(θ))/P) for i in 1:P, j in 1:P]
                        for ω in RADII, θ in range(0, π, length=NTH+1)[1:NTH]]
    function descr_polar(p, w, Ks)
        A = w .* p; sw = Float32(sum(w)); a0 = sum(A)/sw
        E = Float32[abs2(sum(Ks[a,b] .* A)/sw) for a in 1:length(RADII), b in 1:NTH]
        pw = sum(E)
        pw<=0 && return (a0=a0, ac=0f0, E2=ComplexF32(0), E4=ComplexF32(0), rings=zeros(Float32,3), F=E)
        s2=ComplexF32(0); s4=ComplexF32(0)
        for a in 1:length(RADII), b in 1:NTH
            θ=(b-1)*π/NTH; s2+=E[a,b]*cis(2θ); s4+=E[a,b]*cis(4θ)
        end
        (a0=a0, ac=sqrt(pw), E2=s2/pw, E4=s4/pw, rings=vec(sum(E,dims=2))./pw, F=E)
    end

    "N×N grid of cell descriptors over `img`. `ov` > 1 lets the analysis windows overlap."
    function grid_descriptors(img; N=3, K=3, ov=1.3, sampling="polar")
        S=size(img,1); cs=S/N; P=round(Int,ov*cs); isodd(P)||(P+=1)
        w=gauss_window(P); Ks = sampling=="polar" ? polar_kernels(P) : nothing
        [sampling=="polar" ? descr_polar(patch_at(img,(i-0.5)*cs,(j-0.5)*cs,P), w, Ks) :
                             descr_lattice(patch_at(img,(i-0.5)*cs,(j-0.5)*cs,P), w, K)
         for i in 1:N, j in 1:N]
    end

    stroke_angle(d) = mod(rad2deg(angle(d.E2))/2 + 90, 180)
    loop_score(d)   = (d.rings[1]+d.rings[2]) > 0 ? d.rings[2]/(d.rings[1]+d.rings[2]) : 0f0

    "The continuous 9-vector per cell — this, not the label, is the descriptor."
    cell_vector(d) = Float32[d.a0, d.ac, real(d.E2), imag(d.E2), abs(d.E2), abs(d.E4),
                             d.rings[1], d.rings[2], d.rings[3]]
    feature_vector(G) = reduce(vcat, cell_vector.(vec(G)))
    const FEATNAMES = ["a0","ac","ReE2","ImE2","|E2|","|E4|","e1","e2","e3"]
end

# ╔═╡ 90000000-0000-0000-0000-000000000008
# ---- DISPLAY AID ONLY: nearest-prototype label for a cell ----
begin
    "A readable label for a cell. Thresholds here exist purely to draw the figure —
     nothing downstream uses them, and no number quoted in the Notes comes from them."
    function cell_label(d; ink=0.04, aniso=0.20, loop=0.45)
        d.a0 < ink && return ("·", :grey)
        if abs(d.E2) >= aniso
            a = stroke_angle(d)
            # angles are measured with y pointing *down*, so 45° runs top-left→bottom-right
            k = argmin(abs.([0,45,90,135,180] .- a))
            return (["—","\\","|","/","—"][k], :yellow)
        end
        loop_score(d) >= loop && return ("O", :cyan)
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
**window overlap** (× cell): $(@bind ov Slider(1.0:0.1:2.0, default=1.3, show_value=true))
sampling: $(@bind sampling Select(["polar"=>"polar ω×θ (unbiased angles)", "lattice"=>"lattice (v,u) — literal DFT orders"]))
max order K (lattice only): $(@bind Klat Slider(1:1:5, default=3, show_value=true))

label thresholds *(display only)* — ink $(@bind t_ink Slider(0.0:0.01:0.20, default=0.04, show_value=true)) · anisotropy $(@bind t_aniso Slider(0.05:0.05:0.80, default=0.20, show_value=true)) · loop $(@bind t_loop Slider(0.20:0.05:0.90, default=0.45, show_value=true))
"""

# ╔═╡ 90000000-0000-0000-0000-00000000000a
begin
    img = startswith(src, "EMNIST: ") ?
          upsample(em.class_images[findfirst(==(replace(src,"EMNIST: "=>"")), em.class_names)][inst]) :
          synth_img(src, srad)
    G   = grid_descriptors(img; N=gridN, K=Klat, ov=ov, sampling=sampling)
    CS  = IMG/gridN
    PATCH = (p=round(Int, ov*CS); isodd(p) ? p : p+1)
    Markdown.parse("**source** `$(src)` — grid **$(gridN)×$(gridN)**, cell **$(round(CS,digits=1)) px**, " *
        "analysis patch **$(PATCH)×$(PATCH)** (Gaussian σ = $(round(PATCH/4,digits=1))) — sampling **$(sampling)** — " *
        "descriptor = **$(9*gridN^2) numbers** ($(gridN^2) cells × 9)")
end

# ╔═╡ 90000000-0000-0000-0000-00000000000b
let
    # clims up to 1.8 dims the ink to mid-grey so the yellow/green overlays read
    # everywhere — on the strokes as well as on the background
    p = heatmap(img; c=:grays, clims=(0f0,1.8f0), yflip=true, aspect_ratio=:equal,
                axis=false, ticks=false, cbar=false, xlims=(1,IMG), ylims=(1,IMG),
                size=(560,560), title="$(src) — tic-tac-toe signature", titlefontsize=10)
    for k in 1:(gridN-1)                                    # the grid itself
        plot!(p, [1,IMG], [k*CS,k*CS]; lc=:red, lw=1, alpha=0.5, label="")
        plot!(p, [k*CS,k*CS], [1,IMG]; lc=:red, lw=1, alpha=0.5, label="")
    end
    for i in 1:gridN, j in 1:gridN
        d = G[i,j]; cy = (i-0.5)*CS; cx = (j-0.5)*CS
        lab, col = cell_label(d; ink=t_ink, aniso=t_aniso, loop=t_loop)
        # oriented bar: angle = stroke angle, length ∝ anisotropy, opacity ∝ ink
        if d.a0 >= t_ink && abs(d.E2) > 0.02
            a = stroke_angle(d); ℓ = 0.45*CS*clamp(abs(d.E2)/0.6, 0.15, 1.0)
            plot!(p, [cx-ℓ*cosd(a), cx+ℓ*cosd(a)], [cy-ℓ*sind(a), cy+ℓ*sind(a)];
                  lc=:yellow, lw=4, alpha=0.85, label="")
        end
        if d.a0 >= t_ink && abs(d.E2) < t_aniso              # isotropic ⇒ blob/loop marker
            r = 0.30*CS*(loop_score(d) >= t_loop ? 1.0 : 0.55)
            θs = range(0,2π,length=60)
            plot!(p, cx .+ r.*cos.(θs), cy .+ r.*sin.(θs);
                  lc=(loop_score(d) >= t_loop ? :cyan : :orange), lw=3, alpha=0.9, label="")
        end
        # text in the cell corners (less likely to land on ink than the centre);
        # springgreen reads on both the black background and the white strokes
        annotate!(p, cx-0.44CS, cy-0.36CS, text(lab, 12, col, :left))
        annotate!(p, cx-0.44CS, cy+0.38CS,
                  text(@sprintf("a₀%.2f |E₂|%.2f", d.a0, abs(d.E2)), 6, :springgreen, :left))
    end
    p
end

# ╔═╡ 90000000-0000-0000-0000-00000000000c
let
    hdr = "**Per-cell descriptor** — `label` is the display aid; the row of numbers is the " *
          "actual continuous descriptor.\n\n" *
          "| cell | label | a₀ (ink) | ac | \\|E₂\\| | stroke° | \\|E₄\\| | e₁ | e₂ | e₃ | loop |\n" *
          "|:--|:--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n"
    rows = String[]
    for i in 1:gridN, j in 1:gridN
        d = G[i,j]; lab, _ = cell_label(d; ink=t_ink, aniso=t_aniso, loop=t_loop)
        push!(rows, @sprintf("| (%d,%d) | %s | %.3f | %.3f | %.3f | %.0f | %.3f | %.3f | %.3f | %.3f | %.3f |",
              i, j, lab, d.a0, d.ac, abs(d.E2), stroke_angle(d), abs(d.E4),
              d.rings[1], d.rings[2], d.rings[3], loop_score(d)))
    end
    Markdown.parse(hdr * join(rows, "\n"))
end

# ╔═╡ 90000000-0000-0000-0000-00000000000d
md"""
### Angular energy profile per cell

The power spectrum of each cell folded onto orientation: `Σ_ω |F(ω,θ)|²`, plotted
against the **stroke** angle it implies (`θ + 90°`, since a stroke's spectrum lies
*perpendicular* to it), 0–180°, polar sampling only. **One lobe = one stroke** at
that angle; **flat = no preferred orientation** (blob, loop, or crossing strokes that
cancel). The red line is the reported stroke angle, its height `|E₂|` — it should sit
on the lobe.
"""

# ╔═╡ 90000000-0000-0000-0000-00000000000e
let
    if sampling != "polar"
        md"*(switch sampling to **polar** to see the angular profiles)*"
    else
        # plotted against STROKE angle = (frequency angle + 90°) mod 180, so the lobe
        # sits at the orientation of the stroke that produced it
        θs = collect(range(0, π, length=NTH+1)[1:NTH])
        sa = mod.(rad2deg.(θs) .+ 90, 180); perm = sortperm(sa)
        panels = Any[]
        for i in 1:gridN, j in 1:gridN
            d = G[i,j]; prof = vec(sum(d.F, dims=1))
            m = maximum(prof); prof = m > 0 ? prof ./ m : prof
            pp = plot(sa[perm], prof[perm]; lw=2, lc=:steelblue, label="", ylims=(0,1.15),
                      xlims=(0,180), xticks=([0,45,90,135,180],["0","45","90","135","180"]),
                      title=@sprintf("(%d,%d) a₀=%.2f |E₂|=%.2f", i, j, d.a0, abs(d.E2)),
                      titlefontsize=7, tickfontsize=6, grid=false)
            if d.a0 >= t_ink && abs(d.E2) > 0.02
                a = stroke_angle(d); a = a >= 180 ? a-180 : a
                plot!(pp, [a,a], [0, clamp(abs(d.E2)/0.6,0,1)]; lc=:red, lw=3, label="")
            end
            push!(panels, pp)
        end
        plot(panels...; layout=(gridN,gridN), size=(240gridN, 190gridN))
    end
end

# ╔═╡ 90000000-0000-0000-0000-00000000000f
md"""
### Calibration: does the orientation readout actually read the angle?

A single bar swept through the centre cell, true angle vs. reported. This is the
measurement behind the *lattice vs polar* claim in the header.
"""

# ╔═╡ 90000000-0000-0000-0000-000000000010
let
    angles = 0:15:165
    read_p = Float64[]; read_l = Float64[]; an_p = Float64[]; an_l = Float64[]
    for θ in angles
        im = z(); L = 40
        drawbar!(im, 56-L*sind(θ), 56-L*cosd(θ), 56+L*sind(θ), 56+L*cosd(θ), srad)
        dp = grid_descriptors(im; N=3, ov=ov, sampling="polar")[2,2]
        dl = grid_descriptors(im; N=3, K=Klat, ov=ov, sampling="lattice")[2,2]
        push!(read_p, stroke_angle(dp)); push!(read_l, stroke_angle(dl))
        push!(an_p, abs(dp.E2));         push!(an_l, abs(dl.E2))
    end
    wrap(a,b) = abs(mod(a-b+90,180)-90)
    ep = [wrap(r,θ) for (r,θ) in zip(read_p,angles)]
    el = [wrap(r,θ) for (r,θ) in zip(read_l,angles)]
    p1 = plot(angles, ep; lw=2, marker=:circle, label=@sprintf("polar (max %.2f°)", maximum(ep)),
              xlabel="true angle (°)", ylabel="|error| (°)", legend=:top,
              title="orientation error", titlefontsize=9, guidefontsize=8, tickfontsize=7)
    plot!(p1, angles, el; lw=2, marker=:square, label=@sprintf("lattice K=%d (max %.2f°)", Klat, maximum(el)))
    p2 = plot(angles, an_p; lw=2, marker=:circle, label="polar", ylims=(0,1),
              xlabel="true angle (°)", ylabel="|E₂|", legend=:bottom,
              title="anisotropy of a single bar", titlefontsize=9, guidefontsize=8, tickfontsize=7)
    plot!(p2, angles, an_l; lw=2, marker=:square, label="lattice K=$(Klat)")
    plot(p1, p2; layout=(1,2), size=(950,340))
end

# ╔═╡ 90000000-0000-0000-0000-000000000011
md"""
### Is the signature diagnostic of letter identity?

Same protocol as `New_Gabor_FPE/KeyPointDiagnosticity.md`: EMNIST, 12 classes
(`O C I L T X K A H Y E F`), leave-one-out **nearest-class-mean** on the
standardised feature vector. **Chance = 8.3 %.** For reference, the best previous
descriptor in this project (global shape harmonics `|M1..6|` + radial profile)
reached **≈ 61 %**.

Tick the box to run it live (≈ 20–60 s depending on the settings).

run: $(@bind run_diag CheckBox(default=false)) · instances per class: $(@bind n_per Slider(5:5:30, default=15, show_value=true))
"""

# ╔═╡ 90000000-0000-0000-0000-000000000012
begin
    function eta2(X, y)
        n,p = size(X); out = zeros(Float64,p)
        for j in 1:p
            col = @view X[:,j]; gm = mean(col); tot = sum((col .- gm).^2); tot<=0 && continue
            bet = 0.0
            for c in unique(y); idx = findall(==(c),y); bet += length(idx)*(mean(col[idx])-gm)^2; end
            out[j] = bet/tot
        end; out
    end
    function loo_ncm(X, y)
        n,p = size(X); Z = copy(X)
        for j in 1:p
            s = std(@view Z[:,j]); s<=0 ? (Z[:,j] .= 0) : (Z[:,j] .= (Z[:,j] .- mean(Z[:,j]))./s)
        end
        classes = unique(y); correct = 0
        for i in 1:n
            best = Inf; bestc = classes[1]
            for c in classes
                idx = [k for k in 1:n if y[k]==c && k!=i]
                m = vec(mean(Z[idx,:], dims=1)); dd = sum(abs2, Z[i,:] .- m)
                dd < best && (best = dd; bestc = c)
            end
            bestc == y[i] && (correct += 1)
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
            for k in 1:min(n_per, length(v)); push!(imgs, upsample(v[k])); push!(labs, ci); end
        end
        X = reduce(vcat, [feature_vector(grid_descriptors(im; N=gridN, K=Klat, ov=ov,
                                                          sampling=sampling))' for im in imgs])
        e = eta2(X, labs)
        subsets = [("ink only  a₀", [1]), ("orientation only  (Re E₂, Im E₂)", [3,4]),
                   ("ink + orientation", [1,3,4]), ("ink + orientation + loop", [1,3,4,8]),
                   ("all 9 per cell", collect(1:9))]
        hdr = "**$(length(imgs)) images, $(gridN)×$(gridN) grid, $(sampling) sampling.** " *
              "Chance = $(round(100/length(CLASSES),digits=1)) %.\n\n" *
              "| feature subset | numbers/cell | total | LOO nearest-class-mean |\n|:--|--:|--:|--:|\n"
        rows = [ (cols = vcat([fi:9:size(X,2) for fi in fs]...);
                  @sprintf("| %s | %d | %d | **%.1f %%** |", nm, length(fs), length(cols),
                           100*loo_ncm(X[:,cols], labs)))
                 for (nm,fs) in subsets ]
        hdr2 = "\n\n**Per-feature η²** (mean over the $(gridN^2) cells): " *
               join([@sprintf("`%s` %.2f", FEATNAMES[k], mean(e[k:9:end])) for k in 1:9], " · ")
        Markdown.parse(hdr * join(rows,"\n") * hdr2)
    end
end

# ╔═╡ 90000000-0000-0000-0000-000000000014
md"""
### Notes — what was measured

**1. Orientation works, and it is the whole signal.** A bar swept through 12 angles
is read to within **0.16°** with polar sampling (up to **3.5°**, biased toward
0/45/90/135°, on the square DFT lattice — the lattice's own 4-fold symmetry leaking
into the 2nd harmonic). `|E₂|` is ≈ 0.51–0.57 for one clean stroke and **0.000** for
a plus, an X, a disc or a ring — the anisotropy really is an "is there a single
dominant stroke here" gauge.

**2. `|E₄|` does *not* detect crossings.** The hope was that two strokes ~90° apart
would ring the 4th harmonic. Measured (polar): single bar **0.18–0.23**, plus
**0.085**, X **0.142** — a *single* stroke has more `|E₄|` than a crossing. The
crossing region radiates broadband isotropic energy that dilutes the harmonic faster
than the second stroke adds to it. Use `|E₂| ≈ 0` **with** high `ac` as the
"something non-oriented is here" cue instead; `|E₄|` carries little (η² ≈ 0.30 on the
lattice, and most of *that* is lattice-alignment, not shape).

**3. Loop vs blob works only when the cell is matched to the figure.** With a disc
and a ring of the *same outer size*, both filling the cell, the radial profile
separates them cleanly — measured `e₂/e₁`: disc **0.015**, ring **1.33** (a ×90
difference; it is the `J₀` zero of the ring's transform falling between rings 1 and 2,
exactly as Bessel predicts). But the ratio confounds *hollowness* with *object size
relative to the cell*: at N=1 the same disc scores **0.52** and the same ring
**0.39** — inverted. On real EMNIST at 3×3 the centre-cell loop score does **not**
pick out `O` (mean 0.20 vs `C` 0.26, `F` 0.26), because an EMNIST `O`'s loop spans
the whole grid rather than sitting in one cell. At N=1 the whole-image score does
separate `O` from the rest (AUC **0.80**) but with the **opposite sign** to the loop
hypothesis — it is reading object scale and smoothness, not enclosure. Even the
synthetic demo shows how brittle it is: the same ring drawn with stroke radius **4**
scores **0.57** (reads as a loop at the default threshold) and with radius **6**
scores **0.41** (reads as a blob), because what the ratio really measures is *how much
of the cell the hole occupies*. **If you want loops, the cell has to be about the size
of the loop, or you need a different construct** (enclosure/topology, not a low-order
radial profile).

**4. The 9-cell signature is strongly diagnostic of letter identity** — the headline
result. 360 EMNIST instances, 12 classes, LOO nearest-class-mean, chance 8.3 %:

| descriptor | numbers | LOO accuracy |
|:--|--:|--:|
| ink only (`a₀` per cell) | 9 | 56.7 % |
| **orientation only (`Re E₂, Im E₂` per cell)** | **18** | **75.3 %** |
| **ink + orientation** | **27** | **76.1 %** |
| ink + orientation + loop score | 36 | **76.4 %** |
| all 9 features per cell | 81 | 74.2 % |
| all 9, but **no grid** (1×1) | 9 | 49.7 % |
| all 9, 2×2 grid | 36 | 71.4 % |
| all 9, 4×4 grid | 144 | 77.8 % |
| *(previous best in this project: global shape harmonics)* | *~20* | *≈ 61 %* |

Three things to read off it. **(a)** The grid is doing the work — the same nine
numbers computed over the whole image score 49.7 %, over a 3×3 grid 74.2 %.
**(b)** Two numbers per cell — the orientation vector — are enough; adding `ac`,
`|E₄|` and the ring profile *lowers* the unweighted nearest-mean score by diluting
good dimensions with noisy ones (the same artifact documented in
`KeyPointDiagnosticity.md`; η²-weighting recovers it to 77.2 %). **(c)** Bounding-box
normalising the letter before gridding does not help (72.8 % vs 74.2 %) — EMNIST is
already size-normalised.

**5. So the answer to the original question.** *Vertical / horizontal / diagonal per
area*: **yes**, cleanly, from two numbers per cell (`Re E₂, Im E₂`) — and those two
numbers are exactly a π-periodic orientation code, ready to bind as FPE with integer
frequencies. *Blob vs loop*: **no**, not at this grid scale, and not from a low-order
radial profile in general. And the honest surprise is that the crude 3×3 orientation
signature outperforms every more sophisticated local descriptor tried in this project
so far — which says the information that separates letters is **where the oriented
strokes are**, not what type of junction sits at each keypoint.

**6. Relation to the Gabor front end.** `F(ω,θ)` with a Gaussian window *is* a Gabor
filter response (see `LocalFourierPatches.jl`), so this whole notebook is: run a
Gabor bank at 3 scales × 12 orientations, pool the energy over 9 large regions, and
take the 2nd circular harmonic over orientation in each. Nothing here needs the
Fourier framing — but the Fourier framing is what makes "the first two or three
coefficients" the natural thing to keep.
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
