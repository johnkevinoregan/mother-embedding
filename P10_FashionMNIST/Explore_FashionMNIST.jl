### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ f2000000-0000-0000-0000-000000000001
md"""
# The front end on Fashion-MNIST, layer by layer

Every dataset the front end has been tested on is a **line drawing on an empty background** —
EMNIST letters, synthetic single strokes. Fashion-MNIST is the first thing here that is not:
these are filled silhouettes with **texture** — weave, ribbing, sole tread — and multi-scale
oriented energy is a texture descriptor, which has never been examined.

This notebook shows what each stage actually computes on a chosen garment, as dense maps
*before* pooling. The pooled feature vector that a classifier sees is at the bottom, so you
can see what pooling keeps and what it discards.

Worth looking for while you click through:

* **the oriented energy** should light up on fabric texture, not only on the silhouette;
* **A₂ (end-stopping)** should fire at hems, cuffs and strap ends;
* **c₀ (ray count)** should be high where straps meet bags and where sleeves meet bodies;
* and the **winning scale** map is a local stroke-width readout — it should separate a thin
  sandal strap from a thick boot sole.
"""

# ╔═╡ f2000000-0000-0000-0000-000000000002
begin
    using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
    using Statistics, Printf, PlutoUI, Plots
    R = joinpath(@__DIR__, "..", "P0-8_RationalGaborFeatures")
    include(joinpath(@__DIR__, "..", "LoadEMNIST.module.jl"))
    include(joinpath(R, "GaborStack.module.jl"))
    include(joinpath(R, "AndLayer.module.jl"))
    include(joinpath(R, "RayHarmonics.module.jl"))
    include(joinpath(R, "Pooling.module.jl"))
    using .LoadEMNIST, .GaborStack, .AndLayer, .RayHarmonics, .Pooling
    md"*packages loaded*"
end

# ╔═╡ f2000000-0000-0000-0000-000000000003
begin
    const IMG = 112
    const CLASSES = ["T-shirt", "trouser", "pullover", "dress", "coat",
                     "sandal", "shirt", "sneaker", "bag", "ankle boot"]
    const DIR = joinpath(homedir(), "Julia", "DATABASES", "FashionMNIST")

    # The EMNIST reader un-transposes, because EMNIST stores its images transposed relative
    # to MNIST. On Fashion-MNIST that would *introduce* a transpose and lay every garment on
    # its side, so it is undone here. Labels come back 1-based already.
    RAW = read_emnist_images(joinpath(DIR, "train-images-idx3-ubyte"); max_images=4000)
    LAB = Int.(read_emnist_labels(joinpath(DIR, "train-labels-idx1-ubyte")))[1:4000]

    function upsample(a, N=IMG)
        H, W = size(a); out = zeros(Float32, N, N)
        @inbounds for i in 1:N, j in 1:N
            y = 1 + (i-1)*(H-1)/(N-1); x = 1 + (j-1)*(W-1)/(N-1)
            y0 = floor(Int, y); x0 = floor(Int, x); y1 = min(y0+1,H); x1 = min(x0+1,W)
            fy = y - y0; fx = x - x0
            out[i,j] = (1-fy)*(1-fx)*a[y0,x0] + fy*(1-fx)*a[y1,x0] +
                       (1-fy)*fx*a[y0,x1] + fy*fx*a[y1,x1]
        end
        out
    end
    getimg(k) = Float32.(upsample(permutedims(RAW[:, :, k])))

    HF, WF, _ = field_for((IMG,IMG), [2.0,3.742,7.0]; n_orient=[8,12,16], beta=[2.0,1.6,1.2])
    BANK = make_bank((HF,WF), [2.0,3.742,7.0]; imwidth=IMG, n_orient=[8,12,16], beta=[2.0,1.6,1.2])
    SCALES = unique(m.rho0 for m in BANK.meta if m.kind === :oriented)
    md"*4,000 images loaded; bank built for $(IMG)×$(IMG) at ρ = $(join(round.(SCALES; digits=2), ", "))*"
end

# ╔═╡ f2000000-0000-0000-0000-000000000004
md"""
## Pick an image

garment $(@bind cls Select([i => "$i — $(CLASSES[i])" for i in 1:10]))
instance $(@bind which Slider(1:40, default=1, show_value=true))

scale for the per-orientation panels $(@bind si Select([1 => "ρ 2.00 (coarse, λ 56 px)",
   2 => "ρ 3.74 (λ 30 px)", 3 => "ρ 7.00 (fine, λ 16 px)"]))
pooling grid for the feature vector $(@bind grid Slider(1:5, default=3, show_value=true))
"""

# ╔═╡ f2000000-0000-0000-0000-000000000005
begin
    idxs = findall(LAB .== cls)
    k = idxs[mod1(which, length(idxs))]
    IMGK = getimg(k)
    ES = energy_stack(IMGK, BANK)
    AM, AL = and_maps(ES, BANK.meta; forms=(:A1, :A2))
    RM, RL = ray_maps(ES, BANK.meta)
    hm(m; t="") = heatmap(m; c=:viridis, axis=false, ticks=false, colorbar=false,
                          yflip=true, aspect_ratio=1, title=t, titlefontsize=8)
    md"*image $(k) — $(CLASSES[cls])*"
end

# ╔═╡ f2000000-0000-0000-0000-000000000006
plot(heatmap(IMGK; c=:grays, axis=false, ticks=false, colorbar=false, yflip=true,
             aspect_ratio=1, title="input — $(CLASSES[cls])", titlefontsize=10),
     hm(sum(ES[:,:,[i for (i,m) in enumerate(BANK.meta) if m.kind === :oriented]], dims=3)[:,:,1];
        t="total oriented energy"),
     hm(ES[:,:,findfirst(m -> m.kind !== :oriented, BANK.meta)]; t="lowpass"),
     layout=(1,3), size=(760, 270))

# ╔═╡ f2000000-0000-0000-0000-000000000007
md"""
### Oriented energy at one scale, every orientation

Each panel is one carrier angle. A stroke runs **perpendicular** to its carrier, so a panel
labelled 0° responds to *vertical* structure. On a garment these should pick out the weave
and the silhouette separately.
"""

# ╔═╡ f2000000-0000-0000-0000-000000000008
begin
    ρ = SCALES[si]
    chans = [(i, m.theta) for (i, m) in enumerate(BANK.meta)
             if m.kind === :oriented && m.rho0 ≈ ρ]
    sort!(chans; by=last)
    plot([hm(ES[:,:,i]; t=@sprintf("%.0f°", rad2deg(θ))) for (i, θ) in chans]...;
         layout=(1, length(chans)), size=(150*length(chans), 175))
end

# ╔═╡ f2000000-0000-0000-0000-000000000009
md"""
### The conjunction layer and the ray transform, per scale

`A₁` is large where two perpendicular orientations coincide at a point. `A₂` fires at line
endings — hems, cuffs, strap ends. `c₀` counts contour branches radiating from a point, and
`|c₁|/c₀` is high at terminations, low where a contour passes straight through.
"""

# ╔═╡ f2000000-0000-0000-0000-00000000000a
begin
    pans = Any[]
    for (j, ρj) in enumerate(SCALES)
        a1 = findfirst(l -> l.form === :A1 && l.rho0 ≈ ρj, AL)
        a2 = findfirst(l -> l.form === :A2 && l.rho0 ≈ ρj, AL)
        r0 = findfirst(l -> l.form === :R0 && l.rho0 ≈ ρj, RL)
        r1 = findfirst(l -> l.form === :R1 && l.rho0 ≈ ρj, RL)
        push!(pans, hm(AM[:,:,a1]; t=@sprintf("A₁  ρ %.2f", ρj)))
        push!(pans, hm(AM[:,:,a2]; t=@sprintf("A₂  ρ %.2f", ρj)))
        push!(pans, hm(RM[:,:,r0]; t=@sprintf("c₀  ρ %.2f", ρj)))
        # |c1| is returned unnormalised; the ratio is formed here, as the front end does
        # after pooling — a per-pixel ratio would be meaningless where c₀ is small
        push!(pans, hm(RM[:,:,r1] ./ (RM[:,:,r0] .+ 1f-3*mean(RM[:,:,r0]));
                       t=@sprintf("|c₁|/c₀  ρ %.2f", ρj)))
    end
    plot(pans...; layout=(length(SCALES), 4), size=(760, 200*length(SCALES)))
end

# ╔═╡ f2000000-0000-0000-0000-00000000000b
md"""
### What survives pooling

The maps above are what the operators compute; the classifier never sees them. It sees the
grid-pooled vector below. Move the **grid** slider to watch spatial detail being traded for a
shorter description — at grid 1 the whole image becomes one number per channel.
"""

# ╔═╡ f2000000-0000-0000-0000-00000000000c
begin
    W = grid_weights(IMG, IMG, grid)
    f1, l1 = assemble(ES, BANK.meta, AM, AL,
                      PoolSpec(grid=grid, blocks=(:orient,:lowpass,:A1,:A2)); Wts=W)
    PR = pool_maps(RM, W)
    blocks = ["orient" => count(l -> startswith(l, "orient"), l1),
              "lowpass" => count(l -> startswith(l, "lowpass"), l1),
              "A1" => count(l -> startswith(l, "A1"), l1),
              "A2" => count(l -> startswith(l, "A2"), l1),
              "rays" => length(PR)]
    v = vcat(f1, vec(PR))
    bar(v; lw=0, c=:steelblue, legend=false, size=(880, 230),
        xlabel="feature", ylabel="value", grid=true, gridalpha=0.25,
        title="pooled feature vector at grid $(grid) — $(length(v)) numbers  (" *
              join(["$n×$c" for (n,c) in blocks], ", ") * ")", titlefontsize=9)
end

# ╔═╡ f2000000-0000-0000-0000-00000000000d
md"""
## What this dataset does and does not test

**Does:** texture, for the first time. Filled regions rather than thin strokes. Ten classes
with published baselines to calibrate against — roughly 84 % for a linear model on pixels,
88 % for an MLP, 93 % for a good CNN.

**Does not:** the background is black and uniform, there is exactly one centred object, local
contrast barely varies within a frame, and the resolution is 28×28 upsampled to 112. Those are
precisely the conditions under which **divisive normalisation** would matter, so that design
question stays open after this. It needs natural greyscale images — BSDS boundary detection is
the intended target.

One prediction worth checking with the grid slider: on the synthetic stroke dataset **grid 1
beat grid 3**, because position was randomised there and a fixed grid was pure liability. Here
garments are centred with their parts in consistent places — sleeves up, soles down — so grid 3
should win. If it does not, that earlier result was about pooling in general rather than about
position randomisation.
"""

# ╔═╡ Cell order:
# ╟─f2000000-0000-0000-0000-000000000001
# ╟─f2000000-0000-0000-0000-000000000002
# ╟─f2000000-0000-0000-0000-000000000003
# ╟─f2000000-0000-0000-0000-000000000004
# ╟─f2000000-0000-0000-0000-000000000005
# ╠═f2000000-0000-0000-0000-000000000006
# ╟─f2000000-0000-0000-0000-000000000007
# ╠═f2000000-0000-0000-0000-000000000008
# ╟─f2000000-0000-0000-0000-000000000009
# ╠═f2000000-0000-0000-0000-00000000000a
# ╟─f2000000-0000-0000-0000-00000000000b
# ╠═f2000000-0000-0000-0000-00000000000c
# ╟─f2000000-0000-0000-0000-00000000000d
