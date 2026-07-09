### A Pluto.jl notebook ###
# v1.0.3

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

# ╔═╡ 11aa22bb-0c1d-4e2f-9a3b-101112131415
begin
    import Pkg
    Pkg.activate(@__DIR__)
end

# ╔═╡ 22bb33cc-1d2e-4f30-8b4c-212223242526
begin
    using PlutoUI
    using Plots
    using Colors
end

# ╔═╡ 33cc44dd-2e3f-4041-9c5d-313233343536
begin
    # Config must be included before any component module that depends on it.
    include(joinpath(@__DIR__, "Config.jl"))
    using .Config
    include(joinpath(@__DIR__, "CreateGaborLifting.jl"))
    using .CreateGaborLifting
    include(joinpath(@__DIR__, "CreateTJunctionLifting.jl"))
    using .CreateTJunctionLifting
end

# ╔═╡ 44dd55ee-3f40-4152-8d6e-414243444546
md"""
# View: Gabor kernels and T-junction kernel pairs

The complex Gabor kernel at phase φ is displayed as `real(kernel · e^(-iφ))` —
this is exactly the image pattern that would produce a response with phase φ,
so the phase slider shows *what the detector is looking for* at that phase.
Red = positive lobes, blue = negative. Display convention matches the other
notebooks: up on screen = up in the image, α = 270° points up.
"""

# ╔═╡ 55ee66ff-4051-4263-9e7f-515253545556
md"""
## Single Gabor kernel

**Scale index**: $(@bind s1_idx Slider(1:length(SCALES), default=3, show_value=true))

**Orientation index**: $(@bind o1_idx Slider(1:length(ORIENTATIONS), default=1, show_value=true))

**Phase (deg)**: $(@bind φ1_deg Slider(-180:22.5:180, default=0, show_value=true))
"""

# ╔═╡ 66ff77aa-5162-4374-8f80-616263646566
let
    λ = SCALES[s1_idx]
    θ = ORIENTATIONS[o1_idx]
    Kr = real.(gabor_kernel(λ, θ) .* cis(-deg2rad(φ1_deg)))
    m = maximum(abs, Kr)
    # :RdBu runs red->blue from low to high; reverse it so positive = red.
    heatmap(reverse(Kr, dims=1), color=cgrad(:RdBu, rev=true), clim=(-m, m),
            aspect_ratio=:equal, axis=false, colorbar=false, size=(320, 320),
            title="λ=$(Int(λ)), θ=$(round(θ * 180 / π, digits=1))°, φ=$(φ1_deg)°",
            titlefontsize=9)
end

# ╔═╡ 77aa88bb-6273-4485-9091-717273747576
md"""
## T-junction kernel pair

The stem kernel sits at the center; the crossbar kernel (orthogonal
orientation) sits one grid step away in direction α — exactly the geometry
`t_junction_lift` uses. Each kernel has its own phase.

**Scale index**: $(@bind s2_idx Slider(1:length(SCALES), default=3, show_value=true))

**Direction α**: $(@bind a2_idx Slider(1:8, default=7, show_value=true)) (α = (value−1)·45°)

**Stem phase (deg)**: $(@bind φ2s_deg Slider(-180:22.5:180, default=0, show_value=true))

**Crossbar phase (deg)**: $(@bind φ2c_deg Slider(-180:22.5:180, default=0, show_value=true))
"""

# ╔═╡ 88bb99cc-7384-4596-81a2-818283848586
let
    λ = SCALES[s2_idx]
    α = Float32((a2_idx - 1) * π / 4)
    step = max(1, round(Int, (1 - OVERLAP_FRAC) * λ))
    stem_θ = CreateTJunctionLifting.closest_orientation(mod(α, Float32(π)), ORIENTATIONS)
    cross_θ = CreateTJunctionLifting.closest_orientation(mod(α + Float32(π / 2), Float32(π)), ORIENTATIONS)
    Ks = real.(gabor_kernel(λ, stem_θ) .* cis(-deg2rad(φ2s_deg)))
    Kc = real.(gabor_kernel(λ, cross_θ) .* cis(-deg2rad(φ2c_deg)))
    radius = (size(Ks, 1) - 1) ÷ 2
    Δrow = round(Int, sin(α)) * step
    Δcol = round(Int, cos(α)) * step

    half = radius + step + 2
    canvas = zeros(Float32, 2half + 1, 2half + 1)
    c0 = half + 1
    for i in -radius:radius, j in -radius:radius
        canvas[c0 + i, c0 + j] += Ks[i + radius + 1, j + radius + 1]
        canvas[c0 + Δrow + i, c0 + Δcol + j] += Kc[i + radius + 1, j + radius + 1]
    end
    m = maximum(abs, canvas)
    heatmap(reverse(canvas, dims=1), color=cgrad(:RdBu, rev=true), clim=(-m, m),
            aspect_ratio=:equal, axis=false, colorbar=false, size=(360, 360),
            title="λ=$(Int(λ)), α=$((a2_idx - 1) * 45)°, stem φ=$(φ2s_deg)°, cross φ=$(φ2c_deg)°",
            titlefontsize=9)
end

# ╔═╡ 99ccaadd-8495-46a7-92b3-919293949596
# Synthetic T: bright strokes on dark background, stroke thickness 4 px
# (roughly matching an upscaled EMNIST stroke). Junction at about row 14,
# column 28.
synth_T = let
    img = zeros(Float32, IMG_SIZE, IMG_SIZE)
    img[13:16, 12:44] .= 1   # crossbar
    img[13:48, 27:30] .= 1   # stem
    img
end

# ╔═╡ aaddbbee-95a6-47b8-83c4-a1a2a3a4a5a6
md"""
## Sliding the kernel pair over a synthetic T

The kernel pair is overlaid (red = positive lobes, blue = negative) on the
synthetic T at a position you control: (`x`, `y`) is the stem kernel's
center; the crossbar kernel sits one grid step away in direction α. Below
the figure, the actual complex responses at that position are shown — set
the phase sliders to the *measured* phases to see the pattern the responses
correspond to.

**Scale index**: $(@bind s3_idx Slider(1:length(SCALES), default=3, show_value=true))

**Direction α**: $(@bind a3_idx Slider(1:8, default=7, show_value=true)) (α = (value−1)·45°)

**Stem phase (deg)**: $(@bind φ3s_deg Slider(-180:22.5:180, default=0, show_value=true))

**Crossbar phase (deg)**: $(@bind φ3c_deg Slider(-180:22.5:180, default=0, show_value=true))

**Stem x (column)**: $(@bind px3 Slider(1:IMG_SIZE, default=28, show_value=true))

**Stem y (row from top)**: $(@bind py3 Slider(1:IMG_SIZE, default=24, show_value=true))
"""

# ╔═╡ bbeeccff-a6b7-48c9-94d5-b1b2b3b4b5b6
let
    λ = SCALES[s3_idx]
    α = Float32((a3_idx - 1) * π / 4)
    step = max(1, round(Int, (1 - OVERLAP_FRAC) * λ))
    stem_θ = CreateTJunctionLifting.closest_orientation(mod(α, Float32(π)), ORIENTATIONS)
    cross_θ = CreateTJunctionLifting.closest_orientation(mod(α + Float32(π / 2), Float32(π)), ORIENTATIONS)
    Ks = real.(gabor_kernel(λ, stem_θ) .* cis(-deg2rad(φ3s_deg)))
    Kc = real.(gabor_kernel(λ, cross_θ) .* cis(-deg2rad(φ3c_deg)))
    radius = (size(Ks, 1) - 1) ÷ 2
    Δrow = round(Int, sin(α)) * step
    Δcol = round(Int, cos(α)) * step

    composite = zeros(Float32, IMG_SIZE, IMG_SIZE)
    for (Kr, r0, c0) in ((Ks, py3, px3), (Kc, py3 + Δrow, px3 + Δcol))
        for i in -radius:radius, j in -radius:radius
            r, c = r0 + i, c0 + j
            (1 <= r <= IMG_SIZE && 1 <= c <= IMG_SIZE) || continue
            composite[r, c] += Kr[i + radius + 1, j + radius + 1]
        end
    end
    composite ./= max(maximum(abs, composite), eps(Float32))

    rgb = [begin
               g = 0.5f0 * synth_T[i, j]
               k = composite[i, j]
               kp, kn = max(k, 0f0), max(-k, 0f0)
               RGB(clamp(g + kp, 0, 1), clamp(g - 0.5f0 * (kp + kn), 0, 1), clamp(g + kn, 0, 1))
           end
           for i in 1:IMG_SIZE, j in 1:IMG_SIZE]
    p_img = plot(rgb, aspect_ratio=:equal, axis=false,
                 title="λ=$(Int(λ)), α=$((a3_idx - 1) * 45)°, stem at (x=$(px3), y=$(py3))",
                 titlefontsize=9)

    # Responses at this position (complex kernels, replicate-padded), and the
    # combined feature response as used in the project:
    # min(modulus_stem, modulus_cross) × (1 + cos(Δphase))/2.
    Ksc = gabor_kernel(λ, stem_θ)
    Kcc = gabor_kernel(λ, cross_θ)
    function resp(K, r0, c0)
        acc = zero(ComplexF32)
        for i in -radius:radius, j in -radius:radius
            acc += K[i + radius + 1, j + radius + 1] *
                   synth_T[clamp(r0 + i, 1, IMG_SIZE), clamp(c0 + j, 1, IMG_SIZE)]
        end
        acc
    end
    rs = resp(Ksc, py3, px3)
    rc = resp(Kcc, py3 + Δrow, px3 + Δcol)
    strength = min(abs(rs), abs(rc)) / λ * (1 + cos(angle(rc) - angle(rs))) / 2
    p_bar = bar(["stem", "crossbar", "combined"], [abs(rs) / λ, abs(rc) / λ, strength],
                ylim=(-1, 1), legend=false, title="responses (modulus/λ)",
                titlefontsize=9)
    hline!(p_bar, [0], color=:black, label=false)

    plot(p_img, p_bar, layout=(1, 2), size=(780, 380))
end

# ╔═╡ ccffddaa-b7c8-49da-85e6-c1c2c3c4c5c6
let
    λ = SCALES[s3_idx]
    α = Float32((a3_idx - 1) * π / 4)
    step = max(1, round(Int, (1 - OVERLAP_FRAC) * λ))
    stem_θ = CreateTJunctionLifting.closest_orientation(mod(α, Float32(π)), ORIENTATIONS)
    cross_θ = CreateTJunctionLifting.closest_orientation(mod(α + Float32(π / 2), Float32(π)), ORIENTATIONS)
    Ks = gabor_kernel(λ, stem_θ)
    Kc = gabor_kernel(λ, cross_θ)
    radius = (size(Ks, 1) - 1) ÷ 2
    Δrow = round(Int, sin(α)) * step
    Δcol = round(Int, cos(α)) * step

    # Correlation of the complex kernel with the image at one position,
    # replicate-padded — the same quantity imfilter gives gabor_lift.
    function resp(K, r0, c0)
        acc = zero(ComplexF32)
        for i in -radius:radius, j in -radius:radius
            acc += K[i + radius + 1, j + radius + 1] *
                   synth_T[clamp(r0 + i, 1, IMG_SIZE), clamp(c0 + j, 1, IMG_SIZE)]
        end
        acc
    end
    rs = resp(Ks, py3, px3)
    rc = resp(Kc, py3 + Δrow, px3 + Δcol)
    compat = (1 + cos(angle(rc) - angle(rs))) / 2
    strength = min(abs(rs), abs(rc)) / λ * compat

    md"""
    **Measured responses at this position** (moduli include the /λ normalization):

    | | modulus/λ | phase |
    |---|---|---|
    | stem | $(round(abs(rs) / λ, digits=4)) | $(round(Int, angle(rs) * 180 / π))° |
    | crossbar | $(round(abs(rc) / λ, digits=4)) | $(round(Int, angle(rc) * 180 / π))° |

    Phase compatibility = $(round(compat, digits=3)), **strength = $(round(strength, digits=4))**
    """
end

# ╔═╡ Cell order:
# ╠═11aa22bb-0c1d-4e2f-9a3b-101112131415
# ╠═22bb33cc-1d2e-4f30-8b4c-212223242526
# ╠═33cc44dd-2e3f-4041-9c5d-313233343536
# ╟─44dd55ee-3f40-4152-8d6e-414243444546
# ╟─55ee66ff-4051-4263-9e7f-515253545556
# ╠═66ff77aa-5162-4374-8f80-616263646566
# ╟─77aa88bb-6273-4485-9091-717273747576
# ╠═88bb99cc-7384-4596-81a2-818283848586
# ╠═99ccaadd-8495-46a7-92b3-919293949596
# ╟─aaddbbee-95a6-47b8-83c4-a1a2a3a4a5a6
# ╠═bbeeccff-a6b7-48c9-94d5-b1b2b3b4b5b6
# ╠═ccffddaa-b7c8-49da-85e6-c1c2c3c4c5c6
