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
# column 28. Pixel values are in [0, 1], the range every input image
# (EMNIST included) uses — 1 = full brightness.
synth_T = let
    img = zeros(Float32, IMG_SIZE, IMG_SIZE)
    img[13:16, 12:44] .= 1   # crossbar
    img[13:48, 27:30] .= 1   # stem
    img
end

# ╔═╡ aaddbbee-95a6-47b8-83c4-a1a2a3a4a5a6
md"""
## Sliding the kernel pair over a synthetic T

Each kernel is shown in its own panel (red = positive weights, blue =
negative), overlaid on the synthetic T at a position you control: (`x`, `y`)
is the stem kernel's center; the crossbar kernel sits one grid step away in
direction α. The two kernels are correlated with the image **independently**
— each panel shows exactly the weights of one linear template. The combined
feature is computed from the two scalar responses afterwards, nonlinearly
(min of the moduli × phase compatibility); it is *not* the response of any
summed pattern.

The bar panel shows, for each kernel, the projection of its response onto
the displayed template `P(φ)`, onto the same template a quarter-cycle later
`P(φ+90°)`, and the modulus `|r| = √(P(φ)² + P(φ+90°)²)` — the maximum
projection achievable over all φ. Set a phase slider to the *measured*
phase (table below) and watch `P(φ)` rise to the modulus while `P(φ+90°)`
drops to zero. The kernels are *ideal-response normalized* (a white bar
exactly filling the positive lobe responds 1), so every value here is
directly the fraction of the best a [0,1]-valued image can possibly do.

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
    Ksc = gabor_kernel(λ, stem_θ)
    Kcc = gabor_kernel(λ, cross_θ)
    radius = (size(Ksc, 1) - 1) ÷ 2
    Δrow = round(Int, sin(α)) * step
    Δcol = round(Int, cos(α)) * step

    # One overlay per kernel: each panel shows exactly the weights of one
    # linear template — the two kernels are correlated with the image
    # independently, nothing is ever summed across them.
    function overlay(K, φ_deg, r0, c0, ttl)
        Kr = real.(K .* cis(-deg2rad(φ_deg)))
        pattern = zeros(Float32, IMG_SIZE, IMG_SIZE)
        for i in -radius:radius, j in -radius:radius
            r, c = r0 + i, c0 + j
            (1 <= r <= IMG_SIZE && 1 <= c <= IMG_SIZE) || continue
            pattern[r, c] = Kr[i + radius + 1, j + radius + 1]
        end
        pattern ./= max(maximum(abs, pattern), eps(Float32))
        # Blend the grayscale image toward pure red (positive kernel weights)
        # or pure blue (negative), so the tint stays visible on white strokes
        # as well as on black background.
        rgb = [begin
                   g = synth_T[i, j]
                   k = pattern[i, j]
                   w = abs(k)
                   RGB((1 - w) * g + (k > 0 ? w : 0f0),
                       (1 - w) * g,
                       (1 - w) * g + (k < 0 ? w : 0f0))
               end
               for i in 1:IMG_SIZE, j in 1:IMG_SIZE]
        plot(rgb, aspect_ratio=:equal, axis=false, title=ttl, titlefontsize=9)
    end
    p_stem = overlay(Ksc, φ3s_deg, py3, px3,
                     "stem kernel — λ=$(Int(λ)), α=$((a3_idx - 1) * 45)°, (x=$(px3), y=$(py3))")
    p_cross = overlay(Kcc, φ3c_deg, py3 + Δrow, px3 + Δcol, "crossbar kernel")

    # Responses at these positions (complex kernels, replicate-padded), and
    # the combined feature response as used in the project:
    # min(modulus_stem, modulus_cross) × (1 + cos(Δphase))/2.
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
    # Kernels are ideal-response normalized, so these values are already
    # fractions of the best any [0,1]-valued image could produce.
    strength = min(abs(rs), abs(rc)) * (1 + cos(angle(rc) - angle(rs))) / 2

    # Projection of a response onto the template displayed at phase φ; the
    # modulus is the root-sum-of-squares of the projections at φ and φ+90°,
    # so P(φ) reaches the modulus exactly when φ is the measured phase.
    proj(r, φ_deg) = real(r * cis(-deg2rad(φ_deg)))
    values = [proj(rs, φ3s_deg), proj(rs, φ3s_deg + 90), abs(rs),
              proj(rc, φ3c_deg), proj(rc, φ3c_deg + 90), abs(rc),
              strength]
    labels = ["s P(φ)", "s P(φ+90°)", "s |r|", "c P(φ)", "c P(φ+90°)", "c |r|", "comb"]
    bar_colors = [:steelblue, :steelblue, :steelblue,
                  :darkorange, :darkorange, :darkorange, :firebrick]
    p_bar = bar(labels, values, color=bar_colors, ylim=(-1, 1), legend=false,
                title="responses (fraction of ideal): s=stem, c=crossbar", titlefontsize=9,
                xrotation=30, tickfontsize=7)
    hline!(p_bar, [0], color=:black, label=false)

    plot(p_stem, p_cross, p_bar, layout=(1, 3), size=(1140, 380))
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
    strength = min(abs(rs), abs(rc)) * compat

    md"""
    **Measured responses at this position** (kernels are ideal-response
    normalized, so 1.0 = perfectly matched full-contrast bar):

    | | modulus (fraction of ideal) | phase |
    |---|---|---|
    | stem | $(round(abs(rs), digits=4)) | $(round(Int, angle(rs) * 180 / π))° |
    | crossbar | $(round(abs(rc), digits=4)) | $(round(Int, angle(rc) * 180 / π))° |

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
