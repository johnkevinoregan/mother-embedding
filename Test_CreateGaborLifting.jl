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

# ╔═╡ 2376c093-f905-42c7-a350-cd1c529437b2
begin
    import Pkg
    Pkg.activate(@__DIR__)
end

# ╔═╡ a141848a-da33-44da-9c2e-500ac33fa56f
begin
    using PlutoUI
    using Plots
    using Colors
end

# ╔═╡ 59658887-1fc7-493e-95d8-d5b58422ea6c
begin
    # Config must be included before any component module that depends on it.
    include(joinpath(@__DIR__, "Config.jl"))
    using .Config
    include(joinpath(@__DIR__, "LoadEMNIST.jl"))
    using .LoadEMNIST
    include(joinpath(@__DIR__, "CreateGaborLifting.jl"))
    using .CreateGaborLifting
end

# ╔═╡ 91b99331-c212-46b4-b79f-9e09f5447925
md"""
# Test: CreateGaborLifting

Sanity-check notebook for the `CreateGaborLifting` module. Loads a few EMNIST
images (via `LoadEMNIST`), runs `gabor_lift` on a chosen image, and visualizes
the raw, unthresholded complex Gabor response: modulus and phase at every
sampled grid point, for a chosen orientation/scale, plus an at-a-glance grid
across every filter in the bank.
"""

# ╔═╡ 96694ae2-430d-4175-92bf-f7e1a1aef8c1
md"""
**Classes to load**: $(@bind n_classes_ui Slider(1:47, default=5, show_value=true))

**Images to scan**: $(@bind n_images_ui Slider(2000:2000:40000, default=3000, show_value=true))
"""

# ╔═╡ e7f59bd3-f6a2-49a1-a6ec-74b6bef73a5f
result = load_emnist(n_images_to_load=n_images_ui, n_classes=n_classes_ui)

# ╔═╡ e3965220-4119-49e0-bdc1-d2d488d9c7af
md"""
**Class**: $(@bind class_idx Slider(1:n_classes_ui, default=1, show_value=true))

**Image index within class**: $(@bind img_idx Slider(1:20, default=1, show_value=true))
"""

# ╔═╡ 38a43d4c-93bd-474f-a4e8-b5bbb3b084be
selected_image = result.class_images[class_idx][min(img_idx, length(result.class_images[class_idx]))]

# ╔═╡ 885373bb-6317-4256-bfef-a174e70e8a84
# Raw, unthresholded Gabor lift of the currently selected image — every grid
# point at every scale/orientation, each carrying modulus and phase.
# gabor_lift's own defaults already come from Config, so no need to pass them.
samples = gabor_lift(selected_image)

# ╔═╡ 7d8ebfaf-502c-439c-a027-8fb1d921cde8
md"""
## Single-filter overlay

Line color encodes **phase** (cyclic colormap); line opacity encodes
**modulus**, proportional to it, normalized to the largest modulus among
this filter's samples — so a zero response is fully transparent (invisible),
and faint lines are genuinely small (but non-zero) responses. Line length
equals the filter's wavelength $\lambda$; line orientation matches the
filter's $\theta$. No threshold is applied: every sampled grid point is
drawn, visible in proportion to its response.

**Orientation index**: $(@bind orient_idx Slider(1:length(ORIENTATIONS), default=2, show_value=true))

**Scale index**: $(@bind scale_idx Slider(1:length(SCALES), default=2, show_value=true))
"""

# ╔═╡ c02ccbe6-bb78-407b-96a1-bc11485a2019
# Map (phase, modulus) to an RGBA color: hue <- phase, alpha <- normalized
# modulus. Opacity is directly proportional to the modulus, so a zero
# response draws nothing at all.
function phase_modulus_color(phase::Real, modulus::Real, max_modulus::Real)
    hue = 180 * (phase / π + 1)  # phase in (-π, π] -> hue in [0, 360)
    alpha = max_modulus > 0 ? clamp(modulus / max_modulus, 0.0, 1.0) : 0.0
    return RGBA(convert(RGB, HSV(hue, 1.0, 1.0)), alpha)
end

# ╔═╡ bd917179-309b-4dee-9eb2-0c5e4161b2bb
let
    θ = ORIENTATIONS[orient_idx]
    λ = SCALES[scale_idx]
    filtered = filter(s -> s.θ == θ && s.s == Float32(scale_idx), samples)
    max_mod = maximum(s.modulus for s in filtered; init=0f0)

    img_size = IMG_SIZE
    display_img = size(selected_image) != (img_size, img_size) ?
        imresize_simple(selected_image, img_size) : selected_image

    p = heatmap(to_image(display_img), color=:grays, aspect_ratio=:equal,
                axis=false, colorbar=false,
                title="θ=$(round(θ*180/π, digits=1))°, λ=$(λ)px  ($(length(filtered)) samples)",
                xlim=(0.5, img_size + 0.5), ylim=(0.5, img_size + 0.5))

    # Line along the bar the filter responds to (the carrier varies along
    # (cos θ, sin θ) in (row, col); the bar runs perpendicular to that):
    # θ=0 -> horizontal.
    line_half = λ / 5
    dx = line_half * cos(θ)
    dy = line_half * sin(θ)
    for s in filtered
        cx = s.x * (img_size - 1) + 1
        cy = s.y * (img_size - 1) + 1
        # Match the image's to_image() vertical flip for overlay alignment.
        plot_y = img_size - cy + 1
        color = phase_modulus_color(s.phase, s.modulus, max_mod)
        plot!(p, [cx - dx, cx + dx], [plot_y - dy, plot_y + dy],
              color=color, linewidth=2.0, label=false)
    end
    p
end

# ╔═╡ 4b77102f-ddb8-4e1b-8de6-cf497a61424b
md"""
## Modulus distribution for the selected filter
"""

# ╔═╡ 6b0ccf27-8e25-40fe-8097-9b139a1f6454
let
    θ = ORIENTATIONS[orient_idx]
    filtered = filter(s -> s.θ == θ && s.s == Float32(scale_idx), samples)
    histogram([s.modulus for s in filtered], bins=30,
              xlabel="modulus", ylabel="count", legend=false,
              title="Modulus distribution — θ=$(round(θ*180/π, digits=1))°, scale=$scale_idx",
              size=(500, 280))
end

# ╔═╡ 606fa6a6-4ce9-48f0-b7a5-9e658bf9bbfe
md"""
## Full filter bank, at a glance

Every (scale, orientation) combination for the same image, modulus only
(brightness), no phase — a quick way to see how responses shift across the
whole bank at once.
"""

# ╔═╡ 14d46eef-2518-4ec8-b0a0-b6bdecb724a8
let
    img_size = IMG_SIZE
    display_img = size(selected_image) != (img_size, img_size) ?
        imresize_simple(selected_image, img_size) : selected_image

    panels = Plots.Plot[]
    for (s_idx, λ) in enumerate(SCALES)
        for θ in ORIENTATIONS
            filtered = filter(s -> s.θ == θ && s.s == Float32(s_idx), samples)
            max_mod = maximum(s.modulus for s in filtered; init=0f0)

            p = heatmap(to_image(display_img), color=:grays, aspect_ratio=:equal,
                        axis=false, colorbar=false, margin=0Plots.mm,
                        title="λ=$(Int(λ)),θ=$(round(Int, θ*180/π))°", titlefontsize=7,
                        xlim=(0.5, img_size + 0.5), ylim=(0.5, img_size + 0.5))
            # Bar orientation, as in the single-filter overlay: θ=0 -> horizontal.
            line_half = λ / 5
            dx = line_half * cos(θ)
            dy = line_half * sin(θ)
            for s in filtered
                cx = s.x * (img_size - 1) + 1
                cy = s.y * (img_size - 1) + 1
                plot_y = img_size - cy + 1
                alpha = max_mod > 0 ? clamp(s.modulus / max_mod, 0.0, 1.0) : 0.0
                plot!(p, [cx - dx, cx + dx], [plot_y - dy, plot_y + dy],
                      color=RGBA(1.0, 0.6, 0.0, alpha),
                      linewidth=1.2, label=false)
            end
            push!(panels, p)
        end
    end
    # Canvas proportioned to the panel grid (square panels + title strip),
    # zero margins: the image fills each slot.
    plot(panels..., layout=(length(SCALES), length(ORIENTATIONS)),
         size=(180 * length(ORIENTATIONS), 200 * length(SCALES)))
end

# ╔═╡ Cell order:
# ╠═2376c093-f905-42c7-a350-cd1c529437b2
# ╠═a141848a-da33-44da-9c2e-500ac33fa56f
# ╠═59658887-1fc7-493e-95d8-d5b58422ea6c
# ╟─91b99331-c212-46b4-b79f-9e09f5447925
# ╟─96694ae2-430d-4175-92bf-f7e1a1aef8c1
# ╠═e7f59bd3-f6a2-49a1-a6ec-74b6bef73a5f
# ╟─e3965220-4119-49e0-bdc1-d2d488d9c7af
# ╠═38a43d4c-93bd-474f-a4e8-b5bbb3b084be
# ╠═885373bb-6317-4256-bfef-a174e70e8a84
# ╟─7d8ebfaf-502c-439c-a027-8fb1d921cde8
# ╠═c02ccbe6-bb78-407b-96a1-bc11485a2019
# ╠═bd917179-309b-4dee-9eb2-0c5e4161b2bb
# ╟─4b77102f-ddb8-4e1b-8de6-cf497a61424b
# ╠═6b0ccf27-8e25-40fe-8097-9b139a1f6454
# ╟─606fa6a6-4ce9-48f0-b7a5-9e658bf9bbfe
# ╠═14d46eef-2518-4ec8-b0a0-b6bdecb724a8
