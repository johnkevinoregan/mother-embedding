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

# ╔═╡ 74451829-a028-4da2-93c6-b0b2f681ce57
begin
    import Pkg
    Pkg.activate(@__DIR__)
end

# ╔═╡ b66ef95a-3ed3-45c4-aa21-0d311a39e49d
begin
    using PlutoUI
    using Plots
    using Colors
end

# ╔═╡ f3782ead-6065-4105-85d3-d97d92b23b85
begin
    # Config must be included before any component module that depends on it.
    include(joinpath(@__DIR__, "Config.jl"))
    using .Config
    include(joinpath(@__DIR__, "LoadEMNIST.jl"))
    using .LoadEMNIST
    include(joinpath(@__DIR__, "CreateGaborLifting.jl"))
    using .CreateGaborLifting
    include(joinpath(@__DIR__, "CreateTJunctionLifting.jl"))
    using .CreateTJunctionLifting
end

# ╔═╡ 4b74faaf-f5b3-42f0-9081-3066d07bc52e
ORIENTATIONS

# ╔═╡ 3a76c864-f6c5-49fe-b13b-3767615ff12d
md"""
# Test: CreateTJunctionLifting — synthetic T

Same code as `Test_CreateTJunctionLifting.jl`, but the input is the
synthetic T from `View_GaborKernels.jl` instead of an EMNIST image — a
controlled stimulus for looking in detail at the combined stem/crossbar
responses. Runs `gabor_lift` on the synthetic T, then `t_junction_lift`
on that result, and visualizes the strongest candidates as T glyphs made
of two ellipses — direction `α` is 2π-periodic (8 possible values),
distinguishing e.g. a stem pointing left from one pointing right even
though both share the same underlying π-periodic Gabor orientation.
"""

# ╔═╡ 01375ee5-4227-4768-a96b-70b3c637659b
# Synthetic T instead of an EMNIST image: bright ([0,1]) strokes on black,
# thickness 4 px, junction at about row 14, column 28 — the same image
# as in View_GaborKernels.jl.
selected_image = let
    img = zeros(Float32, IMG_SIZE, IMG_SIZE)
    img[13:16, 12:44] .= 0   # crossbar
    img[13:48, 27:30] .= 1   # stem
    img
end

# ╔═╡ aa11bb22-cc33-dd44-ee55-ff6677889900
md"""
**Gabor aspect** (σ along ÷ across; 1 = isotropic, >1 = elongated along the stroke): $(@bind aspect_ui Slider(1.0:0.25:4.0, default=1.0, show_value=true))
"""

# ╔═╡ 98f93280-7e6d-4872-8dd9-b356ea033ebc
# Raw Gabor lift, then the T-junction lift computed directly from it — no FPE
# embedding involved. Both unthresholded; filtering happens only below, for
# display.
begin
    gsamples = gabor_lift(selected_image; aspect=aspect_ui)
    tsamples = t_junction_lift(gsamples)
end

# ╔═╡ f866d9bb-daa2-486e-93f0-01808450d4ad
md"""
## Strongest T-junction candidates

Only the top `n_show` candidates (by strength) are drawn, purely for
legibility — `t_junction_lift` itself applies no threshold; every grid
point/neighbor-direction pair is present in `tsamples`.

Each candidate also gets a dashed ellipse around its stem sample point —
the kernel's true receptive-field extent, λ across the stroke (radius
2σ = λ) and λ·aspect along it (a circle at aspect=1), much larger than the
glyph at the bigger scales: a glyph in a blank area can still carry a
genuine response if the receptive field reaches image content.

**Number of top candidates to show**: $(@bind n_show Slider(5:5:100, default=30, show_value=true))
"""

# ╔═╡ f0133005-19f2-467f-a4db-d4e27bed4f91
let
    img_size = IMG_SIZE
    display_img = size(selected_image) != (img_size, img_size) ?
        imresize_simple(selected_image, img_size) : selected_image

    sorted = sort(tsamples, by = t -> -t.strength)
    top = sorted[1:min(n_show, length(sorted))]
    max_strength = isempty(top) ? 1f0 : top[1].strength

    p = heatmap(to_image(display_img), color=:grays, aspect_ratio=:equal,
                axis=false, colorbar=false,
                title="Top $(length(top)) T-junction candidates",
                xlim=(0.5, img_size + 0.5), ylim=(0.5, img_size + 0.5))

    # A filled ellipse centered at (cx0, cy0), oriented along (dx, dy) and
    # sized 2λ/3 x λ/3 — the same glyph as the Gabor test notebook.
    function ellipse_at!(p, cx0, cy0, dx, dy, λ, color)
        L = hypot(dx, dy)
        ux, uy = dx / L, dy / L
        a, b = λ / 3 * aspect_ui, λ / 6   # along-stroke axis scales with the aspect slider
        ts = range(0, 2π, length=24)
        xs = [cx0 + a * cos(t) * ux - b * sin(t) * uy for t in ts]
        ys = [cy0 + a * cos(t) * uy + b * sin(t) * ux for t in ts]
        plot!(p, Shape(xs, ys), fillcolor=color, linewidth=0, label=false)
    end

    for t in top
        px = t.x * (img_size - 1) + 1
        py = t.y * (img_size - 1) + 1
        plot_y = img_size - py + 1   # match to_image()'s vertical flip

        λ = SCALES[round(Int, t.s)]
        step = max(1, round(Int, (1 - OVERLAP_FRAC) * λ))   # sample spacing at this scale

        # Hue = the stem Gabor's phase; opacity proportional to strength
        # (relative to the strongest shown), as in the Gabor test notebook.
        alpha = max_strength > 0 ? clamp(t.strength / max_strength, 0.0, 1.0) : 0.0
        hue = mod(t.phase, 2f0π) * 180 / π
        color = RGBA(RGB(HSV(hue, 1.0, 1.0)), alpha)

        # Stem: an ellipse centered on the sampled point, pointing along α
        # toward the adjacent sampled point where the crossbar orientation
        # was determined.
        nx = px + round(Int, cos(t.α)) * step
        ny = plot_y - round(Int, sin(t.α)) * step
        ellipse_at!(p, px, plot_y, cos(t.α), -sin(t.α), λ, color)

        # Crossbar: an ellipse centered on that adjacent sampled point,
        # perpendicular to α.
        ellipse_at!(p, nx, ny, cos(t.α + Float32(π / 2)), -sin(t.α + Float32(π / 2)), λ, color)

        # Dashed ellipse around the stem sample point: the kernel's true
        # receptive-field extent — λ across the stroke (radius 2σ = λ),
        # λ·aspect along it — reducing to a circle at aspect=1, and
        # much larger than the glyph at the bigger scales.
        circ = range(0, 2π, length=48)
        plot!(p, px .+ (λ*aspect_ui) .* cos.(circ) .* cos(t.α) .+ λ .* sin.(circ) .* sin(t.α),
              plot_y .- (λ*aspect_ui) .* cos.(circ) .* sin(t.α) .+ λ .* sin.(circ) .* cos(t.α),
              color=color, linestyle=:dash, linewidth=1, label=false)
    end
    p
end

# ╔═╡ 1397d847-3730-44d4-b786-57ac6b5e7e8b
md"""
## Strength distribution (all candidates, unthresholded)
"""

# ╔═╡ ad4d1025-5d3c-4179-8f81-94be47b143a9
histogram([t.strength for t in tsamples], bins=40, xlabel="strength",
          ylabel="count", legend=false, title="T-junction strength distribution",
          size=(500, 280))

# ╔═╡ 6df6fe80-93b7-40ae-8de5-0e4e310577ef
md"""
Total candidates: $(length(tsamples)) (from $(length(gsamples)) Gabor tokens × up to 8 neighbor directions each, minus border clipping).
"""

# ╔═╡ 1960a1cb-1d64-41a7-9655-91895af0a3d2
md"""
## Best orientation per location, by scale

One panel per scale. At every sampled grid point, only the strongest-strength
candidate (over all 8 directions at that point) is kept, and of those only
the top `n_show_panel` (by strength, per panel) are drawn, as T glyphs made
of two ellipses: the stem ellipse is centered on the sampled point, pointing
along α toward the adjacent sampled point where the crossbar orientation was
determined, and the crossbar ellipse is centered on that adjacent point,
perpendicular. Hue encodes the stem Gabor's phase; opacity is proportional
to strength, normalized once across all panels (to the strongest
best-per-location candidate at any scale), so panel brightness is
comparable between scales. The dashed ellipse around each stem point shows
the kernel's true receptive-field extent (λ across, λ·aspect along).

**Top candidates per panel**: $(@bind n_show_panel Slider(5:5:100, default=20, show_value=true))
"""

# ╔═╡ 1b236d8f-0cc5-4ba4-a564-12d4218af0fe
let
    img_size = IMG_SIZE
    display_img = size(selected_image) != (img_size, img_size) ?
        imresize_simple(selected_image, img_size) : selected_image

    # Best (highest-strength) candidate at each sampled location, per scale.
    best_per_point = Dict{Tuple{Float32,Float32,Float32}, eltype(tsamples)}()
    for t in tsamples
        key = (t.x, t.y, t.s)
        if !haskey(best_per_point, key) || t.strength > best_per_point[key].strength
            best_per_point[key] = t
        end
    end

    # A filled ellipse centered at (cx0, cy0), oriented along (dx, dy) and
    # sized 2λ/3 x λ/3 — the same glyph as the Gabor test notebook.
    function ellipse_at!(p, cx0, cy0, dx, dy, λ, color)
        L = hypot(dx, dy)
        ux, uy = dx / L, dy / L
        a, b = λ / 3 * aspect_ui, λ / 6   # along-stroke axis scales with the aspect slider
        ts = range(0, 2π, length=24)
        xs = [cx0 + a * cos(t) * ux - b * sin(t) * uy for t in ts]
        ys = [cy0 + a * cos(t) * uy + b * sin(t) * ux for t in ts]
        plot!(p, Shape(xs, ys), fillcolor=color, linewidth=0, label=false)
    end

    # One normalization across all panels: opacity is comparable between
    # scales, as in the Gabor test notebook's filter-bank grid.
    max_strength = maximum((t.strength for t in values(best_per_point)); init=0f0)

    panels = Plots.Plot[]
    for (s_idx, λ) in enumerate(SCALES)
        pts = sort([t for t in values(best_per_point) if t.s == Float32(s_idx)],
                   by = t -> -t.strength)
        pts = pts[1:min(n_show_panel, length(pts))]
        step = max(1, round(Int, (1 - OVERLAP_FRAC) * λ))   # sample spacing at this scale

        p = heatmap(to_image(display_img), color=:grays, aspect_ratio=:equal,
                    axis=false, colorbar=false, title="scale $(s_idx) (λ=$(Int(λ)))",
                    titlefontsize=9, xlim=(0.5, img_size + 0.5), ylim=(0.5, img_size + 0.5))

        for t in pts
            px = t.x * (img_size - 1) + 1
            py = t.y * (img_size - 1) + 1
            plot_y = img_size - py + 1   # match to_image()'s vertical flip

            # Hue = the stem Gabor's phase; opacity proportional to strength
            # (relative to the strongest candidate over all scales).
            alpha = max_strength > 0 ? clamp(t.strength / max_strength, 0.0, 1.0) : 0.0
            hue = mod(t.phase, 2f0π) * 180 / π
            color = RGBA(RGB(HSV(hue, 1.0, 1.0)), alpha)

            # Stem: an ellipse centered on the sampled point, pointing along
            # α toward the adjacent sampled point where the crossbar
            # orientation was determined.
            nx = px + round(Int, cos(t.α)) * step
            ny = plot_y - round(Int, sin(t.α)) * step
            ellipse_at!(p, px, plot_y, cos(t.α), -sin(t.α), λ, color)

            # Crossbar: an ellipse centered on that adjacent sampled point,
            # perpendicular to α.
            ellipse_at!(p, nx, ny, cos(t.α + Float32(π / 2)), -sin(t.α + Float32(π / 2)), λ, color)

            # Dashed circle of radius λ around the stem sample point: the
            # kernel's true receptive-field extent (radius 2σ = λ), which is
            # much larger than the glyph at the bigger scales.
            circ = range(0, 2π, length=48)
            plot!(p, px .+ (λ*aspect_ui) .* cos.(circ) .* cos(t.α) .+ λ .* sin.(circ) .* sin(t.α),
              plot_y .- (λ*aspect_ui) .* cos.(circ) .* sin(t.α) .+ λ .* sin.(circ) .* cos(t.α),
                  color=color, linestyle=:dash, linewidth=1, label=false)
        end
        push!(panels, p)
    end
    plot(panels..., layout=(1, length(SCALES)), size=(350 * length(SCALES), 380))
end

# ╔═╡ Cell order:
# ╠═74451829-a028-4da2-93c6-b0b2f681ce57
# ╠═b66ef95a-3ed3-45c4-aa21-0d311a39e49d
# ╠═f3782ead-6065-4105-85d3-d97d92b23b85
# ╠═4b74faaf-f5b3-42f0-9081-3066d07bc52e
# ╟─3a76c864-f6c5-49fe-b13b-3767615ff12d
# ╠═01375ee5-4227-4768-a96b-70b3c637659b
# ╟─aa11bb22-cc33-dd44-ee55-ff6677889900
# ╠═98f93280-7e6d-4872-8dd9-b356ea033ebc
# ╟─f866d9bb-daa2-486e-93f0-01808450d4ad
# ╠═f0133005-19f2-467f-a4db-d4e27bed4f91
# ╟─1397d847-3730-44d4-b786-57ac6b5e7e8b
# ╠═ad4d1025-5d3c-4179-8f81-94be47b143a9
# ╟─6df6fe80-93b7-40ae-8de5-0e4e310577ef
# ╟─1960a1cb-1d64-41a7-9655-91895af0a3d2
# ╠═1b236d8f-0cc5-4ba4-a564-12d4218af0fe
