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

# ╔═╡ d4cbcd52-b5d9-4e8b-840c-6356269b19ef
begin
    import Pkg
    Pkg.activate(@__DIR__)
end

# ╔═╡ 26ff5b32-8764-459f-8c2d-fd5c5bd0c7d4
begin
    using PlutoUI
    using Plots
end

# ╔═╡ 5981a9ce-3c86-4cac-a395-f485a83f9c41
begin
    include(joinpath(@__DIR__, "LoadEMNIST.jl"))
    using .LoadEMNIST
end

# ╔═╡ 02222113-c7c9-441b-90e1-21d64fc7fae8
md"""
# Test: LoadEMNIST

Sanity-check notebook for the `LoadEMNIST` module. Loads a parametrizable number of
EMNIST balanced-train images and classes, and displays per-class counts, a sample
image grid, and a count bar chart, so you can visually confirm the loader is doing
what you expect before moving on to the next pipeline component.
"""

# ╔═╡ cdddcf79-4c9d-4dfc-801e-0ca3bee7b8fb
md"""
**Classes to load**: $(@bind n_classes_ui Slider(1:47, default=5, show_value=true))

**Images to scan**: $(@bind n_images_ui Slider(2000:2000:40000, default=20000, show_value=true))
"""

# ╔═╡ 4aef9f65-fd3a-4b42-aff4-21c2b92ca5ab
result = load_emnist(n_images_to_load=n_images_ui, n_classes=n_classes_ui)

# ╔═╡ 17211971-f39a-44e6-bc44-e01001db3a52
let
    lines = String["Per-class image counts (scanned $(n_images_ui) images):"]
    for c in 1:n_classes_ui
        push!(lines, "- **" * result.class_names[c] * "** (index " *
                     string(result.selected_classes[c]) * "): " *
                     string(length(result.class_images[c])) * " images")
    end
    Markdown.parse(join(lines, "\n"))
end

# ╔═╡ ea82376a-ac63-4d87-8338-bf203d30fc96
md"""
**Class to preview**: $(@bind preview_class_idx Slider(1:n_classes_ui, default=1, show_value=true))
"""

# ╔═╡ db7de237-6dba-4617-870f-5fcbe80e8266
let
    imgs = result.class_images[preview_class_idx]
    name = result.class_names[preview_class_idx]
    n_show = min(5, length(imgs))

    plots = Plots.Plot[]
    for k in 1:n_show
        push!(plots, heatmap(to_image(imgs[k]), color=:grays, aspect_ratio=:equal,
                              axis=false, colorbar=false, title="$(name) #$(k)",
                              titlefontsize=9))
    end
    for _ in (n_show + 1):5
        push!(plots, plot(framestyle=:none))
    end
    plot(plots..., layout=(1, 5), size=(1200, 260))
end

# ╔═╡ 9296320b-3403-4338-ba72-eaf164c795cd
let
    counts = length.(result.class_images)
    bar(result.class_names, counts, xlabel="class", ylabel="count",
        title="Images per class (of $(n_images_ui) scanned)", legend=false,
        size=(600, 300))
end

# ╔═╡ Cell order:
# ╠═d4cbcd52-b5d9-4e8b-840c-6356269b19ef
# ╠═26ff5b32-8764-459f-8c2d-fd5c5bd0c7d4
# ╠═5981a9ce-3c86-4cac-a395-f485a83f9c41
# ╟─02222113-c7c9-441b-90e1-21d64fc7fae8
# ╟─cdddcf79-4c9d-4dfc-801e-0ca3bee7b8fb
# ╠═4aef9f65-fd3a-4b42-aff4-21c2b92ca5ab
# ╠═17211971-f39a-44e6-bc44-e01001db3a52
# ╟─ea82376a-ac63-4d87-8338-bf203d30fc96
# ╠═db7de237-6dba-4617-870f-5fcbe80e8266
# ╠═9296320b-3403-4338-ba72-eaf164c795cd
