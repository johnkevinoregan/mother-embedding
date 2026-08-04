### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# This Pluto "bind" macro is defined so the notebook still loads outside Pluto.
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

# ╔═╡ a1000000-0000-0000-0000-000000000001
md"""
# The stimulus generator, interactively

Every image in `P9_P12_SimpleStrokeTests` is **one stroke on a uniform grey field**, and the label
is a vector of eight graded properties rather than a class. This notebook lets you drive the
generator by hand and watch both the picture and its target vector.

Two things are worth doing here that the static contact sheet cannot show:

1. **Move one parameter and watch only that change.** The contact sheet shows random draws,
   in which everything moves at once.
2. **Watch the target vector.** Several rows are *measured back out of the drawn geometry*
   rather than taken from the parameter that generated it — the corner angle especially,
   because the tangent discontinuity at a vertex is not what is visible when the arms
   themselves curve. Set a curved base and a corner and you can see the two diverge.
"""

# ╔═╡ a1000000-0000-0000-0000-000000000002
begin
    using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
    using Random, Statistics, Printf, PlutoUI, Plots
    include(joinpath(@__DIR__, "Contours.module.jl"))
    using .Contours
    md"*packages loaded*"
end

# ╔═╡ a1000000-0000-0000-0000-000000000003
md"""
## Geometry

| | |
|:--|:--|
| curvature κ (1/px) — 0 is dead straight $(@bind κ Slider(0.0:0.001:0.028, default=0.012, show_value=true)) | closed loop $(@bind closed CheckBox(false)) |
| arclength (px) $(@bind arclen Slider(68.0:2.0:280.0, default=90.0, show_value=true)) | aspect (1 = circular) $(@bind aspect Slider(0.62:0.01:1.0, default=1.0, show_value=true)) |

**event** $(@bind ev Select(["none" => "none — plain stroke", "gap" => "gap — a break",
   "kink" => "kink — a corner", "tee" => "tee — 3 arms", "cross" => "cross — 4 arms"]))

| | |
|:--|:--|
| vertex angle (°), for a kink $(@bind vang Slider(32:1:180, default=90, show_value=true)) | gap (px), for a gap $(@bind gap Slider(1.0:0.5:40.0, default=8.0, show_value=true)) |
| branch angle (°), for tee/cross $(@bind branch Slider(-90:1:90, default=60, show_value=true)) | |
"""

# ╔═╡ a1000000-0000-0000-0000-000000000004
md"""
## How it is drawn — the nuisances

None of these change the geometry. A good description of the picture should be unaffected by
all of them, which is what the extrapolation splits in `RESULTS.md` test.

| | |
|:--|:--|
| thickness (px) $(@bind w Slider(3.0:0.5:12.0, default=6.0, show_value=true)) | fuzziness — edge ramp (px) $(@bind ramp Slider(0.8:0.2:20.0, default=2.0, show_value=true)) |
| contrast, as a fraction of headroom $(@bind amp Slider(0.35:0.01:1.0, default=0.9, show_value=true)) | background level $(@bind bg Slider(0.40:0.01:0.60, default=0.50, show_value=true)) |
| polarity $(@bind pol Select([1 => "light stroke", -1 => "dark stroke"])) | noise $(@bind noise Slider(0.0:0.005:0.06, default=0.02, show_value=true)) |
| rotation (°) $(@bind rot Slider(0:5:355, default=35, show_value=true)) | seed $(@bind seed Slider(1:50, default=7, show_value=true)) |
"""

# ╔═╡ a1000000-0000-0000-0000-000000000005
begin
    p = Params(closed ? max(κ, 1/45) : κ,
               closed ? 2π : min(arclen * max(κ, 1e-9), 2π/3),
               closed ? 2π/max(κ, 1/45) : arclen,
               aspect, Symbol(ev), gap, deg2rad(180 - vang), deg2rad(branch),
               w, ramp, amp * 0.88 * min(bg, 1 - bg), pol, bg, noise)
    img, tgt, _ = stimulus(p, MersenneTwister(seed); N=112, rot=deg2rad(rot), at=(56.0, 56.0))
    heatmap(img; c=:grays, clims=(0,1), yflip=true, aspect_ratio=1, axis=false,
            ticks=false, colorbar=false, size=(430, 430))
end

# ╔═╡ a1000000-0000-0000-0000-000000000006
md"""
## The target vector

What a model is asked to predict from this image. **Nothing is masked** — every row is
defined for every stimulus, which is why the corner row is an *angle* (180° = passes
straight through) rather than an angle plus a separate "is there a corner" flag.
"""

# ╔═╡ a1000000-0000-0000-0000-000000000007
Markdown.parse("""
| property | value | |
|:--|--:|:--|
| `curvedness` | $(round(tgt[1], digits=3)) | 0 = straight |
| `brokenness` | $(round(tgt[2], digits=3)) | gap in units of stroke width |
| `closedness` | $(Int(round(tgt[3]))) | $(tgt[3] > 0.5 ? "closed loop" : "open") |
| `vangle` | $(round(tgt[4], digits=1))° | $(String(band_of(tgt[4]))) |
| `arms` | $(Int(round(tgt[5]))) | $(tgt[5] == 2 ? "no junction" : tgt[5] == 3 ? "tee" : "crossing") |
| `thickness` | $(round(tgt[6], digits=1)) px | |
| `fuzziness` | $(round(tgt[7], digits=1)) px | |
| `polarity` | $(Int(tgt[8])) | $(tgt[8] > 0 ? "light on dark" : "dark on light") |
""")

# ╔═╡ a1000000-0000-0000-0000-000000000008
md"""
## Try these

**The corner angle is measured, not asserted.** Set `event = kink`, `vangle = 90`, and
`curvature = 0`. The reported `vangle` is 90. Now raise curvature to 0.028 without touching
anything else — the reported value stays at 90, because it is computed as *excess* turn: the
jump in direction at the vertex, over and above the turn the smooth curvature accounts for.
An earlier version measured the angle between the two limbs over a fixed window, and on a
curved base that put **a third of all labels in the wrong band**.

**Fuzziness does not change contrast.** Sweep `fuzziness` from 0.8 to 20 with everything
else fixed. The stroke blurs but its peak stays put. Without that normalisation, a ramp wider
than the stroke never reaches full amplitude, so blurring would silently dim the stroke and
two target rows would be entangled.

**Polarity is invisible to the front end.** Flip `polarity` and nothing about the geometry
changes. Our features are identical to seven decimal places across that flip — which is why,
trained on light strokes and tested on dark, they score exactly what they scored i.i.d.,
while a pixel model falls below chance.

**Why closed loops are always big.** Tick `closed` and watch `arclength` jump. In a frame of
this size, a closed contour can be about π·D long while an open arc capped at 2π/3 of turn
is limited to ~1.2·D. Closure *buys* length — which is why the `closedness` row is
confounded and should not be read as a closure result. See `RESULTS.md`.
"""

# ╔═╡ Cell order:
# ╟─a1000000-0000-0000-0000-000000000001
# ╟─a1000000-0000-0000-0000-000000000002
# ╟─a1000000-0000-0000-0000-000000000003
# ╟─a1000000-0000-0000-0000-000000000004
# ╠═a1000000-0000-0000-0000-000000000005
# ╟─a1000000-0000-0000-0000-000000000006
# ╟─a1000000-0000-0000-0000-000000000007
# ╟─a1000000-0000-0000-0000-000000000008
