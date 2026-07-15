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

# ╔═╡ 10000000-0000-0000-0000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 10000000-0000-0000-0000-000000000002
begin
    using PlutoUI
    using Plots
    using Colors
    using ImageFiltering
    using Random
end

# ╔═╡ 10000000-0000-0000-0000-000000000003
begin
    # Only LoadEMNIST is reused from the project — for the real EMNIST samples.
    # Everything else in this notebook is a self-contained port of the Python
    # demo and does NOT use Config.jl / CreateGaborLifting.jl (those use a
    # different Gabor convention; see the note below).
    include(joinpath(@__DIR__, "..", "LoadEMNIST.jl"))
    using .LoadEMNIST
end

# ╔═╡ 10000000-0000-0000-0000-000000000004
md"""
# Gabor orientation demo (Julia port)

A faithful Julia/Pluto port of `gabor_orientation_demo.py`.

**What it does.** At *one* chosen scale we convolve a character image with a bank
of **complex (quadrature) Gabor filters** spanning orientation. At every pixel
we take the response **modulus** (oriented energy $\sqrt{\text{even}^2+\text{odd}^2}$)
and **phase**, then keep the orientation that maximises the modulus (the
*argmax orientation*). Four views come out of this:

1. **Flow field** — the argmax-orientation as an HSV image + a "needle" field
   (contour tangents; watch the flip at junctions).
2. **Why mask** — raw vs modulus-masked argmax (flat regions produce random
   speckle that the modulus channel suppresses).
3. **Profiles** — the *full* $\text{modulus}(\theta)$ curve at a clean stroke, a
   junction, and the background: **unimodal vs bimodal vs noise floor** — the
   information the argmax throws away.
4. **Phase** — phase at the winning orientation → **even (line)** vs
   **odd (edge)** classification.

**Orientation convention.** `θ` is the **contour-tangent** orientation. The
carrier modulates along the *normal*, so the stripes run along the tangent and
the filter matches a line/edge oriented at `θ`. Sanity check: a vertical line →
`θ = 90°`, a horizontal line → `0°`. This is enforced by the self-test cell below.

**Relation to the rest of the project.** This is a standalone analysis notebook,
so — like the Python original — it defines its own Gabor bank rather than
calling `CreateGaborLifting.gabor_kernel`. The two differ deliberately:

| | this notebook (Python port) | `CreateGaborLifting.jl` |
|---|---|---|
| envelope | anisotropic ($\sigma_t$ along tangent, $\sigma_n$ across) | isotropic-ish, single $\sigma$, aspect $\gamma$ |
| carrier | along the **normal** (θ = tangent) | along $x_\theta$ (θ = carrier axis) |
| DC | mean-subtracted (zero DC) | ideal-response normalized |
| readout | **argmax over θ** per pixel | full grid of (modulus, phase) tokens |

**Port notes.** `scipy.signal.fftconvolve(…, "same")` → `ImageFiltering.imfilter`
with `Fill(0)` (zero padding, matching SciPy). `imfilter` is technically
correlation, not convolution, which conjugates the response — but modulus is
unchanged and the even/odd test uses $|\sin(\text{phase})|$, so every figure is
identical. The font-render fallback for missing EMNIST is dropped: the lab
server has EMNIST, so we always use `LoadEMNIST`.
"""

# ╔═╡ 10000000-0000-0000-0000-000000000005
md"""
## Parameters

These are the `Config` dataclass fields, live. Changing any of them re-runs the
whole analysis (a few seconds — each figure convolves the image with `n_orient`
filters).

**λ carrier wavelength (the "scale")**: $(@bind lam_ui Slider(4.0:1.0:24.0, default=12.0, show_value=true))

**σₙ across-contour width (normal)**: $(@bind sigman_ui Slider(2.0:0.5:12.0, default=6.0, show_value=true))

**σₜ along-contour width (tangent → elongation)**: $(@bind sigmat_ui Slider(2.0:0.5:16.0, default=11.0, show_value=true))

**n orientations (field resolution)**: $(@bind norient_ui Slider(8:2:48, default=36, show_value=true))

**noise floor σ**: $(@bind noise_ui Slider(0.0:0.01:0.1, default=0.03, show_value=true))

**needle mask threshold (fraction of max modulus)**: $(@bind mask_ui Slider(0.0:0.02:0.4, default=0.14, show_value=true))

**display γ (modulus brightness)**: $(@bind gamma_ui Slider(0.3:0.05:1.0, default=0.6, show_value=true))
"""

# ╔═╡ 10000000-0000-0000-0000-000000000006
# The Python `Config` dataclass, as a plain NamedTuple. ksize (kernel support)
# and n_orient_prof (fine sampling for the point profiles) are left fixed.
cfg = (
    lam      = lam_ui,
    sigma_n  = sigman_ui,
    sigma_t  = sigmat_ui,
    ksize    = 41,
    n_orient = norient_ui,
    n_orient_prof = 180,
    upsample = 4,
    noise_sigma = noise_ui,
    seed     = 0,
    mask_thresh = mask_ui,
    gamma    = gamma_ui,
    needle_step = 10,
    needle_len  = 3.2,
    chars    = ("X", "T", "A", "K", "S"),
)

# ╔═╡ 10000000-0000-0000-0000-000000000007
md"""
## The Gabor bank and readout

`gabor(θ, cfg)` builds one complex Gabor whose preferred (stripe) orientation is
`θ`. `analyze(img, cfg)` convolves with the whole `θ`-bank and collapses to the
argmax orientation, its modulus, and the phase at that winner. `modulus_profile`
keeps the *whole* $\text{modulus}(\theta)$ curve at a single pixel.
"""

# ╔═╡ 10000000-0000-0000-0000-000000000008
# Complex Gabor whose preferred (stripe) orientation is θ.
# Note np.mgrid[-h:h+1, -h:h+1] gives (y=rows, x=cols); we mirror that so the
# orientation convention (θ = contour tangent) matches the Python source.
function gabor(θ, cfg)
    h = cfg.ksize ÷ 2
    g = zeros(ComplexF64, cfg.ksize, cfg.ksize)
    for (ii, y) in enumerate(-h:h), (jj, x) in enumerate(-h:h)
        xt =  x*cos(θ) + y*sin(θ)          # along tangent
        xn = -x*sin(θ) + y*cos(θ)          # along normal
        env = exp(-(xt^2/(2cfg.sigma_t^2) + xn^2/(2cfg.sigma_n^2)))
        car = cis(2π * xn / cfg.lam)       # modulate across the contour
        g[ii, jj] = env * car
    end
    return g .- sum(g)/length(g)           # zero DC → uniform regions give ~0
end

# ╔═╡ 10000000-0000-0000-0000-000000000009
# Return (orientation, modulus, winning_phase) fields via argmax over θ.
function analyze(img, cfg)
    thetas = collect(range(0, π; length=cfg.n_orient + 1))[1:end-1]   # endpoint=false
    Rs = [imfilter(img, centered(gabor(t, cfg)), Fill(zero(ComplexF64))) for t in thetas]
    hh, ww = size(img)
    orientation = zeros(hh, ww); modulus = zeros(hh, ww); win_phase = zeros(hh, ww)
    for i in 1:hh, j in 1:ww
        k = argmax(abs(Rs[t][i, j]) for t in 1:length(thetas))
        orientation[i, j] = thetas[k]
        modulus[i, j]     = abs(Rs[k][i, j])
        win_phase[i, j]   = angle(Rs[k][i, j])
    end
    return orientation, modulus, win_phase
end

# ╔═╡ 10000000-0000-0000-0000-00000000000a
# Full modulus(θ) at a single pixel (fine orientation sampling).
function modulus_profile(img, y, x, cfg)
    thetas = collect(range(0, π; length=cfg.n_orient_prof + 1))[1:end-1]
    vals = [abs(imfilter(img, centered(gabor(t, cfg)), Fill(zero(ComplexF64)))[y, x])
            for t in thetas]
    return thetas, vals
end

# ╔═╡ 10000000-0000-0000-0000-00000000000b
md"""
## Characters

We pull one real EMNIST sample per class — the **whole EMNIST-balanced alphabet**
(A–Z, the lowercase subset, and digits 0–9), selectable below — via `LoadEMNIST`,
then reproduce the Python preprocessing: **bilinear upsample 28 → 112** (Python
uses cubic `scipy.ndimage.zoom`; bilinear is close enough for this demo) and add
a Gaussian **noise floor**. Each character gets its own seed so results are
reproducible.
"""

# ╔═╡ 10000000-0000-0000-0000-00000000000c
# Simple bilinear upscale (Python used cubic scipy.ndimage.zoom).
function upsample_bilinear(img, factor)
    h, w = size(img)
    nh, nw = h*factor, w*factor
    out = zeros(Float64, nh, nw)
    for i in 1:nh, j in 1:nw
        si = 1 + (i-1)*(h-1)/(nh-1)
        sj = 1 + (j-1)*(w-1)/(nw-1)
        i0 = floor(Int, si); j0 = floor(Int, sj)
        i1 = min(i0+1, h);    j1 = min(j0+1, w)
        fi = si - i0; fj = sj - j0
        out[i, j] = (1-fi)*(1-fj)*img[i0, j0] + fi*(1-fj)*img[i1, j0] +
                    (1-fi)*fj*img[i0, j1]     + fi*fj*img[i1, j1]
    end
    return out
end

# ╔═╡ 10000000-0000-0000-0000-00000000000d
# Load all 47 EMNIST-balanced classes (A–Z, the lowercase subset a,b,d,e,f,g,h,
# n,q,r,t, and digits 0–9) so the whole alphabet is selectable below.
emnist = load_emnist(n_images_to_load=5000, n_classes=47)

# ╔═╡ 10000000-0000-0000-0000-00000000000e
name_to_class = Dict(emnist.class_names[i] => i for i in 1:length(emnist.class_names))

# ╔═╡ 10000000-0000-0000-0000-00000000001f
# Every class that actually turned up at least once in the scanned images, in
# EMNIST's canonical order (uppercase, then the lowercase subset, then digits).
available = [emnist.class_names[i] for i in 1:length(emnist.class_names)
             if !isempty(emnist.class_images[i])]

# ╔═╡ 10000000-0000-0000-0000-00000000000f
# One prepped 112×112 image per available character: upsample + noise floor.
# Int(c[1]) is the character's codepoint — a stable, always-positive per-class
# seed offset (works for digits and lowercase, not just A–Z).
imgs = Dict(c => begin
        base = emnist.class_images[name_to_class[c]][1]
        big  = upsample_bilinear(base, cfg.upsample)
        rng  = MersenneTwister(cfg.seed + Int(c[1]))
        clamp.(big .+ cfg.noise_sigma .* randn(rng, size(big)...), 0, 1)
    end for c in available)

# ╔═╡ 10000000-0000-0000-0000-000000000011
md"""
## Self-test: argmax orientation == contour tangent

A vertical white bar should report `θ ≈ 90°`; a horizontal bar `≈ 0°` (or `180°`).
"""

# ╔═╡ 10000000-0000-0000-0000-000000000012
let
    v = zeros(112, 112); v[:, 54:57] .= 1.0
    ov, _, _ = analyze(v, cfg)
    h = zeros(112, 112); h[54:57, :] .= 1.0
    oh, _, _ = analyze(h, cfg)
    dv = rad2deg(ov[56, 56]); dh = rad2deg(oh[56, 56])
    ok_v = abs(dv - 90) < 6
    ok_h = dh < 6 || dh > 174
    md"""
    - vertical line → **$(round(dv, digits=1))°** (expect ≈ 90) $(ok_v ? "✅" : "❌")
    - horizontal line → **$(round(dh, digits=1))°** (expect ≈ 0/180) $(ok_h ? "✅" : "❌")
    """
end

# ╔═╡ 10000000-0000-0000-0000-000000000013
md"""
## Pick a character

$(@bind which_char Select(available, default="X"))
"""

# ╔═╡ 10000000-0000-0000-0000-000000000010
# argmax analysis for the SELECTED character only (the expensive step). With the
# whole alphabet selectable it would be wasteful to analyze all 47 up front, so
# this recomputes on each character switch or Config-slider move.
sel = analyze(imgs[which_char], cfg)

# ╔═╡ 10000000-0000-0000-0000-000000000014
md"""
### Fig 1 — argmax-orientation flow field

Left: the character. Middle: **HSV** map, hue = orientation (mod π), brightness =
modulus. Right: the **needle** field (each needle a contour tangent, colored by
orientation, faded by modulus, hidden below the mask threshold). Watch the
needles *flip* where strokes cross.
"""

# ╔═╡ 10000000-0000-0000-0000-000000000015
# Hue = orientation (mod π, cyclic), Value = normalised modulus. → Matrix{RGB}.
hsv_field(orientation, modulus, gamma) =
    map((o, m) -> RGB(HSV((o/π)*360, 1.0, m)),
        orientation, (modulus ./ maximum(modulus)) .^ gamma)

# ╔═╡ 10000000-0000-0000-0000-000000000016
# Needle plot: contour tangents on black, colored by θ, faded by modulus.
function needle_plot(orientation, modulus, cfg)
    vmax = maximum(modulus)
    p = plot(background_color=:black, yflip=true, aspect_ratio=:equal,
             axis=false, grid=false, legend=false, size=(340, 340))
    for yy in 1:cfg.needle_step:size(orientation, 1),
        xx in 1:cfg.needle_step:size(orientation, 2)
        m = modulus[yy, xx] / vmax
        m < cfg.mask_thresh && continue
        t = orientation[yy, xx]
        dx = cos(t) * cfg.needle_len; dy = sin(t) * cfg.needle_len
        plot!(p, [xx-dx, xx+dx], [yy-dy, yy+dy],
              color=RGB(HSV(t/π*360, 1, 1)), lw=1.1, alpha=min(1.0, m*1.3))
    end
    return p
end

# ╔═╡ 10000000-0000-0000-0000-000000000017
let
    o, m, _ = sel
    p_char = plot(Gray.(imgs[which_char]), axis=false,
                  title="character $which_char", titlefontsize=10)
    p_hsv  = plot(hsv_field(o, m, cfg.gamma), axis=false,
                  title="orientation (hue=θ, bright=modulus)", titlefontsize=9)
    p_ndl  = needle_plot(o, m, cfg)
    plot!(p_ndl, title="argmax flow (masked)", titlefontsize=9)
    plot(p_char, p_hsv, p_ndl, layout=(1, 3), size=(900, 320))
end

# ╔═╡ 10000000-0000-0000-0000-000000000018
md"""
### Fig 2 — why the modulus mask is needed

Left: raw argmax with hue only (no modulus) — flat regions have no real
orientation, so the argmax is **random speckle**. Right: brightness = modulus, so
only the contours survive.
"""

# ╔═╡ 10000000-0000-0000-0000-000000000019
let
    o, m, _ = sel
    raw = map(o -> RGB(HSV((o/π)*360, 1.0, 1.0)), o)     # value=1 everywhere
    p_raw = plot(raw, axis=false, title="raw argmax, UNMASKED\n(flat → speckle)",
                 titlefontsize=9)
    p_msk = plot(hsv_field(o, m, cfg.gamma), axis=false,
                 title="masked by modulus\n(only contours survive)", titlefontsize=9)
    plot(p_raw, p_msk, layout=(1, 2), size=(680, 360))
end

# ╔═╡ 10000000-0000-0000-0000-00000000001a
md"""
### Fig 3 — the full modulus(θ) profile at three points

The argmax keeps only the peak; here is the whole curve. A **clean stroke** is
*unimodal*, a **junction** is *bimodal* (two strokes cross → two peaks — this is
the runner-up orientation the argmax discards), and the **background** is a flat
noise floor. Red dashed line = the argmax.
"""

# ╔═╡ 10000000-0000-0000-0000-00000000001b
# center of mass and a stroke/junction/background triplet, as in Python.
function pick_points(img)
    tot = sum(img); hh, ww = size(img)
    cy = round(Int, sum(i*img[i, j] for i in 1:hh, j in 1:ww) / tot)
    cx = round(Int, sum(j*img[i, j] for i in 1:hh, j in 1:ww) / tot)
    bright = [(i, j) for i in 1:hh, j in 1:ww if img[i, j] > 0.5]
    rad = [hypot(i-cy, j-cx) for (i, j) in bright]
    stroke = bright[argmin(abs.(rad .- 0.55*maximum(rad)))]
    return ["stroke (clean edge)" => stroke,
            "crossing (junction)" => (cy, cx),
            "background (flat)"    => (10, 10)]
end

# ╔═╡ 10000000-0000-0000-0000-00000000001c
let
    img = imgs[which_char]
    subs = Plots.Plot[]
    for (label, (y, x)) in pick_points(img)
        th, pr = modulus_profile(img, y, x, cfg)
        sp = plot(rad2deg.(th), pr, lw=2, legend=false, title=label,
                  titlefontsize=10, xlabel="orientation θ (deg)", xlims=(0, 180))
        vline!(sp, [rad2deg(th[argmax(pr)])], color=:red, ls=:dash, lw=1)
        push!(subs, sp)
    end
    plot!(subs[1], ylabel="modulus |Gabor(θ)|")
    plot(subs..., layout=(1, 3), size=(950, 300))
end

# ╔═╡ 10000000-0000-0000-0000-00000000001d
md"""
### Fig 4 — phase at the winning orientation → even/odd

The quadrature filter's **phase** at the winning θ tells line from edge: phase
near 0/π is **even** (a bright/dark *line*), phase near ±π/2 is **odd** (an
*edge*, one side brighter). We color by $|\sin(\text{phase})|$: **yellow =
even/line-like**, **teal = odd/edge-like**, brightness = modulus.
"""

# ╔═╡ 10000000-0000-0000-0000-00000000001e
let
    o, m, ph = sel
    edge = abs.(sin.(ph))                    # 0 = even/line, 1 = odd/edge
    v = (m ./ maximum(m)) .^ cfg.gamma
    yellow = (1.0, 0.8, 0.1); teal = (0.1, 0.7, 0.9)
    img = map((e, vv) -> RGB(
            clamp(((1-e)*yellow[1] + e*teal[1]) * vv, 0, 1),
            clamp(((1-e)*yellow[2] + e*teal[2]) * vv, 0, 1),
            clamp(((1-e)*yellow[3] + e*teal[3]) * vv, 0, 1)), edge, v)
    plot(img, axis=false, size=(360, 400),
         title="$which_char: phase at winning θ\nyellow = even/line   teal = odd/edge",
         titlefontsize=9)
end

# ╔═╡ Cell order:
# ╠═10000000-0000-0000-0000-000000000001
# ╠═10000000-0000-0000-0000-000000000002
# ╠═10000000-0000-0000-0000-000000000003
# ╟─10000000-0000-0000-0000-000000000004
# ╟─10000000-0000-0000-0000-000000000005
# ╠═10000000-0000-0000-0000-000000000006
# ╟─10000000-0000-0000-0000-000000000007
# ╠═10000000-0000-0000-0000-000000000008
# ╠═10000000-0000-0000-0000-000000000009
# ╠═10000000-0000-0000-0000-00000000000a
# ╟─10000000-0000-0000-0000-00000000000b
# ╠═10000000-0000-0000-0000-00000000000c
# ╠═10000000-0000-0000-0000-00000000000d
# ╠═10000000-0000-0000-0000-00000000000e
# ╠═10000000-0000-0000-0000-00000000001f
# ╠═10000000-0000-0000-0000-00000000000f
# ╠═10000000-0000-0000-0000-000000000010
# ╟─10000000-0000-0000-0000-000000000011
# ╟─10000000-0000-0000-0000-000000000012
# ╟─10000000-0000-0000-0000-000000000013
# ╟─10000000-0000-0000-0000-000000000014
# ╠═10000000-0000-0000-0000-000000000015
# ╠═10000000-0000-0000-0000-000000000016
# ╠═10000000-0000-0000-0000-000000000017
# ╟─10000000-0000-0000-0000-000000000018
# ╠═10000000-0000-0000-0000-000000000019
# ╟─10000000-0000-0000-0000-00000000001a
# ╠═10000000-0000-0000-0000-00000000001b
# ╠═10000000-0000-0000-0000-00000000001c
# ╟─10000000-0000-0000-0000-00000000001d
# ╠═10000000-0000-0000-0000-00000000001e
