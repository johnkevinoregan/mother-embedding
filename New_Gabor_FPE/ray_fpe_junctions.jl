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

# ╔═╡ b0000002-0001-4000-8000-000000000002
begin
    using LinearAlgebra
    using Statistics
    using Random
    using FFTW
    using Plots
    using PlutoUI
    Random.seed!(42)
end

# ╔═╡ b0000001-0001-4000-8000-000000000001
md"""
# Ray-harmonic / FPE junction detection

**The claim this notebook tests:** T-junctions, corners, endpoints and crossings
can be read out from a Gabor lift by *linear projections only* — no ring
run-length analysis, no peak counting, no `if npk >= 2`, no thresholds.

**Why the previous attempt failed.** The orientation fibre `E(x,y,θ)` is
**π-periodic**. At the centre of an L-corner, a T-junction and an X-crossing,
`E(θ)` is bimodal with peaks at the *same* orientations — the three are nearly
indistinguishable (cosine similarity ≈ 0.9). Junction *type* is a **2π
(directional)** property: it is about which **rays** leave the point, and
orientation mod π cannot count rays.

**The fix.** Build a *ray profile*

```
R(p, φ) = E(p + d·u(φ),  θ = φ mod π),     φ ∈ [0, 2π),  u(φ) = (cos φ, sin φ)
```

"is there a contour at distance `d` in direction `φ`, oriented *along* `φ`?"
`R` is 2π-periodic and literally has one lobe per branch. Its circular
harmonics `cₙ[R]` are the type signature — and those harmonics are exactly
what an **FPE bundle with integer frequencies** computes.
"""

# ╔═╡ b0000003-0001-4000-8000-000000000003
md"""
## 1. Constants

Values transferred from a verified reference implementation. `LAM`/`SIGMA_*` are
in pixels of the **upsampled** image; `D_RAY` is the ray-probe radius.
"""

# ╔═╡ b0000004-0001-4000-8000-000000000004
begin
    const IMG        = 112          # analysis size (28 × 4)
    const N_THETA    = 48           # orientation channels (mod π)
    const K_PHI      = 96           # ray directions (mod 2π)
    const N_MAX      = 4            # highest harmonic
    const LAM        = 12.0f0       # Gabor wavelength
    const SIGMA_N    = 6.0f0        # across-contour envelope
    const SIGMA_T    = 10.0f0       # along-contour envelope (elongated)
    const KSIZE      = 39
    const D_RAY      = 15.0f0       # ray probe radius
    const THETAS     = Float32.(range(0, π, length=N_THETA+1)[1:N_THETA])
    const PHIS       = Float32.(range(0, 2π, length=K_PHI+1)[1:K_PHI])
end

# ╔═╡ b0000005-0001-4000-8000-000000000005
md"""
## 2. Gabor lift → `E[θ, y, x]`

`θ` is the **contour-tangent** orientation: the carrier modulates along the
normal, so the filter matches a line/edge at `θ`. Sanity-checked in §7.
"""

# ╔═╡ b0000006-0001-4000-8000-000000000006
function gabor_kernel(θ::Float32; λ=LAM, σn=SIGMA_N, σt=SIGMA_T, ks=KSIZE)
    h = ks ÷ 2
    K = zeros(ComplexF32, ks, ks)
    c, s = cos(θ), sin(θ)
    for i in -h:h, j in -h:h
        y, x = Float32(i), Float32(j)
        xt =  x*c + y*s          # along tangent
        xn = -x*s + y*c          # along normal
        env = exp(-(xt^2/(2σt^2) + xn^2/(2σn^2)))
        K[i+h+1, j+h+1] = env * cis(2f0π * xn / λ)
    end
    K .-= mean(K)                # zero DC: flat regions → 0
    return K
end

# ╔═╡ b0000007-0001-4000-8000-000000000007
# FFT-based 'same' convolution (no ImageFiltering dependency)
function conv_same(img::Matrix{Float32}, K::Matrix{ComplexF32})
    H, W   = size(img)
    kh, kw = size(K)
    ph, pw = H + kh - 1, W + kw - 1
    A = zeros(ComplexF32, ph, pw); A[1:H, 1:W] .= img
    B = zeros(ComplexF32, ph, pw); B[1:kh, 1:kw] .= K
    full = ifft(fft(A) .* fft(B))
    o1, o2 = kh ÷ 2, kw ÷ 2
    return full[o1+1:o1+H, o2+1:o2+W]
end

# ╔═╡ b0000008-0001-4000-8000-000000000008
"""Oriented energy stack: E[θi, y, x] = |Gabor_θi * img| (phase-invariant)."""
function energy_stack(img::Matrix{Float32})
    E = zeros(Float32, N_THETA, size(img)...)
    for (i, θ) in enumerate(THETAS)
        E[i, :, :] .= abs.(conv_same(img, gabor_kernel(θ)))
    end
    return E
end

# ╔═╡ b0000009-0001-4000-8000-000000000009
md"""
## 3. Ray profile and its harmonics

`ray_harmonics` is the heart of the notebook. Note what it is: **K rigid shifts
of orientation channels, linearly combined.** That is a filter over the lifted
field — same class of operation as the Gabor itself. Nothing symbolic.

```
cₙ(y,x) = (1/K) Σ_φ  E[φ mod π](y + d·sin φ, x + d·cos φ) · exp(-i n φ)
```
"""

# ╔═╡ b0000010-0001-4000-8000-000000000010
# bilinear sample of a 2-D map at (y, x), zero outside
@inline function bilinear(M::AbstractMatrix{Float32}, y::Float32, x::Float32)
    H, W = size(M)
    (y < 1 || x < 1 || y > H || x > W) && return 0f0
    y0, x0 = floor(Int, y), floor(Int, x)
    y1, x1 = min(y0+1, H), min(x0+1, W)
    fy, fx = y - y0, x - x0
    return (1-fy)*(1-fx)*M[y0,x0] + fy*(1-fx)*M[y1,x0] +
           (1-fy)*fx    *M[y0,x1] + fy*fx    *M[y1,x1]
end

# ╔═╡ b0000011-0001-4000-8000-000000000011
"""Ray profile R(φ) at one point — the diagnostic view (one lobe per branch)."""
function ray_profile(E::Array{Float32,3}, y::Real, x::Real; d=D_RAY)
    R = zeros(Float32, K_PHI)
    for (i, φ) in enumerate(PHIS)
        ti = mod(round(Int, (mod(φ, π)/π) * N_THETA), N_THETA) + 1
        R[i] = bilinear(view(E, ti, :, :), Float32(y + d*sin(φ)), Float32(x + d*cos(φ)))
    end
    return R
end

# ╔═╡ b0000012-0001-4000-8000-000000000012
"""Dense ray-harmonic maps C[n+1, y, x] for n = 0 … N_MAX. Pure linear filtering."""
function ray_harmonics(E::Array{Float32,3}; d=D_RAY, nmax=N_MAX)
    _, H, W = size(E)
    C = zeros(ComplexF32, nmax+1, H, W)
    for φ in PHIS
        ti = mod(round(Int, (mod(φ, π)/π) * N_THETA), N_THETA) + 1
        Eθ = view(E, ti, :, :)
        dy, dx = d*sin(φ), d*cos(φ)
        for y in 1:H, x in 1:W                       # shift by (+dy,+dx)
            v = bilinear(Eθ, Float32(y + dy), Float32(x + dx))
            v == 0f0 && continue
            for n in 0:nmax
                C[n+1, y, x] += v * cis(-Float32(n) * φ)
            end
        end
    end
    return C ./ K_PHI
end

# ╔═╡ b0000013-0001-4000-8000-000000000013
md"""
## 4. FPE with **integer** frequencies — where the harmonics come from

This is the bridge to the VSA side, and it contains the bug from the previous
notebook.

FPE encodes a scalar `u` as `z^u`, components `exp(i φₖ u)`. Bundle the ray
profile weighted by its energy:

```
V = Σ_φ  R(φ) · z^φ           (z = FPE base for the ray direction)
```

Component `k` of `V` is `Σ_φ R(φ) exp(i φₖ φ)` — **which is the circular
harmonic `c_{φₖ}[R]`.** FPE + bundling *is* the harmonic expansion; you do not
need a separate harmonics module.

**But only if `φₖ ∈ ℤ.`** The previous notebook drew base phases from a
*continuous* uniform distribution:

```julia
Θ_BASE = cis.(Float32(2π) .* rand(rng, Float32, d))    # φₖ ∈ ℝ  ← not periodic
```

With continuous `φₖ`, `z^θ` is **not π-periodic**: `θ = 0` and `θ = π` encode to
maximally different vectors although they are the *same orientation*. The `2θ`
in `fpow(Θ_BASE, 2*s.θ)` was intended to fix this but cannot — periodicity is a
property of the **base frequencies**, not of the exponent. Fix: draw the
frequencies from the integers.
"""

# ╔═╡ b0000014-0001-4000-8000-000000000014
begin
    const DIM = 1024
    # PERIODIC FPE base: integer frequencies -> z^u is 2π-periodic in u.
    # Concentrate on low |n|: that is where junction structure lives.
    const RAY_FREQS = Int.(rand(MersenneTwister(0xBEEF), -N_MAX:N_MAX, DIM))
    ray_base_pow(u::Real) = cis.(Float32(u) .* Float32.(RAY_FREQS))

    # for reference: the NON-periodic version used previously
    bad_base = cis.(Float32(2π) .* rand(MersenneTwister(1), Float32, DIM))
    bad_pow(u::Real) = cis.(Float32(u) .* angle.(bad_base))
end

# ╔═╡ b0000015-0001-4000-8000-000000000015
md"""
### Periodicity check
`z^0` and `z^2π` must be identical (they are the same direction).
"""

# ╔═╡ b0000016-0001-4000-8000-000000000016
let
    good = abs(dot(ray_base_pow(0.0), ray_base_pow(2π))) / DIM
    bad  = abs(dot(bad_pow(0.0),      bad_pow(2π)))      / DIM
    md"""
    | base | ⟨z⁰, z²ᵖⁱ⟩ / d | periodic? |
    |---|---|---|
    | integer frequencies (fixed) | $(round(good, digits=4)) | ✅ |
    | continuous phases (previous notebook) | $(round(bad, digits=4)) | ❌ |
    """
end

# ╔═╡ b0000017-0001-4000-8000-000000000017
"""FPE-bundle a ray profile. Equivalent to the harmonic expansion, sampled at RAY_FREQS."""
function fpe_bundle_ray(R::Vector{Float32})
    V = zeros(ComplexF32, DIM)
    for (i, φ) in enumerate(PHIS)
        R[i] == 0f0 && continue
        V .+= R[i] .* ray_base_pow(φ)
    end
    return V ./ K_PHI
end

# ╔═╡ b0000018-0001-4000-8000-000000000018
md"""
## 5. Synthetic canonical junctions — the ground truth

Ideal ray configurations give `cₙ = Σⱼ exp(-i n φⱼ)`. Predicted signatures:

| config | c₀ | \\|c₁\\|/c₀ | \\|c₂\\|/c₀ |
|---|---|---|---|
| endpoint (1 ray) | 1 | 1.000 | 1.000 |
| straight (2 opposite) | 2 | 0.000 | 1.000 |
| L-corner (2 @ 90°) | 2 | 0.707 | 0.000 |
| **T-junction (3 rays)** | **3** | **0.333** | **0.333** |
| X-crossing (4 rays) | 4 | 0.000 | 0.000 |

All five are distinct in `(c₀, |c₁|/c₀, |c₂|/c₀)`. Real Gabors attenuate the
ratios by a lobe form-factor but preserve the pattern (reference values in §6).
"""

# ╔═╡ b0000019-0001-4000-8000-000000000019
function draw_rays(φs::Vector{<:Real}; S=121, L=45, w=2.2)
    im = zeros(Float32, S, S); c = S ÷ 2 + 1
    for φ in φs, t in 0:0.25:L
        y0, x0 = c + t*sin(φ), c + t*cos(φ)
        for i in max(1,floor(Int,y0-w)):min(S,ceil(Int,y0+w)),
            j in max(1,floor(Int,x0-w)):min(S,ceil(Int,x0+w))
            (i-y0)^2 + (j-x0)^2 < w^2 && (im[i,j] = 1f0)
        end
    end
    return im
end

# ╔═╡ b0000020-0001-4000-8000-000000000020
const CANON = Dict(
    "endpoint"   => [0.0],
    "straight"   => [0.0, π],
    "L-corner"   => [0.0, π/2],
    "T-junction" => [0.0, π/2, π],
    "X-crossing" => [0.0, π/2, π, 3π/2],
)

# ╔═╡ b0000021-0001-4000-8000-000000000021
md"""
## 6. Measured signatures on the canonical figures

Reference values from the verified implementation (λ=10, d=16, ideal in
parentheses) — your numbers should land close to these:

```
endpoint     c₀= 6.8   |c₁|/c₀=0.671 (1.000)   |c₂|/c₀=0.682 (1.000)
straight     c₀=10.5   |c₁|/c₀=0.000 (0.000)   |c₂|/c₀=0.896 (1.000)
L-corner     c₀=11.8   |c₁|/c₀=0.592 (0.707)   |c₂|/c₀=0.006 (0.000)
T-junction   c₀=16.5   |c₁|/c₀=0.326 (0.333)   |c₂|/c₀=0.282 (0.333)
X-crossing   c₀=22.4   |c₁|/c₀=0.000 (0.000)   |c₂|/c₀=0.000 (0.000)
```
"""

# ╔═╡ b0000022-0001-4000-8000-000000000022
canon_results = let
    rows = []
    for k in ["endpoint","straight","L-corner","T-junction","X-crossing"]
        im = draw_rays(CANON[k])
        E  = energy_stack(im)
        c  = size(im,1) ÷ 2 + 1
        R  = ray_profile(E, c, c)
        cn = [abs(sum(R .* cis.(-Float32(n) .* PHIS))) / K_PHI for n in 0:N_MAX]
        push!(rows, (k, cn))
    end
    rows
end

# ╔═╡ b0000023-0001-4000-8000-000000000023
let
    hdr = "| config | c₀ | \\|c₁\\|/c₀ | \\|c₂\\|/c₀ | \\|c₃\\|/c₀ |\n|---|---|---|---|---|\n"
    body = join(["| $k | $(round(c[1],digits=2)) | $(round(c[2]/c[1],digits=3)) | " *
                 "$(round(c[3]/c[1],digits=3)) | $(round(c[4]/c[1],digits=3)) |"
                 for (k,c) in canon_results], "\n")
    Markdown.parse(hdr * body)
end

# ╔═╡ b0000024-0001-4000-8000-000000000024
md"""
### The decisive comparison: E(θ) cannot do this

Below, the orientation profile at the centre for L / T / X. They are *the same
shape*. This is why the previous architecture could not build a T-detector —
the information was never in the representation.
"""

# ╔═╡ b0000025-0001-4000-8000-000000000025
let
    p1 = plot(title="orientation profile E(θ)  [mod π]", xlabel="θ (deg)", legend=:top)
    p2 = plot(title="ray profile R(φ)  [mod 2π]", xlabel="φ (deg)", legend=:top)
    for k in ["L-corner","T-junction","X-crossing"]
        im = draw_rays(CANON[k]); E = energy_stack(im); c = size(im,1) ÷ 2 + 1
        plot!(p1, rad2deg.(THETAS), E[:, c, c], lw=2, label=k)
        plot!(p2, rad2deg.(PHIS), ray_profile(E, c, c), lw=2, label=k)
    end
    plot(p1, p2, layout=(1,2), size=(900, 340))
end

# ╔═╡ b0000026-0001-4000-8000-000000000026
md"""
## 7. Sanity check: orientation convention

A vertical line must give argmax-θ = 90°; horizontal → 0°.
"""

# ╔═╡ b0000027-0001-4000-8000-000000000027
let
    v = zeros(Float32, 112, 112); v[:, 54:58] .= 1f0
    h = zeros(Float32, 112, 112); h[54:58, :] .= 1f0
    Ev, Eh = energy_stack(v), energy_stack(h)
    θv = rad2deg(THETAS[argmax(Ev[:, 56, 56])])
    θh = rad2deg(THETAS[argmax(Eh[:, 56, 56])])
    md"vertical line → **$(round(θv, digits=1))°** (expect 90) · horizontal → **$(round(θh, digits=1))°** (expect 0 or 180)"
end

# ╔═╡ b0000028-0001-4000-8000-000000000028
md"""
## 8. Dense maps on a letter

`c₀` = ray energy ≈ how many branches leave this point. **Junctions are simply
the brightest points** — a junction detector with no threshold anywhere.

`|c₁|/c₀` = asymmetry: ≈1 at endpoints (one ray, no opposite), ≈0 mid-stroke
(two opposite rays cancel) and at X-crossings.

Load a real EMNIST letter here; `letter_T()` is a stand-in so the notebook runs
standalone.
"""

# ╔═╡ b0000029-0001-4000-8000-000000000029
function letter_T(; S=IMG, w=7)
    im = zeros(Float32, S, S)
    im[round(Int,0.25S):round(Int,0.25S)+w, round(Int,0.22S):round(Int,0.78S)] .= 1f0
    im[round(Int,0.25S):round(Int,0.80S), round(Int,0.5S)-w÷2:round(Int,0.5S)+w÷2] .= 1f0
    return im
end

# ╔═╡ b0000030-0001-4000-8000-000000000030
letter_img = letter_T()

# ╔═╡ b0000031-0001-4000-8000-000000000031
letter_E = energy_stack(letter_img)

# ╔═╡ b0000032-0001-4000-8000-000000000032
letter_C = ray_harmonics(letter_E)

# ╔═╡ b0000033-0001-4000-8000-000000000033
let
    c0 = abs.(letter_C[1, :, :])
    r1 = abs.(letter_C[2, :, :]) ./ (c0 .+ 1f-6)
    mask = c0 .> 0.25f0 * maximum(c0)
    p1 = heatmap(letter_img, color=:grays, yflip=true, aspect_ratio=:equal,
                 axis=false, colorbar=false, title="letter")
    p2 = heatmap(c0, color=:magma, yflip=true, aspect_ratio=:equal, axis=false,
                 title="c₀ — ray energy (junction = brightest)")
    p3 = heatmap(map(i -> mask[i] ? r1[i] : NaN, CartesianIndices(r1)),
                 color=:viridis, clims=(0,1), yflip=true, aspect_ratio=:equal,
                 axis=false, title="|c₁|/c₀ — asymmetry (endpoint ≈ 1)")
    plot(p1, p2, p3, layout=(1,3), size=(1150, 380))
end

# ╔═╡ b0000034-0001-4000-8000-000000000034
md"""
### Probe a point interactively

Move the sliders onto the T-junction (≈ y=32, x=56), an arm tip, and mid-stem.
The polar plot shows the ray structure directly: **3 lobes at the junction, 1 at
an endpoint, 2 opposite mid-stroke.**
"""

# ╔═╡ b0000035-0001-4000-8000-000000000035
md"""
**y** $(@bind probe_y Slider(10:IMG-10, default=32, show_value=true))
**x** $(@bind probe_x Slider(10:IMG-10, default=56, show_value=true))
"""

# ╔═╡ b0000036-0001-4000-8000-000000000036
let
    R  = ray_profile(letter_E, probe_y, probe_x)
    cn = [abs(sum(R .* cis.(-Float32(n) .* PHIS))) / K_PHI for n in 0:N_MAX]
    p1 = heatmap(letter_img, color=:grays, yflip=true, aspect_ratio=:equal,
                 axis=false, colorbar=false, title="probe")
    scatter!(p1, [probe_x], [probe_y], m=:circle, ms=7, mc=:magenta, label=false)
    p2 = plot(vcat(PHIS, PHIS[1]), vcat(R, R[1]), proj=:polar, lw=2.5,
              label=false, title="ray profile R(φ)")
    info = md"""
    c₀ = **$(round(cn[1], digits=2))**  ·
    |c₁|/c₀ = **$(round(cn[2]/cn[1], digits=3))**  ·
    |c₂|/c₀ = **$(round(cn[3]/cn[1], digits=3))**
    """
    plot(p1, p2, layout=(1,2), size=(820, 380))
end

# ╔═╡ b0000037-0001-4000-8000-000000000037
md"""
## 9. Where bundling *is* the right operation

The dense maps above involve **no bundling** — bundling a whole image into one
vector is what broke junction detection before (see the diagnosis note).

Bundling belongs *downstream*, at the **object** level, over a **sparse** set of
keypoints, binding each local signature to its object-relative position:

```
Descriptor = Σ_keypoints  V_ray(p)  ⊙  z_r^{r(p)}  ⊙  z_α^{α(p)}
```

with `(r, α)` polar coordinates relative to the glyph centroid — i.e. the
Pasupathy–Connor angular-position code. Sparse because bundling `N` items into
`d` dimensions has SNR ≈ √(d/N): with `d = 1024` and dense `112² × 48 ≈ 600k`
tokens the readout is pure noise. A few dozen keypoints is fine.

**So: dense + convolutional for the feature maps, sparse + bundled for the
object code.** That split is forced by crosstalk, and it happens to mirror
dense retinotopic maps in V1/V2/V4 vs sparse object codes in IT.
"""

# ╔═╡ b0000038-0001-4000-8000-000000000038
"""Object-level descriptor: bundle sparse keypoint signatures with polar position."""
function object_descriptor(E::Array{Float32,3}, C::Array{ComplexF32,3},
                           img::Matrix{Float32}; n_keypoints=32)
    c0 = abs.(C[1, :, :])
    H, W = size(c0)
    # centroid of the glyph → object-relative frame
    tot = sum(img); cy = sum(i*img[i,j] for i in 1:H, j in 1:W)/tot
    cx = sum(j*img[i,j] for i in 1:H, j in 1:W)/tot
    # sparse: take the n strongest ray-energy locations
    idx = sortperm(vec(c0), rev=true)[1:n_keypoints]
    Desc = zeros(ComplexF32, DIM)
    for k in idx
        y, x = Tuple(CartesianIndices(c0)[k])
        R = ray_profile(E, y, x)
        α = atan(y - cy, x - cx)             # object-relative angle
        r = sqrt((y-cy)^2 + (x-cx)^2) / (H/2)
        Desc .+= fpe_bundle_ray(R) .* ray_base_pow(α) .* cis.(Float32(r) .* Float32.(RAY_FREQS))
    end
    return Desc ./ (norm(Desc) + 1f-8)
end

# ╔═╡ b0000039-0001-4000-8000-000000000039
md"""
## 10. Notes / next steps

* **Scale.** Everything here is one Gabor scale. Add scale as another *encoded
  axis* (bind `z_s^{log σ}`), not as a survival conjunction. A curve's harmonic
  signature drifts with scale; a corner's does not — that becomes a readable
  property of the multiscale vector rather than an `if`.
* **`D_RAY` is the one real hyperparameter left.** It must exceed the stroke
  half-width and be smaller than the distance to the next junction. Better:
  compute `cₙ` at several `d` and keep the profile over `d` — a *ray-transform*,
  still linear.
* **Rotation covariance is free.** Rotating the image by `α` gives
  `R(φ) → R(φ-α)` and hence `cₙ → cₙ·e^{-inα}`, i.e. `V → z^α ⊙ V`.
  Rotation = binding. `|cₙ|` is rotation-invariant by construction.
* **Don't threshold into labels.** `(c₀, |c₁|/c₀, |c₂|/c₀, …)` *is* the
  descriptor. A "corner" is a region of that space, not a symbol; corner-ness is
  graded exactly as V4 curvature tuning is graded.
"""

# ╔═╡ Cell order:
# ╟─b0000001-0001-4000-8000-000000000001
# ╠═b0000002-0001-4000-8000-000000000002
# ╟─b0000003-0001-4000-8000-000000000003
# ╠═b0000004-0001-4000-8000-000000000004
# ╟─b0000005-0001-4000-8000-000000000005
# ╠═b0000006-0001-4000-8000-000000000006
# ╠═b0000007-0001-4000-8000-000000000007
# ╠═b0000008-0001-4000-8000-000000000008
# ╟─b0000009-0001-4000-8000-000000000009
# ╠═b0000010-0001-4000-8000-000000000010
# ╠═b0000011-0001-4000-8000-000000000011
# ╠═b0000012-0001-4000-8000-000000000012
# ╟─b0000013-0001-4000-8000-000000000013
# ╠═b0000014-0001-4000-8000-000000000014
# ╟─b0000015-0001-4000-8000-000000000015
# ╠═b0000016-0001-4000-8000-000000000016
# ╠═b0000017-0001-4000-8000-000000000017
# ╟─b0000018-0001-4000-8000-000000000018
# ╠═b0000019-0001-4000-8000-000000000019
# ╠═b0000020-0001-4000-8000-000000000020
# ╟─b0000021-0001-4000-8000-000000000021
# ╠═b0000022-0001-4000-8000-000000000022
# ╠═b0000023-0001-4000-8000-000000000023
# ╟─b0000024-0001-4000-8000-000000000024
# ╠═b0000025-0001-4000-8000-000000000025
# ╟─b0000026-0001-4000-8000-000000000026
# ╠═b0000027-0001-4000-8000-000000000027
# ╟─b0000028-0001-4000-8000-000000000028
# ╠═b0000029-0001-4000-8000-000000000029
# ╠═b0000030-0001-4000-8000-000000000030
# ╠═b0000031-0001-4000-8000-000000000031
# ╠═b0000032-0001-4000-8000-000000000032
# ╠═b0000033-0001-4000-8000-000000000033
# ╟─b0000034-0001-4000-8000-000000000034
# ╟─b0000035-0001-4000-8000-000000000035
# ╠═b0000036-0001-4000-8000-000000000036
# ╟─b0000037-0001-4000-8000-000000000037
# ╠═b0000038-0001-4000-8000-000000000038
# ╟─b0000039-0001-4000-8000-000000000039
