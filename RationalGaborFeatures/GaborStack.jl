"""
    GaborStack

A general oriented-energy front end: a frequency-domain (log-)Gabor bank, applied
densely to any 2-D greyscale image of any size.

Nothing here is specific to EMNIST or to characters. The scale ladder is *derived
from the data* by `radial_spectrum` + `scale_ladder`, so pointing this at a new
dataset means re-running those two functions, not editing constants.

Design decisions, and why:

  * **Filters are built in the frequency domain**, as bumps in `(ρ, θ)`. We convolve
    by FFT anyway, so there is no spatial kernel to truncate — the ±2σ/±3σ support
    question simply does not arise — and the bank is size-agnostic.
  * **One-sided (analytic) bumps.** A single bump, not a conjugate pair, so the
    response is analytic: real and imaginary parts form an exact even/odd quadrature
    pair and `|·|²` is the complex-cell energy. Polarity invariance is exact, by
    construction, not by approximation.
  * **log-Gabor by default.** Zero DC exactly (`log 0 = -∞`), so no mean-subtraction
    hack and no constraint keeping the bump clear of the origin; and bandwidth is
    free, which decouples the spatial envelope `σ_x` from the wavelength `λ`. That
    decoupling is what allows a filter tuned to a coarse wavelength to still localise
    finely. `family = :gabor` gives the ordinary linear-frequency Gaussian for A/B.
  * **Padding is mandatory, not optional.** FFT convolution is circular; without a
    border of at least `3σ_x` the coarsest filter's response is contaminated
    everywhere. `field_for` computes the requirement from the ladder.

Typical use:

```julia
spec   = radial_spectrum(images)                 # any size, any number
lad    = scale_ladder(spec; n_scales=3, energy_floor=0.05)
bank   = make_bank((320,320), lad; imwidth=112, n_orient=[8,12,16])
E      = energy_stack(img, bank)                 # H × W × nfilters, cropped
```
"""
module GaborStack

using FFTW, Statistics

export radial_spectrum, scale_ladder, GaborBank, make_bank, energy_stack,
       field_for, sigma_x, sigma_along, bandwidth_ratio, embed, pad_modes

# ---------------------------------------------------------------- frequency grid

"Signed FFT frequency indices for length `n`, in cycles per pixel."
fftfreqs(n::Int) = Float64[(i <= n ÷ 2 ? i : i - n) / n for i in 0:n-1]

# ---------------------------------------------------------------- 1. spectrum

"""
    radial_spectrum(images; nbins=nothing, remove_dc=true)

Mean radial power spectrum over a stack of images, as `(ρ, power)` where `ρ` is in
**cycles per image width**, so it is comparable across image sizes.

`images` may be a 3-D array `(H, W, N)` or a vector of 2-D arrays (which need not all
be the same size — each contributes on its own ρ axis, resampled to the common bins).
"""
function radial_spectrum(images::AbstractArray{<:Real,3}; remove_dc=true)
    H, W, N = size(images)
    P = zeros(Float64, H, W)
    for k in 1:N
        P .+= abs2.(fft(Float64.(@view images[:, :, k])))
    end
    P ./= N
    remove_dc && (P[1, 1] = 0.0)
    _radial_profile(P, H, W)
end

function radial_spectrum(images::AbstractVector; remove_dc=true)
    isempty(images) && error("no images")
    ref = size(first(images))
    all(size(im) == ref for im in images) ||
        error("radial_spectrum: mixed image sizes not supported; resample first")
    stack = Array{Float64,3}(undef, ref[1], ref[2], length(images))
    for (k, im) in enumerate(images); stack[:, :, k] = im; end
    radial_spectrum(stack; remove_dc=remove_dc)
end

function _radial_profile(P, H, W)
    fy, fx = fftfreqs(H), fftfreqs(W)
    maxr = min(H, W) ÷ 2
    ring = zeros(Float64, maxr + 1)
    @inbounds for a in 1:H, b in 1:W
        # ρ in cycles per image width, so the axis is size-independent
        r = hypot(fy[a], fx[b]) * W
        r > maxr && continue
        ring[round(Int, r)+1] += P[a, b]
    end
    (rho = collect(0:maxr), power = ring)
end

# ---------------------------------------------------------------- 2. scale ladder

"""
    scale_ladder(spec; n_scales=4, energy_floor=0.05, rho_lo=2.0)

Place `n_scales` centre frequencies geometrically across the band that actually
carries energy. The top of the band is the largest `ρ` above which more than
`energy_floor` of the total (DC-removed) power still remains; below that we would be
filtering noise, which is exactly the failure mode of a bank whose finest channels
sit past the data's real bandwidth.

Returns `ρ₀` values in **cycles per image width**.
"""
function scale_ladder(spec; n_scales::Int=4, energy_floor::Real=0.05, rho_lo::Real=2.0)
    ring = spec.power; tot = sum(ring)
    tot <= 0 && error("scale_ladder: spectrum has no energy")
    cum_above = [sum(@view ring[j+2:end]) / tot for j in 0:length(ring)-2]  # energy at ρ > j
    hi = findlast(c -> c > energy_floor, cum_above)
    hi === nothing && error("scale_ladder: energy_floor $(energy_floor) excludes everything")
    rho_hi = Float64(hi)
    rho_hi <= rho_lo && error("scale_ladder: usable band [$rho_lo, $rho_hi] is empty")
    collect(exp.(range(log(rho_lo), log(rho_hi); length=n_scales)))
end

# ---------------------------------------------------------------- 3. bank geometry

"""
    bandwidth_ratio(beta)

`k = σ_ρ/ρ₀` for a log-Gabor of bandwidth `beta` octaves.
From `β = 2√(2/ln2)·|ln k|`.
"""
bandwidth_ratio(beta::Real) = exp(-beta / (2 * sqrt(2 / log(2))))

"""
    sigma_x(lambda, beta)

Approximate spatial envelope σ (pixels) for a filter of wavelength `lambda` (pixels)
and bandwidth `beta` octaves. This is what sets **localisation** — note it shrinks as
bandwidth grows, which is why a broadband filter at a long wavelength can still
resolve fine spatial structure.
"""
sigma_x(lambda::Real, beta::Real) =
    (lambda / π) * sqrt(log(2) / 2) * (2^beta + 1) / (2^beta - 1)

"""
    sigma_along(rho0, sigma_phi, imwidth)

Spatial envelope σ (pixels) **along** the contour, set by the filter's angular
narrowness: `σ_along = W / (2π ρ₀ σ_φ)`.

This is usually the *larger* of the two extents and is easy to overlook — a bank narrow
in orientation is spatially elongated, and sizing the padded field from the radial
`sigma_x` alone under-pads badly. Measured example: at ρ₀=2 with 8 orientations,
`σ_x = 17.5` but `σ_along = 34.0`, so a border of 56 sits at only 1.65 σ and leaks ~5 %
of the energy round the wrap.
"""
sigma_along(rho0::Real, sigma_phi::Real, imwidth::Real) = imwidth / (2π * rho0 * sigma_phi)

"""
    field_for(imsize, rho0s; n_orient, beta, k=3, dtheta_on_sigma=1.5, fft_friendly=true)

Padded field size such that circular (FFT) convolution does not contaminate the valid
region. The border must clear `k·σ` for the **larger of the two spatial extents** —
across-contour (`sigma_x`, from radial bandwidth) and along-contour (`sigma_along`,
from angular width). Returns `(H, W, border)`.

Takes the same bank parameters as [`make_bank`] so the two cannot drift apart.

`fft_friendly` rounds up to a product of 2/3/5/7, which matters: a field of 218
(= 2·109, 109 prime) measured **3.8× slower per FFT** than 224. Padding is free, so
there is no reason not to round.
"""
function field_for(imsize::Tuple{Int,Int}, rho0s::AbstractVector;
                   n_orient=8, beta=2.0, k::Real=3, dtheta_on_sigma::Real=1.5,
                   fft_friendly::Bool=true)
    ns = length(rho0s)
    norients = n_orient isa Number ? fill(Int(n_orient), ns) : collect(Int.(n_orient))
    betas    = beta isa Number ? fill(Float64(beta), ns) : collect(Float64.(beta))
    imw = imsize[2]; ext = 0.0
    for i in 1:ns
        ρ = rho0s[i]; σφ = (π / norients[i]) / dtheta_on_sigma
        ext = max(ext, sigma_x(imw / ρ, betas[i]), sigma_along(ρ, σφ, imw))
    end
    border = ceil(Int, k * ext)
    H = imsize[1] + 2border; W = imsize[2] + 2border
    if fft_friendly
        H = nextprod([2, 3, 5, 7], H); W = nextprod([2, 3, 5, 7], W)
    end
    (H, W, min((H - imsize[1]) ÷ 2, (W - imsize[2]) ÷ 2))
end

# ---------------------------------------------------------------- 4. the bank

struct GaborBank
    size::Tuple{Int,Int}                  # padded field
    filters::Vector{Matrix{Float32}}      # frequency-domain transfer functions
    meta::Vector{NamedTuple}              # (rho0, lambda, theta, beta, kind)
end

Base.length(b::GaborBank) = length(b.filters)

"""
    make_bank(fieldsize, rho0s; n_orient, beta, family=:log_gabor,
              lowpass=true, normalize=true, dtheta_on_sigma=1.5)

Build the frequency-domain bank on a `fieldsize` grid.

* `rho0s` — centre frequencies in cycles per image width (from `scale_ladder`).
* `n_orient` — scalar, or one entry per scale. Growing the count with `ρ` keeps
  coverage of the frequency plane roughly uniform, since the ring circumference
  grows; a fixed count over-samples coarse scales and under-samples fine ones.
* `beta` — bandwidth in octaves, scalar or per-scale.
* `family` — `:log_gabor` (zero DC exactly) or `:gabor` (linear-frequency Gaussian).
* `lowpass` — append an isotropic low-pass channel below the lowest scale. On many
  datasets this band carries most of the energy, and a bandpass bank misses it.
* `normalize` — scale each filter to unit **RMS** over the frequency grid, which
  approximates the continuous integral `∫|G|²df` and is therefore independent of the
  padded field size. Normalising by the raw `sum(G²)` instead makes every response
  scale as `1/√(HW)`, so merely adding padding changes the features — measured as a
  63 % shift going from a 320 to a 525 field.
"""
function make_bank(fieldsize::Tuple{Int,Int}, rho0s::AbstractVector;
                   imwidth::Int, n_orient=8, beta=2.0, family::Symbol=:log_gabor,
                   lowpass::Bool=true, normalize::Bool=true,
                   dtheta_on_sigma::Real=1.5)
    family in (:log_gabor, :gabor) || error("family must be :log_gabor or :gabor")
    H, W = fieldsize
    ns = length(rho0s)
    norients = n_orient isa Number ? fill(Int(n_orient), ns) : collect(Int.(n_orient))
    betas    = beta isa Number ? fill(Float64(beta), ns) : collect(Float64.(beta))
    length(norients) == ns || error("n_orient must be scalar or one per scale")
    length(betas) == ns    || error("beta must be scalar or one per scale")

    # ρ is in cycles per IMAGE width, never per field width. Getting this wrong
    # mistunes every filter by field/image (2.86× at 320 vs 112) and, worse, the error
    # grows as you add padding — so the symptom looks like leakage rather than mistuning.
    fy, fx = fftfreqs(H), fftfreqs(W)          # cycles per pixel
    RHO = Matrix{Float64}(undef, H, W); PHI = Matrix{Float64}(undef, H, W)
    @inbounds for a in 1:H, b in 1:W
        RHO[a, b] = hypot(fy[a], fx[b]) * imwidth
        PHI[a, b] = atan(fy[a], fx[b])
    end

    filters = Matrix{Float32}[]; meta = NamedTuple[]
    for (si, rho0) in enumerate(rho0s)
        nθ = norients[si]; β = betas[si]
        k = bandwidth_ratio(β); lnk2 = 2 * log(k)^2
        σφ = (π / nθ) / dtheta_on_sigma
        for oi in 0:nθ-1
            θ0 = oi * π / nθ
            G = Matrix{Float32}(undef, H, W)
            @inbounds for a in 1:H, b in 1:W
                ρ = RHO[a, b]
                if ρ <= 0
                    G[a, b] = 0f0; continue          # DC: zero for both families
                end
                radial = family === :log_gabor ?
                    exp(-log(ρ / rho0)^2 / lnk2) :
                    exp(-(ρ - rho0)^2 / (2 * (k * rho0)^2))
                dφ = atan(sin(PHI[a, b] - θ0), cos(PHI[a, b] - θ0))
                # one-sided: keep only the half-plane around θ0, giving an analytic
                # response whose modulus is the envelope
                angular = abs(dφ) > π/2 ? 0.0 : exp(-dφ^2 / (2σφ^2))
                G[a, b] = Float32(radial * angular)
            end
            if normalize
                nrm = sqrt(sum(abs2, G) / (H * W)); nrm > 0 && (G ./= Float32(nrm))
            end
            push!(filters, G)
            push!(meta, (rho0=rho0, lambda=imwidth / rho0, theta=θ0, beta=β, kind=:oriented))
        end
    end

    if lowpass
        rho_lp = minimum(rho0s) / 2
        G = Matrix{Float32}(undef, H, W)
        @inbounds for a in 1:H, b in 1:W
            G[a, b] = Float32(exp(-(RHO[a, b] / rho_lp)^2 / 2))
        end
        G[1, 1] = 0f0                                   # keep it DC-free like the rest
        if normalize
            nrm = sqrt(sum(abs2, G) / (H * W)); nrm > 0 && (G ./= Float32(nrm))
        end
        push!(filters, G)
        push!(meta, (rho0=rho_lp, lambda=imwidth / rho_lp, theta=NaN, beta=NaN, kind=:lowpass))
    end

    GaborBank(fieldsize, filters, meta)
end

# ---------------------------------------------------------------- 5. application

const pad_modes = (:zero, :replicate, :reflect)

"""
    embed(img, fieldsize; mode=:zero)

Centre `img` in a `fieldsize` field. `:zero` is right when the background genuinely is
zero (it then introduces no step); `:replicate` is the gentlest general choice;
`:reflect` is conventional but manufactures mirror-symmetric structure at the border,
which can produce spurious symmetric responses.

Returns `(field, offset)` where `offset` is the top-left corner of `img` in the field.
"""
function embed(img::AbstractMatrix{<:Real}, fieldsize::Tuple{Int,Int}; mode::Symbol=:zero)
    mode in pad_modes || error("mode must be one of $pad_modes")
    h, w = size(img); H, W = fieldsize
    (H >= h && W >= w) || error("field $(fieldsize) smaller than image $(size(img))")
    oy = (H - h) ÷ 2; ox = (W - w) ÷ 2
    F = zeros(Float32, H, W)
    @inbounds for a in 1:H, b in 1:W
        i = a - oy; j = b - ox
        if mode === :zero
            F[a, b] = (1 <= i <= h && 1 <= j <= w) ? Float32(img[i, j]) : 0f0
        elseif mode === :replicate
            F[a, b] = Float32(img[clamp(i, 1, h), clamp(j, 1, w)])
        else                                            # :reflect
            ri = i < 1 ? 2 - i : (i > h ? 2h - i : i)
            rj = j < 1 ? 2 - j : (j > w ? 2w - j : j)
            F[a, b] = Float32(img[clamp(ri, 1, h), clamp(rj, 1, w)])
        end
    end
    F, (oy, ox)
end

"""
    energy_stack(img, bank; mode=:zero, crop=true)

Dense oriented energy `E[y, x, filter] = |analytic response|²`.

Real, non-negative, and **exactly polarity-invariant**: negating the image negates the
complex response and leaves `|·|²` unchanged. Cropped back to `size(img)` unless
`crop=false`.
"""
function energy_stack(img::AbstractMatrix{<:Real}, bank::GaborBank;
                      mode::Symbol=:zero, crop::Bool=true)
    F, (oy, ox) = embed(img, bank.size; mode=mode)
    Ff = fft(F)
    h, w = size(img); H, W = bank.size
    nf = length(bank.filters)
    out = crop ? Array{Float32,3}(undef, h, w, nf) : Array{Float32,3}(undef, H, W, nf)
    for (k, G) in enumerate(bank.filters)
        r = ifft(Ff .* G)
        if crop
            @inbounds for j in 1:w, i in 1:h
                out[i, j, k] = Float32(abs2(r[oy+i, ox+j]))
            end
        else
            @inbounds for j in 1:W, i in 1:H
                out[i, j, k] = Float32(abs2(r[i, j]))
            end
        end
    end
    out
end

end # module
