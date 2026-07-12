module CreateGaborLifting

using ImageFiltering

include(joinpath(@__DIR__, "Config.jl"))
using .Config

export GaborSample, gabor_kernel, imresize_simple, gabor_lift

# A single Gabor sample: position, filter orientation/scale, and the complex
# response's modulus and phase (no thresholding or normalization applied).
struct GaborSample
    x::Float32       # normalized to [0, 1]
    y::Float32
    θ::Float32       # filter orientation, radians, [0, π)
    s::Float32       # scale index (1, 2, 3, ...)
    modulus::Float32
    phase::Float32   # radians, (-π, π]
end

# `aspect` is the envelope's along-contour ÷ across-contour σ ratio (Config's
# GABOR_ASPECT): 1 = isotropic, >1 = elongated along the stroke. It stretches
# the σ of the y_θ (tangent) axis by `aspect` while leaving the x_θ (carrier /
# across-contour) axis — hence the scale/frequency selectivity — untouched. The
# kernel support grows with `aspect` so the longer envelope isn't truncated; at
# aspect=1 both the support and the envelope are identical to before.
function gabor_kernel(λ::Float32, θ::Float32; σ_factor=0.5f0, aspect=GABOR_ASPECT)
    σ = σ_factor * λ
    radius = ceil(Int, 2σ * max(1f0, aspect))
    kernel = zeros(ComplexF32, 2radius+1, 2radius+1)
    cos_θ, sin_θ = cos(θ), sin(θ)
    for i in -radius:radius, j in -radius:radius
        x_θ =  i * cos_θ + j * sin_θ
        y_θ = -i * sin_θ + j * cos_θ
        envelope = exp(-(x_θ^2 + (y_θ / aspect)^2) / (2σ^2))
        carrier  = cis(Float32(2π) * x_θ / λ)
        kernel[i+radius+1, j+radius+1] = envelope * carrier
    end
    # Zero the DC (mean) of the kernel. Corrects a slight earlier error: the
    # even/cosine channel carried a small residual DC (~1.5% of the
    # ideal-response scale at σ=0.5λ; the "~1% DC leak" noted in
    # PROGRESS_2026-07-11.md), so a uniform region of value c leaked a
    # spurious response ∝ c — inflating modulus and biasing phase toward
    # "line". It happened to be harmless on black-background EMNIST (c≈0, so
    # the leak is ≈0), which is why it went unnoticed, but it was still wrong.
    # Subtracting the mean makes every uniform region — at any brightness —
    # give exactly 0.
    kernel .-= sum(kernel) / length(kernel)
    # Ideal-response normalization — deliberately NOT the usual energy
    # normalization (Σ|K|² = 1). Energy normalization calibrates responses
    # against a unit-energy stimulus, which leaves them growing ~∝ λ on real
    # [0,1]-valued images (larger kernels integrate more image energy).
    # Instead we scale so the kernel's response to its ideal achievable
    # stimulus — a white (value 1) bar covering exactly the positive lobe,
    # on black — is 1. Every response modulus is then directly the fraction
    # of the best possible response, comparable across scales, with no
    # further correction needed anywhere downstream. Re-derived AFTER the DC
    # subtraction above, since deepening the negative lobes changes the
    # positive-lobe sum.
    kernel ./= sum(max.(real.(kernel), 0f0))
    return kernel
end

# Simple bilinear upscale.
function imresize_simple(img::Matrix{Float32}, new_size::Int)
    h, w = size(img)
    out = zeros(Float32, new_size, new_size)
    for i in 1:new_size, j in 1:new_size
        src_i = 1 + (i - 1) * (h - 1) / (new_size - 1)
        src_j = 1 + (j - 1) * (w - 1) / (new_size - 1)
        i0, j0 = floor(Int, src_i), floor(Int, src_j)
        i1, j1 = min(i0 + 1, h), min(j0 + 1, w)
        fi, fj = src_i - i0, src_j - j0
        out[i, j] = (1 - fi) * (1 - fj) * img[i0, j0] +
                          fi  * (1 - fj) * img[i1, j0] +
                    (1 - fi) *       fj  * img[i0, j1] +
                          fi  *       fj  * img[i1, j1]
    end
    return out
end

"""
    gabor_lift(image; scales=SCALES, orientations=ORIENTATIONS,
                 overlap_frac=OVERLAP_FRAC, img_size=IMG_SIZE)

Convolve `image` with a bank of complex Gabor filters at every combination of
`scales` x `orientations`, sample the complex response on a grid (spacing set
by `overlap_frac` relative to each scale's wavelength), and return every
sampled point as a `GaborSample` carrying its modulus and phase.

This is the raw filter-bank response: no thresholding is applied — every grid
point at every scale and orientation is returned. No normalization is applied
here either: the kernels themselves are ideal-response normalized (see
`gabor_kernel`), so each modulus is already the fraction of the best
achievable response — a perfectly matched full-contrast bar reads 1.0 at
every scale, making moduli comparable across scales.

Boundary handling: convolution uses replicate-padding (`Pad(:replicate)`), so
kernels that extend past the image border are computed against replicated
edge pixels rather than dropped or zero-padded. This matters most at the
largest scale, where the kernel radius can be a large fraction of `img_size`.
"""
function gabor_lift(image::Matrix{Float32}; scales::Vector{Float32}=SCALES,
                     orientations::Vector{Float32}=ORIENTATIONS,
                     overlap_frac::Float32=OVERLAP_FRAC,
                     img_size::Int=IMG_SIZE,
                     aspect=GABOR_ASPECT)
    img = size(image) != (img_size, img_size) ? imresize_simple(image, img_size) : image

    samples = GaborSample[]
    for (s_idx, λ) in enumerate(scales)
        step = max(1, round(Int, (1 - overlap_frac) * λ))
        for θ in orientations
            kernel = gabor_kernel(λ, θ; aspect=aspect)
            response = imfilter(img, centered(kernel), Pad(:replicate))
            for cx in step ÷ 2 + 1 : step : img_size
                for cy in step ÷ 2 + 1 : step : img_size
                    r = response[cx, cy]
                    # response is (row, col); cx indexes rows, cy indexes
                    # columns — so .x (horizontal) comes from cy, .y
                    # (vertical) comes from cx, matching normal image/plot
                    # convention.
                    push!(samples, GaborSample(
                        (cy - 1) / (img_size - 1),
                        (cx - 1) / (img_size - 1),
                        θ,
                        Float32(s_idx),
                        abs(r),
                        angle(r)
                    ))
                end
            end
        end
    end
    return samples
end

#=
Kept for reference in case a dense (non-grid-sampled) per-pixel response map
is ever needed for debugging, e.g. to visualize the continuous filter
response field rather than just the sampled grid points. Superseded for
normal use by `gabor_lift` itself, which now returns every grid point for
every filter unthresholded — the same information for a single (λ, θ) can be
obtained by filtering `gabor_lift(image)`'s output down to that scale and
orientation.

function compute_response_grid(image::Matrix{Float32}, λ::Float32, θ::Float32;
                                overlap_frac::Float32=OVERLAP_FRAC, img_size::Int=IMG_SIZE)
    img = size(image) != (img_size, img_size) ? imresize_simple(image, img_size) : image
    kernel = gabor_kernel(λ, θ)
    response = imfilter(img, centered(kernel), Pad(:replicate))
    step = max(1, round(Int, (1 - overlap_frac) * λ))
    points = Tuple{Int, Int, ComplexF32}[]
    for cx in step ÷ 2 + 1 : step : img_size
        for cy in step ÷ 2 + 1 : step : img_size
            push!(points, (cx, cy, response[cx, cy]))
        end
    end
    return img, points
end
=#

end # module
