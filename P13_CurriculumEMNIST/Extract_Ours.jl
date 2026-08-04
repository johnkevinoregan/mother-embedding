# ── PLAIN SCRIPT ────────────────────────────────────────────────────────────
# `julia --project=.. -t 14 Extract_Ours.jl`
#
# Extract the best front-end configuration on all of EMNIST balanced, and cache it.
#
# CONFIG: the 4-scale ladder (λ = 56/29.9/16/8 px) plus the spatial-max features, at grid 3.
# Grid 3 rather than the stroke set's grid 1 because EMNIST characters are CENTRED — Phase 10
# showed grid 3 beating grid 1 by 8 points on Fashion-MNIST for exactly that reason, and grid 1
# only wins where position is randomised.
#
# CAVEAT ON λ = 8. EMNIST is 28×28 upsampled 4× to 112, so its original Nyquist limit lands at
# λ = 8 px. That channel therefore sees mostly bilinear interpolation here, unlike the stroke set
# which is natively 112 px. Included because it is part of the best configuration, but if it
# underperforms this is why.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Statistics, Printf, Serialization, LinearAlgebra, FFTW
include(joinpath(@__DIR__, "..", "LoadEMNIST.module.jl"))
include(joinpath(@__DIR__, "..", "P9_P12_SimpleStrokeTests", "Frontend.module.jl"))
using .LoadEMNIST, .Frontend
BLAS.set_num_threads(min(16, Sys.CPU_THREADS)); FFTW.set_num_threads(1)

const SRC   = joinpath(homedir(), "Julia", "DATABASES", "EMNIST", "emnist_source_files")
const CACHE = joinpath(@__DIR__, "cache"); mkpath(CACHE)
const N     = 112
const CHUNK = 4000

"Bilinear 28 → 112, as every other phase does it."
function up(img, M=N)
    H, W = size(img); out = zeros(Float32, M, M)
    @inbounds for i in 1:M, j in 1:M
        y = 1 + (i-1)*(H-1)/(M-1); x = 1 + (j-1)*(W-1)/(M-1)
        y0 = floor(Int, y); x0 = floor(Int, x)
        y1 = min(y0+1, H); x1 = min(x0+1, W); fy = y - y0; fx = x - x0
        out[i,j] = (1-fy)*(1-fx)*img[y0,x0] + fy*(1-fx)*img[y1,x0] +
                   (1-fy)*fx*img[y0,x1] + fy*fx*img[y1,x1]
    end
    out
end

spec = build_frontend(N; grid=3, ladder=[2.0, 3.742, 7.0, 14.0],
                      betas=[2.0, 1.6, 1.2, 1.0], nori=[8, 12, 16, 20],
                      spatial_max=true)
@printf("front end: %d features, %d channels, grid 3\n\n", spec.n, length(spec.bank.filters))

for (kind, tag) in (("train", "train"), ("test", "test"))
    ck = joinpath(CACHE, "ours_$(tag).jls")
    isfile(ck) && (@printf("%s already cached\n", tag); continue)
    A = read_emnist_images(joinpath(SRC, "emnist-balanced-$(kind)-images-idx3-ubyte"))
    y = Int.(read_emnist_labels(joinpath(SRC, "emnist-balanced-$(kind)-labels-idx1-ubyte")))
    n = size(A, 3)
    F = zeros(Float32, n, spec.n)
    t0 = time()
    # chunked: 112,800 images at 112² Float32 would be 5.7 GB held at once
    for lo in 1:CHUNK:n
        hi = min(lo + CHUNK - 1, n)
        imgs = [up(Float32.(@view A[:, :, i])) for i in lo:hi]
        F[lo:hi, :] = featurize(imgs, spec)
        @printf("  %s %6d / %6d   %5.0f s\n", tag, hi, n, time()-t0); flush(stdout)
    end
    serialize(ck, (F, y))
    @printf("%s: %d × %d in %.0f s (%.1f ms/img)\n\n", tag, n, spec.n, time()-t0, 1000*(time()-t0)/n)
end
println("done")
