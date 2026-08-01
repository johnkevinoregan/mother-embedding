# ── PLAIN SCRIPT, not a Pluto notebook ──────────────────────────────────────
# `julia --project=.. -t 14 Add_DeepHead.jl`   (from the SimpleStrokeTests directory)
#
# Adds a fourth arm: our 31 features with a **deeper, tapering head** — 31 → 512 → 128 → 64 → 8 —
# against the 31 → 256 → 256 → 8 used everywhere else.
#
# The point is to separate two things the tables cannot currently distinguish. Where our features
# score below frozen ConvNeXt, is that because the 31 numbers do not carry the information, or
# because a two-layer 256-wide head cannot extract it? A wider first layer and a taper give the
# head more room to fold the input before compressing, at a similar parameter count (~91 k against
# ~76 k), so a large gain would point at the readout and a null result at the representation.
#
# Everything else is held fixed: same cached features, same images, same targets, same standard-
# isation, same Adam 1e-3, same batch 128, same snapshot epochs. Only the head's shape changes.
#
# Merges into `figures_predictions/predictions.jls` rather than rewriting it, so the three
# existing arms are untouched and the figures redraw with four rows.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Statistics, Printf, Random, LinearAlgebra, Serialization, FFTW
using Flux, CUDA
include(joinpath(@__DIR__, "..", "ConVNextTest", "Readout.module.jl"))
using .Readout
BLAS.set_num_threads(min(16, Sys.CPU_THREADS)); FFTW.set_num_threads(1)

const DATA  = joinpath(@__DIR__, "..", "ConVNextTest", "data")
const CACHE = joinpath(@__DIR__, "..", "ConVNextTest", "cache")
const FIG   = joinpath(@__DIR__, "figures_predictions")
const NTR, NTE = 16000, 4000
# Which head to add. Generalised rather than copied into a second script, so the two variants
# cannot drift apart in the parts that are meant to be identical.
const HEAD  = get(ENV, "HEAD", "deep")
const NAME  = HEAD == "res"  ? "ours res 256-128-64+skip" :
              HEAD == "fine"  ? "ours + fine λ=8 (41)" :
              HEAD == "ultra" ? "ours + λ=8,4 (51)" :
              HEAD == "smax"  ? "ours + spatial max (40)" :
              HEAD == "both"  ? "ours + λ=8 + smax (53)" : "our features deep (31)"

d = deserialize(joinpath(FIG, "predictions.jls"))
PROPS, SNAPS, Yte = d.props, d.snaps, d.Yte
np = length(PROPS)

read_mat(p, n, k) = permutedims(reshape(read!(open(p), Vector{Float32}(undef, n*k)), k, n))
Ytr = read_mat(joinpath(DATA, "iid_train_y.f32"), NTR, np)
nva = NTR ÷ 6; tr = 1:NTR-nva; va = NTR-nva+1:NTR
μy, σy = zfit(Ytr[tr, :]); Zt = zapply(Ytr[tr, :], μy, σy)

# `fine` adds a fourth oriented scale at λ = 8 px (ρ = 14), continuing the ladder's own
# geometric spacing — 56 / 29.9 / 16 has ratio 1.87, so the next rung is 8.55 px and 8 is
# essentially it. That needs its own extraction; the other heads reuse the 31-feature cache.
#
# λ = 8 is real signal here because SimpleStrokeTests is natively 112 px. On EMNIST and
# Fashion-MNIST, which are 28×28 upsampled 4×, λ = 8 sits exactly at the original Nyquist and
# would mostly measure bilinear interpolation — so this scale should not be made a default
# without checking per dataset.
Ftr, Fte = if HEAD in ("fine", "ultra", "smax", "both")
    # `ultra` adds a FIFTH oriented scale at λ = 4 px (ρ = 28). Still well clear of Nyquist —
    # these images are natively 112 px, so the limit is λ = 2 — but close to the point where the
    # generator's own sharpest edge (a 0.8 px ramp) is all there is to see.
    ck = joinpath(CACHE, "ours_$(HEAD)_g1_iid_$(NTR)_$(NTE).jls")
    if isfile(ck)
        deserialize(ck)
    else
        include(joinpath(@__DIR__, "Frontend.module.jl"))
        lad, bet, nori_ = HEAD == "ultra" ?
            ([2.0, 3.742, 7.0, 14.0, 28.0], [2.0, 1.6, 1.2, 1.0, 0.9], [8, 12, 16, 20, 24]) :
            HEAD == "smax" ?
            ([2.0, 3.742, 7.0], [2.0, 1.6, 1.2], [8, 12, 16]) :
            HEAD == "both" ?
            ([2.0, 3.742, 7.0, 14.0], [2.0, 1.6, 1.2, 1.0], [8, 12, 16, 20]) :
            ([2.0, 3.742, 7.0, 14.0],       [2.0, 1.6, 1.2, 1.0],      [8, 12, 16, 20])
        # `smax` keeps the ORIGINAL three-scale ladder and changes only the spatial summary, so it
        # isolates mean-vs-max from every wavelength change.
        sp = Main.Frontend.build_frontend(112; grid=1, ladder=lad, betas=bet, nori=nori_,
                                          spatial_max = HEAD in ("smax", "both"))
        @printf("fine bank: %d features, %d channels\n", sp.n, length(sp.bank.filters))
        rdimg(path, n) = (a = read!(open(path), Vector{Float32}(undef, n*112*112));
                          [permutedims(reshape(@view(a[(i-1)*112*112+1 : i*112*112]), 112, 112))
                           for i in 1:n])
        t = @elapsed (a = Main.Frontend.featurize(rdimg(joinpath(DATA,"iid_train_img.f32"), NTR), sp);
                      b = Main.Frontend.featurize(rdimg(joinpath(DATA,"iid_test_img.f32"),  NTE), sp))
        @printf("extracted in %.0f s (%.1f ms/img)\n", t, 1000t/(NTR+NTE))
        serialize(ck, (a, b)); (a, b)
    end
else
    deserialize(joinpath(CACHE, "ours_g1_iid_$(NTR)_$(NTE).jls"))
end
μ, σ = zfit(Ftr[tr, :]); A = zapply(Ftr, μ, σ); T = zapply(Fte, μ, σ)

Random.seed!(1)
dev = CUDA.functional() ? gpu : cpu
Xg = dev(permutedims(A[tr, :])); Tg = dev(permutedims(T)); Yg = dev(permutedims(Zt))
nin = size(A, 2)
m = if HEAD == "res"
    # SkipConnection(f, vcat) computes vcat(f(x), x), so the trunk's 64 activations arrive at
    # the output layer alongside the raw `nin` features — 64 + 31 = 95 inputs. The output layer
    # can therefore express a linear readout of the features plus a nonlinear correction, rather
    # than having to rebuild the linear part through three ReLU layers.
    trunk = Chain(Dense(nin => 256, relu), Dense(256 => 128, relu), Dense(128 => 64, relu))
    Chain(SkipConnection(trunk, vcat), Dense(64 + nin => np))
elseif HEAD in ("fine", "ultra", "smax", "both")
    # the standard head, so this arm isolates the REPRESENTATION change
    Chain(Dense(nin => 256, relu), Dense(256 => 256, relu), Dense(256 => np))
else
    Chain(Dense(nin => 512, relu), Dense(512 => 128, relu),
          Dense(128 => 64, relu), Dense(64 => np))
end |> dev
@printf("head %s: %s   (%d parameters)\n", HEAD,
        HEAD == "res"  ? "$nin → 256 → 128 → 64 ⊕ $nin = $(64+nin) → $np" :
        HEAD in ("fine","ultra","smax","both") ? "$nin → 256 → 256 → $np" :
                         "$nin → 512 → 128 → 64 → $np",
        sum(length, Flux.trainables(m)))
opt = Flux.setup(Flux.Adam(1f-3), m); n = size(Xg, 2)
out = Dict{Int,Matrix{Float32}}()
t = @elapsed for e in 1:maximum(SNAPS)
    for i in Iterators.partition(randperm(n), 128)
        _, gs = Flux.withgradient(mm -> Flux.mse(mm(Xg[:,i]), Yg[:,i]), m)
        Flux.update!(opt, m, gs[1])
    end
    e in SNAPS && (out[e] = permutedims(Array(m(Tg))) .* σy' .+ μy')
end
@printf("trained in %.0f s\n\n", t)

d.preds[NAME] = out
serialize(joinpath(FIG, "predictions.jls"), d)

# ── the table, every arm at every snapshot epoch, from one run on identical data
arms = ["our CNN (end to end)", "our features (31)", "ours + fine λ=8 (41)",
        "ours + λ=8,4 (51)", "ours + spatial max (40)", "ours + λ=8 + smax (53)",
        "our features deep (31)",
        "ours res 256-128-64+skip", "frozen ConvNeXt (1024)"]
r2of(ŷ, y) = 1 - sum(abs2, ŷ .- y)/sum(abs2, y .- mean(y))
for e in SNAPS
    @printf("\nepoch %d\n%-26s", e, "arm")
    for p in PROPS; @printf("%11s", p[1:min(10,end)]); end; println()
    println("-"^(26 + 11*np))
    for a in arms
        haskey(d.preds, a) || continue
        @printf("%-26s", a)
        for j in 1:np
            @printf("%11.3f", r2of(Float64.(d.preds[a][e][:, j]), Float64.(Yte[:, j])))
        end
        println()
    end
end
println("\nmerged into $(joinpath(FIG, "predictions.jls"))")
