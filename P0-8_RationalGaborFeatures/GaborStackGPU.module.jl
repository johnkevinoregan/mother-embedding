# ── PLAIN MODULE — .module.jl, not a Pluto notebook ─────────────────────────
# Included by other files. Opening it in Pluto rewrites it.

"""
    GaborStackGPU

The expensive half of the front end, batched on a CUDA device.

**Why this exists.** `energy_stack` does one forward and `nf` inverse 2D FFTs per image — 38
transforms of a 224×224 complex field for the production bank — and measures ~21 ms/image on 14
CPU threads. That single number is what makes Phase 6 a three-hour job (augmentation multiplies
extraction tenfold) and what would make BSDS painful. Batched complex FFTs are the workload
CUFFT is built for.

**What is and is not ported.**

| stage | where | why |
|:--|:--|:--|
| embed + pad, FFT, per-channel IFFT, `abs2` | **GPU** | ~80 % of the runtime |
| `A₁`, `A₂` | **GPU** | broadcasts and one `argmax`; the flank probe is a constant-offset shift |
| ray transform | *not ported* | see below |
| spatial pooling and feature assembly | **CPU**, existing code | cheap, and reuse means it cannot diverge |

Pooling and assembly deliberately stay on the host, so `Pooling.assemble` runs **unchanged** and
the feature vector comes from the same validated code as the CPU path — the only thing this module
can get wrong is the maps, which `Validate_GPU.jl` checks directly.

**That choice is measured, and it is now the bottleneck.** Per image on an RTX 4090, 112² input,
grid 1, batches of 64:

| stage | ms/image | |
|:--|--:|:--|
| GPU: embed, FFT, 37 inverse transforms, `abs2` | **0.10** | ~100× the CPU cost of the same work |
| GPU: `A₁`, `A₂` | **0.24** | |
| transfer 43 maps to host | 0.75 | |
| **CPU: pooling and assembly** | **1.18** | **52 % of the total** |
| total | **2.27** | against 11.95 CPU, so **4.6× end to end** |

So the FFT port did exactly what it promised — 100× on the part that was 80 % of the CPU cost —
and the end-to-end gain is capped at 4.6× because 85 % of what remains is the host-side pooling
plus the transfer that feeds it.

**The obvious next step, not taken here.** Pooling is `Wt' * reshape(maps, N², k)`, a matmul, and
doing it on the device would shrink the transfer from 43 maps of 112² to 9 × 43 pooled scalars —
about 1500× less traffic — putting the total near 0.4 ms and the overall speedup near 30×. The
cost is that the feature assembly would then have to be reimplemented against pooled arrays
rather than reused, which is precisely the divergence risk this version was built to avoid. Worth
doing, but worth doing with the gate already in place, which it now is.

**The ray transform is not ported.** It needs the same constant-offset sampling primitive that
`shift_sample` already provides, so it is an increment rather than a redesign — but it is 81 of
279 columns and both Phase 10 and Phase 11 measured it as not earning them on these datasets
(+0.54 over the base alone, −0.04 on top of A₁+A₂). The blocks that *are* covered —
`orient, lowpass, A1, A2` — are exactly the configuration every EMNIST phase uses, including
Phase 6.

**Numerics.** CUFFT and FFTW do not agree bit-for-bit, so this path cannot inherit the CPU
path's exactness claims (polarity invariance to 2.4 × 10⁻⁷, padding bit-identical on EMNIST).
The CPU implementation stays the reference; `Validate_GPU.jl` asserts agreement to a stated
tolerance, and any table produced here should say which backend made it.
"""
module GaborStackGPU

using CUDA, Statistics, FFTW

export GPUBank, upload_bank, energy_batch, and_batch, gpu_available

gpu_available() = CUDA.functional()

"The bank's transfer functions resident on the device, uploaded once and reused."
struct GPUBank
    size::Tuple{Int,Int}
    filters::CuArray{ComplexF32,3}       # HF × WF × nf
    meta::Vector{NamedTuple}
end

function upload_bank(bank)
    HF, WF = bank.size
    nf = length(bank.filters)
    F = Array{ComplexF32,3}(undef, HF, WF, nf)
    for k in 1:nf; F[:, :, k] = ComplexF32.(bank.filters[k]); end
    GPUBank(bank.size, CuArray(F), bank.meta)
end

"""
Embed a batch into the padded field with **replicate** padding.

Replicate padding is exactly a clamp of the source index, so the whole operation is one gather
with clamped index vectors — no branching and no separate border pass. `:replicate` is the
production default because zero-padding a non-zero background puts a full-contrast step round
the border whose cross term flips sign with contrast polarity, which is the bug Phase 9 found.
"""
function embed_batch(imgs::CuArray{Float32,3}, fieldsize::Tuple{Int,Int})
    h, w, B = size(imgs)
    HF, WF = fieldsize
    oy = (HF - h) ÷ 2; ox = (WF - w) ÷ 2
    iy = CuArray(clamp.((1:HF) .- oy, 1, h))
    ix = CuArray(clamp.((1:WF) .- ox, 1, w))
    imgs[iy, ix, :], (oy, ox)
end

"""
    energy_batch(imgs, gb; crop_to) -> E  (N × N × nf × B)

Quadrature energy for a whole batch. One batched forward transform, then one batched inverse per
channel — the loop is over channels, not images, so every transform is `B`-wide.
"""
function energy_batch(imgs::CuArray{Float32,3}, gb::GPUBank; crop_to::Int=size(imgs, 1))
    h, w, B = size(imgs)
    HF, WF = gb.size
    F, (oy, ox) = embed_batch(imgs, gb.size)
    Ff = fft(ComplexF32.(F), (1, 2))
    nf = size(gb.filters, 3)
    ys = oy+1 : oy+crop_to
    xs = ox+1 : ox+crop_to
    E = CuArray{Float32,4}(undef, crop_to, crop_to, nf, B)
    for k in 1:nf
        r = ifft(Ff .* view(gb.filters, :, :, k), (1, 2))
        E[:, :, k, :] .= abs2.(view(r, ys, xs, :))
    end
    E
end

"""
    shift_sample(M, dy, dx) -> array of the same shape

Bilinear sample of `M[:, :, b]` at `(y + dy, x + dx)`, zero outside the frame, for a **constant**
offset. Because the offset does not vary with position, the four bilinear taps are fixed integer
shifts with fixed weights, so this is four gathers and a broadcast rather than a scatter.

Zero-outside matters: it is what `AndLayer.bilin` does, and an end-stop or a ray probe that fell
back on the border value instead would read structure that is not there.
"""
function shift_sample(M::AbstractArray{Float32,3}, dy::Float64, dx::Float64)
    N = size(M, 1)
    ys = (1:N) .+ dy; xs = (1:N) .+ dx
    y0 = floor.(Int, ys); x0 = floor.(Int, xs)
    fy = Float32.(ys .- y0); fx = Float32.(xs .- x0)
    okY = (ys .>= 1) .& (ys .<= N); okX = (xs .>= 1) .& (xs .<= N)
    cy0 = CuArray(clamp.(y0, 1, N)); cy1 = CuArray(clamp.(y0 .+ 1, 1, N))
    cx0 = CuArray(clamp.(x0, 1, N)); cx1 = CuArray(clamp.(x0 .+ 1, 1, N))
    wy0 = CuArray(reshape((1 .- fy) .* okY, N, 1, 1)); wy1 = CuArray(reshape(fy .* okY, N, 1, 1))
    wx0 = CuArray(reshape((1 .- fx) .* okX, 1, N, 1)); wx1 = CuArray(reshape(fx .* okX, 1, N, 1))
    @views (wy0 .* wx0) .* M[cy0, cx0, :] .+ (wy1 .* wx0) .* M[cy1, cx0, :] .+
           (wy0 .* wx1) .* M[cy0, cx1, :] .+ (wy1 .* wx1) .* M[cy1, cx1, :]
end

_scales(meta) = unique(Float64(m.rho0) for m in meta if m.kind === :oriented)
function _channels(meta, ρ)
    ch = Tuple{Int,Float64}[(Int(i), Float64(m.theta)) for (i, m) in enumerate(meta)
                            if m.kind === :oriented && m.rho0 ≈ ρ]
    sort!(ch; by = last); ch
end
function _sigma_phi(meta, ρ)
    for m in meta
        m.kind === :oriented && m.rho0 ≈ ρ && return Float64(m.sigma_phi)
    end
    error("no oriented channel at ρ=$ρ")
end

"""
    and_batch(E, meta; a1_floor, kappa, d_factor, ...) -> (A, labels)

`A₁` and `A₂` for a batch, returning `N × N × 2·nscales × B` in the same channel order the CPU
`and_maps` produces, so the host side can hand it straight to `Pooling.assemble`.

`A₁` is `max(0, S/C₀ − c·C₀)` with `c` the closed-form i1D floor — the production default since
the A₁ leakage finding. `A₂` takes the **locally dominant** orientation via `argmax` over
channels, then probes the two flanks along the stroke at a constant offset.
"""
function and_batch(E::CuArray{Float32,4}, meta;
                   a1_floor::Symbol=:analytic, a1_floor_fn=nothing,
                   kappa::Float32=0.5f0, eps::Float32=1f-12,
                   d_factor::Real=1.0, d_anchor::Symbol=:envelope,
                   structure_scale::Union{Nothing,Real}=nothing)
    N, _, _, B = size(E)
    ρs = _scales(meta)
    maps = CuArray{Float32,3}[]; labels = NamedTuple[]

    for ρ in ρs                                             # ---- A1
        ch = _channels(meta, ρ); n = length(ch)
        iseven(n) || error("A1 needs an even orientation count at ρ=$ρ (got $n)")
        half = n ÷ 2
        idx = [i for (i, _) in ch]
        C0 = sum(view(E, :, :, i, :) for i in idx)
        S = sum(view(E, :, :, idx[k], :) .* view(E, :, :, idx[mod1(k + half, n)], :)
                for k in 1:n)
        c = a1_floor === :analytic ?
            Float32(a1_floor_fn === nothing ? 0.0 : a1_floor_fn(n, _sigma_phi(meta, ρ))) : 0f0
        A = ifelse.(C0 .> eps, max.(0f0, S ./ max.(C0, eps) .- c .* C0), 0f0)
        push!(maps, A); push!(labels, (form=:A1, rho0=ρ))
    end

    for ρ in ρs                                             # ---- A2
        ch = _channels(meta, ρ)
        dρ = if d_anchor === :structure
            structure_scale === nothing && error("d_anchor=:structure needs structure_scale")
            d_factor * Float64(structure_scale)
        else
            m = first(m for m in meta if m.kind === :oriented && m.rho0 ≈ ρ)
            d_factor * m.imwidth / (2π * ρ * m.sigma_phi)
        end
        # Winner-take-all over orientation, as a real end-stopped cell does: an off-orientation
        # channel has near-empty flanks, so its ratio is ill-conditioned and contributes spurious
        # asymmetry if allowed in.
        #
        # `>` and not `>=`, and built by the same running scan the CPU uses. The CPU replaces the
        # incumbent only on a *strict* improvement, so on an exact tie the lowest-θ channel wins.
        # A single `argmax`/`maximum` plus `.>=` would hand ties to the highest θ instead and the
        # two paths would disagree on flat regions — which is exactly the kind of difference a
        # tolerance test can miss, because it shows up on a few pixels at full magnitude rather
        # than everywhere at small magnitude.
        best = CUDA.fill(-Inf32, N, N, B)
        bidx = CUDA.zeros(Int32, N, N, B)
        for (ci, (i, _)) in enumerate(ch)
            Ei = view(E, :, :, i, :)
            newbest = Ei .> best
            bidx = ifelse.(newbest, Int32(ci), bidx)
            best = ifelse.(newbest, Ei, best)
        end
        A = CUDA.zeros(Float32, N, N, B)
        for (ci, (i, θc)) in enumerate(ch)
            θs = θc + π/2
            dy = dρ * sin(θs); dx = dρ * cos(θs)
            Ei = view(E, :, :, i, :)
            ep = shift_sample(Ei, dy, dx)
            em = shift_sample(Ei, -dy, -dx)
            win = (bidx .== Int32(ci)) .& (best .> eps)
            A = ifelse.(win, best .* abs.(ep .- em) ./ (ep .+ em .+ kappa .* best), A)
        end
        push!(maps, A); push!(labels, (form=:A2, rho0=ρ, d=dρ))
    end

    out = CuArray{Float32,4}(undef, N, N, length(maps), B)
    for (k, M) in enumerate(maps); out[:, :, k, :] .= M; end
    out, labels
end

end # module
