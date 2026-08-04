# ── PLAIN MODULE, not a notebook — do not open in Pluto ─────────────────────
#
# The Phase 9 readout protocol, so a ConvNeXt arm is scored by exactly the same rules as
# every arm in `P9+12_SimpleStrokeTests/RESULTS.md`.
#
# EVERY FUNCTION HERE IS COPIED VERBATIM FROM `P9+12_SimpleStrokeTests/Phase9_Readouts.jl`, with
# only the ENV-driven constants turned into keyword arguments. It is duplicated rather than
# `include`d because that file runs `main()` on load — including it would regenerate 20,000
# stimuli and run the whole Phase 9 experiment as a side effect. **If the Phase 9 readout
# changes, change it here too**, or the comparison stops being a comparison.
#
# The docstrings are kept because each one records a bug that the implementation is shaped
# around, and those are the parts most easily lost in a copy.
module Readout

using Statistics, Random, LinearAlgebra, Printf
using Flux, CUDA

export r2, nanmean, zfit, zapply, ridge, mlp, trivial_baseline

"R² with a NaN rather than a number when the target has no variance to explain."
function r2(ŷ, y)
    sst = sum(abs2, y .- mean(y))
    sst <= 0 ? NaN : 1 - sum(abs2, ŷ .- y) / sst
end

"""
Per-property R² of a linear fit on three scalar image summaries — total contrast mass, mean
level, sd. This is the floor every arm has to clear to have shown anything, and it exists
because Phase 8 mistook a cue nobody had measured for a result.
"""
function trivial_baseline(imgs, Ytr, Yte, ntr)
    summ(im) = (sum(abs.(im .- median(im))), mean(im), std(im))
    S = reduce(vcat, [collect(summ(im))' for im in imgs])
    A = hcat(ones(size(S,1)), S)
    Atr, Ate = A[1:ntr, :], A[ntr+1:end, :]
    [r2(Ate * (Atr \ Ytr[:, j]), Yte[:, j]) for j in 1:size(Ytr, 2)]
end

"""
Mean over the properties that have a defined R².

A held-out nuisance is constant in training, so its own row has zero variance and scores
`NaN`. Averaging that in makes the selection metric `NaN`, `NaN > best` is false at every
epoch, and the model records no prediction at all — which showed up as both MLP arms and
the CNN scoring a flat 0.000 across the polarity split, looking exactly like three
independent collapses rather than one bug in a comparison.
"""
nanmean(v) = (u = filter(!isnan, v); isempty(u) ? NaN : mean(u))

"""
Column standardisation fitted on training data only. Constant columns get σ = 1 so they map
to zero rather than to NaN — a held-out nuisance makes whole columns constant, and silently
producing NaNs there would poison every property, not just the held-out one.
"""
function zfit(X)
    μ = vec(mean(X, dims=1)); σ = vec(std(X, dims=1)); σ[σ .<= 1e-8] .= 1f0
    Float32.(μ), Float32.(σ)
end
zapply(X, μ, σ) = (X .- μ') ./ σ'

"""
Closed-form ridge with the penalty chosen **per property** on a validation split, because
the properties differ by orders of magnitude in how much signal they carry and a single λ
tuned on their average would under-regularise the easy rows and over-regularise the hard
ones.

The penalty is scaled by the mean diagonal of the Gram matrix rather than given in absolute
units: columns are standardised so that diagonal is ≈ n, and the same grid then means the
same thing whatever the sample size. With wide feature matrices an absolute λ of 0.01
leaves the matrix numerically singular in Float32 and the factorisation simply fails.
"""
function ridge(Xtr, Ytr, Xva, Yva, Xte; cs=Float32[1f-3, 1f-2, 1f-1, 1f0, 1f1, 1f2, 1f3])
    G  = Xtr' * Xtr
    B  = Xtr' * Ytr
    P  = size(Ytr, 2)
    best = fill(-Inf, P); bestλ = zeros(Float32, P)
    Pte = zeros(Float32, size(Xte, 1), P)
    scale = Float32(mean(diag(G)))
    for c in cs
        F = try cholesky(Symmetric(G + (c*scale)*I)) catch; continue end
        W = F \ B
        Va = Xva * W; Te = Xte * W
        for j in 1:P
            s = r2(Va[:, j], Yva[:, j])
            if s > best[j]; best[j] = s; bestλ[j] = c*scale; Pte[:, j] = Te[:, j]; end
        end
    end
    Pte, bestλ
end

"""
Two hidden layers, Adam, with the epoch chosen by validation R² averaged over the properties
that have one. The reported prediction is from the **best** epoch, not the last, so a
diverging run is scored where it was actually good rather than where it stopped.

Returns `(predictions, history)`; the history is kept because this project's convention is to
plot accuracy against epoch rather than report a final number.
"""
function mlp(Xtr, Ytr, Xva, Yva, Xte; hidden=256, epochs=100, seed=1, bs=128, usegpu=true)
    Random.seed!(seed)
    dev = (usegpu && CUDA.functional()) ? gpu : cpu
    A = dev(permutedims(Xtr)); V = dev(permutedims(Xva)); T = dev(permutedims(Xte))
    Yt = dev(permutedims(Ytr)); Yv = permutedims(Yva)
    m = Chain(Dense(size(Xtr,2) => hidden, relu), Dense(hidden => hidden, relu),
              Dense(hidden => size(Ytr,2))) |> dev
    opt = Flux.setup(Flux.Adam(1f-3), m)
    n = size(A, 2); best = -Inf; bestP = zeros(Float32, size(T,2), size(Yt,1))
    hist = fill(NaN, epochs, size(Yv, 1)); lossh = fill(NaN, epochs)
    for e in 1:epochs
        tot = 0.0; nb = 0
        for i in Iterators.partition(randperm(n), bs)
            l, gs = Flux.withgradient(mm -> Flux.mse(mm(A[:, i]), Yt[:, i]), m)
            Flux.update!(opt, m, gs[1]); tot += l; nb += 1
        end
        lossh[e] = tot / max(nb, 1)
        Pv = Array(m(V))
        for j in 1:size(Yv, 1); hist[e, j] = r2(Pv[j, :], Yv[j, :]); end
        s = nanmean(hist[e, :])
        if !isnan(s) && s > best; best = s; bestP = permutedims(Array(m(T))); end
    end
    bestP, (val=hist, loss=lossh, best=best)
end

end # module
