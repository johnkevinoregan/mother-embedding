# ── PLAIN SCRIPT, not a Pluto notebook — runs headless as a gate ────────────
# `julia --project=.. Validate_GPU.jl`   (from the P0-8_RationalGaborFeatures directory)
#
# Does the CUDA path compute the same thing as the CPU path?
#
# This gate exists because the GPU path **cannot inherit the CPU path's exactness claims**.
# CUFFT and FFTW do not agree bit-for-bit, so results like "polarity invariance exact to
# 2.4 × 10⁻⁷" and "padding bit-identical on EMNIST" are properties of the reference
# implementation and have to be re-established, not assumed, for the accelerated one. This
# project has already been bitten once by a number that would not reproduce with no code change
# to explain it (Phase 5a), so the CPU implementation stays authoritative and this asserts
# agreement to a stated tolerance.
#
# WHAT IS COMPARED. The maps, not the features — because the GPU path deliberately hands its maps
# to the *same* `Pooling.assemble` the CPU path uses, so if the maps agree the features agree by
# construction. Checking the maps also localises any failure to the operator that caused it.
#
# TOLERANCES. Relative error against the CPU value, scaled by each map's own RMS so a channel
# that is small everywhere is not held to an absolute bar it cannot meet. Float32 FFT round-trip
# on a 224² field accumulates ~1e-6 relative; A₂ divides by a conditioned denominator and can
# amplify that, so it gets a looser bound. Anything far outside these is a real disagreement.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Printf, Statistics, LinearAlgebra, Random
using CUDA
include(joinpath(@__DIR__, "GaborStack.module.jl"))
include(joinpath(@__DIR__, "AndLayer.module.jl"))
include(joinpath(@__DIR__, "GaborStackGPU.module.jl"))
include(joinpath(@__DIR__, "Stimuli.module.jl"))
using .GaborStack, .AndLayer, .GaborStackGPU, .Stimuli

const N      = 112
const LADDER = [2.0, 3.742, 7.0]
const BETAS  = [2.0, 1.6, 1.2]
const NORI   = [8, 12, 16]
const TOL_E  = 1f-4     # oriented energy, relative to the channel's RMS
const TOL_A  = 3f-3     # A₁/A₂ — a conditioned ratio amplifies the FFT's Float32 error

rel(a, b) = maximum(abs.(a .- b)) / max(sqrt(mean(abs2, a)), eps(Float32))

function main()
    if !CUDA.functional()
        println("CUDA not functional — nothing to validate. Skipping (exit 0).")
        return true
    end
    @printf("Validating the CUDA front end against the CPU reference\n")
    @printf("  %s\n\n", CUDA.name(CUDA.device()))

    HF, WF, _ = field_for((N, N), LADDER; n_orient=NORI, beta=BETAS)
    bank = make_bank((HF, WF), LADDER; imwidth=N, n_orient=NORI, beta=BETAS)
    gb = upload_bank(bank)

    # A batch that exercises every branch: an empty field (the eps guards), a plain bar (i1D),
    # a corner and a cross (i2D), a bar at a diagonal (the padding path), and noise.
    rng = MersenneTwister(7)
    imgs = [
        fill(0.5f0, N, N),
        barstim(N, 0.0; w=13.0, len=70.0),
        barstim(N, deg2rad(37); w=9.0, len=80.0),
        corner(N, deg2rad(90); w=13.0, len=40.0),
        cross_bars(N; w=11.0, len=70.0),
        tee(N; w=13.0, len=70.0),
        Float32.(rand(rng, N, N)),
        Float32.(0.5 .+ 0.4 .* sin.((1:N) .* 0.3) .* ones(1, N)),
    ]
    B = length(imgs)
    @printf("%d stimuli in one batch: empty, bar 0°, bar 37°, corner, cross, tee, noise, grating\n\n", B)

    # ── CPU reference, one image at a time, exactly as production does it
    Ecpu = Array{Float32,4}(undef, N, N, length(bank.filters), B)
    Acpu = nothing; alab = nothing
    for b in 1:B
        E = energy_stack(imgs[b], bank; mode=:replicate, crop=true)
        Ecpu[:, :, :, b] = E
        A, l = and_maps(E, bank.meta; forms=(:A1, :A2))
        if Acpu === nothing
            Acpu = Array{Float32,4}(undef, N, N, size(A, 3), B); alab = l
        end
        Acpu[:, :, :, b] = A
    end

    # ── GPU, whole batch at once
    gimgs = CuArray(reshape(reduce(hcat, [vec(i) for i in imgs]), N, N, B))
    Egpu = energy_batch(gimgs, gb; crop_to=N)
    Agpu, glab = and_batch(Egpu, bank.meta; a1_floor=:analytic,
                           a1_floor_fn=AndLayer.a1_i1d_floor)
    Eh = Array(Egpu); Ah = Array(Agpu)

    fails = String[]
    @printf("%-26s %12s %12s %10s\n", "quantity", "rel error", "tolerance", "verdict")

    # labels must line up or the comparison is meaningless even if the numbers agree
    if [(l.form, round(Float64(l.rho0), digits=3)) for l in alab] !=
       [(l.form, round(Float64(l.rho0), digits=3)) for l in glab]
        push!(fails, "A-map channel order differs between CPU and GPU")
        println("  CPU: ", [(l.form, l.rho0) for l in alab])
        println("  GPU: ", [(l.form, l.rho0) for l in glab])
    end

    for (k, m) in enumerate(bank.meta)
        e = rel(Ecpu[:, :, k, :], Eh[:, :, k, :])
        ok = e < TOL_E
        nm = m.kind === :lowpass ? "lowpass" :
             @sprintf("orient ρ=%.2f θ=%3.0f°", m.rho0, rad2deg(m.theta))
        ok || push!(fails, @sprintf("energy channel %d (%s): rel %.2e", k, nm, e))
        # only print the worst few, otherwise 37 lines of pass
        e > TOL_E / 10 && @printf("%-26s %12.2e %12.1e %10s\n", nm, e, TOL_E, ok ? "pass" : "FAIL")
    end
    worstE = maximum(rel(Ecpu[:, :, k, :], Eh[:, :, k, :]) for k in 1:length(bank.meta))
    @printf("%-26s %12.2e %12.1e %10s\n", "energy, worst channel", worstE, TOL_E,
            worstE < TOL_E ? "pass" : "FAIL")

    for (k, l) in enumerate(alab)
        e = rel(Acpu[:, :, k, :], Ah[:, :, k, :])
        ok = e < TOL_A
        nm = @sprintf("%s ρ=%.2f", l.form, l.rho0)
        ok || push!(fails, @sprintf("%s: rel %.2e", nm, e))
        @printf("%-26s %12.2e %12.1e %10s\n", nm, e, TOL_A, ok ? "pass" : "FAIL")
    end

    # A2's winner-take-all is a discrete choice, so a tolerance test can hide a tie-break
    # disagreement: it would show up on a handful of pixels at full magnitude rather than
    # everywhere at small magnitude. Count them directly.
    nbig = 0
    for k in 1:size(Acpu, 3)
        d = abs.(Acpu[:, :, k, :] .- Ah[:, :, k, :])
        s = sqrt(mean(abs2, Acpu[:, :, k, :]))
        nbig += count(>(0.05f0 * max(s, eps(Float32))), d)
    end
    tot = length(Acpu)
    @printf("\npixels differing by >5%% of RMS: %d of %d (%.4f %%)\n", nbig, tot, 100nbig/tot)
    nbig > tot ÷ 1000 && push!(fails, "$nbig pixels differ by >5 % of RMS — likely a tie-break or indexing difference, not float error")

    println("\n" * "="^72)
    if isempty(fails)
        println("ALL GATES PASSED — the CUDA path matches the CPU reference within tolerance.")
        println("""
The CPU implementation remains the reference. Tables produced on the GPU path should say
so: agreement here is numerical, not bit-exact, and CUFFT's rounding is not FFTW's.
Not covered: the ray transform, which is not ported — see GaborStackGPU.""")
    else
        println("GATES FAILED:"); for f in fails; println("  - ", f); end
    end
    isempty(fails)
end

main() || exit(1)
