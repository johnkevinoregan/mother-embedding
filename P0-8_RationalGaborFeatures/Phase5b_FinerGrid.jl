# ── PLAIN SCRIPT, not a Pluto notebook ──────────────────────────────────────
# Run with `julia --project=. -t 8 <this file>`. Opening it in Pluto rewrites it.
#
# Phase 5b — is the AND layer redundant, or is 3×3 pooling destroying it?
#
# Phase 5a found A1+A2 reach 88.45 % alone from 54 columns yet add nothing on top of the
# orientation statistics. Two explanations fit that:
#
#   (a) REDUNDANT  — the co-location signal is real but the classes are already separable
#                    by the orientation statistics, so it buys nothing.
#   (b) DESTROYED  — A1 is a *point* property read out over 37 px cells; a sharp peak
#                    averaged over a cell that size may simply not survive to the vector.
#
# (b) predicts the A blocks improve with a finer grid; (a) predicts they do not. So: pool
# the A blocks at 3×3, 6×6 and 11×11 (the latter being the demodulation-Nyquist grid at
# σ_along = 9.7 px) while holding the orient/lowpass baseline at 3×3.
#
# Two controls, both necessary:
#
#   * ORIENT AT THE SAME GRIDS. If finer pooling helps the A blocks it may simply help
#     everything, in which case the finding is about the grid and not about conjunction.
#   * A SHUFFLED TWIN for every A arm. Going 3×3 → 11×11 takes the A blocks from 54 to
#     726 columns, and §7.8 established that a fixed projection into a few hundred
#     dimensions plus a trained head is a strong baseline WHATEVER the projection is.
#     Without the twin, "finer pooling helped" and "13× more columns helped" are the same
#     measurement.
#
# ~11 minutes of extraction, then seconds per arm.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Statistics, Printf, Random, LinearAlgebra, Serialization, FFTW
using Flux, OneHotArrays, Plots
include(joinpath(@__DIR__, "GaborStack.module.jl"))
include(joinpath(@__DIR__, "AndLayer.module.jl"))
include(joinpath(@__DIR__, "Pooling.module.jl"))
include(joinpath(@__DIR__, "..", "LoadEMNIST.module.jl"))
using .GaborStack, .AndLayer, .Pooling, .LoadEMNIST

const IMG   = 112
const GRIDS = [3, 6, 11]
const CACHE = get(ENV, "P5B_CACHE", joinpath(tempdir(), "p5b_features.jls"))
const NIMG  = parse(Int, get(ENV, "P5B_NIMG", "0"))
BLAS.set_num_threads(2); FFTW.set_num_threads(1)

@inline function bilinear(M, y, x)
    H, W = size(M); (y < 1 || x < 1 || y > H || x > W) && return 0f0
    y0, x0 = floor(Int, y), floor(Int, x); y1, x1 = min(y0+1,H), min(x0+1,W)
    fy, fx = y - y0, x - x0
    (1-fy)*(1-fx)*M[y0,x0] + fy*(1-fx)*M[y1,x0] + (1-fy)*fx*M[y0,x1] + fy*fx*M[y1,x1]
end
function upsample(img, N=IMG)
    H, W = size(img); out = zeros(Float32, N, N)
    @inbounds for i in 1:N, j in 1:N
        out[i,j] = bilinear(img, Float32(1+(i-1)*(H-1)/(N-1)), Float32(1+(j-1)*(W-1)/(N-1)))
    end
    out
end

const LADDER = [2.0, 3.742, 7.0]
const BETAS  = [2.0, 1.6, 1.2]
const NORI   = [8, 12, 16]
const HF, WF, _ = field_for((IMG,IMG), LADDER; n_orient=NORI, beta=BETAS)
const BANK = make_bank((HF,WF), LADDER; imwidth=IMG, n_orient=NORI, beta=BETAS)
const WTS  = Dict(g => grid_weights(IMG, IMG, g) for g in GRIDS)
const SPEC = Dict(g => PoolSpec(grid=g, blocks=(:orient,:lowpass,:A1,:A2)) for g in GRIDS)

"One dense pass per image; pooled at every grid from the same maps."
function all_grids(big)
    Es = energy_stack(big, BANK)
    A, al = and_maps(Es, BANK.meta; forms=(:A1,:A2))
    [assemble(Es, BANK.meta, A, al, SPEC[g]; Wts=WTS[g]) for g in GRIDS]
end

function extract(imgs)
    n = size(imgs,3)
    probe = all_grids(upsample(@view imgs[:,:,1]))
    Fs  = [zeros(Float32, n, length(p[1])) for p in probe]
    LAB = [p[2] for p in probe]
    Threads.@threads for i in 1:n
        r = all_grids(upsample(@view imgs[:,:,i]))
        for k in eachindex(Fs); Fs[k][i,:] = r[k][1]; end
    end
    Fs, LAB
end

D = joinpath(homedir(),"Julia","DATABASES","EMNIST"); S = joinpath(D,"emnist_source_files")
tr_i = read_emnist_images(joinpath(D,"emnist-balanced-train-images-idx3-ubyte"))
tr_l = read_emnist_labels(joinpath(D,"emnist-balanced-train-labels-idx1-ubyte"))
te_i = read_emnist_images(joinpath(S,"emnist-balanced-test-images-idx3-ubyte"))
te_l = read_emnist_labels(joinpath(S,"emnist-balanced-test-labels-idx1-ubyte"))
if NIMG > 0
    tr_i = tr_i[:,:,1:NIMG]; tr_l = tr_l[1:NIMG]
    te_i = te_i[:,:,1:min(NIMG,size(te_i,3))]; te_l = te_l[1:min(NIMG,length(te_l))]
end

if isfile(CACHE)
    C = deserialize(CACHE); TR, TE, LAB = C.TR, C.TE, C.LAB
    println("loaded cached features from $CACHE")
else
    t0 = time(); TR, LAB = extract(tr_i); TE, _ = extract(te_i)
    @printf("extraction: %.1f min on %d threads\n", (time()-t0)/60, Threads.nthreads())
    serialize(CACHE, (; TR, TE, LAB))
end
for (k,g) in enumerate(GRIDS)
    @printf("grid %2d×%-2d → %5d features\n", g, g, size(TR[k],2))
end

NAMES = [LoadEMNIST.emnist_class_name(i) for i in 1:47]
HOMO  = [["0","O"],["1","I","L"],["2","Z"],["5","S"],["9","g","q"]]
canon = Dict(s=>s for s in NAMES); for g in HOMO, s in g; canon[s]=g[1]; end
reps  = sort(unique(values(canon))); const NC = length(reps)
mlab(y) = [findfirst(==(canon[NAMES[c]]), reps) for c in y]
ytr, yte = mlab(tr_l), mlab(te_l)
@printf("homoglyph-merged: %d classes, chance %.2f %%\n\n", NC, 100/NC)

function standardise(a, b; clip=3f0)
    μ = vec(mean(a,dims=1)); σ = vec(std(a,dims=1)); σ[σ .<= 0] .= 1f0
    clamp.((a .- μ')./σ', -clip, clip), clamp.((b .- μ')./σ', -clip, clip)
end
function fit(Xtr, Xte; hidden=256, epochs=15, batch=128, lr=1f-3, seed=1)
    a,b = standardise(Xtr,Xte); A = permutedims(a); B = permutedims(b)
    Random.seed!(seed)
    m = Chain(Dense(size(A,1)=>hidden,relu), Dense(hidden=>NC))
    opt = Flux.setup(Flux.Adam(lr), m); Y = onehotbatch(ytr,1:NC); n = size(A,2)
    curve = Float64[]
    for _ in 1:epochs
        p = randperm(n)
        for i in 1:batch:n
            idx = p[i:min(i+batch-1,n)]
            _,gs = Flux.withgradient(mm->Flux.logitcrossentropy(mm(A[:,idx]),Y[:,idx]), m)
            Flux.update!(opt,m,gs[1])
        end
        pr = Int[]
        for i in 1:10000:size(B,2)
            j = min(i+9999,size(B,2)); append!(pr, onecold(m(view(B,:,i:j)),1:NC))
        end
        push!(curve, mean(pr .== yte))
    end
    curve
end

gi(g) = findfirst(==(g), GRIDS)
sel(g, pfx) = findall(x -> any(startswith(x, p) for p in pfx), LAB[gi(g)])
Ablk(g) = sel(g, ("A1","A2")); OLblk(g) = sel(g, ("orient","lowpass"))

"""
The dimensionality control: permute the block's rows independently in train and test, so
the columns keep their marginal distribution and their count but carry no per-sample
information. Any gain that survives this is the block; any gain that does not was the
column count.
"""
function with_shuffled(base_tr, base_te, add_tr, add_te; seed=7)
    a = copy(add_tr); b = copy(add_te)
    shuffle_block!(a, 1:size(a,2); seed=seed)
    shuffle_block!(b, 1:size(b,2); seed=seed+1)
    hcat(base_tr, a), hcat(base_te, b)
end

println("="^86)
println("PHASE 5b — does a finer grid rescue the AND blocks?")
println("="^86)
arms = Tuple{String,Matrix{Float32},Matrix{Float32}}[]
base_tr = TR[gi(3)][:, OLblk(3)]; base_te = TE[gi(3)][:, OLblk(3)]
push!(arms, ("orient+lp 3×3 (baseline)", base_tr, base_te))
for g in GRIDS
    push!(arms, ("orient+lp $(g)×$(g) alone", TR[gi(g)][:,OLblk(g)], TE[gi(g)][:,OLblk(g)]))
    push!(arms, ("A1+A2 $(g)×$(g) alone",     TR[gi(g)][:,Ablk(g)],  TE[gi(g)][:,Ablk(g)]))
    push!(arms, ("base + A $(g)×$(g)",
                 hcat(base_tr, TR[gi(g)][:,Ablk(g)]), hcat(base_te, TE[gi(g)][:,Ablk(g)])))
    st, se = with_shuffled(base_tr, base_te, TR[gi(g)][:,Ablk(g)], TE[gi(g)][:,Ablk(g)])
    push!(arms, ("base + A $(g)×$(g) SHUFFLED", st, se))
end

@printf("\n%-34s %6s %9s %9s\n", "arm", "n", "final", "best")
curves = Dict{String,Vector{Float64}}()
for (nm,a,b) in arms
    c = fit(a,b); curves[nm] = c
    @printf("%-34s %6d %8.2f %% %8.2f %%\n", nm, size(a,2), 100c[end], 100maximum(c))
    flush(stdout)
end

base = 100*curves["orient+lp 3×3 (baseline)"][end]
println("\n" * "="^86)
@printf("baseline: orient+lowpass 3×3 = %.2f %%\n\n", base)
@printf("%-8s %10s %11s %11s %11s %11s\n",
        "A grid", "A alone", "base+A", "Δ vs base", "shuffled", "Δ shuffled")
for g in GRIDS
    a  = 100*curves["A1+A2 $(g)×$(g) alone"][end]
    ba = 100*curves["base + A $(g)×$(g)"][end]
    sh = 100*curves["base + A $(g)×$(g) SHUFFLED"][end]
    @printf("%-8s %9.2f %% %10.2f %% %+10.2f %10.2f %% %+10.2f\n",
            "$(g)×$(g)", a, ba, ba - base, sh, sh - base)
end
println("""
Read it this way. `Δ vs base` is the effect of adding the A block. The shuffled twin has
the same column count with the correspondence destroyed, so `Δ shuffled` is what those
columns cost as pure noise — it is a floor, not a rival. A positive `Δ vs base` is a real
gain; a negative one means the block does not help however it is pooled.

The grid control is the `orient+lp g×g alone` rows: if finer pooling lifts those too, the
finding is about the grid rather than about conjunction.""")

serialize(joinpath(dirname(CACHE), "p5b_curves.jls"), curves)
p = plot(xlabel="epoch", ylabel="test accuracy (%)", legend=:bottomright, xlims=(1,15),
         grid=true, gridalpha=0.25, size=(950,470), titlefontsize=10,
         left_margin=8Plots.mm, bottom_margin=4Plots.mm,
         title="Phase 5b — pooling the AND blocks more finely ($(NC) merged classes)")
for nm in ["orient+lp 3×3 (baseline)", "A1+A2 3×3 alone", "A1+A2 11×11 alone",
           "base + A 11×11", "base + A 11×11 SHUFFLED"]
    haskey(curves,nm) || continue
    plot!(p, 1:length(curves[nm]), 100 .*curves[nm]; lw=2, marker=:circle, ms=3, label=nm)
end
savefig(p, joinpath(@__DIR__, "figures", "phase5b_curves.png"))
println("\nwrote figures/phase5b_curves.png")
