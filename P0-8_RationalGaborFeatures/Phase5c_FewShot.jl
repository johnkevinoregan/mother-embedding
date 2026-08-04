# ── PLAIN SCRIPT, not a Pluto notebook ──────────────────────────────────────
# Run with `julia --project=. -t 8 <this file>`. Opening it in Pluto rewrites it.
#
# Phase 5c — does the AND layer help when data is scarce?
#
# Phase 5a found A1+A2 add +0.01 at full data. Reading the accuracy-vs-epoch curves
# suggested a mechanism: the +A1+A2 arm sits slightly above the others for the first four
# epochs (+0.16 mean) and the gap closes by epoch 12 (+0.01). That is below the 0.186 %
# standard error at one seed, so it is not evidence on its own — but it points at a
# hypothesis that IS testable: the conjunctions supply something the MLP can otherwise
# LEARN from the orientation statistics, given enough data. If so the advantage should
# grow as data shrinks.
#
# That is the project's actual thesis — built-in invariances should make generalisation
# from few examples easier — so it deserves a proper measurement rather than an argument
# about a 0.16 % difference.
#
#   PREDICTION (on record): if the mechanism is right, A1+A2 add several points at k = 5-10
#   per class and decay toward the +0.01 measured at full data.
#
# Design:
#   * features come from the Phase 5a cache (3x3, 198 columns) - no re-extraction
#   * subsets are PAIRED: one draw per (k, seed), every arm trains on the same images
#   * fixed 4000-step budget, so k does not silently buy extra gradient updates
#   * a SHUFFLED twin at every k, because 54 extra columns can help a data-starved model
#     for reasons that have nothing to do with their content

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Statistics, Printf, Random, LinearAlgebra, Serialization
using Flux, OneHotArrays, Plots
include(joinpath(@__DIR__, "Pooling.module.jl"))
include(joinpath(@__DIR__, "..", "LoadEMNIST.module.jl"))
using .Pooling, .LoadEMNIST

const CACHE = get(ENV, "P5C_CACHE", joinpath(tempdir(), "p5a_features.jls"))
const KS    = parse.(Int, split(get(ENV, "P5C_KS", "5,10,20,50,100,400"), ","))
const SEEDS = 1:parse(Int, get(ENV, "P5C_SEEDS", "5"))
const STEPS = parse(Int, get(ENV, "P5C_STEPS", "4000"))
BLAS.set_num_threads(8)

isfile(CACHE) || error("no Phase 5a feature cache at $CACHE — run Phase5a_EMNIST.jl first")
C = deserialize(CACHE)
NEWtr, NEWte, LAB = C.NEWtr, C.NEWte, C.LAB

D = joinpath(homedir(),"Julia","DATABASES","EMNIST"); S = joinpath(D,"emnist_source_files")
tr_l = read_emnist_labels(joinpath(D,"emnist-balanced-train-labels-idx1-ubyte"))
te_l = read_emnist_labels(joinpath(S,"emnist-balanced-test-labels-idx1-ubyte"))
NAMES = [LoadEMNIST.emnist_class_name(i) for i in 1:47]
HOMO  = [["0","O"],["1","I","L"],["2","Z"],["5","S"],["9","g","q"]]
canon = Dict(s=>s for s in NAMES); for g in HOMO, s in g; canon[s]=g[1]; end
reps  = sort(unique(values(canon))); const NC = length(reps)
mlab(y) = [findfirst(==(canon[NAMES[c]]), reps) for c in y]
ytr, yte = mlab(tr_l), mlab(te_l)

cols(p) = findall(x -> startswith(x, p), LAB)
const OL = vcat(cols("orient"), cols("lowpass"))
const AB = vcat(cols("A1"), cols("A2"))
@printf("orient+lowpass %d cols · A1+A2 %d cols · %d merged classes, chance %.2f %%\n\n",
        length(OL), length(AB), NC, 100/NC)

"k training rows per merged class, drawn once; every arm sees the same rows."
function subsample(y, k; seed=1)
    rng = MersenneTwister(seed); idx = Int[]
    for c in 1:NC
        pool = findall(==(c), y)
        append!(idx, pool[randperm(rng, length(pool))[1:min(k, length(pool))]])
    end
    sort(idx)
end

function standardise(a, b; clip=3f0)
    μ = vec(mean(a,dims=1)); σ = vec(std(a,dims=1)); σ[σ .<= 0] .= 1f0
    clamp.((a .- μ')./σ', -clip, clip), clamp.((b .- μ')./σ', -clip, clip)
end

"Fixed STEP budget, so a larger k does not also buy more gradient updates."
function fit(Xtr, ysub, Xte; hidden=256, steps=STEPS, batch=32, lr=1f-3, seed=1)
    a, b = standardise(Xtr, Xte); A = permutedims(a); B = permutedims(b)
    Random.seed!(seed)
    m = Chain(Dense(size(A,1)=>hidden,relu), Dense(hidden=>NC))
    opt = Flux.setup(Flux.Adam(lr), m); Y = onehotbatch(ysub,1:NC)
    n = size(A,2); bs = min(batch,n); order = Int[]; pos = 1
    for _ in 1:steps
        if pos+bs-1 > length(order); order = randperm(n); pos = 1; end
        idx = order[pos:pos+bs-1]; pos += bs
        _,gs = Flux.withgradient(mm->Flux.logitcrossentropy(mm(A[:,idx]),Y[:,idx]), m)
        Flux.update!(opt,m,gs[1])
    end
    pr = Int[]
    for i in 1:10000:size(B,2)
        j = min(i+9999,size(B,2)); append!(pr, onecold(m(view(B,:,i:j)),1:NC))
    end
    mean(pr .== yte)
end

println("="^80)
println("PHASE 5c — does the AND layer pay when data is scarce?")
println("="^80)
res = Dict{Tuple{String,Int,Int},Float64}()
for k in KS, s in SEEDS
    idx = subsample(ytr, k; seed=s); ys = ytr[idx]
    base_tr = NEWtr[idx, OL]
    a_tr    = NEWtr[idx, AB]
    sh_tr   = copy(a_tr);        shuffle_block!(sh_tr, 1:size(sh_tr,2); seed=100s)
    sh_te   = copy(NEWte[:, AB]); shuffle_block!(sh_te, 1:size(sh_te,2); seed=200s)
    res[("orient+lp", k, s)]   = fit(base_tr, ys, NEWte[:, OL]; seed=s)
    res[("+A1+A2", k, s)]      = fit(hcat(base_tr, a_tr),  ys, hcat(NEWte[:,OL], NEWte[:,AB]); seed=s)
    res[("+shuffled", k, s)]   = fit(hcat(base_tr, sh_tr), ys, hcat(NEWte[:,OL], sh_te);        seed=s)
    @printf("  k=%-4d seed %d  base %.4f   +A %.4f   +shuf %.4f\n", k, s,
            res[("orient+lp",k,s)], res[("+A1+A2",k,s)], res[("+shuffled",k,s)])
    flush(stdout)
end

m(nm,k) = mean(res[(nm,k,s)] for s in SEEDS)
sd(nm,k) = std([res[(nm,k,s)] for s in SEEDS])
println("\n" * "="^80)
println("Test accuracy (%), mean ± sd over $(length(SEEDS)) seeds. Δ is paired per seed.")
println("="^80)
@printf("\n%-6s %8s %16s %16s %14s %14s\n",
        "k", "images", "orient+lp", "+A1+A2", "Δ from A", "Δ from shuffle")
for k in KS
    dA  = [res[("+A1+A2",k,s)]    - res[("orient+lp",k,s)] for s in SEEDS] .* 100
    dS  = [res[("+shuffled",k,s)] - res[("orient+lp",k,s)] for s in SEEDS] .* 100
    @printf("%-6d %8d %8.2f ±%4.2f %8.2f ±%4.2f %7.2f ±%4.2f %7.2f ±%4.2f\n",
            k, k*NC, 100m("orient+lp",k), 100sd("orient+lp",k),
            100m("+A1+A2",k), 100sd("+A1+A2",k),
            mean(dA), std(dA), mean(dS), std(dS))
end
println("\nfull data (112,800 imgs, 15 epochs, Phase 5a):  Δ from A = +0.01")

serialize(joinpath(dirname(CACHE), "p5c_results.jls"), res)
p = plot(xlabel="training images per merged class (log scale)", ylabel="Δ accuracy (points)",
         xscale=:log10, xticks=(KS, string.(KS)), grid=true, gridalpha=0.25,
         size=(880,430), legend=:topright, titlefontsize=10,
         left_margin=7Plots.mm, bottom_margin=5Plots.mm,
         title="Does the AND layer pay when data is scarce? ($(NC) merged classes)")
for (nm, lbl, c) in [("+A1+A2", "adding A₁+A₂", :firebrick),
                     ("+shuffled", "adding the same columns, shuffled", :grey)]
    ys = [mean([100*(res[(nm,k,s)] - res[("orient+lp",k,s)]) for s in SEEDS]) for k in KS]
    es = [std( [100*(res[(nm,k,s)] - res[("orient+lp",k,s)]) for s in SEEDS]) for k in KS]
    plot!(p, KS, ys; yerror=es, lw=2.5, marker=:circle, ms=5, c=c, label=lbl)
end
hline!(p, [0]; lc=:black, ls=:dash, lw=1, label="no effect")
hline!(p, [0.01]; lc=:firebrick, ls=:dot, lw=1.5, label="full data (+0.01)")
savefig(p, joinpath(@__DIR__, "figures", "phase5c_fewshot.png"))
println("\nwrote figures/phase5c_fewshot.png")
