# ── PLAIN SCRIPT, not a Pluto notebook ──────────────────────────────────────
# Run with `julia --project=. -t 8 <this file>`.
#
# Phase 8 — the missing positive control: a task where i2D structure is decisive.
#
# Everything measured so far establishes that the conjunction layer adds nothing ON EMNIST,
# and Phase 7 explains why (93 % linearly recoverable from the orientation statistics there).
# What has NOT been established is that it helps anything, anywhere. Phase 3 showed the
# operators work, but that is a measurement of the operator, not of a classifier.
#
# So: a classification task built so that co-location is the ONLY discriminating cue.
#
# THE DESIGN. Every stimulus is the same two bars — a "bar" and a "stem" perpendicular to
# it — and the class is decided purely by HOW THEY MEET:
#
#     :touch   the stem's end just reaches the bar   (a T-junction, 3 rays)
#     :gap     the stem's end stops a hair short      (no junction, 2 disjoint strokes)
#
# ONLY TWO CLASSES, and they are the minimum contrast that can be made airtight. A first
# attempt used four (:cross / :tee / :corner / :separate) and SCORED 100 % ON THE ORIENT
# CONTROL, i.e. it was void. The leak: matching the *global* orientation histogram is not
# enough, because `orient` is a 3×3 grid of PER-CELL statistics and therefore reads spatial
# layout — a corner puts ink at the bar's end, a separated stem sits further out, and those
# are visible without any co-location machinery.
#
# With :touch versus :gap the bar and the stem have identical lengths, widths, orientations
# and positions drawn from the same distributions; the stem's far end is in the same place;
# and the ONLY difference is whether its near end reaches the bar or stops a few pixels
# short. Nothing about the layout distinguishes them.
#
# That matching is the whole point, and it is the third time in this project that failing to
# control it produced a spurious result — total ink faked a "ray count" ordering (RESULTS.md
# §5.1), and 3-vs-4 equal rays faked a c0-versus-A1 comparison (Validate_RayHarmonics.jl).
# Here it is designed in, and the orient arm's score is the check: if the design is sound,
# orient should sit near chance.
#
# PREDICTION, on record before running: orient near 25 % chance; A1+A2 and ray harmonics
# well above. If orient scores high, the design leaks and the benchmark is void.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Statistics, Printf, Random, LinearAlgebra, Serialization, FFTW
using Flux, OneHotArrays, Plots
include(joinpath(@__DIR__, "GaborStack.module.jl"))
include(joinpath(@__DIR__, "AndLayer.module.jl"))
include(joinpath(@__DIR__, "RayHarmonics.module.jl"))
include(joinpath(@__DIR__, "Stimuli.module.jl"))
include(joinpath(@__DIR__, "Pooling.module.jl"))
using .GaborStack, .AndLayer, .RayHarmonics, .Stimuli, .Pooling

const N     = 112
const NTR   = parse(Int, get(ENV, "P8_NTR", "1500"))     # per class
const NTE   = parse(Int, get(ENV, "P8_NTE", "400"))
const CLS   = (:touch, :gap)
const NC    = length(CLS)
BLAS.set_num_threads(2); FFTW.set_num_threads(1)

"""
One stimulus. `bar` of length L1 and `stem` of length L2, perpendicular, with the class
deciding only where the stem sits relative to the bar. All nuisance parameters are drawn
from class-independent distributions, so the marginal orientation energy is matched.
"""
function junction(cls::Symbol, rng)
    θ  = rand(rng) * π                       # global orientation, uniform
    w  = 9.0 + 6.0 * rand(rng)               # stroke width  9–15 px
    L1 = 55.0 + 20.0 * rand(rng)             # bar length
    L2 = 30.0 + 15.0 * rand(rng)             # stem length
    jy = (rand(rng) - 0.5) * 14; jx = (rand(rng) - 0.5) * 14   # position jitter
    c  = (N + 1) / 2
    u  = (cos(θ), sin(θ)); v = (-sin(θ), cos(θ))               # along bar, along stem
    bx, by = c + jx, c + jy

    # :touch — the stem's near end reaches the bar. :gap — it stops GAP px short, and the
    # stem is SHORTENED rather than translated, so its far end stays put. The only change is
    # the few pixels of ink adjacent to the junction.
    # GAP is measured from the bar's SURFACE, not its centre. Measuring from the centre
    # meant that any gap below w/2 (the bar is 9-15 px wide) still left the stem overlapping
    # the bar, so "gap = 2" and "gap = 4" produced bit-identical stimuli and bit-identical
    # scores — an artefact, not a resolution limit.
    g  = cls === :touch ? 0.0 : GAP
    L2e = L2 - g
    av  = L2e/2 + w/2 + g
    sy = by + av*v[2]; sx = bx + av*v[1]
    max.(barstim(N, θ;      w=w, len=L1,  cy=by, cx=bx),
         barstim(N, θ+π/2;  w=w, len=L2e, cy=sy, cx=sx))
end

const LADDER=[2.0,3.742,7.0]; const BETAS=[2.0,1.6,1.2]; const NORI=[8,12,16]
const HF,WF,_ = field_for((N,N),LADDER;n_orient=NORI,beta=BETAS)
const BANK = make_bank((HF,WF),LADDER;imwidth=N,n_orient=NORI,beta=BETAS)
const WTS  = grid_weights(N,N,3)

function feats(img)
    Es = energy_stack(img, BANK)
    A, al = and_maps(Es, BANK.meta; forms=(:A1,:A2))
    Rm, rl = ray_maps(Es, BANK.meta)
    f1, l1 = assemble(Es, BANK.meta, A, al,
                      PoolSpec(grid=3, blocks=(:orient,:lowpass,:A1,:A2)); Wts=WTS)
    # ray_maps now returns unnormalised moments; ratios are formed after pooling
    PR = pool_maps(Rm, WTS); fr = Float32[]; lr = String[]
    for ρ in unique(l.rho0 for l in rl)
        k0 = findfirst(l -> l.rho0 == ρ && l.form === :R0, rl)
        k1 = findfirst(l -> l.rho0 == ρ && l.form === :R1, rl)
        k2 = findfirst(l -> l.rho0 == ρ && l.form === :R2, rl)
        fl = 1f-3 * max(mean(@view PR[:, k0]), 1f-12)
        for c in 1:9
            push!(fr, PR[c,k0]);                     push!(lr, "R0.ρ$(round(ρ,digits=2)).cell$(c)")
            push!(fr, PR[c,k1]/(PR[c,k0]+fl));       push!(lr, "R1.ρ$(round(ρ,digits=2)).cell$(c)")
            push!(fr, PR[c,k2]/(PR[c,k0]+fl));       push!(lr, "R2.ρ$(round(ρ,digits=2)).cell$(c)")
        end
    end
    vcat(f1, fr), vcat(l1, lr)
end

function build(n_per, seed)
    rng = MersenneTwister(seed)
    imgs = Matrix{Float32}[]; y = Int[]
    for (ci,cl) in enumerate(CLS), _ in 1:n_per
        push!(imgs, junction(cl, rng)); push!(y, ci)
    end
    p = randperm(MersenneTwister(seed+1), length(imgs))
    imgs = imgs[p]; y = y[p]
    f1, lab = feats(imgs[1])
    F = zeros(Float32, length(imgs), length(f1))
    Threads.@threads for i in eachindex(imgs); F[i,:] = feats(imgs[i])[1]; end
    F, y, lab, imgs
end

println("Phase 8 — at what gap can each representation still tell touching from not?\n")
GAP = 0.0   # set per sweep step below

stdz(a,b)=(μ=vec(mean(a,dims=1));σ=vec(std(a,dims=1));σ[σ.<=0].=1f0;
           (clamp.((a.-μ')./σ',-3,3), clamp.((b.-μ')./σ',-3,3)))
function acc(FTR,YTR,FTE,YTE,cs; seed=1, hidden=128, epochs=40)
    a,b = stdz(FTR[:,cs],FTE[:,cs]); A=permutedims(a); B=permutedims(b); Random.seed!(seed)
    m=Chain(Dense(length(cs)=>hidden,relu),Dense(hidden=>NC))
    opt=Flux.setup(Flux.Adam(1f-3),m); Y=onehotbatch(YTR,1:NC); n=size(A,2); best=0.0
    for _ in 1:epochs
        p=randperm(n)
        for i in 1:64:n
            idx=p[i:min(i+63,n)]
            _,gs=Flux.withgradient(mm->Flux.logitcrossentropy(mm(A[:,idx]),Y[:,idx]),m)
            Flux.update!(opt,m,gs[1])
        end
        best=max(best, mean(onecold(m(B),1:NC).==YTE))
    end
    best
end

println("="^76)
println("Gap sweep — gap measured from the bar's SURFACE; stroke width is 9–15 px")
println("="^76)
@printf("\n%-9s %12s %12s %12s %12s\n","gap (px)","orient+lp","A₁+A₂","rays","orient+A+rays")
rows = Tuple{Float64,Float64,Float64,Float64,Float64}[]
for gp in (16.0, 12.0, 8.0, 5.0, 3.0, 1.5)
    global GAP = gp
    FTR,YTR,LAB,_ = build(NTR, 11); FTE,YTE,_,_ = build(NTE, 99)
    cols(p...) = findall(x->any(startswith(x,q) for q in p), LAB)
    OL=cols("orient","lowpass"); AB=cols("A1","A2"); RY=cols("R0","R1","R2")
    a(cs) = mean(acc(FTR,YTR,FTE,YTE,cs;seed=s) for s in 1:3)
    r = (gp, a(OL), a(AB), a(RY), a(vcat(OL,AB,RY)))
    push!(rows, r)
    @printf("%-9.0f %11.2f %% %11.2f %% %11.2f %% %11.2f %%\n",
            r[1], 100r[2], 100r[3], 100r[4], 100r[5])
    flush(stdout)
end
@printf("%-9s %11.2f %% %11.2f %% %11.2f %% %11.2f %%\n","chance",50.0,50.0,50.0,50.0)
println("""

Read the columns downward. Whichever representation stays above chance at the smallest gap
is the one that resolves co-location most finely. If they degrade together, co-location is
not a separable dimension at this pooling resolution — it is entangled with local ink
density, which any per-cell energy statistic already reports.""")
p = plot(xlabel="gap between stem and bar (px)", ylabel="accuracy (%)", xflip=true,
         legend=:bottomleft, size=(820,400), grid=true, gridalpha=0.25, titlefontsize=10,
         left_margin=6Plots.mm, bottom_margin=5Plots.mm,
         title="how small a gap can each representation still detect?")
for (k,(nm,c)) in enumerate([("orient+lowpass",:steelblue),("A₁+A₂",:firebrick),
                             ("ray harmonics",:seagreen),("everything",:purple)])
    plot!(p, [r[1] for r in rows], [100r[k+1] for r in rows]; lw=2.5, marker=:circle, ms=5,
          c=c, label=nm)
end
hline!(p, [50]; ls=:dash, lc=:black, lw=1, label="chance")
savefig(p, joinpath(@__DIR__, "figures", "phase8_gapsweep.png"))
println("\nwrote figures/phase8_gapsweep.png")
