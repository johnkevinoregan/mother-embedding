# ── PLAIN SCRIPT, not a Pluto notebook ──────────────────────────────────────
# `julia --project=.. -t 16 ConvNextReadout.jl`   (from the P11_ConVNextTest directory)
#
# Step 3 of 3. Score frozen ConvNeXt against our front end on the Phase 9 properties.
#
# THE QUESTION. Our front end is a hand-designed representation with **zero learned
# parameters**, and on Phase 9 it beat a properly trained CNN on four of five geometric
# properties. The obvious objection is that the CNN had only 10,000 stroke images to learn
# from. A frozen ImageNet model answers that objection directly: ConvNeXt-Base has ~88 M
# parameters fitted to 1.28 M natural images, and if a large learned general-purpose
# representation already makes curvedness, brokenness and vertex angle linearly available,
# then designing operators for them was unnecessary.
#
# WHY THIS IS THE RIGHT COMPARISON, and it is not obvious. Both arms are *fixed* functions of
# the image, so nothing is fitted to strokes except the readout, and the readout is identical:
# same ridge, same per-property λ selection, same two-hidden-layer MLP, same epoch budget,
# same R². The only thing that differs is the representation. Phase 9's CNN arm confounded
# representation with training data; this does not.
#
# All arms are scored inside a single run on **byte-identical images**, deserialised from what
# `ConvNextStimuli.jl` wrote, rather than against numbers recorded in another script's log.
#
# PREDICTIONS, on record before the first run:
#
#   1. ConvNeXt will beat raw pixels comfortably on every property. It is a strong general
#      representation and the pixel arms are the floor.
#   2. On the i.i.d. split it should be **competitive on `thickness` and `fuzziness`** — those
#      are scale and blur, which ImageNet features encode well — and **worse on `vangle` and
#      `arms`**, which are 2π ray-counting properties. Geirhos et al. (ICLR 2019) measured
#      ImageNet CNNs at 22.1 % shape bias against humans' 95.9 %, so a texture-biased
#      representation should be weakest exactly where geometry is the whole task.
#   3. **The sharpest one: it will collapse on the polarity extrapolation split.** Trained on
#      light strokes and tested on dark, our features are unchanged because quadrature energy
#      discards the sign of contrast by construction. ImageNet features certainly do not —
#      dark-on-light and light-on-dark are different inputs to a network that has learned
#      about surfaces and shading. Phase 9's trained CNN reached −2.107 on `vangle` here.
#   4. Stage 3 may beat stage 4. ImageNet's last stage is tuned to object category, and none
#      of these properties is an object category.
#
# If prediction 3 fails — if frozen ConvNeXt transfers across polarity — that is a genuine
# problem for the argument that built-in invariance is worth designing for, and it should be
# reported as such rather than explained away.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Statistics, Printf, Random, LinearAlgebra, Serialization, FFTW
using Flux, CUDA
include(joinpath(@__DIR__, "Readout.module.jl"))
include(joinpath(@__DIR__, "..", "P9_P12_SimpleStrokeTests", "Frontend.module.jl"))
using .Readout, .Frontend

const DATA   = joinpath(@__DIR__, "data")
const FEAT   = joinpath(@__DIR__, "features")
const OUT    = joinpath(@__DIR__, "results")
const N      = 112
const GRID   = parse(Int, get(ENV, "CX_GRID", "1"))       # Phase 9's winning configuration
const EPOCHS = parse(Int, get(ENV, "CX_EPOCHS", "100"))
const MODELS = split(get(ENV, "CX_MODELS", "tiny,base"), ",")
const USEGPU = get(ENV, "CX_GPU", "1") == "1"
BLAS.set_num_threads(min(16, Sys.CPU_THREADS)); FFTW.set_num_threads(1)

const PROPNAMES = [strip(l) for l in readlines(joinpath(DATA, "props.txt")) if !isempty(strip(l))]

read_manifest() = [(m[1], parse(Int, m[2]), parse(Int, m[3]))
                   for m in split.(filter(l -> !startswith(l, "#") && !isempty(strip(l)),
                                          readlines(joinpath(DATA, "manifest.txt"))))]

"Read an (n × p) Float32 matrix written row-major."
function read_matrix(path, n, p)
    a = read!(path, Vector{Float32}(undef, n * p))
    permutedims(reshape(a, p, n))
end

"""
Read back the images `ConvNextStimuli.jl` wrote — the same bytes Python reads.

On disk each image is row-major, so `reshape(·, N, N)` recovers its transpose and one more
`permutedims` restores the original. Round-tripping through the file rather than keeping a
parallel `.jls` is what makes "identical stimuli" checkable instead of assumed.
"""
function read_images(path, n; nn=N)
    a = read!(path, Vector{Float32}(undef, n * nn * nn))
    [permutedims(reshape(@view(a[(i-1)*nn*nn+1 : i*nn*nn]), nn, nn)) for i in 1:n]
end

stage_dims(m) = parse.(Int, split(strip(read(joinpath(FEAT, "$(m)_dims.txt"), String))))

"""
    pca_project(Xtr, Xte, k) -> (Ztr, Zte)

Project both splits onto the top `k` principal components of the **training** features.

Needed because the ConvNeXt stages double in width with depth — 96 / 192 / 384 / 768 for Tiny —
so scoring each at its native size confounds *which stage* with *how many columns the readout
gets*, and the confound points the same way as the effect being measured. Reducing every stage to
the width of the narrowest removes it.

Fitted on train only, so it is a legitimate part of the pipeline rather than a peek at the test
set. It is still mildly generous to ConvNeXt: these `k` directions are chosen for variance *on
this dataset*, where our own 31 features are fixed a priori.
"""
function pca_project(Xtr, Xte, k)
    μ = vec(mean(Xtr, dims=1))
    A = Xtr .- μ'
    F = svd(A)
    V = F.V[:, 1:min(k, size(F.V, 2))]
    (A * V, (Xte .- μ') * V)
end

"Concatenate the requested ConvNeXt stages into one feature matrix."
function convnext_features(model, split, kind, n; stages)
    dims = stage_dims(model)
    reduce(hcat, [read_matrix(joinpath(FEAT, "$(model)_$(split)_$(kind)_s$(s).f32"), n, dims[s])
                  for s in stages])
end

"""
Score one arm. `Xtr`/`Xte` are already-built feature matrices; the split into train and
validation, the standardisation, and the metric all follow Phase 9 exactly.
"""
function score_arm(Xtr, Ytr, Xte, Yte; drop=nothing, linear=true, tag="")
    ntr = size(Xtr, 1)
    nva = clamp(ntr ÷ 6, 40, ntr - 40); va = ntr-nva+1:ntr; tr = 1:ntr-nva
    μy, σy = zfit(Ytr[tr, :])
    Zt = zapply(Ytr[tr, :], μy, σy); Zv = zapply(Ytr[va, :], μy, σy)
    μ, σ = zfit(Xtr[tr, :])
    A = zapply(Xtr, μ, σ); T = zapply(Xte, μ, σ)
    # The MLP's per-epoch validation history is kept, not discarded. An earlier version took
    # `[1]` and threw it away — which is how the Phase 9 CNN's non-convergence went unnoticed
    # for a whole phase. Any arm whose curve is still climbing at the last epoch is reporting
    # a sample of a trajectory, not a ceiling, and that has to be visible.
    P, hist = if linear
        ridge(A[tr, :], Zt, A[va, :], Zv, T)[1], nothing
    else
        mlp(A[tr, :], Zt, A[va, :], Zv, T; epochs=EPOCHS, usegpu=USEGPU)
    end
    pred = P .* σy' .+ μy'
    R = [(drop !== nothing && PROPNAMES[j] == String(drop)) ? NaN : r2(pred[:, j], Yte[:, j])
         for j in 1:size(Yte, 2)]
    R, hist
end

function show_table(title, rows, base)
    println("\n" * "="^108); println(title); println("="^108)
    @printf("\n%-24s%7s", "arm", "nfeat")
    for p in PROPNAMES; @printf("%11s", p[1:min(10, end)]); end; println()
    @printf("%-24s%7s", "trivial baseline", "3")
    for x in base; isnan(x) ? @printf("%11s", "—") : @printf("%11.3f", x); end; println()
    println("-"^119)
    for (nm, nf, v) in rows
        @printf("%-24s%7d", nm[1:min(24, end)], nf)
        for x in v; isnan(x) ? @printf("%11s", "—") : @printf("%11.3f", x); end
        println()
    end
    flush(stdout)
end

function main()
    mkpath(OUT)
    @printf("Frozen ConvNeXt vs our front end on the Phase 9 stroke properties\n")
    @printf("grid %d for our arm (Phase 9's best), %d MLP epochs, GPU %s\n\n",
            GRID, EPOCHS, (USEGPU && CUDA.functional()) ? "yes" : "no")
    spec = build_frontend(N; grid=GRID)
    results = Dict{String,Any}()

    # `Base.split` explicitly: the loop variable below is also called `split`, and relying on
    # which one the parser picks is not worth the five characters saved.
    want = Base.split(get(ENV, "CX_SPLITS", ""), ",", keepempty=false)
    for (split, ntr, nte) in read_manifest()
        (isempty(want) || split in want) || continue
        drop = split == "iid" ? nothing : Symbol(split)
        np = length(PROPNAMES)
        itr = read_images(joinpath(DATA, "$(split)_train_img.f32"), ntr)
        ite = read_images(joinpath(DATA, "$(split)_test_img.f32"),  nte)
        Ytr = read_matrix(joinpath(DATA, "$(split)_train_y.f32"), ntr, np)
        Yte = read_matrix(joinpath(DATA, "$(split)_test_y.f32"),  nte, np)

        base = trivial_baseline(vcat(itr, ite), Ytr, Yte, ntr)
        rows = Tuple{String,Int,Vector{Float64}}[]
        hists = Dict{String,Any}()
        # one helper so every arm records its curve the same way
        function add!(nm, nf, X1, X2; kw...)
            R, h = score_arm(X1, Ytr, X2, Yte; kw...)
            push!(rows, (nm, nf, R)); h === nothing || (hists[nm] = h)
            R
        end

        # pixels — the floor
        flat(v) = permutedims(reduce(hcat, [vec(x) for x in v]))
        Xp_tr, Xp_te = flat(itr), flat(ite)
        add!("pixels·linear", size(Xp_tr, 2), Xp_tr, Xp_te; drop=drop, linear=true)
        Xp_tr = nothing; Xp_te = nothing; GC.gc()

        # ours. Cached because extraction is ~21 ms/image and dominates the whole script —
        # ~7 min per split against seconds for every fit, exactly as in Phase 10.
        mkpath(joinpath(@__DIR__, "cache"))
        ck = joinpath(@__DIR__, "cache", "ours_g$(GRID)_$(split)_$(ntr)_$(nte).jls")
        Ftr, Fte = if isfile(ck)
            deserialize(ck)
        else
            a = featurize(itr, spec); b = featurize(ite, spec)
            serialize(ck, (a, b)); (a, b)
        end
        add!("ours·linear", spec.n, Ftr, Fte; drop=drop, linear=true)
        add!("ours·MLP",    spec.n, Ftr, Fte; drop=drop, linear=false)
        Ftr = nothing; Fte = nothing; GC.gc()

        # convnext, per model: stage 4 alone (the standard frozen feature), stage 3, and all four
        for m in MODELS
            isfile(joinpath(FEAT, "$(m)_dims.txt")) || continue
            probe = get(ENV, "CX_STAGEPROBE", "0") == "1"
            stagesets = probe ?
                [("s1", [1]), ("s2", [2]), ("s3", [3]), ("s4", [4]),
                 ("s1-2", [1,2]), ("s1-3", [1,2,3]), ("s1-4", [1,2,3,4])] :
                [("s4", [4]), ("s3", [3]), ("s1-4", [1,2,3,4])]
            for (label, stages) in stagesets
                Ctr = convnext_features(m, split, "train", ntr; stages=stages)
                Cte = convnext_features(m, split, "test",  nte; stages=stages)
                add!("convnext_$m $label ·lin", size(Ctr, 2), Ctr, Cte; drop=drop, linear=true)
                if stages == [4] || probe
                    add!("convnext_$m $label ·MLP", size(Ctr, 2), Ctr, Cte; drop=drop, linear=false)
                end
                # dimension-matched: every stage reduced to the width of the narrowest, so the
                # comparison is between stages rather than between column counts.
                if probe && length(stages) == 1
                    kdim = parse(Int, get(ENV, "CX_PCA", "96"))
                    Ptr, Pte = pca_project(Float64.(Ctr), Float64.(Cte), kdim)
                    add!("convnext_$m $label pca$kdim ·lin", size(Ptr, 2),
                         Float32.(Ptr), Float32.(Pte); drop=drop, linear=true)
                    add!("convnext_$m $label pca$kdim ·MLP", size(Ptr, 2),
                         Float32.(Ptr), Float32.(Pte); drop=drop, linear=false)
                end
                Ctr = nothing; Cte = nothing; GC.gc()
            end
        end

        title = split == "iid" ? "i.i.d. split — test R² per property" :
                "extrapolation: $split — trained on one range, tested on the other"
        show_table(title, rows, base)
        results[split] = (rows=rows, base=base, props=PROPNAMES, hists=hists)
        serialize(joinpath(OUT, "$(split).jls"), results[split])
    end
    # MERGE, never overwrite. A partial run (`CX_SPLITS=iid`) previously rewrote this file with
    # only the split it had run, destroying the extrapolation results it did not touch — the
    # per-split files survived and `Deltas.jl` now reads those instead, but there is no reason
    # for this file to be destructive either.
    prev = isfile(joinpath(OUT, "all.jls")) ? deserialize(joinpath(OUT, "all.jls")) : Dict{String,Any}()
    serialize(joinpath(OUT, "all.jls"), merge(prev, results))
    println("\nwrote $OUT")
end

main()
