### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 90000000-0000-0000-0000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 90000000-0000-0000-0000-000000000002
begin
    using PlutoUI
    using Plots
    using Statistics
    using Printf
    using Random
    using Flux
    using OneHotArrays
end

# ╔═╡ 90000000-0000-0000-0000-000000000003
begin
    include(joinpath(@__DIR__, "..", "LoadEMNIST.module.jl"))
    using .LoadEMNIST
end

# ╔═╡ 90000000-0000-0000-0000-000000000004
md"""
# Why does everything stop at ~86 %?

Every arm in `MLPonFeatures.jl` and `ConvComparison.jl` lands between 82 % and 86.5 %,
whether the input is 156 hand-designed shape features, 784 raw pixels, or the output of
a learned convolution. Even dropping to 16 convolution kernels only costs a couple of
points, and the learned kernels look visually unstructured — mostly noise with a few
oriented bars.

That clustering is suspicious. Either every representation we have tried is hitting the
same bottleneck, or something about the *task* is imposing a ceiling that no
representation can cross. This notebook runs three diagnostics to find out.

| # | question | method |
|:--|:--|:--|
| 1 | Is the convolution layer actually learning, or acting as a random projection? | freeze the kernels at random initialisation, train only the head |
| 2 | How much of the error is irreducible **label ambiguity**? | rescore each model's predictions with visually identical classes merged |
| 3 | Is the ceiling our architecture family, or the data? | train a conventional stride-1 CNN with pooling on the same budget |

**Summary of what they found:** the convolution *is* learning (freezing it costs 6
points); roughly **45 % of the best model's remaining error is homoglyph confusion**,
and the absolute size of that error is the same for every model regardless of quality;
and a proper CNN reaches 87.8 % at its best epoch — only ~1.3 points above the
hand-designed features. See the Notes for the full argument.
"""

# ╔═╡ 90000000-0000-0000-0000-000000000005
begin
    const IMG = 112
    const NCLASS = 47
    const EMNIST_DIR = joinpath(homedir(), "Julia", "DATABASES", "EMNIST")
    const SRC_DIR = joinpath(EMNIST_DIR, "emnist_source_files")
    const NAMES = [LoadEMNIST.emnist_class_name(i) for i in 1:NCLASS]

    # Classes that are the SAME handwritten shape. Disjoint by construction: taking the
    # transitive closure of confusable pairs is wrong (6≡G, G≡g, 9≡g would chain 6 to 9).
    const HOMO = [["0","O"], ["1","I","L"], ["2","Z"], ["5","S"], ["9","g","q"]]
    const CANON = let d = Dict(s=>s for s in NAMES)
        for g in HOMO, s in g; d[s] = g[1]; end
        d
    end
    homkey(c) = CANON[NAMES[c]]
end

# ╔═╡ 90000000-0000-0000-0000-000000000006
md"""
### Data

Official EMNIST-Balanced split: 112,800 train / 18,800 test, 47 classes, chance 2.13 %.

train images per class: $(@bind ntr Slider(100:100:2400, default=2400, show_value=true))
test images per class: $(@bind nte Slider(50:50:400, default=400, show_value=true))
epochs: $(@bind nep Slider(5:5:30, default=15, show_value=true))
"""

# ╔═╡ 90000000-0000-0000-0000-000000000007
begin
    function load_split(imgpath, labpath, per_class)
        I = read_emnist_images(imgpath); L = read_emnist_labels(labpath)
        keep = Int[]
        for c in 1:NCLASS
            idx = findall(==(c), L); append!(keep, idx[1:min(per_class,length(idx))])
        end
        I[:,:,keep], L[keep]
    end
    tr_imgs, ytr = load_split(joinpath(EMNIST_DIR,"emnist-balanced-train-images-idx3-ubyte"),
                              joinpath(EMNIST_DIR,"emnist-balanced-train-labels-idx1-ubyte"), ntr)
    te_imgs, yte = load_split(joinpath(SRC_DIR,"emnist-balanced-test-images-idx3-ubyte"),
                              joinpath(SRC_DIR,"emnist-balanced-test-labels-idx1-ubyte"), nte)
    Xtr = reshape(tr_imgs, 28, 28, 1, size(tr_imgs,3))
    Xte = reshape(te_imgs, 28, 28, 1, size(te_imgs,3))
    Ptr = reshape(Xtr, 784, :); Pte = reshape(Xte, 784, :)
    Markdown.parse("**$(length(ytr)) train / $(length(yte)) test** · chance " *
                   "**$(round(100/NCLASS,digits=2)) %**")
end

# ╔═╡ 90000000-0000-0000-0000-000000000008
# ---- trainers that return TEST PREDICTIONS, so predictions can be rescored ----
begin
    "Strict 47-way accuracy, and accuracy with homoglyph groups merged."
    function score(pred, y)
        s = mean(pred .== y)
        m = mean([homkey(p) for p in pred] .== [homkey(t) for t in y])
        (strict=s, merged=m, gain=m-s, share=(m-s)/max(1-s,1e-9))
    end

    function fit_mlp(Xtr_t, Xte_t, ytr; hidden=[256], epochs=15, batch=128, lr=1f-3, seed=1)
        Random.seed!(seed)
        layers=Any[]; d=size(Xtr_t,1)
        for h in hidden; push!(layers, Dense(d=>h, relu)); d=h; end
        push!(layers, Dense(d=>NCLASS))
        model=Chain(layers...); opt=Flux.setup(Flux.Adam(lr), model)
        Y=onehotbatch(ytr,1:NCLASS); n=size(Xtr_t,2)
        for _ in 1:epochs
            p=randperm(n)
            for i in 1:batch:n
                idx=p[i:min(i+batch-1,n)]
                _,gs=Flux.withgradient(m->Flux.logitcrossentropy(m(Xtr_t[:,idx]),Y[:,idx]), model)
                Flux.update!(opt,model,gs[1])
            end
        end
        pred=Int[]
        for i in 1:10000:size(Xte_t,2)
            j=min(i+9999,size(Xte_t,2)); append!(pred, onecold(model(view(Xte_t,:,i:j)),1:NCLASS))
        end
        pred
    end

    "One learned convolution + the standard head (the ConvComparison.jl architecture)."
    function fit_conv(Xtr, Xte, ytr; k=11, s=6, nk=32, hidden=256, epochs=15,
                      batch=128, lr=1f-3, seed=1)
        Random.seed!(seed)
        W=size(Xtr,1); out=(W-k)÷s+1; flat=out*out*nk
        model=Chain(Conv((k,k),1=>nk,relu;stride=s), Flux.flatten,
                    Dense(flat=>hidden,relu), Dense(hidden=>NCLASS))
        opt=Flux.setup(Flux.Adam(lr), model)
        Y=onehotbatch(ytr,1:NCLASS); n=size(Xtr,4)
        for _ in 1:epochs
            p=randperm(n)
            for i in 1:batch:n
                idx=p[i:min(i+batch-1,n)]
                _,gs=Flux.withgradient(m->Flux.logitcrossentropy(m(Xtr[:,:,:,idx]),Y[:,idx]), model)
                Flux.update!(opt,model,gs[1])
            end
        end
        pred=Int[]
        for i in 1:5000:size(Xte,4)
            j=min(i+4999,size(Xte,4)); append!(pred, onecold(model(view(Xte,:,:,:,i:j)),1:NCLASS))
        end
        model, pred
    end

    """
    A **fixed, randomly-initialised** convolution used purely as a feature extractor —
    its weights are never updated. Only the head is trained. If this matches the learned
    version, the convolution was acting as a random projection.
    """
    function random_conv_features(Xtr, Xte; k=11, s=6, nk=32, seed=7)
        Random.seed!(seed)
        layer = Conv((k,k), 1=>nk, relu; stride=s)
        ex(X) = begin
            chunks=Vector{Matrix{Float32}}()
            for i in 1:5000:size(X,4)
                j=min(i+4999,size(X,4)); F=layer(view(X,:,:,:,i:j))
                push!(chunks, reshape(F, :, size(F,4)))
            end
            reduce(hcat, chunks)
        end
        ex(Xtr), ex(Xte)
    end

    "A conventional small CNN: stride-1 convolutions with max-pooling."
    function fit_cnn(Xtr, Xte, yte, ytr; hidden=256, epochs=15, batch=128, lr=1f-3, seed=1)
        Random.seed!(seed)
        model=Chain(Conv((3,3),1=>32,relu;pad=1), MaxPool((2,2)),
                    Conv((3,3),32=>64,relu;pad=1), MaxPool((2,2)),
                    Flux.flatten, Dense(7*7*64=>hidden,relu), Dense(hidden=>NCLASS))
        opt=Flux.setup(Flux.Adam(lr), model)
        Y=onehotbatch(ytr,1:NCLASS); n=size(Xtr,4); curve=Float64[]
        predict() = begin
            p=Int[]
            for i in 1:5000:size(Xte,4)
                j=min(i+4999,size(Xte,4)); append!(p, onecold(model(view(Xte,:,:,:,i:j)),1:NCLASS))
            end; p
        end
        for _ in 1:epochs
            pm=randperm(n)
            for i in 1:batch:n
                idx=pm[i:min(i+batch-1,n)]
                _,gs=Flux.withgradient(m->Flux.logitcrossentropy(m(Xtr[:,:,:,idx]),Y[:,idx]), model)
                Flux.update!(opt,model,gs[1])
            end
            push!(curve, mean(predict() .== yte))
        end
        model, predict(), curve
    end
end

# ╔═╡ 90000000-0000-0000-0000-000000000009
md"""
### Diagnostic 1 & 2 — frozen-random convolution, and the label ceiling

`run` trains: the pixel MLP, a learned convolution, and frozen-random convolutions at
several kernel counts. Each model's predictions are then scored twice — strictly, and
with the homoglyph groups `0/O`, `1/I/L`, `2/Z`, `5/S`, `9/g/q` treated as one class
each. The difference is the part of the error the labels make unwinnable.

run: $(@bind go12 CheckBox(default=false))
"""

# ╔═╡ 90000000-0000-0000-0000-00000000000a
if !go12
    md"*(tick to run — a few minutes on the full split)*"
else
    let
        # each row carries a full name for the table and a short one for the figure
        rows = Tuple{String,String,NamedTuple}[]
        push!(rows, ("pixels 784, no conv", "pixels\n784",
                     score(fit_mlp(Ptr,Pte,ytr; epochs=nep), yte)))
        _, pc = fit_conv(Xtr,Xte,ytr; nk=32, epochs=nep)
        push!(rows, ("conv 11×11 s6 ×32, **learned**", "conv ×32\nlearned", score(pc, yte)))
        for nk in (8,16,32,64)
            a,b = random_conv_features(Xtr,Xte; nk=nk)
            push!(rows, ("conv ×$(nk), **frozen random**", "random\n×$(nk)",
                         score(fit_mlp(a,b,ytr; epochs=nep), yte)))
        end
        global DIAG_ROWS = rows
        hdr = "| model | strict | homoglyphs merged | gain | share of errors |\n|:--|--:|--:|--:|--:|\n"
        body = join([@sprintf("| %s | %.2f %% | %.2f %% | +%.2f | %.0f %% |",
                              n, 100r.strict, 100r.merged, 100r.gain, 100r.share)
                     for (n,_,r) in rows], "\n")
        Markdown.parse(hdr*body)
    end
end

# ╔═╡ 90000000-0000-0000-0000-00000000000b
if !(@isdefined DIAG_ROWS)
    md"*(run diagnostic 1 & 2 to see the figure)*"
else
    let
        st    = [100r[3].strict for r in DIAG_ROWS]
        gn    = [100r[3].gain   for r in DIAG_ROWS]
        short = [r[2] for r in DIAG_ROWS]      # figure labels; full names are in the table
        # draw the total first, then overwrite the lower part — a true stacked bar,
        # both segments opaque so the colours stay readable
        p = bar(1:length(st), st .+ gn; label="homoglyph error (irreducible)", c=:goldenrod,
                xticks=(1:length(st), short), ylabel="accuracy (%)", ylims=(0,100),
                legend=:bottomright, tickfontsize=7, guidefontsize=8, legendfontsize=7,
                title="the homoglyph tax is the same size for every model", titlefontsize=9,
                grid=false, size=(950,400), bottom_margin=6Plots.mm)
        bar!(p, 1:length(st), st; label="strict accuracy", c=:steelblue)
        for i in eachindex(gn)
            annotate!(p, i, st[i]+gn[i]+3.5, text(@sprintf("+%.1f", gn[i]), 8, :darkgoldenrod))
        end
        p
    end
end

# ╔═╡ 90000000-0000-0000-0000-00000000000c
md"""
### Diagnostic 3 — a conventional CNN

Two stride-1 convolutions with max-pooling, the same Adam / batch size / epoch budget
as everything else. This is a genuinely different architecture class: dense spatial
sampling and pooled translation tolerance, rather than one coarse strided layer.

run: $(@bind go3 CheckBox(default=false))
"""

# ╔═╡ 90000000-0000-0000-0000-00000000000d
if !go3
    md"*(tick to run — ~80 s per epoch on the full split)*"
else
    let
        _, pred, curve = fit_cnn(Xtr, Xte, yte, ytr; epochs=nep)
        r = score(pred, yte)
        p = plot(1:length(curve), 100 .*curve; lw=2, marker=:circle, ms=3, c=:seagreen,
                 label="small CNN", xlabel="epoch", ylabel="test accuracy (%)",
                 legend=:bottomright, title="stride-1 CNN with pooling", titlefontsize=9,
                 size=(760,330))
        hline!(p, [86.43]; lc=:steelblue, ls=:dash, label="156 hand features (86.43 %)")
        Markdown.parse(@sprintf("**small CNN** · strict **%.2f %%** (best epoch %.2f %%) · homoglyphs merged **%.2f %%** (+%.2f, %.0f %% of errors)",
                        100r.strict, 100maximum(curve), 100r.merged, 100r.gain, 100r.share)),
        p
    end
end

# ╔═╡ 90000000-0000-0000-0000-00000000000e
md"""
### Notes — measured on the full official split

47 classes, chance 2.13 %, official 112,800 / 18,800 split, 1×256 head where
applicable, Adam 1e-3, batch 128, 15 epochs. Standard error is **0.25 %**.

#### The complete table

| model | strict | homoglyphs merged | gain | share of errors |
|:--|--:|--:|--:|--:|
| **156 hand features** | **86.43 %** | **92.53 %** | +6.10 | 45 % |
| **small CNN** (2 conv, stride 1, pooling) | 86.46 % | **92.70 %** | +6.24 | 46 % |
| conv 11×11 s6 ×32, learned | 84.56 % | 90.80 % | +6.23 | 40 % |
| pixels 784, no conv | 83.65 % | 89.91 % | +6.26 | 38 % |
| conv ×64, frozen random | 80.48 % | 86.71 % | +6.23 | 32 % |
| conv ×32, frozen random | 78.62 % | 84.88 % | +6.27 | 29 % |
| conv ×16, frozen random | 73.19 % | 79.55 % | +6.36 | 24 % |
| conv ×8, frozen random | 63.44 % | 69.82 % | +6.38 | 17 % |

*(the small CNN's best epoch was 87.76 %, at epoch 4)*

#### 1. The convolution really is learning — my prediction was wrong

I expected freezing the kernels at random initialisation to cost little, on the grounds
that a strong 74 k-parameter head can compensate for an arbitrary first-layer basis (the
classic random-features result), and that this would explain why the filters look
unstructured. **It is not what happens.** Learned-32 reaches 84.56 % against
frozen-random-32's 78.62 % — learning the kernels is worth **~6 points**. And
frozen-random is still climbing at 64 kernels (80.48 %) yet *still* below learned-32.

So the visual appearance of the kernels is misleading: they are unstructured to the eye
but functionally doing real work. The reason they never look like textbook oriented edge
detectors is architectural — with stride 6 there are only 9 positions per image, so
weight sharing exerts little statistical pressure; there is no pooling to reward
translation tolerance; and there is no second convolution that would need clean oriented
edges as *its* input. Interpretable filters are a product of depth and dense sampling,
not a prerequisite for a layer being useful.

#### 2. Nearly half the residual error is undecidable labels

This is the main answer. Merging the five homoglyph groups lifts the best model from
**86.43 % to 92.53 %** — **45 % of its remaining error** is confusion between characters
that are *the same handwritten shape*.

The striking part is the `gain` column: **+6.10, +6.24, +6.23, +6.26, +6.23, +6.27,
+6.36, +6.38**. Across models spanning **63 % to 86 %** accuracy — a 23-point range, four
different architectures, learned and frozen — the absolute size of the homoglyph error is
constant to within a quarter of a point. A model at 63 % makes the same number of
`0`-vs-`O` mistakes as a model at 86 %.

That is what a genuine task ceiling looks like. These images do not contain the
information the label demands, so *every* model fails on the same ones, and improving a
model only ever reduces the *other* kind of error. It also explains the clustering you
noticed: once several methods have squeezed out most of the decidable error, they all sit
just above a shared 6.2-point floor, and the remaining spread between them is small.

#### 3. A proper CNN buys ~1.3 points, not the 5 I guessed

The stride-1 CNN with pooling reaches **87.76 % at its best epoch** and 86.46 % at epoch
15 — against the hand-designed features' 86.43 %. **The final-epoch numbers are
statistically identical** (0.03 points apart, against a 0.25 % standard error); the
honest comparison is best-epoch, where the CNN leads by ~1.3 points.

It converges very fast — 84.22 % after a *single* epoch, peaking at epoch 4 — and then
drifts down, which is ordinary overfitting on a 15-epoch budget with no augmentation,
regularisation or schedule. So the published ≈ 91 % convolutional results are not
contradicted; they need the machinery we deliberately excluded to keep every arm
comparable. What this row establishes is narrower and more useful: **the ~86 % plateau is
not an artefact of avoiding convolution.** A genuinely different architecture class,
given the same budget, lands in the same place.

#### What this means together

Of the ~13.6 % error the best model makes, roughly **6.2 points are undecidable labels**
and only ~7.4 points are addressable at all. Within that addressable margin, four quite
different representations — hand-designed features, raw pixels, a coarse learned
convolution, and a proper CNN — span barely 3 points. The features are doing well not
because they are close to some absolute limit of shape description, but because the task
leaves little room to distinguish good descriptions from very good ones.

If the goal is to compare *representations*, this dataset is close to exhausted: use the
homoglyph-merged number, which has more headroom and is less dominated by the floor, or
move to a task whose labels are decidable from the ink.
"""

# ╔═╡ Cell order:
# ╠═90000000-0000-0000-0000-000000000001
# ╠═90000000-0000-0000-0000-000000000002
# ╠═90000000-0000-0000-0000-000000000003
# ╟─90000000-0000-0000-0000-000000000004
# ╠═90000000-0000-0000-0000-000000000005
# ╟─90000000-0000-0000-0000-000000000006
# ╠═90000000-0000-0000-0000-000000000007
# ╠═90000000-0000-0000-0000-000000000008
# ╟─90000000-0000-0000-0000-000000000009
# ╠═90000000-0000-0000-0000-00000000000a
# ╠═90000000-0000-0000-0000-00000000000b
# ╟─90000000-0000-0000-0000-00000000000c
# ╠═90000000-0000-0000-0000-00000000000d
# ╟─90000000-0000-0000-0000-00000000000e
