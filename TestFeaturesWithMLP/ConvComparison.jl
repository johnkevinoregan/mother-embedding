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
    include(joinpath(@__DIR__, "..", "LoadEMNIST.jl"))
    using .LoadEMNIST
end

# ╔═╡ 90000000-0000-0000-0000-000000000004
md"""
# A convolutional arm, for comparison

`MLPonFeatures.jl` compared three ways of coding 156 hand-designed shape features
against a **raw-pixel** baseline, all through the same fully-connected net with **no
convolution anywhere**. This notebook adds the missing row: **one convolution layer**
on the raw pixels, everything else held identical.

That notebook is left untouched; this is a separate, self-contained addition.

### The architecture

```
28×28 image  →  Conv 11×11, stride 6, 32 kernels, ReLU  →  3×3×32 = 288  →  Dense 256 ReLU  →  47
```

A single convolution: 32 learnable 11×11 kernels, slid with stride 6 so consecutive
placements overlap by roughly half. On a 28×28 input that yields a **3×3 grid of
positions**, giving 288 numbers, which then go through **the same 1×256 head, the same
Adam optimiser, the same batch size and the same 15 epochs** as every other arm.

### Why this is the interesting comparison

The 3×3 output grid is a **learned counterpart of the hand-designed "tic-tac-toe"
grid** used for the `F` features: that one also divides the image into 3×3 cells and
describes each with 9 numbers derived from a windowed Fourier transform. Here each cell
gets **32 learned kernels** instead. So the row answers a sharp question — given the
same spatial layout and the same classifier head, how much better is *learning* the
per-cell descriptors than *designing* them?

### A note on resolution

The hand-designed `Z` and `F` features are computed on a **112×112 bilinear upsample**
of the character, because they place discs and analysis windows at sub-pixel positions
and a 28×28 grid is too coarse for that to be stable. The **pixel and convolution arms
use the raw 28×28**, because upsampling is bilinear interpolation and adds **no
information** — feeding 12,544 interpolated pixels would supply identical content with
16× the first-layer parameters, making for a worse baseline, not a fairer one.

The kernel geometry was chosen for 28×28 specifically: an 11×11 kernel covers ~39 % of
the image width, so with stride 6 the three placements per axis tile the character much
as the 3×3 hand-designed grid does. At 112×112 the same 11×11 kernel would span under
10 % of the width — less than one 13 px stroke — and would be a fine-scale edge
detector rather than a cell descriptor.
"""

# ╔═╡ 90000000-0000-0000-0000-000000000005
begin
    const NCLASS = 47
    const EMNIST_DIR = joinpath(homedir(), "Julia", "DATABASES", "EMNIST")
    # official test split ships as idx inside emnist_source_files/ — same parser, no
    # CSV-orientation risk
    const SRC_DIR = joinpath(EMNIST_DIR, "emnist_source_files")
end

# ╔═╡ 90000000-0000-0000-0000-000000000006
md"""
### Data

Official EMNIST-Balanced split: **112,800 train / 18,800 test**, 2400 and 400 per class,
47 classes, chance **2.13 %**. Sliders subsample for quick iteration — use the full
split for anything you intend to quote.

train images per class: $(@bind ntr Slider(100:100:2400, default=2400, show_value=true))
test images per class: $(@bind nte Slider(50:50:400, default=400, show_value=true))
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
    # Flux wants WHCN
    Xtr = reshape(tr_imgs, 28, 28, 1, size(tr_imgs,3))
    Xte = reshape(te_imgs, 28, 28, 1, size(te_imgs,3))
    Markdown.parse("**$(length(ytr)) train / $(length(yte)) test** · " *
                   "tensors `$(size(Xtr))` and `$(size(Xte))` · chance " *
                   "**$(round(100/NCLASS,digits=2)) %**")
end

# ╔═╡ 90000000-0000-0000-0000-000000000008
# ---- one conv layer + the standard head ----
begin
    """
    `k` kernel size, `s` stride, `nk` number of learnable kernels, `pad` padding.
    Everything after the convolution is identical to the other arms: flatten, one
    hidden layer of `hidden` ReLU units, linear output to 47, Adam, softmax
    cross-entropy, minibatches, official test split scored every epoch.
    """
    function train_conv(Xtr, ytr, Xte, yte; k=11, s=6, nk=32, pad=0, hidden=256,
                        epochs=15, batch=128, lr=1f-3, seed=1, verbose=false)
        Random.seed!(seed)
        W = size(Xtr,1); out = (W + 2pad - k) ÷ s + 1; flat = out*out*nk
        model = Chain(Conv((k,k), 1=>nk, relu; stride=s, pad=pad),
                      Flux.flatten, Dense(flat=>hidden, relu), Dense(hidden=>NCLASS))
        opt = Flux.setup(Flux.Adam(lr), model)
        Ytr = onehotbatch(ytr,1:NCLASS); n = size(Xtr,4)
        hist=(loss=Float64[], test=Float64[])
        function acc(X,y)
            c=0
            for i in 1:5000:size(X,4)
                j=min(i+4999,size(X,4))
                c += sum(onecold(model(view(X,:,:,:,i:j)),1:NCLASS) .== view(y,i:j))
            end
            c/length(y)
        end
        for ep in 1:epochs
            perm=randperm(n); tot=0.0; nb=0
            for i in 1:batch:n
                idx=perm[i:min(i+batch-1,n)]
                l,gs=Flux.withgradient(m->Flux.logitcrossentropy(m(Xtr[:,:,:,idx]),Ytr[:,idx]), model)
                Flux.update!(opt,model,gs[1]); tot+=l; nb+=1
            end
            push!(hist.loss,tot/nb); push!(hist.test,acc(Xte,yte))
            verbose && @printf("  ep %2d  loss %.4f  test %.4f\n", ep, hist.loss[end], hist.test[end])
        end
        model, hist, out, flat, sum(length, Flux.trainables(model))
    end

    "Same head with no convolution at all, for the like-for-like pixel row."
    function train_mlp(Xtr_t, ytr, Xte_t, yte; hidden=[256], epochs=15, batch=128,
                       lr=1f-3, seed=1)
        Random.seed!(seed)
        layers=Any[]; d_in=size(Xtr_t,1)
        for h in hidden; push!(layers, Dense(d_in=>h, relu)); d_in=h; end
        push!(layers, Dense(d_in=>NCLASS))
        model=Chain(layers...); opt=Flux.setup(Flux.Adam(lr), model)
        Ytr=onehotbatch(ytr,1:NCLASS); n=size(Xtr_t,2); hist=(loss=Float64[], test=Float64[])
        function acc(X,y)
            c=0
            for i in 1:10000:size(X,2)
                j=min(i+9999,size(X,2))
                c += sum(onecold(model(view(X,:,i:j)),1:NCLASS) .== view(y,i:j))
            end
            c/length(y)
        end
        for ep in 1:epochs
            perm=randperm(n); tot=0.0; nb=0
            for i in 1:batch:n
                idx=perm[i:min(i+batch-1,n)]
                l,gs=Flux.withgradient(m->Flux.logitcrossentropy(m(Xtr_t[:,idx]),Ytr[:,idx]), model)
                Flux.update!(opt,model,gs[1]); tot+=l; nb+=1
            end
            # NB: Xte_t, the function's own 784×N argument — not the global 4-D `Xte`
            push!(hist.loss,tot/nb); push!(hist.test,acc(Xte_t,yte))
        end
        model, hist
    end
end

# ╔═╡ 90000000-0000-0000-0000-000000000009
md"""
### Controls

kernel size: $(@bind ksz Select([5,7,9,11,13], default=11))
stride: $(@bind kst Select([2,3,4,6,8], default=6))
kernels: $(@bind nker Select([8,16,32,64], default=32))
padding: $(@bind kpad Select([0,1,2], default=0))
epochs: $(@bind nep Slider(5:5:30, default=15, show_value=true))

run: $(@bind go CheckBox(default=false))
"""

# ╔═╡ 90000000-0000-0000-0000-00000000000a
if !go
    md"*(tick **run** — a full-split conv run is ~2–4 minutes)*"
else
    let
        t0=time()
        model, h, out, flat, np = train_conv(Xtr, ytr, Xte, yte;
                                             k=ksz, s=kst, nk=nker, pad=kpad, epochs=nep)
        global CONV_HIST = h; global CONV_MODEL = model
        Markdown.parse(@sprintf("**%d×%d kernels, stride %d, %d of them, pad %d** → %d×%d×%d = **%d conv features**, %d params · **test %.2f %%** (best %.2f %%) · %.0f s",
                        ksz,ksz,kst,nker,kpad,out,out,nker,flat,np,
                        100h.test[end], 100maximum(h.test), time()-t0)),
        plot(1:nep, 100 .*h.test; lw=2, marker=:circle, ms=3, label="conv, test",
             xlabel="epoch", ylabel="accuracy (%)", legend=:bottomright,
             title="one conv layer on 28×28 pixels", titlefontsize=9, size=(760,320))
    end
end

# ╔═╡ 90000000-0000-0000-0000-00000000000b
md"""
### The learned kernels

What the 32 filters converged to. Compare with the hand-designed alternative, whose
per-cell descriptors are fixed low-order Fourier components chosen in advance.
"""

# ╔═╡ 90000000-0000-0000-0000-00000000000c
if !(@isdefined CONV_MODEL)
    md"*(run the conv arm first)*"
else
    let
        W = CONV_MODEL[1].weight            # k × k × 1 × nk
        nk = size(W,4)
        kw=(aspect_ratio=:equal, axis=false, ticks=false, cbar=false, yflip=true)
        div = cgrad(:RdBu, rev=true)
        panels = [begin
            f = W[:,:,1,i]; m = max(maximum(abs,f), 1e-9)
            heatmap(f; c=div, clims=(-m,m), title="", kw...)
        end for i in 1:nk]
        nc = 8; nr = ceil(Int, nk/nc)
        plot(panels...; layout=(nr,nc), size=(110nc, 110nr))
    end
end

# ╔═╡ 90000000-0000-0000-0000-00000000000d
md"""
### Reference row: the same head with no convolution

For a like-for-like comparison in this same session, the raw 784 pixels through the
identical 1×256 head. If the conv arm above has already been run at the same epoch
count, its curve is overlaid so the two can be read against each other directly.

run: $(@bind go_ref CheckBox(default=false))
"""

# ╔═╡ 90000000-0000-0000-0000-00000000000e
if !go_ref
    md"*(tick to run the no-convolution pixel reference)*"
else
    let
        P_tr = reshape(Xtr, 784, :); P_te = reshape(Xte, 784, :)
        t0=time(); _, h = train_mlp(P_tr, ytr, P_te, yte; hidden=[256], epochs=nep)
        global REF_HIST = h
        p = plot(1:nep, 100 .*h.test; lw=2, marker=:square, ms=3, c=:goldenrod,
                 label="pixels 784, no conv", xlabel="epoch", ylabel="test accuracy (%)",
                 legend=:bottomright, title="convolution vs none, identical head",
                 titlefontsize=9, size=(760,340))
        # overlay the conv curve when it has been run at the same epoch count
        if (@isdefined CONV_HIST) && length(CONV_HIST.test) == nep
            plot!(p, 1:nep, 100 .*CONV_HIST.test; lw=2, marker=:circle, ms=3,
                  c=:steelblue, label="conv $(ksz)×$(ksz), stride $(kst), $(nker) kernels")
        end
        Markdown.parse(@sprintf("**pixels 784, no convolution, 1×256** · test **%.2f %%** (best %.2f %%) · %.0f s%s",
                        100h.test[end], 100maximum(h.test), time()-t0,
                        (@isdefined CONV_HIST) && length(CONV_HIST.test)==nep ?
                          @sprintf(" — conv arm reached **%.2f %%**, a gap of **%+.2f** points",
                                   100CONV_HIST.test[end], 100*(CONV_HIST.test[end]-h.test[end])) :
                          " — run the conv arm above to overlay it")),
        p
    end
end

# ╔═╡ 90000000-0000-0000-0000-00000000000f
md"""
### Notes — measured on the full official split

All three rows below were run **in the same session**, with the identical 1×256 head,
Adam at 1e-3, batch 128, 15 epochs, official 112,800/18,800 split, 47 classes, chance
2.13 %. Standard error is **0.25 %**, so gaps under ~0.5 points are not meaningful.

| arm | features into the head | total params | final | best |
|:--|--:|--:|--:|--:|
| **conv 11×11, stride 6, 32 kernels** | 288 | 89,967 | **84.56 %** | 84.85 % |
| pixels 784, no convolution | 784 | 213,551 | 83.65 % | 83.95 % |
| **156 hand-designed features** | 156 | 52,271 | **86.43 %** | 86.71 % |

**1. Convolution helps over raw pixels, but only a little.** 84.56 % vs 83.65 % — about
0.9 points, roughly 3.6 standard errors, so real but modest. One convolution layer with
a stride-6 3×3 output is a weak convolutional model: no pooling, no second stage, no
translation robustness beyond what a single stride buys. Full convolutional networks
reach ≈ 91 % on this dataset, and nothing here contradicts that; this arm is deliberately
matched to the others, not tuned for accuracy.

**2. The hand-designed features still win, by 1.9 points** (86.43 % vs 84.56 %) — about
7.5 standard errors, and they do it with **156 numbers against 288**, and **52 k
parameters against 90 k**. So on this comparison, *learning* 32 kernels per cell of a
3×3 grid does **not** recover what the designed per-cell descriptors provide.

That is the result worth taking seriously, because the comparison is close to
apples-to-apples: both descriptions carve the image into a 3×3 spatial layout and hand
the classifier a per-cell summary. The difference is only in where those summaries come
from. What the designed features have that the learned kernels lack is **scale and
invariance structure that a single 11×11 kernel cannot express** — the Zernike block
describes the character as a centred whole with explicit rotation behaviour, and the
Fourier block's per-cell orientation tensor is built to be translation-invariant *inside*
each cell. A single convolution layer has to discover any such structure from data, with
only 3×3 = 9 spatial positions and no depth to build it in.

**3. Read this as a floor for convolution, not a ceiling.** A second conv layer,
pooling, or simply more kernels would very likely close and then reverse the gap — that
is what the published ≈ 91 % convolutional results are. The claim here is narrow and
specific: *at matched depth, matched head, and matched training budget*, the designed
features beat a single learned convolution layer. It says the features are carrying real
structure, not that convolution is inferior.

**4. Parameter efficiency runs the same way.** The features arm uses the fewest
parameters (52 k) and gets the best accuracy; the pixel arm uses the most (214 k, almost
all in the 784→256 first layer) and does worst. The conv arm sits between on both counts.

### Caveats

- One seed per configuration; differences under ~0.5 points should not be read.
- The conv arm sees the **28×28** image while the hand-designed features are computed on
  a **112×112 bilinear upsample**. That upsample adds no information, so the two arms
  have access to the same content — but the feature extractors get sub-pixel geometry
  that the 28×28 conv grid cannot represent. This is a genuine asymmetry in *how* the
  same information can be addressed, and part of why the designed features do well.
- Kernel geometry was chosen to match the 3×3 hand-designed grid at 28×28. The sliders
  above let you vary it; the defaults are not a tuned optimum.
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
# ╟─90000000-0000-0000-0000-00000000000b
# ╠═90000000-0000-0000-0000-00000000000c
# ╟─90000000-0000-0000-0000-00000000000d
# ╠═90000000-0000-0000-0000-00000000000e
# ╟─90000000-0000-0000-0000-00000000000f
