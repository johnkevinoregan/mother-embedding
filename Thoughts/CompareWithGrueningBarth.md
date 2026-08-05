# Same theorem, opposite direction

*On Grüning & Barth, "Bio-inspired Min-Nets Improve the Performance and Robustness of Deep
Networks", SVRHM 2021 workshop @ NeurIPS ([arXiv:2201.02149](https://arxiv.org/abs/2201.02149)).
Local copy: `GrueningBarth.pdf` in the repo root, excluded from the repo via `.git/info/exclude`.*

*Nothing here is a new experiment. It is a comparison, plus two things worth stealing.*

---

## Why this paper is the closest relative we have found

They cite **Zetzsche & Barth (1990)** for exactly the reason we do: no linear filter can be
i2D-selective, so detecting corners, junctions and line ends requires an **AND of two filters**.
Barth is a co-author of that paper. We share the starting theorem with the people who proved it.

Then we go opposite directions. They let a CNN **learn** the pairs. We **fix** them by
construction and learn nothing in the front end.

## What they built

A "Min-block", substituted for the first block of each stack in a ResNet or DenseNet — three
blocks in a hundred-layer network. Inside it, two depthwise-separable 3×3 filters `v`, `g` are
learned per feature map, instance-normalised, ReLU'd, and combined by a minimum:

```math
T_2[i,j,m] \;=\; \min\!\left(\frac{\mathrm{ReLU}(v_m^{\mathsf T}x_m - \mu_{v_m})}{\sigma_{v_m}},\;
                             \frac{\mathrm{ReLU}(g_m^{\mathsf T}x_m - \mu_{g_m})}{\sigma_{g_m}}\right)
```

The unit has significant output only if **both** filters are active. That is the AND. Their
stated justification is *hyperselectivity* (see below) plus natural-image statistics: i0D patches
are redundant and frequent, i1D structure is also redundant and more frequent than i2D, so a
representation built on i2D events has lower entropy.

## The differences

### 1. Learned pairs vs. a fixed 90° pairing

Their `v, g` are free parameters — thousands of pairs, discovered by SGD. Our `A₁`
(`P0-8_RationalGaborFeatures/AndLayer.module.jl`) pairs orientation channel *k* with channel
*k + n/2*, i.e. **exactly orthogonal, by construction**:

```math
A_1(x) \;=\; \frac{1}{C_0(x)}\sum_k E_k(x)\,E_{k+n/2}(x)
```

Zero learned parameters, one map per scale.

**They never check whether the learned pairs come out orthogonal, or oriented, or anything else.**
This is the verification their argument needs and does not have. The entire biological motivation
rests on the learned pair being an end-stopped conjunction of oriented filters; the paper measures
only downstream accuracy. Everything they report is equally consistent with "an extra
multiplicative nonlinearity helps optimisation", which is a much weaker claim than the title.

### 2. min vs. product — and they say it does not matter

The most directly useful sentence in the paper is in the Discussion. Prior work from the same
group got similar results with **explicit multiplication** (FP-nets, Grüning et al. 2021) and with
**log-space convolution** (log-nets, Grüning et al. 2020), so:

> the key to the improvements demonstrated here is mainly the AND combination of filter pairs and
> less the way in which the AND is implemented.

We use the product. Their ablation-by-publication-history says that choice is not the interesting
axis. The interesting axis is fixed-vs-learned pairing, which nobody has tested.

### 3. Phase and polarity — the real divide

Their min sits on **signed** ReLU'd linear responses. A Min-unit is therefore phase-sensitive and
polarity-sensitive: invert the contrast and it stops firing. Ours sits on quadrature energy, phase
discarded in the first line of `energy_stack`, polarity-invariant by construction — which for this
project is a **requirement**, not a detail. Nothing in a Min-Net would survive contrast inversion
without retraining.

But note where we agree, because it matters for [[ExploitingFourierHarmonics]]. Both operators get
their co-location signal from being **pointwise before pooling**, not from phase. Their min is at a
pixel of a feature map; our A₁ is at a pixel of the energy stack. The `AndLayer` docstring makes
exactly this argument — multiply-then-pool minus pool-then-multiply is the within-window spatial
covariance, and that covariance *is* the co-location signal. So the phase dead end documented in
the other note is specific to the **pooled summary statistics**; the dense-operator route is the
one both projects independently took.

### 4. Where the AND sits

Theirs applies to *learned* feature maps three levels deep, so it can build ANDs of ANDs. Ours
applies once, to the image, and everything downstream is a linear or MLP readout. Theirs is
strictly more expressive. Ours is the one you can inspect, ablate, and hold fixed across datasets.

### 5. Normalisation scope

They instance-normalise each filter's **whole feature map** before taking the min, so the min
compares relative activations under a global gain. We divide by `C₀`, the orientation sum **at that
pixel** — local divisive normalisation. Ours is the better match to cortical contrast gain control;
theirs is the one that plays well with SGD.

### 6. No scale ladder, no explicit orientation, nothing cross-scale

3×3 kernels throughout; scale comes only from network stride. There is no counterpart to our
log-scale ladder, to `A₂`'s end-stopping with its flank offset anchored on a measured structure
scale, or to `A₃`'s cross-scale conjunction.

### 7. What counts as success

They test Cifar-10 accuracy and JPEG robustness. **They never test any invariance** — not polarity,
not size, not few-shot — and never the AND in isolation, because it is always embedded in a trained
network. They cannot say what the AND buys by itself. We can, and do: Phase 8's junction benchmark,
Phase 13's kNN baseline and few-shot arms.

## The route they do not consider: no AND at all

Their thesis is that i2D selectivity **requires** an AND of two filters. Our **ray transform**
(`P0-8_RationalGaborFeatures/RayHarmonics.module.jl`) is a counterexample to a narrower but more
interesting version of it.

```math
R(p, \varphi) \;=\; E\big(p + d\,u(\varphi),\; \theta = \varphi \bmod \pi\big), \qquad u(\varphi) = (\cos\varphi, \sin\varphi)
```

*"Is there a contour at distance `d` in direction `φ`, oriented along `φ`?"* `R` has one lobe per
branch and its circular harmonics are the junction-type signature — `c₀` counts rays, `|c₁|/c₀`
measures asymmetry:

| configuration | c₀ | \|c₁\|/c₀ | \|c₂\|/c₀ |
|:--|--:|--:|--:|
| endpoint (1 ray) | 1 | 1.000 | 1.000 |
| straight (2 opposite) | 2 | 0.000 | 1.000 |
| L-corner (2 at 90°) | 2 | 0.707 | 0.000 |
| T-junction (3 rays) | 3 | 0.333 | 0.333 |
| X-crossing (4 rays) | 4 | 0.000 | 0.000 |

**This operator is linear in the energy field** — a weighted sum of rigidly shifted orientation
channels — where A₁ and their min are both bilinear. The discriminating work is done by the
**geometry of the sampling**, not by a product.

No contradiction with Zetzsche & Barth: `E` is already quadratic in the image, so `R` is too. But
it sharpens the Min-Net claim considerably. **The AND buys you the energy field. It does not buy
you junction type.** A second AND stacked on top adds nothing there; a spatial offset does. Nothing
in their framing anticipates this, because they treat "AND of two filters" as the whole mechanism
rather than as one step of two.

### And it exposes a limitation of ours that they do not share

The orientation fibre `E(x, y, θ)` is **π-periodic**; ray count is a **2π** property — it is about
which rays *leave* a point, and mod-π orientation cannot tell east from west. A₁ takes a product
across orientation channels *at a single point*, discarding spatial layout, so it **provably cannot
count rays**. Measured cosine similarity of `E(θ)` at figure centres:

```
L-corner   vs T-junction : 0.9031
L-corner   vs X-crossing : 0.8868
T-junction vs X-crossing : 0.9234
```

Effectively the same vector. No downstream learning recovers what the representation never encoded.
The ray transform exists because of this, and the fix was offset sampling rather than a bigger
nonlinearity.

**Their min does not have this limitation.** Each of `v`, `g` reads a 3×3 *spatial patch* of a
feature map, so spatial layout survives the product and the operator is in principle 2π-capable.
That is a genuine advantage of their formulation over A₁ — and they never test it, because Cifar-10
has no junction taxonomy and the paper never asks what the learned pairs discriminate.

One reservation, stated as a hypothesis rather than a measurement: a single 3×3 depthwise kernel
spans ±1 map pixel, which at their coarsest Cifar stack (8×8) is ±4 input pixels, and at the first
stack ±1. Our ray offset is on the order of one stroke width. So an offset large enough to separate
branches is reachable only at the stacks where the spatial resolution to distinguish T from X has
already been thrown away. Whether depth composes around this is exactly the question their
evaluation cannot answer.

### What the ray block actually costs us

Stated plainly, because it is the weakest part of our own position. The ray harmonics are
**expensive relative to their return, and the return is dataset-dependent**. On Fashion-MNIST:
+0.54 points over the base alone, and **−0.04 when added on top of A₁+A₂** — eighty-one features
for nothing. Expected on silhouettes, which have no junctions to count, but it is a real cost. The
block earns its columns only under Phase 9's question — how much of a *graded* property is
*linearly* available — not under classification accuracy.

So: we have a route they do not consider, it answers a question their operator class cannot, and it
has not yet paid for itself on a classification benchmark.

## How big their effect is

Cifar-10, final epoch, no early stopping (they report min-over-epochs separately in the appendix,
which is the right way round):

| model | params | test error @ 300 ep |
|:--|--:|--:|
| DenseNet L=100, k=12 | 769 k | 4.79 ± 0.06 |
| **Min-Net** L=100, k=12 | 752 k | **4.55 ± 0.14** |
| DenseNet L=58 | 314 k | 5.90 ± 0.30 |
| **Min-Net** L=58 | 299 k | **5.67 ± 0.07** |
| DenseNet L=22 | 72 k | 9.76 ± 0.19 |
| **Min-Net** L=22 | 57 k | **9.22 ± 0.17** |

0.2–0.5 points, with **fewer** parameters, consistent across three depths × two architectures ×
3–5 seeds. Real, small, and honestly reported.

Robustness, as percentage of changed predictions at JPEG quality Q=90: DenseNet-100 8.1 %,
Min-Net-100 7.3 %. By Q=10 the advantage has essentially gone (62.6 vs 61.4 test error) and both
models are shredded. They say so: "even the more robust Min-Nets remain sensitive to compression
artifacts given that 8 % of the predictions change with a slightly altered test set."

## Two things worth taking

**POCP — percentage of changed predictions.** Instead of reporting the accuracy delta between clean
and perturbed input, report the fraction of images whose *prediction changed*:

```math
\mathrm{POCP}(f, Q) \;=\; \frac{1}{|X|}\sum_{I \in X} \mathbb{1}\big(f(I) \neq f(\mathrm{perturb}(I,Q))\big)
```

This cancels the baseline accuracy difference between arms, which is exactly the confound that
makes our changed-conditions comparisons awkward to read — a weaker arm can look more "robust"
simply by having less accuracy to lose. Directly transferable to the polarity-inversion,
size-change and stroke-thickening splits. Note their caution: POCP is larger than the error
increase, because it also counts already-wrong predictions changing to a different wrong class, and
occasionally-right ones going wrong and back.

**Hyperselectivity as a stated mechanism.** Their equation 4:

```math
f \text{ is hyperselective} \iff \exists\, o : f(x^* + o) < f(x^*), \quad x^{*\mathsf T}o = 0
```

A perturbation **orthogonal** to the optimal stimulus reduces the output. This is impossible for a
linear filter with a pointwise nonlinearity, where `ReLU(wᵀ(x* + o)) = ReLU(wᵀx*)` for any
orthogonal `o`. It is the formal statement of what an AND buys you.

Our `A₁` is hyperselective in exactly this sense and we have never framed it that way or measured
it. The junction benchmark currently reports contrast ratios between stimulus classes; it could
instead report the orthogonal-perturbation falloff, which is a property of the operator rather than
of the stimulus set we happened to choose.

## What we could tell them that they do not know

**The fixed 90° pairing is a control they never ran.** If a hand-built orthogonal pairing on a fixed
Gabor bank recovers most of the Min-block's gain, the "learning finds the right pairs" story is
unnecessary — and if it does not, that is the first evidence that the learned pairs are doing
something other than end-stopping.

**And the AND is not the only way to get i2D.** Their argument stops at "you need a product". The
ray transform shows that once the quadrature energy field exists, junction *type* comes from offset
sampling — an operator that is linear in that field. If the Min-block's gain survives replacing the
min with a fixed pair of offset taps, the paper's mechanism is misattributed.
