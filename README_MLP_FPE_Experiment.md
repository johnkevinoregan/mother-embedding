# Hand-designed shape features vs. FPE codes, through a plain MLP on EMNIST

**A self-contained report.** Nothing here assumes you have read any other file in this
repository, or that you already know what Zernike moments, Fractional Power Encoding,
or Vector-Symbolic Architectures are. Every term is defined where it is first used, and
every number quoted was measured by the runs described below.

Date: 2026-07-26. Companion notebook: `ExptsWithZernike/MLPonFeatures.jl`.

---

## 1. The question

We had built a **156-number description of a handwritten character** out of two very
different families of shape measurements (§4). Up to this point we had only ever scored
that description with a deliberately weak classifier, which gave 62.5 % on a 47-way
problem. Two questions followed:

1. **How good are these features really**, when scored with a classifier that is
   actually trying — and how do they compare to just handing a network the raw image?
2. **Does it help to re-code the features before classification?** Specifically, does
   embedding each number into a **Fractional Power Encoding** (§6) — the representation
   this whole project is built around — make the information easier or harder for a
   network to use?

The second question matters beyond benchmark numbers. FPE is a way of representing
quantities as vectors so that they can be *combined algebraically* — bound to roles,
superposed, queried. If that representation costs little or nothing in classification
accuracy, it can be adopted freely. If it costs a lot, its algebraic convenience has a
price that has to be justified.

---

## 2. The dataset

**EMNIST-Balanced.** EMNIST is a set of handwritten characters rendered as 28×28
greyscale images, derived from the same source as the classic MNIST digit set but
extended to letters. The *Balanced* variant has **47 classes with equal numbers of
examples each**:

- the 10 digits `0`–`9`,
- the 26 uppercase letters `A`–`Z`,
- 11 lowercase letters `a b d e f g h n q r t`.

Only 11 lowercase letters appear because the dataset's designers merged every
upper/lower pair whose handwritten forms are essentially identical (`C`/`c`, `O`/`o`,
`S`/`s`, …). The 11 that survive as separate classes are exactly those whose lowercase
shape genuinely differs from the uppercase one.

**Official split**, used here exactly as published so the numbers are comparable to
other people's:

| | images | per class |
|:--|--:|--:|
| train | 112,800 | 2,400 |
| test | 18,800 | 400 |

**Chance accuracy is 1/47 = 2.13 %.**

For context on what "good" means on this dataset: published *convolutional* networks
reach roughly **91 %**; published *non-convolutional* multilayer perceptrons on the raw
pixels land around **84–85 %**; the linear baseline in the original EMNIST paper is
about **78 %**.

**Statistical resolution.** With 18,800 test images at ~86 % accuracy, the standard
error on any single accuracy figure is `√(0.86 × 0.14 / 18800) ≈ 0.25 %`. **Differences
smaller than about 0.5 percentage points are not meaningful.** This matters repeatedly
below.

---

## 3. Image preprocessing

Every 28×28 image is **bilinearly upsampled to 112×112** before feature extraction.
This is not for extra information — there is none to add — but because the feature
extractors below place analysis windows and discs at sub-pixel positions, and a 28×28
grid is too coarse for that to be stable. At 112×112 a typical pen stroke is about
**13 pixels wide**, which is the natural length scale for everything that follows.

---

## 4. The 156 features

The description is the concatenation of two blocks that measure genuinely different
things. Both are hand-designed: no learning is involved in producing them.

### 4.1 Block Z — global Zernike moments (75 numbers)

**What Zernike polynomials are.** They are a family of functions defined on a unit disc
(a circle of radius 1), which are *orthogonal* over that disc — meaning each one
measures a pattern of variation that none of the others can express. Each is indexed by
two integers:

```
V_nm(ρ, θ) = R_n^{|m|}(ρ) · e^{imθ}       for ρ ≤ 1,  n ≥ 0,  |m| ≤ n,  n − |m| even
```

where `ρ` is distance from the disc centre (0 at the centre, 1 at the rim) and `θ` is
the angle around the disc.

- **`m` is the angular index** — how the pattern varies as you go *around* the disc.
  `m = 0` means rotationally symmetric (a bullseye); `m = 2` means two-lobed (like a
  bar); `m = 4` means four-lobed (like a cross).
- **`n` is the radial index** — how much structure there is going *outward* from centre
  to rim. `R_n^{|m|}(ρ)` is a polynomial in `ρ`, and larger `n` means more rings.

**The moments.** Projecting the image `f` onto each polynomial gives one complex number:

```
A_nm = (n+1)/π · ∫∫_{ρ≤1}  f(ρ,θ) · R_n^{|m|}(ρ) · e^{−imθ}  dA
```

`A_nm` says "how much of this pattern is present in the character". Because the image
is real-valued, `A_{n,−m}` is just the complex conjugate of `A_nm`, so only `m ≥ 0`
carries new information. Keeping all orders up to **`n ≤ 8`** gives **25 moments**.

**The key property.** If you rotate the character by an angle `α`, every moment simply
picks up a phase: `A_nm → A_nm · e^{−imα}`. So the **magnitude `|A_nm|` is unchanged by
rotation** — it is a rotation-invariant shape measurement — while the **phase carries
the orientation**. We keep both, because for upright letters the orientation is
informative (an `M` and a `W` differ by rotation, and we want to tell them apart).

**The 75 numbers** are therefore `|A_nm|`, `Re A_nm` and `Im A_nm` for the 25 moments.

**Fitting the disc.** Zernike polynomials only exist on the unit disc, so the character
must be placed inside one. We do this by finding the centre of mass of the ink and
scaling so that the 98th percentile of ink radii lands on the rim. (The 98th percentile
rather than the maximum, so that a single stray speck cannot shrink the whole letter.)
This matters: an unfitted disc wastes most of the basis describing empty corners.

**The DC term.** Before taking moments we subtract the mean intensity inside the disc,
which zeroes `A_00`. This makes every remaining moment a statement about *structure*
rather than about how much ink there is. (Because of the orthogonality, adding a
constant to the image only affects `A_00` and nothing else.)

### 4.2 Block F — the "tic-tac-toe" Fourier grid (81 numbers)

Where block Z describes the character as one centred whole, block F describes **where
things are**. The 112×112 image is divided into a **3×3 grid of cells** — hence
"tic-tac-toe" — and each of the 9 cells is described by 9 numbers, giving 81.

Within a cell we take a **windowed 2D Fourier transform**. Concretely: multiply the
cell by a smooth Gaussian window (so the cell's edges don't create artefacts), and
compute the Fourier coefficients `F(v,u)` for the low orders `|v|, |u| ≤ 3`. Each
coefficient corresponds to a plane wave across the cell: `u` counts how many cycles fit
horizontally, `v` vertically. A coefficient's magnitude says how much of that wave is
present.

The useful fact is that **a straight stroke puts its energy along a line in the `(v,u)`
plane perpendicular to the stroke**. So rather than keep the raw coefficients, we
summarise the pattern of energy:

| feature | what it is | what it means |
|:--|:--|:--|
| `a₀` | the `F(0,0)` coefficient | how much **ink** is in this cell |
| `ac` | `√Σ|F|²` over all non-zero orders | how much **structure** is in this cell |
| `Re E₂`, `Im E₂` | `E₂ = Σ|F|² e^{2iθ} / Σ|F|²`, where `θ` is the direction of each coefficient in the `(v,u)` plane | the **orientation** of the dominant stroke, encoded as a 2-vector |
| `|E₂|` | magnitude of the above | **how oriented** the cell is: ~0.5 for one clean stroke, 0.0 for a crossing or a blob |
| `|E₄|` | the same construction with `e^{4iθ}` | intended to detect two strokes at ~90°; measured to work poorly |
| `e₁, e₂, e₃` | fraction of energy at radius 1, 2, 3 in the `(v,u)` plane | the **radial profile** — coarse vs. fine structure in the cell |

`E₂` deserves a note because the same idea recurs later. Orientation is a quantity that
repeats every 180°, not 360° (a stroke at 10° and a stroke at 190° are the same stroke).
Multiplying the angle by 2 inside the exponential is what makes the code respect that:
`e^{2iθ}` gives the same value for `θ` and `θ+180°`. `|E₂|` then measures how
concentrated the energy is in one direction, and `arg(E₂)/2` recovers the direction
itself.

### 4.3 Why two blocks

Block Z is **rotation-aware and centred**: it describes the character as one object
about its own centre, and knows nothing about where any part of it sits. Block F is
**translation-aware and local**: it knows exactly which ninth of the image each
measurement came from, but each cell's measurement is insensitive to where within that
cell the ink lies. They are, by construction, blind in different directions.

---

## 5. The classifier

A **multilayer perceptron (MLP)** — the plainest kind of neural network. Every unit in
each layer is connected to every unit in the next, with no convolution, no weight
sharing, no spatial structure of any kind. The network is simply given a vector of
numbers and asked to output one of 47 class scores.

Architecture and training, held identical across every condition:

| | |
|:--|:--|
| hidden layers | **1 layer of 256 ReLU units** (justified in §7.1) |
| output | linear layer to 47 class scores |
| loss | softmax cross-entropy |
| optimiser | Adam, learning rate 1e-3 |
| batch size | 128 |
| epochs | 15 |
| framework | Flux.jl 0.16.10 |

Inputs are **standardised** (each feature shifted and scaled to zero mean and unit
variance using *training-set* statistics only) and clipped to ±3 standard deviations.
Standardising with training statistics and applying them unchanged to the test set is
what keeps the evaluation honest.

We report **two numbers** per run: the accuracy at the final epoch, and the best
accuracy reached at any epoch. When these diverge, the network is overfitting late in
training, and the gap is itself informative.

---

## 6. Fractional Power Encoding, and the three input codings

### 6.1 What FPE is

Fractional Power Encoding is a way of turning a **number** into a **vector**, such that
similar numbers get similar vectors and the vectors can be combined algebraically.

Pick a set of `d` random frequencies `θ_1 … θ_d`. Encode the number `x` as the vector
whose `k`-th component is `e^{i θ_k x}` — a unit complex number whose phase advances at
rate `θ_k` as `x` grows. Written as real numbers, that is the pair
`[cos(θ_k x), sin(θ_k x)]` for each `k`.

Why "fractional power": if you call the vector for `x = 1` the base `z`, then the vector
for any `x` is `z` raised to the power `x`, with fractional powers perfectly
well-defined because raising a unit complex number to a fractional power just scales its
phase.

The essential property is the **similarity kernel**. The inner product between the codes
of two numbers depends only on their difference, and if the frequencies are drawn from a
Gaussian with standard deviation `σφ`:

```
⟨ V(x), V(y) ⟩  ≈  exp( −σφ² (x−y)² / 2 )
```

So **`σφ` is an inverse similarity width**. Small `σφ` → a broad, smooth code where
numbers far apart still look similar. Large `σφ` → a sharp, high-resolution code where
only very close numbers look alike. Since our features are standardised, `σφ` is in
units of "radians of phase per standard deviation of the feature".

**Two more VSA operations** are needed for the third coding below:

- **Binding** — combining two vectors by elementwise multiplication (`⊙`). Binding a
  value code to a random **role vector** `R_j` tags it as "this is feature *j*'s value".
  Binding is invertible: multiplying by the conjugate of `R_j` recovers the value code.
- **Bundling** — superposing several vectors by *adding* them. The sum is similar to
  each of its parts, so one vector can hold many items at once — but the parts interfere,
  and the interference grows with how many you pack in.

### 6.2 The three codings compared

Take the 156 standardised feature values `x_1 … x_156` of one character.

**(a) RAW.** Feed the 156 numbers to the network directly. **156 inputs.**

**(b) CONCAT-FPE.** Give each feature its own FPE code and lay them end to end:

```
input = [ V_1 | V_2 | … | V_156 ]     where V_j = [cos(θ_jk x_j), sin(θ_jk x_j)]_k
```

Here **`d_feat`** is the number of real numbers spent on **one** feature — it is
`d_feat/2` frequencies, each contributing a cosine and a sine. The total input width is
`156 × d_feat`, so `d_feat = 8` gives **1,248 inputs** and `d_feat = 32` gives
**4,992 inputs**.

This coding is mathematically identical to **random Fourier features**, a standard trick
for approximating a kernel machine: `d_feat/2` is the number of Monte-Carlo samples used
to approximate the Gaussian kernel of width `1/σφ`. It tests whether a fixed nonlinear
expansion of each input helps the network.

**(c) BUNDLE-FPE.** The Vector-Symbolic coding. Bind each feature's value code to that
feature's role vector, and superpose everything into **one** vector:

```
V = (1/√156) · Σ_j  R_j ⊙ z_j^{x_j}
```

fed to the network as `[Re V, Im V]`. Here **`D`** is the width of that single vector,
so the input is **`2D` numbers regardless of how many features there are** — `D = 256`
gives **512 inputs** for all 156 features. That fixed width is the entire point of the
representation: it composes.

The cost is interference. Recovering one item from a bundle of `N` gives signal 1
against crosstalk of roughly `√(N/D)`. With `N = 156`, that is 0.78 at `D = 256` and
0.28 at `D = 2048`.

> **Note on notation.** `d_feat` (concat) and `D` (bundle) are *not comparable numbers*.
> `d_feat` is per-feature; `D` is the total for all features. Concat at `d_feat = 32`
> uses 4,992 inputs; bundle at `D = 256` uses 512.

### 6.3 The unbinding diagnostic

If the bundle coding underperforms, there are two quite different possible reasons:
the superposition destroyed the information, or the information is present but gradient
descent failed to find it. To tell these apart we test recoverability directly, with no
learning involved:

1. build the bundle `V`;
2. unbind feature `j` analytically: `u_j = V ⊙ conj(R_j)`, which yields `z_j^{x_j}` plus
   crosstalk from the other 155 features;
3. decode the value by correlating `u_j` against the code for every candidate value on a
   grid and taking the best match;
4. score the recovered value against the true one with **R²** (1.0 = perfect recovery,
   0.0 = no better than always guessing the mean, negative = worse than that).

---

## 7. Results

All figures are accuracy on the **official 18,800-image test set**, 47 classes,
chance 2.13 %. Format: *final epoch / best epoch*. Recall that **±0.5 % is the
resolution limit**.

### 7.1 How deep does the network need to be?

Measured on the raw 156 features at 30 epochs:

| hidden layers | final | best |
|:--|--:|--:|
| 256 | **85.87 %** | 86.76 % |
| 512 | 85.56 % | 86.63 % |
| 512, 512 | 85.29 % | 86.68 % |
| 512, 512, 512 | 85.03 % | 86.61 % |

**Depth does not help at all.** Every configuration peaks at essentially the same place
(86.6–86.8 %), and the final-epoch number gets *worse* with depth purely because deeper
networks overfit more in the last epochs. All subsequent experiments therefore use **one
hidden layer of 256 units**, which is also the fastest. Separately, 15 epochs gave a
better final accuracy than 30 (86.43 % vs 85.87 %), so 15 epochs is used throughout.

The interpretation is that the features have already done the hard nonlinear work. What
remains for the network is close to a linear separation, and extra capacity only buys
extra overfitting.

### 7.2 Reference points

| input | numbers | final | best |
|:--|--:|--:|--:|
| **raw 156 features** | 156 | **86.43 %** | 86.71 % |
| raw 784 pixels | 784 | 83.65 % | 83.95 % |
| block Z alone (Zernike) | 75 | 83.96 %\* | 84.27 %\* |
| block F alone (Fourier grid) | 81 | 85.86 %\* | 86.32 %\* |

<sub>\* measured at 30 epochs</sub>

**Three findings.**

**(i) The 156 features beat the 784 raw pixels by 2.8 points** under an identical
network. This is the headline result. The features are a 5× smaller representation, and
yet they are *more* useful — so they are not a lossy compression of the image but a
genuine re-description that exposes structure a fully-connected network cannot easily
find for itself. (A convolutional network would find much of it, which is why
convolutional EMNIST results are higher; the comparison here is like-for-like against a
network with no spatial prior.)

**(ii) A weak classifier badly understated these features.** The same 156 numbers scored
**62.5 %** under leave-one-out nearest-class-mean — a classifier that represents each
class by its average feature vector and assigns each item to the nearest one. The MLP
gets **86.4 %** from identical inputs. That 24-point gap is entirely the classifier.
Nearest-class-mean weights every feature equally and assumes classes are blobs around
their means; when neither holds, it reports a floor, not a measurement.

**(iii) A conclusion we had drawn earlier does not survive.** Under nearest-class-mean,
combining blocks Z and F was worth **+6.4 points** over either alone, and we had
interpreted that as strong evidence of complementarity. Under the MLP, block F alone
gets **85.86 %** and Z+F gets **86.43 %** — a 0.6-point difference, at the edge of
significance, and block Z contributes almost nothing on top of F. The complementarity
was largely an artefact of the weak classifier's inability to exploit F's features on
its own, not a property of the representations. This is recorded here because it
corrects the earlier claim.

### 7.3 CONCAT-FPE: giving each feature its own code

| `σφ` | `d_feat` = 8 (1,248 inputs) | `d_feat` = 32 (4,992 inputs) |
|--:|--:|--:|
| 0.5 | **85.58 %** / 85.68 | 84.72 % / 84.78 |
| 1.0 | 85.11 % / 85.47 | 84.27 % / 84.77 |
| 2.0 | 82.94 % / 84.22 | 82.30 % / 83.73 |

**Every cell is below the 86.43 % raw baseline, and accuracy falls monotonically in both
directions** — the more code dimensions, and the sharper the kernel, the worse it gets.
The best cell is the one that least resembles an FPE code at all: the narrowest
bandwidth and the smallest code, i.e. the setting closest to just passing the number
through.

This is a clean negative result, and in hindsight it is the expected one. Random Fourier
features are valuable when you need a *fixed* nonlinear expansion because your model
cannot learn one — a kernel machine, a linear readout. An MLP's first layer *is* a
learned nonlinear expansion. Replacing a scalar by a random sinusoidal basis destroys
the monotone ordering that first layer could have exploited, and hands it 8–32× more
parameters to overfit with. The encoding is solving a problem the network does not have.

### 7.4 BUNDLE-FPE: the Vector-Symbolic coding

| `σφ` | `D`=256 (512 in) | `D`=512 (1,024 in) | `D`=1024 (2,048 in) | `D`=2048 (4,096 in) |
|--:|--:|--:|--:|--:|
| 0.5 | 85.63 % / 85.75 | 85.75 % / 85.75 | **85.79 %** / 85.79 | 84.00 % / 85.04 |
| 1.0 | 84.98 % / 85.15 | 84.98 % / 85.14 | 85.00 % / 85.37 | 84.21 % / 84.63 |
| 2.0 | 81.85 % / 82.74 | 82.43 % / 83.64 | 82.59 % / 84.14 | 82.50 % / 83.38 |

**The headline is how cheap bundling is.** At `σφ = 0.5` the bundle reaches **85.79 %**
against raw's 86.43 % — a **0.6-point cost**, which is barely more than the measurement
resolution. All 156 features, superposed into a single vector of 1,024 numbers with
their role tags, cost essentially nothing in accuracy.

Two structural observations:

- **Accuracy is flat in `D` from 256 to 1024** (85.63 → 85.75 → 85.79 at `σφ=0.5`), then
  *drops* at `D = 2048`. The drop is overfitting, not information loss: the final-vs-best
  gap widens sharply there (84.00 vs 85.04) as the first layer grows past a million
  parameters, while the recoverability of the information is still *increasing* (§7.5).
- **Narrow bandwidth wins**, consistently, at every width: `σφ = 0.5` > `1.0` > `2.0`.

### 7.5 What survives superposition: the unbinding diagnostic

Median R² of recovering the 156 individual feature values from the bundle, by analytic
unbinding and matched-filter decoding — **no learning involved**:

| `D` | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|:--|--:|--:|--:|--:|--:|--:|
| `σφ` = 0.5 | −1.19 | −0.31 | 0.37 | 0.75 | 0.88 | 0.94 |
| `σφ` = 1.0 | −1.36 | −0.17 | 0.56 | 0.90 | 0.96 | 0.98 |
| `σφ` = 2.0 | −1.38 | −0.48 | 0.32 | 0.86 | 0.99 | 1.00 |
| `√(D/156)` | 0.91 | 1.28 | 1.81 | 2.56 | 3.62 | 5.12 |

This behaves exactly as superposition theory predicts. Below `D ≈ N = 156` the R² is
**negative** — the recovered values are worse than simply guessing the mean, because
crosstalk from the other 155 items exceeds the signal. Recovery then climbs steadily
with width, reaching near-perfect only around `D = 4096`.

The bandwidth optimum **moves with width**: `σφ = 1` is best at `D` = 512–1024, `σφ = 2`
is best at 2048–4096. Bandwidth buys resolution, width buys capacity, and you can only
afford resolution once you have capacity — pushing `σφ = 2` at `D = 512` costs you
(0.32 vs 0.56) because the code is finer than the dimensionality can support.

### 7.6 The most interesting result: recovery and classification come apart

Put §7.4 and §7.5 side by side at `σφ = 0.5`:

| `D` | 256 | 512 | 1024 | 2048 |
|:--|--:|--:|--:|--:|
| **decode R²** (can you get the numbers back?) | **−0.31** | 0.37 | 0.75 | 0.88 |
| **classification** (can you tell the letter?) | **85.63 %** | 85.75 % | 85.79 % | 84.00 % |

At `D = 256` the individual feature values are **unrecoverable** — R² is negative, the
decoder does worse than guessing — and yet the network classifies at **85.63 %**, within
a point of having the clean features. Across the whole range, decode R² moves from
*negative* to 0.75 while accuracy moves by 0.16 points.

**These two capacities are close to unrelated, and conflating them would be a mistake.**
The reason is that classification never requires inverting the superposition. The
decoder is solving a hard problem — recover 156 specific values exactly — whereas the
network only needs to find directions in the bundle along which the 47 classes separate.
Those directions are linear combinations of many features at once, they are numerous,
and they survive levels of crosstalk that destroy per-item recovery.

The practical consequence for this project is concrete: **the `√(D/N)` capacity rule is
the right yardstick for retrieval and the wrong one for classification.** Sizing a
bundle so that its contents can be read back item-by-item is the correct discipline when
you intend to query it symbolically. If the bundle is only ever going to be consumed by
a downstream discriminative readout, that rule over-provisions the width by roughly an
order of magnitude — here, `D = 256` classifies as well as `D = 1024`, while needing
`D ≈ 4096` for faithful readback.

---

## 8. Summary of conclusions

1. **The 156 hand-designed features are genuinely good**: 86.4 % on 47-way EMNIST,
   beating a 784-pixel raw-image MLP (83.7 %) trained identically, at a fifth the size.
2. **They are shallow-friendly.** One hidden layer of 256 units is as good as three of
   512; the features have already done the nonlinear work.
3. **Weak classifiers give badly misleading readings.** The same features score 62.5 %
   under leave-one-out nearest-class-mean — 24 points lower — and, worse, that classifier
   produced a *qualitative* conclusion (strong Z+F complementarity) that the MLP shows to
   be an artefact.
4. **Concatenated FPE hurts, monotonically.** It is random Fourier features, which solve
   a problem a learned first layer does not have.
5. **Bundled FPE is nearly free**: 85.8 % against 86.4 %, packing 156 features into a
   single 1,024-number vector with role tags. For a representation whose whole purpose is
   algebraic composability, a 0.6-point cost is a bargain.
6. **Recoverability and discriminability are different capacities.** A bundle too narrow
   for its contents to be read back at all can still be classified almost perfectly.

### What this does not show

- Only one classifier family was tried. A convolutional network on raw pixels would beat
  everything here; the pixel arm is a like-for-like control, not an attempt at the state
  of the art.
- Bandwidth and width were swept coarsely (3 × 4 grid), with one random seed per cell.
  Differences below ~0.5 points anywhere in these tables should not be interpreted.
- The bundle used one role vector per feature *type*, with all 156 features always
  present. It does not test the case FPE is really for — variable-length, variable-content
  structures — where the fixed width becomes a genuine advantage rather than a
  constraint the raw coding does not have to satisfy.

---

## 9. Reproducing

`ExptsWithZernike/MLPonFeatures.jl` is a Pluto notebook implementing all of the above
with the arms, code sizes and bandwidths on sliders. Feature extraction over the full
131,600 images takes about 105 s; individual training runs range from 12 s (raw
features) to ~20 min (the widest codes).

```bash
cd mother-embedding
julia --project=. -t 4 -e 'using Pluto; Pluto.run(notebook="ExptsWithZernike/MLPonFeatures.jl")'
```

The `-t 4` matters: the FPE encoders are threaded, and are slow without it.

Requirements: Julia 1.11, the project's `Project.toml` environment (Flux 0.16.10 is
included), and the EMNIST-Balanced idx files in `~/Julia/DATABASES/EMNIST/` — the train
split at the top level and the test split in the `emnist_source_files/` subdirectory.
